# Realtime decode viewer backend for Proxy-GS.
#
# Serves live neural-rendered frames over WebSocket:
#   client sends orbit camera (eye/target/up + fov + resolution)
#   server: nvdiffrast proxy-mesh depth -> prefilter_voxel (frustum+occlusion)
#           -> generate_neural_gaussians (MLP decode, amortized every K frames)
#           -> diff-gaussian rasterization -> JPEG frame back
#
# The depth stage uses NvdiffrastMeshDepthRenderer, a drop-in for
# VK2TorchRenderer (render_real.py path); on native Linux with the Vulkan
# extension built, only the renderer construction line needs to change.
#
# Usage:
#   python viewer_server.py -m output/block_5 \
#       -s data/MatrixCity/small_city/street/pose_block/block_5 \
#       --mesh mesh/block_5_reduce5.ply --port 8765
#
#   # offline smoke test (no websocket): render N orbit frames to JPG
#   python viewer_server.py -m output/block_5 \
#       -s data/MatrixCity/small_city/street/pose_block/block_5 \
#       --mesh mesh/block_5_reduce5.ply --selftest 8 --selftest_out selftest_frames
#
# Protocol:
#   server -> client  json  {"type":"scene_info", center, radius, fovx_deg, up}
#   client -> server  json  {"type":"camera", seq, eye, target, up,
#                            fovx_deg, width, height, refine}
#   server -> client  binary  b"PG" + u32 meta_len + meta_json + jpeg_bytes

import argparse
import asyncio
import io
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaussian_renderer import GaussianModel, prefilter_voxel, generate_neural_gaussians
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.system_utils import searchForMaxIteration
from export_decoded_ply import load_cfg
from nvdiffrast_depth_renderer import NvdiffrastMeshDepthRenderer

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class Pipe:
    compute_cov3D_python = False
    debug = False


# ---------------------------------------------------------------- camera ----

def build_camera(eye, target, up, fovx_deg, W, H, znear=0.01, zfar=100.0):
    """Orbit camera -> MiniCam + intrinsics for the depth renderer.

    eye/target/up are world-space (dataset is Z-up). We build the OpenCV
    c2w directly (camera +z = backward, +x right, +y down), which is exactly
    what load_cams produces after its OpenGL->OpenCV flip.
    """
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    # OpenCV look-at: +z forward, +x right, +y down
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R_c2w = np.stack([xc, yc, zc], axis=1)
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ eye

    R = np.transpose(R_w2c)          # 3DGS stored convention (transposed)
    T = T_w2c
    fovx = math.radians(fovx_deg)
    fovy = 2.0 * math.atan(math.tan(fovx * 0.5) * H / W)

    world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
    proj = torch.tensor(getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)).float().transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cam = MiniCam(W, H, fovy, fovx, znear, zfar, world_view, full_proj)
    cam.image_name = "live"

    fx = W / (2.0 * math.tan(fovx * 0.5))
    fy = H / (2.0 * math.tan(fovy * 0.5))
    intr = (R.astype(np.float32), T.astype(np.float32), fx, fy, W / 2.0, H / 2.0)
    return cam, intr


def scene_info_from_transforms(source_path, which="transforms_train.json"):
    with open(os.path.join(source_path, which)) as f:
        data = json.load(f)
    centers = []
    for fr in data["frames"]:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        centers.append(c2w[:3, 3])
    centers = np.stack(centers)
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).mean()) * 1.2
    fovx_deg = math.degrees(float(data["camera_angle_x"]))
    # Seed the initial view from a real dataset camera (stays in-distribution):
    # orbit target sits a few meters ahead of cam0 along its view direction.
    c2w0 = np.array(data["frames"][0]["transform_matrix"], dtype=np.float64)
    eye0 = c2w0[:3, 3]
    fwd0 = -c2w0[:3, 2]                       # raw json is OpenGL c2w (-z forward)
    fwd0 = fwd0 / np.linalg.norm(fwd0)
    look_dist = max(radius * 2.0, 5.0)
    target0 = eye0 + fwd0 * look_dist
    return {
        "center": center.tolist(), "radius": radius, "fovx_deg": fovx_deg,
        "eye": eye0.tolist(), "target": target0.tolist(),
    }


# ---------------------------------------------------------------- engine ----

class Engine:
    def __init__(self, args):
        cfg = load_cfg(args.model_path)
        self.gaussians = GaussianModel(
            cfg["feat_dim"], cfg["n_offsets"], cfg["fork"], cfg["use_feat_bank"], cfg["appearance_dim"],
            cfg["add_opacity_dist"], cfg["add_cov_dist"], cfg["add_color_dist"], cfg["add_level"],
            cfg["visible_threshold"], cfg["dist2level"], cfg["base_layer"], cfg["progressive"], cfg["extend"])
        iteration = args.iteration
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        ckpt_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}")
        self.gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"), None)
        self.gaussians.load_mlp_checkpoints(ckpt_dir)
        self.gaussians.eval()
        self.iteration = iteration
        print(f"[engine] loaded iteration_{iteration}, anchors={self.gaussians.get_anchor.shape[0]}")

        self.depth_renderer = NvdiffrastMeshDepthRenderer(args.mesh, 512, 512)
        self.pipe = Pipe()
        self.bg = torch.zeros(3, device="cuda")
        self.decode_every = args.decode_every
        self.znear, self.zfar = args.znear, args.zfar

        self.frame_idx = 0
        self.decoded = None          # (xyz, color, opacity, scaling, rot)
        self.visible_count = 0
        self.fps_ema = 0.0

    def rasterize_decoded(self, cam):
        xyz, color, opacity, scaling, rot = self.decoded
        screenspace = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=False, device="cuda")
        settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=self.bg, scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
            sh_degree=1, campos=cam.camera_center, prefiltered=False, debug=False)
        rasterizer = GaussianRasterizer(raster_settings=settings)
        image, _radii = rasterizer(
            means3D=xyz, means2D=screenspace, shs=None, colors_precomp=color,
            opacities=opacity, scales=scaling, rotations=rot, cov3D_precomp=None)
        return image

    @torch.no_grad()
    def render_frame(self, req):
        W, H = int(req["width"]), int(req["height"])
        cam, (R, T, fx, fy, cx, cy) = build_camera(
            req["eye"], req["target"], req.get("up", [0, 0, 1]),
            float(req["fovx_deg"]), W, H, self.znear, self.zfar)

        t = {}
        t0 = time.perf_counter()
        need_full = (self.decoded is None or req.get("refine") or
                     self.frame_idx % self.decode_every == 0)
        if need_full:
            depth_m = self.depth_renderer.render(
                camera_R=R, camera_T=T, fx=fx, fy=fy, cx=cx, cy=cy,
                znear=self.znear, zfar=self.zfar, width=W, height=H)
            torch.cuda.synchronize()
            t["depth"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            vmask, _ = prefilter_voxel(cam, self.gaussians, self.pipe, self.bg, depth_map=depth_m)
            torch.cuda.synchronize()
            t["prefilter"] = time.perf_counter() - t1
            self.visible_count = int(vmask.sum().item())

            t2 = time.perf_counter()
            xyz, color, opacity, scaling, rot, _mask = generate_neural_gaussians(
                cam, self.gaussians, vmask)
            self.decoded = (xyz, color, opacity, scaling, rot)
            torch.cuda.synchronize()
            t["decode"] = time.perf_counter() - t2
        else:
            t["depth"] = t["prefilter"] = t["decode"] = 0.0

        t3 = time.perf_counter()
        image = self.rasterize_decoded(cam)
        torch.cuda.synchronize()
        t["raster"] = time.perf_counter() - t3

        t["total"] = time.perf_counter() - t0
        self.frame_idx += 1
        fps = 1.0 / max(t["total"], 1e-6)
        self.fps_ema = fps if self.fps_ema == 0 else self.fps_ema * 0.9 + fps * 0.1

        img = (torch.clamp(image, 0.0, 1.0) * 255.0).byte()
        arr = img.permute(1, 2, 0).detach().cpu().numpy()
        return arr, {
            "type": "frame", "seq": int(req.get("seq", 0)), "w": W, "h": H,
            "ms": {k: round(v * 1000.0, 2) for k, v in t.items()},
            "fps": round(self.fps_ema, 1),
            "visible": self.visible_count,
            "decoded": bool(need_full),
            "iteration": self.iteration,
        }


def encode_packet(meta, rgb_arr, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb_arr).save(buf, format="JPEG", quality=quality)
    meta_b = json.dumps(meta).encode()
    return struct.pack("<2sI", b"PG", len(meta_b)) + meta_b + buf.getvalue()


# --------------------------------------------------------------- server ----

async def handle_client(ws, engine):
    latest = {"req": None}
    peer = ws.remote_address
    print(f"[server] client connected: {peer}")
    await ws.send(json.dumps(SCENE_INFO))

    async def render_loop():
        rendered_seq = -1
        while True:
            req = latest["req"]
            if req is not None and req["seq"] != rendered_seq:
                rendered_seq = req["seq"]
                try:
                    arr, meta = await asyncio.get_event_loop().run_in_executor(
                        None, engine.render_frame, req)
                    await ws.send(encode_packet(meta, arr))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    try:
                        await ws.send(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        return
            else:
                await asyncio.sleep(0.005)

    loop_task = asyncio.create_task(render_loop())
    try:
        async for msg in ws:
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "camera":
                    latest["req"] = data          # keep only the newest
    finally:
        loop_task.cancel()
        print(f"[server] client disconnected: {peer}")


async def amain(args):
    engine = Engine(args)
    import websockets
    async with websockets.serve(
            lambda ws: handle_client(ws, engine),
            args.host, args.port, max_size=64 * 1024 * 1024):
        print(f"[server] listening on ws://{args.host}:{args.port} "
              f"(decode_every={engine.decode_every})")
        await asyncio.Future()


def selftest(args):
    engine = Engine(args)
    pivot = np.array(SCENE_INFO["target"])
    eye0 = np.array(SCENE_INFO["eye"])
    d = eye0 - pivot
    radius = float(np.linalg.norm(d))
    el0 = math.asin(d[2] / radius)
    az0 = math.atan2(d[1], d[0])
    os.makedirs(args.selftest_out, exist_ok=True)
    n = args.selftest
    for i in range(n):
        az = az0 + 2 * math.pi * i / n
        eye = pivot + radius * np.array([math.cos(el0) * math.cos(az),
                                         math.cos(el0) * math.sin(az),
                                         math.sin(el0)])
        req = {"seq": i, "eye": eye.tolist(), "target": pivot.tolist(),
               "up": [0, 0, 1], "fovx_deg": SCENE_INFO["fovx_deg"],
               "width": 500, "height": 500, "refine": i == 0}
        arr, meta = engine.render_frame(req)
        from PIL import Image
        path = os.path.join(args.selftest_out, f"frame_{i:03d}.jpg")
        Image.fromarray(arr).save(path, quality=90)
        print(f"[selftest] {path}  ms={meta['ms']} visible={meta['visible']} decoded={meta['decoded']}")


SCENE_INFO = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("--mesh", type=str, default="mesh/block_5_reduce5.ply")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--decode_every", type=int, default=8,
                        help="full depth+prefilter+decode every K frames; "
                             "other frames reproject the cached decode")
    parser.add_argument("--znear", type=float, default=0.01)
    parser.add_argument("--zfar", type=float, default=100.0)
    parser.add_argument("--selftest", type=int, default=0, metavar="N")
    parser.add_argument("--selftest_out", type=str, default="selftest_frames")
    args = parser.parse_args()

    info = scene_info_from_transforms(args.source_path)
    SCENE_INFO.update({"type": "scene_info", "up": [0, 0, 1], **info})
    print(f"[server] scene center={np.round(info['center'], 2).tolist()} "
          f"radius={info['radius']:.1f} fovx={info['fovx_deg']:.1f}")

    if args.selftest > 0:
        selftest(args)
    else:
        asyncio.run(amain(args))
