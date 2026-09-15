# Benchmark: per-frame cost of the Proxy-GS neural pipeline, split into
#   prefilter (anchor frustum culling) / decode (anchors -> gaussians via MLP)
#   / rasterize, vs an amortized-decode model (decode once every K frames).
#
# This answers the viewer-design question: is it cheaper to decode anchors
# to gaussians and reuse them, or to run the full neural render each frame.
import os
import json
import math
import glob
import time
import argparse
import numpy as np
import torch

from gaussian_renderer import GaussianModel, prefilter_voxel, render, generate_neural_gaussians
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from export_decoded_ply import load_cfg


class Pipe:
    compute_cov3D_python = False
    debug = False


def load_cams(source_path, which="transforms_test.json", max_cams=20, downscale=4):
    path = os.path.join(source_path, which)
    if not os.path.exists(path):
        path = sorted(glob.glob(os.path.join(source_path, "transforms_*.json")))[0]
    with open(path) as f:
        data = json.load(f)
    frames = data["frames"][:max_cams]
    fovx = data["camera_angle_x"]
    cams = []
    for i, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        # NeRF c2w is OpenGL convention (y up, -z forward); flip to OpenCV/COLMAP
        # (same convention as scene/dataset_readers.py readCamerasFromTransforms)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        w = frame.get("w", data.get("w", 1920)) // downscale
        h = frame.get("h", data.get("h", 1080)) // downscale
        fovy = 2 * math.atan(math.tan(fovx * 0.5) * h / w)
        world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
        proj = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)).float().transpose(0, 1).cuda()
        full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        cams.append(MiniCam(w, h, fovy, fovx, 0.01, 100.0, world_view, full_proj))
    return cams


def timed(fn, warmup=2, rep=5):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--max_cams", type=int, default=20)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--rep", type=int, default=5)
    args = parser.parse_args()

    cfg = load_cfg(args.model_path)
    gaussians = GaussianModel(
        cfg["feat_dim"], cfg["n_offsets"], cfg["fork"], cfg["use_feat_bank"], cfg["appearance_dim"],
        cfg["add_opacity_dist"], cfg["add_cov_dist"], cfg["add_color_dist"], cfg["add_level"],
        cfg["visible_threshold"], cfg["dist2level"], cfg["base_layer"], cfg["progressive"], cfg["extend"])
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        iteration = args.iteration
    ckpt_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}")
    gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"), None)
    gaussians.load_mlp_checkpoints(ckpt_dir)
    gaussians.eval()
    print(f"loaded iteration_{iteration}, anchors={gaussians.get_anchor.shape[0]}")

    cams = load_cams(args.source_path, max_cams=args.max_cams, downscale=args.downscale)
    pipe = Pipe()
    background = torch.zeros(3, device="cuda")

    rows = []
    with torch.no_grad():
        for cam in cams:
            t_pre, (vmask, _) = timed(lambda: prefilter_voxel(cam, gaussians, pipe, background), rep=args.rep)
            n_vis = int(vmask.sum().item())
            t_dec, _ = timed(lambda: generate_neural_gaussians(cam, gaussians, vmask), rep=args.rep)
            t_ren, out = timed(lambda: render(cam, gaussians, pipe, background, visible_mask=vmask), rep=args.rep)
            rows.append((t_pre, t_dec, t_ren, n_vis))

    t_pre = np.mean([r[0] for r in rows])
    t_dec = np.mean([r[1] for r in rows])
    t_ren = np.mean([r[2] for r in rows])
    n_vis = int(np.mean([r[3] for r in rows]))
    t_raster_only = t_ren - t_dec

    print(f"\ncameras: {len(cams)} @ {cams[0].image_width}x{cams[0].image_height}, visible anchors avg {n_vis}")
    print(f"{'stage':<28}{'ms':>10}{'FPS':>10}")
    print(f"{'prefilter_voxel':<28}{t_pre:>10.2f}")
    print(f"{'decode (MLP anchors->gs)':<28}{t_dec:>10.2f}")
    print(f"{'rasterize (render-decode)':<28}{t_raster_only:>10.2f}")
    print(f"{'direct render (pre+render)':<28}{t_pre+t_ren:>10.2f}{1000/(t_pre+t_ren):>10.1f}")
    for K in (1, 5, 20, 100):
        t = t_pre + t_raster_only + t_dec / K
        print(f"{'amortized decode K=' + str(K):<28}{t:>10.2f}{1000/t:>10.1f}")


if __name__ == "__main__":
    main()
