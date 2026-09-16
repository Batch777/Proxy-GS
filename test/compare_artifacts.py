# Isolate the source of needle artifacts in the realtime viewer.
# Renders the SAME pose (transforms_test.json frame N) four ways:
#   A) official eval path: prefilter_voxel(no depth) + render()
#   B) viewer path: nvdiffrast reduce5 depth + prefilter + decode + rasterize
#   C) viewer path with FULL proxy mesh instead of reduce5
#   D) viewer decode+rasterize but NO occlusion depth (frustum only)
# If A is clean but B is not, the proxy-depth culling is the culprit;
# if A and D differ, the decode/rasterize split differs from render().
#
#   python test/compare_artifacts.py [--frame 3] [--width 500]

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # test/ siblings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from gaussian_renderer import GaussianModel, prefilter_voxel, render, generate_neural_gaussians
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.system_utils import searchForMaxIteration
from export_decoded_ply import load_cfg
from nvdiffrast_depth_renderer import NvdiffrastMeshDepthRenderer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class Pipe:
    compute_cov3D_python = False
    debug = False


def build_cam(eye, target, fovx_deg, W, H):
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.array([0, 0, 1], np.float64)
    zc = target - eye
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R_c2w = np.stack([xc, yc, zc], axis=1)
    R_w2c = R_c2w.T
    T = -R_w2c @ eye
    R = np.transpose(R_w2c)
    fovx = math.radians(fovx_deg)
    fovy = 2.0 * math.atan(math.tan(fovx * 0.5) * H / W)
    world_view = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
    proj = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)).float().transpose(0, 1).cuda()
    full_proj = (world_view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cam = MiniCam(W, H, fovy, fovx, 0.01, 100.0, world_view, full_proj)
    cam.image_name = "cmp"
    intr = (R.astype(np.float32), T.astype(np.float32),
            W / (2 * math.tan(fovx / 2)), H / (2 * math.tan(fovy / 2)), W / 2.0, H / 2.0)
    return cam, intr


def rasterize_decoded(cam, bg, decoded):
    xyz, color, opacity, scaling, rot = decoded
    screenspace = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=False, device="cuda")
    settings = GaussianRasterizationSettings(
        image_height=int(cam.image_height), image_width=int(cam.image_width),
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=1, campos=cam.camera_center, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(raster_settings=settings)
    image, _ = rasterizer(means3D=xyz, means2D=screenspace, shs=None, colors_precomp=color,
                          opacities=opacity, scales=scaling, rotations=rot, cov3D_precomp=None)
    return image


def save(img, path):
    from PIL import Image
    arr = (torch.clamp(img, 0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path, quality=92)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", default="output/block_5")
    ap.add_argument("-s", "--source_path", default="data/MatrixCity/small_city/street/pose_block/block_5")
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--out", default="selftest_frames")
    args = ap.parse_args()

    cfg = load_cfg(args.model_path)
    g = GaussianModel(cfg["feat_dim"], cfg["n_offsets"], cfg["fork"], cfg["use_feat_bank"], cfg["appearance_dim"],
                      cfg["add_opacity_dist"], cfg["add_cov_dist"], cfg["add_color_dist"], cfg["add_level"],
                      cfg["visible_threshold"], cfg["dist2level"], cfg["base_layer"], cfg["progressive"], cfg["extend"])
    it = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    ck = os.path.join(args.model_path, "point_cloud", f"iteration_{it}")
    g.load_ply_sparse_gaussian(os.path.join(ck, "point_cloud.ply"), None)
    g.load_mlp_checkpoints(ck)
    g.eval()
    pipe = Pipe()
    bg = torch.zeros(3, device="cuda")

    d = json.load(open(os.path.join(args.source_path, "transforms_test.json")))
    fovx_deg = math.degrees(float(d["camera_angle_x"]))
    m = np.array(d["frames"][args.frame]["transform_matrix"], dtype=np.float64)
    eye = m[:3, 3]
    fwd = -m[:3, 2]
    fwd = fwd / np.linalg.norm(fwd)
    W = H = args.width
    cam, intr = build_cam(eye, eye + fwd * 5.0, fovx_deg, W, H)
    name = os.path.splitext(os.path.basename(d["frames"][args.frame]["file_path"]))[0]
    print(f"test frame {args.frame} ({name}) eye={eye.round(3).tolist()}")

    with torch.no_grad():
        # A) official: frustum-only prefilter + render()
        vm_a, _ = prefilter_voxel(cam, g, pipe, bg)
        out_a = render(cam, g, pipe, bg, visible_mask=vm_a)["render"]
        save(out_a, os.path.join(args.out, f"cmp_A_official_{name}.jpg"))
        print("A visible:", int(vm_a.sum()))

        # B) viewer: reduce5 depth + prefilter + decode + rasterize
        dr5 = NvdiffrastMeshDepthRenderer("mesh/block_5_reduce5.ply", W, H)
        depth_b = dr5.render(camera_R=intr[0], camera_T=intr[1], fx=intr[2], fy=intr[3],
                             cx=intr[4], cy=intr[5], znear=0.01, zfar=100.0)
        vm_b, _ = prefilter_voxel(cam, g, pipe, bg, depth_map=depth_b)
        dec_b = generate_neural_gaussians(cam, g, vm_b)
        save(rasterize_decoded(cam, bg, dec_b[:5]), os.path.join(args.out, f"cmp_B_viewer_reduce5_{name}.jpg"))
        print("B visible:", int(vm_b.sum()))

        # C) viewer with FULL proxy mesh
        dr_full = NvdiffrastMeshDepthRenderer("mesh/tsdf_fusion_post_block5.ply", W, H)
        depth_c = dr_full.render(camera_R=intr[0], camera_T=intr[1], fx=intr[2], fy=intr[3],
                                 cx=intr[4], cy=intr[5], znear=0.01, zfar=100.0)
        vm_c, _ = prefilter_voxel(cam, g, pipe, bg, depth_map=depth_c)
        dec_c = generate_neural_gaussians(cam, g, vm_c)
        save(rasterize_decoded(cam, bg, dec_c[:5]), os.path.join(args.out, f"cmp_C_viewer_fullmesh_{name}.jpg"))
        print("C visible:", int(vm_c.sum()))

        # D) no depth, but viewer decode+rasterize split
        dec_d = generate_neural_gaussians(cam, g, vm_a)
        save(rasterize_decoded(cam, bg, dec_d[:5]), os.path.join(args.out, f"cmp_D_nodecode_split_{name}.jpg"))

        # also diff stats between depth maps
        finite = torch.isfinite(depth_b) & torch.isfinite(depth_c)
        print("depth reduce5-vs-full mean|d|:",
              float((depth_b[finite] - depth_c[finite]).abs().mean()))


if __name__ == "__main__":
    main()
