# Render the exported 3DGS PLY with the standard diff-gaussian rasterizer,
# from the same test cameras used by the training eval. Decisive check:
# if this looks clean, the PLY is good and any viewer artifact is Spark-side;
# if it looks broken, the export math is wrong.
import os
import math
import argparse
import numpy as np
import torch
import plyfile

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # test/ siblings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from bench_decode_vs_render import load_cams

SH_C0 = 0.28209479177387814


def load_ply(path):
    ply = plyfile.PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacity = v["opacity"][:, None]
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")
    color = (0.5 + SH_C0 * t(f_dc)).clamp(0, 1)
    return t(xyz), color, torch.sigmoid(t(opacity)), torch.exp(t(scale)), t(rot)


def render_ply(ply_path, source_path, out_dir, n=3, downscale=4):
    xyz, color, opac, scale, rot = load_ply(ply_path)
    print(f"loaded {xyz.shape[0]} gaussians from {ply_path}")
    scale = scale.clamp_min(1e-8)
    rot = torch.nn.functional.normalize(rot, dim=1)
    cams = load_cams(source_path, max_cams=n, downscale=downscale)
    bg = torch.zeros(3, device="cuda")
    os.makedirs(out_dir, exist_ok=True)

    for i, cam in enumerate(cams):
        settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height), image_width=int(cam.image_width),
            tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg, scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=0,
            campos=torch.inverse(cam.world_view_transform)[:3, 3].contiguous(),
            prefiltered=False, debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        means2d = torch.zeros_like(xyz, requires_grad=True)
        img, radii = rasterizer(
            means3D=xyz, means2D=means2d, shs=None, colors_precomp=color,
            opacities=opac, scales=scale, rotations=rot, cov3D_precomp=None)
        arr = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        out = os.path.join(out_dir, f"ply_render_{i:05d}.png")
        Image.fromarray(arr).save(out)
        print(f"saved {out} (radii>0: {(radii > 0).sum().item()})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("-o", "--out_dir", default="/tmp/ply_render_test")
    p.add_argument("-n", type=int, default=3)
    a = p.parse_args()
    with torch.no_grad():
        render_ply(a.ply, a.source_path, a.out_dir, a.n)
