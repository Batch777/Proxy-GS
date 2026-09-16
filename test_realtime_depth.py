# Verify NvdiffrastMeshDepthRenderer (VK2TorchRenderer drop-in) against the
# precomputed mesh_depth_block_5 npy files, and measure per-frame latency.
#
#   python test_realtime_depth.py -s data/MatrixCity/small_city/street \
#       --mesh mesh/tsdf_fusion_post_block5.ply --depth_dir mesh_depth_block_5
#
# The npy files were produced by mesh_render.py's mesh_depth_render() with
# flip_y=True / linear_depth=True at the dataset's -r 4 resolution (250x250),
# so the adapter at the same resolution should reproduce them almost exactly
# (only difference: zfar 1000 vs 100 used during npy generation).

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_decode_vs_render import load_cams, load_depth, timed
from nvdiffrast_depth_renderer import NvdiffrastMeshDepthRenderer

ZFAR = 100.0  # match npy generation (mesh_render.py used znear=0.01, zfar=100)


def cam_rt_intr(cam):
    """Recover (R in 3DGS stored convention, T, fx, fy, cx, cy) from a MiniCam."""
    # load_cams stored world_view_transform = getWorld2View2(R, T).T, and
    # getWorld2View2 itself transposes R internally -> w2c = world_view_transform.T
    w2c = cam.world_view_transform.detach().cpu().numpy().T
    R_3dgs = np.transpose(w2c[:3, :3])          # 3DGS stored convention (transposed)
    T = w2c[:3, 3].astype(np.float32)
    W, H = int(cam.image_width), int(cam.image_height)
    fx = W / (2.0 * math.tan(cam.FoVx * 0.5))
    fy = H / (2.0 * math.tan(cam.FoVy * 0.5))
    return R_3dgs.astype(np.float32), T, fx, fy, W / 2.0, H / 2.0


def render_view(renderer, cam):
    R, T, fx, fy, cx, cy = cam_rt_intr(cam)
    return renderer.render(camera_R=R, camera_T=T,
                           fx=fx, fy=fy, cx=cx, cy=cy,
                           znear=0.01, zfar=ZFAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--which", type=str, default="transforms_test.json")
    parser.add_argument("--max_cams", type=int, default=10)
    parser.add_argument("--downscale", type=int, default=4)
    args = parser.parse_args()

    cams = load_cams(args.source_path, which=args.which,
                     max_cams=args.max_cams, downscale=args.downscale)
    cam0 = cams[0]
    H, W = int(cam0.image_height), int(cam0.image_width)
    print(f"cams: {len(cams)}, resolution {W}x{H}")

    renderer = NvdiffrastMeshDepthRenderer(args.mesh, W, H)
    print("info:", renderer.get_info())

    n_checked = 0
    for cam in cams:
        npy_path = os.path.join(args.depth_dir, cam.image_name + ".npy")
        if not os.path.exists(npy_path):
            continue
        depth_rt = render_view(renderer, cam)
        depth_ref = load_depth(cam, args.depth_dir)
        assert depth_rt.shape == depth_ref.shape, (depth_rt.shape, depth_ref.shape)

        finite = torch.isfinite(depth_ref) & torch.isfinite(depth_rt)
        diff = (depth_rt[finite] - depth_ref[finite]).abs()
        rel = diff / depth_ref[finite].clamp(min=1e-6)
        miss_rt = (~torch.isfinite(depth_rt)).float().mean().item()
        miss_ref = (~torch.isfinite(depth_ref)).float().mean().item()
        print(f"{cam.image_name}: max|d|={diff.max().item():.4f} "
              f"mean|d|={diff.mean().item():.5f} "
              f"rel@p99={rel.quantile(0.99).item():.5f} "
              f"inf rt/ref={miss_rt:.3f}/{miss_ref:.3f}")
        n_checked += 1

    def one_frame():
        for cam in cams[:5]:
            render_view(renderer, cam)
        torch.cuda.synchronize()

    ms, _ = timed(one_frame)
    ms /= 5
    print(f"\nrealtime depth: {ms:.2f} ms/frame ({1000.0 / ms:.1f} FPS) @ {W}x{H}")
    print(f"checked {n_checked} cams against npy")


if __name__ == "__main__":
    main()
