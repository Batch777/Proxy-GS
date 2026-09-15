# Export a trained Proxy-GS anchor model to a standard 3DGS .ply that
# generic splat viewers (SparkJS etc.) can load.
#
# Anchors are decoded to concrete Gaussians with the trained MLPs. Each
# anchor is decoded from its NEAREST real training camera center (parsed
# directly from transforms_*.json to avoid loading the full dataset) — the
# view-dependent opacity/color MLPs are only reliable for on-distribution
# view directions, and a single synthetic reference view bakes visible haze.
# Output layout follows the 3DGS spec:
# x,y,z, nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3.
import os
import json
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import GaussianModel

SH_C0 = 0.28209479177387814


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def load_cfg(model_path):
    cfg = dict(feat_dim=32, n_offsets=10, fork=2, use_feat_bank=False, appearance_dim=0,
               add_opacity_dist=False, add_cov_dist=False, add_color_dist=False, add_level=False,
               visible_threshold=0.0, dist2level="round", base_layer=10, progressive=True, extend=1.1)
    cfg_args = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_args):
        from argparse import Namespace
        with open(cfg_args) as f:
            ns = eval(f.read(), {"Namespace": Namespace})
        for k in cfg:
            if hasattr(ns, k):
                cfg[k] = getattr(ns, k)
    return cfg


def camera_centers(source_path):
    """All camera centers from transforms_*.json (c2w matrices)."""
    centers = []
    for tf in sorted(glob.glob(os.path.join(source_path, "transforms_*.json"))):
        with open(tf) as f:
            data = json.load(f)
        for frame in data.get("frames", []):
            m = np.array(frame["transform_matrix"], dtype=np.float64)
            centers.append(m[:3, 3])
    return torch.tensor(np.stack(centers), dtype=torch.float32, device="cuda")


def decode_chunk(pc, anchor, feat, offsets, scaling, center):
    """Replicates generate_neural_gaussians for the no-dist/no-level config
    (use_feat_bank=False, appearance_dim=0, add_*_dist=False, add_level=False).
    Returns per-Gaussian xyz, color, opacity(post-tanh), scale, rot (w-first)."""
    ob_view = anchor - center
    ob_dist = ob_view.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ob_view = ob_view / ob_dist

    cat_view = torch.cat([feat, ob_view], dim=1)  # [N, 35]

    opacity = pc.get_opacity_mlp(cat_view)              # [N, k] tanh
    color = pc.get_color_mlp(cat_view)                  # [N, 3k] sigmoid
    scale_rot = pc.get_cov_mlp(cat_view)                # [N, 7k]

    N, k = anchor.shape[0], pc.n_offsets
    opacity = opacity.reshape(N * k, 1)
    color = color.reshape(N * k, 3)
    scale_rot = scale_rot.reshape(N * k, 7)
    offsets = offsets.reshape(N * k, 3)

    # per-anchor scaling repeated per offset: [:, :3] scales offsets, [:, 3:] scales gaussians
    scaling_rep = scaling.repeat_interleave(k, dim=0)   # [N*k, 6]
    anchor_rep = anchor.repeat_interleave(k, dim=0)     # [N*k, 3]

    scale = scaling_rep[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = torch.nn.functional.normalize(scale_rot[:, 3:7], dim=1)
    xyz = anchor_rep + offsets * scaling_rep[:, :3]
    return xyz, color, opacity, scale, rot


def main():
    parser = argparse.ArgumentParser(description="Export Proxy-GS anchors to 3DGS ply")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="scene dir containing transforms_*.json (for reference view center)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="-1 = latest point_cloud/iteration_*")
    parser.add_argument("--opacity_threshold", type=float, default=0.02)
    parser.add_argument("--max_scale", type=float, default=1.0,
                        help="drop decoded gaussians whose largest scale axis exceeds this "
                             "(world units); kills giant low-opacity haze blobs from coarse levels")
    parser.add_argument("--max_gaussians", type=int, default=8_000_000)
    parser.add_argument("--chunk", type=int, default=500_000)
    parser.add_argument("-o", "--output", type=str, required=True)
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
    print(f"loading {ckpt_dir}")
    gaussians.load_ply_sparse_gaussian(os.path.join(ckpt_dir, "point_cloud.ply"), None)
    gaussians.load_mlp_checkpoints(ckpt_dir)
    gaussians.eval()

    centers = camera_centers(args.source_path)
    print(f"{centers.shape[0]} dataset camera centers loaded")

    N = gaussians.get_anchor.shape[0]
    k = gaussians.n_offsets
    print(f"anchors: {N}, offsets/anchor: {k} -> {N*k} candidate gaussians")

    out = {"xyz": [], "color": [], "opacity": [], "scale": [], "rot": []}
    with torch.no_grad():
        for s in tqdm(range(0, N, args.chunk), desc="decode"):
            e = min(s + args.chunk, N)
            anchors = gaussians.get_anchor[s:e]
            # nearest real camera center per anchor (on-distribution view dir)
            nearest = torch.cdist(anchors, centers).argmin(dim=1)
            center = centers[nearest]
            xyz, color, opacity, scale, rot = decode_chunk(
                gaussians,
                anchors, gaussians.get_anchor_feat[s:e],
                gaussians._offset[s:e], gaussians.get_scaling[s:e], center)
            keep = (opacity.squeeze(1) > args.opacity_threshold)
            if args.max_scale > 0:
                keep &= scale.max(dim=1).values <= args.max_scale
            out["xyz"].append(xyz[keep].cpu())
            out["color"].append(color[keep].cpu())
            out["opacity"].append(opacity[keep].cpu())
            out["scale"].append(scale[keep].cpu())
            out["rot"].append(rot[keep].cpu())

    xyz = torch.cat(out["xyz"]); color = torch.cat(out["color"])
    opacity = torch.cat(out["opacity"]); scale = torch.cat(out["scale"]); rot = torch.cat(out["rot"])
    print(f"survivors after opacity>{args.opacity_threshold} & scale<={args.max_scale}: {xyz.shape[0]}")

    if xyz.shape[0] > args.max_gaussians:
        idx = torch.topk(opacity.squeeze(1), args.max_gaussians).indices
        xyz, color, opacity, scale, rot = xyz[idx], color[idx], opacity[idx], scale[idx], rot[idx]
        print(f"capped to top-{args.max_gaussians} by opacity")

    # convert to 3DGS ply parameterization
    f_dc = (color - 0.5) / SH_C0
    opacity_logit = inverse_sigmoid(opacity.clamp(1e-6, 1 - 1e-6))
    log_scale = torch.log(scale.clamp_min(1e-12))
    normals = torch.zeros_like(xyz)

    attrs = torch.cat([xyz, normals, f_dc, opacity_logit, log_scale, rot], dim=1).numpy().astype(np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    dtype_full = [(n, "f4") for n in names]
    elements = np.ascontiguousarray(attrs).ravel().view(dtype_full)

    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(args.output)
    print(f"wrote {xyz.shape[0]} gaussians -> {args.output} "
          f"({os.path.getsize(args.output)/2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
