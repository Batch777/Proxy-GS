import torch, os
from gaussian_renderer import GaussianModel, prefilter_voxel, render, generate_neural_gaussians
from bench_decode_vs_render import load_cams, Pipe, timed
from export_decoded_ply import load_cfg
from utils.system_utils import searchForMaxIteration

cfg = load_cfg("output/block_5")
g = GaussianModel(cfg["feat_dim"], cfg["n_offsets"], cfg["fork"], cfg["use_feat_bank"], cfg["appearance_dim"],
    cfg["add_opacity_dist"], cfg["add_cov_dist"], cfg["add_color_dist"], cfg["add_level"],
    cfg["visible_threshold"], cfg["dist2level"], cfg["base_layer"], cfg["progressive"], cfg["extend"])
it = searchForMaxIteration("output/block_5/point_cloud")
ck = f"output/block_5/point_cloud/iteration_{it}"
g.load_ply_sparse_gaussian(os.path.join(ck, "point_cloud.ply"), None)
g.load_mlp_checkpoints(ck); g.eval()
pipe = Pipe(); bg = torch.zeros(3, device="cuda")
SRC = "data/MatrixCity/small_city/street/pose_block/block_5"

print("      res   visible  pre_ms  dec_ms  ren_ms  tot_ms    FPS")
with torch.no_grad():
    for ds in [4, 2, 1]:
        cam = load_cams(SRC, max_cams=1, downscale=ds)[0]
        t_pre, (vmask, _) = timed(lambda: prefilter_voxel(cam, g, pipe, bg), rep=5)
        nv = int(vmask.sum())
        t_all, _ = timed(lambda: render(cam, g, pipe, bg, visible_mask=vmask), rep=5)
        t_dec, _ = timed(lambda: generate_neural_gaussians(cam, g, vmask), rep=5)
        t_r = t_all - t_dec
        tot = t_pre + t_all
        w, h = int(cam.image_width), int(cam.image_height)
        print(f"{w:>6}x{h:<4} {nv:>8} {t_pre:>7.2f} {t_dec:>7.2f} {t_r:>7.2f} {tot:>7.2f} {1000/tot:>6.1f}")
