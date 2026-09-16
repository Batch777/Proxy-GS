# Unit test: GaussianModel.capture/restore dict pack/unpack round-trip.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import torch
import torch.nn as nn
from types import SimpleNamespace
from scene.gaussian_model import GaussianModel

dev = "cuda"
torch.manual_seed(3)
N, K, F = 100, 10, 32

def make_args():
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.0, position_lr_final=0.0, position_lr_delay_mult=0.01, position_lr_max_steps=500,
        offset_lr_init=0.01, offset_lr_final=0.0001, offset_lr_delay_mult=0.01, offset_lr_max_steps=500,
        feature_lr=0.0075, opacity_lr=0.02, scaling_lr=0.007, rotation_lr=0.002,
        mlp_opacity_lr_init=0.002, mlp_opacity_lr_final=2e-5, mlp_opacity_lr_delay_mult=0.01, mlp_opacity_lr_max_steps=500,
        mlp_cov_lr_init=0.004, mlp_cov_lr_final=0.004, mlp_cov_lr_delay_mult=0.01, mlp_cov_lr_max_steps=500,
        mlp_color_lr_init=0.008, mlp_color_lr_final=5e-5, mlp_color_lr_delay_mult=0.01, mlp_color_lr_max_steps=500,
        mlp_featurebank_lr_init=0.01, mlp_featurebank_lr_final=1e-5, mlp_featurebank_lr_delay_mult=0.01, mlp_featurebank_lr_max_steps=500,
        appearance_lr_init=0.05, appearance_lr_final=0.0005, appearance_lr_delay_mult=0.01, appearance_lr_max_steps=500,
    )

g = GaussianModel(F, K, 2, False, 0, False, False, False, False, 0.0, "round", 10, True, 1.1)
g._anchor = nn.Parameter(torch.randn(N, 3, device=dev))
g._level = torch.randint(0, 4, (N, 1), device=dev).float()
g._extra_level = torch.rand(N, device=dev)
g._offset = nn.Parameter(torch.randn(N, K, 3, device=dev))
g._anchor_feat = nn.Parameter(torch.randn(N, F, device=dev))
g._scaling = nn.Parameter(torch.randn(N, 6, device=dev))
g._rotation = nn.Parameter(torch.randn(N, 4, device=dev), requires_grad=False)
g._opacity = nn.Parameter(torch.randn(N, K, device=dev), requires_grad=False)
g.voxel_size = torch.tensor(0.01)
g.standard_dist = torch.tensor(1.5)
g.spatial_lr_scale = 5.0

args = make_args()
g.training_setup(args)

# Fill Adam moments + accumulators with sentinel values
loss = g._anchor.sum() + g._offset.sum() + g._anchor_feat.sum() + g._scaling.sum()
loss += sum(p.sum() for p in g.mlp_opacity.parameters())
loss.backward()
g.optimizer.step()
g.opacity_accum = torch.rand(N, 1, device=dev)
g.offset_gradient_accum = torch.rand(N * K, 1, device=dev)
g.offset_denom = torch.rand(N * K, 1, device=dev)
g.anchor_demon = torch.rand(N, 1, device=dev)

state = g.capture()
assert set(state.keys()) == {"params", "accumulators", "mlps", "meta", "optimizer"}
ref_params = {k: v.detach().clone() for k, v in state["params"].items()}
ref_accums = {k: v.clone() for k, v in state["accumulators"].items()}
ref_mlps = {k: {n: t.clone() for n, t in v.items()} for k, v in state["mlps"].items()}

# Wreck everything, then restore (replace attrs with NEW tensors; capture()
# intentionally stores references because training immediately torch.save()s)
for name in g._CKPT_PARAM_NAMES:
    attr = getattr(g, name)
    z = torch.zeros_like(attr)
    setattr(g, name, nn.Parameter(z) if isinstance(attr, nn.Parameter) else z)
g.voxel_size = torch.tensor(-1.0)
g.standard_dist = torch.tensor(-1.0)
g.spatial_lr_scale = -1.0
g.restore(state, args)

ok = True
for name, ref in ref_params.items():
    got = getattr(g, name)
    same = torch.equal(got.detach() if isinstance(got, nn.Parameter) else got, ref)
    ok &= same
    print(f"param {name:15s} restore={'OK' if same else 'MISMATCH'}")
for name, ref in ref_accums.items():
    same = torch.equal(getattr(g, name), ref)
    ok &= same
    print(f"accum {name:22s} restore={'OK' if same else 'MISMATCH'}")
for name, ref in ref_mlps.items():
    same = all(torch.equal(t, getattr(g, name).state_dict()[n]) for n, t in ref.items())
    ok &= same
    print(f"mlp   {name:15s} restore={'OK' if same else 'MISMATCH'}")
for name in g._CKPT_META_NAMES:
    same = torch.equal(getattr(g, name), state["meta"][name]) if isinstance(state["meta"][name], torch.Tensor) else getattr(g, name) == state["meta"][name]
    ok &= same
    print(f"meta  {name:15s} restore={'OK' if same else 'MISMATCH'}")

# optimizer state actually loaded (Adam exp_avg non-zero after step)
exp_avg = g.optimizer.state_dict()["state"][0]["exp_avg"]
ok &= bool(exp_avg.abs().sum().item() > 0)
print("optimizer state non-trivial:", bool(exp_avg.abs().sum().item() > 0))
print("capture/restore round-trip:", "ALL OK" if ok else "FAILED")
assert ok
