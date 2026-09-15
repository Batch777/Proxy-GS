# Minimal repro for depth-texture pitch alignment fix in visible_filter.
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

torch.manual_seed(0)
dev = "cuda"

W, H = 250, 250  # 250*4=1000 bytes pitch: misaligned without the fix
P = 10000

means3D = torch.randn(P, 3, device=dev) * 2.0
means3D[:, 2] += 5.0
scales = torch.full((P, 3), -4.0, device=dev)
rotations = torch.zeros(P, 4, device=dev)
rotations[:, 0] = 1.0
point_mask = torch.ones(P, dtype=torch.bool, device=dev)

viewmatrix = torch.eye(4, device=dev)
# w2c -> c2w convention: rasterizer expects transposed matrices like the main repo
viewmatrix = viewmatrix.transpose(0, 1).contiguous()

tanfovx, tanfovy = 0.5, 0.5
fx = W / (2 * tanfovx)
fy = H / (2 * tanfovy)
proj = torch.zeros(4, 4, device=dev)
proj[0, 0] = 2 * fx / W
proj[1, 1] = 2 * fy / H
proj[2, 2] = 1.0
proj[3, 2] = 1.0  # perspective
projmatrix = proj.transpose(0, 1).contiguous()

depth = torch.full((H, W), 100.0, device=dev)

settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=torch.zeros(3, device=dev),
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=0,
    campos=torch.zeros(3, device=dev),
    prefiltered=False,
    debug=True,
    depth_mesh=depth,
)

rasterizer = GaussianRasterizer(settings)
radii = rasterizer.visible_filter(means3D, scales=scales, rotations=rotations, point_mask=point_mask)
torch.cuda.synchronize()
print("visible_filter OK, radii>0:", int((radii > 0).sum().item()), "/", P)
