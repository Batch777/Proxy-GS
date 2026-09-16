# Drop-in replacement for VK2TorchRenderer backed by nvdiffrast (pure-CUDA rasterizer).
#
# Why this exists:
#   The official realtime path (render_real.py) uses ProxyGS-Vulkan-Cuda-Interop's
#   VK2TorchRenderer to rasterize the proxy mesh into a linear depth buffer via
#   Vulkan + CUDA external-memory interop. That path needs a native Linux Vulkan
#   driver, which WSL2 does not expose (verified 2026-09-16: no NVIDIA Vulkan ICD
#   under /usr/lib/wsl, jammy mesa 23.2 ships no Dozen/dzn ICD).
#
#   This adapter reproduces the exact render() interface and output convention of
#   VK2TorchRenderer (linear view-space depth, float32 [H, W] on CUDA, +inf for
#   miss) using nvdiffrast's RasterizeCudaContext — the same code path that
#   generated mesh_depth_block_5 npy files, so it is already validated against
#   the training-time occlusion culling.
#
# Usage:
#   from nvdiffrast_depth_renderer import NvdiffrastMeshDepthRenderer
#   renderer = NvdiffrastMeshDepthRenderer(mesh_path="mesh/tsdf_fusion_post_block5.ply",
#                                          width=1024, height=690)
#   depth = renderer.render(camera_R=view.R, camera_T=view.T,
#                           fx=view.Fx, fy=view.Fy, cx=view.Cx, cy=view.Cy,
#                           znear=0.01, zfar=1000.0)

import numpy as np
import torch

from Mesh2DepthHelper import (
    DepthRenderer,
    Load_ply_resource,
    Build_Ply_Render_Camera_Parameters_colmap_correct,
)


class NvdiffrastMeshDepthRenderer:
    """Mimics VK2TorchRenderer(mesh_file).render(...) using nvdiffrast."""

    def __init__(self, mesh_path, width, height, device="cuda"):
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self.mesh_path = str(mesh_path)
        self._depth_renderer = DepthRenderer(device=device)
        self._mesh = Load_ply_resource(self.mesh_path, device)
        print(f"[NvdiffrastMeshDepthRenderer] {self.width}x{self.height}, "
              f"mesh={self.mesh_path} "
              f"(verts={self._mesh['verts'].shape[0]}, faces={self._mesh['faces'].shape[0]})")

    def get_info(self):
        return {
            "backend": "nvdiffrast.RasterizeCudaContext",
            "width": self.width,
            "height": self.height,
            "mesh": self.mesh_path,
            "verts": int(self._mesh["verts"].shape[0]),
            "faces": int(self._mesh["faces"].shape[0]),
        }

    @torch.inference_mode()
    def render(self, camera_R=None, camera_T=None, *,
               fx=None, fy=None, cx=None, cy=None,
               znear=0.01, zfar=1000.0,
               width=None, height=None):
        """Same contract as VK2TorchRenderer.render().

        camera_R follows the 3DGS Camera convention (world->cam stored
        transposed), exactly like render_real.py passes viewpoint_camera.R.
        width/height optionally override the construction resolution so a
        single instance can serve interactive + refine tiers.
        Returns: float32 CUDA tensor [H, W], linear view-space depth,
        +inf where the proxy mesh misses.
        """
        W = int(width) if width else self.width
        H = int(height) if height else self.height
        # Undo the 3DGS transpose (identical to to_vulkan_viewproj_match_nvdiffrast)
        R = np.swapaxes(np.asarray(camera_R, np.float32), -1, -2)
        T = np.asarray(camera_T, np.float32)

        cam_params = Build_Ply_Render_Camera_Parameters_colmap_correct(
            fx, fy, cx, cy, W, H, znear, zfar, R, T, self.device
        )
        depth, _mask = self._depth_renderer.render_depth_batched(
            mesh=self._mesh,
            camera_parameters=cam_params,
            flip_y=True,
            max_tris_per_pass=100_000_000,
            verbose=False,
            linear_depth=True,
        )
        return depth.contiguous()
