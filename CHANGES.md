# CHANGES — Proxy-GS 本地改动台账

本文件记录本工作区内每一处新增文件与修改，用于筛查 review。
按时间倒序，最新在最上。每条注明：类型（新增/修改/删除）、文件、原因、对应 commit。

---

## 2026-09-16 · 实时 Decode Viewer（P0：Vulkan 环境验证）

| 类型 | 文件 | 说明 | Commit |
|---|---|---|---|
| 删除 | `viewer/`（整个目录） | 静态 SparkJS viewer，被实时 decode viewer 方案取代（git 历史可恢复） | `119373e` |
| 新增 | `CHANGES.md` | 本台账文件 | 本次 |

> P0 进行中：cmake / Vulkan SDK 安装、vk2torch_ext 编译均只影响 conda 环境
> 与 `~/VulkanSDK`、`ProxyGS-Vulkan-Cuda-Interop/build-py/`（构建产物，不入库）。

---

## 2026-09-15 · 论文口径 FPS 复现 + 预览修复

| 类型 | 文件 | 说明 | Commit |
|---|---|---|---|
| 修改 | `bench_decode_vs_render.py` | prefilter_voxel 传入 `--depth_dir mesh_depth_block_5`（250² npy 遮挡深度），复现论文 FPS 口径：可见 anchors 4.3M→~180k，500² 53 FPS | `ab1336d` |
| 修改 | `plot_render_pipeline.py` / `render_pipeline.png` | 流程图更新为新实测数据 | `ab1336d` |
| 修改 | `viewer/package.json` | dev 脚本平台自定位 node（修复 Windows 预览 UNC cwd 问题） | `b92143e` |
| 修改 | `bench_decode_vs_render.py` | 相机 NeRF→OpenCV 翻转修复 `c2w[:3,1:3]*=-1`，对齐 `scene/dataset_readers.py` | `1d03189` |
| 新增 | `export_decoded_ply.py` | 最近相机现场 decode + scale≤1 过滤 + topk 8M，导出标准 PLY | `2ea108d` |
| 新增 | `bench_decode_vs_render.py` | 「decode 出 gaussians 再渲染」vs「直接 render」对比基准 | `2ea108d` |
| 新增 | `test_ply_render.py` | 标准 3DGS 光栅器渲染烘焙 PLY 的验证脚本 | `2ea108d` |
| 新增 | `viewer/` | SparkJS 静态 PLY viewer（已于 2026-09-16 删除） | `2ea108d` |
| 新增 | `depth_viewer.py` | npy 深度图可视化工具（colormap/统计/导出 PNG） | 早前 commit |
| 新增 | `plot_render_pipeline.py` / `render_pipeline.png` | 渲染流程图生成脚本与产物 | 早前 commit |

---

## 2026-09-15 · 训练修复链（commits 7342a33…d7c3776）

| 类型 | 文件 | 说明 |
|---|---|---|
| 修改 | `gaussian_renderer/__init__.py` | 深度纹理 pitch 对齐修复（mesh_render 训练 CUDA 崩溃） |
| 修改 | `scene/gaussian_model.py` | `anchor_growing` 显存爆炸修复；checkpoint pack/unpack 字典化（anchor/offset/feat/opacity/scaling/rotation 打包） |
| 修改 | `train.py` | 测试用训练阶段 iter 暂时调到 1000 以内 |
| 新增 | `test_ckpt_roundtrip.py` | checkpoint pack/unpack 往返一致性测试 |
| 新增 | `repro_depth_pitch.py` | 深度 pitch 问题最小复现 |

> 结果：100k 训练已跑通，模型在 `output/block_5/point_cloud/iteration_100000/`（7.8M anchors，PSNR 26.9）。

---

## 待办

- [ ] 推送本地 commit 到 fork `github.com/Batch777/Proxy-GS`（待 token 确认）
- [ ] P1–P4：PLY→GLB、后端推流、前端实时模式、调优
