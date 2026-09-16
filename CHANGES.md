# CHANGES — Proxy-GS 本地改动台账

本文件记录本工作区内每一处新增文件与修改，用于筛查 review。
按时间倒序，最新在最上。每条注明：类型（新增/修改/删除）、文件、原因、对应 commit。

---

## 2026-09-16 · P2+P3：实时 Decode Viewer（viewer_server.py + rtviewer/）

| 类型 | 文件 | 说明 | Commit |
|---|---|---|---|
| 新增 | `viewer_server.py` | WebSocket 推流后端：收相机(eye/target/up+fov+分辨率) → nvdiffrast 深度 → prefilter → MLP decode（每 K 帧，默认 8）→ 光栅化 → JPEG 推流；`--selftest N` 离线出环绕帧 | 本次 |
| 新增 | `rtviewer/`（index.html / main.js / style.css / server.js / package.json） | 前端：第一人称式控制（原地转视角安全，滚轮沿视线前进/后退，Shift/右键平移），静止 0.3s 自动请求 1000² 精修，HUD 显示 FPS/分段耗时/可见 anchors | 本次 |
| 修改 | `nvdiffrast_depth_renderer.py` | `render()` 增加可选 `width/height` 覆盖（单实例同时服务交互档与精修档） | 本次 |
| 环境 | conda 安装 `websockets 16.1.1` | 后端依赖 | — |

**踩坑记录**：
- `build_camera` 初版把 OpenCV c2w 的 +z 当成 backward（实际 OpenCV +z=forward），视角朝后 180°，
  渲染出模糊墙面；以真实数据集相机渲染对比 GT 图（`street/train/small_city_road_horizon.tar`
  内 0000.png）后定位修复，修正后渲染与 GT 街景一致，可见 anchors 286k 与 bench 263k 吻合
- 场景是街道数据集（相机沿 x ∈ [-5,5]、z≈0.03 一线分布），绕远处 pivot 环绕会撞墙出雾；
  初始视角改为种子化 cam0 位姿（eye=cam0 位置，target=前方 ~6.2m），前端改第一人称式控制
- Windows 版 node 的 fs API 对 `\\wsl.localhost\` UNC 路径返回 ENOENT（readdir/stat 均失败，
  但 exe 加载正常）；`path.normalize` 还会吞掉 UNC 前导 `\\`。
  解法：`npm run dev` 的 win32 分支改为 spawn **WSL 内 node**（`~/.local/node`），
  server.js 内去掉 `path.normalize`（`path.join` 已解析 `..` 且保留 UNC 前缀）

**验证**：
- `--selftest 8`：首帧全管线 ~1.0s（含 warmup），缓存帧光栅化 0.5–63ms（视角相关），帧图目视正确
- WebSocket 协议端到端：scene_info 握手 → 6 帧 PG 二进制包（39KB JPEG @500²），
  缓存帧 11–18ms，EMA FPS 27+（同机测试，不含网络）
- 静态服务器：index/main.js/style.css 全 200（`npm run dev`，WSL node v22）

**使用**：先在 proxy-gs 环境终端启动后端
`python viewer_server.py -m output/block_5 -s data/MatrixCity/small_city/street/pose_block/block_5 --mesh mesh/block_5_reduce5.ply --port 8765`，
再打开预览页（rtviewer，`npm run dev`）。

---

## 2026-09-16 · 实时 Decode Viewer（P0：Vulkan 环境验证 → 结论不可用，转 nvdiffrast）

**P0 结论：WSL2 无法使用官方 Vulkan-Cuda-Interop 路径**，证据链：

- `/usr/lib/wsl/lib` 只有 CUDA/D3D12 用户态库，**无 NVIDIA Vulkan ICD**；驱动目录里的
  `nv-vk64.json` 指向 Windows DLL（`nvoglv64.dll`），Linux loader 用不了
- jammy mesa 23.2 的 `mesa-vulkan-drivers` 不含 Dozen(dzn)，只有 lavapipe(CPU)/intel/radeon
- SDK loader 实测：`vulkaninfo` → `vkCreateInstance: Found no drivers!`
- 因此 `ProxyGS-Vulkan-Cuda-Interop` 编译暂缓（编得出也跑不了）；原生 Linux 环境可重启此路径，
  Vulkan SDK 已备好在 `~/VulkanSDK/1.4.321.1/x86_64`（不入库）

**环境变更（不入库）**：conda 环境安装 `cmake 4.4.3`、`ninja 1.13.2`；
`~/VulkanSDK/` 下载解压 SDK 1.4.321.1；`~/mesa-pkg/` 解包 mesa deb（验证用）。

| 类型 | 文件 | 说明 | Commit |
|---|---|---|---|
| 删除 | `viewer/`（整个目录） | 静态 SparkJS viewer，被实时 decode viewer 方案取代（git 历史可恢复） | `119373e` |
| 新增 | `CHANGES.md` | 本台账文件 | `a380a38` |
| 新增 | `nvdiffrast_depth_renderer.py` | **VK2TorchRenderer 的 drop-in 替代**：同 `render(camera_R, camera_T, fx..znear, zfar)` 接口、同输出约定（线性深度 float32 [H,W] cuda，miss=+inf），后端换 nvdiffrast RasterizeCudaContext（纯 CUDA，WSL 可用） | 本次 |
| 新增 | `test_realtime_depth.py` | 适配器对照测试：与 `mesh_depth_block_5` npy 逐像素对比 + 测速 | 本次 |
| 修改 | `bench_decode_vs_render.py` | 新增 `--which` 参数。**修正重大错配**：`mesh_depth_block_5` 的 1580 张 npy 是 train 相机生成的，此前 bench 默认读 transforms_test.json 按 image_name 匹配 → 深度与相机对不上，ab1336d 的 FPS 数字偏乐观 | 本次 |
| 修改 | `plot_render_pipeline.py` / `render_pipeline.png` | 流程图数字更新为修正后实测 | 本次 |

**适配器验证结果**（`--which transforms_train.json` 配对后）：
与 npy 逐像素差 mean ≈ 2 mm（reduce5 网格），inf 像素比例一致；实时深度 **6.7 ms/帧 @250²**（reduce5，21 万面；全网格 430 万面为 24 ms）。

**修正后 FPS 实测**（train 相机 + 配对 npy 深度，100k 模型 7.8M anchors，4090D）：

| 分辨率 | 可见 anchors | prefilter | decode | direct render | K=20 摊销 |
|---|---|---|---|---|---|
| 250² | 263k | 6.1 ms | 17.0 ms | 18.0 FPS | 25.5 FPS |
| 500² | 261k | 6.1 ms | 16.7 ms | 25.0 FPS | 41.3 FPS |
| 1000² | 251k | 6.1 ms | 16.6 ms | 25.9 FPS | 43.7 FPS |

> 旧数字（37/53/50 FPS）作废——那是错误深度多剔了 anchor（180k）的结果。
> decode（MLP 16.6 ms）是瓶颈，摊销复用是 viewer 上 40 FPS 的关键。

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
