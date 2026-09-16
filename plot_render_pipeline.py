# Proxy-GS rendering pipeline diagram:
# left = per-frame neural pipeline (measured), right = static export + SparkJS.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C_STORE = "#dbeafe"   # data/store
C_OP = "#fef3c7"      # compute op
C_OUT = "#dcfce7"     # output
C_EDGE = "#475569"
C_BAD = "#fee2e2"


def box(x, y, w, h, text, fc, fs=10.5, weight="normal", ec="#64748b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2, label=None, color=C_EDGE, style="-", lw=1.8, fs=9.5, lpos=0.5, ldy=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, linestyle=style, zorder=1,
                                 shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + ldy if lpos == 0.5 else y1 * (1 - lpos) + y2 * lpos + ldy,
                label, ha="center", fontsize=fs, color=color, zorder=3)


ax.text(25, 97.5, "① Python 神经渲染管线（每帧）", ha="center", fontsize=14, weight="bold", color="#1e3a8a")
ax.text(76, 97.5, "② 一次性离线导出 → SparkJS 静态渲染", ha="center", fontsize=14, weight="bold", color="#14532d")

# ---------------- left column ----------------
LX = 6
box(LX, 86, 38, 8, "checkpoint（iteration_100000）\nanchors 7.80M · feat 32d · offsets ×10 · scaling", C_STORE)
box(LX, 72.5, 38, 8, "prefilter_voxel\n视锥剔除（深度纹理，256B pitch 对齐）", C_OP)
box(LX, 59, 38, 8, "generate_neural_gaussians\nopacity / color / cov 三个 MLP 解码（视角相关）", C_OP)
box(LX, 45.5, 38, 8, "neural gaussians（每帧生成）\nxyz + opacity + color + scale + rot", C_STORE)
box(LX, 32, 38, 8, "diff-gaussian-rasterization\n标准 3DGS 光栅化", C_OP)
box(LX, 20, 38, 7, "渲染图像 250×250", C_OUT)

arrow(25, 86, 25, 81.3)
arrow(25, 72.5, 25, 67.8)
arrow(25, 59, 25, 54.3)
arrow(25, 45.5, 25, 40.8)
arrow(25, 32, 25, 27.8)

# timing annotations on the right of left column
ax.text(45.5, 76.5, "4.8 ms", fontsize=10, color="#b45309", weight="bold")
ax.text(45.5, 68.6, "≈124 ms ≈ 80%", fontsize=10, color="#b45309", weight="bold", ha="left", va="center")
ax.text(45.5, 36, "23.7 ms @250²\n68.0 ms @500²", fontsize=10, color="#b45309", weight="bold", ha="left", va="center")
ax.text(45.5, 50, "可见 anchors\n≈ 4.3M（街景正视）", fontsize=9.5, color="#475569", ha="left", va="center")

box(LX, 6, 38, 9, "每帧全跑 @250²：156 ms ⇒ 6.4 FPS\ndecode 摊销 K=20 @250²：34.6 ms ⇒ 28.9 FPS\n@500²：直接 5.1 FPS ／ 摊销 12.7 FPS", C_BAD, fs=10)

# ---------------- right column ----------------
RX = 57
box(RX, 86, 38, 8, "同一 checkpoint\n（anchors + MLP 权重）", C_STORE)
box(RX, 72.5, 38, 8, "export_decoded_ply.py\n每个 anchor 从最近训练相机解码 78M 候选", C_OP)
box(RX, 59, 38, 8, "过滤：opacity > 0.02 ＆ scale ≤ 1.0\n→ 23.9M，按 opacity topk → 8.0M", C_OP)
box(RX, 45.5, 38, 8, "标准 3DGS PLY（519 MiB）\nf_dc · logit opacity · log scale · quat", C_STORE)
box(RX, 32, 38, 8, "SparkJS SplatMesh（viewer/）\nWASM 解析 + GPU 排序光栅化", C_OP)
box(RX, 20, 38, 7, "浏览器实时画面", C_OUT)

arrow(76, 86, 76, 81.3)
arrow(76, 72.5, 76, 67.8)
arrow(76, 59, 76, 54.3)
arrow(76, 45.5, 76, 40.8)
arrow(76, 32, 76, 27.8)

ax.text(96, 76.5, "一次性\n≈ 40 s", fontsize=9.5, color="#b45309", ha="left", va="center")
box(RX, 6, 38, 9, "实测 60 FPS（vsync 上限）\n视角相关效果被冻结在参考视角\n（opacity / 颜色不再随相机更新）", C_OUT, fs=10)

# cross arrow: static bake trade-off
ax.add_patch(FancyArrowPatch((44.5, 63), (57.5, 63), arrowstyle="-|>", mutation_scale=16,
                             color="#7c3aed", lw=1.8, linestyle="--", zorder=1))
ax.text(51, 60.9, "烘焙一次\n省去每帧 decode", ha="center", fontsize=9.5, color="#7c3aed")

ax.text(50, 1.5, "结论：decode 占神经管线约 8 成（4.3M 可见 anchors）；静态烘焙 + SparkJS 以“视角相关效果冻结”为代价换 ~9 倍帧率",
        ha="center", fontsize=11, color="#334155", weight="bold")

fig.savefig(r"\\wsl.localhost\Ubuntu-22.04\home\steven\Proxy-GS\render_pipeline.png", bbox_inches="tight")
print("saved render_pipeline.png")
