#!/usr/bin/env python
"""depth_viewer.py — .npy 深度图可视化工具

用法:
    # 启动交互式网页查看器（默认当前目录，端口 8000）
    python depth_viewer.py mesh_depth_block_5
    python depth_viewer.py mesh_depth_block_5 --port 8000

    # 批量导出着色 PNG（不启动服务器）
    python depth_viewer.py mesh_depth_block_5 --save preview_png

    # 只导出单张
    python depth_viewer.py mesh_depth_block_5 --file 0000.npy --save .

网页端功能: 文件列表切换（支持 ← / → 方向键）、colormap 选择、
百分位拉伸范围 (pmin/pmax)、对数拉伸开关、inf/NaN 区域显示为黑色。

依赖: numpy / Pillow / matplotlib（项目 conda 环境均已具备）。
"""

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
from matplotlib import colormaps as plt_colormaps

CMAPS = ["turbo", "viridis", "magma", "inferno", "plasma", "jet", "gray"]


def list_npys(root):
    return sorted(f for f in os.listdir(root) if f.endswith(".npy"))


def normalize(d, pmin=2.0, pmax=98.0, log=False):
    """返回 (norm in [0,1] float32, valid_mask)。inf/NaN 视为无效。"""
    d = np.asarray(d, dtype=np.float64)
    valid = np.isfinite(d)
    if not valid.any():
        return np.zeros(d.shape, dtype=np.float32), valid
    v = d[valid]
    lo, hi = np.percentile(v, pmin), np.percentile(v, pmax)
    if log:
        lo, hi = np.log1p(max(lo, 0)), np.log1p(max(hi, 0))
        work = np.log1p(np.clip(d, 0, None))
    else:
        work = d
    if hi <= lo:
        hi = lo + 1e-12
    norm = np.clip((work - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return norm, valid


def colorize(d, cmap="turbo", pmin=2.0, pmax=98.0, log=False):
    norm, valid = normalize(d, pmin, pmax, log)
    lut = (plt_colormaps[cmap](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb = lut[(norm * 255).astype(np.uint8)]
    rgb[~valid] = (30, 30, 30)  # 无效区域深灰
    return rgb


def depth_info(d):
    valid = np.isfinite(d)
    v = d[valid]
    info = {
        "shape": list(d.shape),
        "dtype": str(d.dtype),
        "valid_pct": round(float(valid.mean()) * 100, 2),
    }
    if v.size:
        info.update({
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
            "p2": round(float(np.percentile(v, 2)), 4),
            "p50": round(float(np.percentile(v, 50)), 4),
            "p98": round(float(np.percentile(v, 98)), 4),
        })
    return info


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Depth Viewer</title>
<style>
  body { margin:0; display:flex; height:100vh; font:14px/1.4 monospace; background:#1e1e1e; color:#ddd; }
  #side { width:260px; overflow-y:auto; border-right:1px solid #444; padding:8px; }
  #side div { padding:2px 6px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #side div.sel { background:#094771; }
  #main { flex:1; display:flex; flex-direction:column; }
  #bar { padding:8px; border-bottom:1px solid #444; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  #view { flex:1; display:flex; align-items:center; justify-content:center; overflow:auto; }
  #view img { max-width:100%; max-height:100%; image-rendering:pixelated; }
  #info { padding:4px 8px; color:#9c9; border-top:1px solid #444; }
  input[type=number] { width:60px; background:#2d2d2d; color:#ddd; border:1px solid #555; }
  select,button { background:#2d2d2d; color:#ddd; border:1px solid #555; }
</style>
</head>
<body>
<div id="side"></div>
<div id="main">
  <div id="bar">
    <label>cmap <select id="cmap"></select></label>
    <label>pmin <input type="number" id="pmin" value="2" step="0.5"></label>
    <label>pmax <input type="number" id="pmax" value="98" step="0.5"></label>
    <label><input type="checkbox" id="log"> log</label>
    <button id="apply">应用</button>
    <span id="pos"></span>
  </div>
  <div id="view"><img id="img"></div>
  <div id="info"></div>
</div>
<script>
let files = [], cur = -1;
const $ = id => document.getElementById(id);
CMAPS.forEach(c => { let o = document.createElement("option"); o.value = o.text = c; $("cmap").appendChild(o); });

async function loadList() {
  files = (await (await fetch("/api/list")).json()).files;
  const side = $("side");
  files.forEach((f, i) => {
    const d = document.createElement("div");
    d.textContent = f; d.onclick = () => show(i);
    side.appendChild(d);
  });
  if (files.length) show(0);
}

function params() {
  return `cmap=${$("cmap").value}&pmin=${$("pmin").value}&pmax=${$("pmax").value}&log=${$("log").checked ? 1 : 0}`;
}

async function show(i) {
  if (i < 0 || i >= files.length) return;
  cur = i;
  [...$("side").children].forEach((d, j) => d.classList.toggle("sel", j === cur));
  $("pos").textContent = `${cur + 1}/${files.length}`;
  const f = encodeURIComponent(files[cur]);
  $("img").src = `/img?f=${f}&${params()}&t=${Date.now()}`;
  const info = await (await fetch(`/api/info?f=${f}`)).json();
  $("info").textContent = JSON.stringify(info);
}

$("apply").onclick = () => show(cur);
document.onkeydown = e => {
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") show(cur - 1);
  if (e.key === "ArrowRight" || e.key === "ArrowDown") show(cur + 1);
};
loadList();
</script>
</body>
</html>
""".replace("CMAPS.forEach", "CMAP_LIST.forEach").replace(
    "<script>", f"<script>\nconst CMAP_LIST = {json.dumps(CMAPS)};")


class Handler(BaseHTTPRequestHandler):
    root = "."

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, name):
        # 防路径穿越：只允许 root 目录下的 .npy
        p = os.path.realpath(os.path.join(self.root, os.path.basename(name)))
        if not p.startswith(os.path.realpath(self.root) + os.sep) or not p.endswith(".npy"):
            return None
        return p if os.path.exists(p) else None

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(INDEX_HTML)
        elif u.path == "/api/list":
            self._send(json.dumps({"files": list_npys(self.root)}), "application/json")
        elif u.path == "/api/info":
            p = self._resolve(q.get("f", [""])[0])
            if not p:
                return self._send("{}", "application/json", 404)
            self._send(json.dumps(depth_info(np.load(p))), "application/json")
        elif u.path == "/img":
            p = self._resolve(q.get("f", [""])[0])
            if not p:
                return self._send(b"", "image/png", 404)
            rgb = colorize(
                np.load(p),
                cmap=q.get("cmap", ["turbo"])[0],
                pmin=float(q.get("pmin", [2])[0]),
                pmax=float(q.get("pmax", [98])[0]),
                log=q.get("log", ["0"])[0] == "1",
            )
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "PNG")
            self._send(buf.getvalue(), "image/png")
        else:
            self._send("not found", code=404)

    def log_message(self, *a):
        pass


def export(root, out_dir, files, cmap, pmin, pmax, log):
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        rgb = colorize(np.load(os.path.join(root, f)), cmap, pmin, pmax, log)
        out = os.path.join(out_dir, os.path.splitext(f)[0] + ".png")
        Image.fromarray(rgb).save(out)
        print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=".npy 深度图可视化")
    ap.add_argument("dir", nargs="?", default=".", help="npy 所在目录")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cmap", default="turbo", choices=CMAPS)
    ap.add_argument("--pmin", type=float, default=2.0)
    ap.add_argument("--pmax", type=float, default=98.0)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--file", default=None, help="只处理单个 npy 文件名")
    ap.add_argument("--save", default=None, help="批量导出 PNG 到该目录（不启动服务器）")
    args = ap.parse_args()

    if args.save:
        files = [args.file] if args.file else list_npys(args.dir)
        export(args.dir, args.save, files, args.cmap, args.pmin, args.pmax, args.log)
        return

    Handler.root = args.dir
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving {os.path.abspath(args.dir)}  ->  http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
