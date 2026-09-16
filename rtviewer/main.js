// Proxy-GS realtime decode viewer — websocket streaming client.
// Backend: viewer_server.py (ws://<host>:8765). No build step, no deps.

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);

const WS_PORT = 8765;
$("ws-addr").textContent = location.hostname + ":" + WS_PORT;

// ------------------------------------------------------------ orbit state --
// First-person style: drag rotates the view IN PLACE (eye fixed, safe for
// street data); wheel dollies along the view direction; shift/right-drag pans.
const orbit = {
  eye: [0, 0, 0],
  pivotDist: 8,             // |target - eye|, kept for pan/tilt pivot
  az: Math.PI * 0.25,       // radians
  el: Math.PI * 0.12,
  fovx_deg: 60,
  up: [0, 0, 1],            // dataset world is Z-up
  ready: false,
};

function viewDir() {
  const { az, el } = orbit;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
}
function targetPoint() {
  return add3(orbit.eye, mul3(viewDir(), orbit.pivotDist));
}

// ------------------------------------------------------------ sensitivity --
const sens = { move: 1.0, rotate: 1.0, wheel: 1.0 };
function bindSens(id, key, fmt) {
  const el = $(id), lab = $(id + "-v");
  const apply = () => {
    sens[key] = parseFloat(el.value);
    lab.textContent = fmt(sens[key]);
  };
  el.addEventListener("input", apply);
  apply();
}

// ------------------------------------------------------- dataset cameras --
let CAMERAS = { train: [], test: [] };
let currentCam = null;   // {set, i, n}
let gtMode = false;
let lastGtUrl = null;

function markFreeView() {
  if (currentCam) {
    currentCam = null;
    $("hud-cam").textContent = "自由";
    $("hud-psnr").textContent = "–";
  }
}

function jumpCam(set, i) {
  const list = CAMERAS[set] || [];
  if (list.length === 0) return;
  i = ((i % list.length) + list.length) % list.length;
  const c = list[i];
  markDirty();
  orbit.eye = c.e.slice();
  orbit.az = Math.atan2(c.f[1], c.f[0]);
  orbit.el = Math.asin(Math.max(-1, Math.min(1, c.f[2])));
  orbit.pivotDist = 5.0;
  currentCam = { set, i, n: c.n };
  $("hud-cam").textContent = set + "/" + c.n;
  $("cam-idx").value = i;
  lastInputTs = -1e9;      // force an immediate full-pipeline refine frame
}

function camIndexNow() {
  const set = $("cam-set").value;
  if (currentCam && currentCam.set === set) return currentCam.i;
  return parseInt($("cam-idx").value || "0", 10) || 0;
}

function updateCamCount() {
  const l = CAMERAS[$("cam-set").value] || [];
  $("cam-count").textContent = l.length ? "0–" + (l.length - 1) : "–";
  $("cam-idx").max = Math.max(0, l.length - 1);
}

$("cam-go").addEventListener("click", () => jumpCam($("cam-set").value, camIndexNow()));
$("cam-prev").addEventListener("click", () => jumpCam($("cam-set").value, camIndexNow() - 1));
$("cam-next").addEventListener("click", () => jumpCam($("cam-set").value, camIndexNow() + 1));
$("cam-set").addEventListener("change", updateCamCount);
$("cam-idx").addEventListener("keydown", (e) => { if (e.key === "Enter") $("cam-go").click(); });

$("gt-toggle").addEventListener("change", (e) => {
  gtMode = e.target.checked;
  $("gt-state").textContent = gtMode ? "开" : "关";
  document.body.classList.toggle("gt-mode", gtMode);
  if (!gtMode) $("hud-psnr").textContent = "–";
  markDirty();
  lastInputTs = -1e9;   // canvas width changed -> resend immediately with GT flag
});

// ------------------------------------------------------------ interaction --
let dragging = false, panning = false, lastX = 0, lastY = 0;
let dirty = true;                 // camera changed, need to (re)send
let lastInputTs = 0;
let refineSent = false;

function markDirty() {
  dirty = true;
  refineSent = false;
  lastInputTs = performance.now();
}

// WASD + QE fly controls (held keys, integrated per-frame in tick)
const keysDown = new Set();
const MOVE_CODES = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"]);
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (!MOVE_CODES.has(e.code)) return;
  e.preventDefault();
  if (!keysDown.has(e.code)) { keysDown.add(e.code); }
});
window.addEventListener("keyup", (e) => keysDown.delete(e.code));
window.addEventListener("blur", () => keysDown.clear());

function moveStep(dt) {
  if (keysDown.size === 0) return;
  const fwd = viewDir();
  const right = norm3(cross3(fwd, orbit.up));
  // base speed scales with scene distance so it feels right at any zoom
  const speed = orbit.pivotDist * 0.8 * sens.move;   // units per second
  let v = [0, 0, 0];
  if (keysDown.has("KeyW")) v = add3(v, fwd);
  if (keysDown.has("KeyS")) v = sub3(v, fwd);
  if (keysDown.has("KeyD")) v = add3(v, right);
  if (keysDown.has("KeyA")) v = sub3(v, right);
  if (keysDown.has("KeyE")) v = add3(v, orbit.up);
  if (keysDown.has("KeyQ")) v = sub3(v, orbit.up);
  if (v[0] === 0 && v[1] === 0 && v[2] === 0) return;
  orbit.eye = add3(orbit.eye, mul3(norm3(v), speed * dt));
  markFreeView();
  markDirty();
}

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  panning = e.button === 2 || e.shiftKey;
  lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  const fwd = viewDir();
  if (panning) {
    // pan eye laterally in the camera's right/up plane (Z-up world)
    const right = norm3(cross3(fwd, orbit.up));
    const upv = cross3(right, fwd);
    const scale = orbit.pivotDist * Math.tan(orbit.fovx_deg * Math.PI / 360) * 2 / canvas.clientHeight;
    orbit.eye = add3(orbit.eye, add3(mul3(right, -dx * scale), mul3(upv, dy * scale)));
    markFreeView();
  } else {
    orbit.az -= dx * 0.005 * sens.rotate;
    orbit.el = Math.min(Math.PI / 2 - 0.02, Math.max(-Math.PI / 2 + 0.02, orbit.el - dy * 0.005 * sens.rotate));
  }
  markDirty();
});
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  // dolly along the view direction, clamp to a sane corridor
  const step = orbit.pivotDist * (Math.exp(-e.deltaY * 0.0012 * sens.wheel) - 1);
  orbit.eye = add3(orbit.eye, mul3(viewDir(), step));
  markFreeView();
  markDirty();
}, { passive: false });

// vec helpers
function sub3(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function add3(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function mul3(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function cross3(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm3(a) { const l = Math.hypot(...a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

// ------------------------------------------------------------ websocket ----
let ws = null, seq = 0;
let lastFrameUrl = null;

function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${WS_PORT}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("overlay-title").textContent = "已连接，等待场景信息…"; };
  ws.onclose = () => {
    $("overlay").classList.remove("hidden");
    $("overlay-title").textContent = "后端未连接 — 请先启动 viewer_server.py";
    orbit.ready = false;
    setTimeout(connect, 2000);
  };
  ws.onerror = () => {};
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const msg = JSON.parse(ev.data);
      if (msg.type === "scene_info") {
        orbit.eye = msg.eye;
        orbit.fovx_deg = msg.fovx_deg;
        orbit.up = msg.up;
        // derive az/el/pivotDist from the seeded eye->target vector
        const d = sub3(msg.target, msg.eye);
        orbit.pivotDist = Math.hypot(...d);
        orbit.el = Math.asin(d[2] / orbit.pivotDist);
        orbit.az = Math.atan2(d[1], d[0]);
        CAMERAS = msg.cameras || { train: [], test: [] };
        updateCamCount();
        if (CAMERAS.train.length > 0) {
          currentCam = { set: "train", i: 0, n: CAMERAS.train[0].n };
          $("hud-cam").textContent = "train/" + CAMERAS.train[0].n;
        }
        orbit.ready = true;
        $("overlay").classList.add("hidden");
        markDirty();
      } else if (msg.type === "error") {
        console.error("backend:", msg.message);
      }
      return;
    }
    // binary frame: "PG"|"GT" + u32 meta_len + meta json + jpeg
    const buf = ev.data;
    const dv = new DataView(buf);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1));
    const metaLen = dv.getUint32(2, true);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 6, metaLen)));
    const jpeg = buf.slice(6 + metaLen);
    if (magic === "GT") drawGt(jpeg, meta);
    else drawFrame(jpeg, meta);
  };
}

function drawGt(jpeg, meta) {
  const blob = new Blob([jpeg], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  $("gt-img").src = url;
  if (lastGtUrl) URL.revokeObjectURL(lastGtUrl);
  lastGtUrl = url;
  $("gt-label").textContent = "GT · " + (meta.cam || "");
}

function drawFrame(jpeg, meta) {
  const blob = new Blob([jpeg], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    canvas.width = meta.w;
    canvas.height = meta.h;
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    updateHud(meta);
  };
  img.src = url;
}

function updateHud(meta) {
  $("hud-fps").textContent = meta.fps.toFixed(1);
  $("hud-ms").textContent = meta.ms.total.toFixed(1) + " ms";
  $("hud-ms-depth").textContent = meta.ms.depth.toFixed(1) + " / " + meta.ms.prefilter.toFixed(1);
  $("hud-ms-rest").textContent = meta.ms.decode.toFixed(1) + " / " + meta.ms.raster.toFixed(1);
  $("hud-visible").textContent = (meta.visible / 1000).toFixed(0) + " k";
  $("hud-res").textContent = meta.w + "×" + meta.h;
  $("hud-decode").textContent = meta.decoded ? "本帧全跑" : "复用缓存";
  if (meta.psnr !== undefined) $("hud-psnr").textContent = meta.psnr.toFixed(2);
}

// ------------------------------------------------------------ send loop ----
const INTERACTIVE_W = 500;
const REFINE_W = 1000;
const REFINE_IDLE_MS = 300;

let lastTickTs = performance.now();

function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastTickTs) / 1000);
  lastTickTs = now;
  if (!orbit.ready || !ws || ws.readyState !== WebSocket.OPEN) return;

  moveStep(dt);   // WASD/QE held-key movement (calls markDirty when moving)

  const idleMs = now - lastInputTs;
  const wantRefine = idleMs > REFINE_IDLE_MS;
  if (!dirty && !(wantRefine && !refineSent)) return;

  const refine = wantRefine && !refineSent;
  const w = refine ? REFINE_W : INTERACTIVE_W;
  const aspect = canvas.clientHeight / Math.max(1, canvas.clientWidth);
  const h = Math.round(w * aspect);
  ws.send(JSON.stringify({
    type: "camera", seq: ++seq,
    eye: orbit.eye, target: targetPoint(), up: orbit.up,
    fovx_deg: orbit.fovx_deg, width: w, height: h, refine,
    cam: currentCam ? { set: currentCam.set, n: currentCam.n } : null,
    gt: gtMode && !!currentCam,
  }));
  dirty = false;
  if (refine) refineSent = true;
}

bindSens("sens-move", "move", (v) => v.toFixed(1) + "×");
bindSens("sens-rotate", "rotate", (v) => v.toFixed(1) + "×");
bindSens("sens-wheel", "wheel", (v) => v.toFixed(1) + "×");
connect();
tick();
