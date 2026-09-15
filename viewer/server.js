// Dependency-free static file server for the SparkJS viewer.
// Usage: node server.js [--port 7100] [--host 0.0.0.0]
const http = require("http");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
function argOf(name, dflt) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
}
// Also accept npm-style forwarded args like "--port=7100"
function argEq(name, dflt) {
  const hit = args.find((a) => a.startsWith(name + "="));
  return hit ? hit.split("=")[1] : dflt;
}
const PORT = parseInt(argOf("--port", argEq("--port", process.env.PORT || "7100")), 10);
const HOST = argOf("--host", argEq("--host", "0.0.0.0"));

const ROOT = __dirname;
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".ply": "application/octet-stream",
  ".spz": "application/octet-stream",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/index.html";
  const filePath = path.normalize(path.join(ROOT, urlPath));
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    return res.end("forbidden");
  }
  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404);
      return res.end("not found: " + urlPath);
    }
    const type = MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream";
    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      if (m) {
        let start = m[1] ? parseInt(m[1], 10) : 0;
        let end = m[2] ? parseInt(m[2], 10) : st.size - 1;
        if (m[1] === "" && m[2] !== "") {
          start = Math.max(0, st.size - parseInt(m[2], 10));
          end = st.size - 1;
        }
        end = Math.min(end, st.size - 1);
        if (start > end || start >= st.size) {
          res.writeHead(416, { "Content-Range": `bytes */${st.size}` });
          return res.end();
        }
        res.writeHead(206, {
          "Content-Type": type,
          "Content-Range": `bytes ${start}-${end}/${st.size}`,
          "Accept-Ranges": "bytes",
          "Content-Length": end - start + 1,
        });
        return fs.createReadStream(filePath, { start, end }).pipe(res);
      }
    }
    res.writeHead(200, {
      "Content-Type": type,
      "Content-Length": st.size,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-cache",
    });
    fs.createReadStream(filePath).pipe(res);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`viewer serving ${ROOT}`);
  console.log(`open http://localhost:${PORT}/`);
});
