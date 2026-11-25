# -*- coding: utf-8 -*-
"""phone_gauge_server.py - iPhone の Safari で計器を写し、円の枠に合わせて「読む」と圧力の値を返す

仕組み:
  iPhone: カメラ映像の中央に円を表示 → 黒い縁を円に合わせて「読む」→ 円のまわりを切り出した JPEG を PC へ送る
  PC    : 黒縁の内側の楕円を探す(05_ml/kuro.py)→ 大きさ・位置をそろえる(canvas.py)
          → 赤チャンネルの 96x96 → CNN(models/net_phone_v1.pt)→ 値と自信を返す
  縁が見つからない・おかしいときは、案内の円をそのまま縁として使う(利用者が円に合わせているため)

HTTPS の証明書は G:\\iphone_capture\\certs のもの(iPhone に信頼済みのローカルCA)を借りて使う。iphone_capture は変更しない。
  ポート: iPhone 用 https 8444、PC の案内ページ http://127.0.0.1:8766
使い方: start_phone_reader.bat(または python phone_gauge_server.py)→ PC のブラウザに出る QR を iPhone で読む
「読む」1回で 0.25 秒おきに3枚撮り、自信が下限以上の画像が2枚以上・その値の幅が SPREAD_MAX 以内のときだけ中央値を出す。

誤差の測定(疑似圧力計を使う):
  python phone_gauge_server.py --needle COM8 --targets 0.2,0.5,0.8 --per 10
  → 針を 0.2 に止め、「読む」10回ごとに次の値へ動かす。iPhone の画面で姿勢(正面・左30° など)を選んでから読む。
    results.csv に正解(truth)と姿勢(pose)が残る。集計: python summarize_phone.py
"""
import argparse
import base64
import csv
import io
import json
import os
import secrets
import socket
import ssl
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "05_ml"
sys.path.insert(0, str(ML))
import canvas  # noqa: E402
import kuro    # noqa: E402
from train_persp import Net, decode, norm  # noqa: E402

GUIDE_MARGIN = 1.5      # 切り出す正方形の1辺 = 案内の円の直径 × 1.5
SPREAD_MAX = 0.02       # 3枚の値の幅の上限 [MPa](= 2%FS)。超えたら未取得
LOG_COLS = ["time", "file", "value", "conf", "ok", "reason", "okn", "spread", "vals", "confs",
            "rim", "cx", "cy", "d", "w", "h", "ms", "truth", "pose"]
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


class Reader:
    def __init__(self, model_path, cf):
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()
        self.cf = cf
        self.lock = threading.Lock()

    def read(self, jpeg, guide=None):
        t0 = time.time()
        rgb = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
        if guide:                                   # 映像1枚 + 円の位置(gx, gy, gr) → 直径×GUIDE_MARGIN の正方形を切り出す
            gx, gy, gr = guide
            side = max(16, int(round(2 * gr * GUIDE_MARGIN)))
            x0, y0 = int(round(gx - side / 2)), int(round(gy - side / 2))
            H, W = rgb.shape[:2]
            crop = np.full((side, side, 3), 128, np.uint8)       # 映像の外は灰色
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
            if sx1 > sx0 and sy1 > sy0:
                crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
            if side > 960:
                crop = cv2.resize(crop, (960, 960), interpolation=cv2.INTER_AREA)
            rgb = crop
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        guide_d = min(w, h) / GUIDE_MARGIN
        src = "guide"
        cx, cy, d1, d2, ang = w / 2.0, h / 2.0, guide_d, guide_d, 0.0
        # 縁の検出は明るさの絶対値で判定するので、暗い写真は明るさを伸ばしてから探す(読み取りには元の画像を使う)
        lo, hi = np.percentile(g, (1, 99.5))
        gd = np.clip((g.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
        try:
            seed, pts = kuro.boundary(gd)
        except (ValueError, ZeroDivisionError):     # 白い所が無いなど → 案内の円を使う
            pts = []
        if len(pts) >= 25:
            (ex, ey), (e1, e2), ea = kuro.fit_ellipse(pts)[0]
            dmaj = max(e1, e2) if np.isfinite([ex, ey, e1, e2]).all() else 0.0
            # 案内の円に近い楕円だけ採用(中心のずれ 25% 以内、長径 0.6〜1.25倍)
            if (abs(ex - w / 2) < 0.25 * guide_d and abs(ey - h / 2) < 0.25 * guide_d
                    and 0.6 * guide_d < dmaj < 1.25 * guide_d and min(e1, e2) > 0.5 * dmaj):
                cx, cy, d1, d2, ang, src = ex, ey, e1, e2, ea, "rim"
        cv_ = canvas.to_canvas(np.ascontiguousarray(rgb[:, :, 0]), cx, cy, max(d1, d2))
        x = cv2.resize(cv_, (96, 96), interpolation=cv2.INTER_AREA)[None]
        with self.lock, torch.no_grad():
            v, c = decode(self.net(torch.tensor(norm(x)[:, None], dtype=torch.float32)))
        v, c = float(v[0]), float(c[0])
        ok = c >= self.cf and -0.05 < v < 1.05
        # 確認用: 切り出した画像に使った楕円を描く
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        col = (40, 200, 40) if src == "rim" else (0, 160, 255)
        cv2.ellipse(vis, ((cx, cy), (d1, d2), ang), col, max(2, w // 200))
        s = 360 / max(w, h)
        vis = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        okj, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return {"ok": bool(ok), "value": round(min(max(v, 0.0), 1.0) if ok else v, 3), "raw": v, "conf": round(c, 2), "cf": self.cf,
                "rim": src, "cx": round(cx, 1), "cy": round(cy, 1), "d": round(max(d1, d2), 1), "size": [w, h],
                "ms": int((time.time() - t0) * 1000),
                "check": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()}


PHONE_PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>圧力計を読む</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #f3f3f3; font-family: -apple-system, "Hiragino Sans", sans-serif; overflow: hidden; }
  #wrap { position: fixed; inset: 0; }
  video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; background: #000; }
  svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  #hint { position: absolute; left: 0; right: 0; top: max(14px, env(safe-area-inset-top)); text-align: center; font-size: 17px; text-shadow: 0 1px 3px #000; }
  #panel { position: absolute; left: 0; right: 0; bottom: 0; padding: 12px 16px max(18px, env(safe-area-inset-bottom)); background: rgba(0,0,0,.62); }
  #val { font-size: 44px; font-weight: 700; text-align: center; min-height: 52px; }
  #val small { font-size: 18px; font-weight: 400; margin-left: 4px; }
  #sub { text-align: center; font-size: 14px; color: #bbb; min-height: 1.3em; }
  button { display: block; width: 100%; padding: 16px; margin-top: 10px; border: 0; border-radius: 14px; font-size: 20px; font-weight: 600; background: #f4f4f4; color: #111; }
  button:disabled { background: #666; color: #ccc; }
  #meas { display: none; text-align: center; font-size: 15px; margin-top: 6px; }
  #meas select { font-size: 16px; padding: 4px; border-radius: 8px; }
  #meas button { display: inline-block; width: auto; padding: 6px 12px; margin: 0 0 0 6px; font-size: 14px; background: #444; color: #eee; }
  #check { display: none; position: absolute; right: 10px; top: 52px; width: 34vw; max-width: 180px; border: 2px solid #fff; border-radius: 8px; }
</style></head>
<body><div id="wrap">
  <video id="v" autoplay playsinline muted></video>
  <svg id="ov"><circle id="ring" fill="none" stroke="#38d26b" stroke-width="3" stroke-dasharray="10 8"/>
       <line id="h1" stroke="#38d26b" stroke-width="1.5"/><line id="h2" stroke="#38d26b" stroke-width="1.5"/></svg>
  <div id="hint">計器の黒い縁を、緑の円に合わせてください</div>
  <img id="check" alt="">
  <div id="panel"><div id="val"></div><div id="sub">「カメラ開始」を押してください</div>
    <div id="meas">姿勢 <select id="pose"><option>正面</option><option>左30</option><option>右30</option><option>上30</option><option>下30</option><option>その他</option></select>
      <span id="mstat"></span><button id="restart">最初から</button></div>
    <button id="go">カメラ開始</button><button id="auto" style="display:none">自動で続けて読む</button></div>
</div>
<script>
const KEY = __KEY__, MARGIN = __MARGIN__;
const v = document.getElementById("v"), go = document.getElementById("go");
const val = document.getElementById("val"), sub = document.getElementById("sub"), chk = document.getElementById("check");
let stream = null, busy = false;
function guide() {                       // 画面上の円(中心と半径)
  const w = innerWidth, h = innerHeight, r = 0.40 * Math.min(w, h);
  return { x: w / 2, y: h * 0.45, r: r };
}
function drawGuide() {
  const g = guide(), ring = document.getElementById("ring");
  ring.setAttribute("cx", g.x); ring.setAttribute("cy", g.y); ring.setAttribute("r", g.r);
  const l1 = document.getElementById("h1"), l2 = document.getElementById("h2");
  l1.setAttribute("x1", g.x - 12); l1.setAttribute("x2", g.x + 12); l1.setAttribute("y1", g.y); l1.setAttribute("y2", g.y);
  l2.setAttribute("x1", g.x); l2.setAttribute("x2", g.x); l2.setAttribute("y1", g.y - 12); l2.setAttribute("y2", g.y + 12);
}
addEventListener("resize", drawGuide); drawGuide();
async function startCam() {
  sub.textContent = "カメラを開始しています…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
  } catch (e) { sub.textContent = "カメラを開始できません(証明書の信頼設定・カメラの許可を確認)"; return; }
  v.srcObject = stream; await v.play();
  go.textContent = "読む"; sub.textContent = "縁を円に合わせて「読む」";
  autoBtn.style.display = "block";
}
function grabFrame() {                   // 映像1枚をそのまま(長辺1920まで)JPEG にし、円の位置を画像の座標で返す
  const vw = v.videoWidth, vh = v.videoHeight, ew = v.clientWidth, eh = v.clientHeight;
  const s = Math.max(ew / vw, eh / vh), ox = (ew - vw * s) / 2, oy = (eh - vh * s) / 2;
  const k = Math.min(1, 1920 / Math.max(vw, vh)), w = Math.round(vw * k), h = Math.round(vh * k);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(v, 0, 0, w, h);   // 切り出しは PC 側で行う(Safari の部分描画を使わない)
  const g = guide();
  const info = [(g.x - ox) / s * k, (g.y - oy) / s * k, g.r / s * k, vw, vh].map((x) => x.toFixed(1)).join(",");
  return new Promise((res) => c.toBlob((b) => res({ blob: b, info: info }), "image/jpeg", 0.92));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const poseSel = document.getElementById("pose"), meas = document.getElementById("meas"), mstat = document.getElementById("mstat");
try { const p = localStorage.getItem("pose"); if (p) poseSel.value = p; } catch (e) {}
poseSel.addEventListener("change", () => { try { localStorage.setItem("pose", poseSel.value); } catch (e) {} });
function showState(st) {
  if (!st || !st.meas) return;
  meas.style.display = "block";
  mstat.textContent = st.error ? st.error : st.done ? "測定終了" : st.moving ? "針を移動中…" : "目標 " + st.target.toFixed(2) + " MPa  " + st.n + "/" + st.per + "  (" + (st.idx + 1) + "/" + st.targets.length + ")";
}
async function pollState() {
  try { showState(await (await fetch("/state?k=" + KEY, { cache: "no-store" })).json()); } catch (e) {}
}
document.getElementById("restart").addEventListener("click", async () => {
  if (!confirm("測定を最初の目標値からやり直しますか?")) return;
  await fetch("/restart", { method: "POST", headers: { "X-Key": KEY } }); pollState();
});
async function readGauge() {
  if (busy || !v.videoWidth) return;
  busy = true; go.disabled = true; sub.textContent = "読み取り中…(3枚撮ります。動かさないで)";
  try {
    const fs = [];
    for (let i = 0; i < 3; i++) { if (i) await sleep(250); fs.push(await grabFrame()); }
    const body = new Blob(fs.map((f) => f.blob), { type: "application/octet-stream" });
    const r = await fetch("/read", { method: "POST", headers: { "X-Key": KEY, "X-Parts": fs.map((f) => f.blob.size).join(","),
      "X-Guide": fs.map((f) => f.info).join(";"), "X-Pose": encodeURIComponent(poseSel.value) }, body: body });
    const d = await r.json();
    if (d.error) { sub.textContent = d.error; pollState(); return; }
    val.innerHTML = d.ok ? d.value.toFixed(3) + "<small>MPa</small>" : "未取得";
    val.style.color = d.ok ? "#fff" : "#ff6b6b";
    sub.textContent = (d.ok ? "" : d.reason + " ／ ") + "3枚: " + d.vals.map((x, i) => x.toFixed(3) + "(" + d.confs[i].toFixed(2) + ")").join(" ") + " ／ 縁: " + (d.rim === "rim" ? "検出" : "円の枠") + " ／ " + d.ms + " ms";
    chk.src = d.check; chk.style.display = "block";
    showState(d.state);
  } catch (e) { sub.textContent = "PC に届きません(同じWi-Fi・サーバーの起動を確認)"; }
  finally { busy = false; go.disabled = auto; }
}
let auto = false;
const autoBtn = document.getElementById("auto");
async function autoLoop() {        // 三脚で固定したまま、読み取りを繰り返す(測定中は針の移動を待つ)
  while (auto) {
    let st = null;
    try { st = await (await fetch("/state?k=" + KEY, { cache: "no-store" })).json(); showState(st); } catch (e) {}
    if (st && st.meas && (st.done || st.error)) { setAuto(false); break; }
    if (st && st.moving) { await sleep(700); continue; }
    await readGauge();
    await sleep(800);
  }
}
function setAuto(on) {
  auto = on; autoBtn.textContent = on ? "自動を止める" : "自動で続けて読む";
  autoBtn.style.background = on ? "#e0533d" : ""; autoBtn.style.color = on ? "#fff" : "";
  go.disabled = on;
  if (on) autoLoop();
}
autoBtn.addEventListener("click", () => setAuto(!auto));
pollState();
go.addEventListener("click", () => (stream ? readGauge() : startCam()));
chk.addEventListener("click", () => { chk.style.display = "none"; });
</script></body></html>"""

PC_PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>圧力計を読む(iPhone)</title>
<style>body{font-family:"Yu Gothic UI",sans-serif;background:#1b1d22;color:#eee;margin:24px}
.q{display:inline-block;margin:0 24px 24px 0;text-align:center}.q div{background:#fff;padding:10px;border-radius:8px}
a{color:#8cc8ff}table{border-collapse:collapse;margin-top:12px}td,th{border:1px solid #444;padding:4px 10px;font-size:14px}
img.t{height:72px}</style></head><body>
<h2>iPhone で圧力計を読む</h2>
<p>iPhone のカメラで QR を読み、開いたページで「カメラ開始」→ 計器の黒い縁を緑の円に合わせて「読む」。<br>
iPhone と同じネットワークのアドレスの QR を使ってください。証明書は iphone_capture のもの(iPhone に信頼済み)です。</p>
<div id="qrs"></div>
<h3>最近の読み取り</h3><table id="log"></table>
<script src="/qrcode.js"></script>
<script>
const URLS = __URLS__;
const box = document.getElementById("qrs");
for (const u of URLS) {
  const d = document.createElement("div"); d.className = "q";
  let svg = "(qrcode.js が見つかりません)";
  if (window.qrcode) { const q = qrcode(0, "M"); q.addData(u.url); q.make(); svg = q.createSvgTag(5, 0); }
  d.innerHTML = "<div>" + svg + "</div><p>" + u.label + "<br><a href='" + u.url + "'>" + u.url.split("?")[0] + "</a></p>";
  box.appendChild(d);
}
async function refresh() {
  try {
    const rows = await (await fetch("/log.json", { cache: "no-store" })).json();
    const t = document.getElementById("log");
    t.innerHTML = "<tr><th>時刻</th><th>値</th><th>3枚の値(自信)</th><th>正解</th><th>姿勢</th><th>画像</th></tr>" + rows.map(r =>
      `<tr><td>${r.time}</td><td>${r.ok === "1" ? r.value : "未取得: " + (r.reason || "")}</td><td>${r.vals || ""} (${r.confs || r.conf || ""})</td><td>${r.truth || ""}</td><td>${r.pose || ""}</td>` +
      `<td><img class="t" src="/shots/${encodeURIComponent(r.file)}"></td></tr>`).join("");
  } catch (e) {}
}
refresh(); setInterval(refresh, 3000);
</script></body></html>"""


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.0.1", 9))          # 送信はしない(経路の確認だけ)
        ips.add(s.getsockname()[0]); s.close()
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def main():
    ap = argparse.ArgumentParser(description="iPhone で圧力計を読む(PC 側サーバー)")
    ap.add_argument("--certs", default=r"G:\iphone_capture\certs", help="iphone_capture の certs フォルダ")
    ap.add_argument("--qrjs", default=r"G:\iphone_capture\qrcode.js", help="QR 表示用の qrcode.js")
    ap.add_argument("--model", default=str(ML / "models" / "net_phone_v1.pt"))
    ap.add_argument("--cf", type=float, default=0.9, help="自信の下限(これ未満は未取得)。0.8 では取り違えが通ることがあった")
    ap.add_argument("--port", type=int, default=8444, help="iPhone 用 https ポート")
    ap.add_argument("--pc-port", type=int, default=8766, help="PC の案内ページ(127.0.0.1 のみ)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--needle", default="", help="誤差の測定: 疑似圧力計(UNO)の COM ポート(例 COM8)")
    ap.add_argument("--targets", default="0.2,0.5,0.8", help="誤差の測定: 針を止める値 [MPa]")
    ap.add_argument("--per", type=int, default=10, help="誤差の測定: 1つの値で読む回数")
    a = ap.parse_args()

    certs = Path(a.certs)
    chain, key = certs / "server-chain.pem", certs / "server.key"
    if not chain.is_file() or not key.is_file():
        sys.exit(f"証明書が見つかりません: {certs}(先に iphone_capture の start.bat を一度起動してください)")
    cert_ips = set((certs / "ips.txt").read_text(encoding="utf-8").split()) if (certs / "ips.txt").is_file() else set()
    reader = Reader(a.model, a.cf)
    key_tok = secrets.token_urlsafe(9)
    out_dir = HERE / "results" / time.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.csv"
    if log_path.exists():                       # 列の違う古い形式は別名にして残す
        with open(log_path, encoding="utf-8") as f:
            head = f.readline().strip().split(",")
        if head != LOG_COLS:
            log_path.rename(out_dir / time.strftime("results_old_%H%M%S.csv"))
    log_lock = threading.Lock()
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_COLS)

    # 誤差の測定: 疑似圧力計を目標値に止め、per 回読んだら次へ
    st = {"meas": bool(a.needle), "targets": [float(x) for x in a.targets.split(",")], "idx": 0, "n": 0,
          "per": a.per, "moving": False, "done": False, "error": ""}
    st["target"] = st["targets"][0]
    st_lock = threading.Lock()
    needle = None
    if a.needle:
        sys.path.insert(0, str(HERE.parent / "01_stepper_needle"))
        from needle import Needle  # noqa: E402
        print("疑似圧力計に接続:", a.needle)
        needle = Needle(a.needle)

    def move_to(i, home=False):
        with st_lock:
            st.update(idx=i, n=0, target=st["targets"][i], moving=True, done=False, error="")
        try:
            if home:
                print("原点復帰中…")
                needle.home()
            r = needle.move_mpa(st["targets"][i])
            print(f"針を {st['targets'][i]:.3f} MPa に移動  ({r.get('raw', '')})")
            time.sleep(1.0)                     # 針の揺れが収まるのを待つ
        except Exception as e:  # noqa: BLE001
            print("針を動かせません:", e)
            with st_lock:
                st["error"] = f"針を動かせません: {e}"
        finally:
            with st_lock:
                st["moving"] = False

    if needle:
        threading.Thread(target=move_to, args=(0, True), daemon=True).start()

    def recent(n=20):
        dirs = sorted((HERE / "results").glob("*/results.csv"))[-2:]
        rows = []
        for p in dirs:
            with open(p, encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
            if not lines:
                continue
            head = next(csv.reader(lines[:1]))
            rd = csv.DictReader(lines) if "file" in head else csv.DictReader(lines, fieldnames=LOG_COLS)   # 見出し行が無くても読む
            for r in rd:
                if r.get("file"):
                    r["file"] = p.parent.name + "/" + r["file"]
                    rows.append(r)
        return rows[-n:][::-1]

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path, _, query = self.path.partition("?")
            pc = self.server.server_address[1] == a.pc_port
            if pc and path == "/":
                urls = [{"label": ("iPhone 用" if ip in cert_ips else "証明書に無いアドレス(使えない可能性)") + " " + ip,
                         "url": f"https://{ip}:{a.port}/?k={key_tok}"} for ip in local_ips()]
                urls.sort(key=lambda u: "使えない" in u["label"])
                self.send(200, PC_PAGE.replace("__URLS__", json.dumps(urls)).encode("utf-8"), "text/html; charset=utf-8")
            elif pc and path == "/qrcode.js" and Path(a.qrjs).is_file():
                self.send(200, Path(a.qrjs).read_bytes(), "text/javascript")
            elif pc and path == "/log.json":
                self.json(recent())
            elif pc and path.startswith("/shots/"):
                p = (HERE / "results" / path[7:].replace("%2F", "/")).resolve()
                if p.is_file() and (HERE / "results") in p.parents:
                    self.send(200, p.read_bytes(), "image/jpeg")
                else:
                    self.send(404, b"not found", "text/plain")
            elif not pc and path == "/state" and query == f"k={key_tok}":
                with st_lock:
                    self.json(dict(st))
            elif not pc and path == "/" and query == f"k={key_tok}":
                page = PHONE_PAGE.replace("__KEY__", json.dumps(key_tok)).replace("__MARGIN__", str(GUIDE_MARGIN))
                self.send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self.send(404, "ページがありません(PC に出ている QR から開き直してください)".encode("utf-8"), "text/plain; charset=utf-8")

        def do_POST(self):
            if self.server.server_address[1] == a.pc_port or self.headers.get("X-Key") != key_tok:
                return self.json({"error": "キーが一致しません。PC の QR から開き直してください"}, 403)
            if self.path == "/restart":
                if needle:
                    threading.Thread(target=move_to, args=(0, True), daemon=True).start()
                return self.json({"ok": True})
            if self.path != "/read":
                return self.json({"error": "不明な要求"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n < 24_000_000:
                return self.json({"error": "画像の大きさが不正です"}, 400)
            data = self.rfile.read(n)
            with st_lock:
                if st["moving"]:
                    return self.json({"error": "針を移動中です。止まってから「読む」", "state": dict(st)})
                if st["error"]:
                    return self.json({"error": st["error"] + "(PC の画面を確認し、サーバーを起動し直す)", "state": dict(st)})
                if st["done"]:
                    return self.json({"error": "測定は終わりました(「最初から」でやり直し)", "state": dict(st)})
            try:
                sizes = [int(x) for x in (self.headers.get("X-Parts") or str(n)).split(",")]
            except ValueError:
                sizes = [n]
            guides = (self.headers.get("X-Guide") or "").split(";")
            parts, pos = [], 0
            for k, sz in enumerate(sizes):
                g = None
                try:
                    gv = [float(x) for x in guides[k].split(",")] if k < len(guides) else []
                    if len(gv) >= 3 and all(np.isfinite(gv[:3])) and gv[2] > 4:
                        g = gv[:3]
                except ValueError:
                    pass
                parts.append((data[pos:pos + sz], g)); pos += sz
            try:
                rs = [reader.read(b, g) for b, g in parts if b]
            except Exception as e:  # noqa: BLE001
                return self.json({"error": f"読み取りに失敗: {e}"}, 500)
            vals = [r["raw"] for r in rs]; confs = [r["conf"] for r in rs]
            acc = [r["raw"] for r in rs if r["ok"]]
            spread = (max(acc) - min(acc)) if acc else None
            if len(acc) < min(2, len(rs)):
                ok, reason = False, f"自信 {reader.cf} 以上が {len(acc)}/{len(rs)} 枚"
            elif spread is not None and spread > SPREAD_MAX:
                ok, reason = False, f"3枚の値がそろわない(幅 {spread:.3f})"
            else:
                ok, reason = True, ""
            value = float(np.median(acc)) if ok else float(np.median(vals))
            mid = int(np.argsort(vals)[len(vals) // 2])
            pose = ""
            try:
                from urllib.parse import unquote
                pose = unquote(self.headers.get("X-Pose") or "")[:20]
            except Exception:  # noqa: BLE001
                pass
            with st_lock:
                truth = st["target"] if st["meas"] else ""
                if st["meas"]:
                    st["n"] += 1
                    if st["n"] >= st["per"]:
                        if st["idx"] + 1 < len(st["targets"]):
                            threading.Thread(target=move_to, args=(st["idx"] + 1,), daemon=True).start()
                            st["moving"] = True
                        else:
                            st["done"] = True
                state = dict(st)
            name = time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.jpg"
            (out_dir / name).write_bytes(parts[mid][0])
            r0 = rs[mid]
            with log_lock, open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, round(min(max(value, 0.0), 1.0) if ok else value, 4),
                                        round(min(confs), 2), int(ok), reason, len(acc), "" if spread is None else round(spread, 4),
                                        " ".join(f"{x:.4f}" for x in vals), " ".join(f"{x:.2f}" for x in confs),
                                        r0["rim"], r0["cx"], r0["cy"], r0["d"], r0["size"][0], r0["size"][1],
                                        sum(r["ms"] for r in rs), truth, pose])
            print(f"{time.strftime('%H:%M:%S')}  {'値 %.3f MPa' % value if ok else '未取得(' + reason + ')'}  3枚 {' '.join(f'{x:.3f}' for x in vals)}"
                  + (f"  正解 {truth}" if truth != "" else "") + (f"  {pose}" if pose else ""))
            self.json({"ok": ok, "value": round(min(max(value, 0.0), 1.0) if ok else value, 3), "reason": reason,
                       "vals": [round(x, 3) for x in vals], "confs": confs, "cf": reader.cf, "rim": r0["rim"],
                       "ms": sum(r["ms"] for r in rs), "check": r0["check"], "state": state})

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(chain), str(key))
    https = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    https.socket = ctx.wrap_socket(https.socket, server_side=True)
    pc = ThreadingHTTPServer(("127.0.0.1", a.pc_port), H)
    threading.Thread(target=pc.serve_forever, daemon=True).start()
    pc_url = f"http://127.0.0.1:{a.pc_port}/"
    print("PC の案内ページ:", pc_url)
    for ip in local_ips():
        print(f"iPhone 用: https://{ip}:{a.port}/?k={key_tok}" + ("" if ip in cert_ips else "  (証明書に無いアドレス)"))
    print("結果の保存先:", out_dir)
    if not a.no_browser:
        webbrowser.open(pc_url)
    try:
        https.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
