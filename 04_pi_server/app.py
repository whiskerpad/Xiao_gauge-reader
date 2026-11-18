"""
app.py - 圧力計読み取り値の受信・保存・トレンド表示(Flask)

  XIAO(定期読み取り) --POST /api/reading--> このサーバー --> SQLite(data/gauge.db)
  疑似針の指令値(needle_wander.py) --POST /api/truth--> 同上(比較用の真値)
  ブラウザ: http://<このPCのIP>:5000/  トレンドグラフ(未取得は赤い点)

起動:
  python -m pip install flask requests
  python app.py --xiao http://192.168.11.5 --interval 5
    --xiao を付けると、起動時に XIAO の送信先をこのPCに設定し、定期読み取りを開始する
    (XIAO の /log/set?on=1&intv=5&n=3&url=http://<このPCのIP>:5000/api/reading)

PCで動かして確認し、最終的にはラズパイで同じものを動かす想定。
"""
import argparse
import csv
import io
import os
import socket
import sqlite3
import threading
import time
from urllib.parse import urlparse

from flask import Flask, Response, g, jsonify, request

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "data", "gauge.db")

app = Flask(__name__)
_lock = threading.Lock()


# ------------------------------------------------------------------ DB
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close(_exc):
    d = g.pop("db", None)
    if d is not None:
        d.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS reading(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,            -- 受信時刻(UNIX秒)
      dev TEXT, seq INTEGER,
      ok INTEGER NOT NULL,         -- 1=値あり 0=未取得
      value REAL, raw REAL, snr REAL, contrast REAL,
      mk_ok INTEGER, n INTEGER, okn INTEGER, ms INTEGER, uptime_s INTEGER,
      addr TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_reading_ts ON reading(ts);
    CREATE TABLE IF NOT EXISTS truth(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL, value REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_truth_ts ON truth(ts);
    """)
    con.commit()
    con.close()


# ------------------------------------------------------------------ API
@app.post("/api/reading")
def api_reading():
    d = request.get_json(force=True, silent=True) or {}
    ok = 1 if d.get("ok") else 0
    with _lock:
        db().execute(
            "INSERT INTO reading(ts,dev,seq,ok,value,raw,snr,contrast,mk_ok,n,okn,ms,uptime_s,addr)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), d.get("dev"), d.get("seq"), ok,
             d.get("value") if ok else None, d.get("raw") if ok else None,
             d.get("snr"), d.get("contrast"), d.get("mk_ok"), d.get("n"), d.get("okn"),
             d.get("ms"), d.get("uptime_s"), request.remote_addr))
        db().commit()
    return jsonify(status="ok")


@app.post("/api/truth")
def api_truth():
    d = request.get_json(force=True, silent=True) or {}
    if "value" not in d:
        return jsonify(status="error", msg="value required"), 400
    with _lock:
        db().execute("INSERT INTO truth(ts,value) VALUES(?,?)", (d.get("ts") or time.time(), float(d["value"])))
        db().commit()
    return jsonify(status="ok")


def _range():
    minutes = float(request.args.get("minutes", 30))
    t1 = time.time()
    return t1 - minutes * 60, t1


def _truth_at(truth, t):
    """t 以前で最後の指令値(なければ None)"""
    lo, hi = 0, len(truth) - 1
    if hi < 0 or truth[0][0] > t:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if truth[mid][0] <= t:
            lo = mid
        else:
            hi = mid - 1
    return truth[lo][1]


@app.get("/api/data")
def api_data():
    t0, t1 = _range()
    rows = db().execute("SELECT ts,ok,value,snr,ms,seq FROM reading WHERE ts>=? ORDER BY ts", (t0,)).fetchall()
    tr = db().execute("SELECT ts,value FROM truth WHERE ts>=? ORDER BY ts", (t0 - 600,)).fetchall()
    truth = [(r["ts"], r["value"]) for r in tr]
    ok_pts, ng_pts, errs = [], [], []
    for r in rows:
        if r["ok"]:
            ok_pts.append([r["ts"], r["value"]])
            # 読み取りは受信より ms 前から行っているので、その中間時刻の指令値と比べる
            tv = _truth_at(truth, r["ts"] - (r["ms"] or 0) / 2000.0)
            if tv is not None:
                errs.append(r["value"] - tv)
        else:
            ng_pts.append([r["ts"]])
    n = len(rows)
    stats = {
        "n": n, "ok": len(ok_pts), "ng": len(ng_pts),
        "ng_rate": (len(ng_pts) / n) if n else None,
        "last": dict(rows[-1]) if rows else None,
        "err_n": len(errs),
        "mae": (sum(abs(e) for e in errs) / len(errs)) if errs else None,
        "max_err": max((abs(e) for e in errs), default=None),
        "bias": (sum(errs) / len(errs)) if errs else None,
        "within2": (sum(1 for e in errs if abs(e) <= 0.02) / len(errs)) if errs else None,
    }
    return jsonify(t0=t0, t1=t1, ok=ok_pts, ng=ng_pts,
                   truth=[[t, v] for t, v in truth if t >= t0 - 600], stats=stats)


@app.get("/api/export.csv")
def api_export():
    t0, _ = _range()
    rows = db().execute("SELECT * FROM reading WHERE ts>=? ORDER BY ts", (t0,)).fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time"] + list(rows[0].keys()) if rows else ["time"])
    for r in rows:
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))] + list(r))
    return Response(buf.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=gauge_readings.csv"})


@app.get("/")
def index():
    return Response(open(os.path.join(BASE, "index.html"), encoding="utf-8").read(), mimetype="text/html")


# ------------------------------------------------------------------ 起動時に XIAO を設定
def my_ip_toward(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 80))
        return s.getsockname()[0]
    finally:
        s.close()


def setup_xiao(xiao, interval, n, port):
    import requests
    host = urlparse(xiao).hostname
    ip = my_ip_toward(socket.gethostbyname(host))
    url = f"http://{ip}:{port}/api/reading"
    r = requests.get(f"{xiao.rstrip('/')}/log/set",
                     params={"on": 1, "intv": interval, "n": n, "url": url}, timeout=10)
    print(f"XIAO 定期読み取り設定: {r.text}")
    return ip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--xiao", default="", help="例: http://192.168.11.5 (指定すると起動時に送信先を設定)")
    ap.add_argument("--interval", type=int, default=5, help="XIAO の読み取り間隔[秒]")
    ap.add_argument("--n", type=int, default=3, help="1回の読み取り枚数(中央値)")
    args = ap.parse_args()
    init_db()
    ip = "127.0.0.1"
    if args.xiao:
        try:
            ip = setup_xiao(args.xiao, args.interval, args.n, args.port)
        except Exception as e:
            print(f"XIAO の設定に失敗: {e}  (ブラウザで /log/set を開いて手動設定も可)")
    print(f"トレンド表示: http://{ip}:{args.port}/   (このPC: http://127.0.0.1:{args.port}/)")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
