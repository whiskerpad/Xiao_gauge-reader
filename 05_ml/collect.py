"""
collect.py - 機械学習用データセットの自動収集(司令塔: PC)

流れ: UNOで針を乱数の位置へ移動(MPA指令。ガタ取りの寄せは常に減少側で一定)
      → XIAO の /capture.jpg で白黒画像(640x480, JPEG)を撮る → 画像と正解をCSVに記録
XIAOのファームは変更不要(/capture.jpg は既存)。マーカー・解析設定は使わない(生画像を保存)。
切り抜き・縮小は学習側で行う。

  pip install pyserial requests

使い方:
  python collect.py --port COM8 --xiao http://192.168.11.5 --count 2000 --tag front_ceiling
  python collect.py ... --check                 1枚だけ撮って構図を確認(針は動かさない)
  python collect.py ... --out data\\20261118_xxx   既存フォルダに追記(中断からの再開)
主なオプション:
  --frames 1      1位置で撮る枚数(2以上で同じ正解の画像を複数保存)
  --settle 0.3    移動完了後に待つ秒数(XIAO側でも古いフレームを2枚捨てる)
  --rehome 300    この件数ごとに原点復帰(脱調による位置ずれの蓄積を防ぐ)
  --min 0 --max 1.0  針の範囲[MPa]
  --edge 0.2      この割合を範囲の両端(--edge-w 幅、既定0.05MPa)から選ぶ(端の学習データを増やす)

出力: data/<日時>_<tag>/
  images/000001_0.jpg ...   (番号_同一位置での撮影番号)
  labels.csv                idx,file,mpa,deg,steps,frame,time,move_s,cap_s,bytes,tag
  info.json                 収集条件と XIAO の /cam /config
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "01_stepper_needle"))
from needle import Needle  # noqa: E402

FIELDS = ["idx", "file", "mpa", "deg", "steps", "frame", "time", "move_s", "cap_s", "bytes", "tag"]


def capture(xiao, retries=3):
    last = None
    for i in range(retries):
        try:
            r = requests.get(f"{xiao}/capture.jpg", timeout=15)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                return r.content
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(1.0 + i)
    raise RuntimeError(f"capture failed: {last}")


def get_json(xiao, path):
    try:
        return requests.get(f"{xiao}{path}", timeout=10).json()
    except Exception as e:  # 記録用なので失敗しても続行
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--xiao", required=True)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--settle", type=float, default=0.3)
    ap.add_argument("--rehome", type=int, default=300)
    ap.add_argument("--min", type=float, default=0.0)
    ap.add_argument("--max", type=float, default=1.0)
    ap.add_argument("--edge", type=float, default=0.0)
    ap.add_argument("--edge-w", type=float, default=0.05)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--out", default=None, help="既存の出力フォルダに追記する")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="1枚撮って保存するだけ")
    args = ap.parse_args()
    xiao = args.xiao.rstrip("/")

    if args.check:
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        p = os.path.join(HERE, "data", "check.jpg")
        with open(p, "wb") as f:
            f.write(capture(xiao))
        print("saved", p)
        return

    if args.out:
        out = os.path.abspath(args.out)
    else:
        out = os.path.join(HERE, "data", datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.tag)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(out, "labels.csv")

    start_idx = 1
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            start_idx = int(rows[-1]["idx"]) + 1
    new_file = not os.path.exists(csv_path)

    seed = args.seed if args.seed is not None else time.time_ns() % (2**31)
    rng = random.Random(seed)

    info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args), "seed": seed, "start_idx": start_idx,
        "cam": get_json(xiao, "/cam"), "config": get_json(xiao, "/config"),
        "note": "正解 mpa/deg/steps は UNO の応答(指令位置)。MPA指令は最終の寄せを減少側で行う",
    }
    with open(os.path.join(out, f"info_{start_idx:06d}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    print(f"出力: {out}")
    print(f"UNO {args.port} に接続中...")
    nd = Needle(args.port)
    nd.home()

    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    t_start = time.time()
    done = 0
    try:
        for k in range(args.count):
            idx = start_idx + k
            if k > 0 and args.rehome > 0 and k % args.rehome == 0:
                print("  原点復帰")
                nd.home()
            if rng.random() < args.edge:
                lo, hi = (args.min, args.min + args.edge_w) if rng.random() < 0.5 else (args.max - args.edge_w, args.max)
                target = round(rng.uniform(lo, hi), 4)
            else:
                target = round(rng.uniform(args.min, args.max), 4)
            t0 = time.time()
            res = nd.move_mpa(target)
            move_s = time.time() - t0
            time.sleep(args.settle)
            for fr in range(args.frames):
                t1 = time.time()
                jpg = capture(xiao)
                cap_s = time.time() - t1
                name = f"{idx:06d}_{fr}.jpg"
                with open(os.path.join(img_dir, name), "wb") as g:
                    g.write(jpg)
                w.writerow({
                    "idx": idx, "file": f"images/{name}",
                    "mpa": res.get("mpa", target), "deg": res.get("deg", ""), "steps": res.get("steps", ""),
                    "frame": fr, "time": datetime.now().isoformat(timespec="seconds"),
                    "move_s": round(move_s, 2), "cap_s": round(cap_s, 2), "bytes": len(jpg), "tag": args.tag,
                })
            f.flush()
            done += 1
            if done % 20 == 0 or done == args.count:
                el = time.time() - t_start
                rest = el / done * (args.count - done)
                print(f"  {done}/{args.count}  経過 {el/60:.1f}分  残り約 {rest/60:.1f}分")
    except KeyboardInterrupt:
        print("中断しました(--out で同じフォルダを指定すると続きから追記できます)")
    finally:
        f.close()
        try:
            nd.off()
        except Exception:
            pass
        nd.close()
    print(f"完了 {done} 位置 → {csv_path}")


if __name__ == "__main__":
    main()
