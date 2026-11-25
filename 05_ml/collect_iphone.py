"""
collect_iphone.py - iPhone(iphone_capture)で機械学習用データを自動収集する(司令塔: PC)

流れ: UNOで針を乱数の位置へ移動(MPA指令) → iphone_capture に撮影を指示(/ctrl/cmd shoot)
      → captures/ に届いた新しい静止画を取得 → 画像と正解をCSVに記録
      --xiao を付けると、同じ位置で XIAO の画像も撮る(カメラ比較用の組データ)
姿勢(iPhoneの位置・角度)は手で変える。--count 枚ごとに止まるので、iPhoneを動かして Enter。

準備:
  1. G:\\iphone_capture\\start.bat を起動し、iPhoneで撮影ページを開いて「カメラを開始」
  2. iPhoneは給電し、画面の自動ロックを「なし」にする。撮影ページを前面に出したままにする
  pip install pyserial requests

使い方:
  python collect_iphone.py --port COM8 --poses 5 --count 200 --tag iphone_ceiling
  python collect_iphone.py ... --xiao http://192.168.11.5      XIAOも同時に撮る
  python collect_iphone.py ... --check                         iPhoneで1枚撮るだけ(針は動かさない)
  python collect_iphone.py ... --out data\\<フォルダ> --pose-start 6   既存フォルダに姿勢6から追記
主なオプション:
  --iphone-dir G:\\iphone_capture   page.url(接続先とキー)を読む場所
  --quality 1080                    iPhoneの画質(720/1080/2160)。開始時に指示する
  --settle 0.8                      移動後に待つ秒数(iPhoneのピント・露出が落ち着くまで)

出力: data/<日時>_<tag>/
  images/NNNNNN_ip.jpg (iPhone)、NNNNNN_xi.jpg (XIAO、--xiao 時)
  labels.csv   idx,file,cam,mpa,deg,steps,pose,pose_note,width,height,time,move_s,cap_s,bytes,tag
  poses.csv    pose,note,start_idx,time
"""
import argparse
import csv
import json
import os
import random
import struct
import sys
import time
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "01_stepper_needle"))
from needle import Needle  # noqa: E402

FIELDS = ["idx", "file", "cam", "mpa", "deg", "steps", "pose", "pose_note", "width", "height",
          "time", "move_s", "cap_s", "bytes", "tag"]


def jpeg_size(data):
    """JPEGの幅・高さ(SOFマーカーから)。読めなければ (0, 0)"""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return 0, 0


class IPhone:
    def __init__(self, iphone_dir):
        url_file = os.path.join(iphone_dir, "page.url")
        if not os.path.exists(url_file):
            raise SystemExit(f"{url_file} がありません。iphone_capture の start.bat を先に起動してください")
        url = open(url_file, encoding="ascii").read().strip()      # http://127.0.0.1:8765/k/<token>
        base, _, token = url.partition("/k/")
        self.base = base.rstrip("/")
        self.h = {"X-Token": token.strip("/")}

    def state(self):
        return requests.get(f"{self.base}/ctrl/state", headers=self.h, timeout=5).json()

    def cmd(self, action, value=None):
        r = requests.post(f"{self.base}/ctrl/cmd", headers=self.h,
                          data=json.dumps({"action": action, "value": value}), timeout=5)
        r.raise_for_status()
        return r.json()

    def latest_name(self):
        return requests.get(f"{self.base}/latest", headers=self.h, timeout=5).json().get("file")

    def shoot(self, timeout=20.0):
        """撮影を指示し、新しく届いた画像のバイト列を返す"""
        before = self.latest_name()
        res = self.cmd("shoot")
        if not res.get("online"):
            raise RuntimeError("iPhoneの撮影ページが応答していません(画面ロック・Safariが裏に回っていないか確認)")
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.3)
            name = self.latest_name()
            if name and name != before:
                time.sleep(0.2)   # 書き込み完了待ち
                r = requests.get(f"{self.base}/shots/{name}", headers=self.h, timeout=20)
                if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                    return r.content, name
        raise RuntimeError("iPhoneの画像が届きません(タイムアウト)")


def xiao_capture(xiao):
    r = requests.get(f"{xiao}/capture.jpg", timeout=15)
    r.raise_for_status()
    return r.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--iphone-dir", default=r"G:\iphone_capture")
    ap.add_argument("--xiao", default=None)
    ap.add_argument("--poses", type=int, default=5, help="姿勢の数(姿勢ごとに止まる)")
    ap.add_argument("--count", type=int, default=200, help="1姿勢あたりの位置数")
    ap.add_argument("--pose-start", type=int, default=1)
    ap.add_argument("--quality", default="1080", choices=["720", "1080", "2160"])
    ap.add_argument("--settle", type=float, default=0.8)
    ap.add_argument("--rehome", type=int, default=300)
    ap.add_argument("--min", type=float, default=0.0)
    ap.add_argument("--max", type=float, default=1.0)
    ap.add_argument("--tag", default="iphone")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    ip = IPhone(args.iphone_dir)
    st = ip.state()
    s = st.get("state") or {}
    if not s or (st.get("age") or 99) > 5:
        raise SystemExit("iPhoneの撮影ページが接続されていません。iPhoneで撮影ページを開いてください")
    if not s.get("running"):
        raise SystemExit("iPhoneのカメラが停止しています。iPhoneで「カメラを開始」を押してください")
    if str(s.get("quality")) != args.quality:
        ip.cmd("quality", args.quality)
        time.sleep(3.0)
    print(f"iPhone 接続OK: {s.get('width')}x{s.get('height')} facing={s.get('facing')}")

    if args.check:
        os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
        data, name = ip.shoot()
        p = os.path.join(HERE, "data", "check_iphone.jpg")
        with open(p, "wb") as f:
            f.write(data)
        print("saved", p, jpeg_size(data))
        return

    xiao = args.xiao.rstrip("/") if args.xiao else None
    out = os.path.abspath(args.out) if args.out else os.path.join(
        HERE, "data", datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.tag)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(out, "labels.csv")
    pose_path = os.path.join(out, "poses.csv")

    idx = 1
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            idx = int(rows[-1]["idx"]) + 1
    new_csv = not os.path.exists(csv_path)
    new_pose = not os.path.exists(pose_path)

    seed = args.seed if args.seed is not None else time.time_ns() % (2**31)
    rng = random.Random(seed)
    with open(os.path.join(out, f"info_{idx:06d}.json"), "w", encoding="utf-8") as f:
        json.dump({"started": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "seed": seed,
                   "iphone_state": s, "note": "正解は UNO の指令位置。MPA指令は最終の寄せを減少側で行う"},
                  f, ensure_ascii=False, indent=1)

    print(f"出力: {out}")
    nd = Needle(args.port)
    nd.home()
    fc = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fc, fieldnames=FIELDS)
    if new_csv:
        w.writeheader()
    fp = open(pose_path, "a", newline="", encoding="utf-8")
    wp = csv.writer(fp)
    if new_pose:
        wp.writerow(["pose", "note", "start_idx", "time"])

    since_home = 0
    try:
        for pose in range(args.pose_start, args.pose_start + args.poses):
            print("\a")
            note = input(f"[姿勢 {pose}] iPhoneを置き直し、構図を確認してから メモ(例: 斜め右30cm)を入力して Enter: ").strip()
            wp.writerow([pose, note, idx, datetime.now().isoformat(timespec="seconds")])
            fp.flush()
            t_pose = time.time()
            for k in range(args.count):
                if args.rehome > 0 and since_home >= args.rehome:
                    print("  原点復帰")
                    nd.home()
                    since_home = 0
                target = round(rng.uniform(args.min, args.max), 4)
                t0 = time.time()
                res = nd.move_mpa(target)
                move_s = time.time() - t0
                since_home += 1
                time.sleep(args.settle)
                shots = []
                t1 = time.time()
                try:
                    data, _ = ip.shoot()
                except RuntimeError as e:
                    print("  再試行:", e)
                    time.sleep(2.0)
                    data, _ = ip.shoot()
                shots.append(("ip", data, time.time() - t1))
                if xiao:
                    t2 = time.time()
                    shots.append(("xi", xiao_capture(xiao), time.time() - t2))
                for cam, data, cap_s in shots:
                    name = f"{idx:06d}_{cam}.jpg"
                    with open(os.path.join(img_dir, name), "wb") as g:
                        g.write(data)
                    wd, ht = jpeg_size(data)
                    w.writerow({"idx": idx, "file": f"images/{name}", "cam": cam,
                                "mpa": res.get("mpa", target), "deg": res.get("deg", ""), "steps": res.get("steps", ""),
                                "pose": pose, "pose_note": note, "width": wd, "height": ht,
                                "time": datetime.now().isoformat(timespec="seconds"),
                                "move_s": round(move_s, 2), "cap_s": round(cap_s, 2), "bytes": len(data), "tag": args.tag})
                fc.flush()
                idx += 1
                if (k + 1) % 20 == 0 or k + 1 == args.count:
                    el = time.time() - t_pose
                    print(f"  姿勢{pose}: {k+1}/{args.count}  残り約 {el/(k+1)*(args.count-k-1)/60:.1f}分")
    except KeyboardInterrupt:
        print("中断しました(--out と --pose-start で続きから追記できます)")
    finally:
        fc.close()
        fp.close()
        try:
            nd.off()
        except Exception:
            pass
        nd.close()
    print(f"完了 → {csv_path}")


if __name__ == "__main__":
    main()
