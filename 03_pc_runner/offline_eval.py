"""
offline_eval.py - 保存済み画像(run_eval.py --save-all)を使い、PC上で読み取り方式を検証する

XIAOに書き込む前に、帯の半径やハイパス幅などの効果を同じ画像で比較するためのツール。
XIAOの gauge.h の処理(角度0.5°刻み・[1,2,1]平滑化・暗さ%・ハイパス・帯の最小値・
snr判定)をnumpyで再現している。

  pip install numpy pillow
  python offline_eval.py ..\\dataset\\xiao_s3\\<日時_tag>
  python offline_eval.py ..\\dataset\\xiao_s3\\<日時_tag> --bands 50-70,75-95,100-112 --hpw 8
  python offline_eval.py ..\\dataset\\xiao_s3\\<日時_tag> --bands 100-112 --hpw 0   # 単一帯・背景なし

注意:
  - 保存画像はJPEG(XIAO上の生画像より劣化している)。結果はXIAO実機と少し異なる
  - 同じ画像で帯の半径を選ぶと「その画像に合わせすぎ」になる。必ず別の測定で確認すること
  - 中心・目盛り角は測定時の xiao_config.json の値を使う
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image

STEP, MARGIN = 0.5, 5.0
OK_MIN_DARK, OK_SNR, SNR_EXCL = 8.0, 1.5, 6.0


def load(dirpath):
    cfg = json.load(open(os.path.join(dirpath, "xiao_config.json"), encoding="utf-8"))["config"]
    items = []
    for f in sorted(glob.glob(os.path.join(dirpath, "images", "*.jpg"))):
        m = re.search(r"_(-?\d+\.\d+)\.jpg$", f)
        if m:
            items.append((float(m.group(1)), np.asarray(Image.open(f).convert("L"), dtype=float)))
    return cfg, items


def run(cfg, items, bands, hpw, tol):
    span = (cfg["a1"] - cfg["a0"]) % 360 or 360
    rels = np.arange(-MARGIN, span + MARGIN + 1e-6, STEP)
    th = np.radians(cfg["a0"] + rels)[:, None]

    def prof(img, r0, r1):
        rr = np.arange(r0, r1 + 1e-6, 1.0)[None, :]
        x = np.clip(np.rint(cfg["cx"] + rr * np.sin(th)).astype(int), 0, img.shape[1] - 1)
        y = np.clip(np.rint(cfg["cy"] - rr * np.cos(th)).astype(int), 0, img.shape[0] - 1)
        p = img[y, x].mean(1)
        q = p.copy()
        q[1:-1] = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
        return q

    def dark(p):
        d = (1 - p / max(np.median(p), 1)) * 100
        k = int(round(hpw / STEP))
        if k <= 0:
            return d
        return np.array([d[i] - np.median(d[max(0, i - k):i + k + 1]) for i in range(len(d))])

    res = {"ok": 0, "good": 0, "bad": 0, "errs": [], "n": len(items)}
    for t, img in items:
        d = np.minimum.reduce([dark(prof(img, a, b)) for a, b in bands])
        pk = int(np.argmax(d)); peak = d[pk]; thr = peak / 2
        L = pk
        while L > 0 and d[L - 1] > thr: L -= 1
        R = pk
        while R < len(d) - 1 and d[R + 1] > thr: R += 1
        w = np.clip(d[L:R + 1] - thr, 0, None)
        rel = (w * rels[L:R + 1]).sum() / w.sum() if w.sum() > 0 else rels[pk]
        other = d[np.abs(rels - rel) > SNR_EXCL].max()
        if peak >= OK_MIN_DARK and peak / max(other, 0.01) >= OK_SNR:
            res["ok"] += 1
            v = cfg["vmin"] + rel / span * (cfg["vmax"] - cfg["vmin"])
            e = v - t
            if abs(e) <= tol:
                res["good"] += 1; res["errs"].append(e)
            else:
                res["bad"] += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--bands", default="50-70,75-95,100-112", help="帯 r0-r1 をカンマ区切り")
    ap.add_argument("--hpw", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=0.02, help="正解とみなす誤差(目盛り値)")
    args = ap.parse_args()
    bands = [tuple(float(x) for x in b.split("-")) for b in args.bands.split(",")]
    cfg, items = load(args.dir)
    r = run(cfg, items, bands, args.hpw, args.tol)
    e = r["errs"]
    print(f"画像 {r['n']}枚  帯 {bands}  ハイパス±{args.hpw}°")
    print(f"値を出した {r['ok']}  正解(±{args.tol}) {r['good']}  誤読 {r['bad']}  "
          f"点数 {r['good'] / max(r['n'], 1) * 100:.0f}")
    if e:
        print(f"誤差(生値): 平均 {np.mean(e):+.4f}  最大 {np.max(np.abs(e)):.4f}")


if __name__ == "__main__":
    main()
