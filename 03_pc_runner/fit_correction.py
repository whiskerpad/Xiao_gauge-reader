"""
fit_correction.py - 精度評価の結果から直線補正(指示値 = a * 生値 + b)を求める

  python fit_correction.py ..\\dataset\\xiao_s3\\<日時_tag>\\result.csv
  python fit_correction.py ..\\dataset\\xiao_s3\\<日時_tag>\\result.csv --apply --xiao http://192.168.11.5

- 値を出した点(ok=1)だけを使い、真値 = a*生値 + b を最小二乗で求める
- 生値は raw_mpa 列(補正前)。古いCSVで raw_mpa が無い場合は read_mpa を使う
  (その測定時の補正が a=1, b=0 であることが前提)
- 誤差が --outlier を超える点は誤読とみなして除外
- --apply でXIAOへ書き込む(背景は削除されない)
"""
import argparse
import csv
import math

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--xiao")
    ap.add_argument("--outlier", type=float, default=0.03)
    args = ap.parse_args()

    xs, ys = [], []
    with open(args.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("ok") != "1":
                continue
            raw = r.get("raw_mpa") or r.get("read_mpa")
            if not raw:
                continue
            x, y = float(raw), float(r["true_mpa"])
            if abs(x - y) > args.outlier:
                continue
            xs.append(x)
            ys.append(y)
    n = len(xs)
    if n < 5:
        print(f"有効な点が少なすぎます({n}点)")
        return
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx

    before = [x - y for x, y in zip(xs, ys)]
    after = [a * x + b - y for x, y in zip(xs, ys)]
    rms = lambda e: math.sqrt(sum(v * v for v in e) / len(e))
    print(f"使用点数: {n}")
    print(f"補正式: 指示値 = {a:.5f} × 生値 + {b:+.5f}")
    print(f"補正前: 平均 {sum(before)/n:+.4f}  RMSE {rms(before):.4f}  最大 {max(map(abs, before)):.4f} MPa")
    print(f"補正後: 平均 {sum(after)/n:+.4f}  RMSE {rms(after):.4f}  最大 {max(map(abs, after)):.4f} MPa")

    if args.apply:
        if not args.xiao:
            print("--apply には --xiao が必要です")
            return
        r = requests.get(f"{args.xiao.rstrip('/')}/config/set", params={"ca": f"{a:.5f}", "cb": f"{b:.5f}"}, timeout=10)
        cfg = r.json()
        print("XIAOへ書き込み:", {"ca": cfg.get("ca"), "cb": cfg.get("cb"), "bg_valid": cfg.get("bg_valid")})


if __name__ == "__main__":
    main()
