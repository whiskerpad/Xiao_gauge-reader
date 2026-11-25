# -*- coding: utf-8 -*-
"""
augment_dataset.py - 正面から撮った計器の画像を「斜めから見た形」などに変形して、学習データを増やす

正解の値(針の位置)は変形しても変わらないので、元の画像1枚から何枚でも作れます。
変形の内容(すべて乱数で毎回変わる):
  - 文字盤の面を 0〜--tilt 度 傾けた見え方(傾く向きも乱数)… 斜めから撮った写真の代わり
  - 回転 ±--rot 度、拡大縮小 ±--scale、平行移動 ±--shift(画像の幅に対する割合)
  - 明るさ・コントラスト・ガンマ、照明の映り込み(明るいかすみ)、ぼけ、ノイズ(--no-photo で無効)

入力: collect.py の出力フォルダ(labels.csv に file と mpa の列、images/ に画像)
出力: 新しいフォルダ(images/ と labels.csv)。labels.csv は file, mpa, src(元画像), tilt, phi, rot, scale
      出力は collect.py と同じ形なので、そのまま prep.py などの学習側で使える

使い方(例):
  python augment_dataset.py data\\20261119_074420_front_ceiling data\\aug_front --count 5
  python augment_dataset.py <入力> <出力> --count 3 --tilt 30 --center 296,219 --seed 1
必要なもの: Python 3.9 以上、pip install numpy opencv-python
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np


def tilt_homography(w, h, cx, cy, tilt_deg, phi_deg, rot_deg, scale, tx, ty, f=None):
    """(cx, cy) を中心に、平面を phi 方向の軸まわりに tilt 度傾けて見た射影変換(3x3)を返す"""
    if f is None:
        f = 0.9 * max(w, h)                       # スマホ・小型カメラ程度の画角を仮定
    th, ph, ro = np.radians([tilt_deg, phi_deg, rot_deg])
    ax = np.array([np.cos(ph), np.sin(ph), 0.0])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K   # ロドリゲスの回転公式
    half = 0.25 * min(w, h)
    src, dst = [], []
    for X, Y in ((-half, -half), (half, -half), (half, half), (-half, half)):
        src.append((cx + X, cy + Y))
        P = R @ np.array([X, Y, 0.0]) + np.array([0.0, 0.0, f])
        u, v = f * P[0] / P[2], f * P[1] / P[2]                  # 透視投影
        c, s = np.cos(ro), np.sin(ro)
        u, v = scale * (c * u - s * v), scale * (s * u + c * v)
        dst.append((cx + u + tx, cy + v + ty))
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def photometric(img, rng):
    """明るさ・コントラスト・ガンマ・映り込み・ぼけ・ノイズ(0〜255 の画像)"""
    x = img.astype(np.float32) / 255.0
    x = np.clip(x, 1e-3, 1.0) ** rng.uniform(0.6, 1.7)
    m = x.mean()
    x = (x - m) * rng.uniform(0.6, 1.5) + m + rng.uniform(-0.15, 0.15)
    if rng.random() < 0.5:                                         # 照明の映り込み(明るいかすみ)
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        bx, by = rng.uniform(0, w), rng.uniform(0, h)
        sig = rng.uniform(0.15, 0.6) * min(w, h)
        blob = rng.uniform(0.2, 0.7) * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sig * sig))
        if x.ndim == 3:
            blob = blob[:, :, None]
        x = x * (1 - blob) + blob * rng.uniform(0.7, 1.0)
    x = np.clip(x, 0, 1) * 255.0
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    x = x + rng.normal(0, rng.uniform(0, 4), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="計器の画像を変形して学習データを増やす")
    ap.add_argument("src", help="入力フォルダ(labels.csv と images/)")
    ap.add_argument("dst", help="出力フォルダ(新しく作る)")
    ap.add_argument("--count", type=int, default=5, help="元の画像1枚から作る枚数")
    ap.add_argument("--tilt", type=float, default=35.0, help="傾きの最大[度]")
    ap.add_argument("--tilt-prob", type=float, default=0.85, help="傾けた画像にする割合")
    ap.add_argument("--rot", type=float, default=8.0, help="回転の最大[度]")
    ap.add_argument("--scale", type=float, default=0.15, help="拡大縮小の幅(0.15 = 0.85〜1.15倍)")
    ap.add_argument("--shift", type=float, default=0.05, help="平行移動の最大(画像幅に対する割合)")
    ap.add_argument("--center", default=None, help="文字盤の中心 x,y [画素](省略時は画像の中心)")
    ap.add_argument("--no-photo", action="store_true", help="明るさなどの変化を付けない")
    ap.add_argument("--gray", action="store_true", help="白黒にして保存する")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.src, "labels.csv"), encoding="utf-8")))
    if not rows or "file" not in rows[0] or "mpa" not in rows[0]:
        sys.exit("labels.csv に file と mpa の列が必要です")
    os.makedirs(os.path.join(a.dst, "images"), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    out_rows = []
    for k, r in enumerate(rows):
        path = os.path.join(a.src, r["file"])
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE if a.gray else cv2.IMREAD_COLOR)
        if img is None:
            print("読めない:", path)
            continue
        h, w = img.shape[:2]
        cx, cy = (float(v) for v in a.center.split(",")) if a.center else (w / 2.0, h / 2.0)
        base = os.path.splitext(os.path.basename(r["file"]))[0]
        for j in range(a.count):
            tilt = rng.uniform(0, a.tilt) if rng.random() < a.tilt_prob else 0.0
            phi = rng.uniform(0, 360)
            rot = rng.uniform(-a.rot, a.rot)
            sc = rng.uniform(1 - a.scale, 1 + a.scale)
            tx, ty = rng.uniform(-a.shift, a.shift, 2) * w
            H = tilt_homography(w, h, cx, cy, tilt, phi, rot, sc, tx, ty)
            out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if not a.no_photo:
                out = photometric(out, rng)
            name = f"{base}_a{j}.jpg"
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            buf.tofile(os.path.join(a.dst, "images", name))
            out_rows.append({"file": f"images/{name}", "mpa": r["mpa"], "src": r["file"],
                             "tilt": round(tilt, 1), "phi": round(phi, 1), "rot": round(rot, 1), "scale": round(sc, 3)})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(rows)} 枚目まで処理")
    with open(os.path.join(a.dst, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"完了: {len(rows)}枚 → {len(out_rows)}枚 を {a.dst} に保存")


if __name__ == "__main__":
    main()
