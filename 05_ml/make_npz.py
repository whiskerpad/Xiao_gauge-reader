# -*- coding: utf-8 -*-
"""make_npz.py - 収集した画像フォルダから、学習スクリプトが読む .npz を作る

学習スクリプトは 05_ml フォルダで実行し、同じフォルダの .npz を読みます。
  train_polar.py  … reg_<名前>.npz(極座標の帯画像。XIAO に載せる極座標CNN)
  train_persp.py  … h_<名前>.npz  (1/2縮小の画像。斜め変形で学習するモデル、PCのみ)
  train_phone.py  … cv_<名前>.npz (縁の楕円で大きさ・位置をそろえた画像。スマホ写真用)
学習スクリプトが探す名前: front / dark / desk / window(XIAO)、iphone(iPhone)、phone8(テスト用の写真)
ない名前は飛ばすので、まずは front だけでも動きます。

使い方(05_ml フォルダで):
  1) 位置合わせの基準を作る(最初に1回。正面・普段の照明のフォルダで。カメラを置き直したら --force で作り直す)
     python make_npz.py ref    data\\20261119_074420_front_ceiling
  2) 極座標CNN用
     python make_npz.py polar  data\\20261119_074420_front_ceiling --name front
  3) 斜め変形モデル用
     python make_npz.py persp  data\\20261119_074420_front_ceiling --name front
  4) スマホ用(XIAO の画像、iPhone の画像、切り抜いた写真)
     python make_npz.py canvas data\\20261119_074420_front_ceiling --name front
     python make_npz.py canvas data\\20261120_174418_iphone_ceiling data\\20261120_194937_iphone_ceiling --name iphone --cam ip
       (遠くて計器が小さく写った姿勢は、--roi 1:200,150,520,450 のように 姿勢番号:計器のまわりの四角 を指定)
     python make_npz.py canvas phone_test --name phone8 --cam photo
  --limit 20 で先頭20枚だけ(動作確認用)

入力: labels.csv(file と mpa の列。写真は mpa の代わりに truth でもよい)と画像
注意: polar の針の軸の位置(polar.py の CX, CY)と 0目盛り・振れ角(train_polar.py の A0, SPAN)は
      作者のカメラ位置での値です。カメラ位置や計器が違うときは calib_angles.py で求めて書き換えてください。
      canvas の XIAO 用の中心(--center)と縁の直径(--diam)も同じです。
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import canvas  # noqa: E402
import kuro    # noqa: E402
import polar   # noqa: E402
import reg2    # noqa: E402

REF = os.path.join(HERE, "models", "reg_ref.npy")


def rows_of(folder, cam=None):
    rows = list(csv.DictReader(open(os.path.join(folder, "labels.csv"), encoding="utf-8")))
    if cam in ("xi", "ip") and rows and "cam" in rows[0]:
        rows = [r for r in rows if r["cam"] == cam]
    return rows


def value(r):
    return float(r["mpa"] if "mpa" in r else r["truth"])


def gray(path):
    return np.asarray(Image.open(path).convert("L"))


def main():
    ap = argparse.ArgumentParser(description="収集フォルダ → 学習用 .npz")
    ap.add_argument("kind", choices=["ref", "polar", "persp", "canvas"])
    ap.add_argument("src", nargs="+", help="入力フォルダ(labels.csv がある所)。canvas の ip は複数可")
    ap.add_argument("--name", help="出力名(front, dark, desk, window, iphone, phone8 など)")
    ap.add_argument("--cam", default="xi", choices=["xi", "ip", "photo"], help="canvas: xi=XIAO, ip=iPhone収集, photo=写真1枚ずつ")
    ap.add_argument("--center", default="296,219", help="canvas xi: 基準画像での縁の中心 x,y")
    ap.add_argument("--diam", type=float, default=202.0, help="canvas xi: 縁の直径[画素]")
    ap.add_argument("--roi", action="append", default=[], help="canvas ip: 姿勢:x0,y0,x1,y1(例 1:200,150,520,450)。何回でも指定できる")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="ref: 既存の基準を上書きする")
    a = ap.parse_args()
    roi = {}
    for t in a.roi:
        p, box = t.split(":")
        roi[p] = tuple(int(v) for v in box.split(","))
    if a.kind != "ref" and not a.name:
        sys.exit("--name を指定してください")

    if a.kind == "ref":
        if os.path.exists(REF) and not a.force:
            sys.exit(f"{REF} はすでにあります(今のモデルはこの基準で学習済み)。作り直すときだけ --force を付ける")
        rows = rows_of(a.src[0])[: a.limit or 300]
        ref = np.mean([gray(os.path.join(a.src[0], r["file"])).astype(np.float32) for r in rows], 0)
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        np.save(REF, ref.astype(np.float32))
        print("基準を保存:", REF, f"({len(rows)}枚の平均)")
        return

    R = reg2.Ref(np.load(REF)) if os.path.exists(REF) and (a.kind == "polar" or a.cam == "xi") else None
    if a.kind == "polar" and R is None:
        sys.exit("先に python make_npz.py ref <フォルダ> で基準を作ってください")

    X, y, extra, pose = [], [], [], []
    for folder in a.src:
        rows = rows_of(folder, a.cam if a.kind == "canvas" else None)
        if a.limit:
            rows = rows[: a.limit]
        rim = {}
        stamp = (os.path.basename(os.path.normpath(folder)).split("_") + ["", ""])[1]
        if a.kind == "canvas" and a.cam == "ip":
            # 姿勢ごとに中央値の画像を作り、縁の楕円を1回だけ求める(1枚ずつより安定)
            for p in sorted({r["pose"] for r in rows}):
                rr = [r for r in rows if r["pose"] == p]
                med = np.median(np.stack([gray(os.path.join(folder, r["file"])) for r in rr[:60]]), 0).astype(np.uint8)
                x0, y0 = 0, 0
                if p in roi:                       # 計器が小さく写る姿勢は、計器のまわりだけで探す
                    x0, y0, x1, y1 = roi[p]
                    med = med[y0:y1, x0:x1]
                seed, pts = kuro.boundary(med)
                if len(pts) < 25:
                    print("縁が見つからない 姿勢", p, "(--roi で計器のまわりを指定)")
                    continue
                (cx, cy), (d1, d2), ang = kuro.fit_ellipse(pts)[0]
                cx, cy = cx + x0, cy + y0
                if max(d1, d2) > 0.9 * min(med.shape) or min(d1, d2) < 0.5 * max(d1, d2):
                    print(f"姿勢 {p}: 縁の楕円がおかしい(中心 {cx:.0f},{cy:.0f} 径 {d1:.0f}x{d2:.0f})→ 飛ばす。--roi で計器のまわりを指定")
                    continue
                rim[p] = ((cx, cy), (d1, d2), ang)
                print(f"姿勢 {p}: 縁の中心 ({cx:.0f},{cy:.0f}) 長径 {max(d1, d2):.0f}")
        for i, r in enumerate(rows):
            path = os.path.join(folder, r["file"])
            if a.kind == "polar":
                img = gray(path).astype(np.float32)
                dx, dy, s = reg2.shift(img, R)
                X.append(polar.prep_at(img, polar.CX + dx, polar.CY + dy)); extra.append((dx, dy, s))
            elif a.kind == "persp":
                X.append(cv2.resize(gray(path), (320, 240), interpolation=cv2.INTER_AREA))
            elif a.cam == "xi":
                g = gray(path)
                dx, dy = reg2.shift(g.astype(np.float32), R)[:2] if R is not None else (0, 0)
                cx, cy = (float(v) for v in a.center.split(","))
                X.append(canvas.to_canvas(g, cx + dx, cy + dy, a.diam))
            else:
                rgb = np.asarray(Image.open(path).convert("RGB"))
                if a.cam == "ip":
                    if r["pose"] not in rim:
                        continue
                    (cx, cy), (d1, d2), _ = rim[r["pose"]]
                    pose.append(f"{stamp}_{r['pose']}_{r.get('pose_note', '')}")
                else:
                    seed, pts = kuro.boundary(np.asarray(Image.open(path).convert("L")))
                    if len(pts) < 25:
                        print("縁が見つからない:", path)
                        continue
                    (cx, cy), (d1, d2), _ = kuro.fit_ellipse(pts)[0]
                X.append(canvas.to_canvas(np.ascontiguousarray(rgb[:, :, 0]), cx, cy, max(d1, d2)))
            y.append(value(r))
            if (i + 1) % 200 == 0:
                print(f"{folder}: {i + 1}/{len(rows)}")

    prefix = {"polar": "reg", "persp": "h", "canvas": "cv"}[a.kind]
    out = os.path.join(HERE, f"{prefix}_{a.name}.npz")
    y = np.array(y, np.float32)
    if a.kind == "polar":
        np.savez(out, P=np.stack(X).astype(np.float32), y=y, sh=np.array(extra))
    elif pose:
        np.savez(out, X=np.stack(X), y=y, pose=np.array(pose))
    else:
        np.savez(out, X=np.stack(X), y=y)
    print(f"保存: {out}  {len(y)}枚")


if __name__ == "__main__":
    main()
