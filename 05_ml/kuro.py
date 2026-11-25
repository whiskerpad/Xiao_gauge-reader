# -*- coding: utf-8 -*-
"""Very-white center, across the gray face, then the sudden drop to black.

That drop is the inner edge of the black rim. An ellipse is fitted to it.
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image


def boundary(gray):
    h, w = gray.shape
    soft = cv2.GaussianBlur(gray, (0, 0), 1.2)
    yy, xx = np.ogrid[:h, :w]
    cx0, cy0 = w / 2.0, h / 2.0
    central = (xx - cx0) ** 2 + (yy - cy0) ** 2 < (0.22 * min(h, w)) ** 2
    white = (soft > 175) & central
    if int(white.sum()) < 40:
        white = soft > 175
    ys, xs = np.where(white)
    sx, sy = float(xs.mean()), float(ys.mean())
    rmax = int(max(h, w) * 0.78)
    rows = []
    for deg in range(0, 360, 2):
        th = np.radians(deg)
        si, co = np.sin(th), np.cos(th)
        xsamp, ysamp = [], []
        for r in range(0, rmax):
            x = int(round(sx + r * si))
            y = int(round(sy - r * co))
            if not (0 <= x < w and 0 <= y < h):
                break
            xsamp.append(x)
            ysamp.append(y)
        if len(xsamp) < 30:
            continue
        prof = soft[ysamp, xsamp].astype(np.float32)
        for i in range(8, len(prof) - 16):
            inner = float(prof[i])
            if inner < 85:
                continue
            outer = float(prof[i + 4 : i + 8].mean())
            if outer > 58 or inner - outer < 32:
                continue
            window = prof[i + 3 : i + 18]
            if float((window < 68).mean()) < 0.72:
                continue
            rows.append((deg, sx + i * si, sy - i * co, i))
            break
    pts = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4))
    return (sx, sy), pts


def fit_ellipse(rows):
    rad = rows[:, 3].copy()
    keep = np.ones(len(rows), dtype=bool)
    for i in range(len(rows)):
        idx = [(i + d) % len(rows) for d in range(-5, 6)]
        local = float(np.median(rad[idx]))
        if abs(rad[i] - local) > max(14.0, 0.14 * local):
            keep[i] = False
    kept = rows[keep] if keep.sum() >= 20 else rows
    ellipse = cv2.fitEllipse(kept[:, 1:3].astype(np.float32).reshape(-1, 1, 2))
    return ellipse, kept


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else r"G:\N100視覚読取\写真\iPhone"
    out = sys.argv[2] if len(sys.argv) > 2 else r"G:\N100視覚読取\画像処理\out\黒縁楕円"
    os.makedirs(out, exist_ok=True)
    sheets = []
    for name in sorted(os.listdir(src)):
        if not name.lower().endswith(".png"):
            continue
        rgb = np.asarray(Image.open(os.path.join(src, name)).convert("RGB"))
        gray = np.asarray(Image.open(os.path.join(src, name)).convert("L"))
        seed, rows = boundary(gray)
        if len(rows) < 25:
            print(name, "点が足りない")
            continue
        ellipse, kept = fit_ellipse(rows)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.ellipse(vis, ellipse, (40, 210, 40), 2)
        for x, y in kept[:, 1:3]:
            cv2.circle(vis, (int(round(x)), int(round(y))), 2, (240, 170, 30), -1)
        cv2.circle(vis, (int(seed[0]), int(seed[1])), 4, (220, 30, 30), -1)
        Image.fromarray(vis).save(os.path.join(out, "gray_" + name))
        color = rgb.copy()
        cv2.ellipse(color, ellipse, (40, 210, 40), 2)
        Image.fromarray(color).save(os.path.join(out, name))
        (cx, cy), (d1, d2), _ = ellipse
        print(name, "中心", round(cx), round(cy), "径", round(d1), round(d2), "点", len(kept))
        thumb = Image.fromarray(vis)
        thumb.thumbnail((340, 320))
        sheets.append(thumb)
    if sheets:
        board = Image.new("RGB", (340 * 4, 320 * 2), (20, 20, 20))
        for i, im in enumerate(sheets):
            board.paste(im, ((i % 4) * 340, (i // 4) * 320))
        board.save(os.path.join(out, "一覧.png"))
    print("保存", out)


if __name__ == "__main__":
    main()
