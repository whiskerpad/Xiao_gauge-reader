# -*- coding: utf-8 -*-
"""calib_angles.py - 自分のカメラ・計器に合わせて、極座標CNN の定数(針の軸の位置、0目盛りの角度、振れ角)を求める

画像(make_npz.py ref の元にしたフォルダの1枚など)をペイント等で開き、次の3点の画素の位置を読み取って渡す。
  軸   : 針の回転の中心
  ゼロ : 0 の目盛り線の先(外側の端)
  満点 : 最大値(例 1.0MPa)の目盛り線の先
使い方:
  python calib_angles.py --pivot 296,220 --zero 214,309 --full 384,307
出力された CX, CY を polar.py に、A0, SPAN を train_polar.py に書き換える(XIAO 用は export_polar.py の後、
nn_polar_data.h にも同じ値が入る)。角度は画像の座標(右=0°、下=90°、時計回りに増える)。
"""
import argparse
import math


def ang(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0


def xy(t):
    x, y = (float(v) for v in t.split(","))
    return x, y


ap = argparse.ArgumentParser()
ap.add_argument("--pivot", required=True, help="針の軸 x,y")
ap.add_argument("--zero", required=True, help="0目盛りの先 x,y")
ap.add_argument("--full", required=True, help="最大目盛りの先 x,y")
a = ap.parse_args()
p, z, f = xy(a.pivot), xy(a.zero), xy(a.full)
a0 = ang(p, z)
span = (ang(p, f) - a0) % 360.0
r = (math.dist(p, z) + math.dist(p, f)) / 2
print(f"CX, CY = {p[0]:.1f}, {p[1]:.1f}")
print(f"A0, SPAN = {a0:.2f}, {span:.2f}")
print(f"目盛りの先までの半径 約{r:.0f}画素(polar.py の帯は 半径24〜101.5画素。大きく違うときは R0, DR も調整)")
if not 180 < span < 330:
    print("注意: 振れ角が普通の圧力計(約270°)と大きく違います。点の順番(ゼロと満点)や位置を確認してください")
