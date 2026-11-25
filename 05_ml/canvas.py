"""canvas.py - 文字盤の縁(黒縁の内側の楕円)を基準に、大きさ・位置をそろえた 192x192 の画像にする
中心 = 縁の楕円の中心、1辺 = 長径 × K。カラー写真は赤チャンネル(赤い置針が白く飛ぶ)
"""
import numpy as np, cv2
K = 1.35; C = 192
def to_canvas(gray, cx, cy, dmajor):
    side = dmajor * K; s = C / side
    M = np.array([[s, 0, C / 2 - s * cx], [0, s, C / 2 - s * cy]], np.float32)
    return cv2.warpAffine(gray, M, (C, C), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
