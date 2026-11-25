"""polar.py - 文字盤を 角度×半径 の帯画像にし、照明の影響を消す前処理(XIAO と同じ計算にする)
帯: 角度 0〜358°(2°刻み、180列、画像座標で右=0°・下=90°)× 半径 R0〜(32行、2.5px刻み)
暗さ: 各行(リング)で 周りの代表値(中央値)に対する比 d=(med-v)/(med+8)、
      さらに角度方向に ±HP 列の移動中央値を引く(ゆるやかな明暗を消し、細い暗いものを残す)
"""
import numpy as np
NA, NR, R0, DR, HP = 180, 32, 24.0, 2.5, 5
CX, CY = 295.7, 219.6
_th = np.radians(np.arange(NA) * 2.0)
_r = R0 + DR * np.arange(NR)
PX = CX + _r[:, None] * np.cos(_th)[None, :]
PY = CY + _r[:, None] * np.sin(_th)[None, :]
def sample(img):
    x0 = np.floor(PX).astype(int); y0 = np.floor(PY).astype(int); fx = PX - x0; fy = PY - y0
    a = img[y0, x0]; b = img[y0, x0 + 1]; c = img[y0 + 1, x0]; d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
def darkness(p):
    med = np.median(p, 1, keepdims=True)
    d = (med - p) / (med + 8.0)
    idx = (np.arange(NA)[:, None] + np.arange(-HP, HP + 1)[None, :]) % NA
    return d - np.median(d[:, idx], 2)
def prep(img):
    return darkness(sample(img.astype(np.float32))).astype(np.float32)

# ---- 文字盤の中心を画像から探す(カメラ・計器のずれに追従)----
# 明るい文字盤(半径 RI1〜RI2)と暗いケースの縁(半径 RO1〜RO2)の差が最大になる位置
RI1, RI2, RO1, RO2 = 80.0, 96.0, 106.0, 122.0
_ca = np.radians(np.arange(0, 360, 5.0))
_ri = np.linspace(RI1, RI2, 3); _ro = np.linspace(RO1, RO2, 3)
def _score(img, cx, cy):
    """文字盤の縁(半径 RI2 と RO1 の間)で、内側と外側の明るさの差の大きさを全周で平均(明暗の向きは問わない)"""
    H, W = img.shape
    def ring(rs):
        x = np.clip((cx + rs[:, None] * np.cos(_ca)[None]).astype(int), 0, W - 1); y = np.clip((cy + rs[:, None] * np.sin(_ca)[None]).astype(int), 0, H - 1)
        return img[y, x].mean(0)
    return float(np.abs(ring(_ri) - ring(_ro)).mean())
def find_center(img, gx=CX, gy=CY, span=60, step=4):
    img = img.astype(np.float32); best = (-1e9, gx, gy)
    for cy in np.arange(gy - span, gy + span + 1, step):
        for cx in np.arange(gx - span, gx + span + 1, step):
            s = _score(img, cx, cy)
            if s > best[0]: best = (s, cx, cy)
    _, bx, by = best
    for cy in np.arange(by - step, by + step + 0.5, 1.0):
        for cx in np.arange(bx - step, bx + step + 0.5, 1.0):
            s = _score(img, cx, cy)
            if s > best[0]: best = (s, cx, cy)
    return best[1], best[2], best[0]
def prep_auto(img, off=(0.0, 0.0)):
    """中心を探してから帯画像にする。off は (ケース中心 → 針の軸) のずれ[px]"""
    cx, cy, s = find_center(img)
    return prep_at(img, cx + off[0], cy + off[1]), (cx, cy, s)
def prep_at(img, cx, cy):
    th = np.radians(np.arange(NA) * 2.0); r = R0 + DR * np.arange(NR)
    px = cx + r[:, None] * np.cos(th)[None]; py = cy + r[:, None] * np.sin(th)[None]
    im = img.astype(np.float32)
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int); fx = px - x0; fy = py - y0
    a = im[y0, x0]; b = im[y0, x0 + 1]; c = im[y0 + 1, x0]; d = im[y0 + 1, x0 + 1]
    return darkness((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy).astype(np.float32)
