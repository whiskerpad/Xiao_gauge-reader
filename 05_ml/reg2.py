"""reg2.py - XIAO で動かす位置合わせ(位相相関の代わりに、輪郭画像の相関を範囲を絞って探す)
輪郭: E = |∇ log(I+10)|(中心差分)、上限 ECLIP で頭打ち。基準 E は朝の天井照明セットの平均画像から作る
1) 1/2縮小画像で ±24画素(元±48)を 2画素おきに探す  2) その周り±2を1画素おき  3) 元の解像度で ±1
相関は「平均を引いた積和 / ノルム」(正規化相互相関)。対象は文字盤のまわりの四角(ROI)
"""
import numpy as np
ECLIP = 0.5
RX0, RY0, RW, RH = 150, 70, 300, 300          # 元の解像度でのROI(基準画像上)
def edge(a):
    l = np.log(a.astype(np.float32) + 10.0); e = np.zeros_like(l)
    gx = l[1:-1, 2:] - l[1:-1, :-2]; gy = l[2:, 1:-1] - l[:-2, 1:-1]
    e[1:-1, 1:-1] = np.minimum(np.sqrt(gx * gx + gy * gy), ECLIP); return e
def half(a): return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))
def quant(e): return np.round(e / ECLIP * 255.0).astype(np.uint8)
class Ref:
    """基準の輪郭(uint8 に量子化した値を使う。XIAO と同じ)"""
    def __init__(s, refimg=None, q1=None, q2=None):
        if q1 is None:
            q1 = quant(edge(refimg)[RY0:RY0 + RH, RX0:RX0 + RW]); q2 = quant(edge(half(refimg))[RY0 // 2:(RY0 + RH) // 2, RX0 // 2:(RX0 + RW) // 2])
        s.q1, s.q2 = q1, q2
        s.e1 = q1.astype(np.float32) * (ECLIP / 255.0); s.e2 = q2.astype(np.float32) * (ECLIP / 255.0)
        s.z1 = s.e1 - s.e1.mean(); s.z2 = s.e2 - s.e2.mean()
def ncc(z, win):
    w = win - win.mean(); return float((z * w).sum() / (np.sqrt((z * z).sum() * (w * w).sum()) + 1e-9))
def shift(img, R, span2=24):
    e2 = edge(half(img.astype(np.float32))); H2, W2 = e2.shape
    h, w = R.z2.shape; x0, y0 = RX0 // 2, RY0 // 2
    def sc2(dx, dy):
        xs, ys = x0 + dx, y0 + dy
        if xs < 0 or ys < 0 or xs + w > W2 or ys + h > H2: return -2
        return ncc(R.z2, e2[ys:ys + h, xs:xs + w])
    best = max(((sc2(dx, dy), dx, dy) for dy in range(-span2, span2 + 1, 2) for dx in range(-span2, span2 + 1, 2)))
    best = max(((sc2(dx, dy), dx, dy) for dy in range(best[2] - 2, best[2] + 3) for dx in range(best[1] - 2, best[1] + 3)))
    e1 = edge(img.astype(np.float32)); H, W = e1.shape; h1, w1 = R.z1.shape
    def sc1(dx, dy):
        xs, ys = RX0 + dx, RY0 + dy
        if xs < 0 or ys < 0 or xs + w1 > W or ys + h1 > H: return -2
        return ncc(R.z1, e1[ys:ys + h1, xs:xs + w1])
    b1 = max(((sc1(dx, dy), dx, dy) for dy in range(2 * best[2] - 1, 2 * best[2] + 2) for dx in range(2 * best[1] - 1, 2 * best[1] + 2)))
    return b1[1], b1[2], b1[0]
