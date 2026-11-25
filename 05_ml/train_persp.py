"""train_persp.py - 正面の画像を人工的に斜めにする水増しで学習し、実際の15°・30°の画像で試す
入力: 1/2解像度(320x240)の (150,108) 中心 160x160 を 96x96 に(log(I+10) を画像ごとに正規化)
水増し: 文字盤の面を 0〜35° 傾けた射影変換(傾く向きは乱数)、回転±8°、拡大縮小0.85〜1.15、平行移動±25(1/2解像度)、明るさ・映り込み・ぼけ・ノイズ
出力: 値 -0.05〜1.04 を 0.01刻み110区分の確率。山の±3区分の重み付き平均を値、山の確率の和を自信
  python train_persp.py --epochs 40 --out persp.pt
"""
import argparse, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
NB = 110; V0 = -0.05; DV = 0.01
CX, CY, CS, OUT = 150.0, 108.0, 160, 96          # 切り抜き(1/2解像度)
GX, GY, FH = 295.7 / 2, 219.6 / 2, 230.0          # 針の軸(1/2解像度)と焦点距離(1/2解像度)
S_PXMM = 2.09 / 2                                  # 1/2解像度での px/mm(正面)
def crop_mat():
    k = OUT / CS; return np.array([[k, 0, -k * (CX - CS / 2)], [0, k, -k * (CY - CS / 2)], [0, 0, 1]], np.float64)
CM = crop_mat()
def tilt_h(theta, phi, rot, scale, tx, ty):
    """文字盤の面を軸(phi方向)まわりに theta 傾けた見え方への射影変換(1/2解像度の画像座標)"""
    Z0 = FH / S_PXMM
    ax = np.array([np.cos(phi), np.sin(phi), 0.0]); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    src, dst = [], []
    for X, Y in ((-50, -50), (50, -50), (50, 50), (-50, 50)):
        src.append((GX + S_PXMM * X, GY + S_PXMM * Y))
        P = R @ np.array([X, Y, 0.0]) + np.array([0, 0, Z0])
        u, v = FH * P[0] / P[2], FH * P[1] / P[2]
        c, s = np.cos(rot), np.sin(rot)
        u, v = scale * (c * u - s * v), scale * (s * u + c * v)
        dst.append((GX + u + tx, GY + v + ty))
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)).astype(np.float64)
def to_input(img_half, H=None):
    M = CM if H is None else CM @ H
    return cv2.warpPerspective(img_half, M, (OUT, OUT), flags=cv2.INTER_AREA if H is None else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
def norm(x):
    l = np.log(x.astype(np.float32) + 10.0); m = l.reshape(len(l), -1).mean(1)[:, None, None]; s = l.reshape(len(l), -1).std(1)[:, None, None] + 1e-3
    return (l - m) / s
yy, xx = np.mgrid[0:OUT, 0:OUT] / (OUT - 1.0)
def photo(x):
    x = x.astype(np.float32) / 255.0
    x = np.clip(x, 1e-3, 1) ** np.random.uniform(0.6, 1.7)
    x = (x - x.mean()) * np.random.uniform(0.6, 1.5) + x.mean() + np.random.uniform(-0.15, 0.15)
    if np.random.rand() < 0.5:
        cx, cy = np.random.rand(2); s = np.random.uniform(0.15, 0.6); h = np.random.uniform(0.2, 0.7) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        x = x * (1 - h) + h * np.random.uniform(0.7, 1.0)
    x = np.clip(x, 0, 1) * 255
    if np.random.rand() < 0.3: x = cv2.GaussianBlur(x, (3, 3), np.random.uniform(0.3, 1.0))
    return np.clip(x + np.random.normal(0, np.random.uniform(0, 4), x.shape), 0, 255)
def sample_aug(img):
    th = np.radians(np.random.uniform(0, 35)) if np.random.rand() < 0.85 else 0.0
    H = tilt_h(th, np.random.uniform(0, 2 * np.pi), np.radians(np.random.uniform(-8, 8)), np.random.uniform(0.85, 1.15), *np.random.uniform(-25, 25, 2))
    return photo(to_input(img, H))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        c = lambda i, o, st: nn.Sequential(nn.Conv2d(i, o, 3, st, 1), nn.BatchNorm2d(o), nn.ReLU())
        s.f = nn.Sequential(c(1, 16, 2), c(16, 32, 2), c(32, 64, 2), c(64, 64, 1), c(64, 64, 2), c(64, 64, 1))   # 96→6
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * 6 * 6, 128), nn.ReLU(), nn.Linear(128, NB))
    def forward(s, x): return s.h(s.f(x))
def soft_target(y, sig=1.0):
    c = (y - V0) / DV; k = np.arange(NB)[None]; t = np.exp(-0.5 * ((k - c[:, None]) / sig) ** 2); return t / t.sum(1, keepdims=True)
def decode(lg):
    p = torch.softmax(lg, 1).numpy(); k = p.argmax(1); v = []; c = []
    for i in range(len(p)):
        idx = np.clip(k[i] + np.arange(-3, 4), 0, NB - 1); w = p[i, idx]
        v.append(V0 + DV * (w * idx).sum() / w.sum()); c.append(w.sum())
    return np.array(v), np.array(c)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--out", default="persp.pt"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(0)
    Xtr, ytr, tests = [], [], {}
    for k in ["front", "dark", "desk", "window"]:
        if not os.path.exists(f"h_{k}.npz"): print("なし(飛ばす):", f"h_{k}.npz"); continue
        d = np.load(f"h_{k}.npz"); X, y = d["X"], d["y"]; idx = rng.permutation(len(X)); nte = int(len(X) * 0.15)
        if k == "front": idx = np.array([i for i in idx if i not in (278, 279)])
        tests[k] = (X[idx[:nte]], y[idx[:nte]]); Xtr.append(X[idx[nte:]]); ytr.append(y[idx[nte:]])
    for k in ["t_sun", "t_dark", "t_desk", "p_ceil", "obl15", "obl30"]:
        if not os.path.exists(f"h_{k}.npz"): print("なし(飛ばす):", f"h_{k}.npz"); continue
        d = np.load(f"h_{k}.npz"); tests[k] = (d["X"], d["y"])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = a.epochs * ((len(Xtr) + 63) // 64); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    def evaluate(final=False):
        net.eval(); out = {}
        for k, (X, y) in tests.items():
            inp = torch.tensor(norm(np.stack([to_input(x) for x in X]))[:, None], dtype=torch.float32)
            with torch.no_grad(): v, c = decode(net(inp))
            e = abs(v - y); a8 = c >= 0.8; out[k] = (v, c, y)
            print(f"  {k:7s} 平均 {e.mean()*100:5.2f}%FS ±2%以内 {(e<=0.02).mean()*100:3.0f}%  自信≥0.8 正/誤/未 {(a8&(e<=0.02)).sum()}/{(a8&(e>0.02)).sum()}/{(~a8).sum()}", flush=True)
        net.train(); return out
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); p = np.random.permutation(len(Xtr))
        for b in range(0, len(p), 64):
            j = p[b:b + 64]
            xb = torch.tensor(norm(np.stack([sample_aug(Xtr[i]) for i in j]))[:, None], dtype=torch.float32)
            loss = -(torch.tensor(soft_target(ytr[j]), dtype=torch.float32) * F.log_softmax(net(xb), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 50 == 0 or ep == a.epochs - 1:
            print(f"ep{ep+1} {time.time()-t0:.0f}s", flush=True); res = evaluate()
    torch.save(net.state_dict(), a.out); np.save(a.out.replace(".pt", "_res.npy"), res, allow_pickle=True)
