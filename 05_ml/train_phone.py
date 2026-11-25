"""train_phone.py - XIAO(正面)と iPhone(5姿勢)の画像を、縁の楕円で大きさ・位置をそろえて学習する
入力: canvas.py の 192x192(縁の長径×1.35 が1辺)→ 水増しの変形 → 96x96、log(I+10) を画像ごとに正規化
水増し: 文字盤の面を 0〜30° 傾ける射影変換、回転±10°、拡大縮小0.88〜1.12、平行移動±8(192基準)、明るさ・映り込み・ぼけ・ノイズ
テスト: iPhone の 1姿勢(top_tilt)まるごと、他の姿勢と XIAO の各15%、別の計器のスマホ写真8枚(真値0.87)
  python train_phone.py --epochs 200 --out phone.pt
"""
import argparse, os, time, numpy as np, torch, torch.nn.functional as F, cv2
from train_persp import Net, norm, decode, soft_target, photo
C, OUT = 192, 96
def aug_h():
    th = np.radians(np.random.uniform(0, 30)) if np.random.rand() < 0.8 else 0.0
    phi = np.random.uniform(0, 2 * np.pi); rot = np.radians(np.random.uniform(-10, 10)); sc = np.random.uniform(0.88, 1.12); tx, ty = np.random.uniform(-8, 8, 2)
    f, Z = 400.0, 400.0
    ax = np.array([np.cos(phi), np.sin(phi), 0.0]); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    src, dst = [], []
    for X, Y in ((-60, -60), (60, -60), (60, 60), (-60, 60)):
        src.append((C / 2 + X, C / 2 + Y)); P = R @ np.array([X, Y, 0.0]) + np.array([0, 0, Z]); u, v = f * P[0] / P[2], f * P[1] / P[2]
        c, s = np.cos(rot), np.sin(rot); u, v = sc * (c * u - s * v), sc * (s * u + c * v)
        dst.append((C / 2 + u + tx, C / 2 + v + ty))
    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    S = np.array([[OUT / C, 0, 0], [0, OUT / C, 0], [0, 0, 1]]); return S @ H
def sample(x): return photo(cv2.warpPerspective(x, aug_h(), (OUT, OUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
def plain(X): return np.stack([cv2.resize(x, (OUT, OUT), interpolation=cv2.INTER_AREA) for x in X])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--out", default="phone.pt"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-xiao", action="store_true"); ap.add_argument("--no-iphone", action="store_true")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(0)
    Xtr, ytr, tests = [], [], {}
    for k in ["front", "dark", "desk", "window"]:
        if not os.path.exists(f"cv_{k}.npz"): print("なし(飛ばす):", f"cv_{k}.npz"); continue
        d = np.load(f"cv_{k}.npz"); X, y = d["X"], d["y"]; idx = rng.permutation(len(X)); nte = int(len(X) * 0.15)
        if k == "front": idx = np.array([i for i in idx if i not in (278, 279)])
        tests["x_" + k] = (X[idx[:nte]], y[idx[:nte]])
        if not a.no_xiao: Xtr.append(X[idx[nte:]]); ytr.append(y[idx[nte:]])
    d = np.load("cv_iphone.npz") if os.path.exists("cv_iphone.npz") else None
    X, y, P = (d["X"], d["y"], d["pose"]) if d is not None else (None, None, np.array([]))
    for p in sorted(set(P)):
        m = np.where(P == p)[0]
        if "top_tilt" in p: tests["ip_" + p.split("_", 2)[2] + "(未学習の姿勢)"] = (X[m], y[m]); continue
        m = m[rng.permutation(len(m))]; nte = int(len(m) * 0.15)
        tests["ip_" + p.split("_", 2)[2] + "_" + p[:6]] = (X[m[:nte]], y[m[:nte]])
        if not a.no_iphone: Xtr.append(X[m[nte:]]); ytr.append(y[m[nte:]])
    if os.path.exists("cv_phone8.npz"): d = np.load("cv_phone8.npz"); tests["別計器スマホ8"] = (d["X"], d["y"])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * ((len(Xtr) + 63) // 64))
    def evaluate():
        net.eval(); res = {}
        for k, (X, y) in tests.items():
            with torch.no_grad(): v, c = decode(net(torch.tensor(norm(plain(X))[:, None], dtype=torch.float32)))
            e = abs(v - y); a8 = c >= 0.8; res[k] = (v, c, y)
            print(f"  {k:28s} n={len(y):4d} 平均 {e.mean()*100:5.2f}%FS ±2%以内 {(e<=0.02).mean()*100:3.0f}%  自信≥0.8 正/誤/未 {(a8&(e<=0.02)).sum()}/{(a8&(e>0.02)).sum()}/{(~a8).sum()}", flush=True)
        net.train(); return res
    t0 = time.time(); print(f"学習 {len(Xtr)}枚", flush=True)
    for ep in range(a.epochs):
        net.train(); p = np.random.permutation(len(Xtr))
        for b in range(0, len(p), 64):
            j = p[b:b + 64]; xb = torch.tensor(norm(np.stack([sample(Xtr[i]) for i in j]))[:, None], dtype=torch.float32)
            loss = -(torch.tensor(soft_target(ytr[j]), dtype=torch.float32) * F.log_softmax(net(xb), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep + 1) % 50 == 0 or ep == a.epochs - 1:
            print(f"ep{ep+1} {time.time()-t0:.0f}s", flush=True); res = evaluate()
    torch.save(net.state_dict(), a.out); np.save(a.out.replace(".pt", "_res.npy"), res, allow_pickle=True)
    if "別計器スマホ8" in res: d = res["別計器スマホ8"]; print("別計器スマホ8 の値:", np.round(d[0], 3), "自信:", np.round(d[1], 2))
