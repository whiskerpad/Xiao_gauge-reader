"""train_polar.py - 極座標の帯画像(角度180×半径32、照明の影響を消した暗さ)で針の角度を当てるCNN
出力は角度180区分の確率。最大の山の周り±3区分の重み付き平均を角度とし、山の確率の和を「自信」とする。
  python train_polar.py --seed 0 --out pol_s0.pt
"""
import argparse, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
A0, SPAN = 132.49, 275.32      # 画像座標での 0目盛りの角度[deg] と 振れ角(朝の天井照明セットで推定)
NA = 180
TH = np.radians(np.arange(NA) * 2.0)
class Blk(nn.Module):
    def __init__(s, ci, co, kr=3, pr=1):
        super().__init__(); s.c = nn.Conv2d(ci, co, (kr, 3)); s.pr = pr
    def forward(s, x):
        x = F.pad(x, (1, 1, 0, 0), mode="circular"); x = F.pad(x, (0, 0, s.pr, s.pr)); return s.c(x)
class PolNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.b1 = Blk(3, 16); s.b2 = Blk(16, 16); s.b3 = Blk(16, 32); s.b4 = Blk(32, 32, kr=4, pr=0); s.b5 = Blk(32, 1, kr=1, pr=0)
    def forward(s, d):                       # d: [B,32,180]
        B = d.shape[0]
        pos = torch.tensor(np.stack([np.cos(TH), np.sin(TH)]), dtype=torch.float32)[None, :, None, :].expand(B, 2, d.shape[1], NA)
        x = torch.cat([d[:, None] * 5.0, pos], 1)
        x = F.max_pool2d(F.relu(s.b1(x)), (2, 1)); x = F.max_pool2d(F.relu(s.b2(x)), (2, 1)); x = F.max_pool2d(F.relu(s.b3(x)), (2, 1))
        x = F.relu(s.b4(x)); return s.b5(x)[:, 0, 0, :]          # [B,180]
def target_angle(mpa): return (A0 + SPAN * mpa) % 360.0
def soft_target(mpa, sig=1.0):
    c = target_angle(mpa) / 2.0; k = np.arange(NA)[None]
    dd = (k - c[:, None] + NA / 2) % NA - NA / 2
    t = np.exp(-0.5 * (dd / sig) ** 2); return t / t.sum(1, keepdims=True)
def decode(logits):
    p = torch.softmax(logits, 1).numpy(); k = p.argmax(1); out = []; conf = []
    for i in range(len(p)):
        idx = (k[i] + np.arange(-3, 4)) % NA; w = p[i, idx]
        ang = np.degrees(np.arctan2((w * np.sin(TH[idx])).sum(), (w * np.cos(TH[idx])).sum())) % 360
        rel = (ang - A0) % 360.0
        if rel > 320: rel -= 360
        out.append(rel / SPAN); conf.append(w.sum())
    return np.array(out), np.array(conf)
def aug(d):
    d = d * np.random.uniform(0.6, 1.5, (len(d), 1, 1)) + np.random.normal(0, np.random.uniform(0, 0.02), d.shape)
    sh = np.random.randint(-1, 2, len(d))
    return np.stack([np.roll(x, s, 0) for x, s in zip(d, sh)]).astype(np.float32)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="pol.pt"); ap.add_argument("--train", nargs="+", default=["front", "dark", "desk", "window"])
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(0)
    Xtr, ytr, tests = [], [], {}
    for k in ["front", "dark", "desk", "window"]:
        if not os.path.exists(f"reg_{k}.npz"): print("なし(飛ばす):", f"reg_{k}.npz"); continue
        d = np.load(f"reg_{k}.npz"); X, y = d["P"], d["y"]; idx = rng.permutation(len(X)); nte = int(len(X) * 0.15)
        if k == "front": idx = np.array([i for i in idx if i not in (278, 279)])
        tests[k] = (X[idx[:nte]], y[idx[:nte]])
        if k in a.train: Xtr.append(X[idx[nte:]]); ytr.append(y[idx[nte:]])
        else: tests[k + "(未学習)"] = (X, y)
    for k in ["ev2", "ev3", "ev4", "ev5", "t_sun", "t_ceiling", "t_dark", "t_desk"]:
        if not os.path.exists(f"reg_{k}.npz"): print("なし(飛ばす):", f"reg_{k}.npz"); continue
        d = np.load(f"reg_{k}.npz"); tests[k] = (d["P"], d["y"])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    net = PolNet(); opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * ((len(Xtr) + 63) // 64))
    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); p = np.random.permutation(len(Xtr))
        for b in range(0, len(p), 64):
            j = p[b:b + 64]; lg = net(torch.tensor(aug(Xtr[j]))); tg = torch.tensor(soft_target(ytr[j]), dtype=torch.float32)
            loss = -(tg * F.log_softmax(lg, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); print(f"学習 {len(Xtr)}枚 {a.epochs}エポック {time.time()-t0:.0f}s seed={a.seed}")
    res = {}
    for k, (X, y) in tests.items():
        with torch.no_grad(): v, c = decode(net(torch.tensor(X)))
        e = v - y; res[k] = (v, c, y)
        print(f"  {k:12s} n={len(y):4d} 平均 {abs(e).mean()*100:5.2f}%FS 最大 {abs(e).max()*100:6.2f} ±2%FS以内 {(abs(e)<=0.02).mean()*100:3.0f}%  自信の中央値 {np.median(c):.2f}")
    torch.save(net.state_dict(), a.out); np.save(a.out.replace(".pt", "_res.npy"), res, allow_pickle=True)
