"""read_phone.py - スマホ写真(計器を切り抜いた画像)から圧力を読む(net_phone_v1)
  python read_phone.py <画像フォルダ or 画像ファイル> [--model models\\net_phone_v1.pt]
手順: 黒縁の内側の楕円を探す(kuro.py = 黒縁楕円.py と同じ処理)→ 大きさ・位置をそろえる(canvas.py)
      → 赤チャンネルの 96x96 → CNN → 値と自信。自信0.8未満は「未取得」とする
注意: 写真の中央付近に計器が大きく写っていること(縁の探索は画像の中心付近の白から始める)
"""
import argparse, os, sys
import numpy as np, torch, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import kuro, canvas
from train_persp import Net, norm, decode
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--model", default=os.path.join(HERE, "models", "net_phone_v1.pt")); ap.add_argument("--cf", type=float, default=0.8)
a = ap.parse_args()
net = Net(); net.load_state_dict(torch.load(a.model, map_location="cpu")); net.eval()
files = [a.src] if os.path.isfile(a.src) else [os.path.join(a.src, f) for f in sorted(os.listdir(a.src)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
for f in files:
    rgb = np.asarray(Image.open(f).convert("RGB")); g = np.asarray(Image.open(f).convert("L"))
    seed, rows = kuro.boundary(g)
    if len(rows) < 25: print(os.path.basename(f), "縁が見つからない"); continue
    (cx, cy), (d1, d2), ang = kuro.fit_ellipse(rows)[0]
    cv_ = canvas.to_canvas(np.ascontiguousarray(rgb[:, :, 0]), cx, cy, max(d1, d2))
    x = cv2.resize(cv_, (96, 96), interpolation=cv2.INTER_AREA)[None]
    with torch.no_grad(): v, c = decode(net(torch.tensor(norm(x)[:, None], dtype=torch.float32)))
    print(f"{os.path.basename(f)}  値 {v[0]:.3f} MPa  自信 {c[0]:.2f}" + ("" if c[0] >= a.cf else "  → 未取得"))
