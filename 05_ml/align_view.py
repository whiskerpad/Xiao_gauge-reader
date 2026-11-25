# -*- coding: utf-8 -*-
"""align_view.py - XIAO のライブ画像に学習時の基準画像を重ねて、カメラ位置を合わせる確認用

XIAO の /capture.jpg を繰り返し取り、models/reg_ref.npy(学習時の基準画像)と重ねて表示します。
ファームの変更は不要です。

使い方(05_ml フォルダで):
  python align_view.py --xiao http://192.168.11.5
  python align_view.py --file data\\check.jpg      (保存済みの画像1枚で確かめる)

画面の見方:
  緑の十字と円 … 学習時の針の軸(CX, CY)と黒縁の内側(直径 202 画素)。ここに文字盤を合わせる
  黄の楕円     … 基準の輪郭を拡大縮小・移動して照合し、推定した今の文字盤の位置と大きさ。
                 緑の円に重なれば位置・距離・向きが合っている
  赤とシアン   … 輪郭表示のとき、赤が基準、シアンが今の画像。重なると白っぽくなる
  上の文字
    XIAO reg … XIAO と同じ位置合わせ(ずれだけ ±48 画素)の ncc と dx dy。読めるかどうかはこれで決まる
    fit      … 広い範囲で照合した結果。center が今の文字盤の中心、size x/y が基準に対する横・縦の倍率
               (1 より大きい=近すぎ、小さい=遠すぎ、x と y が違う=左右か上下に斜め)
    ->       … 次に直す向き。"position OK" になれば合っている
  目安: 基準と同じ天井照明で XIAO reg の ncc が 0.6 以上、dx dy が 0 付近
        (学習時の画像では天井照明 0.86、ほかの照明 0.4〜0.6。照明が違うと ncc は下がる)
  ずれや倍率が大きいと fit の推定も不安定になります。まず大まかに輪郭を重ねてから数値で詰めてください。

キー:
  m … 表示の切り替え(輪郭の重ね合わせ / 半透明の重ね合わせ / ライブのみ / 基準のみ)
  + - … 半透明の濃さ
  s … 今の画像を data\\align_YYYYmmdd_HHMMSS.jpg に保存
  q または Esc … 終了
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import polar  # noqa: E402
import reg2   # noqa: E402

REF = os.path.join(HERE, "models", "reg_ref.npy")
RIM_D = 202.0          # 学習時の黒縁の内側の直径[画素](make_npz.py canvas の --diam と同じ)
MODES = ["edge", "blend", "live", "ref"]


def grab(xiao):
    import requests
    r = requests.get(f"{xiao}/capture.jpg", timeout=15)
    if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
        raise RuntimeError(f"HTTP {r.status_code}")
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE), r.content


class Fit:
    """基準の輪郭と、拡大率(横 sx、縦 sy)・ずれを変えながら照合して、今の写り方を推定する。
    XIAO の位置合わせ(ずれだけ、±48 画素)より広く、画面全体と 0.76〜1.30 倍を探す"""
    def __init__(s, ref):
        s.x0, s.y0 = reg2.RX0 // 2, reg2.RY0 // 2
        s.T = reg2.edge(reg2.half(ref))[s.y0:s.y0 + reg2.RH // 2, s.x0:s.x0 + reg2.RW // 2].astype(np.float32)
        s.c = (polar.CX / 2, polar.CY / 2)

    def one(s, h, sx, sy):
        cx, cy = s.c
        M = np.float32([[1 / sx, 0, cx - cx / sx], [0, 1 / sy, cy - cy / sy]])
        w = cv2.warpAffine(h, M, (h.shape[1], h.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        r = cv2.matchTemplate(reg2.edge(w).astype(np.float32), s.T, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        return v, sx, sy, loc[0] - s.x0, loc[1] - s.y0

    def __call__(s, g):
        h = reg2.half(g.astype(np.float32))
        best = max(s.one(h, sx, sy) for sx in np.arange(0.76, 1.31, 0.03) for sy in np.arange(0.76, 1.31, 0.03))
        _, a, b, _, _ = best
        best = max([best] + [s.one(h, sx, sy) for sx in np.arange(a - 0.03, a + 0.031, 0.01)
                             for sy in np.arange(b - 0.03, b + 0.031, 0.01)])
        v, sx, sy, ox, oy = best
        return v, sx, sy, polar.CX + 2 * ox * sx, polar.CY + 2 * oy * sy


def edge_u8(g):
    e = reg2.edge(g.astype(np.float32))
    return np.clip(e / reg2.ECLIP * 255, 0, 255).astype(np.uint8)


def compose(live, ref, ref_e, mode, alpha):
    if mode == "edge":
        le = edge_u8(live)
        img = np.zeros(live.shape + (3,), np.uint8)
        img[:, :, 2] = ref_e                  # 基準の輪郭 = 赤
        img[:, :, 1] = le                     # ライブの輪郭 = 緑と青(シアン)
        img[:, :, 0] = le
        return img                            # 重なった所は白っぽくなる
    if mode == "blend":
        b = cv2.addWeighted(live, 1 - alpha, ref, alpha, 0)
        return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if mode == "live":
        return cv2.cvtColor(live, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)


def draw(img, live, R, F, mode, alpha, scale):
    cx, cy = polar.CX, polar.CY
    dx, dy, s = reg2.shift(live.astype(np.float32), R)
    v, sx, sy, fx, fy = F(live)
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    P = lambda x, y: (int(round(x * scale)), int(round(y * scale)))  # noqa: E731
    g = (0, 255, 0)
    cv2.circle(out, P(cx, cy), int(RIM_D / 2 * scale), g, 1, cv2.LINE_AA)
    cv2.line(out, P(cx - 15, cy), P(cx + 15, cy), g, 1)
    cv2.line(out, P(cx, cy - 15), P(cx, cy + 15), g, 1)
    x0, y0 = reg2.RX0, reg2.RY0
    cv2.rectangle(out, P(x0, y0), P(x0 + reg2.RW, y0 + reg2.RH), (90, 90, 90), 1)
    # 推定した今の文字盤の位置と大きさ(黄)
    cv2.ellipse(out, (P(fx, fy), (RIM_D * sx * scale, RIM_D * sy * scale), 0), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(out, P(fx, fy), 3, (0, 255, 255), -1)
    hint = []
    if abs(fx - cx) > 5 or abs(fy - cy) > 5:
        hint.append(f"dial needs to move {cx - fx:+.0f},{cy - fy:+.0f} px")
    m = (sx + sy) / 2
    if m > 1.03:
        hint.append("too close")
    elif m < 0.97:
        hint.append("too far")
    if sx / sy < 0.96 or sx / sy > 1.04:
        hint.append("turn left/right (oblique)")
    lines = [f"mode {mode}" + (f"  alpha {alpha:.1f}" if mode == "blend" else "") + "   [m] mode  [+/-] alpha  [s] save  [q] quit",
             f"XIAO reg: ncc {s:.3f}  dx {dx:+d} dy {dy:+d}  (search +-48 px)",
             f"fit: ncc {v:.2f}  center ({fx:.0f},{fy:.0f})  size x{sx:.2f} y{sy:.2f}  (target 1.00)",
             ("-> " + ", ".join(hint)) if hint else "-> position OK"]
    for i, t in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="ライブ画像と基準画像を重ねてカメラ位置を合わせる")
    ap.add_argument("--xiao", default="http://192.168.11.5")
    ap.add_argument("--file", help="ライブの代わりに保存済みの画像を使う")
    ap.add_argument("--scale", type=float, default=1.5, help="表示倍率")
    ap.add_argument("--out", help="1回描いて画像に保存して終わる(確認用)")
    a = ap.parse_args()
    if not os.path.exists(REF):
        sys.exit(f"{REF} がありません")
    ref_f = np.load(REF)
    ref = np.clip(ref_f, 0, 255).astype(np.uint8)
    ref_e = edge_u8(ref_f)
    R = reg2.Ref(ref_f)
    F = Fit(ref_f)
    mode, alpha = 0, 0.5
    win = "align_view"
    last_jpg = None
    while True:
        try:
            if a.file:
                live = cv2.imdecode(np.fromfile(a.file, np.uint8), cv2.IMREAD_GRAYSCALE)
                last_jpg = None
            else:
                live, last_jpg = grab(a.xiao)
            if live.shape != ref.shape:
                live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
            img = draw(compose(live, ref, ref_e, MODES[mode], alpha), live, R, F, MODES[mode], alpha, a.scale)
        except Exception as e:
            print("取得できない:", e)
            img = np.zeros((int(480 * a.scale), int(640 * a.scale), 3), np.uint8)
            cv2.putText(img, f"capture error: {e}"[:90], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            time.sleep(1)
        if a.out:
            cv2.imencode(".png", img)[1].tofile(a.out)
            print("保存:", a.out)
            return
        cv2.imshow(win, img)
        k = cv2.waitKey(300 if a.file else 30) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("m"):
            mode = (mode + 1) % len(MODES)
        elif k in (ord("+"), ord(";")):
            alpha = min(0.9, alpha + 0.1)
        elif k == ord("-"):
            alpha = max(0.1, alpha - 0.1)
        elif k == ord("s") and last_jpg:
            p = os.path.join(HERE, "data", time.strftime("align_%Y%m%d_%H%M%S.jpg"))
            open(p, "wb").write(last_jpg)
            print("保存:", p)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
