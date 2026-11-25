# -*- coding: utf-8 -*-
"""hom_check.py - カメラ位置を合わせずに読む方式の確認(PC だけで動かす。XIAO のファームは変更しない)

考え方:
  学習時の基準画像と今の画像で、同じ目印(針の軸と目盛り線の先)をクリックで指定し、
  今の画像 → 学習時の見え方 への射影変換(斜め・距離・ずれを含む)を求める。
  変換した画像を、XIAO と同じ位置合わせ → 極座標の帯画像 → 極座標CNN(net_polar_s0.pt)で読む。
  conf と値が学習時の位置と同じように出れば、XIAO のファームに同じ変換を入れる価値がある。

使い方(05_ml フォルダで):
  1) ライブで読む
       python hom_check.py --xiao http://192.168.11.5
  2) 疑似針を動かして正解と比べる(測定)
       python hom_check.py --xiao http://192.168.11.5 --needle COM8 --targets 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --per 3
  3) 保存済みの画像1枚で
       python hom_check.py --file data\\check.jpg
  4) 求めた変換を XIAO に送る(XIAO のファームに /hom があること。以後 XIAO の /nn が補正して読む)
       python hom_check.py --xiao http://192.168.11.5 --upload
     解除: python hom_check.py --xiao http://192.168.11.5 --clear

クリックの手順(初回は基準画像、次に今の画像。同じ順番・同じ場所をクリックする):
  0      … 0 の数字の横の目盛り線の、外側の端
  0.2 0.4 0.6 0.8 1.0 … 同じく各数字の目盛り線の外側の端
  左クリック=指定  右クリック/Backspace=1つ戻す  n=この点を飛ばす(影で見えないとき)  Enter=決定  q=中止
  4点以上あれば変換を求められる。点が多いほど安定する。
  針の軸は使わない(軸のキャップは文字盤から浮いているので、斜めから見ると文字盤の面からずれて見え、変換が歪む)。
  基準画像の点は models\\hom_ref_pts.json、今の画像の点は results\\hom_live_pts.json に保存し、次回から使う。
  クリックし直すときは --redo-ref(基準)/ --redo(今の画像)、ライブ表示中は r キー。

画面:
  左 = 学習時の見え方に変換した今の画像(緑の輪=CNN が見る範囲、線=読んだ針の角度)
  右 = 輪郭の重ね合わせ(赤=基準、シアン=変換後の今の画像。重なれば変換が合っている)
  value / conf … 読んだ値と自信(XIAO は conf 0.9 以上を採用)
  ncc dx dy    … 変換後の画像での XIAO と同じ位置合わせ。変換が合っていれば dx dy は 0 付近
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import polar        # noqa: E402
import reg2         # noqa: E402
from train_polar import PolNet, decode, A0, SPAN  # noqa: E402

REF = os.path.join(HERE, "models", "reg_ref.npy")
REF_PTS = os.path.join(HERE, "models", "hom_ref_pts.json")
LIVE_PTS = os.path.join(HERE, "results", "hom_live_pts.json")
NAMES = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]   # 目盛り線の外側の端(針の軸は使わない)
HELP = {}
for k in NAMES:
    HELP[k] = f"tick '{k}': OUTER end of the tick line"


def grab(xiao):
    import requests
    r = requests.get(f"{xiao}/capture.jpg", timeout=15)
    if r.status_code != 200 or r.content[:2] != b"\xff\xd8":
        raise RuntimeError(f"HTTP {r.status_code}")
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)


def text(img, lines, y0=22, color=(255, 255, 255)):
    for i, t in enumerate(lines):
        y = y0 + 22 * i
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def click_points(gray, title, scale):
    """NAMES の順にクリックで指定。戻り値 {名前: (x, y) または None}。q で None"""
    pts = []
    state = {"mouse": None}
    win = title

    def on(ev, x, y, flags, _):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(NAMES):
            pts.append((x / scale, y / scale))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    while True:
        img = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for i, p in enumerate(pts):
            if p is None:
                continue
            q = (int(p[0] * scale), int(p[1] * scale))
            cv2.drawMarker(img, q, (0, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
            cv2.putText(img, NAMES[i], (q[0] + 6, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # 拡大鏡(カーソルの周り 3 倍)
        m = state["mouse"]
        if m:
            x, y = int(m[0] / scale), int(m[1] / scale)
            r = 20
            x0, y0 = max(0, x - r), max(0, y - r)
            crop = base[y0:y0 + 2 * r, x0:x0 + 2 * r]
            if crop.size:
                z = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                cv2.line(z, (z.shape[1] // 2, 0), (z.shape[1] // 2, z.shape[0]), (0, 255, 0), 1)
                cv2.line(z, (0, z.shape[0] // 2), (z.shape[1], z.shape[0] // 2), (0, 255, 0), 1)
                img[img.shape[0] - z.shape[0]:, img.shape[1] - z.shape[1]:] = z
        n = len(pts)
        msg = [f"[{title}]  click {n + 1}/{len(NAMES)}: {HELP[NAMES[n]]}" if n < len(NAMES) else f"[{title}]  all points set -> Enter",
               "L-click=set  R-click/Backspace=undo  n=skip point  Enter=done  q=quit"]
        text(img, msg)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and sum(p is not None for p in pts) >= 4:
            break
        if k == ord("n") and len(pts) < len(NAMES):
            pts.append(None)
        if k == 8 and pts:
            pts.pop()
        if k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    out = {name: None for name in NAMES}
    for name, p in zip(NAMES, pts):
        out[name] = p
    return out


def homography(live_pts, ref_pts):
    keys = [k for k in NAMES if live_pts.get(k) and ref_pts.get(k)]
    if len(keys) < 4:
        raise RuntimeError("両方で指定された点が4つ未満です")
    s = np.float32([live_pts[k] for k in keys])
    d = np.float32([ref_pts[k] for k in keys])
    H, _ = cv2.findHomography(s, d, 0)
    pr = cv2.perspectiveTransform(s.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pr - d, axis=1)
    return H, keys, err


class Reader:
    def __init__(s, model):
        s.net = PolNet()
        s.net.load_state_dict(torch.load(model, map_location="cpu"))
        s.net.eval()
        ref = np.load(REF)
        s.R = reg2.Ref(ref)
        s.ref_e = np.clip(reg2.edge(ref) / reg2.ECLIP * 255, 0, 255).astype(np.uint8)

    def read(s, gray, H):
        w = cv2.warpPerspective(gray, H, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dx, dy, ncc = reg2.shift(w.astype(np.float32), s.R)
        d = polar.prep_at(w.astype(np.float32), polar.CX + dx, polar.CY + dy)
        with torch.no_grad():
            v, c = decode(s.net(torch.tensor(d[None])))
        return float(v[0]), float(c[0]), dx, dy, ncc, w

    def view(s, w, v, c, dx, dy, ncc, scale, extra=()):
        a = cv2.cvtColor(w, cv2.COLOR_GRAY2BGR)
        cx, cy = polar.CX + dx, polar.CY + dy
        for r in (polar.R0, polar.R0 + polar.DR * (polar.NR - 1)):
            cv2.circle(a, (int(cx), int(cy)), int(r), (0, 255, 0), 1, cv2.LINE_AA)
        ang = np.radians(A0 + SPAN * v)
        cv2.line(a, (int(cx), int(cy)), (int(cx + 110 * np.cos(ang)), int(cy + 110 * np.sin(ang))),
                 (0, 0, 255) if c < 0.9 else (0, 255, 255), 2, cv2.LINE_AA)
        le = np.clip(reg2.edge(w.astype(np.float32)) / reg2.ECLIP * 255, 0, 255).astype(np.uint8)
        b = np.zeros_like(a)
        b[:, :, 2] = s.ref_e
        b[:, :, 1] = le
        b[:, :, 0] = le
        img = np.hstack([a, b])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        text(img, [f"value {v:.3f} MPa   conf {c:.2f} {'OK' if c >= 0.9 else '(< 0.9: XIAO rejects)'}",
                   f"ncc {ncc:.3f}  dx {dx:+d} dy {dy:+d}"] + list(extra) +
             ["[r] re-click points  [s] save  [q] quit"], color=(255, 255, 255))
        return img


def load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="射影変換で補正して読む確認(PC のみ)")
    ap.add_argument("--xiao", default="http://192.168.11.5")
    ap.add_argument("--file", help="ライブの代わりに保存済みの画像を使う")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "net_polar_s0.pt"))
    ap.add_argument("--scale", type=float, default=1.5, help="クリック画面の表示倍率")
    ap.add_argument("--vscale", type=float, default=0.9, help="読み取り画面の表示倍率(左右2枚)")
    ap.add_argument("--redo-ref", action="store_true", help="基準画像の点をクリックし直す")
    ap.add_argument("--redo", action="store_true", help="今の画像の点をクリックし直す")
    ap.add_argument("--needle", help="疑似針の COM ポート(測定するとき)")
    ap.add_argument("--targets", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--per", type=int, default=3, help="測定: 1つの目標値で読む回数")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--upload", action="store_true", help="results\\hom_live_pts.json の変換を XIAO に送って終わる")
    ap.add_argument("--clear", action="store_true", help="XIAO の補正を解除して終わる")
    a = ap.parse_args()

    def get():
        if a.file:
            return cv2.imdecode(np.fromfile(a.file, np.uint8), cv2.IMREAD_GRAYSCALE)
        return grab(a.xiao)

    if a.clear:
        import requests
        print(requests.get(f"{a.xiao}/hom", params={"clear": 1}, timeout=10).text)
        return
    if a.upload:
        import requests
        saved = load_json(LIVE_PTS)
        if not saved:
            sys.exit(f"{LIVE_PTS} がありません。先に python hom_check.py でクリックしてください")
        M = np.linalg.inv(np.array(saved["H"], np.float64))   # XIAO には 基準の座標 → 今の画像の座標 を送る
        M /= M[2, 2]
        m = ",".join(f"{v:.9g}" for v in M.ravel())
        r = requests.get(f"{a.xiao}/hom", params={"m": m}, timeout=10)
        print(r.status_code, r.text)
        if r.status_code == 404:
            print("XIAO に /hom がありません。補正入りのファームを書き込んでください")
        return

    ref_img = np.clip(np.load(REF), 0, 255).astype(np.uint8)
    ref_pts = None if a.redo_ref else load_json(REF_PTS)
    if not ref_pts:
        print("基準画像(学習時の見え方)で点をクリックしてください")
        ref_pts = click_points(ref_img, "reference", a.scale)
        if not ref_pts:
            return
        os.makedirs(os.path.dirname(REF_PTS), exist_ok=True)
        json.dump(ref_pts, open(REF_PTS, "w", encoding="utf-8"), indent=1)
        print("保存:", REF_PTS)

    def click_live():
        print("今の画像で、基準と同じ順番・同じ場所をクリックしてください")
        g = get()
        p = click_points(g, "live", a.scale)
        if not p:
            sys.exit("中止")
        H, keys, err = homography(p, ref_pts)
        os.makedirs(os.path.dirname(LIVE_PTS), exist_ok=True)
        json.dump({"pts": p, "H": H.tolist(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(LIVE_PTS, "w", encoding="utf-8"), indent=1)
        report(keys, err)
        return H

    def report(keys, err):
        print("変換に使った点:", ", ".join(f"{k} {e:.1f}px" for k, e in zip(keys, err)),
              f"/ 平均のずれ {err.mean():.1f} 画素(3 以下が目安。大きい点はクリックし直す)")

    saved = None if a.redo else load_json(LIVE_PTS)
    if saved and not a.file:
        H = np.array(saved["H"], np.float64)
        print("前回の点を使用:", LIVE_PTS, saved.get("time", ""), "(クリックし直すには --redo か r キー)")
    else:
        H = click_live()

    rd = Reader(a.model)
    win = "hom_check"

    if a.needle:
        measure(a, rd, H, get, win)
        return

    while True:
        try:
            g = get()
            v, c, dx, dy, ncc, w = rd.read(g, H)
            img = rd.view(w, v, c, dx, dy, ncc, a.vscale)
            print(f"value {v:.3f}  conf {c:.2f}  ncc {ncc:.3f}  dx {dx:+d} dy {dy:+d}")
        except Exception as e:
            print("取得できない:", e)
            time.sleep(1)
            continue
        cv2.imshow(win, img)
        k = cv2.waitKey(1000 if a.file else 30) & 0xFF
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord("r"):
            cv2.destroyWindow(win)
            H = click_live()
        if k == ord("s"):
            p = os.path.join(HERE, "results", time.strftime("hom_%Y%m%d_%H%M%S.png"))
            cv2.imencode(".png", img)[1].tofile(p)
            print("保存:", p)
    cv2.destroyAllWindows()


def measure(a, rd, H, get, win):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_stepper_needle"))
    from needle import Needle
    targets = [float(t) for t in a.targets.split(",")]
    out = os.path.join(HERE, "results", time.strftime("hom_measure_%Y%m%d_%H%M%S.csv"))
    nd = Needle(a.needle)
    print("原点復帰中…")
    nd.home()
    rows = []
    try:
        with open(out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["truth", "value", "err_fs", "conf", "ncc", "dx", "dy"])
            for t in targets:
                res = nd.move_mpa(t)
                truth = float(res.get("mpa", t)) if isinstance(res, dict) else t
                time.sleep(a.settle)
                for _ in range(a.per):
                    g = get()
                    v, c, dx, dy, ncc, w = rd.read(g, H)
                    e = v - truth
                    rows.append((truth, v, e, c, ncc))
                    wr.writerow([f"{truth:.4f}", f"{v:.4f}", f"{e * 100:.2f}", f"{c:.3f}", f"{ncc:.3f}", dx, dy])
                    f.flush()
                    print(f"truth {truth:.3f}  value {v:.3f}  err {e * 100:+.2f}%FS  conf {c:.2f}  ncc {ncc:.3f}  dx {dx:+d} dy {dy:+d}")
                    cv2.imshow(win, rd.view(w, v, c, dx, dy, ncc, a.vscale, [f"truth {truth:.3f}  err {e * 100:+.2f}%FS"]))
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("中止")
    finally:
        nd.off()
        nd.close()
        cv2.destroyAllWindows()
    if rows:
        r = np.array(rows)
        e = np.abs(r[:, 2]) * 100
        ok = r[:, 3] >= 0.9
        print(f"\n{len(r)} 回  平均誤差 {e.mean():.2f}%FS  最大 {e.max():.2f}%FS  ±2%FS 以内 {(e <= 2).mean() * 100:.0f}%")
        print(f"conf 0.9 以上 {ok.mean() * 100:.0f}%(この中の平均誤差 {e[ok].mean() if ok.any() else float('nan'):.2f}%FS)  conf の中央値 {np.median(r[:, 3]):.2f}")
        print("保存:", out)


if __name__ == "__main__":
    main()
