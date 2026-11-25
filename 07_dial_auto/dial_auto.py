# -*- coding: utf-8 -*-
"""dial_auto.py - 学習なしで丸い圧力計を読む試作(2026-11-22)
縁の楕円で仮に正面化 → 目盛りの外側の円弧に楕円を当てはめ直して正面化 → 等間隔の目盛り線の並びから目盛りの両端の角度
→ EasyOCR で数字(0.2, 0.4 …)を読み、角度と値の関係を当てはめて 最小値・最大値 → 針の角度(赤チャンネルのリング方式)→ 値
  python dial_auto.py 写真1 [写真2 …]      必要: pip install easyocr(初回にモデルを自動ダウンロード)
前提: 時計回りに増える一様な目盛り、目盛り線が円弧の外側か内側に等間隔に並ぶ、数字が2つ以上読める
"""
import sys, math, re
import numpy as np, cv2
from PIL import Image
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "05_ml"))
import kuro  # 05_ml/kuro.py(黒縁の内側の楕円)

S = 600            # 正面化した画像の1辺。縁の内側の半径 R = 0.4*S
R = 0.4 * S

def rim(gray):
    lo, hi = np.percentile(gray, (1, 99.5))
    gd = np.clip((gray.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    seed, pts = kuro.boundary(gd)
    return kuro.fit_ellipse(pts)[0]

def rectify(rgb, ell):
    """縁の楕円を、中心(S/2,S/2)・半径R の正円に写す(アフィン)"""
    (cx, cy), (d1, d2), ang = ell
    a = np.radians(ang)
    # 楕円の軸方向の単位ベクトル(OpenCV: d1 は ang 方向の直径)
    u = np.array([np.cos(a), np.sin(a)]); v = np.array([-np.sin(a), np.cos(a)])
    src = np.float32([[cx, cy], [cx + u[0] * d1 / 2, cy + u[1] * d1 / 2], [cx + v[0] * d2 / 2, cy + v[1] * d2 / 2]])
    # 回転を残さないよう、軸の向きはそのまま、長さだけ R に
    dst = np.float32([[S / 2, S / 2], [S / 2 + u[0] * R, S / 2 + u[1] * R], [S / 2 + v[0] * R, S / 2 + v[1] * R]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def polar(img, r0, r1, nr, na=720):
    th = np.radians(np.arange(na) * 360.0 / na)
    r = np.linspace(r0, r1, nr)
    mx = (S / 2 + r[:, None] * np.cos(th)[None]).astype(np.float32)
    my = (S / 2 + r[:, None] * np.sin(th)[None]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR)     # [nr, na]、角度は画像座標(右0°、下90°、時計回り)

def arc_points(fr_gray, rmin=0.55, rmax=1.0, na=720, step=0.5):
    """正面化した画像で、目盛りの外側の細い円弧の点を探す(角度ごとに細い暗い谷)"""
    nr = int((rmax - rmin) * R / step)
    P = polar(fr_gray.astype(np.float32), rmin * R, rmax * R, nr, na)
    P = cv2.GaussianBlur(P, (1, 5), 0)
    rs = np.linspace(rmin * R, rmax * R, nr)
    th, rr, wt = [], [], []
    for j in range(na):
        p = P[:, j]; med = np.median(p)
        for i in range(3, nr - 3):
            if p[i] <= p[i - 1] and p[i] <= p[i + 1] and med - p[i] > 30:
                half = (med + p[i]) / 2
                a = i; b = i
                while a > 0 and P[a, j] < half: a -= 1
                while b < nr - 1 and P[b, j] < half: b += 1
                if (b - a) * step <= 6:                 # 細い線だけ(目盛り線の長い暗がりは除く)
                    th.append(j * 360.0 / na); rr.append(rs[i]); wt.append(med - p[i])
    return np.array(th), np.array(rr), np.array(wt)

def fit_arc(th, rr, iters=800, tol=2.5, seed=0):
    """r(θ) = c0 + c1 cosθ + c2 sinθ + c3 cos2θ + c4 sin2θ を RANSAC で当てはめ"""
    rng = np.random.default_rng(seed); t = np.radians(th)
    B = np.stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], 1)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(t), 5, replace=False)
        try: c = np.linalg.lstsq(B[idx], rr[idx], rcond=None)[0]
        except np.linalg.LinAlgError: continue
        inl = np.abs(B @ c - rr) < tol
        # 同じ角度の重複を数えないよう、内点の角度の種類数で評価
        score = len(np.unique(np.round(th[inl])))
        if best is None or score > best[0]: best = (score, inl)
    inl = best[1]
    c = np.linalg.lstsq(B[inl], rr[inl], rcond=None)[0]
    inl = np.abs(B @ c - rr) < tol
    return c, inl

def rectify_M(ell, radius):
    (cx, cy), (d1, d2), ang = ell
    a = np.radians(ang); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-np.sin(a), np.cos(a)])
    src = np.float32([[cx, cy], [cx + u[0] * d1 / 2, cy + u[1] * d1 / 2], [cx + v[0] * d2 / 2, cy + v[1] * d2 / 2]])
    dst = np.float32([[S / 2, S / 2], [S / 2 + u[0] * radius, S / 2 + u[1] * radius], [S / 2 + v[0] * radius, S / 2 + v[1] * radius]])
    return cv2.getAffineTransform(src, dst)

def dial_frame(rgb):
    """縁の楕円で仮に正面化 → 目盛りの外側の円弧に楕円を当てはめ直し、その円弧が半径 R になるよう正面化"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    M = rectify_M(rim(g), R / 0.95 * 1.0)          # 仮: 縁の内側 ≒ 円弧の少し外
    for it in range(2):
        fr = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        th, rr, wt = arc_points(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), rmin=0.6, rmax=1.15)
        c, inl = fit_arc(th, rr, tol=3.0)
        t = np.radians(th[inl]); xy = np.stack([S / 2 + rr[inl] * np.cos(t), S / 2 + rr[inl] * np.sin(t)], 1)
        Mi = cv2.invertAffineTransform(M)
        orig = xy @ Mi[:, :2].T + Mi[:, 2]
        ell = cv2.fitEllipse(orig.astype(np.float32).reshape(-1, 1, 2))
        M = rectify_M(ell, R)
    fr = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    th, rr, wt = arc_points(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), rmin=0.9, rmax=1.1)
    near = np.abs(rr - R) < 4
    return fr, ell, th[near]

def scale_ends(arc_th):
    """円弧がある角度の集合から、いちばん大きい空き(目盛りの無い所)を探し、目盛りの始まり・終わりの角度を返す(時計回り)"""
    have = np.zeros(360, bool); have[np.round(arc_th).astype(int) % 360] = True
    # 1°刻みの穴は埋める(3°以下の空きは無視)
    k = np.convolve(np.r_[have, have[:6]], np.ones(7), 'same')[:360] > 0
    best = (0, 0)
    for s in range(360):
        if k[s] and not k[(s + 1) % 360]:            # 円弧が途切れる所 = 空きの始まり
            n = 1
            while not k[(s + n) % 360] and n < 360: n += 1
            if n > best[0]: best = (n, s)
    gap, end = best
    start = (end + gap) % 360
    # 3°の太らせを戻す
    return (start + 3) % 360, (end - 3) % 360, gap

def ticks(fr_gray, r0=0.88, r1=0.99, na=1440):
    """目盛り線の角度(円弧のすぐ内側の帯で、細く暗い所)"""
    P = polar(fr_gray.astype(np.float32), r0 * R, r1 * R, 12, na).mean(0)
    d = np.median(P) - P                               # 暗さ
    k = 40                                             # ±10°の移動中央値を引く
    pad = np.r_[d[-k:], d, d[:k]]
    hp = d - np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(na)])
    thr = max(6.0, 0.5 * np.percentile(hp, 90))
    pk = [i for i in range(na) if hp[i] > thr and hp[i] >= hp[i - 1] and hp[i] >= hp[(i + 1) % na]]
    return np.array(pk) * 360.0 / na, hp

def scale_from_ticks(tk):
    """等間隔に並ぶ目盛り線のいちばん長い並びを探し、始まり・終わり(時計回り)と間隔を返す"""
    tk = np.sort(tk); n = len(tk)
    dif = np.diff(np.r_[tk, tk[0] + 360])
    m = np.median(dif)
    ok = (dif > 0.55 * m) & (dif < 1.6 * m)
    best = (0, 0)
    for s in range(n):                                  # s から ok が続く長さ
        L = 0
        while L < n and ok[(s + L) % n]: L += 1
        if L > best[0]: best = (L, s)
    L, s = best
    return tk[s], tk[(s + L) % n], L, m


def find_scale(fr_gray):
    """円弧の内側・外側の帯の両方で目盛り線を探し、等間隔の並びが長い方を採る"""
    best = None
    for r0, r1 in ((0.88, 0.99), (1.01, 1.10), (0.93, 1.04)):
        tk, hp = ticks(fr_gray, r0, r1)
        if len(tk) < 5: continue
        s, e, L, m = scale_from_ticks(tk)
        if best is None or L > best[2]: best = (s, e, L, m, (r0, r1), tk)
    return best

def scale_from_ticks(tk):
    """等間隔の並び: 近すぎる線(針など)は飛ばし、抜け(最大2本)は許して、いちばん長く続く範囲"""
    tk = np.sort(tk); n = len(tk)
    dif = np.diff(np.r_[tk, tk[0] + 360]); m = np.median(dif)
    best = (0, 0, 0, 0)
    for s in range(n):
        cur = tk[s]; span = 0; cnt = 1; j = s
        while True:
            nxt = None
            for k in range(1, n):
                d = (tk[(j + k) % n] - cur) % 360
                if d < 0.55 * m: continue
                if d <= 2.5 * m: nxt = (j + k) % n
                break
            if nxt is None or span + (tk[nxt] - cur) % 360 >= 355: break
            span += (tk[nxt] - cur) % 360; cur = tk[nxt]; j = nxt; cnt += 1
        if span > best[0]: best = (span, s, j, cnt)
    span, s, e, cnt = best
    return tk[s], tk[e], cnt, m

_reader = None
def numbers(fr_rgb):
    """EasyOCR で数字を読み、(値, 中心からの角度, 半径, 確からしさ, 文字) のリスト"""
    global _reader
    import easyocr
    if _reader is None: _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    out = []
    for box, txt, cf in _reader.readtext(fr_rgb, allowlist='0123456789.,'):
        t = txt.replace(',', '.').strip('.')
        if not re.fullmatch(r'\d+(\.\d+)?', t): continue
        if len(t) >= 2 and t[0] == '0' and '.' not in t: t = '0.' + t[1:]      # 「02」→ 0.2
        b = np.array(box, float); cx, cy = b.mean(0)
        ang = math.degrees(math.atan2(cy - S / 2, cx - S / 2)) % 360
        out.append((float(t), ang, math.hypot(cx - S / 2, cy - S / 2) / R, float(cf), txt))
    return out

def scale_values(nums, s_ang, e_ang):
    """読めた数字(半径 0.6〜0.95、確からしさ 0.5 以上)から、目盛りの両端の値を求める"""
    span = (e_ang - s_ang) % 360
    pts = []
    for v, a, r, cf, t in nums:
        if cf < 0.5 or not (0.6 < r < 0.95): continue
        u = (a - s_ang) % 360                       # 目盛りの始まりから時計回りの角度
        if u > span + 10: continue
        pts.append((u, v))
    vals = sorted({v for u, v in pts})
    if len(vals) < 2: return None
    u = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
    b, a = np.polyfit(u, v, 1)                     # v = a + b u
    resid = np.abs(a + b * u - v)
    keep = resid < 0.5 * np.median(np.diff(vals))
    if keep.sum() >= 2: b, a = np.polyfit(u[keep], v[keep], 1)
    step = np.median(np.diff(vals)) / 2            # 例: 数字が0.2おき → 0.1 に丸める
    vmin = round(a / step) * step; vmax = round((a + b * span) / step) * step
    return vmin, vmax, sorted(set(zip(np.round(u).tolist(), v.tolist())))

def needle(fr_rgb, r0=0.42, r1=0.72):
    """針の角度: 赤チャンネル(赤い置針が白くなる)の極座標で、各リングの暗さ → 下から30%の値が最大の角度。
    その向きの暗い画素に直線を当てはめ、向きを決める(軸の位置のずれに強くする)"""
    red = fr_rgb[:, :, 0].astype(np.float32)
    na = 1440
    P = polar(red, r0 * R, r1 * R, 30, na)
    d = (np.median(P, 1, keepdims=True) - P) / (np.median(P, 1, keepdims=True) + 8)
    k = 40; pad = np.concatenate([d[:, -k:], d, d[:, :k]], 1)
    hp = d - np.stack([np.median(pad[:, i:i + 2 * k + 1], 1) for i in range(na)], 1)
    score = np.quantile(hp, 0.3, axis=0)
    i = int(np.argmax(score)); ang0 = i * 360.0 / na
    # 直線の当てはめ: ang0 ±8°、半径 0.3〜0.9R の暗い画素
    yy, xx = np.mgrid[0:S, 0:S]; dx = xx - S / 2; dy = yy - S / 2
    rr = np.hypot(dx, dy) / R; aa = np.degrees(np.arctan2(dy, dx)) % 360
    dd = np.abs((aa - ang0 + 180) % 360 - 180)
    m = (rr > 0.3) & (rr < 0.9) & (dd < 8)
    thr = np.percentile(red[m], 12)
    sel = m & (red <= thr)
    if sel.sum() > 30:
        pts = np.stack([xx[sel], yy[sel]], 1).astype(np.float64)
        c = pts.mean(0); _, _, vt = np.linalg.svd(pts - c); dvec = vt[0]
        if np.dot(dvec, [math.cos(math.radians(ang0)), math.sin(math.radians(ang0))]) < 0: dvec = -dvec
        ang1 = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
    else:
        ang1 = ang0
    if abs((ang1 - ang0 + 180) % 360 - 180) > 4: ang1 = ang0      # 直線の当てはめが大きくずれたら使わない
    return ang0, ang1, float(score[i])

def read_gauge(rgb):
    fr, ell, ath = dial_frame(rgb)
    fg = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
    s, e, L, m, band, tk = find_scale(fg)
    nums = numbers(fr)
    sv = scale_values(nums, s, e)
    a0, a1, sc = needle(fr)
    span = (e - s) % 360
    out = {"start": s, "end": e, "span": span, "ticks": L, "needle0": a0, "needle": a1, "nums": sv[2] if sv else None}
    if sv:
        vmin, vmax, _, b = sv
        # 1目盛りの値を 1,2,5×10^k に丸め、目盛りの数から最大値を決め直す(数字の当てはめより確か)
        pt = b * m
        e10 = 10 ** math.floor(math.log10(pt)); nice = min((1, 2, 2.5, 5, 10), key=lambda q: abs(q * e10 - pt)) * e10
        vmax2 = vmin + nice * (L - 1)
        if abs(vmax2 - vmax) <= 2 * nice: vmax = vmax2
        out["per_tick"] = nice
        out.update(vmin=vmin, vmax=vmax, value=vmin + ((a1 - s) % 360) / span * (vmax - vmin),
                   value0=vmin + ((a0 - s) % 360) / span * (vmax - vmin))
    return out, fr

def scale_values(nums, s_ang, e_ang):
    """読めた数字から 値 = a + b×(始まりからの角度) を、2点ずつ選ぶ当てはめ(外れ値に強い)で求める"""
    span = (e_ang - s_ang) % 360
    pts = []
    for v, a, r, cf, t in nums:
        if cf < 0.5 or not (0.6 < r < 0.95): continue
        u = (a - s_ang) % 360
        if u > span + 10: continue
        pts.append((u, v))
    if len(pts) < 2: return None
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (u1, v1), (u2, v2) = pts[i], pts[j]
            if abs(u2 - u1) < 20 or v1 == v2: continue
            b = (v2 - v1) / (u2 - u1)
            if b <= 0: continue                       # 時計回りに増える目盛りだけ
            a = v1 - b * u1
            tol = 0.15 * abs(v2 - v1) / max(1, round(abs(u2 - u1) / 50))
            inl = [(u, v) for u, v in pts if abs(a + b * u - v) < max(tol, 0.03 * abs(b) * span)]
            if best is None or len(inl) > len(best): best = inl
    if not best or len(best) < 2: return None
    u = np.array([p[0] for p in best]); v = np.array([p[1] for p in best])
    b, a = np.polyfit(u, v, 1)
    vals = sorted(set(v.tolist()))
    lstep = min(np.diff(vals)) if len(vals) > 1 else vals[0]
    vmin = round(a / (lstep / 2)) * (lstep / 2)
    vmax = round((a + b * span) / (lstep / 2)) * (lstep / 2)
    return vmin, vmax, sorted(set(zip(np.round(u).tolist(), v.tolist()))), b


if __name__ == "__main__":
    for f in sys.argv[1:]:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        o, fr = read_gauge(rgb)
        if "value" in o:
            print(f"{os.path.basename(f)}  値 {o['value']:.3f}  (目盛り {o['vmin']:g}〜{o['vmax']:g}、振れ角 {o['span']:.1f}°、目盛り線 {o['ticks']}本、読めた数字 {[v for u, v in o['nums']]})")
        else:
            print(f"{os.path.basename(f)}  数字が読めず値を出せない(振れ角 {o['span']:.1f}°)")
