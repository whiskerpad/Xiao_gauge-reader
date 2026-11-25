"""
eval_nn.py - XIAO の CNN 読み取り(/nn)を、疑似圧力計の新しい位置で評価する
  python eval_nn.py --port COM8 --xiao http://192.168.11.5 --count 50 --tag nn_front
判定: 誤差±2%FS以内=正解、値が出なかった=未取得、それ以外=誤読
出力: results/<日時>_<tag>/result.csv, summary.txt, images/(全点の最後の1枚。次の評価・学習に使う)
"""
import argparse, csv, json, os, random, statistics, sys, time
from datetime import datetime
import requests
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_stepper_needle"))
from needle import Needle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--xiao", required=True)
ap.add_argument("--count", type=int, default=50); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--settle", type=float, default=0.3); ap.add_argument("--tag", default="nn")
ap.add_argument("--cf", type=float, default=None, help="自信の下限(省略時はXIAOの保存値)")
ap.add_argument("--x0", type=int, default=None); ap.add_argument("--y0", type=int, default=None)
a = ap.parse_args(); xiao = a.xiao.rstrip("/")
out = os.path.join(HERE, "results", datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + a.tag)
os.makedirs(os.path.join(out, "images"), exist_ok=True)
q = f"n={a.n}" + (f"&cf={a.cf}" if a.cf else "") + (f"&x0={a.x0}" if a.x0 is not None else "") + (f"&y0={a.y0}" if a.y0 is not None else "")
r = requests.get(f"{xiao}/nn?n=1", timeout=20).json()
if "nn" not in r: raise SystemExit("XIAO が /nn に対応していません(ファームを書き込んでください)")
nd = Needle(a.port); nd.home()
rows = []
try:
    for i in range(a.count):
        t = round(random.uniform(0, 1.0), 4)
        res = nd.move_mpa(t); truth = float(res.get("mpa", t)); time.sleep(a.settle)
        j = requests.get(f"{xiao}/nn?{q}", timeout=30).json()
        err = (j["value"] - truth) if j.get("ok") else None
        st = "未取得" if err is None else ("正解" if abs(err) <= 0.02 else "誤読")
        rows.append({"i": i + 1, "truth": truth, "value": j.get("value") if j.get("ok") else "", "err": "" if err is None else round(err, 4),
                     "status": st, "v_lo": j.get("v_lo"), "v_hi": j.get("v_hi"), "conf_min": j.get("conf_min"), "rej": j.get("rej"), "dx": j.get("dx"), "dy": j.get("dy"), "ncc": j.get("ncc"), "fv": json.dumps(j.get("fv", [])), "infer_us": j.get("infer_us"), "ms": j.get("ms")})
        print(f"{i+1:3d} 真値 {truth:.4f}  読み {j.get('value'):.4f}  {st}" + ("" if err is None else f"  誤差 {err*100:+.2f}%FS"))
        if True:   # 全点の画像を保存(次の学習・別セッションのテストに使う)
            open(os.path.join(out, "images", f"{i+1:03d}_{truth:.3f}.jpg"), "wb").write(requests.get(f"{xiao}/last.jpg", timeout=10).content)
except KeyboardInterrupt:
    print("中断")
finally:
    nd.off(); nd.close()
with open(os.path.join(out, "result.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
e = [abs(r["err"]) for r in rows if r["err"] != ""]
n = len(rows); ok = sum(r["status"] == "正解" for r in rows); bad = sum(r["status"] == "誤読" for r in rows)
s = (f"件数 {n}  正解 {ok}  誤読 {bad}  未取得 {n-len(e)}  点数 {ok/n*100:.0f}\n"
     + (f"平均絶対誤差 {statistics.mean(e)*100:.2f}%FS  最大 {max(e)*100:.2f}%FS  ±1%FS以内 {sum(x<=0.01 for x in e)/len(e)*100:.0f}%\n" if e else "")
     + f"推論時間 最大 {max(int(r['infer_us'] or 0) for r in rows)/1000:.1f}ms\n")
# 各画像の [値, 自信, dx, dy](fv)から、自信の下限を変えた場合の成績をPCで計算する(XIAOと同じ判定)
def judge(fv, th):
    vals = sorted(f[0] for f in fv if f[1] >= th and f[0] == f[0])
    if not vals: return None
    v = vals[len(vals) // 2]
    if vals[-1] - vals[0] > 0.03 or not (-0.05 < v < 1.05): return None
    return min(max(v, 0.0), 1.0)
if any(r["fv"] != "[]" for r in rows):
    s += "\n自信の下限ごとの成績(同じ画像をPCで再判定)\n"
    for th in (0.95, 0.9, 0.8, 0.7, 0.6, 0.0):
        c = [0, 0, 0]; errs = []
        for r in rows:
            v = judge(json.loads(r["fv"]), th)
            if v is None: c[2] += 1; continue
            e2 = abs(v - r["truth"]); errs.append(e2); c[0 if e2 <= 0.02 else 1] += 1
        s += f"  {th:<5} 正解 {c[0]:2d}  誤読 {c[1]:2d}  未取得 {c[2]:2d}" + (f"  採用分 平均 {statistics.mean(errs)*100:.2f}%FS 最大 {max(errs)*100:.2f}%FS" if errs else "") + "\n"
    s += f"位置ずれ dx={rows[-1]['dx']} dy={rows[-1]['dy']} ncc={rows[-1]['ncc']}\n"
open(os.path.join(out, "summary.txt"), "w", encoding="utf-8").write(s); print(s + "→ " + out)
