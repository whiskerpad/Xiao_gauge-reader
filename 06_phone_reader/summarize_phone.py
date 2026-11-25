# -*- coding: utf-8 -*-
"""summarize_phone.py - iPhone 読み取りの誤差を、姿勢・正解の値ごとに集計する
  python summarize_phone.py [results\\20261122\\results.csv ...][--since "2026-11-22 07:00"](省略時は results フォルダの全 results.csv)
正解(truth)が入っている行だけを使う(phone_gauge_server.py --needle COMx で測った行)。誤差は %FS(1.0MPa = 100%)。
"""
import csv
import glob
import os
import sys
from collections import defaultdict

COLS = ["time", "file", "value", "conf", "ok", "reason", "okn", "spread", "vals", "confs",
        "rim", "cx", "cy", "d", "w", "h", "ms", "truth", "pose"]          # 見出し行が無いときの列
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
since = ""                                   # --since "2026-11-22 07:00" でそれ以降の行だけ
if args[:1] == ["--since"]:
    since, args = args[1], args[2:]
files = args or sorted(glob.glob(os.path.join(HERE, "results", "*", "results.csv")))
rows = []
for p in files:
    with open(p, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    head = next(csv.reader(lines[:1]), [])
    for r in (csv.DictReader(lines) if "truth" in head else csv.DictReader(lines, fieldnames=COLS)):
        if r.get("truth") and r["time"] >= since:
            rows.append(r)
if not rows:
    sys.exit("正解(truth)の入った行がありません。--needle を付けて測定してください")


def stats(rs):
    n = len(rs)
    ok = [r for r in rs if r["ok"] == "1"]
    e = [(float(r["value"]) - float(r["truth"])) * 100 for r in ok]
    if not e:
        return f"{n:3d}回  値あり 0"
    ae = [abs(x) for x in e]
    return (f"{n:3d}回  値あり {len(ok):3d}({len(ok) / n:4.0%})  平均誤差 {sum(e) / len(e):+5.2f}  平均絶対 {sum(ae) / len(ae):4.2f}  "
            f"最大 {max(ae):5.2f}  ±2%FS以内 {sum(x <= 2 for x in ae) / len(ae):4.0%}  ±5%超 {sum(x > 5 for x in ae)}")


by = defaultdict(list)
for r in rows:
    by[(r["pose"] or "-", r["truth"])].append(r)
print("姿勢   正解   (誤差は %FS。平均誤差の符号: + は多め)")
for pose in sorted({k[0] for k in by}):
    for t in sorted({k[1] for k in by if k[0] == pose}, key=float):
        print(f"{pose:6s} {float(t):.2f}  {stats(by[(pose, t)])}")
    print(f"{pose:6s} 全体  {stats([r for r in rows if (r['pose'] or '-') == pose])}")
print(f"全体         {stats(rows)}")
