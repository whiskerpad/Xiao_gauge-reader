"""export_polar.py - 極座標CNN(train_polar.py の .pt)と位置合わせの基準を XIAO 用 nn_polar_data.h に書き出す
  python export_polar.py models\\net_polar_s0.pt models\\reg_ref.npy ..\\02_xiao_reader\\include\\nn_polar_data.h
reg_ref.npy: 基準画像(学習時のカメラ位置の平均画像、640x480 float)
"""
import sys, numpy as np, torch
import reg2, polar
from train_polar import A0, SPAN
pt, refp, out = sys.argv[1:4]
R = reg2.Ref(np.load(refp)); sd = torch.load(pt); L = []
def arr(name, a, typ="float", fmt=lambda v: f"{v:.8e}f", per=8):
    a = np.asarray(a).ravel()
    L.append(f"static const {typ} {name}[{a.size}] = {{\n" + ",\n".join(", ".join(fmt(v) for v in a[i:i + per]) for i in range(0, a.size, per)) + "\n};\n")
for n in ["b1", "b2", "b3", "b4", "b5"]:
    arr(f"PL_{n}_w", sd[f"{n}.c.weight"].numpy()); arr(f"PL_{n}_b", sd[f"{n}.c.bias"].numpy())
arr("PR_Q1", R.q1, "uint8_t", lambda v: str(int(v)), 32); arr("PR_Q2", R.q2, "uint8_t", lambda v: str(int(v)), 32)
h = f"""// nn_polar_data.h - 自動生成(05_ml/export_polar.py)。手で編集しない
#pragma once
#define PL_NA 180
#define PL_NR 32
#define PL_R0 {polar.R0}f
#define PL_DR {polar.DR}f
#define PL_HP {polar.HP}
#define PL_CX {polar.CX}f
#define PL_CY {polar.CY}f
#define PL_A0 {A0}
#define PL_SPAN {SPAN}
#define PR_ECLIP {reg2.ECLIP}f
#define PR_RX0 {reg2.RX0}
#define PR_RY0 {reg2.RY0}
#define PR_RW {reg2.RW}
#define PR_RH {reg2.RH}
#define PR_SPAN2 24
""" + "".join(L)
open(out, "w", encoding="utf-8").write(h); print("written", out)
