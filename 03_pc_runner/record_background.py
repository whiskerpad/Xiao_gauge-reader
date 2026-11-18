"""
record_background.py - 背景(針のない文字盤)をXIAOに記録させる

UNOで針を目盛り全域に動かしながら、XIAOの /bg/add を繰り返す。
XIAOは角度ごとに「針が写っていないフレーム」の平均を取る。
針の位置(PCが指示した値)の前後 ±skipw 度は平均から除外するので、針のない文字盤=背景ができる。
数字・目盛り線・ロゴ・固定の赤針などは背景に含まれ、読み取り時に差し引かれる。

  python record_background.py --port COM4 --xiao http://192.168.x.x
  オプション:
    --step 0.05    針を動かす刻み [MPa](針の幅より細かくする。0.05MPa=13.5°)
    --settle 0.3   移動後の待ち時間 [s]
    --skipw 15     針の前後で背景から除外する角度 [deg]
    --frames 3     1位置あたりの撮影枚数

注意:
  - 先にブラウザで中心・半径・目盛り角を設定保存しておく(変更すると背景は削除される)
  - カメラ位置・照明を変えたら記録し直す
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "01_stepper_needle"))
from needle import Needle  # noqa: E402

SCALE_MAX_MPA = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--xiao", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--settle", type=float, default=0.3)
    ap.add_argument("--skipw", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    xiao = args.xiao.rstrip("/")

    cfg = requests.get(f"{xiao}/config", timeout=10).json()
    print("XIAO設定:", cfg)

    needle = Needle(args.port)
    try:
        print(needle.home()["raw"])
        requests.get(f"{xiao}/bg/reset", timeout=10)
        n = int(round(SCALE_MAX_MPA / args.step))
        vals = [round(i * args.step, 4) for i in range(n + 1)]
        for i, v in enumerate(vals):
            needle.move_mpa(v)
            time.sleep(args.settle)
            st = requests.get(f"{xiao}/bg/add",
                              params={"n": args.frames, "skipv": v, "skipw": args.skipw},
                              timeout=30).json()
            print(f"[{i+1}/{len(vals)}] {v:.3f} MPa  frames={st['count']}  白飛び={st.get('sat', 0)*100:.0f}%")
        r = requests.get(f"{xiao}/bg/save", timeout=10)
        if r.status_code != 200:
            print("背景保存に失敗:", r.text)
            sys.exit(1)
        print("背景保存:", r.json())
        needle.move_mpa(0)
    finally:
        needle.close()

    # 記録(ブログ・検証用)
    bg = requests.get(f"{xiao}/bg", timeout=10).json()
    outdir = os.path.join(ROOT, "dataset", "background")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, datetime.now().strftime("bg_%Y%m%d_%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "background": bg, "step": args.step}, f, ensure_ascii=False)
    print("保存:", path)

    # 背景の品質チェック
    v = bg.get("values", [])
    if v:
        sat = sum(1 for x in v if x >= 245) / len(v)
        rough = sum(abs(v[i - 1] - 2 * v[i] + v[i + 1]) for i in range(1, len(v) - 1)) / (len(v) - 2)
        print(f"背景: 輝度 最小{min(v):.0f} 最大{max(v):.0f}  白飛び{sat*100:.1f}%  凹凸(2次差分平均){rough:.1f}")
        if sat > 0.02:
            print("  注意: 白飛びあり。照明を弱めるか露出を下げて、背景を記録し直すこと")
        if max(v) < 100:
            print("  注意: 全体が暗い。照明を強めるか露出を上げること")


if __name__ == "__main__":
    main()
