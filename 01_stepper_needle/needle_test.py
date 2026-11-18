"""
needle_test.py - 疑似圧力計の単体動作確認・校正ツール

使い方:
  python needle_test.py COM5 sweep          # 原点復帰後、0→1.0MPa→0 を0.1MPa刻み
  python needle_test.py COM5 random 20      # ランダム値へ20回移動(再現性確認)
  python needle_test.py COM5 calib          # 対話式で0目盛り・最大目盛りを校正
  (ラズパイでは /dev/ttyACM0 や /dev/ttyUSB0)
"""
import random
import sys
import time

from needle import Needle

SCALE_MAX_MPA = 1.0


def do_sweep(n):
    n.home()
    vals = [round(i * 0.1, 2) for i in range(11)]
    for v in vals + vals[-2::-1]:
        r = n.move_mpa(v)
        print(f"target={v:.2f}MPa -> {r['raw']}")
        time.sleep(0.5)


def do_random(n, count):
    n.home()
    for i in range(count):
        v = round(random.uniform(0, SCALE_MAX_MPA), 3)
        r = n.move_mpa(v)
        print(f"[{i+1}/{count}] target={v:.3f}MPa -> {r['raw']}")
        time.sleep(0.5)
    n.move_mpa(0)


def do_calib(n):
    print("原点復帰します")
    n.home()
    print("0目盛りに針を合わせる: 数値(半ステップ, 例 5 / -3)を入力、合ったら z")
    while True:
        s = input("step> ").strip().lower()
        if s == "z":
            print(n.zero()["raw"]); break
        if s.lstrip("-").isdigit():
            print(n.step(int(s))["raw"])
    print("最大目盛り付近へ移動します")
    n.move_deg(270)
    print("最大目盛りに針を合わせる: 数値を入力、合ったら f")
    while True:
        s = input("step> ").strip().lower()
        if s == "f":
            print(n.full()["raw"]); break
        if s.lstrip("-").isdigit():
            print(n.step(int(s))["raw"])
    print("表示された ZERO_OFFSET_STEPS と fullSteps を .ino の校正値へ転記して再書き込みする")


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    n = Needle(sys.argv[1])
    try:
        mode = sys.argv[2]
        if mode == "sweep":
            do_sweep(n)
        elif mode == "random":
            do_random(n, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
        elif mode == "calib":
            do_calib(n)
        else:
            print(__doc__)
    finally:
        n.close()


if __name__ == "__main__":
    main()
