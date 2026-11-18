"""
needle_wander.py - 疑似針を 0.5〜1.0 MPa の間でランダムな速さで往復させ、指令値を Flask サーバーへ送る

  python needle_wander.py --port COM8 --server http://127.0.0.1:5000
    --lo 0.5 --hi 1.0         往復範囲 [MPa]
    --vmin 0.003 --vmax 0.03  速さの範囲 [MPa/秒](区間ごとにランダムに選ぶ)
    --seg 5 20                1区間の長さ [秒](この間は同じ速さ)
    --pause 0.15              区間が「停止」になる確率
    --tick 0.5                位置を問い合わせて指令値を送る間隔 [秒]
  移動は UNO の SLEW 指令(加減速付きの連続移動)で行うので、針は滑らかに動く(needle_stepper.ino の書き込みが必要)
  Ctrl+C で停止(針はその位置で止まる)

指令値(UNO が返した実際の位置 mpa)を /api/truth に送るので、トレンド画面で読み取り値と比較できる。
"""
import argparse
import random
import time

import requests

from needle import Needle


def post_truth(server, value):
    if not server:
        return
    try:
        requests.post(f"{server.rstrip('/')}/api/truth", json={"value": value, "ts": time.time()}, timeout=2)
    except requests.RequestException as e:
        print(f"  (サーバーへ送信できません: {e.__class__.__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--vmin", type=float, default=0.003)
    ap.add_argument("--vmax", type=float, default=0.03)
    ap.add_argument("--seg", type=float, nargs=2, default=[5.0, 20.0])
    ap.add_argument("--pause", type=float, default=0.15)
    ap.add_argument("--tick", type=float, default=0.5, help="位置を問い合わせて指令値を送る間隔[秒]")
    ap.add_argument("--no-home", action="store_true", help="原点復帰を省略(電源を切っていない場合)")
    args = ap.parse_args()

    nd = Needle(args.port)
    if not args.no_home:
        print("原点復帰...")
        nd.home()
    pos = nd.move_mpa(args.lo).get("mpa", args.lo)
    post_truth(args.server, pos)
    direction = 1
    print(f"開始: {args.lo}〜{args.hi} MPa を滑らかに往復(SLEW)  (Ctrl+C で停止)")
    try:
        while True:
            seg_len = random.uniform(*args.seg)
            speed = 0.0 if random.random() < args.pause else random.uniform(args.vmin, args.vmax)
            bound = args.hi if direction > 0 else args.lo
            if speed > 0:
                nd.cmd(f"SLEW {bound:.4f} {speed:.4f}")      # UNO 側で加減速しながら連続移動
            else:
                nd.cmd("STOP")
            print(f"区間 {seg_len:4.1f}s  速さ {speed:.4f} MPa/s  {'停止' if speed == 0 else ('上昇' if direction > 0 else '下降')}")
            t_end = time.time() + seg_len
            while time.time() < t_end:
                t = time.time()
                pos = nd.pos().get("mpa", pos)
                post_truth(args.server, pos)
                if speed > 0 and abs(pos - bound) < 0.002:   # 端に着いたら折り返す(速さはそのまま)
                    direction = -direction
                    bound = args.hi if direction > 0 else args.lo
                    nd.cmd(f"SLEW {bound:.4f} {speed:.4f}")
                    print(f"  折り返し → {'上昇' if direction > 0 else '下降'}")
                wait = args.tick - (time.time() - t)
                if wait > 0:
                    time.sleep(wait)
    except KeyboardInterrupt:
        print("停止")
        try:
            nd.cmd("STOP")
        except Exception:
            pass
    finally:
        nd.close()


if __name__ == "__main__":
    main()
