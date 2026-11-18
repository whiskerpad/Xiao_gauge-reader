"""
run_eval.py - XIAO ESP32S3 Sense 単体読み取りの精度評価(司令塔: PC)

流れ: 針を真値へ移動(UNO) → XIAOに /read を要求 → 読み取り値と真値の差を記録
      誤差が閾値を超えた条件は解析画像(/last.jpg)とプロファイルを保存する

  pip install pyserial requests

使い方:
  python run_eval.py --port COM5 --xiao http://gauge-xiao.local --mode grid --step 0.05
  python run_eval.py --port COM5 --xiao http://192.168.1.50 --mode random --count 100
  オプション:
    --n 3            XIAOが1回の読み取りで撮影する枚数(中央値を採用)
    --settle 0.5     移動完了後、読み取り前に追加で待つ秒数
    --err-th 0.01    この誤差[MPa]を超えたら画像を保存
    --save-all       全条件の画像を保存(機械学習用データ収集)
    --tag led_on     照明条件などのラベル(CSVと保存先フォルダ名に付く)
    --no-bg          背景なしで測定する(指定しないと、背景が無効なときは中止する)
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "01_stepper_needle"))
from needle import Needle  # noqa: E402

SCALE_MAX_MPA = 1.0


def targets(args):
    if args.mode == "grid":
        n = int(round(SCALE_MAX_MPA / args.step))
        up = [round(i * args.step, 4) for i in range(n + 1)]
        return up + up[-2::-1]          # 上昇→下降(往復で誤差の向き依存も見る)
    vals = [round(random.uniform(0, SCALE_MAX_MPA), 4) for _ in range(args.count)]
    return vals


def save_image(xiao, path_jpg, path_prof):
    r = requests.get(f"{xiao}/last.jpg", timeout=10)
    with open(path_jpg, "wb") as f:
        f.write(r.content)
    p = requests.get(f"{xiao}/profile", timeout=10).json()
    with open(path_prof, "w", encoding="utf-8") as f:
        json.dump(p, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--xiao", required=True)
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--err-th", type=float, default=0.01)
    ap.add_argument("--save-all", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-bg", action="store_true", help="背景差分なしで測定する")
    args = ap.parse_args()
    xiao = args.xiao.rstrip("/")

    # --- 事前確認: 背景が有効か(フォルダを作る前に判定) ---
    cfg = requests.get(f"{xiao}/config", timeout=10).json()
    mkmode = int(cfg.get("mk", 0)) == 1
    multi = mkmode or int(cfg.get("nb", 0)) >= 2
    if mkmode:
        print(f"マーカーモード(背景なし): マーカー半径 {cfg.get('mkr')}mm, 帯 {cfg.get('m1i')}-{cfg.get('m1o')} / "
              f"{cfg.get('m2i')}-{cfg.get('m2o')} / {cfg.get('m3i')}-{cfg.get('m3o')} mm, 回転中心 ({cfg.get('pcx')},{cfg.get('pcy')}) mm")
        if int(cfg.get("rgon", 0)):
            print(f"  リング方式: 半径 {cfg.get('rg0')}-{cfg.get('rg1')} mm, 割合 {cfg.get('rgq')}(帯の設定は使わない)")
        if int(cfg.get("fpon", 0)):
            print(f"  置針の除外: 位置 {cfg.get('fpv')} ±{cfg.get('fpw')}°")
        print(f"  角度: 0目盛り {cfg.get('ma0')}°, 最大目盛り {cfg.get('ma1')}°")
        mk = requests.get(f"{xiao}/mk/detect", timeout=10).json()
        if not mk.get("ok"):
            print("中止: マーカーを検出できません。ブラウザでマーカー位置を設定し「マーカー検出」で確認してください。")
            sys.exit(1)
    elif multi:
        print(f"多帯モード(背景なし): 帯 {cfg.get('b1i')}-{cfg.get('b1o')} / {cfg.get('b2i')}-{cfg.get('b2o')} / "
              f"{cfg.get('b3i')}-{cfg.get('b3o')} px, ハイパス±{cfg.get('hpw')}°")
    elif not args.no_bg and not cfg.get("bg_valid"):
        print("中止: XIAOの背景が無効です(設定変更で自動削除された可能性があります)。")
        print(f"  先に実行: python record_background.py --port {args.port} --xiao {xiao}")
        print("  背景なしで測定する場合は --no-bg を付けてください。")
        sys.exit(1)
    if not multi and args.no_bg and cfg.get("bg_valid"):
        print("注意: --no-bg 指定ですが、XIAOの背景は有効です。XIAO側で背景差分が使われます。")
        print("  背景なしで測る場合は、ブラウザで「背景 削除」を押してから実行してください。")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "dataset", "xiao_s3", f"{stamp}_{args.tag}")
    imgdir = os.path.join(outdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    csv_path = os.path.join(outdir, "result.csv")

    print(f"直線補正: a={cfg.get('ca', 1)} b={cfg.get('cb', 0)}")
    if not multi:
        print("背景差分:", "使用" if cfg.get("bg_valid") else "未使用(record_background.py で記録可能)")
    with open(os.path.join(outdir, "xiao_config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "args": vars(args), "bg_used": (not multi) and bool(cfg.get("bg_valid")), "mode": "marker" if mkmode else ("multi" if multi else "single")}, f, ensure_ascii=False, indent=1)

    needle = Needle(args.port)
    errors = []
    fails = 0
    try:
        print(needle.home()["raw"])
        tl = targets(args)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "time", "tag", "true_mpa", "needle_deg", "ok", "read_mpa", "raw_mpa", "snr", "mk_ok",
                        "err_mpa", "bg", "read_angle", "v_lo", "v_hi", "width_deg", "contrast", "gain", "sat",
                        "analyze_us", "read_ms", "image"])
            for i, t in enumerate(tl):
                mv = needle.move_mpa(t)
                time.sleep(args.settle)
                try:
                    r = requests.get(f"{xiao}/read", params={"n": args.n}, timeout=20).json()
                except requests.RequestException as e:
                    r = {"ok": False, "error": str(e)}

                if mkmode and "mk" in r and not r["mk"]:
                    print("中止: 測定中にマーカーモードが解除されました(XIAOの設定変更など)。")
                    sys.exit(1)
                if multi and not mkmode and "multi" in r and not r["multi"]:
                    print("中止: 測定中に多帯モードが解除されました(XIAOの設定変更など)。")
                    sys.exit(1)
                if not multi and not args.no_bg and "bg" in r and not r["bg"]:
                    print("中止: 測定中に背景が無効になりました(XIAOの再起動・設定変更など)。")
                    sys.exit(1)
                ok = bool(r.get("ok"))
                err = (r["value"] - t) if ok else float("nan")
                img = ""
                if args.save_all or not ok or abs(err) > args.err_th:
                    img = f"{i:05d}_{t:.4f}.jpg"
                    try:
                        save_image(xiao, os.path.join(imgdir, img),
                                   os.path.join(imgdir, img.replace(".jpg", "_profile.json")))
                    except requests.RequestException:
                        img = "save_failed"

                w.writerow([i, datetime.now().isoformat(timespec="seconds"), args.tag, t,
                            mv.get("deg", ""), int(ok), r.get("value", ""), r.get("raw", ""), r.get("snr", ""), r.get("mk_ok", 1 if r.get("ok") else ""),
                            "" if not ok else round(err, 5), int(bool(r.get("bg"))), r.get("angle", ""),
                            r.get("v_lo", ""), r.get("v_hi", ""), r.get("width", ""),
                            r.get("contrast", ""), r.get("gain", ""), r.get("sat", ""), r.get("analyze_us", ""), r.get("ms", ""), img])
                f.flush()
                if ok:
                    errors.append(err)
                else:
                    fails += 1
                mark = "" if ok and abs(err) <= args.err_th else "  <-- NG"
                print(f"[{i+1}/{len(tl)}] true={t:.4f} read={r.get('value', '-')} "
                      f"err={err:+.4f}{mark}" if ok else f"[{i+1}/{len(tl)}] true={t:.4f} READ FAIL {r}")
        needle.move_mpa(0)
    finally:
        needle.close()

    # 集計
    lines = [f"条件: {args.tag}  試行数: {len(errors) + fails}  読み取り失敗: {fails}"]
    if errors:
        ae = [abs(e) for e in errors]
        fs = SCALE_MAX_MPA
        lines += [
            f"平均誤差(偏り): {statistics.mean(errors):+.5f} MPa",
            f"平均絶対誤差  : {statistics.mean(ae):.5f} MPa ({statistics.mean(ae)/fs*100:.2f}%FS)",
            f"RMSE          : {math.sqrt(statistics.mean([e*e for e in errors])):.5f} MPa",
            f"最大絶対誤差  : {max(ae):.5f} MPa ({max(ae)/fs*100:.2f}%FS)",
            f"±1%FS以内     : {sum(a <= 0.01*fs for a in ae)/len(ae)*100:.1f}%",
            f"±2%FS以内     : {sum(a <= 0.02*fs for a in ae)/len(ae)*100:.1f}%",
        ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n保存先: {outdir}")


if __name__ == "__main__":
    main()
