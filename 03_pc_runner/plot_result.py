"""
plot_result.py - run_eval.py の結果をグラフ化(ブログ掲載用)
  pip install matplotlib
  python plot_result.py ..\\dataset\\xiao_s3\\<日時_tag>\\result.csv
出力: 同じフォルダに error_plot.png(真値-誤差の散布図)
"""
import csv
import os
import sys

import matplotlib.pyplot as plt


def main():
    path = sys.argv[1]
    xs, es = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] == "1":
                xs.append(float(row["true_mpa"]))
                es.append(float(row["err_mpa"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(xs, es, s=12)
    ax.axhline(0, color="gray", lw=0.8)
    for v in (0.01, -0.01):
        ax.axhline(v, color="tab:red", lw=0.8, ls="--")
    ax.set_xlabel("True value [MPa]")
    ax.set_ylabel("Error [MPa]")
    ax.set_title("XIAO ESP32S3 Sense gauge reading error (dashed: ±1%FS)")
    ax.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(path), "error_plot.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
