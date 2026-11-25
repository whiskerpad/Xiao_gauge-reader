# コードの用途と使い方ガイド

gauge-reader の各フォルダにあるコードが何のためのものか、どう使うかをまとめた一覧です(2026-11-25 時点)。
経緯・測定結果・課題は [DEVLOG.md](DEVLOG.md)、カメラを付け直したときの手順は [ALIGNMENT.md](ALIGNMENT.md) を見てください。

## 全体の構成

```
 疑似圧力計                    PC(司令塔・学習)                     XIAO ESP32S3 Sense
 UNO + ステッピングモーター <─USB─> 01 needle.py で針を動かす   <─Wi-Fi─> 02 ファーム
 (正解の値を作る)                   05 画像を集めて学習 → C のヘッダ         /nn で読む(極座標CNN)
                                     03 古典的な方式の評価                    /read で読む(画像処理)
                                     04 受信サーバー・トレンド  <─POST──────  定期読み取り
 iPhone(Safari) ──https──> 06 PC で読む
 写真 ─────────────────────> 07 学習なしで読む(試作)
```

## 準備するもの

| 用途 | 入れるもの |
|---|---|
| Python 全般 | Python 3.10 以上。`pip install pyserial requests numpy opencv-python pillow` |
| 学習(05) | `pip install torch`(CPU 版でよい) |
| トレンド(04) | `pip install flask` |
| 学習なしの試作(07) | `pip install easyocr` |
| XIAO のファーム(02) | VS Code + PlatformIO |
| 疑似圧力計(01) | Arduino IDE(UNO に書き込む) |

コマンドは各フォルダで実行します。PowerShell で .bat を動かすときは `.\run_xxx.bat` のように先頭に `.\` を付けます。
COM 番号(UNO は COM8、XIAO は COM5)と XIAO の IP(192.168.11.5)は作者の環境の値です。違うときは .bat とコマンドを書き換えてください。

## やりたいこと別の入口

| やりたいこと | 使うもの |
|---|---|
| 疑似圧力計を組んで動かす | 01 の `needle_stepper.ino` → `needle_test.py` |
| XIAO にファームを書き込む | 02(VS Code の PlatformIO で Upload) |
| 機械学習用の画像を集める | 05 `run_check.bat` → `run_collect.bat` |
| 学習して XIAO に載せる | 05 `run_train_polar.bat` → 02 を Upload |
| XIAO の読み取りの精度を測る | 05 `run_eval_nn.bat` |
| **カメラ(ブラケット)を付け直した** | 05 `run_hom_check.bat` → `--upload`([ALIGNMENT.md](ALIGNMENT.md)) |
| 読み取り値をグラフで見続ける | 04 `run_server_nn.bat` + 01 `run_wander.bat` |
| iPhone で読む | 06 `start_phone_reader.bat` |
| 学習せずに写真から読む | 07 `dial_auto.py` |
| 古典的な画像処理の方式を評価する | 03 `run_eval.py` |

## 01_stepper_needle 疑似圧力計

本物の文字盤の上で、ステッピングモーター(28BYJ-48)が針を動かします。PC から「0.35MPa に動かせ」と指示でき、その値が学習と評価の正解になります。

| ファイル | 用途 | 使い方 |
|---|---|---|
| `needle_stepper/needle_stepper.ino` | UNO のスケッチ。シリアル(115200)で `HOME` `MPA 0.35` `DEG` `STEP` `POS` `OFF` などを受ける | Arduino IDE で UNO に書き込む |
| `motor_check/motor_check.ino` | 配線確認(±5°往復するだけ) | 配線直後に1回 |
| `needle.py` | 上の指令を Python から送るクラス。05・03・04 が使う | `from needle import Needle` |
| `needle_test.py` | 動作確認と校正 | `python needle_test.py COM8 sweep` / `random 20` / `calib` |
| `needle_wander.py` / `run_wander.bat` | 0.5〜1.0MPa をランダムな速さで往復させ、指令値を 04 のサーバーへ送る(トレンドの正解) | `.\run_wander.bat` |

針を1つの値に動かすだけなら(05 フォルダから):
```
python -c "import sys; sys.path.insert(0, r'..\01_stepper_needle'); from needle import Needle; n=Needle('COM8'); n.home(); print(n.move_mpa(1.0)); n.close()"
```
注意: 28BYJ-48 のギアのガタで、上下どちらから近づけたかによって 0.2〜0.9%FS 位置が違います。1%FS より細かい評価には向きません。

## 02_xiao_reader XIAO のファーム

| ファイル | 用途 |
|---|---|
| `src/main.cpp` | 本体。Wi-Fi、カメラ、HTTP の窓口 |
| `include/nn_polar.h` | 機械学習での読み取り(位置合わせ → 極座標の帯画像 → CNN)と、カメラ位置の補正(射影変換) |
| `include/nn_polar_data.h` | 学習した重みと位置合わせの基準。05 の `export_polar.py` が作る(手で編集しない) |
| `include/gauge.h` | 古典的な画像処理の読み取り(`/read`、マーカー方式など) |
| `include/webpage.h` | 設定画面(`/`) |
| `include/secrets.h` | Wi-Fi の SSID とパスワード。`secrets.h.example` をコピーして作る(公開しない) |
| `platformio.ini` | ビルド設定。`upload_port = COM5` |

書き込み: VS Code で `02_xiao_reader` を開き、PlatformIO の Upload。起動後シリアルに IP が出ます。

主な窓口(ブラウザで開ける):

| URL | 内容 |
|---|---|
| `/nn?n=3` | 機械学習で3枚読み、JSON で返す(value、自信 conf、ok、位置合わせの ncc・dx・dy、補正 hom) |
| `/nn?cf=0.8&save=1` | 採用する自信の下限を変えて保存(初期値 0.8) |
| `/hom` | カメラ位置の補正の状態。`?m=…` で設定、`?clear=1` で解除(普段は 05 の `hom_check.py` が送る) |
| `/capture.jpg` | 今の画像(補正前) |
| `/log/set?on=1&intv=10&n=3&nn=1&url=…` | 定期読み取りと送信先(04 のサーバーが自動で設定する) |
| `/read`、`/config`、`/bg`、`/mk` など | 古典的な方式(記事(1))。詳しくは README.md |

## 03_pc_runner 古典的な方式の評価(記事(1))

画像処理(暗い角度を探す方式、マーカー方式)の `/read` を評価するためのツールです。機械学習版では使いません。

| ファイル | 用途 | 使い方 |
|---|---|---|
| `run_eval.py` | 針を動かして `/read` と正解を比べ、CSV と画像を保存 | `python run_eval.py --port COM8 --xiao http://192.168.11.5 --mode random --count 50 --tag 名前` |
| `run_mk_*.bat` | 記事(1)の各条件(斜め1〜7、リング方式)で上を実行した記録 | そのまま実行 |
| `record_background.py` | 針のない文字盤(背景)を XIAO に記録(単一帯モード用) | `--port COM8 --xiao …` |
| `fit_correction.py` | 測定結果から直線補正を求め、XIAO に書く | `python fit_correction.py <result.csv> [--apply --xiao …]` |
| `offline_eval.py` | 保存画像で `/read` の処理を PC 上に再現し、設定を比べる | `python offline_eval.py <測定フォルダ> --bands 50-70,75-95 --hpw 8` |
| `plot_result.py` | 誤差のグラフ | `python plot_result.py <result.csv>` |

## 04_pi_server 受信サーバーとトレンド

| ファイル | 用途 | 使い方 |
|---|---|---|
| `app.py` | XIAO の定期読み取りを受けて保存(SQLite)し、ブラウザにトレンドグラフ。疑似針の指令値も重ねる | 下の .bat |
| `run_server_nn.bat` | 機械学習版(`/nn`)で10秒ごと | `.\run_server_nn.bat` → `http://127.0.0.1:5000/` |
| `run_server.bat` | 古典的な方式(`/read`)で5秒ごと | `.\run_server.bat` |

トレンドで比較するときは、別の画面で 01 の `run_wander.bat` を動かします。補正中(`/hom`)の XIAO は1回の読み取りに約7秒かかるので、間隔は10秒以上にします。

## 05_ml 機械学習(XIAO 用の極座標CNN と、斜め・スマホ用のモデル)

### XIAO 用(主に使うもの)

| ファイル | 用途 | 使い方 |
|---|---|---|
| `run_check.bat`(`collect.py --check`) | XIAO で1枚撮り `data\check.jpg` に保存(構図の確認) | `.\run_check.bat` |
| `collect.py` / `run_collect*.bat` | 針を乱数の位置に動かして XIAO で撮り、画像と正解を `data\<日時>_<名前>\` に保存。照明ごとの .bat あり | `.\run_collect.bat`(2000位置、約2.7時間) |
| `calib_angles.py` | 自分のカメラ・計器で、針の軸・0目盛りの角度・振れ角を求める | `python calib_angles.py --pivot x,y --zero x,y --full x,y` → 出た値を `polar.py`(CX, CY)と `train_polar.py`(A0, SPAN)に書く |
| `make_npz.py` | 集めた画像を学習用の .npz にする。`ref` で位置合わせの基準を作る | `python make_npz.py ref data\<フォルダ> [--force]`、`python make_npz.py polar data\<フォルダ> --name front` |
| `polar.py` / `reg2.py` | 極座標の帯画像と位置合わせ(XIAO と同じ計算)。ほかのスクリプトが使う | — |
| `train_polar.py` | 極座標CNN の学習 | `python train_polar.py --train front --epochs 60 --out models\my_polar.pt` |
| `export_polar.py` | 学習したモデルと基準を XIAO 用 `nn_polar_data.h` に書き出す | `python export_polar.py models\my_polar.pt models\reg_ref.npy ..\02_xiao_reader\include\nn_polar_data.h` |
| `run_train_polar.bat` | 上の ref → polar → 学習 → 書き出しを一度に | `.\run_train_polar.bat data\<フォルダ>` |
| `test_polar.cpp` | XIAO の C の計算が Python と同じか確かめる(PC で g++) | 開発用 |
| `eval_nn.py` / `run_eval_nn.bat` | 針を新しい位置に動かして XIAO の `/nn` を評価(±2%FS 以内・未取得・誤読) | `.\run_eval_nn.bat` |
| `align_view.py` / `run_align.bat` | XIAO のライブ画像に学習時の基準を重ね、ずれ・倍率・斜めを表示(カメラを元の位置に戻すとき) | `.\run_align.bat` |
| `hom_check.py` / `run_hom_check.bat` | カメラ位置の補正。クリックで射影変換を求め、PC で読む・測る・XIAO に送る | [ALIGNMENT.md](ALIGNMENT.md) |
| `models/net_polar_s0.pt`、`models/reg_ref.npy` | 今 XIAO に載っているモデルと位置合わせの基準 | — |
| `models/hom_ref_pts.json` | 補正用に基準画像でクリックした点 | — |

### 斜め・スマホ用(PC で動かす)

| ファイル | 用途 | 使い方 |
|---|---|---|
| `collect_iphone.py` / `run_collect_iphone*.bat`、`run_check_iphone.bat` | iPhone(iphone_capture)で撮って集める | 05_ml/README.md |
| `kuro.py` | 黒縁の内側の楕円を探す | ほかのスクリプトが使う |
| `canvas.py` | 縁の楕円で大きさ・位置をそろえた画像にする | ほかのスクリプトが使う |
| `train_persp.py` | 正面の画像を人工的に斜めにして学習(PC 用) | `python train_persp.py --epochs 40 --out persp.pt` |
| `train_phone.py` | XIAO と iPhone の画像でスマホ用モデル(net_phone_v1)を学習 | `python train_phone.py --epochs 200 --out phone.pt` |
| `read_phone.py` | 切り抜いた写真をスマホ用モデルで読む | `python read_phone.py <画像かフォルダ>` |
| `augment_dataset.py` / `run_augment.bat` | 画像を斜め・回転・明るさ違いに変形して学習データを増やす | `python augment_dataset.py <元> <先> --count 5` |

データ(`data/`)、結果(`results/`)、学習用の .npz は GitHub に上げません(.gitignore)。

## 06_phone_reader iPhone で読む

| ファイル | 用途 | 使い方 |
|---|---|---|
| `phone_gauge_server.py` / `start_phone_reader.bat` | iPhone の Safari に緑の円を出し、「読む」で3枚撮って PC が読む | `.\start_phone_reader.bat` → PC 画面の QR を iPhone で読む |
| `run_measure_phone.bat` | 疑似針を 0.2 → 0.5 → 0.8 に動かしながら自動で読み続け、誤差を記録 | `.\run_measure_phone.bat` |
| `summarize_phone.py` | 姿勢・値ごとの誤差を集計 | `python summarize_phone.py --since "2026-11-22 07:00"` |

証明書は iphone_capture のものを使います。詳しくは 06_phone_reader/README.md。

## 07_dial_auto 学習なしで読む(試作)

| ファイル | 用途 | 使い方 |
|---|---|---|
| `dial_auto.py` | 目盛りの並びと数字(EasyOCR)から目盛りの範囲を求め、針の角度で値を出す | `python dial_auto.py 写真.jpg …` |

## そのほかのフォルダ

| フォルダ | 内容 |
|---|---|
| `docs/` | DEVLOG、このガイド、ALIGNMENT、ブログ記事(blog = 記事(1)、blog2 = 機械学習編、blog3 = iPhone編)、マーカーリングの PDF |
| `dataset/` | 03 の測定結果(画像は GitHub に上げない) |
| `cad/` | マーカーリング・ひさしの 3D データ |
| `iPhone/` | iPhone の画面の記録画像 |
| `_old/` | 使わなくなったコードと旧モデル(2026-11-25 に移動)。GitHub には上げない |
| `_tmp/` | 作業用の一時ファイル(GitHub に上げない) |

### _old に移したもの(2026-11-25)
- 05_ml: 64x64 切り抜きの CNN(v1〜v5、regcrop)の `train.py` `train2.py` `train3.py` `train4.py` `aug4.py` `train_regcrop.py` `prep.py` `prep_eval.py` `reg.py` `export.py` `export_ens.py` `test_nn.cpp` と、そのモデル 7 個。極座標CNN に置き換え済み
- 02_xiao_reader: 上のモデル用の `nn_gauge.h` `nn_weights.h`(ファームは読み込んでいない)、補正を入れる前の `main.cpp` `nn_polar.h`
- 05_ml/results: 補正方式を試したときの試験用スクリプトと画像
- 05_ml/README.md の v1〜v5 の節は経緯として残しています
