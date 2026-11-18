# 04_pi_server  読み取り値の受信・保存・トレンド表示(Flask)

```
XIAO(定期読み取り) --POST /api/reading--> app.py --> data/gauge.db(SQLite)
needle_wander.py(疑似針の指令値) --POST /api/truth--> app.py
ブラウザ http://<PCのIP>:5000/  … トレンドグラフ(青=読み取り値、灰破線=指令値、赤い点=未取得)
```

## 起動(PC)
1. `run_server.bat` をダブルクリック(初回は flask を自動インストール)
   - XIAO(http://192.168.11.5)に送信先(このPCのIP)を設定し、5秒ごとの定期読み取りを開始する
   - 初回は Windows ファイアウォールの許可画面が出る → 「プライベートネットワーク」を許可(許可しないと XIAO から届かない)
2. `01_stepper_needle/run_wander.bat` をダブルクリック(疑似針を 0.5〜1.0 MPa でランダムな速さで往復)
3. ブラウザで http://127.0.0.1:5000/

## API
- `POST /api/reading` XIAO の読み取り結果(/read と同じJSON + seq, uptime_s, dev)。受信時刻で保存
- `POST /api/truth` `{"value": 0.73}` 指令値(比較用)
- `GET /api/data?minutes=30` グラフ用データと統計
- `GET /api/export.csv?minutes=1440` CSV

## XIAO 側
- `GET /log` 定期読み取りの状態(送信成功数・失敗数・最後のHTTPコード・最後の結果)
- `GET /log/set?on=1&intv=5&n=3&url=http://<サーバー>:5000/api/reading`(NVS保存、再起動後も継続)
- 停止: `/log/set?on=0`

## ラズパイへの移行
同じフォルダをコピーし `python3 app.py --xiao http://192.168.11.5` で起動(Chart.js は static/ に同梱、インターネット不要)。
