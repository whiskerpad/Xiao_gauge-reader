// =====================================================================
// XIAO ESP32S3 Sense 単体ゲージ読み取りファーム
//
// HTTP API
//   GET /             設定・確認画面
//   GET /capture.jpg  最新フレーム(JPEG)
//   GET /last.jpg     直近の /read で解析したフレーム(JPEG)
//   GET /read?n=3     n枚撮影して解析、指示値の中央値をJSONで返す
//   GET /config       設定値(JSON)
//   GET /config/set?cx=&cy=&ri=&ro=&a0=&a1=&vmin=&vmax=   設定変更(NVSへ保存)
//   GET /profile      直近の角度プロファイル・背景・暗さ(JSON、検証・記事用)
//   GET /bg/reset     背景の記録を開始(内容を消去)
//   GET /bg/add?n=2&skipv=0.35&skipw=15
//                     現在の画像を背景の平均に加える。skipv=針の現在値(目盛り値)、その±skipw度は除外
//   GET /bg/save      背景を確定してNVSへ保存(以後 /read で背景差分を使用)
//   GET /bg/clear     背景を削除(背景なしの方式に戻る)
//   GET /bg           背景の状態と値(JSON)
//   GET /cam          露出設定(JSON)
//   GET /cam/set?auto=0&aec=300&gain=0   露出設定(auto=1:自動 / 0:固定。aec=露光量, gain=ゲイン)。NVSへ保存
//   ※ 中心・半径・目盛り角・露出を変更すると背景は自動で削除される
//   マーカーモード: /config/set?mk=1&mkr=52.5&mu0=&mv0=…&mu3=&mv3=&pcx=&pcy=&ma0=&ma1=&m1i=&m1o=&m2i=&m2o=&m3i=&m3o=&mkw=20
//              mu/mv はマーカー初期位置[px](上・右・下・左)。帯 m*i/m*o と回転中心 pcx/pcy は文字盤平面[mm]
//   GET /mk           マーカーの検出状態・射影変換(JSON)
//   GET /mk/detect    新しい画像でマーカーを検出(画像は /last.jpg で確認)
//   GET /mk/find      画像全体からマーカー4点を自動探索し、初期位置を更新・保存(起動時と、定期読み取りで3回続けて
//                     マーカーが見つからないとき(20秒に1回まで)にも自動で実行)
//   リング方式(マーカーモード時): /config/set?rgon=1&rg0=18&rg1=38&rgq=0.3
//   置針の除外: /config/set?fpon=1&fpv=0.80&fpw=3
//   視差補正: /config/set?nd=8&camf=460(nd=針の奥行き[mm]、0=補正なし)
//   GET /hom          射影変換の補正の状態(JSON)。/hom?m=m0,...,m8 で設定(NVSへ保存)、/hom?clear=1 で解除
//                     m は 学習時の見え方の座標 → 今の画像の座標 の3x3行列。05_ml/hom_check.py --upload が送る
//                     設定中は /nn の前に画像を学習時の見え方に変換する(カメラ位置が学習時と違っても読める)
//   GET /log          定期読み取りの状態(JSON)
//   GET /log/set?on=1&intv=5&n=3&url=http://<サーバー>:5000/api/reading
//                     定期読み取り: intv秒ごとにn枚読み取り、結果JSONを url へPOST(NVSへ保存、再起動後も継続)
//   多帯モード: /config/set?nb=3&b1i=50&b1o=70&b2i=75&b2o=95&b3i=100&b3o=112&hpw=8
//              nb>=2 で有効(背景は使わない)。nb=0 で従来の単一帯(ri〜ro、背景差分あり)に戻る
// mDNS: http://gauge-xiao.local/
// =====================================================================
#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "camera_pins.h"
#include "secrets.h"
#define MK_ALLOC(n) ps_malloc(n)   // マーカー自動探索の作業領域はPSRAMに取る
#include "gauge.h"
#include "webpage.h"
#include "nn_polar.h"   // 機械学習(極座標CNN)での読み取り。重み・基準は nn_polar_data.h(05_ml で生成)

static const framesize_t FRAME_SIZE = FRAMESIZE_VGA;  // 640x480 グレースケール
static const int  JPEG_QUALITY = 80;                  // fmt2jpg の品質(0-100)
static const int  DISCARD_FRAMES = 2;    // 針移動直後の古いフレームを捨てる枚数
static const bool CAM_HMIRROR = true;    // 左右反転して表示される場合は切り替える
static const bool CAM_VFLIP   = false;   // 上下反転して表示される場合は切り替える
static const int  READ_MAX_N = 9;
static const int  PIN_LED = 21;      // XIAO ESP32S3 ユーザーLED(橙)。LOWで点灯
inline void led(bool on) { digitalWrite(PIN_LED, on ? LOW : HIGH); }

// 露出設定
struct CamSet { bool autoExp = true; int aec = 300; int gain = 0; };
CamSet camSet;

WebServer   server(80);
Preferences prefs;
GaugeConfig cfg;
GaugeResult lastRes;
uint8_t*    lastFrame = nullptr;
int         frameW = 0, frameH = 0;

// ---- 定期読み取り・送信(2026-11-16) ----
//   lgOn=1 のとき、lgIntv 秒ごとに lgN 枚を読み取り(/read と同じ処理)、結果のJSONを srvUrl へ POST する。
//   送信先(Flaskサーバー)が止まっていても読み取りは続ける(結果は /log で確認できる)。
struct LogSet { int on = 0; int intv = 5; int n = 3; String url = ""; int nn = 0; };   // nn=1: 定期読み取りをCNNで行う
LogSet lg;
bool homOn = false;                         // 射影変換の補正を使う(/hom、NVS: homm)
float homM[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // 学習時の見え方の座標 → 今の画像の座標
uint8_t* homBuf = nullptr;                  // 変換後の画像(PSRAM、640x480)
float nnConfMin = 0.8f;   // CNN: 1枚の「自信」がこれ未満なら使わない(知らない照明など)。/nn?cf=0.7&save=1 で変更(NVS: nncf)
uint32_t lgSeq = 0, lgSent = 0, lgFail = 0, lgOkN = 0, lgNgN = 0;
int      lgLastCode = 0;
String   lgLastJson = "{}";
// 設定画面を操作している間は定期読み取りを止める(2026-11-17)
//   読み取り中(約3秒)は Web サーバーが応答できず、画面の操作・保存が遅れたり失敗したりするため
static const uint32_t UI_PAUSE_MS = 60000;
uint32_t lastUiMs = 0;
bool uiTouched = false;
inline void uiTouch() { lastUiMs = millis(); uiTouched = true; }
inline bool uiActive() { return uiTouched && millis() - lastUiMs < UI_PAUSE_MS; }
// マーカー自動探索の状態(本体は markerAutoFindNow)
String   mkFindInfo = "";
uint32_t mkFindAt = 0;        // 最後に自動探索した時刻 [ms]
int      mkFailRun = 0;       // 定期読み取りでマーカー検出に続けて失敗した回数
bool markerAutoFindNow();

// ---------------------------------------------------------------------
String camDiag;   // カメラ診断結果(失敗時に繰り返し表示)

bool initCamera() {
  // 診断表示: どの段階で失敗したかをシリアルに出す
  camDiag = "PSRAM found=" + String(psramFound()) + " size=" + String(ESP.getPsramSize());

  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 10000000;   // 接触不良対策で20MHz→10MHz
  c.pixel_format = PIXFORMAT_GRAYSCALE;
  c.frame_size   = FRAME_SIZE;
  c.fb_location  = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.fb_count     = psramFound() ? 2 : 1;
  c.grab_mode    = CAMERA_GRAB_LATEST;
  c.jpeg_quality = 12;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    camDiag += " | esp_camera_init failed: 0x" + String(err, HEX) + " " + esp_err_to_name(err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    camDiag += " | sensor PID=0x" + String(s->id.PID, HEX);  // 26=OV2640 3660=OV3660 5640=OV5640
    s->set_hmirror(s, CAM_HMIRROR ? 1 : 0);
    s->set_vflip(s, CAM_VFLIP ? 1 : 0);
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { camDiag += " | first frame capture failed"; return false; }
  frameW = fb->width; frameH = fb->height;
  camDiag += " | frame " + String(frameW) + "x" + String(frameH);
  esp_camera_fb_return(fb);

  lastFrame = (uint8_t*)ps_malloc(frameW * frameH);
  if (!lastFrame) lastFrame = (uint8_t*)malloc(frameW * frameH);
  if (!lastFrame) { camDiag += " | buffer alloc failed"; return false; }
  return true;
}

// センサー別の設定範囲(esp32-camera ドライバ準拠)
uint16_t sensorPID() { sensor_t* s = esp_camera_sensor_get(); return s ? s->id.PID : 0; }
int aecMax()  { return sensorPID() == 0x3660 || sensorPID() == 0x5640 ? 1536 : 1200; }
int gainMax() { return sensorPID() == 0x3660 || sensorPID() == 0x5640 ? 64 : 30; }
const char* sensorName() {
  switch (sensorPID()) { case 0x26: return "OV2640"; case 0x3660: return "OV3660"; case 0x5640: return "OV5640"; default: return "unknown"; }
}

void applyCam() {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  s->set_exposure_ctrl(s, camSet.autoExp ? 1 : 0);   // 自動露出
  s->set_gain_ctrl(s, camSet.autoExp ? 1 : 0);       // 自動ゲイン
  if (!camSet.autoExp) {
    s->set_aec_value(s, camSet.aec);    // 範囲はセンサーにより異なる(aecMax()/gainMax())
    s->set_agc_gain(s, camSet.gain);
  }
}

void loadCam() {
  prefs.begin("gauge", true);
  camSet.autoExp = prefs.getBool("ae", true);
  camSet.aec     = prefs.getInt("aec", 300);
  camSet.gain    = prefs.getInt("agc", 0);
  prefs.end();
}

void saveCam() {
  prefs.begin("gauge", false);
  prefs.putBool("ae", camSet.autoExp);
  prefs.putInt("aec", camSet.aec);
  prefs.putInt("agc", camSet.gain);
  prefs.end();
}

String camJson() {
  char b[160];
  snprintf(b, sizeof(b), "{\"auto\":%d,\"aec\":%d,\"gain\":%d,\"aec_max\":%d,\"gain_max\":%d,\"sensor\":\"%s\"}",
           camSet.autoExp ? 1 : 0, camSet.aec, camSet.gain, aecMax(), gainMax(), sensorName());
  return String(b);
}

camera_fb_t* grabFresh() {
  for (int i = 0; i < DISCARD_FRAMES; i++) {
    camera_fb_t* d = esp_camera_fb_get();
    if (d) esp_camera_fb_return(d);
  }
  return esp_camera_fb_get();
}

// ---------------------------------------------------------------------
String geomSig() {   // 背景が有効な設定の組み合わせ
  char b[96];
  snprintf(b, sizeof(b), "%.1f,%.1f,%.1f,%.1f,%.1f,%.1f",
           cfg.cx, cfg.cy, cfg.rIn, cfg.rOut, cfg.a0, cfg.a1);
  return String(b);
}

void saveBg() {
  prefs.begin("gauge", false);
  prefs.putBytes("bg", bgProf, sizeof(float) * bgN);
  prefs.putInt("bgn", bgN);
  prefs.putInt("bgcnt", bgCount);
  prefs.putString("bgsig", geomSig());
  prefs.end();
}

void clearBgNvs() {
  prefs.begin("gauge", false);
  prefs.remove("bg"); prefs.remove("bgn"); prefs.remove("bgcnt"); prefs.remove("bgsig");
  prefs.end();
}

void loadBg() {
  bgReset();
  prefs.begin("gauge", true);
  int n = prefs.getInt("bgn", 0);
  String sig = prefs.getString("bgsig", "");
  if (n > 0 && n <= PROF_MAX && sig == geomSig() &&
      prefs.getBytes("bg", bgProf, sizeof(float) * n) == sizeof(float) * n) {
    bgN = n;
    bgCount = prefs.getInt("bgcnt", 0);
    bgValid = true;
  }
  prefs.end();
}

void loadConfig() {
  prefs.begin("gauge", true);
  cfg.cx   = prefs.getFloat("cx",   frameW / 2.0f);
  cfg.cy   = prefs.getFloat("cy",   frameH / 2.0f);
  cfg.rIn  = prefs.getFloat("ri",   cfg.rIn);
  cfg.rOut = prefs.getFloat("ro",   cfg.rOut);
  cfg.a0   = prefs.getFloat("a0",   cfg.a0);
  cfg.a1   = prefs.getFloat("a1",   cfg.a1);
  cfg.vMin = prefs.getFloat("vmin", cfg.vMin);
  cfg.vMax = prefs.getFloat("vmax", cfg.vMax);
  cfg.corrA = prefs.getFloat("ca", 1.0f);
  cfg.corrB = prefs.getFloat("cb", 0.0f);
  cfg.nb    = prefs.getInt("nb", 0);
  cfg.mk    = prefs.getInt("mk", 0);
  cfg.mkr   = prefs.getFloat("mkr", cfg.mkr);
  cfg.mkWin = prefs.getInt("mkw", cfg.mkWin);
  cfg.mkPol = prefs.getInt("mkp", cfg.mkPol);
  cfg.pcx   = prefs.getFloat("pcx", cfg.pcx);
  cfg.pcy   = prefs.getFloat("pcy", cfg.pcy);
  cfg.ma0   = prefs.getFloat("ma0", cfg.ma0);
  cfg.ma1   = prefs.getFloat("ma1", cfg.ma1);
  {
    char k[8];
    for (int i = 0; i < 4; i++) {
      snprintf(k, sizeof(k), "mu%d", i); cfg.mu[i] = prefs.getFloat(k, cfg.mu[i]);
      snprintf(k, sizeof(k), "mv%d", i); cfg.mv[i] = prefs.getFloat(k, cfg.mv[i]);
    }
    const char* K[3][2] = {{"m1i", "m1o"}, {"m2i", "m2o"}, {"m3i", "m3o"}};
    for (int b = 0; b < 3; b++) for (int j = 0; j < 2; j++) cfg.mbr[b][j] = prefs.getFloat(K[b][j], cfg.mbr[b][j]);
  }
  cfg.hpw   = prefs.getFloat("hpw", cfg.hpw);
  cfg.rgOn  = prefs.getInt("rgon", cfg.rgOn);
  cfg.rg0   = prefs.getFloat("rg0", cfg.rg0);
  cfg.rg1   = prefs.getFloat("rg1", cfg.rg1);
  cfg.rgq   = prefs.getFloat("rgq", cfg.rgq);
  cfg.fpOn  = prefs.getInt("fpon", cfg.fpOn);
  cfg.fpv   = prefs.getFloat("fpv", cfg.fpv);
  cfg.fpw   = prefs.getFloat("fpw", cfg.fpw);
  cfg.nd    = prefs.getFloat("nd", cfg.nd);
  cfg.camf  = prefs.getFloat("camf", cfg.camf);
  {
    const char* K[3][2] = {{"b1i", "b1o"}, {"b2i", "b2o"}, {"b3i", "b3o"}};
    for (int b = 0; b < 3; b++) for (int j = 0; j < 2; j++) cfg.br[b][j] = prefs.getFloat(K[b][j], cfg.br[b][j]);
  }
  lg.on   = prefs.getInt("lgon", 0);
  lg.intv = prefs.getInt("lgi", 5);
  lg.n    = prefs.getInt("lgn", 3);
  lg.url  = prefs.getString("lgu", "");
  lg.nn   = prefs.getInt("lgnn", 0);
  nnConfMin = prefs.getFloat("nncf", 0.8f);
  homOn = prefs.isKey("homm") && prefs.getBytes("homm", homM, sizeof(homM)) == sizeof(homM);
  prefs.end();
}

void saveConfig() {
  prefs.begin("gauge", false);
  prefs.putFloat("cx", cfg.cx);   prefs.putFloat("cy", cfg.cy);
  prefs.putFloat("ri", cfg.rIn);  prefs.putFloat("ro", cfg.rOut);
  prefs.putFloat("a0", cfg.a0);   prefs.putFloat("a1", cfg.a1);
  prefs.putFloat("vmin", cfg.vMin); prefs.putFloat("vmax", cfg.vMax);
  prefs.putFloat("ca", cfg.corrA);  prefs.putFloat("cb", cfg.corrB);
  prefs.putInt("nb", cfg.nb);       prefs.putFloat("hpw", cfg.hpw);
  prefs.putInt("mk", cfg.mk);       prefs.putFloat("mkr", cfg.mkr);  prefs.putInt("mkw", cfg.mkWin);
  prefs.putInt("mkp", cfg.mkPol);
  prefs.putFloat("pcx", cfg.pcx);   prefs.putFloat("pcy", cfg.pcy);
  prefs.putFloat("ma0", cfg.ma0);   prefs.putFloat("ma1", cfg.ma1);
  prefs.putInt("rgon", cfg.rgOn);   prefs.putFloat("rg0", cfg.rg0);  prefs.putFloat("rg1", cfg.rg1);
  prefs.putFloat("rgq", cfg.rgq);
  prefs.putInt("fpon", cfg.fpOn);   prefs.putFloat("fpv", cfg.fpv);  prefs.putFloat("fpw", cfg.fpw);
  prefs.putFloat("nd", cfg.nd);     prefs.putFloat("camf", cfg.camf);
  {
    char k[8];
    for (int i = 0; i < 4; i++) {
      snprintf(k, sizeof(k), "mu%d", i); prefs.putFloat(k, cfg.mu[i]);
      snprintf(k, sizeof(k), "mv%d", i); prefs.putFloat(k, cfg.mv[i]);
    }
    const char* K[3][2] = {{"m1i", "m1o"}, {"m2i", "m2o"}, {"m3i", "m3o"}};
    for (int b = 0; b < 3; b++) for (int j = 0; j < 2; j++) prefs.putFloat(K[b][j], cfg.mbr[b][j]);
  }
  {
    const char* K[3][2] = {{"b1i", "b1o"}, {"b2i", "b2o"}, {"b3i", "b3o"}};
    for (int b = 0; b < 3; b++) for (int j = 0; j < 2; j++) prefs.putFloat(K[b][j], cfg.br[b][j]);
  }
  prefs.end();
}

String configJson() {
  char b[700];
  snprintf(b, sizeof(b),
    "{\"cx\":%.1f,\"cy\":%.1f,\"ri\":%.1f,\"ro\":%.1f,\"a0\":%.1f,\"a1\":%.1f,"
    "\"vmin\":%.4f,\"vmax\":%.4f,\"span\":%.1f,\"w\":%d,\"h\":%d,"
    "\"bg_valid\":%s,\"bg_count\":%d,\"cam_auto\":%d,\"cam_aec\":%d,\"cam_gain\":%d,\"ca\":%.5f,\"cb\":%.5f,\"nb\":%d,\"hpw\":%.1f,"
    "\"b1i\":%.1f,\"b1o\":%.1f,\"b2i\":%.1f,\"b2o\":%.1f,\"b3i\":%.1f,\"b3o\":%.1f}",
    cfg.cx, cfg.cy, cfg.rIn, cfg.rOut, cfg.a0, cfg.a1, cfg.vMin, cfg.vMax,
    gaugeSpan(cfg), frameW, frameH, bgValid ? "true" : "false", bgCount,
    camSet.autoExp ? 1 : 0, camSet.aec, camSet.gain, cfg.corrA, cfg.corrB, cfg.nb, cfg.hpw,
    cfg.br[0][0], cfg.br[0][1], cfg.br[1][0], cfg.br[1][1], cfg.br[2][0], cfg.br[2][1]);
  String js(b);
  js.remove(js.length() - 1);
  char m[640];
  snprintf(m, sizeof(m),
    ",\"mk\":%d,\"mkr\":%.2f,\"mkw\":%d,\"mkp\":%d,\"pcx\":%.2f,\"pcy\":%.2f,\"ma0\":%.2f,\"ma1\":%.2f,"
    "\"mu0\":%.1f,\"mv0\":%.1f,\"mu1\":%.1f,\"mv1\":%.1f,\"mu2\":%.1f,\"mv2\":%.1f,\"mu3\":%.1f,\"mv3\":%.1f,"
    "\"m1i\":%.1f,\"m1o\":%.1f,\"m2i\":%.1f,\"m2o\":%.1f,\"m3i\":%.1f,\"m3o\":%.1f}",
    cfg.mk, cfg.mkr, cfg.mkWin, cfg.mkPol, cfg.pcx, cfg.pcy, cfg.ma0, cfg.ma1,
    cfg.mu[0], cfg.mv[0], cfg.mu[1], cfg.mv[1], cfg.mu[2], cfg.mv[2], cfg.mu[3], cfg.mv[3],
    cfg.mbr[0][0], cfg.mbr[0][1], cfg.mbr[1][0], cfg.mbr[1][1], cfg.mbr[2][0], cfg.mbr[2][1]);
  js += m;
  js.remove(js.length() - 1);
  char g[200];
  snprintf(g, sizeof(g),
    ",\"rgon\":%d,\"rg0\":%.1f,\"rg1\":%.1f,\"rgq\":%.2f,\"fpon\":%d,\"fpv\":%.3f,\"fpw\":%.1f,\"fpl\":%.1f,\"nd\":%.1f,\"camf\":%.0f}",
    cfg.rgOn, cfg.rg0, cfg.rg1, cfg.rgq, cfg.fpOn, cfg.fpv, cfg.fpw, fpLearn, cfg.nd, cfg.camf);
  js += g;
  return js;
}

// ---------------------------------------------------------------------
void sendGrayJpeg(const uint8_t* buf, int w, int h) {
  uint8_t* out = nullptr; size_t outLen = 0;
  if (!fmt2jpg((uint8_t*)buf, (size_t)w * h, w, h, PIXFORMAT_GRAYSCALE, JPEG_QUALITY, &out, &outLen)) {
    server.send(500, "text/plain", "jpeg encode failed");
    return;
  }
  server.sendHeader("Cache-Control", "no-store");
  server.send_P(200, "image/jpeg", (const char*)out, outLen);
  free(out);
}

void handleCapture() {
  uiTouch();
  camera_fb_t* fb = grabFresh();
  if (!fb) { server.send(500, "text/plain", "capture failed"); return; }
  sendGrayJpeg(fb->buf, fb->width, fb->height);
  esp_camera_fb_return(fb);
}

void handleLast() {
  uiTouch();
  sendGrayJpeg(lastFrame, frameW, frameH);
}

// n 枚を読み取り、結果のJSONを返す(/read と定期読み取りで共用)
String doRead(int n, bool& okOut) {
  n = constrain(n, 1, READ_MAX_N);
  float vals[READ_MAX_N], raws[READ_MAX_N]; int okN = 0;
  float snrSum = 0;
  float wsum = 0, csum = 0, satMax = 0; uint32_t usMax = 0;
  uint32_t t0 = millis();

  for (int i = 0; i < n; i++) {
    camera_fb_t* fb = grabFresh();
    if (!fb) continue;
    GaugeResult r;
    analyzeGauge(fb->buf, fb->width, fb->height, cfg, r);
    memcpy(lastFrame, fb->buf, (size_t)frameW * frameH);
    esp_camera_fb_return(fb);
    lastRes = r;
    float sat = profSaturation(); if (sat > satMax) satMax = sat;
    if (r.us > usMax) usMax = r.us;
    if (r.ok) { raws[okN] = r.raw; vals[okN++] = r.value; wsum += r.width; csum += r.contrast; snrSum += r.snr; }
  }

  char b[420];
  okOut = okN > 0;
  if (okN == 0) {
    snprintf(b, sizeof(b), "{\"ok\":false,\"mk\":%d,\"mk_ok\":%d,\"multi\":%s,\"bg\":%s,\"n\":%d,\"contrast\":%.3f,\"snr\":%.2f,\"ms\":%lu}",
             lastRes.mkUsed ? 1 : 0, lastRes.mkOk ? 1 : 0, lastRes.multi ? "true" : "false", lastRes.bg ? "true" : "false", n, lastRes.contrast, lastRes.snr, (unsigned long)(millis() - t0));
  } else {
    std::sort(vals, vals + okN);
    std::sort(raws, raws + okN);
    float v = vals[okN / 2];
    float rawMed = raws[okN / 2];
    float span = gaugeSpan(cfg);
    float rel  = (rawMed - cfg.vMin) / (cfg.vMax - cfg.vMin) * span;
    float ang  = fmodf(gaugeA0(cfg) + rel + 720.0f, 360.0f);
    snprintf(b, sizeof(b),
      "{\"ok\":true,\"mk\":%d,\"mk_ok\":1,\"multi\":%s,\"bg\":%s,\"value\":%.4f,\"raw\":%.4f,\"snr\":%.2f,\"angle\":%.2f,\"rel\":%.2f,\"v_lo\":%.4f,\"v_hi\":%.4f,"
      "\"width\":%.1f,\"contrast\":%.3f,\"gain\":%.3f,\"sat\":%.3f,\"n\":%d,\"okn\":%d,\"analyze_us\":%lu,\"ms\":%lu}",
      lastRes.mkUsed ? 1 : 0, lastRes.multi ? "true" : "false", lastRes.bg ? "true" : "false", v, rawMed, snrSum / okN, ang, rel, vals[0], vals[okN - 1], wsum / okN, csum / okN, lastRes.gain, satMax, n, okN,
      (unsigned long)usMax, (unsigned long)(millis() - t0));
  }
  return String(b);
}

void handleRead() {
  uiTouch();
  int n = server.hasArg("n") ? server.arg("n").toInt() : 3;
  bool ok;
  String js = doRead(n, ok);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", js);
}

// ---- 機械学習(極座標CNN)での読み取り ----
PolarWork* polW = nullptr;                  // 作業領域(PSRAM、約2.7MB)
int polSinceFull = 0;                       // 位置合わせの全探索からの読み取り回数(30回ごとに全探索)
static const float NN_SPREAD_MAX = 0.03f;   // n枚の推定値の幅がこれを超えたら未取得(手・影などで不安定)
static const float NN_MOVE_STEP = 0.15f;    // 「針が動いている」とみなす1枚ごとの変化の上限(これより大きい飛びは不安定とみなす)
static const float NN_MOVE_BACK = 0.01f;    // 同じ向きの判定で許す逆向きの揺れ
static void* psAlloc(size_t n) { void* p = ps_malloc(n); return p ? p : malloc(n); }

// n 枚撮って 位置合わせ→極座標→CNN で推定し、自信が cf 以上の画像の中央値を返す
// 幅が NN_SPREAD_MAX を超えても、自信のある画像が2枚以上・撮った順に同じ向きへ少しずつ変わっていれば
// 「針が動いている」とみなし、最後の1枚の値を返す(mv=1)。allowMove=false で無効
String doNn(int n, bool& okOut, float cf = -1, bool allowMove = true) {
  if (cf <= 0) cf = nnConfMin;
  n = constrain(n, 1, READ_MAX_N);
  float vals[READ_MAX_N]; float seqv[READ_MAX_N]; int okN = 0, rejN = 0; uint32_t usMax = 0; float cMin = 1;
  int dxL = 0, dyL = 0; float nccL = 0;
  String fv = "[";   // 各画像の [値, 自信, dx, dy](閾値をPCで後から検討するため)
  uint32_t t0 = millis();
  if (!polW) { polW = new PolarWork(); if (!polarInit(*polW, psAlloc)) { delete polW; polW = nullptr; } }
  bool full = polW && (!polW->haveShift || polSinceFull >= 30);   // 普段は前回の位置の近くだけ探す(速い)
  if (full) polSinceFull = 0; else polSinceFull++;
  bool fullUsed = full;
  for (int i = 0; i < n && polW; i++) {
    camera_fb_t* fb = grabFresh();
    if (!fb) continue;
    uint32_t u = micros();
    const uint8_t* src = fb->buf;
    if (homOn) {
      if (!homBuf) homBuf = (uint8_t*)psAlloc((size_t)fb->width * fb->height);
      if (homBuf) { polarWarp(fb->buf, fb->width, fb->height, homM, homBuf); src = homBuf; }
    }
    PolarResult r = polarRead(*polW, src, fb->width, fb->height, full && i == 0);
    u = micros() - u; if (u > usMax) usMax = u;
    memcpy(lastFrame, fb->buf, (size_t)frameW * frameH);
    esp_camera_fb_return(fb);
    if (fv.length() > 1) fv += ",";
    fv += "["; fv += String(r.value, 4); fv += ","; fv += String(r.conf, 3); fv += ","; fv += r.dx; fv += ","; fv += r.dy; fv += "]";
    dxL = r.dx; dyL = r.dy; nccL = r.ncc;
    if (isnan(r.value)) continue;
    if (r.conf < cMin) cMin = r.conf;
    if (r.conf < cf) { rejN++; continue; }    // 自信が低い → 使わない
    seqv[okN] = r.value;                       // 撮った順
    vals[okN++] = r.value;
  }
  char b[420];
  float v = 0, lo = 0, hi = 0;
  if (okN > 0) { std::sort(vals, vals + okN); v = vals[okN / 2]; lo = vals[0]; hi = vals[okN - 1]; }
  bool mv = false;
  if (allowMove && okN >= 2 && (hi - lo) > NN_SPREAD_MAX) {
    bool up = true, down = true, small = true;
    for (int i = 1; i < okN; i++) {
      float d = seqv[i] - seqv[i - 1];
      if (d < -NN_MOVE_BACK) up = false;
      if (d > NN_MOVE_BACK) down = false;
      if (fabsf(d) > NN_MOVE_STEP) small = false;
    }
    float tot = seqv[okN - 1] - seqv[0];
    if (small && ((up && tot > 0) || (down && tot < 0))) { mv = true; v = seqv[okN - 1]; }
  }
  bool ok = okN > 0 && ((hi - lo) <= NN_SPREAD_MAX || mv) && v > -0.05f && v < 1.05f;
  okOut = ok;
  if (ok) v = constrain(v, 0.0f, 1.0f);
  snprintf(b, sizeof(b),
    "{\"ok\":%s,\"nn\":2,\"mv\":%d,\"value\":%.4f,\"raw\":%.4f,\"v_lo\":%.4f,\"v_hi\":%.4f,\"conf_min\":%.3f,\"cf\":%.2f,\"rej\":%d,\"n\":%d,\"okn\":%d,"
    "\"dx\":%d,\"dy\":%d,\"ncc\":%.3f,\"full\":%d,\"hom\":%d,\"infer_us\":%lu,\"ms\":%lu,\"fv\":",
    ok ? "true" : "false", mv ? 1 : 0, v, okN ? vals[okN / 2] : 0.0f, lo, hi, cMin, cf, rejN, n, okN, dxL, dyL, nccL, fullUsed ? 1 : 0, homOn ? 1 : 0, (unsigned long)usMax, (unsigned long)(millis() - t0));
  String js(b);
  js += fv; js += "]}";
  return js;
}

// /nn?n=3&cf=0.8  (cf: 自信の下限。&save=1 で保存し定期読み取りにも使う。mv=0: 動いている判定を使わない)
void handleNn() {
  uiTouch();
  int n  = server.hasArg("n")  ? server.arg("n").toInt()  : 3;
  float cf = server.hasArg("cf") ? server.arg("cf").toFloat() : -1;
  if (cf > 0 && server.hasArg("save")) {
    nnConfMin = constrain(cf, 0.0f, 1.0f);
    prefs.begin("gauge", false); prefs.putFloat("nncf", nnConfMin); prefs.end();
  }
  bool mvOn = !server.hasArg("mv") || server.arg("mv").toInt() != 0;   // /nn?mv=0 で「動いている」判定なし
  bool ok;
  String js = doNn(n, ok, cf, mvOn);
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", js);
}

// /hom: 射影変換の補正。?m=m0,...,m8 で設定、?clear=1 で解除、引数なしで状態
String homJson() {
  String js = "{\"on\":"; js += homOn ? "true" : "false"; js += ",\"m\":[";
  for (int i = 0; i < 9; i++) { if (i) js += ","; js += String(homM[i], 8); }
  js += "]}"; return js;
}
void handleHom() {
  uiTouch();
  if (server.hasArg("clear")) {
    homOn = false;
    prefs.begin("gauge", false); prefs.remove("homm"); prefs.end();
    if (polW) polW->haveShift = false;
  } else if (server.hasArg("m")) {
    String a = server.arg("m"); float m[9]; int k = 0, p = 0;
    while (k < 9 && p <= (int)a.length()) {
      int q = a.indexOf(',', p); if (q < 0) q = a.length();
      m[k++] = a.substring(p, q).toFloat(); p = q + 1;
    }
    if (k != 9 || fabsf(m[8]) < 1e-9f) { server.send(400, "application/json", "{\"error\":\"m needs 9 numbers\"}"); return; }
    for (int i = 0; i < 9; i++) homM[i] = m[i];
    homOn = true;
    prefs.begin("gauge", false); prefs.putBytes("homm", homM, sizeof(homM)); prefs.end();
    if (polW) polW->haveShift = false;            // 次の読み取りで位置合わせを全探索し直す
  }
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", homJson());
}

// 定期読み取り1回分: 読み取り → 通し番号等を付けて POST
void logOnce() {
  bool ok;
  String js = lg.nn ? doNn(lg.n, ok) : doRead(lg.n, ok);
  lgSeq++;
  if (ok) lgOkN++; else lgNgN++;
  // マーカーが続けて見つからない(カメラが動いた等)→ 20秒に1回まで自動探索
  if (!lg.nn && cfg.mk && lastRes.mkUsed && !lastRes.mkOk) mkFailRun++; else mkFailRun = 0;   // CNN読み取り中はマーカーを見ない
  if (mkFailRun >= 3 && millis() - mkFindAt > 20000UL) { if (markerAutoFindNow()) mkFailRun = 0; }   // 20秒に1回まで
  js.remove(js.length() - 1);
  js += ",\"seq\":"; js += String(lgSeq);
  js += ",\"uptime_s\":"; js += String(millis() / 1000);
  js += ",\"dev\":\"gauge-xiao\"}";
  lgLastJson = js;
  if (lg.url.length() == 0 || WiFi.status() != WL_CONNECTED) { lgLastCode = -1; lgFail++; return; }
  HTTPClient http;
  http.setConnectTimeout(1500);
  http.setTimeout(2000);
  if (!http.begin(lg.url)) { lgLastCode = -2; lgFail++; return; }
  http.addHeader("Content-Type", "application/json");
  lgLastCode = http.POST(js);
  http.end();
  if (lgLastCode >= 200 && lgLastCode < 300) lgSent++; else lgFail++;
}

String logJson() {
  String s = "{\"on\":"; s += lg.on;
  s += ",\"intv\":"; s += lg.intv;
  s += ",\"paused_s\":"; s += uiActive() ? (int)((UI_PAUSE_MS - (millis() - lastUiMs)) / 1000) : 0;
  s += ",\"n\":"; s += lg.n;
  s += ",\"nn\":"; s += lg.nn;
  s += ",\"url\":\""; s += lg.url; s += "\"";
  s += ",\"seq\":"; s += lgSeq;
  s += ",\"ok\":"; s += lgOkN;
  s += ",\"ng\":"; s += lgNgN;
  s += ",\"sent\":"; s += lgSent;
  s += ",\"send_fail\":"; s += lgFail;
  s += ",\"last_code\":"; s += lgLastCode;
  s += ",\"last\":"; s += lgLastJson;
  s += "}";
  return s;
}

void handleLog() {
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", logJson());
}

// /log/set?on=1&intv=5&n=3&url=http://192.168.11.x:5000/api/reading
void handleLogSet() {
  if (server.hasArg("on"))   lg.on = server.arg("on").toInt() ? 1 : 0;
  if (server.hasArg("intv")) lg.intv = constrain(server.arg("intv").toInt(), 2, 3600);
  if (server.hasArg("n"))    lg.n = constrain(server.arg("n").toInt(), 1, READ_MAX_N);
  if (server.hasArg("url"))  lg.url = server.arg("url");
  if (server.hasArg("nn"))   lg.nn = server.arg("nn").toInt() ? 1 : 0;
  prefs.begin("gauge", false);
  prefs.putInt("lgnn", lg.nn);
  prefs.putInt("lgon", lg.on); prefs.putInt("lgi", lg.intv); prefs.putInt("lgn", lg.n); prefs.putString("lgu", lg.url);
  prefs.end();
  handleLog();
}

void handleRoot() { uiTouch(); server.send(200, "text/html", INDEX_HTML); }

void handleConfig() { uiTouch(); server.send(200, "application/json", configJson()); }

void handleConfigSet() {
  uiTouch();
  String before = geomSig();
  auto setf = [](const char* k, float& dst) { if (server.hasArg(k)) dst = server.arg(k).toFloat(); };
  setf("cx", cfg.cx); setf("cy", cfg.cy); setf("ri", cfg.rIn); setf("ro", cfg.rOut);
  setf("a0", cfg.a0); setf("a1", cfg.a1); setf("vmin", cfg.vMin); setf("vmax", cfg.vMax);
  setf("ca", cfg.corrA); setf("cb", cfg.corrB);
  setf("hpw", cfg.hpw);
  setf("b1i", cfg.br[0][0]); setf("b1o", cfg.br[0][1]);
  setf("b2i", cfg.br[1][0]); setf("b2o", cfg.br[1][1]);
  setf("b3i", cfg.br[2][0]); setf("b3o", cfg.br[2][1]);
  if (server.hasArg("nb")) cfg.nb = constrain(server.arg("nb").toInt(), 0, 3);
  if (server.hasArg("mk")) cfg.mk = server.arg("mk").toInt() ? 1 : 0;
  if (server.hasArg("mkw")) cfg.mkWin = constrain(server.arg("mkw").toInt(), 5, MK_WIN_MAX);
  if (server.hasArg("mkp")) { int np = server.arg("mkp").toInt() ? 1 : 0; if (np != cfg.mkPol) { cfg.mkPol = np; markerReset(cfg); } }
  setf("mkr", cfg.mkr); setf("pcx", cfg.pcx); setf("pcy", cfg.pcy);
  setf("ma0", cfg.ma0); setf("ma1", cfg.ma1);
  setf("m1i", cfg.mbr[0][0]); setf("m1o", cfg.mbr[0][1]);
  setf("m2i", cfg.mbr[1][0]); setf("m2o", cfg.mbr[1][1]);
  setf("m3i", cfg.mbr[2][0]); setf("m3o", cfg.mbr[2][1]);
  if (server.hasArg("rgon")) cfg.rgOn = server.arg("rgon").toInt() ? 1 : 0;
  setf("rg0", cfg.rg0); setf("rg1", cfg.rg1); setf("rgq", cfg.rgq);
  setf("nd", cfg.nd); setf("camf", cfg.camf);
  if (cfg.rg0 > cfg.rg1) std::swap(cfg.rg0, cfg.rg1);
  if (cfg.rg1 - cfg.rg0 > RG_MAX) cfg.rg0 = cfg.rg1 - RG_MAX;
  cfg.rgq = constrain(cfg.rgq, 0.0f, 0.9f);
  {
    float oldFpv = cfg.fpv, oldFpw = cfg.fpw; int oldOn = cfg.fpOn;
    if (server.hasArg("fpon")) cfg.fpOn = server.arg("fpon").toInt() ? 1 : 0;
    setf("fpv", cfg.fpv); setf("fpw", cfg.fpw);
    cfg.fpw = constrain(cfg.fpw, 0.5f, 15.0f);
    if (cfg.fpv != oldFpv || cfg.fpw != oldFpw || cfg.fpOn != oldOn) fpLearn = -1.0f;   // 置針の学習をやり直す
  }
  {
    bool mkMoved = false;
    char k[8];
    for (int i = 0; i < 4; i++) {
      snprintf(k, sizeof(k), "mu%d", i);
      if (server.hasArg(k)) { float nv = server.arg(k).toFloat(); if (fabsf(nv - cfg.mu[i]) > 0.05f) mkMoved = true; cfg.mu[i] = nv; }
      snprintf(k, sizeof(k), "mv%d", i);
      if (server.hasArg(k)) { float nv = server.arg(k).toFloat(); if (fabsf(nv - cfg.mv[i]) > 0.05f) mkMoved = true; cfg.mv[i] = nv; }
    }
    if (mkMoved || !mks.init) markerReset(cfg);   // 初期位置を変えたら追跡をやり直す
  }
  for (int b = 0; b < 3; b++) if (cfg.mbr[b][0] > cfg.mbr[b][1]) std::swap(cfg.mbr[b][0], cfg.mbr[b][1]);
  for (int b = 0; b < 3; b++) if (cfg.br[b][0] > cfg.br[b][1]) std::swap(cfg.br[b][0], cfg.br[b][1]);
  if (cfg.rIn > cfg.rOut) std::swap(cfg.rIn, cfg.rOut);
  saveConfig();
  if (geomSig() != before) { bgReset(); clearBgNvs(); }   // 形状が変わったら背景は無効
  handleConfig();
}

void appendArray(String& s, const char* key, const float* a, int n) {
  s += ",\""; s += key; s += "\":[";
  for (int i = 0; i < n; i++) { if (i) s += ','; s += String(a[i], 1); }
  s += "]";
}

void handleProfile() {
  uiTouch();
  String s;
  s.reserve(profN * 20 + 128);
  s += "{\"start\":"; s += String(profStart, 1);
  s += ",\"step\":";  s += String(PROF_STEP, 1);
  s += ",\"a0\":";    s += String(gaugeA0(cfg), 1);
  s += ",\"span\":";  s += String(gaugeSpan(cfg), 1);
  s += ",\"bg_used\":"; s += lastRes.bg ? "true" : "false";
  s += ",\"multi\":"; s += lastRes.multi ? "true" : "false";
  s += ",\"rel\":";   s += String(lastRes.rel, 2);
  s += ",\"gain\":";  s += String(lastRes.gain, 3);
  s += ",\"sat\":";   s += String(profSaturation(), 3);
  appendArray(s, "values", prof, profN);
  appendArray(s, "dark", dark, profN);
  if (bgN == profN && bgN > 0) appendArray(s, "bg", bgProf, bgN);
  s += "}";
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", s);
}

// ---- 背景 ----
String bgJson(bool withValues) {
  String s = "{\"valid\":"; s += bgValid ? "true" : "false";
  s += ",\"count\":"; s += String(bgCount);
  s += ",\"n\":";     s += String(bgN);
  if (withValues && bgN > 0) appendArray(s, "values", bgProf, bgN);
  s += "}";
  return s;
}

void handleBgReset() {
  uiTouch();
  bgReset();
  server.send(200, "application/json", bgJson(false));
}

void handleBgAdd() {
  uiTouch();
  int n = server.hasArg("n") ? constrain(server.arg("n").toInt(), 1, READ_MAX_N) : 2;
  float skipW = server.hasArg("skipw") ? server.arg("skipw").toFloat() : 15.0f;
  float skipRel = NAN;
  if (server.hasArg("skipv") && cfg.vMax != cfg.vMin) {
    float v = server.arg("skipv").toFloat();
    skipRel = (v - cfg.vMin) / (cfg.vMax - cfg.vMin) * gaugeSpan(cfg);
  }
  bgValid = false;   // 記録中は背景差分を使わない
  for (int i = 0; i < n; i++) {
    camera_fb_t* fb = grabFresh();
    if (!fb) continue;
    computeProfile(fb->buf, fb->width, fb->height, cfg);
    memcpy(lastFrame, fb->buf, (size_t)frameW * frameH);
    esp_camera_fb_return(fb);
    bgAddCurrent(skipRel, skipW);
  }
  String js = bgJson(false);
  js.remove(js.length() - 1);
  js += ",\"sat\":"; js += String(profSaturation(), 3);
  js += ",\"skip_rel\":"; js += isnan(skipRel) ? String("null") : String(skipRel, 1);
  js += "}";
  server.send(200, "application/json", js);
}

void handleBgSave() {
  uiTouch();
  if (bgCount < 2) { server.send(400, "application/json", "{\"error\":\"too few frames\"}"); return; }
  int minCnt = bgFinalize();
  if (minCnt < 0) {
    server.send(400, "application/json", "{\"error\":\"some angles have no samples (move needle over full range)\"}");
    return;
  }
  bgValid = true;
  saveBg();
  String js = bgJson(false);
  js.remove(js.length() - 1);
  js += ",\"min_samples\":"; js += String(minCnt);
  js += "}";
  server.send(200, "application/json", js);
}

// ---- マーカー ----
String mkJson() {
  String s = "{\"ok\":"; s += mks.ok ? "true" : "false";
  s += ",\"ppm\":"; s += String(mks.ppm, 3);
  s += ",\"fail_idx\":"; s += String(mks.failIdx);
  s += ",\"fail_code\":"; s += String(mks.failCode);
  s += ",\"find_last\":\""; s += mkFindInfo; s += "\"";
  s += ",\"ring\":"; s += String(mks.ring, 2);
  s += ",\"sdx\":"; s += String(mks.sdx, 2); s += ",\"sdy\":"; s += String(mks.sdy, 2);
  s += ",\"cam\":["; s += String(mks.camX, 1); s += ','; s += String(mks.camY, 1); s += ','; s += String(mks.camZ, 1); s += "]";
  s += ",\"u\":["; for (int i = 0; i < 4; i++) { if (i) s += ','; s += String(mks.u[i], 2); } s += "]";
  s += ",\"v\":["; for (int i = 0; i < 4; i++) { if (i) s += ','; s += String(mks.v[i], 2); } s += "]";
  s += ",\"contrast\":["; for (int i = 0; i < 4; i++) { if (i) s += ','; s += String(mks.contrast[i], 2); } s += "]";
  s += ",\"area\":["; for (int i = 0; i < 4; i++) { if (i) s += ','; s += String(mks.area[i]); } s += "]";
  s += ",\"H\":["; for (int i = 0; i < 9; i++) { if (i) s += ','; s += String(mks.H[i], 7); } s += "]";
  s += ",\"Hinv\":["; for (int i = 0; i < 9; i++) { if (i) s += ','; s += String(mks.Hinv[i], 9); } s += "]";
  s += "}";
  return s;
}

void handleMk() {
  uiTouch();
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", mkJson());
}

// ---- マーカー4点の自動探索(2026-11-17) ----

// 新しい画像で自動探索し、見つかれば初期位置(mu/mv)を更新して保存する
bool markerAutoFindNow() {
  camera_fb_t* fb = grabFresh();
  if (!fb) { mkFindInfo = "capture failed"; return false; }
  float u[4], v[4]; char info[96];
  uint32_t t0 = millis();
  bool ok = markerAutoFind(fb->buf, fb->width, fb->height, cfg, u, v, info, sizeof(info));
  if (ok) {
    for (int i = 0; i < 4; i++) { cfg.mu[i] = u[i]; cfg.mv[i] = v[i]; }
    markerReset(cfg);
    ok = detectMarkers(fb->buf, fb->width, fb->height, cfg);   // 射影変換まで通ることを確認
    if (ok) saveConfig();
    else strncat(info, " / detect failed", sizeof(info) - strlen(info) - 1);
  }
  memcpy(lastFrame, fb->buf, (size_t)frameW * frameH);
  esp_camera_fb_return(fb);
  mkFindAt = millis();
  mkFindInfo = String(info) + " (" + String(millis() - t0) + " ms)";
  Serial.printf("[mk-find] %s %s\n", ok ? "OK" : "NG", mkFindInfo.c_str());
  return ok;
}

void handleMkFind() {
  uiTouch();
  bool ok = markerAutoFindNow();
  String js = mkJson();
  js.remove(js.length() - 1);
  js += ",\"find_ok\":"; js += ok ? "true" : "false";
  js += ",\"find_info\":\""; js += mkFindInfo; js += "\"}";
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", js);
}

void handleMkDetect() {
  uiTouch();
  camera_fb_t* fb = grabFresh();
  if (!fb) { server.send(500, "text/plain", "capture failed"); return; }
  detectMarkers(fb->buf, fb->width, fb->height, cfg);
  memcpy(lastFrame, fb->buf, (size_t)frameW * frameH);
  esp_camera_fb_return(fb);
  handleMk();
}

void handleCam() { uiTouch(); server.send(200, "application/json", camJson()); }

void handleCamSet() {
  uiTouch();
  if (server.hasArg("auto")) camSet.autoExp = server.arg("auto").toInt() != 0;
  if (server.hasArg("aec"))  camSet.aec  = constrain(server.arg("aec").toInt(), 0, aecMax());
  if (server.hasArg("gain")) camSet.gain = constrain(server.arg("gain").toInt(), 0, gainMax());
  applyCam();
  saveCam();
  bgReset(); clearBgNvs();   // 露出を変えたら背景は無効
  server.send(200, "application/json", camJson());
}

void handleBgClear() {
  uiTouch();
  bgReset();
  clearBgNvs();
  server.send(200, "application/json", bgJson(false));
}

void handleBg() {
  uiTouch();
  server.sendHeader("Cache-Control", "no-store");
  server.send(200, "application/json", bgJson(true));
}

// ---------------------------------------------------------------------
void setup() {
  // LED表示: 起動直後2秒点灯 → Wi-Fi接続中は0.5秒点滅 → 接続後は5秒ごとに短く点灯
  //          カメラ異常は0.1秒の高速点滅
  pinMode(PIN_LED, OUTPUT);
  led(true);
  delay(2000);
  led(false);
  Serial.begin(115200);
  // USB(CDC)はリセットで一度切断される。PC側が再接続するまで最大5秒待つ
  unsigned long tw = millis();
  while (!Serial && millis() - tw < 5000) delay(10);
  delay(500);
  Serial.println("\n=== gauge-xiao boot ===");
  if (!initCamera()) {
    while (true) {
      Serial.println("camera init failed: " + camDiag);
      for (int i = 0; i < 20; i++) { led(i % 2 == 0); delay(100); }
    }
  }
  loadConfig();
  markerReset(cfg);
  loadCam();
  applyCam();
  loadBg();
  Serial.printf("exposure: %s\n", camJson().c_str());
  Serial.printf("background: %s (frames=%d)\n", bgValid ? "loaded" : "none", bgCount);
  Serial.printf("camera %dx%d gray  (%s)\n", frameW, frameH, camDiag.c_str());

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  unsigned long tc = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print('.');
    led((millis() / 500) % 2);
    if (millis() - tc > 20000) {   // 20秒つながらなければ状態を表示して再試行
      Serial.printf("\nWiFi status=%d (SSID: %s) retry\n", WiFi.status(), WIFI_SSID);
      WiFi.disconnect(); WiFi.begin(WIFI_SSID, WIFI_PASS); tc = millis();
    }
  }
  led(false);
  Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());
  if (MDNS.begin("gauge-xiao")) Serial.println("mDNS: http://gauge-xiao.local/");

  server.on("/",            handleRoot);
  server.on("/capture.jpg", handleCapture);
  server.on("/last.jpg",    handleLast);
  server.on("/read",        handleRead);
  server.on("/config",      handleConfig);
  server.on("/config/set",  handleConfigSet);
  server.on("/profile",     handleProfile);
  server.on("/bg",          handleBg);
  server.on("/bg/reset",    handleBgReset);
  server.on("/bg/add",      handleBgAdd);
  server.on("/bg/save",     handleBgSave);
  server.on("/bg/clear",    handleBgClear);
  server.on("/cam",         handleCam);
  server.on("/cam/set",     handleCamSet);
  server.on("/mk",          handleMk);
  server.on("/mk/detect",   handleMkDetect);
  server.on("/mk/find",     handleMkFind);
  server.on("/log",         handleLog);
  server.on("/log/set",     handleLogSet);
  server.on("/nn",          handleNn);
  server.on("/hom",         handleHom);
  server.begin();

  // 起動時: 保存済みの位置でマーカーが見つからなければ、画像全体から自動探索する
  if (cfg.mk) {
    camera_fb_t* fb = grabFresh();
    bool ok = false;
    if (fb) { ok = detectMarkers(fb->buf, fb->width, fb->height, cfg); esp_camera_fb_return(fb); }
    Serial.printf("marker at saved position: %s\n", ok ? "found" : "not found");
    if (!ok) markerAutoFindNow();
  }
}

void loop() {
  server.handleClient();

  // 定期読み取り・送信
  static unsigned long tLog = 0;
  if (lg.on && !uiActive() && millis() - tLog >= (unsigned long)lg.intv * 1000UL) {
    tLog = millis();
    logOnce();
    Serial.printf("[log] seq=%lu code=%d %s\n", (unsigned long)lgSeq, lgLastCode, lgLastJson.c_str());
  }

  // シリアルモニタを後から開いても分かるよう、5秒ごとに状態を表示
  static unsigned long tPrint = 0;
  if (millis() - tPrint > 5000) {
    tPrint = millis();
    led(true); delay(50); led(false);
    Serial.printf("[status] IP: %s  RSSI: %d dBm  http://gauge-xiao.local/\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  }
}
