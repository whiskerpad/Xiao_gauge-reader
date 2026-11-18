#pragma once
// =====================================================================
// gauge.h  円環走査による針角度検出(背景差分対応)
//
// 角度の定義: 画像の真上(12時)を0°、時計回りを正(0〜360°)
//   x = cx + r*sin(θ),  y = cy - r*cos(θ)
//
// 手順:
//   1. 0目盛り角 a0 から最大目盛り角 a1 まで(時計回り)±マージンを0.5°刻みで走査
//   2. 各角度で半径 rIn〜rOut 上の画素の平均輝度 → 角度プロファイル prof[]
//   3. 「暗さ」dark[] [%] を求める
//        背景あり: 画像/背景 の比の中央値 g で露出変化を補正し、dark = (1 - prof/(g*背景))*100
//                  … 数字・目盛り・赤針などの固定物が消え、照明むら・自動露出の影響も受けにくい
//        背景なし: dark = (1 - prof/中央値)*100
//   4. dark が最大の角度の周囲で、最大値の半分を超える連続区間を針とみなす
//   5. 区間内を (dark - 閾値) で重み付けした重心角度を針角度とする(ぶれ画像では振れの中心)
//
// 背景の作り方: 針を目盛り全域で動かしながら bgAddCurrent() を繰り返す。
//   角度ごとに「針が写っていないフレーム」の平均を取る(針の位置±除外幅は加算しない)。
//   平均なのでノイズが打ち消され、最大値方式のようなノイズ・白飛び側への偏りがない。
// プロファイルは角度方向に [1,2,1]/4 で平滑化する(0.5°刻み → 実効約1°)。
//
// 多帯モード(背景なし・現場向け, cfg.nb >= 2):
//   針は中心から外へ伸びる長い線なので、どの半径の帯でも同じ角度に暗く写る。
//   主目盛り線(外側)、数字(中間)、文字・針のおしり(内側)は特定の帯にしか無い。
//   帯ごとに 暗さ(%) を求め、幅 ±hpw 度の移動中央値を差し引いて照明むら等の
//   「幅の広い変化」を除去し(ハイパス)、角度ごとに全帯の最小値を取る → 針だけが残る。
//   背景(針を動かして作るもの)を必要としないので、針を動かせない現場でも使える。
//
// マーカーモード(cfg.mk = 1、2026-11-15追加):
//   計器の外縁(文字盤φ95〜外形φ115の間)に貼った黒丸マーカー4点(12時・3時・6時・9時、
//   半径 mkr [mm])を毎フレーム検出し、文字盤平面[mm] → 画像[px] の射影変換 H を求める。
//   走査点(針の回転中心 pcx,pcy [mm] から角度・半径[mm])を H で画像座標に変換して読むので、
//   カメラを斜めから向けても、多少動いても、正面から見た文字盤として解析できる。
//   帯の半径は mm で指定(mbr)。角度 ma0/ma1 は文字盤平面上の角度(12時=0°、時計回り)。
//   解析は多帯モードと同じ(帯ごとの暗さ→ハイパス→最小値)。マーカーが見つからなければ値を出さない。
// =====================================================================
#include <Arduino.h>
#include <algorithm>
#include <math.h>

struct GaugeConfig {
  float cx = 320, cy = 240;     // 文字盤中心 [px]
  float rIn = 60, rOut = 150;   // 走査半径 [px]
  float a0 = 225, a1 = 135;     // 0目盛り・最大目盛りの画像角度 [deg]
  float vMin = 0.0, vMax = 1.0; // 目盛り値
  float corrA = 1.0, corrB = 0.0; // 直線補正: 指示値 = corrA * 生値 + corrB
  int   nb = 0;                   // 多帯モードの帯の数(0=従来の単一帯 rIn〜rOut、2〜3=多帯)
  float br[3][2] = {{50, 70}, {75, 95}, {100, 112}};  // 各帯の内半径・外半径 [px]
  float hpw = 8.0f;               // ハイパスの片側幅 [deg](0=無効)。針のぶれ幅より大きくする
  // ---- マーカーモード ----
  int   mk = 0;                   // 1=マーカーモード
  float mkr = 52.5f;              // マーカー中心の半径 [mm](文字盤平面、文字盤中心から)
  float mu[4] = {320, 400, 320, 240};  // マーカー初期位置 [px](上・右・下・左の順)
  float mv[4] = {160, 240, 320, 240};
  int   mkWin = 20;               // マーカー探索窓の半径 [px]
  int   mkPol = 0;                // マーカーの色: 0=白地に黒丸, 1=黒地に白丸
  float mkd = 6.0f;               // マーカー(丸)の直径 [mm]
  float pcx = 0, pcy = 0;         // 針の回転中心 [mm](文字盤平面。マーカー円の中心が原点、右・下が正)
  // ---- 視差補正(2026-11-16) ----
  //   マーカーは計器の縁(前面)に貼るので、針はマーカー面より奥にある。斜めから撮ると、
  //   マーカー面上で見た針の回転中心がカメラ側へずれる。カメラ位置を射影変換から求め、毎フレームこのずれを補正する。
  float nd = 0;                   // 針の奥行き [mm](マーカー面から針までの距離、奥が正)。0=補正なし
  float camf = 460;               // カメラの焦点距離 [px](VGA)
  float ma0 = 225, ma1 = 135;     // 0目盛り・最大目盛りの文字盤平面での角度 [deg]
  float mbr[3][2] = {{19, 26}, {27, 33}, {34, 38}};  // 帯の内半径・外半径 [mm]
  // ---- リング方式(マーカーモード時、2026-11-16) ----
  int   rgOn = 1;                 // 1=細いリングを多数使う方式(帯の設定は使わない)、0=3帯の最小値
  float rg0 = 18, rg1 = 38;       // リングを取る半径範囲 [mm](1mm刻み)
  float rgq = 0.3f;               // 角度ごとに「暗さの小さい方からこの割合」の値を使う(0.3=リングの7割以上で暗ければ針)
  // ---- 固定の置針(赤針など)の除外 ----
  int   fpOn = 0;                 // 1=置針の角度を針の候補から外す
  float fpv = 0.80f;              // 置針の位置(目盛り値)
  float fpw = 3.0f;               // 除外する角度の幅(片側)[deg]
};

struct GaugeResult {
  bool  ok = false;
  bool  bg = false;    // 背景差分を使ったか
  float angle = 0;     // 画像角度 [deg]
  float rel = 0;       // 0目盛りからの角度 [deg]
  float value = 0;     // 指示値(直線補正後)
  float raw = 0;       // 指示値(補正前)
  float snr = 0;       // 針の暗さ / 針以外の最大の暗さ
  bool  multi = false; // 多帯モードで解析したか
  bool  mkUsed = false;// マーカーモードで解析したか
  bool  mkOk = false;  // マーカー4点を検出できたか
  float width = 0;     // 針とみなした区間の幅 [deg]
  float contrast = 0;  // 針の暗さ(背景に対する暗さの割合 0〜1)
  float gain = 1;      // 背景に対する画像の明るさの比(露出補正量)
  uint32_t us = 0;     // 処理時間 [us]
};

static const float  PROF_STEP   = 0.5f;
static const float  PROF_MARGIN = 5.0f;
static const int    PROF_MAX    = 800;
// 合否判定: 針の暗さが「針以外の場所の最大の暗さ」より十分大きいこと
static const float  OK_MIN_DARK = 8.0f;   // 針の暗さの下限 [%]
static const float  OK_SNR      = 1.5f;   // 針の暗さ / 針以外の最大の暗さ の下限
static const float  SNR_EXCL    = 6.0f;   // 針の前後この角度 [deg] は「針以外」から除く

static float prof[PROF_MAX];
static float dark[PROF_MAX];
static float profSorted[PROF_MAX];
static float profTmp[PROF_MAX];
static const int HP_MAXK = 80;                  // ハイパス窓の片側最大点数(40°)
static float hpWin[2 * HP_MAXK + 1];
static int   profN = 0;
static float profStart = 0;   // 0目盛りからの相対角度

// 背景プロファイル
static float bgProf[PROF_MAX];
static float bgSum[PROF_MAX];     // 記録中の合計
static uint16_t bgCnt[PROF_MAX];  // 記録中の加算回数
static int   bgN = 0;         // 背景の点数(profN と一致しないと使わない)
static int   bgCount = 0;     // 背景に取り込んだフレーム数
static bool  bgValid = false; // 保存済みで使用可能

inline float gaugeA0(const GaugeConfig& c) { return c.mk ? c.ma0 : c.a0; }
inline float gaugeA1(const GaugeConfig& c) { return c.mk ? c.ma1 : c.a1; }

inline float gaugeSpan(const GaugeConfig& c) {
  float s = fmodf(gaugeA1(c) - gaugeA0(c) + 720.0f, 360.0f);
  return (s < 1.0f) ? 360.0f : s;
}

// 半径 r0〜r1 の帯の角度プロファイルを out[] に作る(profN, profStart も設定)
inline void computeProfileR(const uint8_t* img, int w, int h, const GaugeConfig& c,
                            float r0, float r1, float* out) {
  float span = gaugeSpan(c);
  profStart = -PROF_MARGIN;
  profN = (int)((span + 2 * PROF_MARGIN) / PROF_STEP) + 1;
  if (profN > PROF_MAX) profN = PROF_MAX;

  for (int i = 0; i < profN; i++) {
    float rel = profStart + i * PROF_STEP;
    float th  = (c.a0 + rel) * DEG_TO_RAD;
    float sx = sinf(th), sy = -cosf(th);
    uint32_t sum = 0, cnt = 0;
    for (float rr = r0; rr <= r1; rr += 1.0f) {
      int x = (int)(c.cx + rr * sx + 0.5f);
      int y = (int)(c.cy + rr * sy + 0.5f);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      sum += img[y * w + x];
      cnt++;
    }
    profTmp[i] = cnt ? (float)sum / cnt : 255.0f;
  }
  // 角度方向の平滑化 [1,2,1]/4(端はそのまま)
  for (int i = 0; i < profN; i++) {
    if (i == 0 || i == profN - 1) out[i] = profTmp[i];
    else out[i] = (profTmp[i - 1] + 2.0f * profTmp[i] + profTmp[i + 1]) * 0.25f;
  }
}

// ---------------------------------------------------------------------
// マーカー検出と射影変換
// ---------------------------------------------------------------------
struct MarkerState {
  float u[4] = {0}, v[4] = {0};  // 画像上のマーカー中心 [px](追跡中の位置)
  bool  ok = false;              // 直近の検出が成功したか
  bool  init = false;            // 初期位置が設定済みか
  float H[9] = {0};              // 文字盤平面[mm] → 画像[px]
  float Hinv[9] = {0};           // 画像[px] → 文字盤平面[mm]
  float ppm = 0;                 // 画像上の縮尺 [px/mm](平均)
  float sdx = 0, sdy = 0;        // 視差による回転中心のずれ [mm](マーカー面上)
  float camX = 0, camY = 0, camZ = 0;  // カメラ位置 [mm](マーカー面座標、Z<0 が手前)
  float contrast[4] = {0};       // マーカーのコントラスト(診断用)
  int   area[4] = {0};           // マーカーとみなした画素数(診断用)
  int   failIdx = -1;            // 失敗したマーカー(0上 1右 2下 3左)、-1=なし
  float ring = 0;                // 紙リングの判定値(マーカー円が外側より明るい角度の割合)
  int   failCode = 0;            // 失敗理由 1:窓が画像外 2:コントラスト不足 3:大きさ不一致 4:変換失敗 5:縮尺異常 6:形が不正 7:紙リングが見えない(どちらも追跡が別の点に移った可能性)
};
static MarkerState mks;

inline void markerReset(const GaugeConfig& c) {
  for (int i = 0; i < 4; i++) { mks.u[i] = c.mu[i]; mks.v[i] = c.mv[i]; }
  mks.ok = false;
  mks.init = true;
}

// 8x8 連立方程式をガウス消去(部分ピボット)で解く。成功で true
inline bool solve8(float A[8][9]) {
  for (int col = 0; col < 8; col++) {
    int piv = col;
    for (int r = col + 1; r < 8; r++) if (fabsf(A[r][col]) > fabsf(A[piv][col])) piv = r;
    if (fabsf(A[piv][col]) < 1e-9f) return false;
    if (piv != col) for (int k = 0; k < 9; k++) std::swap(A[col][k], A[piv][k]);
    for (int r = 0; r < 8; r++) {
      if (r == col) continue;
      float f = A[r][col] / A[col][col];
      for (int k = col; k < 9; k++) A[r][k] -= f * A[col][k];
    }
  }
  for (int r = 0; r < 8; r++) A[r][8] /= A[r][r];
  return true;
}

// 4点の対応 (x,y) → (u,v) から射影変換 H(h22=1)を求める
inline bool homography4(const float x[4], const float y[4], const float u[4], const float v[4], float H[9]) {
  float A[8][9];
  for (int i = 0; i < 4; i++) {
    float* r0 = A[2 * i];
    float* r1 = A[2 * i + 1];
    r0[0] = x[i]; r0[1] = y[i]; r0[2] = 1; r0[3] = 0;    r0[4] = 0;    r0[5] = 0; r0[6] = -u[i] * x[i]; r0[7] = -u[i] * y[i]; r0[8] = u[i];
    r1[0] = 0;    r1[1] = 0;    r1[2] = 0; r1[3] = x[i]; r1[4] = y[i]; r1[5] = 1; r1[6] = -v[i] * x[i]; r1[7] = -v[i] * y[i]; r1[8] = v[i];
  }
  if (!solve8(A)) return false;
  for (int k = 0; k < 8; k++) H[k] = A[k][8];
  H[8] = 1.0f;
  return true;
}

inline bool invert3(const float m[9], float o[9]) {
  float a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  float A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  float det = a * A + b * B + c * C;
  if (fabsf(det) < 1e-12f) return false;
  float inv = 1.0f / det;
  o[0] = A * inv; o[1] = -(b * i - c * h) * inv; o[2] = (b * f - c * e) * inv;
  o[3] = B * inv; o[4] = (a * i - c * g) * inv;  o[5] = -(a * f - c * d) * inv;
  o[6] = C * inv; o[7] = -(a * h - b * g) * inv; o[8] = (a * e - b * d) * inv;
  return true;
}

inline void applyH(const float H[9], float x, float y, float& u, float& v) {
  float w = H[6] * x + H[7] * y + H[8];
  if (fabsf(w) < 1e-9f) w = 1e-9f;
  u = (H[0] * x + H[1] * y + H[2]) / w;
  v = (H[3] * x + H[4] * y + H[5]) / w;
}

// 1個のマーカーを (u0,v0) の周囲 ±win px で探し、黒丸の重心に更新する(2026-11-16 改良2)
//   ・種: 予測位置の周囲で最も暗い画素
//   ・種から、しきい値より暗い画素を4近傍で塗りつぶす。ただし種から半径 rlim(黒丸の半径×1.4)以内に限る
//     → 黒丸が計器の縁・影など暗い部分とつながっていても、黒丸の大きさ以上には広がらない
//   ・得た重心を新しい中心として、もう一度塗りつぶして重心を取り直す(中心のずれを減らす)
//   失敗理由は reason に返す(1:窓が画像外 2:コントラスト不足 3:大きさ不一致)
static const int MK_WIN_MAX = 40;                         // 探索窓の半径の上限 [px]
static const int MK_WSZ = 2 * MK_WIN_MAX + 1;

inline bool findOneMarker(const uint8_t* img, int w, int h, float& u0, float& v0, int win, int pol,
                          float& contrastOut, int& areaOut, float rdot, int& reason) {
  // 2026-11-16 改良3: 黒丸の形で探す(テンプレート照合)
  //   各候補位置で「中心の円(半径0.6r)の平均」と「周囲の輪(半径1.7r)16点の暗い方から6番目」の差を評価し、
  //   予測位置からの距離で減点した値が最大の位置を黒丸とする。計器の縁・影は黒丸と同じ濃さでも、
  //   周囲が明るくないので選ばれない(改良2では斜め撮影で下の黒丸が縁とつながり、中心が5〜10px揺れた)。
  //   その後、その位置から黒丸の半径×1.15以内でしきい値以下の画素の重心を取って中心を精密化する。
  reason = 0;
  if (win > MK_WIN_MAX) win = MK_WIN_MAX;
  if (rdot <= 0) rdot = win * 0.25f;
  if (rdot > 30) rdot = 30;
  int pad = (int)(rdot * 1.8f) + 2;
  int x0 = max(pad, (int)u0 - win), x1 = min(w - 1 - pad, (int)u0 + win);
  int y0 = max(pad, (int)v0 - win), y1 = min(h - 1 - pad, (int)v0 + win);
  if (x1 - x0 < 2 || y1 - y0 < 2) { reason = 1; return false; }
  auto px = [&](int x, int y) -> int { int p = img[y * w + x]; return pol ? 255 - p : p; };
  int8_t ix[16], iy[16], ox[16], oy[16];
  for (int k = 0; k < 16; k++) {
    float a = k * 0.3927f;
    ix[k] = (int8_t)lroundf(rdot * 0.6f * cosf(a)); iy[k] = (int8_t)lroundf(rdot * 0.6f * sinf(a));
    ox[k] = (int8_t)lroundf(rdot * 1.7f * cosf(a)); oy[k] = (int8_t)lroundf(rdot * 1.7f * sinf(a));
  }
  float best = -1e9f, bIn = 0, bOut = 0; int bx = -1, by = -1;
  for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
    int si = px(x, y);
    for (int k = 0; k < 16; k++) si += px(x + ix[k], y + iy[k]);
    int o[16];
    for (int k = 0; k < 16; k++) o[k] = px(x + ox[k], y + oy[k]);
    // 周囲の明るさ = 輪16点の暗い方から6番目の値(2026-11-16 改良4)
    //   黒丸: 輪のほぼ全周が明るい → 高い。縁・影に接する黒丸: 暗いのは数点 → 高いまま
    //   縦長の暗い帯(文字盤の縁など): 輪の半分近くが暗い → 低い → 選ばれない(暗い部屋で左マーカーが帯へずれていった)
    std::nth_element(o, o + 5, o + 16);
    float mi = si / 17.0f, mo = (float)o[5];
    float sc = mo - mi - 1.0f * sqrtf((float)((x - u0) * (x - u0) + (y - v0) * (y - v0)));   // 予測位置から遠いほど減点
    if (sc > best) { best = sc; bx = x; by = y; bIn = mi; bOut = mo; }
  }
  if (bx < 0) { reason = 1; return false; }
  contrastOut = (bOut > 0) ? (bOut - bIn) / bOut : 0;
  if (bOut - bIn < 10 || contrastOut < 0.06f) { reason = 2; return false; }   // 2026-11-16: 映り込み・暗い部屋で薄くなったマーカーも採用
  float thr = (bIn + bOut) * 0.5f, cu = (float)bx, cv = (float)by, rl = rdot * 1.15f;
  int cnt = 0;
  for (int pass = 0; pass < 2; pass++) {
    double sw = 0, sxs = 0, sys = 0; cnt = 0;
    int ra = (int)ceilf(rl);
    for (int y = (int)cv - ra; y <= (int)cv + ra; y++) for (int x = (int)cu - ra; x <= (int)cu + ra; x++) {
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      float dx = x - cu, dy = y - cv;
      if (dx * dx + dy * dy > rl * rl) continue;
      float d = thr - px(x, y);
      if (d <= 0) continue;
      sw += d; sxs += d * x; sys += d * y; cnt++;
    }
    if (sw <= 0) { reason = 3; return false; }
    cu = (float)(sxs / sw); cv = (float)(sys / sw);
  }
  areaOut = cnt;
  if (cnt < 3.14159f * rdot * rdot * 0.3f) { reason = 3; return false; }
  u0 = cu; v0 = cv;
  return true;
}

// H(マーカー面[mm]→画像[px])と焦点距離からカメラ位置を求め、針の回転中心のずれ(視差)を計算する
inline void markerParallax(const GaugeConfig& c, int w, int h) {
  mks.sdx = mks.sdy = 0;
  if (c.camf <= 0) return;
  const float* H = mks.H;
  float cx = w * 0.5f, cy = h * 0.5f, f = c.camf;
  // M = K^-1 * H
  float M[9];
  for (int j = 0; j < 3; j++) {
    M[0 * 3 + j] = (H[0 * 3 + j] - cx * H[2 * 3 + j]) / f;
    M[1 * 3 + j] = (H[1 * 3 + j] - cy * H[2 * 3 + j]) / f;
    M[2 * 3 + j] = H[2 * 3 + j];
  }
  float n1 = sqrtf(M[0] * M[0] + M[3] * M[3] + M[6] * M[6]);
  float n2 = sqrtf(M[1] * M[1] + M[4] * M[4] + M[7] * M[7]);
  if (n1 < 1e-12f || n2 < 1e-12f) return;
  float lam = 2.0f / (n1 + n2);
  if (M[8] < 0) lam = -lam;                       // 平面がカメラの前方(t_z > 0)
  float r1[3] = {M[0] * lam, M[3] * lam, M[6] * lam};
  float r2[3] = {M[1] * lam, M[4] * lam, M[7] * lam};
  float t[3]  = {M[2] * lam, M[5] * lam, M[8] * lam};
  float r3[3] = {r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]};
  // カメラ位置 C = -R^T t
  float Cx = -(r1[0] * t[0] + r1[1] * t[1] + r1[2] * t[2]);
  float Cy = -(r2[0] * t[0] + r2[1] * t[1] + r2[2] * t[2]);
  float Cz = -(r3[0] * t[0] + r3[1] * t[1] + r3[2] * t[2]);
  mks.camX = Cx; mks.camY = Cy; mks.camZ = Cz;
  if (c.nd == 0) return;
  // 奥行き nd の点(回転中心)を、カメラから見てマーカー面に投影した位置とのずれ
  float zp = (Cz < 0) ? c.nd : -c.nd;             // カメラと反対側が奥
  float den = Cz - zp;
  if (fabsf(den) < 1e-3f) return;
  float k = Cz / den;
  mks.sdx = (Cx + (c.pcx - Cx) * k) - c.pcx;
  mks.sdy = (Cy + (c.pcy - Cy) * k) - c.pcy;
  if (fabsf(mks.sdx) > 30 || fabsf(mks.sdy) > 30) mks.sdx = mks.sdy = 0;   // 異常値は使わない
}

// 紙リングの確認(全解像度の画像、H = マーカー面[mm]→画像[px]):
//   マーカー円(半径 mkr ±1.5mm)が 6mm 外側(計器の外)より明るい角度の割合を返す(黒丸の近くは除く)
inline float markerRingFrac(const uint8_t* img, int w, int h, const float H[9], const GaugeConfig& c) {
  int good = 0, tot = 0;
  auto smp = [&](float rm, float th, int& val) -> bool {
    float qu, qv;
    applyH(H, rm * sinf(th), -rm * cosf(th), qu, qv);
    int xi = (int)(qu + 0.5f), yi = (int)(qv + 0.5f);
    if (xi < 0 || yi < 0 || xi >= w || yi >= h) return false;
    int p = img[yi * w + xi]; val = c.mkPol ? 255 - p : p; return true;
  };
  for (int j = 0; j < 48; j++) {
    float a = j * 7.5f;
    float am = fmodf(a, 90.0f); if (am > 45.0f) am = 90.0f - am;
    if (am < 12.0f) continue;
    float th = a * DEG_TO_RAD;
    tot++;
    int b0, b1, b2, bo;
    if (!smp(c.mkr - 1.5f, th, b0) || !smp(c.mkr, th, b1) || !smp(c.mkr + 1.5f, th, b2) || !smp(c.mkr + 6.0f, th, bo)) continue;
    if ((b0 + b1 + b2) / 3.0f - bo > 12.0f) good++;
  }
  return tot ? (float)good / tot : 0;
}
static float g_trackRing = 0.5f;   // 追跡時の紙リング判定の下限(自動探索の0.56より少し緩い)

// 4点を検出して H を更新する。失敗時は追跡位置を変えずに false
inline bool detectMarkers(const uint8_t* img, int w, int h, const GaugeConfig& c) {
  if (!mks.init) markerReset(c);
  mks.failIdx = -1; mks.failCode = 0;
  // 縮尺: 前回の成功値。無ければ現在の追跡位置(=クリックした初期位置)の間隔から見積もる
  float ppmEst = mks.ppm;
  if (ppmEst <= 0) {
    float d1 = hypotf(mks.u[0] - mks.u[2], mks.v[0] - mks.v[2]), d2 = hypotf(mks.u[1] - mks.u[3], mks.v[1] - mks.v[3]);
    ppmEst = (d1 + d2) / (4.0f * c.mkr);
  }
  float rdot = c.mkd * 0.5f * ppmEst;               // 黒丸の半径 [px]
  float u[4], v[4];
  for (int i = 0; i < 4; i++) {
    u[i] = mks.u[i]; v[i] = mks.v[i];
    bool f = false;
    int reason = 0;
    // 通常の窓 → 失敗なら2倍の窓で再探索
    for (int k = 1; k <= 2 && !f; k++) {
      float uu = mks.u[i], vv = mks.v[i];
      float rd = (mks.ok && mks.area[i] > 0) ? sqrtf(mks.area[i] / 3.14159f) : rdot;   // マーカーごとの見かけの半径(遠近で違う)
      if (findOneMarker(img, w, h, uu, vv, c.mkWin * k, c.mkPol, mks.contrast[i], mks.area[i], rd, reason)) {
        u[i] = uu; v[i] = vv; f = true;
      }
    }
    if (!f) { mks.ok = false; mks.failIdx = i; mks.failCode = reason; return false; }
  }
  // 文字盤平面上のマーカー位置(上・右・下・左)
  const float px[4] = {0, c.mkr, 0, -c.mkr};
  const float py[4] = {-c.mkr, 0, c.mkr, 0};
  float H[9], Hi[9];
  if (!homography4(px, py, u, v, H) || !invert3(H, Hi)) { mks.ok = false; mks.failCode = 4; return false; }
  // 妥当性: 対角距離から縮尺を求め、極端な値は失敗扱い
  float d1 = hypotf(u[0] - u[2], v[0] - v[2]), d2 = hypotf(u[1] - u[3], v[1] - v[3]);
  float ppm = (d1 + d2) / (4.0f * c.mkr);
  if (ppm < 0.5f || ppm > 20.0f) { mks.ok = false; mks.failCode = 5; return false; }
  // 形の妥当性(2026-11-17): カメラで撮った円として成り立つか(K^-1 H の第1・2列が同じ長さで直交)。
  //   カメラが大きく動いて追跡が別の暗い点に移ると、ゆがんだ四角形になる → 失敗にして自動探索に任せる
  if (c.camf > 0) {
    float f = c.camf, ccx = w * 0.5f, ccy = h * 0.5f, m[9];
    for (int j = 0; j < 3; j++) {
      m[j] = (H[j] - ccx * H[6 + j]) / f; m[3 + j] = (H[3 + j] - ccy * H[6 + j]) / f; m[6 + j] = H[6 + j];
    }
    float n1 = sqrtf(m[0] * m[0] + m[3] * m[3] + m[6] * m[6]), n2 = sqrtf(m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
    float dot = (m[0] * m[1] + m[3] * m[4] + m[6] * m[7]) / (n1 * n2 + 1e-12f);
    float nr = n1 > n2 ? n1 / n2 : n2 / n1;
    if (nr > 1.35f || fabsf(dot) > 0.3f) { mks.ok = false; mks.failCode = 6; return false; }
  }
  // 紙リングの確認: 追跡が別の暗い点に移ると、マーカー円が白い紙の上を通らない
  mks.ring = markerRingFrac(img, w, h, H, c);
  if (mks.ring < g_trackRing) { mks.ok = false; mks.failCode = 7; return false; }
  for (int i = 0; i < 4; i++) { mks.u[i] = u[i]; mks.v[i] = v[i]; }
  memcpy(mks.H, H, sizeof(H)); memcpy(mks.Hinv, Hi, sizeof(Hi));
  mks.ppm = ppm;
  markerParallax(c, w, h);
  mks.ok = true;
  return true;
}

// ---------------------------------------------------------------------
// マーカー4点の自動探索(2026-11-17)
//   画像全体(1/2に縮小)で「中心が暗く周囲が明るい丸」を半径を変えながら探し、候補を最大24個に絞る。
//   候補から4点の組をすべて試し、「4点が凸四角形で対角線が交わる」「対角線の長さの比が近い」
//   「丸の大きさが4点の間隔(=縮尺)と黒丸の直径から見込まれる大きさと合う」組のうち、点数の合計が最大のものを選ぶ。
//   見つかった4点を上・右・下・左に並べ、通常の検出(findOneMarker)で精密化する。
//   起動時、検出が続けて失敗したとき、/mk/find で実行する。所要時間はおよそ1秒。
// ---------------------------------------------------------------------
#ifndef MK_ALLOC
#define MK_ALLOC(n) malloc(n)
#endif
struct MkCand { int16_t x, y; float r, s; };
static const int MK_NCAND = 40;
static float g_afRing = 0.56f;   // 紙リングの判定: マーカー円が外側より明るい角度の割合の下限

inline int mkScoreAt(const uint8_t* s, int sw, int x, int y, const int8_t* ix, const int8_t* iy,
                     const int8_t* ox, const int8_t* oy, int pol) {
  int si = 0;
  auto P = [&](int xx, int yy) -> int { int p = s[yy * sw + xx]; return pol ? 255 - p : p; };
  si = P(x, y);
  for (int k = 0; k < 16; k++) si += P(x + ix[k], y + iy[k]);
  int o[16];
  for (int k = 0; k < 16; k++) o[k] = P(x + ox[k], y + oy[k]);
  std::nth_element(o, o + 3, o + 16);
  return o[3] - si / 17;   // 周囲(暗い方から4番目) − 中心の平均(フード・縁で輪の一部が暗い黒丸も拾う。偽物は後の4点の判定で落とす)
}

// 成功で u[4], v[4](上・右・下・左、元画像の px)と info(診断文)を返す
inline bool markerAutoFind(const uint8_t* img, int w, int h, const GaugeConfig& c,
                           float u[4], float v[4], char* info, int infoLen) {
  int sw = w / 2, sh = h / 2;
  uint8_t* s  = (uint8_t*)MK_ALLOC((size_t)sw * sh);
  int16_t* sc = (int16_t*)MK_ALLOC((size_t)sw * sh * 2);
  uint8_t* ri = (uint8_t*)MK_ALLOC((size_t)sw * sh);
  if (!s || !sc || !ri) { free(s); free(sc); free(ri); snprintf(info, infoLen, "memory"); return false; }
  for (int y = 0; y < sh; y++) for (int x = 0; x < sw; x++) {
    const uint8_t* p = img + (2 * y) * w + 2 * x;
    s[y * sw + x] = (uint8_t)((p[0] + p[1] + p[w] + p[w + 1] + 2) >> 2);
  }
  memset(sc, 0, (size_t)sw * sh * 2);
  memset(ri, 0, (size_t)sw * sh);
  // 黒丸の半径(縮小画像の px): 2〜9 を 1.3倍刻み
  static const float RAD[6] = {2.0f, 2.6f, 3.4f, 4.4f, 5.7f, 7.4f};
  for (int k = 0; k < 6; k++) {
    float r = RAD[k];
    int8_t ix[16], iy[16], ox[16], oy[16];
    for (int j = 0; j < 16; j++) {
      float a = j * 0.3927f;
      ix[j] = (int8_t)lroundf(r * 0.6f * cosf(a)); iy[j] = (int8_t)lroundf(r * 0.6f * sinf(a));
      ox[j] = (int8_t)lroundf(r * 1.7f * cosf(a)); oy[j] = (int8_t)lroundf(r * 1.7f * sinf(a));
    }
    int pad = (int)ceilf(r * 1.7f) + 1;
    for (int y = pad; y < sh - pad; y++) for (int x = pad; x < sw - pad; x++) {
      int v0 = mkScoreAt(s, sw, x, y, ix, iy, ox, oy, c.mkPol);
      if (v0 > sc[y * sw + x]) { sc[y * sw + x] = (int16_t)v0; ri[y * sw + x] = (uint8_t)k; }
    }
  }
  // 候補: 近傍(半径の2倍)で最大、点数 ≥ 15 の点を点数順に最大 MK_NCAND 個
  MkCand cand[MK_NCAND]; int nc = 0;
  for (int y = 2; y < sh - 2; y++) for (int x = 2; x < sw - 2; x++) {
    int v0 = sc[y * sw + x];
    if (v0 < 15) continue;
    int rr = (int)ceilf(RAD[ri[y * sw + x]] * 2.0f);
    bool mx = true;
    for (int dy = -rr; dy <= rr && mx; dy++) for (int dx = -rr; dx <= rr; dx++) {
      int xx = x + dx, yy = y + dy;
      if ((dx == 0 && dy == 0) || xx < 0 || yy < 0 || xx >= sw || yy >= sh) continue;
      int q = sc[yy * sw + xx];
      if (q > v0 || (q == v0 && (dy < 0 || (dy == 0 && dx < 0)))) { mx = false; break; }
    }
    if (!mx) continue;
    MkCand cd = {(int16_t)x, (int16_t)y, RAD[ri[y * sw + x]], (float)v0};
    if (nc < MK_NCAND) cand[nc++] = cd;
    else { int mi = 0; for (int i = 1; i < nc; i++) if (cand[i].s < cand[mi].s) mi = i; if (cd.s > cand[mi].s) cand[mi] = cd; }
  }
  free(sc); free(ri);
  if (nc < 4) { free(s); snprintf(info, infoLen, "candidates=%d", nc); return false; }
  // 画像全体の明るさの中央値(白い紙リングの判定に使う)
  uint32_t hist[256] = {0};
  for (int i = 0; i < sw * sh; i++) hist[c.mkPol ? 255 - s[i] : s[i]]++;
  int gmed = 0; { uint32_t acc = 0; for (; gmed < 255; gmed++) { acc += hist[gmed]; if (acc * 2 >= (uint32_t)(sw * sh)) break; } }
  const float PX[4] = {0, c.mkr, 0, -c.mkr}, PY[4] = {-c.mkr, 0, c.mkr, 0};
  // 4点の組を評価
  // 点数の上位5組を残し、上から順に精密化して確認する
  const int TOPQ = 5;
  float qs[TOPQ], qf[TOPQ]; int qi[TOPQ][4]; int nq = 0;
  float curFrac = 0;
  for (int a = 0; a < nc; a++) for (int b = a + 1; b < nc; b++) for (int d = b + 1; d < nc; d++) for (int e = d + 1; e < nc; e++) {
    int id[4] = {a, b, d, e};
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; i++) { cx += cand[id[i]].x; cy += cand[id[i]].y; }
    cx /= 4; cy /= 4;
    // 重心まわりの角度(12時=0、時計回り)で並べる
    float ang[4];
    for (int i = 0; i < 4; i++) ang[i] = atan2f(cand[id[i]].x - cx, -(cand[id[i]].y - cy));
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) if (ang[j] < ang[i]) { std::swap(ang[i], ang[j]); std::swap(id[i], id[j]); }
    // 対角線 0-2 と 1-3 が互いに内分すること(凸四角形)
    float x0 = cand[id[0]].x, y0p = cand[id[0]].y, x1 = cand[id[1]].x, y1 = cand[id[1]].y;
    float x2 = cand[id[2]].x, y2 = cand[id[2]].y, x3 = cand[id[3]].x, y3 = cand[id[3]].y;
    float d1x = x2 - x0, d1y = y2 - y0p, d2x = x3 - x1, d2y = y3 - y1;
    float den = d1x * d2y - d1y * d2x;
    if (fabsf(den) < 1e-3f) continue;
    float t = ((x1 - x0) * d2y - (y1 - y0p) * d2x) / den;
    float q = ((x1 - x0) * d1y - (y1 - y0p) * d1x) / den;
    if (t < 0.25f || t > 0.75f || q < 0.25f || q > 0.75f) continue;
    float L1 = hypotf(d1x, d1y), L2 = hypotf(d2x, d2y);
    if (L1 < 110 || L2 < 110) continue;                       // 縮小画像で対角線110px(元220px、縮尺2.1px/mm)未満は対象外
                                                              // → 計器(マーカー円)が画像の高さの半分程度以上に写っていること
    float ratio = L1 > L2 ? L1 / L2 : L2 / L1;
    if (ratio > 2.2f) continue;
    // 黒丸の見込み半径 = 黒丸の半径[mm] × 縮尺[px/mm]
    float ppm = (L1 + L2) / (4.0f * c.mkr);
    float rexp = c.mkd * 0.5f * ppm;
    bool okr = true; float sum = 0;
    for (int i = 0; i < 4; i++) {
      float rr = cand[id[i]].r / rexp;
      if (rr < 0.45f || rr > 2.2f) { okr = false; break; }
      sum += cand[id[i]].s;
    }
    if (!okr) continue;
    // 白い紙リングの確認: 4点から射影変換を求め、マーカー円(半径 mkr ±1.5mm)上の点が、
    //   その6mm外側(計器の外 = ケースの側面や背景)の点より明るい角度の割合を調べる。
    //   照明のむらがあっても角度ごとの比較なので崩れにくい。背景の文字・画面の模様など偶然の4点の組は落ちる
    {
      float UU[4], VV[4], Hc[9];
      int top0 = 0;
      for (int i = 1; i < 4; i++) if (cand[id[i]].y < cand[id[top0]].y) top0 = i;
      for (int i = 0; i < 4; i++) { UU[i] = cand[id[(top0 + i) % 4]].x; VV[i] = cand[id[(top0 + i) % 4]].y; }
      if (!homography4(PX, PY, UU, VV, Hc)) continue;
      // 黒丸ごとの見込み半径(その場所の縮尺 × 黒丸の半径)と、見つけた丸の半径が合うこと
      bool sizeOk = true;
      for (int i = 0; i < 4 && sizeOk; i++) {
        float ax, ay, bx, by, cx2, cy2, dx2, dy2;
        float hr = c.mkd * 0.5f;
        applyH(Hc, PX[i] - hr, PY[i], ax, ay); applyH(Hc, PX[i] + hr, PY[i], bx, by);
        applyH(Hc, PX[i], PY[i] - hr, cx2, cy2); applyH(Hc, PX[i], PY[i] + hr, dx2, dy2);
        float rloc = (hypotf(bx - ax, by - ay) + hypotf(dx2 - cx2, dy2 - cy2)) * 0.25f;
        float ratio2 = cand[id[(top0 + i) % 4]].r / rloc;
        if (ratio2 < 0.6f || ratio2 > 1.7f) sizeOk = false;
      }
      if (!sizeOk) continue;
      // カメラとして成り立つ形か: K^-1 H の第1・第2列(回転行列の2列)が同じ長さで直交すること
      //   (焦点距離 camf を使う。偶然の4点の組はゆがんだ四角形になり、これを満たさない)
      if (c.camf > 0) {
        float f2 = c.camf * 0.5f, ccx = sw * 0.5f, ccy = sh * 0.5f, m[9];
        for (int j = 0; j < 3; j++) {
          m[j] = (Hc[j] - ccx * Hc[6 + j]) / f2; m[3 + j] = (Hc[3 + j] - ccy * Hc[6 + j]) / f2; m[6 + j] = Hc[6 + j];
        }
        float n1 = sqrtf(m[0] * m[0] + m[3] * m[3] + m[6] * m[6]), n2 = sqrtf(m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
        float dot = (m[0] * m[1] + m[3] * m[4] + m[6] * m[7]) / (n1 * n2 + 1e-12f);
        float nr = n1 > n2 ? n1 / n2 : n2 / n1;
        if (nr > 1.25f || fabsf(dot) > 0.2f) continue;
      }
      auto smp = [&](float rm, float th, int& val) -> bool {
        float qu, qv;
        applyH(Hc, rm * sinf(th), -rm * cosf(th), qu, qv);
        int xi = (int)(qu + 0.5f), yi = (int)(qv + 0.5f);
        if (xi < 0 || yi < 0 || xi >= sw || yi >= sh) return false;
        int p = s[yi * sw + xi]; val = c.mkPol ? 255 - p : p; return true;
      };
      int good = 0, tot = 0;
      for (int j = 0; j < 48; j++) {
        float a = j * 7.5f;
        float am = fmodf(a, 90.0f); if (am > 45.0f) am = 90.0f - am;
        if (am < 12.0f) continue;                              // 黒丸の近くは除く
        float th = a * DEG_TO_RAD;
        tot++;
        int b0, b1, b2, bo;
        if (!smp(c.mkr - 1.5f, th, b0) || !smp(c.mkr, th, b1) || !smp(c.mkr + 1.5f, th, b2) || !smp(c.mkr + 6.0f, th, bo)) continue;
        if ((b0 + b1 + b2) / 3.0f - bo > 12.0f) good++;
      }
      float frac = tot ? (float)good / tot : 0;
      if (frac < g_afRing) continue;
      sum += 200.0f * frac;
      curFrac = frac;
    }
    sum -= 10.0f * (ratio - 1.0f);                            // 正面に近いほどわずかに優先
    {
      int pos = nq < TOPQ ? nq : TOPQ;
      while (pos > 0 && qs[pos - 1] < sum) pos--;
      if (pos < TOPQ) {
        int last = nq < TOPQ ? nq : TOPQ - 1;
        for (int k = last; k > pos; k--) { qs[k] = qs[k - 1]; qf[k] = qf[k - 1]; memcpy(qi[k], qi[k - 1], sizeof(qi[0])); }
        qs[pos] = sum; qf[pos] = curFrac; memcpy(qi[pos], id, sizeof(qi[0]));
        if (nq < TOPQ) nq++;
      }
    }
  }
  free(s);
  if (nq == 0) { snprintf(info, infoLen, "no-quad candidates=%d", nc); return false; }
  // 上位の組から順に: 上・右・下・左に並べ、通常の検出(findOneMarker)で4点とも黒丸と確認できた組を採用
  for (int k = 0; k < nq; k++) {
    int top = 0;
    for (int i = 1; i < 4; i++) if (cand[qi[k][i]].y < cand[qi[k][top]].y) top = i;
    bool allOk = true;
    float uu4[4], vv4[4];
    for (int i = 0; i < 4 && allOk; i++) {
      const MkCand& m = cand[qi[k][(top + i) % 4]];
      float uu = m.x * 2.0f + 0.5f, vv = m.y * 2.0f + 0.5f, con; int area, reason;
      if (findOneMarker(img, w, h, uu, vv, 12, c.mkPol, con, area, m.r * 2.0f, reason) &&
          hypotf(uu - (m.x * 2.0f + 0.5f), vv - (m.y * 2.0f + 0.5f)) < m.r * 2.0f) { uu4[i] = uu; vv4[i] = vv; }
      else allOk = false;
    }
    if (!allOk) continue;
    for (int i = 0; i < 4; i++) { u[i] = uu4[i]; v[i] = vv4[i]; }
    float ppm = (hypotf(u[0] - u[2], v[0] - v[2]) + hypotf(u[1] - u[3], v[1] - v[3])) / (4.0f * c.mkr);
    snprintf(info, infoLen, "ok rank=%d candidates=%d score=%.0f ring=%.2f ppm=%.2f", k + 1, nc, qs[k], qf[k], ppm);
    return true;
  }
  snprintf(info, infoLen, "verify-failed quads=%d", nq);
  return false;
}

// マーカーモード: 文字盤平面の半径 r0〜r1 [mm] の帯の角度プロファイルを out[] に作る
inline void computeProfileMk(const uint8_t* img, int w, int h, const GaugeConfig& c,
                             float r0, float r1, float* out) {
  float span = gaugeSpan(c);
  profStart = -PROF_MARGIN;
  profN = (int)((span + 2 * PROF_MARGIN) / PROF_STEP) + 1;
  if (profN > PROF_MAX) profN = PROF_MAX;
  float rstep = (mks.ppm > 0) ? fmaxf(0.15f, 1.0f / mks.ppm) : 0.4f;   // 約1px刻み
  for (int i = 0; i < profN; i++) {
    float rel = profStart + i * PROF_STEP;
    float th  = (c.ma0 + rel) * DEG_TO_RAD;
    float sx = sinf(th), sy = -cosf(th);
    uint32_t sum = 0, cnt = 0;
    for (float rr = r0; rr <= r1; rr += rstep) {
      float uu, vv;
      applyH(mks.H, c.pcx + mks.sdx + rr * sx, c.pcy + mks.sdy + rr * sy, uu, vv);
      int x = (int)(uu + 0.5f), y = (int)(vv + 0.5f);
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      sum += img[y * w + x];
      cnt++;
    }
    profTmp[i] = cnt ? (float)sum / cnt : 255.0f;
  }
  for (int i = 0; i < profN; i++) {
    if (i == 0 || i == profN - 1) out[i] = profTmp[i];
    else out[i] = (profTmp[i - 1] + 2.0f * profTmp[i] + profTmp[i + 1]) * 0.25f;
  }
}

// 従来の単一帯(rIn〜rOut)のプロファイルを prof[] に作る
inline void computeProfile(const uint8_t* img, int w, int h, const GaugeConfig& c) {
  computeProfileR(img, w, h, c, c.rIn, c.rOut, prof);
}

// 暗さ(%): 中央値に対して何%暗いか。p → d
inline void darkVsMedian(const float* p, float* d, int n) {
  memcpy(profSorted, p, sizeof(float) * n);
  std::nth_element(profSorted, profSorted + n / 2, profSorted + n);
  float med = fmaxf(profSorted[n / 2], 1.0f);
  for (int i = 0; i < n; i++) d[i] = (1.0f - p[i] / med) * 100.0f;
}

// ハイパス: d[i] から ±k 点の移動中央値を差し引く(結果は out[])
inline void highPass(const float* d, float* out, int n, int k) {
  if (k <= 0) { memcpy(out, d, sizeof(float) * n); return; }
  if (k > HP_MAXK) k = HP_MAXK;
  for (int i = 0; i < n; i++) {
    int a = max(0, i - k), b = min(n - 1, i + k), m = b - a + 1;
    memcpy(hpWin, d + a, sizeof(float) * m);
    std::nth_element(hpWin, hpWin + m / 2, hpWin + m);
    out[i] = d[i] - hpWin[m / 2];
  }
}

// 白飛びの割合(輝度245以上の点の割合)
inline float profSaturation() {
  int n = 0;
  for (int i = 0; i < profN; i++) if (prof[i] >= 245.0f) n++;
  return profN ? (float)n / profN : 0.0f;
}

// ---- リング方式の作業領域(暗さ% を 0.5% 単位の int8 で保持) ----
static const int RG_MAX = 32;
static int8_t ringBuf[RG_MAX][PROF_MAX];

// 置針の暗さの学習値(針が置針から離れているときの、置針の位置の暗さ)。-1=未学習
static float fpLearn = -1.0f;
static const float FP_RATIO = 1.6f;   // 置針の位置の暗さが学習値のこの倍以上なら、針が重なっているとみなす

// ---- 背景操作 ----
inline void bgReset() {
  bgN = 0; bgCount = 0; bgValid = false;
  memset(bgSum, 0, sizeof(bgSum)); memset(bgCnt, 0, sizeof(bgCnt));
}

// 直前の computeProfile() の結果を背景の平均に加える。
// skipRel: 針の位置(0目盛りからの角度)。その±skipW度は加算しない。skipRel が NAN なら全点加算
inline void bgAddCurrent(float skipRel, float skipW) {
  if (bgCount == 0 || bgN != profN) {
    memset(bgSum, 0, sizeof(bgSum)); memset(bgCnt, 0, sizeof(bgCnt));
    bgN = profN; bgCount = 0;
  }
  for (int i = 0; i < profN; i++) {
    float rel = profStart + i * PROF_STEP;
    if (!isnan(skipRel) && fabsf(rel - skipRel) <= skipW) continue;
    bgSum[i] += prof[i];
    bgCnt[i]++;
  }
  bgCount++;
}

// 平均を確定して bgProf に入れる。加算0回の点があれば -1 を返す(失敗)。戻り値は最小加算回数
inline int bgFinalize() {
  int minCnt = 65535;
  for (int i = 0; i < bgN; i++) {
    if (bgCnt[i] == 0) return -1;
    bgProf[i] = bgSum[i] / bgCnt[i];
    if (bgCnt[i] < minCnt) minCnt = bgCnt[i];
  }
  return minCnt;
}

inline bool bgUsable() { return bgValid && bgN == profN && bgN > 0; }

// ---- 解析 ----
inline bool analyzeGauge(const uint8_t* img, int w, int h,
                         const GaugeConfig& c, GaugeResult& r) {
  uint32_t t0 = micros();
  r = GaugeResult();
  float span = gaugeSpan(c);

  if (c.mk) {
    r.mkUsed = true;
    r.mkOk = detectMarkers(img, w, h, c);
    if (!r.mkOk) { r.us = micros() - t0; return false; }   // マーカーが無ければ値を出さない
  }

  if (c.mk || c.nb >= 2) {
    // ---- 多帯モード(背景なし)。マーカーモードも同じ処理で、走査座標だけ射影変換を使う ----
    // dark[] = 全帯の (暗さ → ハイパス) の最小値。prof[] には最外帯の輝度を残す(グラフ・白飛び判定用)
    r.multi = true;
    r.bg = false;
    r.gain = 1.0f;
    int k = (int)(c.hpw / PROF_STEP + 0.5f);
    if (c.mk && c.rgOn) {
      // ---- リング方式: 1mm幅のリングごとに 暗さ→ハイパス、角度ごとに下から rgq の割合の値を採用 ----
      //   針は中心から先端まで続く線なので、文字や数字で一部のリングが隠れても残る。
      //   文字・数字・目盛り線は一部のリングにしか無いので落ちる。
      int nr = 0;
      for (float rr = c.rg0; rr < c.rg1 - 0.01f && nr < RG_MAX; rr += 1.0f, nr++) {
        computeProfileMk(img, w, h, c, rr, rr + 1.0f, prof);
        darkVsMedian(prof, profTmp, profN);
        highPass(profTmp, profSorted, profN, k);
        for (int i = 0; i < profN; i++) {
          float v = profSorted[i] * 2.0f;
          ringBuf[nr][i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
      }
      if (nr < 1) { r.us = micros() - t0; return false; }
      int q = (int)(nr * c.rgq);
      if (q > nr - 1) q = nr - 1;
      int8_t col[RG_MAX];
      for (int i = 0; i < profN; i++) {
        for (int j = 0; j < nr; j++) col[j] = ringBuf[j][i];
        std::nth_element(col, col + q, col + nr);
        dark[i] = col[q] * 0.5f;
      }
    } else {
    int nb = c.mk ? 3 : min(c.nb, 3);
    for (int b = 0; b < nb; b++) {
      if (c.mk) computeProfileMk(img, w, h, c, c.mbr[b][0], c.mbr[b][1], prof);
      else      computeProfileR(img, w, h, c, c.br[b][0], c.br[b][1], prof);
      darkVsMedian(prof, profTmp, profN);
      if (b == 0) highPass(profTmp, dark, profN, k);
      else {
        highPass(profTmp, profSorted, profN, k);   // profSorted を作業領域に流用
        for (int i = 0; i < profN; i++) if (profSorted[i] < dark[i]) dark[i] = profSorted[i];
      }
    }
    }
    // prof[] は最後に計算した帯/リング(最外側)の輝度
  } else {
  computeProfile(img, w, h, c);

  // 暗さ
  //   背景あり: 露出の変化を補正するため、画像/背景 の比の中央値 g を求め、
  //             dark = (1 - prof/(g*bg)) * 100  [%]  … 背景に対して何%暗いか
  //   背景なし: dark = (1 - prof/中央値) * 100  [%]
  r.bg = bgUsable();
  if (r.bg) {
    for (int i = 0; i < profN; i++) profSorted[i] = prof[i] / fmaxf(bgProf[i], 1.0f);
    std::nth_element(profSorted, profSorted + profN / 2, profSorted + profN);
    float g = profSorted[profN / 2];
    r.gain = g;
    for (int i = 0; i < profN; i++) dark[i] = (1.0f - prof[i] / fmaxf(g * bgProf[i], 1.0f)) * 100.0f;
  } else {
    memcpy(profSorted, prof, sizeof(float) * profN);
    std::nth_element(profSorted, profSorted + profN / 2, profSorted + profN);
    float med = fmaxf(profSorted[profN / 2], 1.0f);
    r.gain = 1.0f;
    for (int i = 0; i < profN; i++) dark[i] = (1.0f - prof[i] / med) * 100.0f;
  }
  }  // 単一帯モードここまで

  // ---- 針の位置の決定 ----
  // 置針の除外が有効なら、置針の角度範囲 [zlo, zhi] を一旦候補から外して探す。
  //   見つからず、かつ置針の位置が「学習した置針だけの暗さ」の FP_RATIO 倍以上暗ければ、針が置針に重なっていると判断する。
  int zlo = -1, zhi = -1;
  if (c.fpOn && (c.mk || c.nb >= 2) && c.vMax != c.vMin) {
    float relZ = (c.fpv - c.vMin) / (c.vMax - c.vMin) * span;
    zlo = max(0, (int)((relZ - c.fpw - profStart) / PROF_STEP));
    zhi = min(profN - 1, (int)((relZ + c.fpw - profStart) / PROF_STEP + 0.999f));
    if (zlo > zhi) zlo = zhi = -1;
  }
  float zonePeak = -999;
  if (zlo >= 0) for (int i = zlo; i <= zhi; i++) if (dark[i] > zonePeak) zonePeak = dark[i];

  auto pick = [&](int lo, int hi, bool skipZone, int& pk, float& relOut, int& Lo, int& Ro) {
    pk = -1;
    for (int i = lo; i <= hi; i++) {
      if (skipZone && zlo >= 0 && i >= zlo && i <= zhi) continue;
      if (pk < 0 || dark[i] > dark[pk]) pk = i;
    }
    if (pk < 0) pk = lo;
    float pv = dark[pk], th = pv * 0.5f;
    int L = pk, R = pk;
    while (L > lo && dark[L - 1] > th && !(skipZone && zlo >= 0 && L - 1 >= zlo && L - 1 <= zhi)) L--;
    while (R < hi && dark[R + 1] > th && !(skipZone && zlo >= 0 && R + 1 >= zlo && R + 1 <= zhi)) R++;
    float ws = 0, was = 0;
    for (int i = L; i <= R; i++) { float wgt = dark[i] - th; if (wgt <= 0) continue; ws += wgt; was += wgt * (profStart + i * PROF_STEP); }
    relOut = (ws > 0) ? was / ws : profStart + pk * PROF_STEP;
    Lo = L; Ro = R;
  };
  auto otherMax = [&](float relC, bool skipZone) {
    float o = 0;
    for (int i = 0; i < profN; i++) {
      float rel = profStart + i * PROF_STEP;
      if (fabsf(rel - relC) <= SNR_EXCL) continue;
      if (skipZone && zlo >= 0 && i >= zlo && i <= zhi) continue;
      if (dark[i] > o) o = dark[i];
    }
    return o;
  };

  int pk, L, R; float rel;
  pick(0, profN - 1, zlo >= 0, pk, rel, L, R);
  float peak = dark[pk];
  float other = otherMax(rel, zlo >= 0);
  float snr = (other > 0.1f) ? peak / other : 99.0f;
  bool ok = (peak >= OK_MIN_DARK) && (snr >= OK_SNR);

  if (zlo >= 0) {
    float relZ = (c.fpv - c.vMin) / (c.vMax - c.vMin) * span;
    if (ok && fabsf(rel - relZ) > c.fpw + SNR_EXCL) {
      // 針が置針から十分離れている → 置針だけの暗さを学習
      fpLearn = (fpLearn < 0) ? zonePeak : fpLearn * 0.9f + zonePeak * 0.1f;
    } else if (!ok && fpLearn > 0 && zonePeak >= OK_MIN_DARK && zonePeak >= fpLearn * FP_RATIO) {
      // 置針の位置が普段より明らかに暗い → 針が重なっている
      int pk2, L2, R2; float rel2;
      pick(zlo, zhi, false, pk2, rel2, L2, R2);
      float other2 = otherMax(rel2, false);
      float snr2 = (other2 > 0.1f) ? dark[pk2] / other2 : 99.0f;
      if (snr2 >= OK_SNR) { pk = pk2; rel = rel2; L = L2; R = R2; peak = dark[pk2]; snr = snr2; ok = true; }
    }
  }

  r.contrast = (peak > 0) ? peak / 100.0f : 0.0f;
  r.rel   = rel;
  r.angle = fmodf(gaugeA0(c) + r.rel + 720.0f, 360.0f);
  r.raw   = c.vMin + r.rel / span * (c.vMax - c.vMin);
  r.value = c.corrA * r.raw + c.corrB;
  r.snr   = snr;
  r.width = (R - L + 1) * PROF_STEP;
  r.ok    = ok;
  r.us    = micros() - t0;
  return r.ok;
}
