// nn_polar.h - 極座標CNNで針の値を読む(05_ml/polar.py, reg2.py, train_polar.py と同じ計算)
// 1) 位置合わせ: log画像の輪郭(|∇log(I+10)|、上限0.5)を基準(学習時のカメラ位置)と正規化相互相関で比べ、平行移動を求める
//    1/2縮小で ±24(2おき)→ ±2 → 元の解像度で ±1
// 2) 極座標: 針の軸(基準の軸+ずれ)を中心に 角度180(2°)×半径32(24〜101.5px)を双線形で読む
//    各リングで d=(中央値−v)/(中央値+8)、角度方向に±5区分の移動中央値を引く
// 3) CNN: 入力 [d×5, cos角, sin角]、角度方向は円環でつなぐ。出力180区分 → 山の±3区分の重み付き平均で角度、山の確率の和=自信
#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include "nn_polar_data.h"   // 重み(PL_*)、基準の輪郭(PR_Q1, PR_Q2)、定数

// ---------- 作業領域 ----------
struct PolarWork {
  float lut[1021];            // log(v/4+10)、v=0..1020(1/2縮小は4画素の和)
  float *h2, *e2;             // 1/2縮小画像と輪郭 [240*320]
  float *e1;                  // 元解像度の輪郭(ROI±2の範囲だけ)
  float *z1, *z2;             // 基準の輪郭(平均を引いたもの)
  double n1, n2;              // 基準のノルム
  float *a, *b;               // CNNの作業 [92160] ×2
  float pol[PL_NR * PL_NA];
  float *b1base;              // 1層目のうち 位置情報(cos,sin)の寄与+バイアス(画像によらず一定なので最初に計算)[16][32][180]
  int lastDx = 0, lastDy = 0; bool haveShift = false; float fullNcc = 0;   // 前回の位置合わせ(追跡用)
  bool ready = false;
};
static int polConv(const float* in, int ci, int H, const float* wt, const float* bs, int co, int kr, int pr, float* out);

static bool polarInit(PolarWork& W, void* (*alloc)(size_t)) {
  for (int v = 0; v <= 1020; v++) W.lut[v] = logf(v / 4.0f + 10.0f);
  W.h2 = (float*)alloc(240 * 320 * 4); W.e2 = (float*)alloc(240 * 320 * 4);
  W.e1 = (float*)alloc((PR_RH + 4) * (PR_RW + 4) * 4);
  W.z1 = (float*)alloc(PR_RH * PR_RW * 4); W.z2 = (float*)alloc((PR_RH / 2) * (PR_RW / 2) * 4);
  W.a = (float*)alloc(92160 * 4); W.b = (float*)alloc(92160 * 4); W.b1base = (float*)alloc(16 * PL_NR * PL_NA * 4);
  if (!W.h2 || !W.e2 || !W.e1 || !W.z1 || !W.z2 || !W.a || !W.b || !W.b1base) return false;
  {  // 位置情報チャンネルだけの1層目(暗さチャンネルを0にして計算)
    for (int j = 0; j < PL_NR; j++) for (int k = 0; k < PL_NA; k++) {
      W.a[(0 * PL_NR + j) * PL_NA + k] = 0.0f;
      W.a[(1 * PL_NR + j) * PL_NA + k] = cosf(k * 2.0f * (float)M_PI / 180.0f);
      W.a[(2 * PL_NR + j) * PL_NA + k] = sinf(k * 2.0f * (float)M_PI / 180.0f);
    }
    polConv(W.a, 3, PL_NR, PL_b1_w, PL_b1_b, 16, 3, 1, W.b1base);
  }
  const float k = PR_ECLIP / 255.0f;
  double m = 0; int n1 = PR_RH * PR_RW, n2 = (PR_RH / 2) * (PR_RW / 2);
  for (int i = 0; i < n1; i++) m += PR_Q1[i] * k;
  m /= n1; W.n1 = 0;
  for (int i = 0; i < n1; i++) { W.z1[i] = (float)(PR_Q1[i] * k - m); W.n1 += (double)W.z1[i] * W.z1[i]; }
  m = 0; for (int i = 0; i < n2; i++) m += PR_Q2[i] * k;
  m /= n2; W.n2 = 0;
  for (int i = 0; i < n2; i++) { W.z2[i] = (float)(PR_Q2[i] * k - m); W.n2 += (double)W.z2[i] * W.z2[i]; }
  W.ready = true; return true;
}

// 正規化相互相関(基準 z [h][w]、対象 e の (xs,ys) から)
// ESP32-S3 は double が遅い(ソフトウェア計算)ので float で積算する。行ごとに部分和を取って誤差を抑える
static float polNcc(const float* z, double nz, int h, int w, const float* e, int ew, int xs, int ys) {
  float s = 0, s2 = 0, p = 0; int n = h * w;
  for (int y = 0; y < h; y++) {
    const float* er = e + (ys + y) * ew + xs; const float* zr = z + y * w;
    float rs = 0, rs2 = 0, rp = 0;
    for (int x = 0; x < w; x++) { float v = er[x]; rs += v; rs2 += v * v; rp += zr[x] * v; }
    s += rs; s2 += rs2; p += rp;
  }
  float var = s2 - s * s / n;      // Σ(v-mean)^2。Σ z(v-mean) = Σ z v(z は平均0)
  return p / (sqrtf((float)nz * fmaxf(var, 0.0f)) + 1e-9f);
}

// 1/2縮小の輪郭を [x0,x1)×[y0,y1) の範囲だけ作る
static void polEdge2(PolarWork& W, const uint8_t* img, int w, int h, int x0, int y0, int x1, int y1) {
  const int W2 = w / 2, H2 = h / 2;
  x0 = std::max(x0 - 1, 0); y0 = std::max(y0 - 1, 0); x1 = std::min(x1 + 1, W2); y1 = std::min(y1 + 1, H2);
  for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
    const uint8_t* p = img + (2 * y) * w + 2 * x;
    W.h2[y * W2 + x] = W.lut[p[0] + p[1] + p[w] + p[w + 1]];      // log(平均+10)
  }
  for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
    float v = 0;
    if (y > y0 && x > x0 && y < y1 - 1 && x < x1 - 1 && y >= 1 && x >= 1 && y < H2 - 1 && x < W2 - 1) {
      float gx = W.h2[y * W2 + x + 1] - W.h2[y * W2 + x - 1], gy = W.h2[(y + 1) * W2 + x] - W.h2[(y - 1) * W2 + x];
      v = fminf(sqrtf(gx * gx + gy * gy), PR_ECLIP);
    }
    W.e2[y * W2 + x] = v;
  }
}

// 位置合わせ。戻り値は相関の値、dx,dy に元解像度のずれ
// full=true: ±48px を探す。false: 前回のずれの近く(1/2縮小で±2)だけ探す(カメラが動いていない前提で速い)
static float polarShift(PolarWork& W, const uint8_t* img, int w, int h, int& dx, int& dy, bool full = true) {
  const int W2 = w / 2, H2 = h / 2;
  const int hh = PR_RH / 2, ww = PR_RW / 2, x0 = PR_RX0 / 2, y0 = PR_RY0 / 2;
  float best = -2; int bx = 0, by = 0;
  auto try2 = [&](int ddx, int ddy) {
    int xs = x0 + ddx, ys = y0 + ddy;
    if (xs < 0 || ys < 0 || xs + ww > W2 || ys + hh > H2) return;
    float s = polNcc(W.z2, W.n2, hh, ww, W.e2, W2, xs, ys);
    if (s > best) { best = s; bx = ddx; by = ddy; }
  };
  if (full || !W.haveShift) {
    polEdge2(W, img, w, h, 0, 0, W2, H2);
    for (int ddy = -PR_SPAN2; ddy <= PR_SPAN2; ddy += 2) for (int ddx = -PR_SPAN2; ddx <= PR_SPAN2; ddx += 2) try2(ddx, ddy);
  } else {
    // 前回の位置(1/2縮小)を中心に。端の差分のため1画素余分に輪郭を作る
    int cx = (int)floorf(W.lastDx / 2.0f), cy = (int)floorf(W.lastDy / 2.0f);
    polEdge2(W, img, w, h, x0 + cx - 5, y0 + cy - 5, x0 + cx + ww + 5, y0 + cy + hh + 5);
    bx = cx; by = cy; best = -2; try2(cx, cy);
  }
  int cx = bx, cy = by;
  for (int ddy = cy - 2; ddy <= cy + 2; ddy++) for (int ddx = cx - 2; ddx <= cx + 2; ddx++) try2(ddx, ddy);
  // 元解像度: ROI の ±(1+1) の範囲だけ輪郭を作る
  const int ex0 = PR_RX0 + 2 * bx - 2, ey0 = PR_RY0 + 2 * by - 2, ew = PR_RW + 4, eh = PR_RH + 4;
  for (int y = 0; y < eh; y++) for (int x = 0; x < ew; x++) {
    int X = ex0 + x, Y = ey0 + y; float v = 0;
    if (X >= 1 && Y >= 1 && X < w - 1 && Y < h - 1) {
      float gx = W.lut[img[Y * w + X + 1] * 4] - W.lut[img[Y * w + X - 1] * 4];
      float gy = W.lut[img[(Y + 1) * w + X] * 4] - W.lut[img[(Y - 1) * w + X] * 4];
      v = fminf(sqrtf(gx * gx + gy * gy), PR_ECLIP);
    }
    W.e1[y * ew + x] = v;
  }
  float b1 = -2; int fx = 2 * bx, fy = 2 * by;
  for (int ddy = 2 * by - 1; ddy <= 2 * by + 1; ddy++) for (int ddx = 2 * bx - 1; ddx <= 2 * bx + 1; ddx++) {
    int xs = PR_RX0 + ddx - ex0, ys = PR_RY0 + ddy - ey0;
    int X = PR_RX0 + ddx, Y = PR_RY0 + ddy;
    if (X < 0 || Y < 0 || X + PR_RW > w || Y + PR_RH > h) continue;
    float s = polNcc(W.z1, W.n1, PR_RH, PR_RW, W.e1, ew, xs, ys);
    if (s > b1) { b1 = s; fx = ddx; fy = ddy; }
  }
  dx = fx; dy = fy;
  W.lastDx = fx; W.lastDy = fy; W.haveShift = true;
  if (full) W.fullNcc = b1;
  return b1;
}

// 極座標の帯と暗さ
static void polarSample(PolarWork& W, const uint8_t* img, int w, int h, float cx, float cy) {
  static float cth[PL_NA], sth[PL_NA]; static bool init = false;
  if (!init) { for (int k = 0; k < PL_NA; k++) { cth[k] = cosf(k * 2.0f * (float)M_PI / 180.0f); sth[k] = sinf(k * 2.0f * (float)M_PI / 180.0f); } init = true; }
  float row[PL_NA], tmp[PL_NA], win[2 * PL_HP + 1];
  for (int j = 0; j < PL_NR; j++) {
    float r = PL_R0 + PL_DR * j;
    for (int k = 0; k < PL_NA; k++) {
      float px = cx + r * cth[k], py = cy + r * sth[k];
      int x0 = (int)floorf(px), y0 = (int)floorf(py); float fx = px - x0, fy = py - y0;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) { row[k] = 0; continue; }
      const uint8_t* p = img + y0 * w + x0;
      row[k] = (p[0] * (1 - fx) + p[1] * fx) * (1 - fy) + (p[w] * (1 - fx) + p[w + 1] * fx) * fy;
    }
    memcpy(tmp, row, sizeof(tmp)); std::sort(tmp, tmp + PL_NA);
    float med = 0.5f * (tmp[PL_NA / 2 - 1] + tmp[PL_NA / 2]);
    float d[PL_NA];
    for (int k = 0; k < PL_NA; k++) d[k] = (med - row[k]) / (med + 8.0f);
    for (int k = 0; k < PL_NA; k++) {
      for (int t = -PL_HP; t <= PL_HP; t++) win[t + PL_HP] = d[(k + t + PL_NA) % PL_NA];
      std::sort(win, win + 2 * PL_HP + 1);
      W.pol[j * PL_NA + k] = d[k] - win[PL_HP];
    }
  }
}

// 畳み込み(半径 kr×角度3、半径方向はゼロ埋め pr、角度方向は円環)+ReLU。in [ci][H][NA] → out [co][H+2pr-kr+1][NA]
static int polConv(const float* in, int ci, int H, const float* wt, const float* bs, int co, int kr, int pr, float* out) {
  const int NA = PL_NA, Ho = H + 2 * pr - kr + 1;
  for (int o = 0; o < co; o++) for (int y = 0; y < Ho; y++) {
    float* op = out + (o * Ho + y) * NA;
    for (int k = 0; k < NA; k++) op[k] = bs[o];
    for (int c = 0; c < ci; c++) for (int ky = 0; ky < kr; ky++) {
      int iy = y + ky - pr; if (iy < 0 || iy >= H) continue;
      const float* ip = in + (c * H + iy) * NA;
      const float* wp = wt + ((o * ci + c) * kr + ky) * 3;
      float w0 = wp[0], w1 = wp[1], w2 = wp[2];
      op[0] += w0 * ip[NA - 1] + w1 * ip[0] + w2 * ip[1];
      for (int k = 1; k < NA - 1; k++) op[k] += w0 * ip[k - 1] + w1 * ip[k] + w2 * ip[k + 1];
      op[NA - 1] += w0 * ip[NA - 2] + w1 * ip[NA - 1] + w2 * ip[0];
    }
  }
  return Ho;
}
static void polRelu(float* x, int n) { for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0; }
static int polPool(const float* in, int c, int H, float* out) {   // 半径方向に2つずつ最大
  const int NA = PL_NA, Ho = H / 2;
  for (int ch = 0; ch < c; ch++) for (int y = 0; y < Ho; y++) for (int k = 0; k < NA; k++)
    out[(ch * Ho + y) * NA + k] = fmaxf(in[(ch * H + 2 * y) * NA + k], in[(ch * H + 2 * y + 1) * NA + k]);
  return Ho;
}

struct PolarResult { float value, conf, ncc; int dx, dy; };

// 1枚を読む。value[MPa]、conf(自信 0〜1)、位置ずれ
// full=false なら前回の位置の近くだけ探す。相関が全探索時の0.7倍未満になったら全探索し直す
static PolarResult polarRead(PolarWork& W, const uint8_t* img, int w, int h, bool full = true) {
  PolarResult R{NAN, 0, 0, 0, 0};
  R.ncc = polarShift(W, img, w, h, R.dx, R.dy, full);
  if (!full && R.ncc < 0.7f * W.fullNcc) R.ncc = polarShift(W, img, w, h, R.dx, R.dy, true);
  polarSample(W, img, w, h, PL_CX + R.dx, PL_CY + R.dy);
  const int NA = PL_NA, NR = PL_NR;
  float* a = W.a; float* b = W.b;
  for (int i = 0; i < NR * NA; i++) a[i] = W.pol[i] * 5.0f;
  // 1層目: 位置情報の寄与(b1base)に、暗さチャンネルの寄与だけを足す
  memcpy(b, W.b1base, 16 * NR * NA * 4);
  for (int o = 0; o < 16; o++) for (int y = 0; y < NR; y++) {
    float* op = b + (o * NR + y) * NA;
    for (int ky = 0; ky < 3; ky++) {
      int iy = y + ky - 1; if (iy < 0 || iy >= NR) continue;
      const float* ip = a + iy * NA; const float* wp = PL_b1_w + ((o * 3 + 0) * 3 + ky) * 3;
      float w0 = wp[0], w1 = wp[1], w2 = wp[2];
      op[0] += w0 * ip[NA - 1] + w1 * ip[0] + w2 * ip[1];
      for (int k = 1; k < NA - 1; k++) op[k] += w0 * ip[k - 1] + w1 * ip[k] + w2 * ip[k + 1];
      op[NA - 1] += w0 * ip[NA - 2] + w1 * ip[NA - 1] + w2 * ip[0];
    }
  }
  int H = NR; polRelu(b, 16 * H * NA); H = polPool(b, 16, H, a);   // 16x16
  H = polConv(a, 16, H, PL_b2_w, PL_b2_b, 16, 3, 1, b); polRelu(b, 16 * H * NA); H = polPool(b, 16, H, a);       // 16x8
  H = polConv(a, 16, H, PL_b3_w, PL_b3_b, 32, 3, 1, b); polRelu(b, 32 * H * NA); H = polPool(b, 32, H, a);       // 32x4
  H = polConv(a, 32, H, PL_b4_w, PL_b4_b, 32, 4, 0, b); polRelu(b, 32 * H * NA);                                 // 32x1
  H = polConv(b, 32, H, PL_b5_w, PL_b5_b, 1, 1, 0, a);                                                           // 1x1x180
  float mx = a[0]; int km = 0;
  for (int k = 1; k < NA; k++) if (a[k] > mx) { mx = a[k]; km = k; }
  double sum = 0; static float p[PL_NA];
  for (int k = 0; k < NA; k++) { p[k] = expf(a[k] - mx); sum += p[k]; }
  double ss = 0, sc = 0, sw = 0;
  for (int t = -3; t <= 3; t++) {
    int k = (km + t + NA) % NA; double wgt = p[k] / sum; double th = k * 2.0 * M_PI / 180.0;
    ss += wgt * sin(th); sc += wgt * cos(th); sw += wgt;
  }
  double ang = fmod(atan2(ss, sc) * 180.0 / M_PI + 360.0, 360.0);
  double rel = fmod(ang - PL_A0 + 720.0, 360.0); if (rel > 320) rel -= 360;
  R.value = (float)(rel / PL_SPAN); R.conf = (float)sw;
  return R;
}

// ---------- 射影変換の補正(カメラ位置が学習時と違うとき) ----------
// m: 基準(学習時の見え方)の座標 → 今の画像の座標 への 3x3 行列(行優先)。PC の hom_check.py がクリック4点以上から求める
// 今の画像を学習時の見え方に変換して dst に書く(双線形、画像の外は端の画素で埋める = OpenCV の BORDER_REPLICATE と同じ)
static void polarWarp(const uint8_t* src, int w, int h, const float* m, uint8_t* dst) {
  for (int y = 0; y < h; y++) {
    float nx = m[1] * y + m[2], ny = m[4] * y + m[5], nz = m[7] * y + m[8];   // x=0 のとき。x が1増えるごとに m[0], m[3], m[6] を足す
    uint8_t* d = dst + y * w;
    for (int x = 0; x < w; x++, nx += m[0], ny += m[3], nz += m[6]) {
      float iz = 1.0f / nz, px = nx * iz, py = ny * iz;
      px = fminf(fmaxf(px, 0.0f), (float)(w - 1)); py = fminf(fmaxf(py, 0.0f), (float)(h - 1));
      int x0 = (int)px, y0 = (int)py; float fx = px - x0, fy = py - y0;
      int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
      const uint8_t* r0 = src + y0 * w; const uint8_t* r1 = src + y1 * w;
      float v = (r0[x0] * (1 - fx) + r0[x1] * fx) * (1 - fy) + (r1[x0] * (1 - fx) + r1[x1] * fx) * fy;
      d[x] = (uint8_t)(v + 0.5f);
    }
  }
}
