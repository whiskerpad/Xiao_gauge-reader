// =====================================================================
// needle_stepper.ino
// 疑似圧力計の指針駆動(28BYJ-48 + ULN2003)
// 対象: Arduino UNO / Nano 等(ピン定数を変えれば他MCUも可)
//
// ラズパイ(またはPC)からUSBシリアルで指令を受け、指定角度・圧力値へ針を移動する。
// 1指令ごとに必ず "OK ..." または "ERR ..." を1行返す(上位側はこれを待って撮影する)。
//
// 指令(115200bps, 改行区切り, 大文字小文字不問)
//   HOME          ストッパーへ押し当てて原点化し、0目盛りへ移動
//   DEG <角度>    0〜SCALE_DEG [deg] へ移動      例: DEG 135
//   MPA <圧力>    0〜SCALE_MAX_MPA [MPa] へ移動  例: MPA 0.45
//   STEP <n>      相対移動(半ステップ数、校正用。ガタ取りなし)
//   POS           現在位置を返す
//   ZERO          現在位置を0目盛りとして登録(校正用)
//   FULL          現在位置を最大目盛りとして登録(校正用)
//   SWEEP <刻み>  0→最大→0 を指定角度刻みで往復(単体動作確認用)
//   OFF           コイル通電を切る(原点情報は破棄)
//   SLEW <圧力> <速さ>  指定圧力へ一定の速さ[MPa/秒]で滑らかに移動(加減速あり)。すぐ OK を返し、移動は裏で続く
//                       移動中も POS で現在位置を返す。移動中に別の SLEW を送ると、速さ・目標を滑らかに切り替える
//                       (ガタ取りの寄せは行わない。上昇中は指令位置より針がガタ分遅れる)
//   STOP          SLEW の移動を減速して止める
//   ?             指令一覧
// =====================================================================

#include <Arduino.h>

// ---------------- 配線 ----------------
const uint8_t PIN_IN1 = 8;
const uint8_t PIN_IN2 = 9;
const uint8_t PIN_IN3 = 10;
const uint8_t PIN_IN4 = 11;

// ---------------- モーター ----------------
const float STEPS_PER_REV = 4096.0;      // 28BYJ-48 半ステップ駆動の公称値(実測は約4076)
const bool  INVERT_DIR    = true;       // 目盛りが増える方向に回らない場合 true

// ---------------- 目盛り ----------------
const float SCALE_DEG     = 270.0;       // 0〜最大目盛りの振れ角
const float SCALE_MAX_MPA = 1.0;         // 最大目盛りの圧力値

// ---------------- 校正値(ZERO / FULL 指令の結果をここへ転記) ----------------
const long  ZERO_OFFSET_STEPS = 0;       // ストッパー位置から0目盛りまでの半ステップ数
//long        fullSteps = (long)(STEPS_PER_REV * SCALE_DEG / 360.0 + 0.5); // 0→最大目盛りの半ステップ数
long        fullSteps = 3094; // 0→最大目盛りの半ステップ数(校正値)
// ---------------- 動作 ----------------
const unsigned int  STEP_US_MOVE   = 1200;  // 移動時の半ステップ周期 [us](小さいほど速い。脱調したら増やす)
const unsigned int  STEP_US_HOME   = 3000;  // 原点復帰時の周期 [us](押し当てるので低速)
const float         HOME_EXTRA_DEG = 100.0;  // 原点復帰で振れ角に追加して戻す量
const long          BACKLASH_STEPS = 40;    // ギアのガタ取り量(最終の寄せ方向を常に「減少側」に揃える)
const unsigned long SETTLE_MS      = 300;   // 停止後、OKを返すまでの静止待ち
const bool          RELEASE_AFTER_MOVE = false; // 移動後に通電を切るか(切ると発熱減、保持力なし)

// ---------------- 内部状態 ----------------
const uint8_t SEQ[8][4] = {
  {1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},
  {0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}
};
int8_t phase = 0;          // 励磁パターン 0〜7
long   posSteps = 0;       // 0目盛りを0とした現在位置 [半ステップ]
long   stopperSteps = 0;   // ストッパー位置(posSteps座標系)
bool   homed = false;

// ---- SLEW(滑らかな連続移動、2026-11-17) ----
const float SLEW_ACCEL = 60.0;       // 加速度 [半ステップ/秒^2](約0.02 MPa/秒^2)
const float SLEW_VMAX  = 500.0;      // 最高速度 [半ステップ/秒]
bool   slewOn = false;
long   slewTarget = 0;
float  slewVmax = 0;                 // 巡航速度 [半ステップ/秒]
float  slewV = 0;                    // 現在速度(符号付き)
float  slewAcc = 0;                  // 端数ステップの積算
unsigned long slewT = 0;

char   lineBuf[48];
uint8_t lineLen = 0;

// ---------------------------------------------------------------------
void writeCoils(int8_t p) {
  digitalWrite(PIN_IN1, SEQ[p][0]);
  digitalWrite(PIN_IN2, SEQ[p][1]);
  digitalWrite(PIN_IN3, SEQ[p][2]);
  digitalWrite(PIN_IN4, SEQ[p][3]);
}

void releaseCoils() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
}

// dir: +1 = 目盛り増加方向, -1 = 減少方向
void stepOnce(int8_t dir, unsigned int us) {
  int8_t d = INVERT_DIR ? -dir : dir;
  phase = (phase + d + 8) % 8;
  writeCoils(phase);
  posSteps += dir;
  delayMicroseconds(us);
}

void moveRelative(long n, unsigned int us) {
  int8_t dir = (n >= 0) ? 1 : -1;
  for (long i = 0; i < labs(n); i++) stepOnce(dir, us);
}

// 目標位置へ移動。最終の寄せは常に減少方向で行い、ガタによる誤差を一定にする
void moveToSteps(long target) {
  writeCoils(phase);  // 通電再開
  if (target > posSteps) {
    moveRelative(target + BACKLASH_STEPS - posSteps, STEP_US_MOVE);
  }
  moveRelative(target - posSteps, STEP_US_MOVE);
  delay(SETTLE_MS);
  if (RELEASE_AFTER_MOVE) releaseCoils();
}

float stepsToDeg(long s) { return (float)s * SCALE_DEG / (float)fullSteps; }
long  degToSteps(float d) { return lround(d * (float)fullSteps / SCALE_DEG); }
float degToMpa(float d)   { return d * SCALE_MAX_MPA / SCALE_DEG; }
float mpaToDeg(float m)   { return m * SCALE_DEG / SCALE_MAX_MPA; }

// SLEW を即座に止める(他の移動指令の前に呼ぶ)
void slewAbort() { slewOn = false; slewV = 0; slewAcc = 0; }

// loop から常に呼ぶ。速度を目標へ近づけ、積算が1ステップを超えたら1ステップ進める
void slewUpdate() {
  unsigned long now = micros();
  float dt = (now - slewT) * 1e-6f;
  slewT = now;
  if (!slewOn) return;
  if (dt > 0.05f) dt = 0.05f;
  long dist = slewTarget - posSteps;
  // 目標で止まれる速度(減速距離から)と巡航速度の小さい方
  float vStop = sqrtf(2.0f * SLEW_ACCEL * (float)labs(dist));
  float vDes = (dist > 0 ? 1 : (dist < 0 ? -1 : 0)) * min(slewVmax, vStop);
  float dv = SLEW_ACCEL * dt;
  if (slewV < vDes) slewV = min(slewV + dv, vDes); else slewV = max(slewV - dv, vDes);
  slewAcc += slewV * dt;
  if (slewAcc >= 1.0f)  { stepOnce(1, 0);  slewAcc -= 1.0f; }
  if (slewAcc <= -1.0f) { stepOnce(-1, 0); slewAcc += 1.0f; }
  if (dist == 0 && fabsf(slewV) < 2.0f) slewAbort();
}

void printPos(const __FlashStringHelper* head) {
  float deg = stepsToDeg(posSteps);
  Serial.print(head);
  Serial.print(F(" deg="));   Serial.print(deg, 2);
  Serial.print(F(" mpa="));   Serial.print(degToMpa(deg), 4);
  Serial.print(F(" steps=")); Serial.println(posSteps);
}

// ---------------------------------------------------------------------
void cmdHome() {
  long back = degToSteps(SCALE_DEG + HOME_EXTRA_DEG);
  writeCoils(phase);
  moveRelative(-back, STEP_US_HOME);   // ストッパーで脱調しながら押し当て
  delay(200);                          // 押し当て状態で励磁を保持し位置を落ち着かせる
  posSteps = -ZERO_OFFSET_STEPS;       // ストッパー位置を座標化
  stopperSteps = posSteps;
  homed = true;
  moveToSteps(0);
  printPos(F("OK HOME"));
}

void cmdMoveDeg(float deg) {
  if (!homed) { Serial.println(F("ERR NOT_HOMED")); return; }
  if (deg < 0.0 || deg > SCALE_DEG) { Serial.println(F("ERR RANGE")); return; }
  moveToSteps(degToSteps(deg));
  printPos(F("OK MOVE"));
}

void cmdSweep(float pitch) {
  if (!homed) { Serial.println(F("ERR NOT_HOMED")); return; }
  if (pitch <= 0.0) pitch = 10.0;
  for (float d = 0.0; d <= SCALE_DEG + 0.001; d += pitch) {
    moveToSteps(degToSteps(d)); printPos(F("SWEEP")); delay(300);
  }
  for (float d = SCALE_DEG; d >= -0.001; d -= pitch) {
    moveToSteps(degToSteps(d)); printPos(F("SWEEP")); delay(300);
  }
  printPos(F("OK SWEEP"));
}

void cmdZero() {
  if (!homed) { Serial.println(F("ERR NOT_HOMED")); return; }
  long offset = posSteps - stopperSteps;   // ストッパーから現在位置までの距離
  stopperSteps = -offset;
  posSteps = 0;
  Serial.print(F("OK ZERO ZERO_OFFSET_STEPS=")); Serial.println(offset);
}

void cmdFull() {
  if (!homed || posSteps <= 0) { Serial.println(F("ERR FULL")); return; }
  fullSteps = posSteps;
  Serial.print(F("OK FULL fullSteps=")); Serial.println(fullSteps);
}

void cmdSlew(char* a1, char* a2) {
  if (!homed) { Serial.println(F("ERR NOT_HOMED")); return; }
  if (!a1 || !a2) { Serial.println(F("ERR ARG")); return; }
  float m = atof(a1), v = atof(a2);
  if (m < 0.0 || m > SCALE_MAX_MPA) { Serial.println(F("ERR RANGE")); return; }
  float vs = v / SCALE_MAX_MPA * (float)fullSteps;          // MPa/秒 → 半ステップ/秒
  if (vs < 0.5f) vs = 0.5f;
  if (vs > SLEW_VMAX) vs = SLEW_VMAX;
  if (!slewOn) { writeCoils(phase); slewV = 0; slewAcc = 0; slewT = micros(); }
  slewTarget = degToSteps(mpaToDeg(m));
  slewVmax = vs;
  slewOn = true;
  printPos(F("OK SLEW"));
}

void cmdStop() {
  if (slewOn) {
    // 今の速度から減速して止まれる位置を新しい目標にする
    long d = (long)(slewV * slewV / (2.0f * SLEW_ACCEL) + 0.5f);
    slewTarget = posSteps + (slewV >= 0 ? d : -d);
  }
  printPos(F("OK STOP"));
}

void cmdHelp() {
  Serial.println(F("HOME | DEG <d> | MPA <m> | STEP <n> | POS | ZERO | FULL | SWEEP <pitch> | OFF | SLEW <m> <mpa/s> | STOP"));
  Serial.println(F("OK HELP"));
}

void handleLine(char* s) {
  for (char* p = s; *p; p++) *p = toupper(*p);
  char* cmd = strtok(s, " \t");
  char* arg = strtok(NULL, " \t");
  char* arg2 = strtok(NULL, " \t");
  if (!cmd) return;
  // SLEW / STOP / POS 以外の指令は、連続移動を止めてから実行する
  if (strcmp(cmd, "SLEW") && strcmp(cmd, "STOP") && strcmp(cmd, "POS")) slewAbort();

  if      (!strcmp(cmd, "HOME"))  cmdHome();
  else if (!strcmp(cmd, "DEG"))   { if (arg) cmdMoveDeg(atof(arg)); else Serial.println(F("ERR ARG")); }
  else if (!strcmp(cmd, "MPA"))   { if (arg) cmdMoveDeg(mpaToDeg(atof(arg))); else Serial.println(F("ERR ARG")); }
  else if (!strcmp(cmd, "STEP"))  {
    if (!homed) { Serial.println(F("ERR NOT_HOMED")); return; }
    if (!arg)   { Serial.println(F("ERR ARG")); return; }
    writeCoils(phase); moveRelative(atol(arg), STEP_US_MOVE); delay(SETTLE_MS);
    printPos(F("OK STEP"));
  }
  else if (!strcmp(cmd, "POS"))   { if (homed) printPos(F("OK POS")); else Serial.println(F("ERR NOT_HOMED")); }
  else if (!strcmp(cmd, "ZERO"))  cmdZero();
  else if (!strcmp(cmd, "FULL"))  cmdFull();
  else if (!strcmp(cmd, "SWEEP")) cmdSweep(arg ? atof(arg) : 10.0);
  else if (!strcmp(cmd, "OFF"))   { releaseCoils(); homed = false; Serial.println(F("OK OFF")); }
  else if (!strcmp(cmd, "SLEW"))  cmdSlew(arg, arg2);
  else if (!strcmp(cmd, "STOP"))  cmdStop();
  else if (!strcmp(cmd, "?"))     cmdHelp();
  else Serial.println(F("ERR UNKNOWN"));
}

// ---------------------------------------------------------------------
void setup() {
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  releaseCoils();
  Serial.begin(115200);
  Serial.println(F("READY needle_stepper"));
}

void loop() {
  slewUpdate();
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineLen > 0) handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}
