// =====================================================================
// motor_check.ino  基板・配線・モーターの生存確認用(単体スケッチ)
// 28BYJ-48 + ULN2003 を 5° ずつ ゆっくり往復させる
//
// 確認できること
//   - UNO内蔵LED(D13)が 1秒周期で点滅 → UNO は動作している
//   - シリアルモニタ(115200bps)に状態が表示される → USB通信は正常
//   - ULN2003 の LED(A〜D)が順番に点灯 → UNO→ドライバの配線は正常
//   - 軸が +5°/-5° で往復 → ドライバ→モーター・電源は正常
// =====================================================================

const uint8_t PIN_IN1 = 8;
const uint8_t PIN_IN2 = 9;
const uint8_t PIN_IN3 = 10;
const uint8_t PIN_IN4 = 11;
const uint8_t PIN_LED = 13;

const float STEPS_PER_REV = 4096.0;   // 半ステップ駆動
const float SWING_DEG     = 5.0;      // 往復の角度
const unsigned int STEP_MS = 10;      // 1半ステップの間隔 [ms](大きいほど遅い。LEDの点灯順も目視できる)
const unsigned int PAUSE_MS = 1000;   // 往復の端での停止時間 [ms]

const uint8_t SEQ[8][4] = {
  {1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},
  {0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}
};
int8_t phase = 0;

void writeCoils(int8_t p) {
  digitalWrite(PIN_IN1, SEQ[p][0]);
  digitalWrite(PIN_IN2, SEQ[p][1]);
  digitalWrite(PIN_IN3, SEQ[p][2]);
  digitalWrite(PIN_IN4, SEQ[p][3]);
}

void moveSteps(long n) {
  int8_t dir = (n >= 0) ? 1 : -1;
  for (long i = 0; i < labs(n); i++) {
    phase = (phase + dir + 8) % 8;
    writeCoils(phase);
    delay(STEP_MS);
  }
}

void blinkPause(unsigned int ms) {
  // 停止中もLEDを点滅させ、UNOが止まっていないことを示す
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    digitalWrite(PIN_LED, (millis() / 250) % 2);
    delay(10);
  }
  digitalWrite(PIN_LED, LOW);
}

void setup() {
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  Serial.begin(115200);
  Serial.println(F("motor_check start"));
  writeCoils(phase);
}

void loop() {
  long n = lround(STEPS_PER_REV * SWING_DEG / 360.0);  // 5° ≒ 57 半ステップ

  Serial.print(F("+")); Serial.print(SWING_DEG, 1); Serial.print(F(" deg  steps=")); Serial.println(n);
  digitalWrite(PIN_LED, HIGH);
  moveSteps(n);
  blinkPause(PAUSE_MS);

  Serial.print(F("-")); Serial.print(SWING_DEG, 1); Serial.print(F(" deg  steps=")); Serial.println(-n);
  digitalWrite(PIN_LED, HIGH);
  moveSteps(-n);
  blinkPause(PAUSE_MS);
}
