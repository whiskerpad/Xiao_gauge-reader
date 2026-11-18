"""
needle.py - 疑似圧力計(needle_stepper.ino)をシリアルで操作するクラス
ラズパイ/PC共通。後段の自動撮影スクリプトから import して使う。

  pip install pyserial
"""
import time
import serial


class NeedleError(RuntimeError):
    pass


class Needle:
    def __init__(self, port, baud=115200, timeout=60.0):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.timeout = timeout
        time.sleep(2.0)                    # UNOはポートを開くとリセットされる
        self._wait_line("READY", 10.0)

    def close(self):
        self.ser.close()

    def _wait_line(self, prefix, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith(prefix):
                return line
            if line.startswith("ERR"):
                raise NeedleError(line)
        raise NeedleError(f"timeout waiting {prefix}")

    def cmd(self, text):
        """指令を送り、OK行を解析して dict で返す"""
        self.ser.reset_input_buffer()
        self.ser.write((text + "\n").encode())
        line = self._wait_line("OK", self.timeout)
        info = {"raw": line}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    info[k] = float(v)
                except ValueError:
                    info[k] = v
        return info

    def home(self):          return self.cmd("HOME")
    def move_deg(self, deg): return self.cmd(f"DEG {deg:.2f}")
    def move_mpa(self, mpa): return self.cmd(f"MPA {mpa:.4f}")
    def step(self, n):       return self.cmd(f"STEP {int(n)}")
    def pos(self):           return self.cmd("POS")
    def zero(self):          return self.cmd("ZERO")
    def full(self):          return self.cmd("FULL")
    def off(self):           return self.cmd("OFF")
