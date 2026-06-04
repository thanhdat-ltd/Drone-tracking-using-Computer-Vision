"""
FLOW-UFO Drone Communication (E88 Pro / FLOW-UFO-XXXXXX)
---------------------------------------------------------
Protocol: hakimjanov/pyDroneWire

Network:
  Drone WiFi SSID : FLOW-UFO-XXXXXX
  Drone IP        : 192.168.1.1
  Control port    : UDP 7099
  Video stream    : rtsp://192.168.1.1:7070/webcam

Packet format (21 bytes):
  [0x03, 0x66, 0x14, roll, pitch, throttle, yaw, cm , speed,
   0x00×10, xor_checksum, 0x99]

  Stick values  : 0–255, center = 0x80 (128)
  XOR checksum  : XOR of bytes at index 3–18
  Heartbeat     : [0x01, 0x01] at 1 Hz
  Control rate  : 20 Hz
"""

import socket
import threading
import time
import logging

logger = logging.getLogger(__name__)

DRONE_IP   = "192.168.1.1"
DRONE_PORT = 7099
VIDEO_URL  = f"rtsp://{DRONE_IP}:7070/webcam"

HEARTBEAT_INTERVAL = 1.0    # 1 Hz
RC_INTERVAL        = 0.05   # 20 Hz

# Header / trailer
_HDR  = bytes([0x03, 0x66, 0x14])
_TAIL = 0x99

# Command bytes
CMD_NONE      = 0x00
CMD_TAKEOFF   = 0x01   # toggle takeoff / land
CMD_EMERGENCY = 0x02
CMD_FLIP      = 0x08
CMD_HEADLESS  = 0x10
CMD_CALIBRATE = 0x80

_SPEED_BYTE = 0x02     # always 0x02 on wire
_HEARTBEAT  = bytes([0x01, 0x01])


def _stick(v: float) -> int:
    """Convert [-1.0, 1.0] → [0, 255], center = 128."""
    v = max(-1.0, min(1.0, v))
    return int(v * 127) + 128


def _build_packet(roll=0x80, pitch=0x80, throttle=0x80, yaw=0x80,
                  cmd=CMD_NONE) -> bytes:
    body = bytes([roll, pitch, throttle, yaw, cmd, _SPEED_BYTE]) + b'\x00' * 10
    chk = 0
    for b in body:
        chk ^= b
    return _HDR + body + bytes([chk, _TAIL])


class FlowUFODrone:
    """
    Controls a FLOW-UFO drone (E88 Pro) over UDP.

    RC values: floats in [-1.0, 1.0] for each axis.
      roll     : negative = left,  positive = right
      pitch    : negative = back,  positive = forward
      throttle : negative = down,  positive = up
      yaw      : negative = left,  positive = right
    """

    def __init__(self, ip: str = DRONE_IP, port: int = DRONE_PORT,
                 trim_roll: int = 0, trim_pitch: int = -1,
                 trim_throttle: int = 0, trim_yaw: int = 0):
        """
        trim_*: offset byte (−20 đến +20) cộng vào giá trị trung tâm (128)
                khi hover/send_rc gửi 0.0.
                Dương = tăng giá trị stick, âm = giảm.
        Ví dụ: drone hay bay về phía trước → trim_pitch = -5
                (gửi 123 thay vì 128, kéo nhẹ về sau)
        """
        self.ip   = ip
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._flying = False
        self._stop   = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._last_rc = {"roll": 0, "pitch": 0, "throttle": 0, "yaw": 0}
        self._trim = {
            "roll": trim_roll, "pitch": trim_pitch,
            "throttle": trim_throttle, "yaw": trim_yaw,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self._sock.sendto(_HEARTBEAT, (self.ip, self.port))
            self._start_heartbeat()
            logger.info(f"Connected to FLOW-UFO at {self.ip}:{self.port}")
            return True
        except OSError as e:
            logger.error(f"Connect failed: {e}")
            return False

    def disconnect(self):
        self._stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=2)
        self._sock.close()

    def takeoff(self):
        logger.info("Takeoff")
        self._send_cmd_timed(CMD_TAKEOFF, duration=1.0)
        self._flying = True
        # Không block main thread — period ổn định được xử lý ở TrackingSystem._stabilize_until

    def land(self):
        logger.info("Land")
        self.send_rc(0, 0, 0, 0)
        time.sleep(0.2)
        self._send_cmd_timed(CMD_TAKEOFF, duration=0.5)
        self._flying = False

    def emergency_stop(self):
        logger.warning("EMERGENCY STOP")
        pkt = _build_packet(cmd=CMD_EMERGENCY)
        try:
            self._sock.sendto(pkt, (self.ip, self.port))
        except OSError:
            pass
        self._flying = False

    def hover(self):
        self.send_rc(0, 0, 0, 0)

    def send_rc(self, roll: float, pitch: float,
                throttle: float, yaw: float):
        """Send RC command. Values in [-1.0, 1.0]. Dead zone ±0.05."""
        def dz(v: float) -> float:
            return 0.0 if abs(v) < 0.05 else v

        # Áp dụng trim: cộng offset byte vào giá trị center (128)
        # clamp về [0, 255] để tránh overflow
        r = max(0, min(255, _stick(dz(roll))     + self._trim["roll"]))
        p = max(0, min(255, _stick(dz(pitch))    + self._trim["pitch"]))
        t = max(0, min(255, _stick(dz(throttle)) + self._trim["throttle"]))
        y = max(0, min(255, _stick(dz(yaw))      + self._trim["yaw"]))

        self._last_rc = {
            "roll":     int((r - 128) / 1.27),
            "pitch":    int((p - 128) / 1.27),
            "throttle": int((t - 128) / 1.27),
            "yaw":      int((y - 128) / 1.27),
        }
        pkt = _build_packet(roll=r, pitch=p, throttle=t, yaw=y)
        try:
            self._sock.sendto(pkt, (self.ip, self.port))
        except OSError as e:
            logger.warning(f"RC send failed: {e}")

    def flip(self):
        self._send_cmd_timed(CMD_FLIP, duration=0.5)

    def calibrate(self):
        self._send_cmd_timed(CMD_CALIBRATE, duration=0.5)

    def toggle_headless(self):
        self._send_cmd_timed(CMD_HEADLESS, duration=0.5)

    @property
    def is_flying(self) -> bool:
        return self._flying

    @property
    def rc_values(self) -> dict:
        return dict(self._last_rc)

    @property
    def video_url(self) -> str:
        return f"rtsp://{self.ip}:7070/webcam"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_cmd_timed(self, cmd: int, duration: float):
        pkt = _build_packet(cmd=cmd)
        deadline = time.time() + duration
        while time.time() < deadline:
            try:
                self._sock.sendto(pkt, (self.ip, self.port))
            except OSError:
                pass
            time.sleep(RC_INTERVAL)

    def _start_heartbeat(self):
        self._stop.clear()
        self._hb_thread = threading.Thread(
            target=self._hb_loop, daemon=True)
        self._hb_thread.start()

    def _hb_loop(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._sock.sendto(_HEARTBEAT, (self.ip, self.port))
            except OSError:
                pass


# Alias để tương thích với main_tracking.py
E88ProDrone = FlowUFODrone
