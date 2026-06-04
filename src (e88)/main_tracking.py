"""
Drone Person Tracking — E88 Pro + YOLOv8 Pose 5 keypoints
==========================================================
Keypoints (custom model): 0=mũi  1=vai_trái  2=vai_phải  3=hông_trái  4=hông_phải

Chạy:
  python main_tracking.py                          # RTSP drone + best.pt (khuyên dùng)
  python main_tracking.py --no-fly                 # test không cần drone (xem dashboard)
  python main_tracking.py --source 0 --flip        # webcam (test bàn làm việc)
  python main_tracking.py --model yolov8n-pose.pt  # đổi model

Phím tắt:
  T : Takeoff / Land (toggle)
  H : Hover (giữ yên)
  S : Safety mode ON/OFF (khi OFF: drone không tự hover khi mất tracking)
  E : Emergency stop (tắt motor ngay)
  Q : Thoát + hạ cánh an toàn
"""

import argparse
import logging
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from pid_controller import DroneController
    from drone_comm import E88ProDrone, VIDEO_URL
except ImportError:
    logger.error("Không tìm thấy pid_controller.py hoặc drone_comm.py — "
                 "đảm bảo 3 file nằm cùng thư mục.")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Chưa cài ultralytics: pip install ultralytics")
    sys.exit(1)

# ── Keypoint indices ─────────────────────────────────────────────────────────
# 5-kp custom model (best.pt): nose=0, l_shoul=1, r_shoul=2, l_hip=3, r_hip=4
# 17-kp COCO standard (yolov8*-pose.pt): nose=0, l_shoul=5, r_shoul=6, l_hip=11, r_hip=12
# → tự động detect khi chạy (xem _detect)
KP_INDICES_5KP  = (0, 1, 2, 3, 4)
KP_INDICES_COCO = (0, 5, 6, 11, 12)

# ── Config ────────────────────────────────────────────────────────────────────
DET_CONF    = 0.40   # ngưỡng confidence detection (thấp hơn để bắt tư thế xoay lưng)
KP_CONF     = 0.30   # ngưỡng confidence từng keypoint
INFER_SKIP  = 2      # chạy YOLO mỗi N frame (1 = mọi frame, 2 = cách 1 frame, ...)

LOST_HOVER_FRAMES  = 15    # mất tracking N frame → hover, giữ lock
LOST_UNLOCK_FRAMES = 100   # mất tracking N frame → xóa lock, tìm đối tượng mới (SEARCHING)
SEARCH_LAND_FRAMES = 80    # trong SEARCHING không thấy ai N frame → tự hạ cánh

YAW_DEADBAND_PX      = 20   # sai số X < giá trị này → không xoay  (triệt jitter ngang)
THROTTLE_DEADBAND_PX = 10   # sai số Y < giá trị này → không lên/xuống (triệt jitter dọc)
DIST_DEADBAND_PX     = 15   # dist_error < giá trị này → không tiến/lùi (triệt jitter khoảng cách)

RC_SEND_INTERVAL  = 0.1    # giây giữa mỗi lần gửi lệnh RC (0.1 = 10Hz, 0.2 = 5Hz)

IOU_LOCK_MIN      = 0.25   # IoU tối thiểu để xác nhận vẫn là người đang tracking

# ── Màu HUD ───────────────────────────────────────────────────────────────────
C_OK     = (80,  200, 100)   # xanh lá
C_WARN   = (50,  180, 240)   # vàng cam
C_DANGER = (60,  60,  220)   # đỏ
C_TEXT   = (230, 230, 230)
C_DIM    = (130, 130, 130)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


# ════════════════════════════════════════════════════════════════════════════
class FrameGrabber(threading.Thread):
    """
    Thread riêng đọc frame từ RTSP/webcam.

    Lý do cần thread riêng: cv2.VideoCapture.read() với RTSP có thể block
    20–100ms khi drone đang encode. Nếu chạy trong main loop, mọi xử lý
    (YOLO inference, PID, display) đều bị trì hoãn theo.

    Thread này liên tục đọc và giữ frame mới nhất. Main loop chỉ cần gọi
    read() để lấy frame ngay lập tức, không cần chờ network.
    Tự động reconnect khi stream bị ngắt (WiFi drone hay camera khởi động lại).
    """
    RECONNECT_DELAY = 2.0   # giây chờ giữa các lần reconnect

    def __init__(self, source, flip: bool = False):
        super().__init__(daemon=True, name="frame-grabber")
        self.source      = source
        self.flip        = flip
        self._lock       = threading.Lock()
        self._frame      = None
        self._ok         = False
        self._stop_evt   = threading.Event()
        self.reconnects  = 0

    def _open_cap(self):
        """Mở VideoCapture với transport settings phù hợp."""
        if isinstance(self.source, str) and self.source.startswith("rtsp"):
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|max_delay;500000|reorder_queue_size;0"
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # giảm latency buffer
        return cap if cap.isOpened() else None

    def run(self):
        while not self._stop_evt.is_set():
            cap = self._open_cap()
            if cap is None:
                logger.warning("Không mở được stream '%s', thử lại sau %.1fs",
                               self.source, self.RECONNECT_DELAY)
                self._stop_evt.wait(self.RECONNECT_DELAY)
                self.reconnects += 1
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Stream OK: %s  (%dx%d)", self.source, w, h)

            while not self._stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Mất stream → reconnecting...")
                    self.reconnects += 1
                    with self._lock:
                        self._ok = False
                    break
                if self.flip:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame
                    self._ok    = True

            cap.release()

    def read(self):
        """Trả về (stream_ok, frame | None). Thread-safe."""
        with self._lock:
            return self._ok, (self._frame.copy() if self._frame is not None else None)

    def stop(self):
        self._stop_evt.set()


# ════════════════════════════════════════════════════════════════════════════
class KeypointAnalyzer:
    """
    Chuyển đổi 5 raw keypoints → các giá trị điều khiển drone.

    Output chính:
      error_x       : pixels, người lệch trái/phải so với tâm frame
      error_y       : pixels, người lệch lên/xuống so với tâm frame
      shoulder_span : pixels, khoảng cách 2 vai (ước lượng khoảng cách người-drone)
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cx = frame_w / 2.0
        self.cy = frame_h / 2.0

    def analyze(self, kp_xy: np.ndarray, kp_conf: np.ndarray,
                kp_indices: tuple = KP_INDICES_5KP) -> dict | None:
        """
        kp_xy      : (N, 2) — tọa độ pixel của N keypoints
        kp_conf    : (N,)   — confidence của từng keypoint
        kp_indices : (nose, l_shoul, r_shoul, l_hip, r_hip)

        Trả về None nếu không đủ keypoints quan trọng (2 vai bắt buộc phải có).
        """
        idx_nose, idx_ls, idx_rs, idx_lh, idx_rh = kp_indices

        def get(idx):
            """Lấy keypoint nếu confidence đủ ngưỡng."""
            if idx < len(kp_conf) and kp_conf[idx] >= KP_CONF:
                return kp_xy[idx].astype(float)
            return None

        nose = get(idx_nose)
        ls   = get(idx_ls)
        rs   = get(idx_rs)
        lh   = get(idx_lh)
        rh   = get(idx_rh)

        # 2 vai là bắt buộc — dùng để tính khoảng cách và orientation
        if ls is None or rs is None:
            return None

        valid   = [p for p in [nose, ls, rs, lh, rh] if p is not None]
        centroid = np.mean(valid, axis=0)

        shoulder_span = float(np.linalg.norm(ls - rs))
        shoulder_mid  = (ls + rs) / 2.0

        # Orientation: góc đường vai với trục ngang (0° = vai song song, ±90° = nghiêng)
        orientation = float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))

        # Facing: mũi phía trên vai → nhìn vào camera; phía dưới → quay lưng
        facing = "back" if (nose is not None and nose[1] > shoulder_mid[1] + 20) else "front"

        # Chiều cao thân (nose → hip): bất biến khi người đi ngang, dùng cho pitch distance
        body_height = None
        if nose is not None:
            hip_ys = [p[1] for p in [lh, rh] if p is not None]
            if hip_ys:
                body_height = max(0.0, float(np.mean(hip_ys)) - nose[1])

        return {
            # Dùng cho điều khiển
            "error_x":       float(centroid[0] - self.cx),   # + = lệch phải
            "error_y":       float(centroid[1] - self.cy),   # + = lệch xuống
            "shoulder_span": shoulder_span,
            "body_height":   body_height,  # None nếu thiếu nose/hip
            # Metadata
            "centroid":      centroid,
            "shoulder_mid":  shoulder_mid,
            "orientation":   orientation,
            "facing":        facing,
            "valid_kps":     len(valid),
            # Raw points để vẽ skeleton
            "nose":          nose,
            "left_shoul":    ls,
            "right_shoul":   rs,
            "left_hip":      lh,
            "right_hip":     rh,
        }


# ════════════════════════════════════════════════════════════════════════════
class Overlay:
    """Vẽ HUD, skeleton, và thông số PID lên frame OpenCV."""

    # Khung xương: các cặp keypoint nối với nhau
    _SKELETON = [
        ("left_shoul",  "right_shoul"),
        ("left_shoul",  "left_hip"),
        ("right_shoul", "right_hip"),
        ("left_hip",    "right_hip"),
    ]
    _KP_COLORS = {
        "nose":        (255, 220,  80),
        "left_shoul":  ( 80, 160, 255),
        "right_shoul": ( 80, 160, 255),
        "left_hip":    (180,  80, 255),
        "right_hip":   (180,  80, 255),
    }

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self._fps_buf = deque(maxlen=30)
        self._last_t  = time.time()

    def tick_fps(self):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-4))
        self._last_t = now

    @property
    def fps(self) -> float:
        return sum(self._fps_buf) / max(len(self._fps_buf), 1)

    def draw(self, frame: np.ndarray, analysis: dict | None,
             pid_out: dict, rc_vals: dict, state: str,
             safety: bool, lost_frames: int,
             stream_ok: bool, reconnects: int) -> np.ndarray:
        out = frame.copy()
        cx, cy = self.w // 2, self.h // 2

        # Crosshair tâm frame
        cv2.line(out, (cx - 25, cy), (cx + 25, cy), C_DIM, 1)
        cv2.line(out, (cx, cy - 25), (cx, cy + 25), C_DIM, 1)

        if analysis:
            self._draw_skeleton(out, analysis)
            self._draw_centroid(out, analysis["centroid"])
            self._draw_error_bar(out, analysis["error_x"], analysis["error_y"])

        self._draw_state_panel(out, state, safety, lost_frames, stream_ok, reconnects)
        self._draw_pid_panel(out, pid_out, rc_vals)

        fps_color = C_OK if self.fps > 20 else C_WARN
        cv2.putText(out, f"FPS {self.fps:.1f}", (10, 22), FONT, 0.55, fps_color, 1)

        if analysis is None and lost_frames > 0:
            self._draw_lost_warning(out, lost_frames, state)

        return out

    def _draw_skeleton(self, f, a):
        for k1, k2 in self._SKELETON:
            p1, p2 = a[k1], a[k2]
            if p1 is not None and p2 is not None:
                cv2.line(f, _pt(p1), _pt(p2), (100, 220, 140), 2)

        for name, color in self._KP_COLORS.items():
            pt = a[name]
            if pt is not None:
                cv2.circle(f, _pt(pt), 5, color, -1)
                cv2.circle(f, _pt(pt), 7, color, 1)

        # Mũi tên hướng vai (cho thấy orientation)
        ls, rs = a["left_shoul"], a["right_shoul"]
        if ls is not None and rs is not None:
            cv2.arrowedLine(f, _pt(ls), _pt(rs), (255, 200, 0), 2, tipLength=0.3)

    def _draw_centroid(self, f, centroid):
        cv2.drawMarker(f, _pt(centroid), (0, 220, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(f, _pt(centroid), 10, (0, 220, 255), 1)

    def _draw_state_panel(self, f, state, safety, lost_frames, stream_ok, reconnects):
        state_color = C_OK if state == "TRACKING" else (C_WARN if state != "IDLE" else C_DIM)
        lines = [
            (f"State  : {state}",                              state_color),
            (f"Safety : {'ON' if safety else 'OFF'}",          C_OK if safety else C_WARN),
            (f"Stream : {'OK' if stream_ok else 'LOST'}"
             f"  rc={reconnects}",                             C_OK if stream_ok else C_DANGER),
            (f"Lost   : {lost_frames}f",                       C_DIM),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(f, text, (10, 44 + i * 22), FONT, 0.50, color, 1)

    def _draw_pid_panel(self, f, pid_out, rc_vals):
        px, py = self.w - 220, 22
        cv2.putText(f, "     PID out     RC%", (px, py), FONT, 0.42, C_DIM, 1)
        for i, axis in enumerate(("yaw", "throttle", "pitch")):
            pid = pid_out.get(axis, 0.0)
            rc  = rc_vals.get(axis, 0)
            c   = C_OK if abs(pid) < 0.25 else C_WARN
            cv2.putText(f,
                        f"{axis.upper():<9} {pid:+.3f}   {rc:+4d}",
                        (px, py + (i + 1) * 20), FONT, 0.42, c, 1)

    def _draw_error_bar(self, f, ex, ey):
        bw = 220
        bx = (self.w - bw) // 2
        by = self.h - 24
        cv2.rectangle(f, (bx, by), (bx + bw, by + 12), (40, 40, 44), -1)
        cx_bar = bx + bw // 2
        mx = max(bx, min(bx + bw, int(cx_bar + ex * bw / (self.w / 2))))
        cv2.line(f, (mx, by), (mx, by + 12), C_OK if abs(ex) < 40 else C_WARN, 2)
        cv2.line(f, (cx_bar, by + 2), (cx_bar, by + 10), C_DIM, 1)
        cv2.putText(f, f"err_x:{ex:+.0f}px  err_y:{ey:+.0f}px",
                    (bx, by - 6), FONT, 0.38, C_DIM, 1)

    def _draw_lost_warning(self, f, n, state="LOST"):
        alpha = min(1.0, n / 30)
        if state == "SEARCHING":
            remain = max(0, LOST_UNLOCK_FRAMES + SEARCH_LAND_FRAMES - n)
            text  = f"SEARCHING... land in {remain}f"
            color = (int(30 * alpha), int(160 * alpha), int(240 * alpha))  # cam
        else:
            text  = f"TARGET LOST ({n}f)"
            color = (int(60 * alpha), int(60 * alpha), int(220 * alpha))   # đỏ
        sz = cv2.getTextSize(text, FONT, 0.9, 2)[0]
        cv2.putText(f, text, ((self.w - sz[0]) // 2, self.h // 2),
                    FONT, 0.9, color, 2)


def _pt(arr) -> tuple:
    """Chuyển array 2 phần tử → (int, int) cho OpenCV."""
    return (int(arr[0]), int(arr[1]))


def _iou(a, b) -> float:
    """Intersection-over-Union của 2 bounding box [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ════════════════════════════════════════════════════════════════════════════
class EMAFilter:
    """
    Exponential Moving Average — làm mượt jitter từ YOLO detection.

    smoothed = alpha * new + (1 - alpha) * prev
      alpha gần 1 → phản ứng nhanh, ít lọc
      alpha gần 0 → rất mượt, phản ứng chậm hơn
    Giá trị khuyên dùng: 0.3–0.5 cho người đi bộ.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def __call__(self, key: str, value: float) -> float:
        if key not in self._state:
            self._state[key] = value          # khởi tạo với giá trị đầu tiên
        else:
            self._state[key] = (self.alpha * value
                                + (1.0 - self.alpha) * self._state[key])
        return self._state[key]

    def reset(self):
        self._state.clear()


# ════════════════════════════════════════════════════════════════════════════
class TrackingSystem:
    """
    Vòng lặp chính kết nối tất cả thành phần:
      FrameGrabber → KeypointAnalyzer → DroneController → E88ProDrone → Overlay
    """

    def __init__(self, args):
        self.args        = args
        self.no_fly      = args.no_fly
        self.target_span   = args.target_dist
        self.target_height = args.target_height
        self.source      = args.source

        # Flip: tự động bật cho webcam (index), tắt cho RTSP/file
        self.flip = args.flip or isinstance(self.source, int)

        # State machine
        self.state       = "IDLE"   # IDLE | FLYING | TRACKING | LOST | SAFETY
        self.safety_mode = True
        self.lost_frames = 0
        self.pid_out     = {"yaw": 0.0, "throttle": 0.0, "pitch": 0.0}
        self.rc_vals     = {"roll": 0,  "pitch": 0,  "throttle": 0,  "yaw": 0}
        self._last_analysis: dict | None = None  # cache detection giữa các lần infer
        self._frame_count = 0
        self._stabilize_until = 0.0  # thời điểm kết thúc period ổn định sau takeoff
        self._kp_indices: tuple | None = None    # auto-detect từ output shape model
        self.ema     = EMAFilter(alpha=0.4)      # làm mượt jitter detection trước PID
        self.ema_out = EMAFilter(alpha=0.5)      # làm mượt RC output trước khi gửi drone
        self._last_rc_send = 0.0                 # timestamp lần gửi lệnh RC gần nhất
        self._locked_bbox  = None                # bbox [x1,y1,x2,y2] của người đang tracking

        # Load YOLOv8 model
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error("Không tìm thấy model: %s", model_path)
            sys.exit(1)
        logger.info("Loading model: %s", model_path)
        self.model = YOLO(str(model_path))

        # Frame grabber thread
        self.grabber = FrameGrabber(self.source, flip=self.flip)
        self.grabber.start()

        # Chờ frame đầu tiên (timeout 10s)
        logger.info("Chờ frame từ: %s ...", self.source)
        deadline = time.time() + 10.0
        frame    = None
        while time.time() < deadline:
            ok, frame = self.grabber.read()
            if ok and frame is not None:
                break
            time.sleep(0.1)
        else:
            logger.error("Không nhận được frame sau 10s. Kiểm tra kết nối.")
            self.grabber.stop()
            sys.exit(1)

        h, w = frame.shape[:2]
        logger.info("Frame size: %dx%d", w, h)

        self.analyzer = KeypointAnalyzer(w, h)
        self.overlay  = Overlay(w, h)
        self.pid      = DroneController()

        # Kết nối drone
        self.drone = None
        if not self.no_fly:
            self.drone = E88ProDrone(
                trim_pitch=args.trim_pitch,
                trim_roll=args.trim_roll,
                trim_yaw=args.trim_yaw,
            )
            if self.drone.connect():
                logger.info("Drone connected")
            else:
                logger.warning("Không kết nối được drone → chạy --no-fly mode")
                self.drone = None

    # ── Vòng lặp chính ──────────────────────────────────────────────────────
    def run(self):
        logger.info("Bắt đầu. Phím: T=Takeoff/Land  H=Hover  S=Safety  E=Emergency  Q=Quit")
        try:
            while True:
                ok, frame = self.grabber.read()

                # Nếu mất stream: frame đen, reset analysis
                if not ok or frame is None:
                    h = self.overlay.h
                    w = self.overlay.w
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                    self._last_analysis = None
                else:
                    # Chạy YOLO mỗi INFER_SKIP frame, dùng kết quả cũ ở giữa
                    self._frame_count += 1
                    if self._frame_count % INFER_SKIP == 0:
                        self._last_analysis = self._detect(frame)

                analysis = self._last_analysis

                # Cập nhật lost counter
                if analysis is None:
                    self.lost_frames += 1
                else:
                    if self.lost_frames > 0:
                        # Re-acquire: xóa state cũ để tránh jerk do integral/dt stale
                        self.ema.reset()
                        self.ema_out.reset()
                        self.pid.reset_all()
                        logger.info("Tracking re-acquired sau %d frames", self.lost_frames)
                    self.lost_frames = 0

                # State machine + gửi lệnh điều khiển
                self._update_state(analysis)
                self._send_control(analysis)

                # Vẽ HUD và hiển thị
                self.overlay.tick_fps()
                display = self.overlay.draw(
                    frame, analysis, self.pid_out, self.rc_vals,
                    self.state, self.safety_mode, self.lost_frames,
                    ok, self.grabber.reconnects
                )
                cv2.imshow("Drone Tracking — E88 Pro", display)

                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                if   key == ord('q'):                   break
                elif key == ord('t'):                   self._toggle_fly()
                elif key == ord('h'):                   self._hover()
                elif key == ord('e'):                   self._emergency()
                elif key == ord('s'):
                    self.safety_mode = not self.safety_mode
                    logger.info("Safety mode: %s", "ON" if self.safety_mode else "OFF")

        finally:
            self._shutdown()

    # ── Detection ───────────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> dict | None:
        """
        Chạy YOLOv8 và chọn đúng người đang tracking (lock-on).

        Lần đầu chưa có lock: chọn người gần tâm frame nhất — người dùng nên
        đứng chính giữa trước khi nhấn T.

        Các frame sau: chỉ chọn detection có IoU >= IOU_LOCK_MIN với bbox lần
        trước. Người khác bước vào frame sẽ bị bỏ qua hoàn toàn.
        """
        results = self.model(frame, verbose=False, conf=DET_CONF)
        if not results or results[0].keypoints is None:
            return None

        kps   = results[0].keypoints
        boxes = results[0].boxes
        if kps.xy is None or len(kps.xy) == 0:
            return None

        n = len(kps.xy)

        # Auto-detect keypoint layout lần đầu tiên có detection
        if self._kp_indices is None:
            n_kp = kps.xy.shape[1]
            if n_kp >= 17:
                self._kp_indices = KP_INDICES_COCO
                logger.info("Model 17-kp COCO: nose=0, l_shoul=5, r_shoul=6, l_hip=11, r_hip=12")
            else:
                self._kp_indices = KP_INDICES_5KP
                logger.info("Model 5-kp custom: nose=0, l_shoul=1, r_shoul=2, l_hip=3, r_hip=4")

        if self._locked_bbox is None:
            # Chưa lock → chọn người gần tâm frame nhất
            cx, cy = self.analyzer.cx, self.analyzer.cy
            best, best_dist = 0, float("inf")
            for i in range(n):
                if boxes is not None:
                    b = boxes.xyxy[i].cpu().numpy()
                    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                else:
                    pts = kps.xy[i].cpu().numpy()
                    bcx, bcy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                d = (bcx - cx) ** 2 + (bcy - cy) ** 2
                if d < best_dist:
                    best_dist, best = d, i

            # Lưu bbox để lock
            if boxes is not None:
                self._locked_bbox = boxes.xyxy[best].cpu().numpy().copy()
                logger.info("Lock-on: detection %d, bbox=(%.0f,%.0f,%.0f,%.0f)",
                            best, *self._locked_bbox)
        else:
            # Đã lock → tìm detection có IoU cao nhất với bbox đã lưu
            if boxes is None:
                return None
            best, best_iou = -1, 0.0
            for i in range(n):
                iou = _iou(self._locked_bbox, boxes.xyxy[i].cpu().numpy())
                if iou > best_iou:
                    best_iou, best = iou, i

            if best == -1 or best_iou < IOU_LOCK_MIN:
                return None  # người bị lock không còn trong frame

            # Cập nhật bbox để bám theo chuyển động
            self._locked_bbox = boxes.xyxy[best].cpu().numpy().copy()

        xy   = kps.xy[best].cpu().numpy()
        conf = kps.conf[best].cpu().numpy()
        analysis = self.analyzer.analyze(xy, conf, self._kp_indices)

        if analysis is not None and boxes is not None:
            # Dùng bbox center Y cho error_y (ổn định hơn keypoint centroid khi ngồi/cúi)
            box = boxes.xyxy[best].cpu().numpy()
            analysis["error_y"] = float((box[1] + box[3]) / 2.0 - self.analyzer.cy)
        elif analysis is None and self._locked_bbox is not None and boxes is not None:
            # Bbox IoU vẫn khớp nhưng vai/keypoint khuất (xoay lưng, xoay ngang).
            # Trả về analysis tối giản từ bbox: giữ yaw + throttle, freeze pitch.
            # shoulder_span = target_span → dist_error = 0 → không tiến/lùi.
            box = self._locked_bbox
            bcx = (box[0] + box[2]) / 2.0
            bcy = (box[1] + box[3]) / 2.0
            centroid = np.array([bcx, bcy])
            analysis = {
                "error_x":      float(bcx - self.analyzer.cx),
                "error_y":      float(bcy - self.analyzer.cy),
                "shoulder_span": self.target_span,
                "body_height":  None,
                # Các field cần cho Overlay — None để skeleton không vẽ
                "centroid":     centroid,
                "shoulder_mid": centroid,
                "orientation":  0.0,
                "facing":       "back",
                "valid_kps":    0,
                "nose":         None,
                "left_shoul":   None,
                "right_shoul":  None,
                "left_hip":     None,
                "right_hip":    None,
            }

        return analysis

    # ── State machine ────────────────────────────────────────────────────────
    def _update_state(self, analysis):
        if self.state == "IDLE":
            return

        # Chờ ổn định sau takeoff — giữ FLYING, không bắt đầu tracking sớm
        if self.state == "FLYING" and time.time() < self._stabilize_until:
            return

        # Cập nhật state theo detection
        if analysis is not None:
            self.state = "TRACKING"
        elif self.lost_frames >= LOST_UNLOCK_FRAMES:
            # Lần đầu vượt ngưỡng: xóa lock để YOLO tự do detect người mới
            if self._locked_bbox is not None:
                self._locked_bbox = None
                self.pid.reset_all()
                self.ema.reset()
                self.ema_out.reset()
                logger.info("Target mất %d frames → xóa lock, bắt đầu tìm đối tượng mới",
                            self.lost_frames)
            self.state = "SEARCHING"
        elif self.lost_frames >= LOST_HOVER_FRAMES:
            self.state = "SAFETY" if self.safety_mode else "LOST"

        # Hạ cánh nếu tìm đối tượng mới quá lâu vẫn không thấy
        if self.state == "SEARCHING" and self.lost_frames >= LOST_UNLOCK_FRAMES + SEARCH_LAND_FRAMES:
            logger.warning("Tìm đối tượng mới %d frames không thấy → tự hạ cánh",
                           SEARCH_LAND_FRAMES)
            if self.drone:
                self.drone.land()
            self.state = "IDLE"
            self.lost_frames = 0
            self.pid.reset_all()

    # ── Control ──────────────────────────────────────────────────────────────
    def _send_control(self, analysis):
        """
        Tính PID và gửi lệnh RC.
        - TRACKING : PID chạy đầy đủ
        - SAFETY   : hover (PID reset)
        - LOST     : giữ lệnh cũ hoặc hover tùy config
        - IDLE     : hover, PID reset
        """
        if self.state not in ("TRACKING", "LOST"):
            if self.drone:
                self.drone.hover()
            self.pid_out = {"yaw": 0.0, "throttle": 0.0, "pitch": 0.0}
            if self.state == "IDLE":
                self.pid.reset_all()
            return

        if analysis is None:
            # LOST state: hover để tránh trôi
            if self.drone:
                self.drone.hover()
            return

        # Rate-limit: chỉ tính PID + gửi lệnh mỗi RC_SEND_INTERVAL giây.
        # Giữa các lần gửi, drone tự giữ lệnh RC cũ → không cần gửi liên tục.
        # PID chỉ được compute đúng tại thời điểm gửi để dt chính xác, tránh tích lũy integral sai.
        now = time.time()
        if now - self._last_rc_send < RC_SEND_INTERVAL:
            return
        self._last_rc_send = now

        # Distance error: ưu tiên body_height (ổn định khi người đi ngang),
        # fallback shoulder_span khi thiếu nose/hip.
        bh = analysis.get("body_height")
        if bh is not None and bh >= 20.0:
            dist_error = self.target_height - bh
        else:
            dist_error = self.target_span - analysis["shoulder_span"]

        # EMA — làm mượt jitter detection trước PID
        ex = self.ema("error_x",    analysis["error_x"])
        ey = self.ema("error_y",    analysis["error_y"])
        de = self.ema("dist_error", dist_error)

        # Deadband — jitter keypoint nhỏ hơn ngưỡng → zero out, không kích hoạt trục đó
        if abs(ex) < YAW_DEADBAND_PX:      ex = 0.0
        if abs(ey) < THROTTLE_DEADBAND_PX: ey = 0.0
        if abs(de) < DIST_DEADBAND_PX:     de = 0.0

        self.pid_out = self.pid.compute(
            error_x        = ex,
            error_y        = ey,
            distance_error = de,
        )

        # Làm mượt output RC cho yaw/pitch — tránh lệnh nhảy đột ngột
        # Throttle KHÔNG dùng ema_out: hysteresis khi đổi chiều (xuống→lên) gây delay, drone không nâng được
        yaw_rc      = self.ema_out("yaw",   self.pid_out["yaw"])
        throttle_rc = self.pid_out["throttle"]
        pitch_rc    = self.ema_out("pitch", self.pid_out["pitch"])

        if self.drone:
            self.drone.send_rc(
                roll     = 0.0,
                pitch    = pitch_rc,
                throttle = throttle_rc,
                yaw      = yaw_rc,
            )
            self.rc_vals = self.drone.rc_values
        else:
            # No-fly: simulate RC values để hiển thị
            self.rc_vals = {
                "yaw":      int(yaw_rc      * 100),
                "throttle": int(throttle_rc * 100),
                "pitch":    int(pitch_rc    * 100),
                "roll":     0,
            }

    # ── Actions ──────────────────────────────────────────────────────────────
    def _toggle_fly(self):
        if self.state == "IDLE":
            if self.drone:
                logger.info("Takeoff...")
                self.drone.takeoff()
            self.state = "FLYING"
            self.pid.reset_all()
            self.ema.reset()
            self.ema_out.reset()
            self._last_analysis = None
            self.lost_frames = 0
            self._last_rc_send = 0.0
            self._locked_bbox  = None
            self._stabilize_until = time.time() + 3.0
            logger.info("Drone đang bay — giữ yên 3s để ổn định...")
        else:
            if self.drone:
                self.drone.land()
            self.state = "IDLE"
            logger.info("Hạ cánh")

    def _hover(self):
        if self.drone:
            self.drone.hover()
        self.pid.reset_all()
        logger.info("Hover")

    def _emergency(self):
        logger.warning("EMERGENCY STOP!")
        if self.drone:
            self.drone.emergency_stop()
        self.state = "IDLE"

    def _shutdown(self):
        logger.info("Đang thoát...")
        self.grabber.stop()
        if self.drone and self.drone.is_flying:
            logger.info("Hạ cánh trước khi thoát...")
            for _ in range(3):
                self.drone.send_rc(0, 0, 0, 0)
                time.sleep(0.05)
            self.drone.land()
            time.sleep(3.0)
            self.drone.emergency_stop()
        if self.drone:
            self.drone.disconnect()
        cv2.destroyAllWindows()
        logger.info("Done.")


# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Drone Person Tracking — E88 Pro + YOLOv8 Pose 5 keypoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--model",
                   default="best.pt",
                   help="YOLOv8 pose model (.pt). Mặc định: best.pt")
    p.add_argument("--source",
                   default=VIDEO_URL,
                   help=f"Camera index hoặc RTSP URL. Mặc định: {VIDEO_URL}")
    p.add_argument("--no-fly",
                   action="store_true",
                   help="Chạy dashboard không kết nối drone (test tracking)")
    p.add_argument("--target-dist",
                   type=float, default=130.0,
                   help="Target shoulder span (px), dùng khi không có body_height. Mặc định: 80")
    p.add_argument("--target-height",
                   type=float, default=240.0,
                   help="Target body height px (nose→hip), ưu tiên hơn shoulder span. Mặc định: 220")
    p.add_argument("--flip",
                   action="store_true",
                   help="Flip frame ngang (webcam thường cần flip)")
    p.add_argument("--trim-pitch",  type=int, default=0,
                   help="Trim pitch (byte offset, âm=lùi, dương=tiến). Mặc định: 0")
    p.add_argument("--trim-roll",   type=int, default=0,
                   help="Trim roll (byte offset, âm=trái, dương=phải). Mặc định: 0")
    p.add_argument("--trim-yaw",    type=int, default=0,
                   help="Trim yaw (byte offset). Mặc định: 0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Chuyển source thành int nếu là số (index camera)
    try:
        args.source = int(args.source)
    except (ValueError, TypeError):
        pass

    system = TrackingSystem(args)
    system.run()
