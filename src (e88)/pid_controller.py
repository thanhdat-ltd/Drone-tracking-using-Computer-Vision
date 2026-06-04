"""
PID Controller cho drone tracking
----------------------------------
Mỗi trục (yaw, throttle, pitch) có 1 PID riêng.
Dùng derivative-on-measurent để tránh derivative kick khi setpoint thay đổi đột ngột.
"""

import time


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_limit: float = 0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit  # anti-windup

        self._integral = 0.0
        self._last_measurement = None
        self._last_time = None

    def reset(self):
        self._integral = 0.0
        self._last_measurement = None
        self._last_time = None

    def compute(self, setpoint: float, measurement: float) -> float:
        now = time.time()
        error = setpoint - measurement

        if self._last_time is None:
            dt = 0.033  # assume ~30fps for first frame
        else:
            dt = now - self._last_time
            if dt <= 0:
                dt = 0.001

        # Proportional
        p_term = self.kp * error

        # Integral với anti-windup (clamp)
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # Derivative-on-measurement (tránh kick khi setpoint nhảy)
        if self._last_measurement is None:
            d_term = 0.0
        else:
            d_measurement = (measurement - self._last_measurement) / dt
            d_term = -self.kd * d_measurement  # âm vì derivative-on-measurement

        output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, output))

        self._last_measurement = measurement
        self._last_time = now
        return output


class DroneController:
    """
    3 PID loop riêng cho 3 trục điều khiển.

    Giá trị trả về trong khoảng [-1.0, 1.0]:
      yaw_cmd     : âm = xoay trái, dương = xoay phải
      throttle_cmd: âm = hạ xuống, dương = lên cao
      pitch_cmd   : âm = lùi ra xa, dương = tiến lại gần

    Tuning guide (bắt đầu với các giá trị này, điều chỉnh dần):
      - Tăng Kp nếu drone phản ứng chậm
      - Giảm Kp nếu drone lắc/oscillate
      - Thêm Kd (0.01-0.05) để giảm overshoot
      - Ki giữ nhỏ hoặc = 0 để tránh drift
    """

    def __init__(self):
        # Yaw: xoay theo người (error = pixel offset X)
        # Kd phải rất nhỏ: với pixel/s rate ~150 px/s, kd=0.005 → D term đảo chiều P term
        # Công thức an toàn: kd ≤ kp * error_typical / approach_rate_px_per_s
        # → kd ≤ 0.0008*100/150 ≈ 0.0005; dùng 0.0002 để có biên an toàn
        self.pid_yaw = PIDController(
            kp=0.0015, ki=0.00001, kd=0.00018,
            output_min=-0.25, output_max=0.25
        )
        # Throttle: giữ người ở giữa khung hình theo chiều dọc
        self.pid_throttle = PIDController(
            kp=0.012, ki=0.00001, kd=0.0005,
            output_min=-0.35, output_max=0.35
        )
        # Pitch: giữ khoảng cách với người (dựa trên shoulder span / body height)
        self.pid_pitch = PIDController(
            kp=0.005, ki=0.00001, kd=0.0005,
            output_min=-0.25, output_max=0.25
        )

    def compute(self,
                error_x: float,       # pixel: centroid X - frame_center_x (+ = người bên phải)
                error_y: float,       # pixel: centroid Y - frame_center_y (+ = người phía dưới)
                distance_error: float # pixel: target_span - current_span (+ = người quá xa)
                ) -> dict:
        yaw = self.pid_yaw.compute(0, -error_x)      # setpoint = 0 (người ở giữa)
        throttle = self.pid_throttle.compute(0, error_y)
        pitch = self.pid_pitch.compute(0, -distance_error)

        return {
            "yaw": round(yaw, 4),
            "throttle": round(throttle, 4),
            "pitch": round(pitch, 4),
        }

    def reset_all(self):
        self.pid_yaw.reset()
        self.pid_throttle.reset()
        self.pid_pitch.reset()
