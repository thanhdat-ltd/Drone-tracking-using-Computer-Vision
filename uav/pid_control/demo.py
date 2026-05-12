"""
UAV + H1 humanoid robot (RL flat policy) — YOLO Autonomous Person Tracking.

Drone tự động bám theo người được phát hiện qua camera YOLO-Pose:

  Bước 1 — YAW alignment:
    Tính cx_err = (bbox_center_x - cam_cx) / cam_w
    PID_yaw_track → yaw_sp  (chỉ xoay, chưa tiến)

  Bước 2 — FORWARD/BACKWARD khi đã căn giữa:
    Bbox được tính từ 4 keypoints: vai-trái, vai-phải, hông-trái, hông-phải
    Tính bbox_h = y_hip - y_shoulder (chiều cao shoulder→hip)
    dist_err = TARGET_BBOX_H - bbox_h  →  PID_forward → pitch_d
    (bbox_h lớn = gần quá → lùi, bbox_h nhỏ = xa quá → tiến)

  Khi không thấy người: hover tại chỗ (pitch=0, roll=0, yaw giữ nguyên).
  R (focus Isaac Sim window) → reset drone về vị trí ban đầu.

Kiến trúc điều khiển drone:
    z_err       → [PID_z]       → thrust          (200 Hz)
    cx_err      → [PID_yaw_trk] → yaw_sp          (cam_interval)
    dist_err    → [PID_fwd]     → pitch_d          (cam_interval, chỉ khi aligned)
    angle_err   → [PID_att]     → rate_des         (50 Hz, outer)
    rate_err    → [PID_rate]    → moment            (200 Hz, inner)

Run:
    ./isaaclab.sh -p demo.py --policy_pt /path/to/policy.pt
"""

import argparse
import os
import math
import collections
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAV autonomous tracking of H1 humanoid with RL flat policy")
_DEFAULT_POLICY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../model/h1_flat/2026-05-12_01-59-31/exported/policy.pt",
)
parser.add_argument("--policy_pt", type=str, default=_DEFAULT_POLICY,
                    help="Path to JIT-exported policy (.pt).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

from dataclasses import replace

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera, CameraCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg_PLAY
from isaaclab_assets.uav.uav_cfg import UAV_CFG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import AttitudeController
from pid_controller import PIDController, make_alloc_inv, wrap_angle


# H1 CONFIG
H1_VEL_X         = 1.0    # m/s tiến thẳng
H1_ARRIVE_THRESH = 1.5    # m
H1_N_WAYPOINTS   = 8
H1_WP_RADIUS     = 5.0    # m
H1_WP_SEED       = 42

# H1FlatEnvCfg_PLAY observation layout:
# [base_lin_vel(3), base_ang_vel(3), proj_grav(3), vel_commands(3),
#  joint_pos(19), joint_vel(19), actions(19)]  → velocity commands at 9:12
H1_CMD_SLICE = slice(9, 12)   # [vel_x, vel_y, ang_vel_z]

def make_h1_waypoints(n=H1_N_WAYPOINTS, radius=H1_WP_RADIUS, seed=H1_WP_SEED):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.4, radius)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


#  UAV/SIM CONF
class Cfg:
    # Altitude hold
    TARGET_Z = 1.5
    Z_KP = 2.5;  Z_KI = 0.85;  Z_KD = 0.52;  Z_ILIM = 1.0

    # Outer loop: angle → rate_des  (50 Hz)
    ATT_KP     = 5.5;  ATT_KI     = 0.3;  ATT_KD     = 0.0
    YAW_ATT_KP = 1.0;  YAW_ATT_KI = 0.0;  YAW_ATT_KD = 0.0

    # Inner loop: rate → moment  (200 Hz)
    RATE_KP     = 0.0002;   RATE_KI     = 0.00015;  RATE_KD     = 0.0000185; RATE_LIM     = 1.0
    YAW_RATE_KP = 0.00015;  YAW_RATE_KI = 0.0005;   YAW_RATE_KD = 0.00001;   YAW_RATE_LIM = 0.2

    MAX_RATE       = math.radians(180.0)
    MAX_YAW_RATE   = math.radians(90.0)
    MAX_MOMENT     = 0.02
    MAX_YAW_MOMENT = 0.0002

    # Frequency
    SIM_HZ      = 200
    DECIMATION  = 4
    OUTER_HZ    = 50
    OUTER_EVERY = SIM_HZ // OUTER_HZ   # 4 (= DECIMATION)

    #  Autonomous tracking — Yaw alignment 
    # cx_err = (bbox_cx - cam_cx) / cam_w  ∈ [-0.5, 0.5]
    # YAW_ALIGN_THRESH: nếu |cx_err| < ngưỡng → coi là đã căn giữa
    YAW_TRK_KP      = 1.8
    YAW_TRK_KI      = 0.05
    YAW_TRK_KD      = 0.12
    YAW_TRK_ILIM    = 0.5
    YAW_ALIGN_THRESH = 0.08   # ~5% chiều rộng frame

    # Autonomous tracking — Forward/backward (pitch control)
    # bbox_h = pixel-height từ shoulder đến hip (COCO idx 5,6,11,12)
    # TARGET_BBOX_H: chiều cao bbox mong muốn (pixels) ~2m
    # dist_err = TARGET_BBOX_H - bbox_h  ⇒ dương=xa⇒tiến, âm=gần⇒lùi
    TARGET_BBOX_H  = 140    # pixels
    FWD_KP         = 0.0008
    FWD_KI         = 0.00002
    FWD_KD         = 0.0003
    FWD_ILIM       = 0.03
    PITCH_MAX_AUTO = math.radians(8.0)
    ROLL_MAX_AUTO  = math.radians(5.0)

    # COCO keypoint raw indices for tracking bbox
    KP_IDX_SHOULDERS = [5, 6]
    KP_IDX_HIPS      = [11, 12]

    # Camera / YOLO
    CAM_WIDTH     = 640
    CAM_HEIGHT    = 480
    CAM_UPDATE_HZ = 10
    YOLO_WEIGHTS  = "yolov8n-pose.pt"
    YOLO_CONF     = 0.3
    YOLO_KP_CONF  = 0.25

    # Camera degradation — giả lập OV2640 FPV thực
    CAM_BRIGHTNESS = 0.62   # nhân độ sáng, < 1 làm tối
    CAM_BLUR_SIGMA = 1.2    # Gaussian blur sigma [pixel]
    CAM_NOISE_STD  = 15.0   # nhiễu Gaussian std (0–255)

    # Plot window
    WINDOW_S = 30.0

    # Drone spawn: 3m phía sau H1 (theo trục -Y)
    DRONE_INIT_POS = (1.5, -3.0, 1.5)

    #  Nhiễu môi trường
    # Gió nhẹ, không đổi, không giật
    #   F ≈ 0.014 N  ~5% thrust_hover → đủ để drone phải bù,
    #   nhưng PID vẫn giữ được ổn định
    WIND_FORCE_N    = (0.012, 0.008, 0.0)   # (Fx, Fy, Fz) thế giới

    # Nhiễu động cơ: phân phối đều ε_i ∈ [-1%, +1%] per motor
    #   → chênh lệch lực đẩy lớn nhất giữa 2 motor bất kỳ = 2%
    MOTOR_NOISE_MAX = 0.02                  # 2 % (peak-to-peak)
    CRAZYFLIE_MASS  = 0.027                 # kg  (Crazyflie 2.x)
    CRAZYFLIE_ARM   = 0.046                 # m   (tâm prop đến tâm khung)


#hằng số điều khiển H1
H1_VEL_X         = 1.0    # m/s
H1_ARRIVE_THRESH = 1.5    # m
H1_ANGULAR_GAIN  = 2.0    # (rad/s) per rad error
H1_MAX_ANGULAR   = math.radians(60.0)  # rad/s



# CAMERA CONFIG
def make_front_camera_cfg(prim_path="/World/Crazyflie/body/camera_front") -> CameraCfg:
    # OV2640: sensor 1/4" (3.6×2.7 mm), lens 2.8 mm → HFOV ≈ 66°
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=Cfg.CAM_WIDTH,
        height=Cfg.CAM_HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            horizontal_aperture=3.6,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


# YOLO DETECTION
_COCO_IDX   = [0, 5, 6, 11, 12]
_KPT_COLORS = [(0,255,255),(0,200,255),(0,100,255),(0,255,100),(255,100,0)]
_SKELETON   = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

def _extract_kpts(raw):
    if len(raw) >= 17:
        return [tuple(raw[i]) for i in _COCO_IDX]
    return [tuple(raw[i]) for i in range(min(5, len(raw)))]

def _is_valid_person(kpts5):
    c = Cfg.YOLO_KP_CONF
    return any(kpts5[i][2] >= c for i in [1, 2]) and any(kpts5[i][2] >= c for i in [3, 4])

def _draw_skeleton(bgr, kpts5):
    c = Cfg.YOLO_KP_CONF
    for a, b in _SKELETON:
        xa, ya, ca = kpts5[a]; xb, yb, cb = kpts5[b]
        if ca > c and cb > c:
            cv2.line(bgr, (int(xa), int(ya)), (int(xb), int(yb)), (180,180,180), 2)
    for i, (x, y, conf) in enumerate(kpts5):
        if conf > c:
            cv2.circle(bgr, (int(x), int(y)), 5, _KPT_COLORS[i % len(_KPT_COLORS)], -1)

_BLUR_KERNEL: torch.Tensor | None = None

def _get_blur_kernel() -> torch.Tensor:
    global _BLUR_KERNEL
    if _BLUR_KERNEL is None:
        sigma = Cfg.CAM_BLUR_SIGMA
        k = max(3, 2 * int(3 * sigma) + 1) | 1
        x = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-x**2 / (2 * sigma**2))
        g = g / g.sum()
        kern = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(3, 1, k, k).contiguous()
        _BLUR_KERNEL = kern.cuda() if torch.cuda.is_available() else kern
    return _BLUR_KERNEL

def _degrade_rgb(frame: np.ndarray) -> np.ndarray:
    """Làm tối + Gaussian blur + nhiễu — giả lập OV2640 FPV."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t   = torch.from_numpy(frame).to(dev, dtype=torch.float32)   # H,W,3
    t   = t * Cfg.CAM_BRIGHTNESS
    t   = t.permute(2, 0, 1).unsqueeze(0)                        # 1,3,H,W
    k   = _get_blur_kernel()
    pad = k.shape[-1] // 2
    t   = F.conv2d(t, k, padding=pad, groups=3)
    t   = t.squeeze(0).permute(1, 2, 0)                          # H,W,3
    t   = t + torch.randn_like(t) * Cfg.CAM_NOISE_STD
    return t.clamp(0, 255).byte().cpu().numpy()

def detect_person(frame_rgb, model):
    """Phát hiện người, trả về tracking info từ keypoints vai+hông.

    Returns:
        track_info: dict với keys:
            'bbox4'    — [x1,y1,x2,y2] bounding box toàn thân (lớn nhất)
            'kp_bbox'  — (cx, cy, kp_h) từ 4 keypoints vai+hông, hoặc None
                         cx, cy: tâm bbox keypoint (pixels)
                         kp_h  : chiều cao shoulder→hip (pixels)
            'found'    — bool
        annotated: frame RGB đã vẽ annotation
    """
    frame_rgb = _degrade_rgb(frame_rgb)
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    results   = model(frame_rgb, conf=Cfg.YOLO_CONF, device=0, verbose=False)
    annotated = bgr.copy()
    best_bbox    = None
    best_kp_bbox = None
    max_area     = 0

    if results[0].boxes is not None:
        boxes    = results[0].boxes.xyxy.cpu().numpy()
        confs    = results[0].boxes.conf.cpu().numpy()
        has_kpts = results[0].keypoints is not None
        kpts_raw = results[0].keypoints.data.cpu().numpy() if has_kpts else []

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            # Tính keypoint bbox từ vai + hông (COCO idx 5,6,11,12)
            kp_bbox_this = None
            if has_kpts and i < len(kpts_raw):
                raw_kpts = kpts_raw[i]   # shape (17,3)
                if len(raw_kpts) >= 13:
                    c_th  = Cfg.YOLO_KP_CONF
                    s_pts = []  # shoulders
                    h_pts = []  # hips
                    for idx in Cfg.KP_IDX_SHOULDERS:
                        x_k, y_k, c_k = raw_kpts[idx]
                        if c_k >= c_th:
                            s_pts.append((x_k, y_k))
                    for idx in Cfg.KP_IDX_HIPS:
                        x_k, y_k, c_k = raw_kpts[idx]
                        if c_k >= c_th:
                            h_pts.append((x_k, y_k))

                    if len(s_pts) >= 1 and len(h_pts) >= 1:
                        # Trung bình tọa độ vai và hông
                        sx = sum(p[0] for p in s_pts) / len(s_pts)
                        sy = sum(p[1] for p in s_pts) / len(s_pts)
                        hx = sum(p[0] for p in h_pts) / len(h_pts)
                        hy = sum(p[1] for p in h_pts) / len(h_pts)

                        all_x = [p[0] for p in s_pts + h_pts]
                        all_y = [p[1] for p in s_pts + h_pts]
                        kp_x1 = int(min(all_x)); kp_x2 = int(max(all_x))
                        kp_y1 = int(sy);          kp_y2 = int(hy)

                        cx_kp = (sx + hx) / 2.0
                        cy_kp = (sy + hy) / 2.0
                        kp_h  = max(0.0, hy - sy)   # chiều cao shoulder→hip
                        kp_bbox_this = (cx_kp, cy_kp, kp_h)

                        # Vẽ keypoint bbox (màu vàng) cho detection lớn nhất
                        if area > max_area:
                            cv2.rectangle(annotated, (kp_x1, kp_y1), (kp_x2, kp_y2), (0, 220, 255), 2)
                            cv2.circle(annotated, (int(cx_kp), int(cy_kp)), 6, (0, 255, 255), -1)
                            cv2.putText(annotated, f"h={kp_h:.0f}px | target={Cfg.TARGET_BBOX_H}px",
                                        (kp_x1, kp_y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

                # Skeleton
                kpts5 = _extract_kpts(raw_kpts)
                if _is_valid_person(kpts5):
                    _draw_skeleton(annotated, kpts5)

            if area > max_area:
                max_area     = area
                best_bbox    = [x1, y1, x2, y2]
                best_kp_bbox = kp_bbox_this

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, f"H1 {conf:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    cx_cam, cy_cam = w // 2, h // 2
    cv2.drawMarker(annotated, (cx_cam, cy_cam), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    # Status bar
    cv2.rectangle(annotated, (0, h - 24), (w, h), (25, 25, 25), -1)
    found = best_bbox is not None and best_kp_bbox is not None
    if found:
        cx_kp, cy_kp, kp_h = best_kp_bbox
        cx_err  = (cx_kp - cx_cam) / w
        aligned = abs(cx_err) < Cfg.YAW_ALIGN_THRESH
        status  = (f"TRACKING | cx_err={cx_err:+.3f} | "
                   f"kp_h={kp_h:.0f}px | {'ALIGNED ▶ FWD' if aligned else 'ALIGNING YAW'}")
        cv2.line(annotated, (cx_cam, cy_cam), (int(cx_kp), int(cy_kp)), (255, 100, 0), 1)
    else:
        status = "NO TARGET — hovering"
    cv2.putText(annotated, status, (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    track_info = {
        "bbox4":   best_bbox,
        "kp_bbox": best_kp_bbox,
        "found":   found,
    }
    return track_info, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# SCENE HELPERS
def load_scene_extras():
    """Load lights + warehouse background. Phải gọi SAU env.sim.reset()."""
    # Lights
    sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.88)).func(
        "/World/Lights/Sun",
        sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 0.95, 0.88)),
    )
    sim_utils.DomeLightCfg(intensity=300.0, color=(0.7, 0.8, 1.0)).func(
        "/World/Lights/Dome",
        sim_utils.DomeLightCfg(intensity=300.0, color=(0.7, 0.8, 1.0)),
    )
    print("[ENV] Lights OK")

    # Warehouse background (Simple_Warehouse — nhỏ, load nhanh)
    warehouse_urls = [
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
        "/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    ]
    for url in warehouse_urls:
        try:
            cfg = sim_utils.UsdFileCfg(usd_path=url)
            cfg.func("/World/Warehouse", cfg, translation=(0.0, 0.0, 0.0))
            print(f"[ENV] Warehouse OK: {url[:70]}...")
            return
        except Exception as e:
            print(f"[ENV] Warehouse load failed ({url[:50]}...): {e}")
    print("[ENV] Chạy không có warehouse background.")


def setup_simulation(policy_pt_path: str):
    """Khởi tạo Isaac Lab simulation, H1 env, drone, camera."""
    env_cfg = H1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs    = 1
    env_cfg.scene.env_spacing = 10.0
    env_cfg.curriculum        = None
    env_cfg.episode_length_s  = 3600.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    device = getattr(args_cli, "device", "cuda:0")
    env_cfg.sim.device = device
    if device == "cpu":
        env_cfg.sim.use_fabric = False

    print("[H1] Creating ManagerBasedRLEnv ...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Policy
    policy = None
    if policy_pt_path and os.path.isfile(policy_pt_path):
        print(f"[POLICY] Loading: {policy_pt_path}")
        policy = torch.jit.load(policy_pt_path, map_location=device)
        policy.eval()
        print("[POLICY] OK")
    else:
        print(f"[POLICY] Không tìm thấy: {policy_pt_path}")
        print("[POLICY] H1 sẽ nhận zero actions.")

    # Drone (Crazyflie)
    robot_cfg = replace(UAV_CFG)
    robot_cfg.prim_path      = "/World/Crazyflie"
    robot_cfg.init_state.pos = Cfg.DRONE_INIT_POS
    robot = Articulation(robot_cfg)

    camera = Camera(make_front_camera_cfg())

    # reset() TRƯỚC load_scene_extras (đúng thứ tự Isaac Lab)
    env.sim.reset()

    load_scene_extras()   # lights + warehouse

    robot.reset()
    camera.reset()
    obs, _ = env.reset()
    return env, policy, robot, camera, obs


# DRONE RESET
def reset_drone(robot, ctrl, pid_z, pid_yaw_trk, pid_fwd):
    """Reset drone về trạng thái khởi tạo và clear toàn bộ PID state."""
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    default_root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(default_root[:, :7])
    robot.write_root_velocity_to_sim(default_root[:, 7:])
    robot.reset()

    ctrl.reset()
    pid_z.reset()
    pid_yaw_trk.reset()
    pid_fwd.reset()
    print("[RESET] Drone reset OK → pos =", Cfg.DRONE_INIT_POS)


# PLOTTING
def make_plots():
    plt.ion()
    fig = plt.figure("UAV Autonomous YOLO Tracking", figsize=(18, 9))
    fig.suptitle("UAV Autonomous Tracking  (blue=actual  red--=setpoint)", fontweight="bold")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.set_title("FPV Camera (YOLO-Pose)", fontsize=9)
    ax_cam.axis("off")
    im_rgb = ax_cam.imshow(
        np.zeros((Cfg.CAM_HEIGHT, Cfg.CAM_WIDTH, 3), dtype=np.uint8),
        extent=[0, Cfg.CAM_WIDTH, Cfg.CAM_HEIGHT, 0],
        aspect="equal",
    )

    subplot_specs = [
        (gs[0, 1], "Z [m]"),
        (gs[1, 1], "Drone X [m]"),
        (gs[2, 1], "Drone Y [m]"),
        (gs[0, 2], "Pitch [°]"),
        (gs[1, 2], "Roll [°]"),
        (gs[2, 2], "Yaw [°]"),
    ]
    axes_data, lines = [], []
    for spec, title in subplot_specs:
        ax = fig.add_subplot(spec)
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        l_act, = ax.plot([], [], "b-",  lw=1.5, label="actual")
        l_des, = ax.plot([], [], "r--", lw=1.2, label="setpoint")
        ax.legend(fontsize=6, loc="upper left")
        axes_data.append(ax)
        lines.append((l_act, l_des))
    return fig, im_rgb, axes_data, lines


# MAIN LOOP
def main():
    print("=" * 60)
    print("UAV AUTONOMOUS Person Tracking + H1 RL Policy")
    print("Drone tự động bám theo người qua YOLO-Pose keypoints")
    print("  Bước 1: Xoay yaw căn giữa người vào tâm camera")
    print("  Bước 2: Tiến/lùi theo chiều cao bbox vai-hông")
    print("R (focus Isaac Sim) = reset drone")
    print("=" * 60)

    env, policy, robot, camera, obs = setup_simulation(args_cli.policy_pt)

    physics_dt = env.sim.get_physics_dt()
    decimation = env.cfg.decimation
    dt_outer   = decimation * physics_dt

    print(f"[TIMING] physics_dt={physics_dt*1000:.1f}ms | "
          f"decimation={decimation} | outer_dt={dt_outer*1000:.1f}ms")

    device   = env.device
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv    = make_alloc_inv(device=device)

    # Disturbance setup (tính trước 1 lần)
    # Gió nhẹ
    _wind_f   = torch.tensor(Cfg.WIND_FORCE_N, dtype=torch.float32,
                             device=device).view(1, 1, 3)

    # Nhiễu động cơ: arm theo layout Crazyflie (45°, m1=+x+y, m2=-x+y, m3=-x-y, m4=+x-y)
    _L        = Cfg.CRAZYFLIE_ARM / math.sqrt(2.0)
    _rx       = torch.tensor([ _L, -_L, -_L,  _L], device=device)   # x arm
    _ry       = torch.tensor([ _L,  _L, -_L, -_L], device=device)   # y arm
    _yaw_sgn  = torch.tensor([ 1., -1.,  1., -1.], device=device)   # drag hướng xoay
    _T_hover  = Cfg.CRAZYFLIE_MASS * 9.81 / 4.0                     # N / motor khi hover

    # PIDs
    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    ctrl = AttitudeController(
        robot=robot, prop_body_ids=prop_ids, root_body_ids=body_ids, A_inv=A_inv,
        att_kp=Cfg.ATT_KP,         att_ki=Cfg.ATT_KI,         att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP,             rate_ki=Cfg.RATE_KI,
        rate_kd=Cfg.RATE_KD,             rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP,     yaw_rate_ki=Cfg.YAW_RATE_KI,
        yaw_rate_kd=Cfg.YAW_RATE_KD,     yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE, max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT, max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    # PID yaw: cx_err (normalized) → delta yaw setpoint
    pid_yaw_trk = PIDController(
        Cfg.YAW_TRK_KP, Cfg.YAW_TRK_KI, Cfg.YAW_TRK_KD,
        integral_limit=Cfg.YAW_TRK_ILIM,
        derivative_limit=2.0,
    )
    # PID forward: dist_err (pixels) → pitch_d (rad)
    pid_fwd = PIDController(
        Cfg.FWD_KP, Cfg.FWD_KI, Cfg.FWD_KD,
        integral_limit=Cfg.FWD_ILIM,
        derivative_limit=0.005,
    )


    yolo_model = YOLO(Cfg.YOLO_WEIGHTS)
    print(f"[YOLO] {Cfg.YOLO_WEIGHTS} OK")

    waypoints = make_h1_waypoints()
    target_wp = waypoints[0] if waypoints else (0.0, 0.0)
    wp_idx    = 0
    n_act     = env.action_space.shape[-1]
    h1_action = torch.zeros(1, n_act, device=device)
    h1_x = h1_y = 0.0

    # Plotting buffers
    fig, im_rgb, axes_data, lines = make_plots()

    maxlen = int(Cfg.WINDOW_S / physics_dt) + 200
    def _dq(): return collections.deque(maxlen=maxlen)
    times         = _dq(); zs    = _dq(); xs    = _dq(); ys    = _dq()
    pitchs        = _dq(); rolls = _dq(); yaws  = _dq()
    target_zs     = _dq(); target_xs = _dq(); target_ys = _dq()
    pitch_des_hist= _dq(); roll_sp_hist = _dq(); yaw_des_hist = _dq()

    cam_interval = max(1, round(1.0 / (physics_dt * Cfg.CAM_UPDATE_HZ)))

    # Control state
    sim_time  = 0.0
    phys_step = 0
    yaw_sp    = 0.0     # yaw setpoint tích lũy (rad, world frame)
    pitch_d   = 0.0     # pitch setpoint từ tracking (rad)
    roll_d    = 0.0     # roll luôn = 0
    p_des_hold = q_des_hold = r_des_hold = 0.0
    data      = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

    # Tracking state
    trk_cx_err   = 0.0    # normalized horizontal error [-0.5, 0.5]
    trk_dist_err = 0.0    # pixel error: TARGET_BBOX_H - kp_h
    trk_aligned  = False  # True khi yaw đã nhắm vào tâm
    trk_found    = False  # True khi YOLO thấy người
    

    while simulation_app.is_running():

        # YOLO tracking
        if phys_step % cam_interval == 0:
            try:
                rgb = camera.data.output.get("rgb")
                if rgb is not None and rgb.shape[0] > 0:
                    frame = rgb[0, ..., :3].cpu().numpy().astype(np.uint8)
                    if frame.max() > 0:
                        track_info, annotated = detect_person(frame, yolo_model)
                        trk_found = track_info["found"]
                        dt_cam    = cam_interval * physics_dt

                        if trk_found:
                            cx_kp, cy_kp, kp_h = track_info["kp_bbox"]
                            cam_cx = Cfg.CAM_WIDTH / 2.0

                            # cx_err: [-0.5, 0.5], dương = người lệch phải
                            trk_cx_err  = (cx_kp - cam_cx) / Cfg.CAM_WIDTH
                            trk_aligned = abs(trk_cx_err) < Cfg.YAW_ALIGN_THRESH

                            # Bước 1: Yaw tracking — luôn chạy
                            # PID output là delta yaw tích lũy vào yaw_sp hiện tại
                            # (dùng yaw_sp làm base, không phải cur_yaw, tránh jump)
                            delta_yaw = pid_yaw_trk.update(trk_cx_err, dt_cam)
                            yaw_sp    = wrap_angle(yaw_sp + delta_yaw)

                            # Bước 2: Forward/backward — chỉ khi đã aligned
                            if trk_aligned:
                                trk_dist_err = Cfg.TARGET_BBOX_H - kp_h
                                raw_pitch    = pid_fwd.update(trk_dist_err, dt_cam)
                                # dist_err dương (xa) → cần tiến → pitch âm (lean forward)
                                pitch_d = -float(
                                    max(-Cfg.PITCH_MAX_AUTO,
                                        min(Cfg.PITCH_MAX_AUTO, raw_pitch))
                                )
                            else:
                                # Đang align yaw → đứng yên, reset PID forward
                                pitch_d = 0.0
                                pid_fwd.reset()

                        else:
                            # Không thấy người: hover tại chỗ
                            trk_cx_err   = 0.0
                            trk_dist_err = 0.0
                            trk_aligned  = False
                            pitch_d      = 0.0
                            # Đồng bộ yaw_sp về yaw thực tế để tránh drift
                            yaw_sp = wrap_angle(data["yaw"])
                            pid_yaw_trk.reset()
                            pid_fwd.reset()

                        im_rgb.set_data(annotated)
            except Exception:
                pass

        roll_d = 0.0   # không dùng roll trong autonomous tracking

        # H1 policy
        is_h1_step = (phys_step % decimation == 0)
        if is_h1_step:
            # Lấy vị trí và quaternion của H1
            h1_root = env.unwrapped.scene["robot"].data.root_state_w[0]  # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            h1_x, h1_y, h1_z = h1_root[0].item(), h1_root[1].item(), h1_root[2].item()
            h1_qw, h1_qx, h1_qy, h1_qz = h1_root[3], h1_root[4], h1_root[5], h1_root[6]
            
            # Tính yaw hiện tại (Euler từ quaternion)
            siny_cosp = 2.0 * (h1_qw * h1_qz + h1_qx * h1_qy)
            cosy_cosp = 1.0 - 2.0 * (h1_qy**2 + h1_qz**2)
            h1_yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Kiểm tra đến waypoint
            dx = target_wp[0] - h1_x
            dy = target_wp[1] - h1_y
            dist = math.hypot(dx, dy)
            if dist < H1_ARRIVE_THRESH:
                wp_idx = (wp_idx + 1) % len(waypoints)
                target_wp = waypoints[wp_idx]
                dx = target_wp[0] - h1_x
                dy = target_wp[1] - h1_y
                dist = math.hypot(dx, dy)
                print(f"[H1] New waypoint: {target_wp}")
            
            # Tính góc mong muốn và vận tốc góc
            target_yaw = math.atan2(dy, dx)
            angle_error = wrap_angle(target_yaw - h1_yaw)
            ang_vel_z = np.clip(H1_ANGULAR_GAIN * angle_error, -H1_MAX_ANGULAR, H1_MAX_ANGULAR)
            
            # Vận tốc dài: giảm khi gần
            if dist < H1_ARRIVE_THRESH * 2:
                vel_x = H1_VEL_X * (dist / (H1_ARRIVE_THRESH * 2))
            else:
                vel_x = H1_VEL_X
            
            # Cập nhật velocity command trong observation
            obs["policy"][:, H1_CMD_SLICE] = torch.tensor([[vel_x, 0.0, ang_vel_z]], device=device, dtype=torch.float32)
            
            # Gọi policy để sinh action
            with torch.inference_mode():
                if policy is not None:
                    h1_action = policy(obs["policy"])
                else:
                    h1_action = torch.zeros(1, n_act, device=device)
            env.action_manager.process_action(h1_action)




        # Altitude PID (200 Hz) — z đọc sau sim.step() ở dưới, dùng biến z hiện tại
        thrust = max(0.0, pid_z.update(Cfg.TARGET_Z - z, physics_dt))

        # Outer loop (50 Hz)
        if phys_step % Cfg.OUTER_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_outer(
                dt_outer, roll_d, pitch_d, yaw_sp
            )

        # Inner loop (200 Hz)
        data = ctrl.step_inner(physics_dt, p_des_hold, q_des_hold, r_des_hold, thrust=thrust)

        # Gió nhẹ + nhiễu động cơ (áp dụng lên body chính của drone)
        # Motor noise: ε_i ∈ [-1%, +1%] per motor → moment vi sai
        eps_i = (torch.rand(4, device=device) - 0.5) * Cfg.MOTOR_NOISE_MAX
        dF_i  = eps_i * _T_hover                              # N mỗi motor
        tau_roll  = (dF_i * _ry).sum()
        tau_pitch = -(dF_i * _rx).sum()
        tau_yaw   = (dF_i * _yaw_sgn * 0.006).sum()
        # Shape: (num_instances=1, num_bodies=1, xyz=3)
        _noise_torque = torch.tensor(
            [[[ tau_roll.item(), tau_pitch.item(), tau_yaw.item() ]]],
            device=device
        )
        robot.set_external_force_and_torque(
            _wind_f, _noise_torque, body_ids=body_ids
        )


        robot.write_data_to_sim()

        if is_h1_step:
            env.action_manager.apply_action()

        env.scene.write_data_to_sim()

        is_render = (phys_step % env.cfg.sim.render_interval == 0)
        env.sim.step(render=is_render)
        env.scene.update(dt=physics_dt)
        robot.update(physics_dt)
        drone_pos = robot.data.root_pos_w[0].cpu().numpy()
        drone_x, drone_y, z = drone_pos[0], drone_pos[1], drone_pos[2]
        
        if is_render:
            camera.update(physics_dt)

        # H1 observation
        if (phys_step + 1) % decimation == 0:
            obs_buf = env.observation_manager.compute()
            obs = {"policy": obs_buf["policy"]}

        sim_time  += physics_dt
        phys_step += 1

        # Log mỗi giây
        if phys_step % round(1.0 / physics_dt) == 0:
            align_str = "ALIGNED ▶ FWD" if trk_aligned else "ALIGNING YAW"
            found_str = "YES" if trk_found else "NO"
            print(
                f"t={sim_time:6.1f}s | "
                f"z={z:+.2f}m (sp={Cfg.TARGET_Z:.1f}) | "
                f"target={found_str} | {align_str} | "
                f"cx_err={trk_cx_err:+.3f} | dist_err={trk_dist_err:+.1f}px | "
                f"pitch_d={math.degrees(pitch_d):+.1f}° | "
                f"yaw_sp={math.degrees(yaw_sp):+.1f}° | "
                f"yaw={math.degrees(data['yaw']):+.1f}°"
            )

        # Plotting buffers 
        times.append(sim_time)
        zs.append(z);          target_zs.append(Cfg.TARGET_Z)
        xs.append(drone_x);    target_xs.append(h1_x)
        ys.append(drone_y);    target_ys.append(h1_y)
        pitchs.append(math.degrees(data["pitch"])); pitch_des_hist.append(math.degrees(pitch_d))
        rolls.append(math.degrees(data["roll"]));   roll_sp_hist.append(math.degrees(roll_d))
        yaws.append(math.degrees(data["yaw"]));     yaw_des_hist.append(math.degrees(yaw_sp))

        # Plot update
        if phys_step % cam_interval == 0 and times:
            t = list(times)
            datasets = [
                (list(zs),     list(target_zs)),
                (list(xs),     list(target_xs)),
                (list(ys),     list(target_ys)),
                (list(pitchs), list(pitch_des_hist)),
                (list(rolls),  list(roll_sp_hist)),
                (list(yaws),   list(yaw_des_hist)),
            ]
            xlim = (max(0.0, t[-1] - Cfg.WINDOW_S), t[-1] + 0.5)
            for ax, (l_act, l_des), (act, des) in zip(axes_data, lines, datasets):
                l_act.set_data(t, act)
                l_des.set_data(t, des)
                ax.set_xlim(*xlim)
                all_v = act + des
                if all_v:
                    mn, mx = min(all_v), max(all_v)
                    pad = max(0.2, (mx - mn) * 0.1)
                    ax.set_ylim(mn - pad, mx + pad)
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Chase camera
        p = robot.data.root_pos_w[0].cpu().numpy()
        env.sim.set_camera_view(
            eye=[p[0] - 2.0, p[1] - 2.0, p[2] + 1.2],
            target=[p[0], p[1], p[2]],
        )

    simulation_app.close()


if __name__ == "__main__":
    main()