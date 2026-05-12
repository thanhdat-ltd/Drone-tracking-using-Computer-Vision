# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
UAV Person Tracking + Pedestrian walking in scene.
3‑tier cascade control:
  - Position (20 Hz, full PID) → pitch/roll setpoint
  - Attitude (50 Hz, PD) → rate setpoint
  - Rate (200 Hz, PID) → motor moments
Pedestrian walks random waypoints.
Camera YOLO for visualization only (not used for control).
"""

import argparse
import math
import collections
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAV person tracking with pedestrian")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

import os
import omni
import omni.usd
import omni.kit.commands
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets.uav.uav_cfg import UAV_CFG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import AttitudeController
from pid_controller import PIDController, make_alloc_inv, wrap_angle

# Enable animation extensions
from isaacsim.core.utils.extensions import enable_extension
for ext in ["omni.anim.graph.core", "omni.anim.graph.schema", "omni.anim.graph.ui"]:
    try:
        enable_extension(ext)
    except Exception:
        pass
for _ in range(60):
    simulation_app.update()

try:
    import omni.anim.graph.core as ag
except ImportError:
    ag = None
    print("[WARN] Animation graph not available, pedestrian will not animate.")

# ========== PEDESTRIAN CONFIGURATION ==========
CHAR_ROOT = "/World/Character"
BIPED_ROOT = "/World/Biped_Setup"
_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People"
CHARACTER_CANDIDATES = [
    f"{_BASE}/Characters/F_Medical_01/F_Medical_01.usd",
    f"{_BASE}/Characters/M_Medical_01/M_Medical_01.usd",
]
BIPED_CANDIDATES = [
    f"{_BASE}/Animation/Biped_Setup.usd",
    f"{_BASE}/Biped_Setup.usd",
]
N_WAYPOINTS = 6
WAYPOINT_RADIUS = 5.0      # meters around center (3.0, 0.0)
WALK_SPEED = 0.5           # m/s
ARRIVE_THRESH = 0.4
CHAR_Z_OFFSET = 0.0        # adjust if sinking
YAW_OFFSET_DEG = -90.0     # Isaac Sim forward +Y
DT_PED = 1.0 / 60.0

# ========== UAV CONFIGURATION ==========
class Cfg:
    TARGET_Z = 3.0
    Z_KP = 1.0; Z_KI = 0.85; Z_KD = 0.72; Z_ILIM = 1.0

    # Attitude (outer loop) gains – PD only (I=0)
    ATT_KP = 5.5; ATT_KI = 0.0; ATT_KD = 1.2
    YAW_ATT_KP = 1.0; YAW_ATT_KI = 0.0; YAW_ATT_KD = 0.0

    # Rate (inner loop) gains
    RATE_KP = 0.0002; RATE_KI = 0.00015; RATE_KD = 0.0000185; RATE_LIM = 1.0
    YAW_RATE_KP = 0.00015; YAW_RATE_KI = 0.0005; YAW_RATE_KD = 0.00001; YAW_RATE_LIM = 0.2

    MAX_RATE = math.radians(180.0)
    MAX_YAW_RATE = math.radians(90.0)
    MAX_MOMENT = 0.03
    MAX_YAW_MOMENT = 0.0003

    SIM_HZ = 200
    OUTER_HZ = 50               # attitude loop frequency
    OUTER_EVERY = SIM_HZ // OUTER_HZ   # = 4

    # Camera
    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    CAM_UPDATE_HZ = 30
    CAM_FOCAL_LENGTH = 24.0
    CAM_HORIZONTAL_APERTURE = 20.955
    CAM_OFFSET_POS = (0.05, 0.0, 0.0)
    CAM_OFFSET_ROT = (0.5, -0.5, -0.5, 0.5)

    YOLO_WEIGHTS = "yolov8n-pose.pt"
    YOLO_CONF = 0.3
    YOLO_KP_CONF = 0.25

    # Position control (outermost loop) – FULL PID
    POS_DIST = 3.0                     # desired X gap behind person [m]
    POS_X_KP = 0.1                     # pitch P gain [rad/m]
    POS_X_KI = 0.05                    # pitch I gain [rad/(m·s)]
    POS_X_KD = 0.2                     # pitch D gain [rad/(m/s)]
    POS_Y_KP = 0.1                     # roll P gain [rad/m]
    POS_Y_KI = 0.05                    # roll I gain
    POS_Y_KD = 0.2                     # roll D gain
    MAX_POS_ANGLE = math.radians(15.0) # max pitch/roll from pos controller (15 deg)

    # Position control frequency (should be lowest)
    POS_CTRL_HZ = 20
    POS_CTRL_EVERY = SIM_HZ // POS_CTRL_HZ   # = 10

    # Visualization
    WINDOW_S = 30.0
    PLOT_EVERY = 5


# ========== CAMERA CONFIG ==========
def make_front_camera_cfg(prim_path="/World/Crazyflie/body/camera_front") -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=Cfg.CAM_WIDTH,
        height=Cfg.CAM_HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=Cfg.CAM_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=Cfg.CAM_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=Cfg.CAM_OFFSET_POS,
            rot=Cfg.CAM_OFFSET_ROT,
            convention="ros",
        ),
    )


# ========== PEDESTRIAN HELPERS ==========
def load_reference(stage, prim_path, candidates, wait_frames=240, label=""):
    for url in candidates:
        print(f"[LOAD:{label}] Trying: {url}")
        old = stage.GetPrimAtPath(prim_path)
        if old.IsValid():
            stage.RemovePrim(prim_path)
            for _ in range(10):
                simulation_app.update()
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(assetPath=url)
        success = False
        for _ in range(3):
            for _ in range(wait_frames//3):
                simulation_app.update()
            if len(list(prim.GetChildren())) > 0:
                success = True
                break
        if success:
            print(f"[LOAD:{label}] SUCCESS: {url}\n")
            return prim, url
        else:
            print(f"[LOAD:{label}] Failed, trying next...\n")
    return None, None

def find_anim_graph(stage):
    for guess in [f"{BIPED_ROOT}/CharacterAnimation/AnimationGraph", f"{BIPED_ROOT}/AnimationGraph"]:
        if stage.GetPrimAtPath(guess).IsValid():
            return guess
    for p in Usd.PrimRange.Stage(stage):
        if "AnimationGraph" in p.GetName() and str(p.GetPath()).startswith(BIPED_ROOT):
            return str(p.GetPath())
    return None

def set_world_transform(prim, x, y, yaw_deg, z=CHAR_Z_OFFSET):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(x, y, z))
    xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

def make_waypoints(n, radius, center=(3.0, 0.0), first_pos=(2.5, 0.0), seed=7):
    rng = random.Random(seed)
    pts = [first_pos]
    for _ in range(n-1):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.4, radius)
        x = center[0] + r * math.cos(a)
        y = center[1] + r * math.sin(a)
        pts.append((x, y))
    return pts

class PedestrianController:
    def __init__(self, char_prim, character_obj, waypoints):
        self.char_prim = char_prim
        self.character_obj = character_obj
        self.waypoints = waypoints
        self.wp_idx = 0
        self.x = waypoints[0][0]
        self.y = waypoints[0][1]
        self.z = CHAR_Z_OFFSET
        self.yaw = 0.0
        self._has_anim = character_obj is not None
        self._speed_var = None
        self._walk_var = None
        set_world_transform(self.char_prim, self.x, self.y, 0.0)
        self._log_target()

    def _current_target(self):
        return self.waypoints[self.wp_idx % len(self.waypoints)]

    def _log_target(self):
        tx, ty = self._current_target()
        print(f"[PED] Waypoint {self.wp_idx}: ({tx:.2f}, {ty:.2f})")

    def _try_set_var(self, name, value):
        try:
            self.character_obj.set_variable(name, value)
            return True
        except Exception:
            return False

    def _discover_and_set_speed(self, speed_value):
        if self._speed_var:
            self._try_set_var(self._speed_var, speed_value)
            return
        for name in ("Speed", "ForwardSpeed", "WalkSpeed", "speed"):
            if self._try_set_var(name, speed_value):
                self._speed_var = name
                print(f"[ANIM] Speed var: {name}")
                return
        if self._walk_var:
            self._try_set_var(self._walk_var, 1.0 if speed_value > 0 else 0.0)
            return
        for name in ("Walk", "IsWalking", "IsMoving"):
            if self._try_set_var(name, 1.0):
                self._walk_var = name
                print(f"[ANIM] Walk var: {name}")
                return

    def _set_idle(self):
        if self._speed_var:
            self._try_set_var(self._speed_var, 0.0)
        if self._walk_var:
            self._try_set_var(self._walk_var, 0.0)

    def step(self, dt):
        tx, ty = self._current_target()
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < ARRIVE_THRESH:
            if self._has_anim:
                self._set_idle()
            self.wp_idx += 1
            self._log_target()
            return
        step_dist = WALK_SPEED * dt
        self.x += (dx / dist) * step_dist
        self.y += (dy / dist) * step_dist
        raw_yaw = math.degrees(math.atan2(dy, dx))
        self.yaw = raw_yaw + YAW_OFFSET_DEG
        set_world_transform(self.char_prim, self.x, self.y, self.yaw)
        if self._has_anim:
            self._discover_and_set_speed(WALK_SPEED)

    def get_position(self):
        return self.x, self.y


# ========== POSITION CONTROLLER (FULL PID) ==========
class PositionPIDController:
    """Independent PID controllers for X (error→pitch) and Y (error→roll)."""
    def __init__(self, dt, kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, output_limits=(-0.5, 0.5)):
        self.dt = dt
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        self.kp_y = kp_y
        self.ki_y = ki_y
        self.kd_y = kd_y
        self.limits = output_limits  # rad
        self.reset()

    def reset(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

    def update(self, error_x, error_y):
        # P
        p_x = self.kp_x * error_x
        p_y = self.kp_y * error_y

        # I with anti-windup (clamp before applying output limit)
        self.integral_x += error_x * self.dt
        self.integral_y += error_y * self.dt
        i_x = self.ki_x * self.integral_x
        i_y = self.ki_y * self.integral_y

        # D
        d_x = self.kd_x * (error_x - self.prev_error_x) / self.dt if self.dt > 0 else 0.0
        d_y = self.kd_y * (error_y - self.prev_error_y) / self.dt if self.dt > 0 else 0.0

        # Raw output (pitch for X, roll for Y)
        out_x = p_x + i_x + d_x
        out_y = p_y + i_y + d_y

        # Limit and anti-windup
        if out_x > self.limits[1]:
            out_x = self.limits[1]
            self.integral_x -= error_x * self.dt   # prevent windup
        elif out_x < self.limits[0]:
            out_x = self.limits[0]
            self.integral_x -= error_x * self.dt

        if out_y > self.limits[1]:
            out_y = self.limits[1]
            self.integral_y -= error_y * self.dt
        elif out_y < self.limits[0]:
            out_y = self.limits[0]
            self.integral_y -= error_y * self.dt

        self.prev_error_x = error_x
        self.prev_error_y = error_y
        return out_x, out_y  # pitch_setpoint, roll_setpoint


# ========== YOLO + DETECTION (visual only) ==========
# (Giữ nguyên các hàm từ code gốc, nhưng không dùng để điều khiển)
_COCO_IDX = [0,5,6,11,12]
_KPT_COLORS = [(0,255,255),(0,200,255),(0,100,255),(0,255,100),(255,100,0)]
_SKELETON = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

def _extract_kpts(raw):
    if len(raw) >= 17:
        return [tuple(raw[i]) for i in _COCO_IDX]
    return [tuple(raw[i]) for i in range(min(5, len(raw)))]

def _is_valid_person(kpts5):
    conf = Cfg.YOLO_KP_CONF
    has_sho = any(kpts5[i][2] >= conf for i in [1,2])
    has_hip = any(kpts5[i][2] >= conf for i in [3,4])
    return has_sho and has_hip

def _draw_skeleton(bgr, kpts5):
    conf = Cfg.YOLO_KP_CONF
    for a,b in _SKELETON:
        xa, ya, ca = kpts5[a]
        xb, yb, cb = kpts5[b]
        if ca>conf and cb>conf:
            cv2.line(bgr, (int(xa),int(ya)), (int(xb),int(yb)), (180,180,180), 2, cv2.LINE_AA)
    for i,(x,y,c) in enumerate(kpts5):
        if c>conf:
            col = _KPT_COLORS[i%len(_KPT_COLORS)]
            cv2.circle(bgr, (int(x),int(y)), 5, col, -1, cv2.LINE_AA)
            cv2.circle(bgr, (int(x),int(y)), 5, (255,255,255), 1, cv2.LINE_AA)

def detect_person(frame_rgb, model):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h,w = bgr.shape[:2]
    results = model(frame_rgb, conf=Cfg.YOLO_CONF, device=0, verbose=False)
    annotated = bgr.copy()
    best_bbox = None
    max_area = 0

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        has_kpts = results[0].keypoints is not None
        if has_kpts:
            kpts_all = results[0].keypoints.data.cpu().numpy()
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1,y1,x2,y2 = map(int, box)
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area
                best_bbox = [x1,y1,x2,y2]
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(annotated, f"person {conf:.2f}", (x1, max(y1-5,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
            if has_kpts and i < len(kpts_all):
                kpts5 = _extract_kpts(kpts_all[i])
                if _is_valid_person(kpts5):
                    _draw_skeleton(annotated, kpts5)

    cx,cy = w//2, h//2
    cv2.drawMarker(annotated, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 20, 1)
    cv2.rectangle(annotated, (0, h-24), (w, h), (25,25,25), -1)
    track_info = "Tracking: " + ("ON" if best_bbox is not None else "NO PERSON")
    cv2.putText(annotated, track_info, (5, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return best_bbox, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ========== SETUP SIMULATION (sửa lỗi) ==========
def setup_simulation():
    # Sửa lỗi: args_cli.device không tồn tại → dùng "cpu"
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0/Cfg.SIM_HZ, device="cpu"))
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])
    stage = omni.usd.get_context().get_stage()

    # Ground and lights
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DistantLightCfg())
    dome_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.85,0.90,1.0))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    # Load pedestrian
    print("[PED] Loading character...")
    char_prim, _ = load_reference(stage, CHAR_ROOT, CHARACTER_CANDIDATES, wait_frames=240, label="CHAR")
    if char_prim is None:
        print("[FATAL] Cannot load character USD. Exiting.")
        simulation_app.close(); return None, None, None, None
    set_world_transform(char_prim, 0.0, 0.0, 0.0, z=CHAR_Z_OFFSET)
    for _ in range(30): simulation_app.update()

    print("[PED] Loading Biped Setup...")
    biped_prim, _ = load_reference(stage, BIPED_ROOT, BIPED_CANDIDATES, wait_frames=240, label="BIPED")
    character_obj = None
    if biped_prim is not None and ag is not None:
        from pxr import Sdf
        skel_root = None
        for p in Usd.PrimRange(stage.GetPrimAtPath(CHAR_ROOT)):
            if p.GetTypeName() == "SkelRoot":
                skel_root = p; break
        if skel_root:
            skel_path = str(skel_root.GetPath())
            graph_path = find_anim_graph(stage)
            if graph_path:
                try:
                    omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=[Sdf.Path(skel_path)])
                except: pass
                try:
                    omni.kit.commands.execute("ApplyAnimationGraphAPICommand",
                                              paths=[Sdf.Path(skel_path)],
                                              animation_graph_path=Sdf.Path(graph_path))
                    print("[ANIM] Animation graph applied.")
                except Exception as e:
                    print(f"[ANIM] Failed: {e}")
                for _ in range(60): simulation_app.update()
                character_obj = ag.get_character(skel_path)
                if character_obj:
                    print("[ANIM] Character object obtained.")
                else:
                    print("[ANIM] No character object.")
    waypoints = make_waypoints(N_WAYPOINTS, WAYPOINT_RADIUS, center=(3.0, 0.0), first_pos=(-10.0, 0.0))
    pedestrian = PedestrianController(char_prim, character_obj, waypoints)

    # Load drone – sửa lỗi: UAV_CFG là dataclass, cần copy và gán lại
    from dataclasses import replace
    robot_cfg = replace(UAV_CFG)
    robot_cfg.prim_path = "/World/Crazyflie"
    robot_cfg.init_state.pos = (0.0, 0.0, 0.05)
    # Spawn
    robot_cfg.spawn.func("/World/Crazyflie", translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    # Camera
    camera = Camera(make_front_camera_cfg())

    sim.reset()
    camera.reset()
    return sim, robot, camera, pedestrian


# ========== PLOTTING (giữ nguyên) ==========
def make_plots():
    plt.ion()
    fig = plt.figure("UAV Tracking", figsize=(18, 9))
    fig.suptitle("UAV Person Tracking  (blue=actual  red--=setpoint)", fontweight='bold')
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.axis('off')
    ax_cam.set_title("FPV Camera (YOLO)", fontsize=9)
    im_rgb = ax_cam.imshow(np.zeros((Cfg.CAM_HEIGHT, Cfg.CAM_WIDTH, 3), dtype=np.uint8))

    subplot_specs = [
        (gs[0, 1], "Z [m]"),
        (gs[1, 1], "X [m]"),
        (gs[2, 1], "Y [m]"),
        (gs[0, 2], "Pitch [°]"),
        (gs[1, 2], "Roll [°]"),
        (gs[2, 2], "Yaw [°]"),
    ]
    axes_data = []
    lines = []
    for spec, title in subplot_specs:
        ax = fig.add_subplot(spec)
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        l_act, = ax.plot([], [], 'b-', lw=1.5, label='actual')
        l_des, = ax.plot([], [], 'r--', lw=1.2, label='setpoint')
        ax.legend(fontsize=6, loc='upper left')
        axes_data.append(ax)
        lines.append((l_act, l_des))

    return fig, im_rgb, axes_data, lines

def update_plots(fig, im_rgb, axes_data, lines, camera, yolo_model, times,
                 zs, xs, ys, pitchs, rolls, yaws,
                 target_zs, target_xs, target_ys, pitch_des_hist, roll_sp_hist, yaw_des_hist):
    try:
        rgb = camera.data.output.get("rgb")
        if rgb is not None and rgb.shape[0] > 0:
            frame = rgb[0, ..., :3].cpu().numpy().astype(np.uint8)
            _, annotated = detect_person(frame, yolo_model)
            im_rgb.set_data(annotated)
    except Exception:
        pass

    if times:
        t = list(times)
        datasets = [
            (list(zs), list(target_zs)),
            (list(xs), list(target_xs)),
            (list(ys), list(target_ys)),
            (list(pitchs), list(pitch_des_hist)),
            (list(rolls), list(roll_sp_hist)),
            (list(yaws), list(yaw_des_hist)),
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


# ========== MAIN (3‑tier cascade) ==========
def main():
    print("=" * 60)
    print("UAV Person Tracking with Walking Pedestrian")
    print("3‑tier cascade control: Position (20Hz PID) → Attitude (50Hz) → Rate (200Hz)")
    print("=" * 60)

    ret = setup_simulation()
    if ret[0] is None:
        return
    sim, robot, camera, pedestrian = ret

    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv = make_alloc_inv(device=sim.device)
    dt = sim.get_physics_dt()          # 0.005 s (200 Hz)
    dt_outer = 1.0 / Cfg.OUTER_HZ      # 0.02 s (50 Hz)

    # Altitude PID
    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    # Attitude & rate controller (inner loops)
    ctrl = AttitudeController(
        robot=robot, prop_body_ids=prop_ids, root_body_ids=body_ids, A_inv=A_inv,
        att_kp=Cfg.ATT_KP, att_ki=Cfg.ATT_KI, att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP, rate_ki=Cfg.RATE_KI, rate_kd=Cfg.RATE_KD, rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP, yaw_rate_ki=Cfg.YAW_RATE_KI, yaw_rate_kd=Cfg.YAW_RATE_KD, yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE, max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT, max_yaw_moment=Cfg.MAX_YAW_MOMENT
    )

    # YOLO model (visual only)
    yolo_model = YOLO(Cfg.YOLO_WEIGHTS)
    print(f"[YOLO] Loaded {Cfg.YOLO_WEIGHTS}")

    # Position controller (full PID, outermost loop)
    pos_ctrl = PositionPIDController(
        dt=dt * Cfg.POS_CTRL_EVERY,   # dt for position loop (0.05 s at 20 Hz)
        kp_x=Cfg.POS_X_KP, ki_x=Cfg.POS_X_KI, kd_x=Cfg.POS_X_KD,
        kp_y=Cfg.POS_Y_KP, ki_y=Cfg.POS_Y_KI, kd_y=Cfg.POS_Y_KD,
        output_limits=(-Cfg.MAX_POS_ANGLE, Cfg.MAX_POS_ANGLE)
    )

    # Plotting setup
    fig, im_rgb, axes_data, lines = make_plots()
    maxlen = int(Cfg.WINDOW_S / dt) + 50
    def _dq(): return collections.deque(maxlen=maxlen)
    times = _dq(); zs = _dq(); xs = _dq(); ys = _dq()
    pitchs = _dq(); rolls = _dq(); yaws = _dq()
    target_zs = _dq(); target_xs = _dq(); target_ys = _dq()
    pitch_des_hist = _dq(); roll_sp_hist = _dq(); yaw_des_hist = _dq()

    sim_time = 0.0
    step = 0
    target_z = Cfg.TARGET_Z
    roll_sp = 0.0
    pitch_des = 0.0
    yaw_des = 0.0
    p_des_hold = q_des_hold = r_des_hold = 0.0
    cam_interval = max(1, int(Cfg.SIM_HZ / Cfg.CAM_UPDATE_HZ))
    ped_timer = 0.0
    data = {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

    # Variables for position error (for logging)
    px = py = drone_x = drone_y = error_x = error_y = 0.0

    while simulation_app.is_running():
        # ---- Altitude control (200 Hz) ----
        z = robot.data.root_pos_w[0][2].item()
        thrust = max(0.0, pid_z.update(target_z - z, dt))

        # ---- Pedestrian update (60 Hz) ----
        ped_timer += dt
        if ped_timer >= DT_PED:
            pedestrian.step(DT_PED)
            ped_timer = 0.0

        # ---- Position control (20 Hz, lowest frequency) ----
        if step % Cfg.POS_CTRL_EVERY == 0:
            px, py = pedestrian.get_position()
            drone_x = robot.data.root_pos_w[0][0].item()
            drone_y = robot.data.root_pos_w[0][1].item()
            # Desired X: person X - desired distance, desired Y: person Y
            error_x = (px - Cfg.POS_DIST) - drone_x
            error_y = py - drone_y
            # Get pitch and roll setpoints from PID
            pitch_des, roll_sp = pos_ctrl.update(error_x, error_y)

            # Yaw setpoint: face the person
            drone_yaw = data['yaw']
            heading_to_person = math.atan2(py - drone_y, px - drone_x)
            yaw_des = wrap_angle(heading_to_person - drone_yaw)

        # ---- Outer attitude loop (50 Hz) ----
        if step % Cfg.OUTER_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_outer(dt_outer, roll_sp, pitch_des, yaw_des)

        # ---- Inner rate loop (200 Hz) ----
        data = ctrl.step_inner(dt, p_des_hold, q_des_hold, r_des_hold, thrust=thrust)

        # ---- Step simulation ----
        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step += 1
        robot.update(dt)
        camera.update(dt)

        # Log every second
        if step % Cfg.SIM_HZ == 0:
            print(f"t={sim_time:5.1f}s | z={z:+.2f}m | "
                  f"yaw={math.degrees(data['yaw']):+5.1f}°(des={math.degrees(yaw_des):+5.1f}) | "
                  f"pitch={math.degrees(data['pitch']):+5.1f}°(des={math.degrees(pitch_des):+5.1f}) | "
                  f"roll={math.degrees(data['roll']):+5.1f}°(des={math.degrees(roll_sp):+5.1f}) | "
                  f"ped=({px:.2f},{py:.2f}) ex={error_x:+.2f} ey={error_y:+.2f}")

        # Collect data for plots
        times.append(sim_time)
        zs.append(z); target_zs.append(target_z)
        xs.append(drone_x); target_xs.append(px - Cfg.POS_DIST)
        ys.append(drone_y); target_ys.append(py)
        pitchs.append(math.degrees(data['pitch'])); pitch_des_hist.append(math.degrees(pitch_des))
        rolls.append(math.degrees(data['roll'])); roll_sp_hist.append(math.degrees(roll_sp))
        yaws.append(math.degrees(data['yaw'])); yaw_des_hist.append(math.degrees(yaw_des))

        if step % cam_interval == 0:
            update_plots(fig, im_rgb, axes_data, lines, camera, yolo_model, times,
                         zs, xs, ys, pitchs, rolls, yaws,
                         target_zs, target_xs, target_ys,
                         pitch_des_hist, roll_sp_hist, yaw_des_hist)

    simulation_app.close()

if __name__ == "__main__":
    main()