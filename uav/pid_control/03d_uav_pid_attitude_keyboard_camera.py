# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
UAV tracking of H1 humanoid robot (RL flat policy) – 4‑tier cascade control.

Position (20Hz) → Velocity (50Hz) → Attitude (50Hz) → Rate (200Hz).
Camera + YOLO for visualization only — control is world‑frame position only.

Run:
    ./isaaclab.sh -p .../track_h1_position_4tier.py --policy_pt /path/to/policy.pt
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
from ultralytics import YOLO

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UAV tracking of H1 humanoid with RL flat policy")
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
from controller import PositionController   # 4‑tầng position → velocity → attitude → rate
from pid_controller import PIDController, make_alloc_inv, wrap_angle


# =============================================================================
# H1 CONFIGURATION
# =============================================================================
H1_VEL_X       = 1.0
H1_ARRIVE_THRESH = 1.5
H1_N_WAYPOINTS  = 8
H1_WP_RADIUS    = 5.0
H1_WP_SEED      = 42

def make_h1_waypoints(n=H1_N_WAYPOINTS, radius=H1_WP_RADIUS, seed=H1_WP_SEED):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.4, radius)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


# =============================================================================
# UAV CONFIGURATION
# =============================================================================
class Cfg:
    TARGET_Z = 1.5
    Z_KP = 2.5;  Z_KI = 0.85;  Z_KD = 0.52;  Z_ILIM = 1.0

    # Tần số
    SIM_HZ = 200
    DECIMATION = 4        # env step decimation (env.step mỗi 4 physics step)
    POS_CTRL_HZ = 20
    POS_CTRL_EVERY = SIM_HZ // POS_CTRL_HZ   # 10

    # Position setpoint
    POS_DIST = 3.0        # desired distance behind H1

    # Camera
    CAM_WIDTH  = 640
    CAM_HEIGHT = 480
    CAM_UPDATE_HZ = 10
    YOLO_WEIGHTS = "yolov8n-pose.pt"
    YOLO_CONF    = 0.3
    YOLO_KP_CONF = 0.25

    # Visualization
    WINDOW_S = 30.0


# =============================================================================
# CAMERA CONFIG
# =============================================================================
def make_front_camera_cfg(prim_path="/World/Crazyflie/body/camera_front") -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=Cfg.CAM_WIDTH,
        height=Cfg.CAM_HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


# =============================================================================
# YOLO DETECTION (visualization only)
# =============================================================================
_COCO_IDX    = [0, 5, 6, 11, 12]
_KPT_COLORS  = [(0,255,255),(0,200,255),(0,100,255),(0,255,100),(255,100,0)]
_SKELETON    = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

def _extract_kpts(raw):
    if len(raw) >= 17:
        return [tuple(raw[i]) for i in _COCO_IDX]
    return [tuple(raw[i]) for i in range(min(5, len(raw)))]

def _is_valid_person(kpts5):
    c = Cfg.YOLO_KP_CONF
    return any(kpts5[i][2] >= c for i in [1,2]) and any(kpts5[i][2] >= c for i in [3,4])

def _draw_skeleton(bgr, kpts5):
    c = Cfg.YOLO_KP_CONF
    for a, b in _SKELETON:
        xa, ya, ca = kpts5[a]; xb, yb, cb = kpts5[b]
        if ca > c and cb > c:
            cv2.line(bgr, (int(xa),int(ya)), (int(xb),int(yb)), (180,180,180), 2)
    for i, (x, y, conf) in enumerate(kpts5):
        if conf > c:
            cv2.circle(bgr, (int(x),int(y)), 5, _KPT_COLORS[i % len(_KPT_COLORS)], -1)

def detect_person(frame_rgb, model):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    results = model(frame_rgb, conf=Cfg.YOLO_CONF, device=0, verbose=False)
    annotated = bgr.copy()
    best_bbox = None
    max_area  = 0
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        has_kpts = results[0].keypoints is not None
        kpts_all = results[0].keypoints.data.cpu().numpy() if has_kpts else []
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = map(int, box)
            area = (x2-x1)*(y2-y1)
            if area > max_area:
                max_area = area; best_bbox = [x1,y1,x2,y2]
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(annotated, f"H1 {conf:.2f}", (x1, max(y1-5,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
            if has_kpts and i < len(kpts_all):
                kpts5 = _extract_kpts(kpts_all[i])
                if _is_valid_person(kpts5):
                    _draw_skeleton(annotated, kpts5)
    cx, cy = w//2, h//2
    cv2.drawMarker(annotated, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 20, 1)
    cv2.rectangle(annotated, (0, h-24), (w, h), (25,25,25), -1)
    label = "Tracking: " + ("ON" if best_bbox else "NO TARGET")
    cv2.putText(annotated, label, (5, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return best_bbox, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# =============================================================================
# SETUP SIMULATION
# =============================================================================
def setup_simulation(policy_pt_path: str):
    env_cfg = H1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 10.0
    env_cfg.curriculum = None
    env_cfg.episode_length_s = 1_000_000
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None

    device = getattr(args_cli, "device", "cuda:0")
    env_cfg.sim.device = device
    if device == "cpu":
        env_cfg.sim.use_fabric = False

    print("[H1] Creating ManagerBasedRLEnv ...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    policy = None
    if policy_pt_path:
        print(f"[POLICY] Loading {policy_pt_path}")
        policy = torch.jit.load(policy_pt_path, map_location=device)
        policy.eval()
        print("[POLICY] Loaded OK")
    else:
        print("[POLICY] No --policy_pt given; H1 will receive zero actions.")

    robot_cfg = replace(UAV_CFG)
    robot_cfg.prim_path = "/World/Crazyflie"
    robot_cfg.init_state.pos = (10.0, 10.0, 0.1)
    robot = Articulation(robot_cfg)

    camera = Camera(make_front_camera_cfg())
    env.sim.reset()
    robot.reset()
    camera.reset()
    obs, _ = env.reset()
    return env, policy, robot, camera, obs


# =============================================================================
# PLOTTING
# =============================================================================
def make_plots():
    plt.ion()
    fig = plt.figure("UAV H1 Tracking", figsize=(18, 9))
    fig.suptitle("UAV → H1 Tracking  (blue=actual  red--=setpoint)", fontweight="bold")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.axis("off")
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

def update_plots(fig, im_rgb, axes_data, lines, camera, yolo_model, times,
                 zs, xs, ys, pitchs, rolls, yaws,
                 target_zs, target_xs, target_ys,
                 pitch_des_hist, roll_sp_hist, yaw_des_hist):
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


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("UAV → H1 Tracking  (4‑tier cascade: Position → Velocity → Attitude → Rate)")
    print("Position 20Hz → Velocity/Attitude 50Hz → Rate 200Hz")
    print("=" * 60)

    env, policy, robot, camera, obs = setup_simulation(args_cli.policy_pt)

    physics_dt = env.sim.get_physics_dt()          # 0.005 s
    decimation = env.cfg.decimation                # 4
    env_step_dt = decimation * physics_dt          # 0.02 s (50 Hz)

    print(f"[TIMING] physics_dt={physics_dt*1000:.1f}ms  decimation={decimation}")

    device = env.device
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv = make_alloc_inv(device=device)

    # Altitude PID riêng
    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    # ── PositionController 4 tầng ──
    pos_ctrl = PositionController(
        robot=robot, prop_body_ids=prop_ids, root_body_ids=body_ids, A_inv=A_inv,
        hover_thrust=0.35,
        # Position gains (tầng ngoài cùng)
        pos_kp_x=0.01, pos_ki_x=0.00, pos_kd_x=0.001, pos_lim_x=0.5,
        pos_kp_y=0.01, pos_ki_y=0.00, pos_kd_y=0.001, pos_lim_y=0.5,
        max_vel=1.0,
        # Velocity gains (vxy → angle)
        vxy_kp=0.05, vxy_ki=0.05, vxy_kd=0.2, vxy_lim=0.5, max_tilt=math.radians(10.0),
        # Attitude gains (angle → rate)
        att_kp=5.5, att_ki=0.55, att_kd=1.2,
        yaw_att_kp=1.0, yaw_att_ki=0.0, yaw_att_kd=0.0,
        # Rate gains (rate → moment)
        rate_kp=0.0002, rate_ki=0.00015, rate_kd=0.0000185, rate_lim=1.0,
        yaw_rate_kp=0.00015, yaw_rate_ki=0.0005, yaw_rate_kd=0.00001, yaw_rate_lim=0.2,
        max_rate=math.radians(180.0), max_yaw_rate=math.radians(90.0),
        max_moment=0.03, max_yaw_moment=0.0003,
    )

    yolo_model = YOLO(Cfg.YOLO_WEIGHTS)
    print(f"[YOLO] Loaded {Cfg.YOLO_WEIGHTS}")

    # H1 waypoints + policy action placeholder
    waypoints = make_h1_waypoints()
    wp_idx = 0
    n_act = env.action_space.shape[-1]
    h1_action = torch.zeros(1, n_act, device=device)
    h1_heading = 0.0

    # Plotting
    fig, im_rgb, axes_data, lines = make_plots()
    maxlen = int(Cfg.WINDOW_S / physics_dt) + 200
    def _dq(): return collections.deque(maxlen=maxlen)
    times = _dq(); zs = _dq(); xs = _dq(); ys = _dq()
    pitchs = _dq(); rolls = _dq(); yaws = _dq()
    target_zs = _dq(); target_xs = _dq(); target_ys = _dq()
    pitch_des_hist = _dq(); roll_sp_hist = _dq(); yaw_des_hist = _dq()

    # State variables
    sim_time = 0.0
    phys_step = 0
    h1_x = h1_y = drone_x = drone_y = 0.0
    vx_des = 0.0
    vy_des = 0.0
    roll_des = 0.0
    pitch_des = 0.0
    yaw_sp = 0.0
    p_des = 0.0
    q_des = 0.0
    r_des = 0.0
    last_pos_update_step = -Cfg.POS_CTRL_EVERY
    last_vel_update_step = -decimation

    cam_interval = max(1, round(1.0 / (physics_dt * Cfg.CAM_UPDATE_HZ)))

    while simulation_app.is_running():
        # ---- H1 policy (mỗi decimation step) ----
        if phys_step % decimation == 0:
            h1_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            h1_x, h1_y = h1_pos[0].item(), h1_pos[1].item()
            tx, ty = waypoints[wp_idx % len(waypoints)]
            if math.hypot(tx - h1_x, ty - h1_y) < H1_ARRIVE_THRESH:
                wp_idx += 1
                tx, ty = waypoints[wp_idx % len(waypoints)]
                print(f"[H1] Waypoint {wp_idx}: ({tx:.1f}, {ty:.1f})")
            h1_heading = math.atan2(ty - h1_y, tx - h1_x)

            with torch.inference_mode():
                if policy is not None:
                    h1_action = policy(obs["policy"])
                else:
                    h1_action = torch.zeros(1, n_act, device=device)
            env.action_manager.process_action(h1_action)

        # ---- Read drone state ----
        z = robot.data.root_pos_w[0][2].item()
        drone_x = robot.data.root_pos_w[0][0].item()
        drone_y = robot.data.root_pos_w[0][1].item()

        # ---- Altitude PID (200Hz) ----
        thrust = max(0.0, pid_z.update(Cfg.TARGET_Z - z, physics_dt))

        # ---- Position control (20Hz) – tính vx_des, vy_des ----
        if phys_step - last_pos_update_step >= Cfg.POS_CTRL_EVERY:
            target_x = h1_x - Cfg.POS_DIST
            target_y = h1_y
            vx_des, vy_des = pos_ctrl.step_pos(
                physics_dt * Cfg.POS_CTRL_EVERY,
                target_x, target_y, drone_x, drone_y
            )
            # Yaw setpoint: hướng về H1
            heading_to_h1 = math.atan2(h1_y - drone_y, h1_x - drone_x)
            yaw_sp = heading_to_h1
            last_pos_update_step = phys_step

        # ---- Velocity + Attitude tầng (50Hz) – tính rate_des ----
        if phys_step % decimation == 0:
            # Tầng velocity: vx_des, vy_des → roll_des, pitch_des
            _, roll_des, pitch_des = pos_ctrl.step_vel(env_step_dt, vx_des, vy_des, thrust, yaw_sp)
            # Tầng attitude: angle → rate_des
            p_des, q_des, r_des = pos_ctrl.step_att(env_step_dt, roll_des, pitch_des, yaw_sp)

        # ---- Rate tầng (200Hz) – tính moment và áp lực ----
        data = pos_ctrl.step_rate(physics_dt, p_des, q_des, r_des, thrust)

        # ---- Physics step ----
        robot.write_data_to_sim()
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        is_render = (phys_step % env.cfg.sim.render_interval == 0)
        env.sim.step(render=is_render)
        env.scene.update(dt=physics_dt)
        robot.update(physics_dt)
        camera.update(physics_dt)

        # ---- Update observation (mỗi decimation step) ----
        if (phys_step + 1) % decimation == 0:
            obs_buf = env.observation_manager.compute()
            obs = {"policy": obs_buf["policy"]}
            with torch.inference_mode():
                obs["policy"][:, 9:13] = torch.tensor(
                    [[H1_VEL_X, 0.0, 0.0, h1_heading]], device=device
                )

        sim_time += physics_dt
        phys_step += 1

        # Logging mỗi giây
        if phys_step % round(1.0 / physics_dt) == 0:
            print(f"t={sim_time:5.1f}s | z={z:+.2f}m | "
                  f"yaw={math.degrees(data['yaw']):+5.1f}°(sp={math.degrees(yaw_sp):+5.1f}) | "
                  f"pitch={math.degrees(data['pitch']):+5.1f}°(sp={math.degrees(pitch_des):+5.1f}) | "
                  f"roll={math.degrees(data['roll']):+5.1f}°(sp={math.degrees(roll_des):+5.1f}) | "
                  f"H1=({h1_x:.1f},{h1_y:.1f}) ex={(h1_x - Cfg.POS_DIST - drone_x):+.2f} ey={(h1_y - drone_y):+.2f}")

        # Thu thập dữ liệu vẽ đồ thị
        times.append(sim_time)
        zs.append(z); target_zs.append(Cfg.TARGET_Z)
        xs.append(drone_x); target_xs.append(h1_x - Cfg.POS_DIST)
        ys.append(drone_y); target_ys.append(h1_y)
        pitchs.append(math.degrees(data["pitch"])); pitch_des_hist.append(math.degrees(pitch_des))
        rolls.append(math.degrees(data["roll"]));   roll_sp_hist.append(math.degrees(roll_des))
        yaws.append(math.degrees(data["yaw"]));     yaw_des_hist.append(math.degrees(yaw_sp))

        if phys_step % cam_interval == 0:
            update_plots(fig, im_rgb, axes_data, lines, camera, yolo_model, times,
                         zs, xs, ys, pitchs, rolls, yaws,
                         target_zs, target_xs, target_ys,
                         pitch_des_hist, roll_sp_hist, yaw_des_hist)
            
        pos = robot.data.root_pos_w[0]
        p = pos.cpu().numpy()
        env.sim.set_camera_view(
            eye=[p[0] - 1.5, p[1] - 1.5, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )

    simulation_app.close()

if __name__ == "__main__":
    main()