# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Attitude PID Control — keyboard interactive (dùng Se2Keyboard của Isaac Lab).

Kiến trúc:
    z_err     → [PID_z]    → thrust        (200 Hz)
    angle_err → [PID_att]  → rate_des      (50 Hz, outer)
    rate_err  → [PID_rate] → moment        (200 Hz, inner)

Bàn phím (cửa sổ Isaac Sim phải được focus):
    Arrow Up / Down   →  pitch  -/+ (nghiêng tiến / lùi)
    Arrow Left / Right→  roll   -/+ (nghiêng trái / phải)
    Z / X             →  yaw    -/+ (xoay trái / phải)
    R                 →  reset drone về vị trí ban đầu
    L                 →  reset lệnh về 0

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/03b_uav_pid_attitude_keyboard.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attitude PID keyboard control cho Crazyflie.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext
from isaaclab_assets.uav.uav_cfg import UAV_CFG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import AttitudeController, make_front_camera_cfg  # noqa: E402
from pid_controller import PIDController, make_alloc_inv, wrap_angle, LivePlotter  # noqa: E402


# === THAM SỐ ===
class Cfg:
    # Altitude hold
    INIT_Z = 1.0
    Z_KP   = 2.5;  Z_KI = 0.85;  Z_KD = 0.52;  Z_ILIM = 1.0

    # Tầng ngoài — angle → rate_des
    ATT_KP     = 5.5;  ATT_KI     = 0.55;  ATT_KD     = 1.2
    YAW_ATT_KP = 1.0;  YAW_ATT_KI = 0.0;   YAW_ATT_KD = 0.0

    # Tầng trong — rate → moment
    RATE_KP     = 0.0002;  RATE_KI     = 0.00015;  RATE_KD     = 0.0000185; RATE_LIM     = 1.0
    YAW_RATE_KP = 0.00015; YAW_RATE_KI = 0.0005;   YAW_RATE_KD = 0.00001;   YAW_RATE_LIM = 0.2

    MAX_RATE       = math.radians(180.0)
    MAX_YAW_RATE   = math.radians(90.0)
    YAW_RATE_LIMIT = math.radians(30.0)   # tốc độ ramp yaw_sp [rad/s]
    MAX_MOMENT     = 0.03
    MAX_YAW_MOMENT = 0.0003

    # Multi-rate ZOH
    SIM_HZ    = 200
    OUTER_HZ  = 50
    OUTER_EVERY = SIM_HZ // OUTER_HZ   # = 4 bước

    # Keyboard — góc tối đa khi giữ phím [deg]
    ROLL_MAX  = 20.0
    PITCH_MAX = 20.0
    YAW_RATE  = 60.0   # [deg/s] tốc độ xoay yaw khi giữ Q/E

    WINDOW_S   = 30.0
    PLOT_EVERY = 5


# === SETUP ===
def setup_simulation():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / Cfg.SIM_HZ, device=args_cli.device))
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot  = Articulation(robot_cfg)
    camera = Camera(make_front_camera_cfg())

    sim.reset()
    camera.reset()
    return sim, robot, camera


# === MAIN ===
def main():
    print("=" * 60)
    print("ATTITUDE PID — Arrow Up/Down=pitch  Left/Right=roll  Z/X=yaw  R=reset")
    print("=" * 60)

    sim, robot, camera = setup_simulation()
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv    = make_alloc_inv(device=sim.device)
    dt       = sim.get_physics_dt()
    dt_outer = 1.0 / Cfg.OUTER_HZ

    pid_z = PIDController(Cfg.Z_KP, Cfg.Z_KI, Cfg.Z_KD, integral_limit=Cfg.Z_ILIM)

    ctrl = AttitudeController(
        robot=robot,
        prop_body_ids=prop_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        att_kp=Cfg.ATT_KP,         att_ki=Cfg.ATT_KI,         att_kd=Cfg.ATT_KD,
        yaw_att_kp=Cfg.YAW_ATT_KP, yaw_att_ki=Cfg.YAW_ATT_KI, yaw_att_kd=Cfg.YAW_ATT_KD,
        rate_kp=Cfg.RATE_KP,           rate_ki=Cfg.RATE_KI,           rate_kd=Cfg.RATE_KD,           rate_lim=Cfg.RATE_LIM,
        yaw_rate_kp=Cfg.YAW_RATE_KP,   yaw_rate_ki=Cfg.YAW_RATE_KI,   yaw_rate_kd=Cfg.YAW_RATE_KD,   yaw_rate_lim=Cfg.YAW_RATE_LIM,
        max_rate=Cfg.MAX_RATE,
        max_yaw_rate=Cfg.MAX_YAW_RATE,
        max_moment=Cfg.MAX_MOMENT,
        max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    plotter = LivePlotter(
        "Attitude Keyboard — Z / Roll / Pitch / Yaw",
        ["Z [m]", "Roll [°]", "Pitch [°]", "Yaw [°]"],
        Cfg.WINDOW_S, dt,
    )

    # Se2Keyboard: UP/DOWN=pitch, LEFT/RIGHT=roll, Z/X=yaw
    # sensitivity âm để: UP → pitch_d < 0 (lean forward), LEFT → roll_d < 0 (lean left)
    kb = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=    -math.radians(Cfg.PITCH_MAX),
        v_y_sensitivity=    -math.radians(Cfg.ROLL_MAX),
        omega_z_sensitivity=-1.0,
        sim_device=sim.device,
    ))

    _do_reset = False

    def _on_reset():
        nonlocal _do_reset
        _do_reset = True

    kb.add_callback("R", _on_reset)
    print("[INFO] Arrow Up/Down=pitch  Left/Right=roll  Z/X=yaw  R=reset  L=zero  (focus Isaac Sim window)")

    sim_time   = 0.0
    step       = 0
    log_hz     = Cfg.SIM_HZ
    yaw_sp     = 0.0
    target_z   = Cfg.INIT_Z
    # ZOH buffers
    p_des_hold = q_des_hold = r_des_hold = 0.0

    while simulation_app.is_running():
        cmd      = kb.advance()
        pitch_d  = cmd[0].item()   # UP=-PITCH_MAX rad, DOWN=+PITCH_MAX rad
        roll_d   = cmd[1].item()   # LEFT=-ROLL_MAX rad, RIGHT=+ROLL_MAX rad
        dyaw_dir = cmd[2].item()   # Z=-1 (yaw left), X=+1 (yaw right)
        yaw_sp  += math.radians(Cfg.YAW_RATE) * dyaw_dir * dt

        if _do_reset:
            _do_reset  = False
            sim_time   = 0.0
            step       = 0
            yaw_sp     = 0.0
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            ctrl.reset()
            pid_z.reset()
            kb.reset()
            p_des_hold = q_des_hold = r_des_hold = 0.0
            print(">>>>>>>> Reset!")

        # Altitude PID — 200 Hz
        z      = robot.data.root_pos_w[0][2].item()
        thrust = max(0.0, pid_z.update(target_z - z, dt))

        # Outer att — 50 Hz ZOH
        if step % Cfg.OUTER_EVERY == 0:
            p_des_hold, q_des_hold, r_des_hold = ctrl.step_outer(
                dt_outer, roll_d, pitch_d, yaw_sp
            )

        # Inner rate — 200 Hz
        data = ctrl.step_inner(dt, p_des_hold, q_des_hold, r_des_hold, thrust=thrust)

        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step     += 1
        robot.update(dt)
        camera.update(dt)

        if step % log_hz == 0:
            print(
                f"t={sim_time:5.1f}s | z={z:+.2f}m | "
                f"roll={math.degrees(data['roll']):+6.1f}°(sp={math.degrees(roll_d):+5.1f}°) | "
                f"pitch={math.degrees(data['pitch']):+6.1f}°(sp={math.degrees(pitch_d):+5.1f}°) | "
                f"yaw={math.degrees(data['yaw']):+6.1f}°(sp={math.degrees(yaw_sp):+5.1f}°)"
            )

        if step % Cfg.PLOT_EVERY == 0:
            plotter.update(
                sim_time,
                [z, math.degrees(data["roll"]), math.degrees(data["pitch"]), math.degrees(data["yaw"])],
                [target_z, math.degrees(roll_d), math.degrees(pitch_d), math.degrees(yaw_sp)],
            )

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] - 1.5, p[1] - 1.5, p[2] + 0.8],
            target=[p[0], p[1], p[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
