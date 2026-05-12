# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Rate PID Control — 1 tầng: điều khiển tốc độ góc (angular velocity).

VÒNG TRONG CÙNG của cascade. Tune file này TRƯỚC HẾT.
- Setpoint: tốc độ góc mong muốn (p_des, q_des, r_des) [rad/s]
- Feedback: angular velocity từ IMU/gyro (robot.data.root_ang_vel_w)
- Output: moment (Tx, Ty, Tz) → motor allocation
- Thrust không đổi (constant hover thrust)

Step test: setpoint đổi dấu mỗi STEP_EVERY giây.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/02_uav_pid_rate.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rate PID — 1 tầng (innermost loop).")
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
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext
from isaaclab_assets.uav.uav_cfg import UAV_CFG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import RateController, make_front_camera_cfg  # noqa: E402
from pid_controller import LivePlotter, make_alloc_inv        # noqa: E402


# === THAM SỐ ===
class Cfg:
    # Step test — chỉ bật 1 trục tại một thời điểm để tránh tích lũy tilt
    # Thứ tự tune: ROLL → PITCH → YAW (zero các trục còn lại)
    STEP_EVERY = 3.0                        # s — đổi dấu setpoint
    ROLL_RATE  = math.radians(20.0)         # rad/s  (set 0 khi đang tune pitch/yaw)
    PITCH_RATE = math.radians(0.0)          # rad/s  (set 0 khi đang tune roll/yaw)
    YAW_RATE   = math.radians(0.0)          # rad/s  (set 0 khi đang tune roll/pitch)

    # Rate PID gains
    ROLL_KP  = 0.0006;  ROLL_KI  = 0.0001;  ROLL_KD  = 0.00001;  ROLL_ILIM  = 0.5
    PITCH_KP = 0.0006;  PITCH_KI = 0.0001;  PITCH_KD = 0.00001;  PITCH_ILIM = 0.5
    # Yaw: I_z ≈ 1.6e-5 kg⋅m² → α = T/I_z rất lớn ⇒ giữ MAX nhỏ để tránh gyro coupling
    # MAX_YAW_MOMENT = 0.0003 Nm → α ≈ 19 rad/s² → rise time ~0.03s (đủ nhanh, ít coupling)
    # KP: err=0.524 rad/s → output = 0.0003 Nm → KP ≈ 0.0003/0.524 ≈ 0.0006
    YAW_KP   = 0.0006;  YAW_KI   = 0.00005; YAW_KD   = 0.0;      YAW_ILIM   = 0.1

    MAX_MOMENT     = 0.03    # Nm (roll/pitch)
    MAX_YAW_MOMENT = 0.0003  # Nm — giảm mạnh để tránh gyro coupling vào roll/pitch
    HOVER_THRUST   = 0.35    # N

    WINDOW_S   = 12.0
    PLOT_EVERY = 5


# === SETUP ===
def setup_simulation():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0)
    )

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 5.0)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    camera = Camera(make_front_camera_cfg())

    sim.reset()
    camera.reset()
    return sim, robot, camera


# === MAIN ===
def main():
    print("=" * 60)
    print("RATE PID — 1 tầng (innermost loop)")
    print(f"Step test: đổi dấu mỗi {Cfg.STEP_EVERY}s")
    print("=" * 60)

    sim, robot, camera = setup_simulation()
    prop_ids = robot.find_bodies("m.*_prop")[0]
    body_ids = robot.find_bodies("body")[0]
    A_inv    = make_alloc_inv(device=sim.device)
    dt       = sim.get_physics_dt()

    ctrl = RateController(
        robot=robot,
        prop_body_ids=prop_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        hover_thrust=Cfg.HOVER_THRUST,
        roll_kp=Cfg.ROLL_KP,   roll_ki=Cfg.ROLL_KI,   roll_kd=Cfg.ROLL_KD,   roll_lim=Cfg.ROLL_ILIM,
        pitch_kp=Cfg.PITCH_KP, pitch_ki=Cfg.PITCH_KI, pitch_kd=Cfg.PITCH_KD, pitch_lim=Cfg.PITCH_ILIM,
        yaw_kp=Cfg.YAW_KP,     yaw_ki=Cfg.YAW_KI,     yaw_kd=Cfg.YAW_KD,     yaw_lim=Cfg.YAW_ILIM,
        max_moment=Cfg.MAX_MOMENT,
        max_yaw_moment=Cfg.MAX_YAW_MOMENT,
    )

    plotter = LivePlotter(
        "Rate PID — 1 tầng (p/q/r → moment)",
        ["Roll rate [°/s]", "Pitch rate [°/s]", "Yaw rate [°/s]"],
        Cfg.WINDOW_S, dt,
    )

    sim_time  = 0.0
    step      = 0
    log_every = int(1.0 / dt)

    while simulation_app.is_running():
        sign  = 1.0 if (sim_time // Cfg.STEP_EVERY) % 2 == 0 else -1.0
        p_des = sign * Cfg.ROLL_RATE
        q_des = sign * Cfg.PITCH_RATE
        r_des = sign * Cfg.YAW_RATE

        data = ctrl.step(dt, p_des=p_des, q_des=q_des, r_des=r_des)

        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step     += 1
        robot.update(dt)
        camera.update(dt)

        if step % log_every == 0:
            print(
                f"t={sim_time:5.1f}s | "
                f"p={math.degrees(data['p']):+6.1f}°/s (des={math.degrees(p_des):+5.1f}) | "
                f"q={math.degrees(data['q']):+6.1f}°/s (des={math.degrees(q_des):+5.1f}) | "
                f"r={math.degrees(data['r']):+6.1f}°/s (des={math.degrees(r_des):+5.1f})"
            )

        if step % Cfg.PLOT_EVERY == 0:
            plotter.update(
                sim_time,
                [math.degrees(data["p"]), math.degrees(data["q"]), math.degrees(data["r"])],
                [math.degrees(p_des),     math.degrees(q_des),     math.degrees(r_des)],
            )

        pos = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[pos[0] - 1.5, pos[1] - 1.5, pos[2] + 0.8],
            target=[pos[0], pos[1], pos[2]],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
