# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Altitude PID Control — điều khiển độ cao Z bằng PID.
Đầu ra PID điều khiển trực tiếp lực/moment vật lý (N, Nm).

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/01_uav_pid_altitude.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Altitude PID control — direct force.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim thêm pip_prebundle (numpy cũ) vào sys.path — filter ra trước khi import.
import sys
sys.path = [p for p in sys.path if "pip_prebundle" not in p]

"""Rest everything follows."""

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationContext

from isaaclab_assets.uav.uav_cfg import UAV_CFG

# Thêm uav/ vào sys.path để import controller
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import AltitudeAttitudeController, make_front_camera_cfg  # noqa: E402
from pid_controller import LivePlotter, make_alloc_inv  # noqa: E402

try:
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    _NUCLEUS_ROOT = get_assets_root_path() or ""
except Exception:
    _NUCLEUS_ROOT = ""

HUMAN_USD_PATH = f"{_NUCLEUS_ROOT}/Isaac/People/Characters/biped_demo/biped_demo.usd"


# === THAM SỐ ĐIỀU KHIỂN ===
class ControlConfig:
    TARGET_ALTITUDE = 1.0   # [m]

    Z_KP,   Z_KI,   Z_KD   = 1.5,  0.7,  0.12
    ATT_KP, ATT_KI, ATT_KD = 0.01, 0.00, 0.02
    YAW_KP, YAW_KI, YAW_KD = 0.01, 0.00, 0.01

    WIND_STD      = 0.0     # [N] nhiễu gió (0 = tắt)
    WIND_INTERVAL = 100     # áp gió mỗi N step


# === SETUP MÔI TRƯỜNG ===
def setup_simulation():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.5, 1.5], target=[0.0, 0.0, 1.0])

    ground_cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(0.95, 0.95, 0.95))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0))
    dome_cfg.func("/World/DomeLight", dome_cfg)

    if _NUCLEUS_ROOT:
        human_cfg = sim_utils.UsdFileCfg(usd_path=HUMAN_USD_PATH)
        human_cfg.func("/World/Human", human_cfg, translation=(1.5, 0.0, 0.0))
    else:
        print("[WARN] Không tìm thấy nucleus — bỏ qua spawn người.")

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    camera = Camera(make_front_camera_cfg())

    sim.reset()
    camera.reset()
    return sim, robot, camera


# === MAIN ===
def main():
    print("[INFO] Initializing simulation...")
    sim, robot, camera = setup_simulation()

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    body_ids      = robot.find_bodies("body")[0]
    A_inv         = make_alloc_inv(device=sim.device)

    ctrl = AltitudeAttitudeController(
        robot=robot,
        prop_body_ids=prop_body_ids,
        root_body_ids=body_ids,
        A_inv=A_inv,
        z_kp=ControlConfig.Z_KP,     z_ki=ControlConfig.Z_KI,   z_kd=ControlConfig.Z_KD,
        att_kp=ControlConfig.ATT_KP, att_ki=ControlConfig.ATT_KI, att_kd=ControlConfig.ATT_KD,
        yaw_kp=ControlConfig.YAW_KP, yaw_ki=ControlConfig.YAW_KI, yaw_kd=ControlConfig.YAW_KD,
    )

    sim_dt  = sim.get_physics_dt()
    plotter = LivePlotter(
        title="UAV Altitude PID — Z / Roll / Pitch / Yaw",
        ylabels=["Z [m]", "Roll [rad]", "Pitch [rad]", "Yaw [rad]"],
        window_s=20.0,
        dt=sim_dt,
    )

    sim_time   = 0.0
    step_count = 0
    log_every  = int(1.0 / sim_dt)

    print(f"[INFO] Target altitude = {ControlConfig.TARGET_ALTITUDE} m")

    while simulation_app.is_running():
        data = ctrl.step(sim_dt, target_z=ControlConfig.TARGET_ALTITUDE)

        # Gió nhiễu tuỳ chọn
        if ControlConfig.WIND_STD > 0 and step_count % ControlConfig.WIND_INTERVAL == 0:
            wind = torch.randn(1, 1, 3, device=sim.device) * ControlConfig.WIND_STD
            robot.set_external_force_and_torque(
                forces=wind, torques=torch.zeros_like(wind), body_ids=body_ids
            )

        robot.write_data_to_sim()
        sim.step()

        # Clear external force sau mỗi step tránh Isaac Lab re-apply
        if ControlConfig.WIND_STD > 0:
            zeros = torch.zeros(1, 1, 3, device=sim.device)
            robot.set_external_force_and_torque(forces=zeros, torques=zeros, body_ids=body_ids)

        sim_time   += sim_dt
        step_count += 1
        robot.update(sim_dt)
        camera.update(sim_dt)

        if step_count % log_every == 0:
            print(
                f"t={sim_time:5.1f}s | z={data['z']:+.3f}m | err={data['z_err']:+.3f}m | "
                f"thrust={data['thrust']:.4f}N | roll={data['roll']:+.3f} pitch={data['pitch']:+.3f}"
            )

        if step_count % 5 == 0:
            plotter.update(
                t=sim_time,
                actual_vals=[data["z"],    data["roll"],  data["pitch"],  data["yaw"]],
                target_vals=[ControlConfig.TARGET_ALTITUDE, 0.0, 0.0, 0.0],
            )

        p = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=(p[0] - 1.0, p[1] - 1.0, p[2] + 1.0),
            target=(p[0], p[1], p[2]),
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
