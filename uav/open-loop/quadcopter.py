# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from isaaclab_assets.uav.uav_cfg import UAV_CFG  # isort:skip
from controller import apply_prop_wrench  # isort:skip


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    root_body_ids = robot.find_bodies("body")[0]  # main drone frame
    print(f"[INFO]: body_names = {robot.body_names}")
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    prop_angles = robot.data.default_joint_pos.clone()
    prop_vel    = robot.data.default_joint_vel.clone()
    # Crazyflie torque-to-thrust ratio (CT_tau / CT_f) from Bitcraze firmware
    CT_RATIO = 0.005971  # meters
    # Chiều quay từng prop: m1 CCW(+), m2 CW(-), m3 CCW(+), m4 CW(-)
    prop_spin_dirs = torch.tensor([1.0, -1.0, 1.0, -1.0], device=sim.device)
    SPIN_TORQUE = 0.01  # Nm — tăng giá trị này để props quay nhanh hơn
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 20000 == 0 and count > 0:
            # reset counters
            sim_time = 0.0
            count = 0
            prop_angles = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(prop_angles, prop_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")

        # Ramp up to hover thrust over 5s, then hold
        thrust_scale = 2.0
        F_base =  robot_mass * gravity / 4.0

        # Yaw command: sinusoidal, chỉ bật sau khi hover ổn định
        yaw_frac = 0.003 * math.sin(2 * math.pi * sim_time / 4.0) if thrust_scale >= 1.0 else 0.0
    
        apply_prop_wrench(
            robot=robot,
            prop_body_ids=prop_body_ids,
            thrust_per_prop=F_base,
            prop_spin_dirs=prop_spin_dirs,
            spin_torque=SPIN_TORQUE,
            root_body_ids=root_body_ids,
            net_yaw=CT_RATIO * yaw_frac * F_base,
        )
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)
        drone_pos = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(
            eye=[drone_pos[0], drone_pos[1], drone_pos[2] + 0.5],
            target=[drone_pos[0], drone_pos[1], drone_pos[2]],
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
