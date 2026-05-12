"""Hover environment for UAV (Crazyflie quadcopter).

Robot: Crazyflie 2.x
Task:  Reach and maintain a randomly sampled target position.
       Policy outputs 4 values: [thrust, moment_roll, moment_pitch, moment_yaw]
       applied as external force+torque on body (same as quadcopter_env.py DirectRL).

Train (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-UAV-Hover \\
        --num_envs 4096 --headless

Play:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Isaac-UAV-Hover \\
        --num_envs 4
"""
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import observations, events, rewards, terminations, commands

from .uav_cfg import UAV_CFG
from . import mdp

CAM_UPDATE_HZ = 30  # Hz


# ─────────────────────────── Scene ────────────────────────────────────────────

@configclass
class UAVSceneCfg(InteractiveSceneCfg):
    """Scene cho UAV hover."""

    num_envs: int = 1
    replicate_physics: bool = True

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1200.0),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: Articulation = UAV_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1))
    )

    # ── Camera trước (OV2640 style) ─────────────────────────────────────────
    camera_front: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera_front",

        update_period=1.0 / CAM_UPDATE_HZ,

        # OV2640 thường chạy VGA/SVGA
        width=640,
        height=480,

        data_types=["rgb", "distance_to_image_plane"],

        spawn=sim_utils.PinholeCameraCfg(

            # Lens OV2640 phổ biến ~2.8mm
            focal_length=2.8,

            # focus xa vừa phải
            focus_distance=400.0,

            # Sensor 1/4"
            # width thực ~3.6mm
            horizontal_aperture=3.6,

            clipping_range=(0.1, 100.0),
        ),

        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),

            # ROS camera frame
            rot=(0.5, -0.5, 0.5, -0.5),

            convention="ros",
        ),
    )


# ─────────────────────────── Actions ──────────────────────────────────────────

@configclass
class ActionCfg:
    """Thrust + moment trực tiếp lên body (giống quadcopter_env DirectRL).

    action[0]   = collective thrust [-1, 1]
    action[1:4] = moments [roll, pitch, yaw] [-1, 1]
    """

    thrust = mdp.UAVThrustActionCfg(
        asset_name="robot",
        body_name="body",
        thrust_to_weight=1.9,
        moment_scale=0.01,
    )


# ─────────────────────────── Commands ─────────────────────────────────────────

@configclass
class CommandsCfg:
    """Vị trí mục tiêu hover (x, y, z) trong world frame."""

    target_pos = mdp.UAVTargetPosCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 20.0),
        debug_vis=True,
        ranges=mdp.UAVTargetPosCommandCfg.Ranges(
            pos_x=(-10.0, 10.0),
            pos_y=(-10.0, 10.0),
            pos_z=(0.2, 10.0),     # độ cao hover [m] — world frame tuyệt đối
        ),
    )


# ─────────────────────────── Observations ─────────────────────────────────────

@configclass
class ObservationsCfg:
    """Observations cho policy và critic."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Body-frame observations (19 dims):
        lin_vel_b(3) + ang_vel_b(3) + projected_gravity_b(3)
        + desired_pos_b(3) + root_quat_w(4) + last_action(4)

        Tất cả velocity/gravity/target đều trong body frame.
        """

        # Velocities in body frame (_w → _b)
        lin_vel_b         = ObservationTermCfg(func=observations.base_lin_vel)
        ang_vel_b         = ObservationTermCfg(func=observations.base_ang_vel)

        # Gravity projected into body frame (orientation indicator)
        projected_gravity = ObservationTermCfg(func=observations.projected_gravity)

        # Target position in body frame (thay thế target_cmd world frame)
        desired_pos_b     = ObservationTermCfg(
            func=mdp.desired_pos_b,
            params={"command_name": "target_pos"},
        )

        # Orientation (world frame — cần thiết cho yaw awareness)
        root_quat_w       = ObservationTermCfg(func=observations.root_quat_w)

        # Last action (temporal smoothness)
        last_action       = ObservationTermCfg(func=observations.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────── Events ───────────────────────────────────────────

@configclass
class EventCfg:
    """Reset events."""

    reset_position = EventTermCfg(
        func=events.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (1.0, 1.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
            },
        },
    )

    reset_joints = EventTermCfg(
        func=events.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (-10.0, 10.0),
        },
    )


# ─────────────────────────── Rewards ──────────────────────────────────────────

@configclass
class RewardCfg:
    """Reward terms cho hover task.

    Positive rewards:
        distance_to_goal_exp  — khuyến khích tiến gần target

    Penalties:
        termination_penalty   — phạt nặng khi terminate sớm
        lin_vel_l2            — giảm vận tốc tịnh tiến (hover ổn định)
        ang_vel_l2            — giảm vận tốc góc
        upright               — phạt nghiêng roll/pitch
        action_rate           — phạt thay đổi action đột ngột
    """

    termination_penalty = RewardTermCfg(
        func=rewards.is_terminated,
        weight=-10000.0,          
    )

    # distance_to_goal = RewardTermCfg(
    #     func=mdp.rewards.distance_to_goal_tanh,
    #     weight=20.0,
    #     params={"command_name": "target_pos", "scale": 0.5},
    # )

    distance_to_goal = RewardTermCfg(
        func=mdp.rewards.distance_to_goal_l1,
        weight=-10.0,
        params={"command_name": "target_pos"},
    )


    lin_vel = RewardTermCfg(
        func=mdp.rewards.lin_vel_xyz_l2,
        weight=-0.05,
    )

    ang_vel = RewardTermCfg(
        func=mdp.rewards.ang_vel_xyz_l2,
        weight=-0.01,
    )

    rpy_alignment = RewardTermCfg(
        func=mdp.rewards.rpy_alignment,
        weight=15.0,             # reward [0,1]: = 1 khi thẳng đứng, 0 khi lật
        params={
            "target_rpy":    (0.0, 0.0, 0.0),
            "tolerance":     0.05,           # ≈ 3° dead zone
            "scale":         3.0,
            "axis_weights":  (5.0, 5.0, 0.2),  # roll/pitch > yaw
            "asset_cfg":     SceneEntityCfg("robot"),
        },
    )

    action_rate = RewardTermCfg(
        func=mdp.rewards.action_rate_l2,
        weight=-0.05,           # tăng 5×: tránh chattering
    )


# ─────────────────────────── Terminations ─────────────────────────────────────

@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = TerminationTermCfg(
        func=terminations.time_out,
        time_out=True,
    )

    # UAV ngã quá nghiêng (> 90 deg)
    bad_orientation = TerminationTermCfg(
        func=terminations.bad_orientation,
        params={
            "limit_angle": math.pi / 2,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # UAV chạm đất
    crashed = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": 0.1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )

    # UAV bay quá xa (out of bounds)
    out_of_bounds = TerminationTermCfg(
        func=terminations.root_height_below_minimum,
        params={
            "minimum_height": -0.1,
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )


# ─────────────────────────── Env ──────────────────────────────────────────────

@configclass
class UAVHoverEnvCfg(ManagerBasedRLEnvCfg):
    """Environment config cho UAV hover task."""

    scene:        UAVSceneCfg      = UAVSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionCfg        = ActionCfg()
    commands:     CommandsCfg      = CommandsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardCfg        = RewardCfg()
    terminations: TerminationsCfg  = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2           # control @ 50 Hz (sim 100 Hz / 2)
        self.episode_length_s = 50.0

        self.sim.dt = 1 / 100.0
        self.sim.render_interval = self.decimation

        self.viewer.eye    = (0.5, 0.5, 0.05)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solver_velocity_iteration_count = 1