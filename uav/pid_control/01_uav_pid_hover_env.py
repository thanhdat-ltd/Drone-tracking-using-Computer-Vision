# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Hover PID — 2 tầng (Altitude + Attitude) trong môi trường kho hàng.

Môi trường: Simple_Warehouse USD từ NVIDIA Isaac Assets.
Người:      Nhân vật nhiều phần (đầu/thân/tay/chân) đi lại dọc theo kệ hàng.
Camera:     Gắn trước drone, nghiêng 15° + xoay 90° nhìn về phía kệ hàng (+Y),
            hiển thị RGB + Depth qua matplotlib.

Chạy:
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/uav/pid_control/04_uav_pid_hover_env.py
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hover PID 2 tầng — Warehouse + Người + Camera.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Import SAU khi Isaac Sim đã khởi động ────────────────────────────────────
import collections
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_assets.uav.uav_cfg import UAV_CFG
from pid_controller import PIDController, make_alloc_inv, apply_wrench, wrap_angle


CAM_UPDATE_HZ = 30

# ── Tham số PID ───────────────────────────────────────────────────────────────
class Cfg:
    ALT_STEPS = [(0.0, 0.5), (8.0, 1.0), (16.0, 1.5), (24.0, 1.0)]

    ALT_KP = 1.5;  ALT_KI = 0.7;  ALT_KD = 0.12;  ALT_ILIM = 1.0
    ROLL_KP  = 4.0;  ROLL_KI  = 0.1;  ROLL_KD  = 0.2;  ROLL_ILIM  = 0.5
    PITCH_KP = 4.0;  PITCH_KI = 0.1;  PITCH_KD = 0.2;  PITCH_ILIM = 0.5
    YAW_KP   = 3.0;  YAW_KI   = 0.0;  YAW_KD   = 0.1;  YAW_ILIM   = 0.3

    MAX_MOMENT  = 0.03
    MAX_YAW_MOM = 0.02
    WINDOW_S    = 20.0
    PLOT_EVERY  = 5
    CAM_UPDATE_HZ = 30

    # Camera degradation — giả lập camera drone thực
    CAM_BRIGHTNESS   = 0.62  # nhân độ sáng, < 1 làm tối
    CAM_BLUR_SIGMA   = 1.2   # Gaussian blur (pixel), tăng → mờ hơn
    CAM_DELAY_FRAMES = 0
    CAM_NOISE_STD    = 15.0  # độ lệch chuẩn nhiễu Gaussian (0-255), tăng → nhiều hạt

    # YOLO
    YOLO_WEIGHTS = "yolov8n-pose.pt"
    YOLO_CONF    = 0.20
    YOLO_KP_CONF = 0.25      # confidence tối thiểu để nhận keypoint


# ── Worker USD (bundled với omni.replicator.core) ─────────────────────────────
_worker_matches = glob.glob(
    "/home/hongquan/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.replicator.core*/omni/replicator/core/tests/data/objects/Worker/Worker.usd"
)
WORKER_USD_PATH = _worker_matches[0] if _worker_matches else None

# ── Tham số người đi lại ──────────────────────────────────────────────────────
PERSON_X    = -4.0  # m — khoảng cách trước camera (dọc hướng nhìn)
WALK_Y_HALF = 3.0   # m — đi lại từ -3m đến +3m theo Y (ngang frame)
WALK_PERIOD = 8.0   # giây cho 1 lượt khứ hồi


# ── Spawn worker (USD thật từ omni.replicator) ────────────────────────────────
def _apply_worker_material(stage) -> None:
    """Override toàn bộ Mesh trong /World/Person bằng PreviewSurface (thay texture bị thiếu)."""
    mat_path = "/World/Person/WorkerMat"
    material = UsdShade.Material.Define(stage, mat_path)
    shader   = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Màu xanh lá đậm kiểu áo công nhân
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.42, 0.22))
    shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.75)
    shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    bound = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Person")):
        if prim.GetTypeName() == "Mesh":
            UsdShade.MaterialBindingAPI(prim).Bind(material,
                UsdShade.Tokens.strongerThanDescendants)
            bound += 1
    print(f"[Worker] Material override applied to {bound} meshes.")


def spawn_person() -> None:
    """Load Worker USD 3D thật — YOLO có thể detect được."""
    if WORKER_USD_PATH is None:
        print("[WARN] Không tìm thấy Worker.usd!")
        return
    worker_cfg = sim_utils.UsdFileCfg(usd_path=WORKER_USD_PATH,
                                      scale=(0.01, 0.01, 0.01))  # cm → m
    worker_cfg.func("/World/Person", worker_cfg,
                    translation=(PERSON_X, 0.0, 0.0))
    print(f"[Worker] Loaded: {WORKER_USD_PATH}")
    stage = omni.usd.get_context().get_stage()
    _apply_worker_material(stage)


def update_person(stage, sim_time: float) -> None:
    """Di chuyển worker theo Y (ngang frame camera)."""
    y    = WALK_Y_HALF * math.sin(2.0 * math.pi * sim_time / WALK_PERIOD)
    prim = stage.GetPrimAtPath("/World/Person")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(PERSON_X, y, 0.0))


# ── Spawn môi trường ──────────────────────────────────────────────────────────
def spawn_environment() -> None:
    dome = sim_utils.DomeLightCfg(intensity=800.0, color=(0.85, 0.90, 1.0))
    dome.func("/World/DomeLight", dome)
    sun  = sim_utils.DistantLightCfg(intensity=4000.0, color=(1.0, 0.95, 0.85), angle=0.53)
    sun.func("/World/Sun", sun)

    # wh_cfg = sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
    # )
    # wh_cfg.func("/World/Warehouse", wh_cfg, translation=(0.0, 0.0, 0.0))

    gnd_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    )
    gnd_cfg.func("/World/GroundPlane", gnd_cfg)

    spawn_person()


# ── Matplotlib ────────────────────────────────────────────────────────────────
def build_figures():
    plt.ion()

    fig_pid, ax = plt.subplots(figsize=(9, 4))
    fig_pid.suptitle("Hover PID — Altitude step test", fontsize=12, fontweight="bold")
    ax.set_ylabel("Z [m]"); ax.set_xlabel("Time [s]"); ax.grid(True, alpha=0.3)
    lz,  = ax.plot([], [], "b-",  lw=2,   label="actual Z")
    ltgt,= ax.plot([], [], "r--", lw=1.5, label="target Z")
    ax.legend(loc="upper right")
    fig_pid.tight_layout()

    fig_cam, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig_cam.suptitle("FPV Camera — nhìn về tường bên kia (-Y)", fontsize=12, fontweight="bold")
    axes[0].set_title("RGB")
    im_rgb = axes[0].imshow(np.zeros((480, 640, 3), dtype=np.uint8)); axes[0].axis("off")
    axes[1].set_title("Depth [m]")
    im_dep = axes[1].imshow(np.zeros((480, 640), dtype=np.float32), cmap="jet", vmin=0, vmax=10)
    plt.colorbar(im_dep, ax=axes[1], fraction=0.046, label="m"); axes[1].axis("off")
    fig_cam.tight_layout()

    return fig_pid, ax, lz, ltgt, fig_cam, im_rgb, im_dep


def update_pid_plot(fig, ax, lz, ltgt, times, zs, tgts):
    if not times:
        return
    t, z, g = list(times), list(zs), list(tgts)
    lz.set_data(t, z); ltgt.set_data(t, g)
    ax.set_xlim(max(0.0, t[-1] - Cfg.WINDOW_S), t[-1] + 0.5)
    all_v = z + g
    ax.set_ylim(min(all_v) - 0.15, max(all_v) + 0.15)
    fig.canvas.draw(); fig.canvas.flush_events()


# ── YOLO Pose helpers ─────────────────────────────────────────────────────────
# 5 điểm trích từ COCO-17: nose(0), L-sho(5), R-sho(6), L-hip(11), R-hip(12)
_COCO_IDX   = [0, 5, 6, 11, 12]
_KPT_COLORS = [
    (0, 255, 255),   # nose      — cyan
    (0, 200, 255),   # L-sho     — xanh nhạt
    (0, 100, 255),   # R-sho     — xanh đậm
    (0, 255, 100),   # L-hip     — xanh lá
    (255, 100,  0),  # R-hip     — cam
]
# Kết nối trong list 5-kpt đã trích: 0=nose,1=L-sho,2=R-sho,3=L-hip,4=R-hip
_SKELETON = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]


def _extract_kpts(raw: np.ndarray) -> list:
    """Trích 5 keypoints từ COCO-17 (hoặc ≤5 kpt custom)."""
    n = len(raw)
    if n >= 17:
        return [tuple(raw[i]) for i in _COCO_IDX]
    return [tuple(raw[i]) for i in range(min(5, n))]


def _is_real_person(kpts5: list) -> bool:
    """Lọc: phải có ít nhất 1 shoulder VÀ 1 hip đủ confidence."""
    c = Cfg.YOLO_KP_CONF
    has_sho = any(kpts5[i][2] >= c for i in [1, 2])  # L-sho, R-sho
    has_hip = any(kpts5[i][2] >= c for i in [3, 4])  # L-hip, R-hip
    return has_sho and has_hip


def _draw_skeleton(bgr: np.ndarray, kpts5: list) -> None:
    """Vẽ đường nối skeleton + chấm tròn màu có viền trắng (giống laptop_tracker)."""
    c = Cfg.YOLO_KP_CONF
    for a, b in _SKELETON:
        xa, ya, ca = kpts5[a]; xb, yb, cb = kpts5[b]
        if ca > c and cb > c:
            cv2.line(bgr, (int(xa), int(ya)), (int(xb), int(yb)),
                     (180, 180, 180), 2, cv2.LINE_AA)
    for i, (x, y, conf) in enumerate(kpts5):
        if conf > c:
            col = _KPT_COLORS[i % len(_KPT_COLORS)]
            cv2.circle(bgr, (int(x), int(y)), 6, col,         -1, cv2.LINE_AA)
            cv2.circle(bgr, (int(x), int(y)), 6, (255,255,255), 1, cv2.LINE_AA)


def _get_centroid(kpts5: list):
    """Trung tâm của các keypoints có đủ confidence."""
    valid = [(x, y) for x, y, c in kpts5 if c > Cfg.YOLO_KP_CONF]
    if not valid:
        return None
    return int(np.mean([p[0] for p in valid])), int(np.mean([p[1] for p in valid]))


def _annotate_yolo(frame_rgb: np.ndarray, model: YOLO) -> np.ndarray:
    """Chạy YOLO-pose trên frame đã degraded (RGB in, RGB out)."""
    bgr     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w    = bgr.shape[:2]
    cx_img, cy_img = w // 2, h // 2

    results  = model.track(bgr, persist=True, classes=[0],
                           conf=Cfg.YOLO_CONF, tracker="bytetrack.yaml",
                           device=0, verbose=False)
    valid    = []   # (bbox, tid, kpts5, centroid, conf)

    if results[0].boxes is not None and results[0].keypoints is not None:
        boxes    = results[0].boxes
        kpts_all = results[0].keypoints.data.cpu().numpy()
        ids      = (boxes.id.cpu().numpy() if boxes.id is not None
                    else np.arange(len(kpts_all)))

        for i in range(len(boxes)):
            bbox   = boxes.xyxy[i].tolist()
            conf   = boxes.conf[i].item()
            tid    = int(ids[i])
            kpts5  = _extract_kpts(kpts_all[i])

            if not _is_real_person(kpts5):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 140), 1)
                cv2.putText(bgr, "no torso", (x1, max(y1-4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 140), 1)
                continue

            centroid = _get_centroid(kpts5)
            valid.append((bbox, tid, kpts5, centroid, conf))

            _draw_skeleton(bgr, kpts5)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, f"ID:{tid} {conf:.2f}", (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 0), 1)

    # ── Target: bbox lớn nhất ──
    if valid:
        t_bbox, t_id, t_kpts5, t_cen, _ = max(
            valid, key=lambda p: (p[0][2]-p[0][0])*(p[0][3]-p[0][1]))
        x1, y1, x2, y2 = map(int, t_bbox)
        cv2.rectangle(bgr, (x1-3, y1-3), (x2+3, y2+3), (0,215,255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"TARGET ID:{t_id}", (x1, min(y2+16, h-30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,215,255), 2, cv2.LINE_AA)
        if t_cen:
            # centroid người (đỏ)
            cv2.drawMarker(bgr, t_cen, (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            # đường nối tâm frame → centroid người
            cv2.line(bgr, (cx_img, cy_img), t_cen, (0, 200, 255), 1, cv2.LINE_AA)

    # ── Crosshair tâm frame ──
    cv2.drawMarker(bgr, (cx_img, cy_img), (255, 255, 255),
                   cv2.MARKER_CROSS, 24, 1, cv2.LINE_AA)

    # ── Status bar ──
    n = len(valid)
    cv2.rectangle(bgr, (0, h-26), (w, h), (25, 25, 25), -1)
    cv2.putText(bgr, f"YOLO  Persons:{n}" + ("  No target" if n == 0 else ""),
                (6, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_BLUR_KERNEL: torch.Tensor | None = None

def _get_blur_kernel() -> torch.Tensor:
    global _BLUR_KERNEL
    if _BLUR_KERNEL is None:
        sigma = Cfg.CAM_BLUR_SIGMA
        k = max(3, 2 * int(3 * sigma) + 1) | 1  # đảm bảo số lẻ
        x = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-x**2 / (2 * sigma**2))
        g = g / g.sum()
        kern = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(3, 1, k, k).contiguous()
        _BLUR_KERNEL = kern.cuda() if torch.cuda.is_available() else kern
    return _BLUR_KERNEL


def _degrade_rgb(frame: np.ndarray) -> np.ndarray:
    """Làm tối + blur + nhiễu trên GPU (torch CUDA)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t   = torch.from_numpy(frame).to(dev, dtype=torch.float32)   # H,W,3
    t   = t * Cfg.CAM_BRIGHTNESS
    # Gaussian blur: conv2d cần 1,3,H,W
    t   = t.permute(2, 0, 1).unsqueeze(0)
    k   = _get_blur_kernel()
    pad = k.shape[-1] // 2
    t   = F.conv2d(t, k, padding=pad, groups=3)
    t   = t.squeeze(0).permute(1, 2, 0)                           # H,W,3
    # Noise trên GPU
    t   = t + torch.randn_like(t) * Cfg.CAM_NOISE_STD
    return t.clamp(0, 255).byte().cpu().numpy()


def update_cam_plot(fig_cam, im_rgb, im_dep, camera: Camera,
                    rgb_buf: collections.deque, yolo_model: YOLO):
    try:
        rgb = camera.data.output.get("rgb")
        if rgb is not None and rgb.shape[0] > 0:
            raw = rgb[0, ..., :3].cpu().numpy().astype(np.uint8)
            rgb_buf.append(_degrade_rgb(raw))
            # delay > 0: hiện frame cũ nhất; delay = 0: hiện ngay frame mới nhất
            display = rgb_buf[0] if Cfg.CAM_DELAY_FRAMES > 0 else rgb_buf[-1]
            # chạy YOLO trên frame đã bị degraded
            annotated = _annotate_yolo(display, yolo_model)
            im_rgb.set_data(annotated)
        dep = camera.data.output.get("distance_to_image_plane")
        if dep is not None and dep.shape[0] > 0:
            d = dep[0].cpu().numpy()
            im_dep.set_data(d)
            im_dep.set_clim(vmin=0, vmax=max(float(d.max()), 0.1))
        fig_cam.canvas.draw(); fig_cam.canvas.flush_events()
    except Exception as e:
        print(f"[Camera] {e}")


# ── Setup simulation ──────────────────────────────────────────────────────────
def setup_simulation():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    # Viewport: nhìn từ phía +X về hướng người ở -X
    sim.set_camera_view(eye=[4.0, 0.0, 3.5], target=[PERSON_X, 0.0, 1.0])

    stage = omni.usd.get_context().get_stage()
    spawn_environment()

    robot_cfg = UAV_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg = robot_cfg.replace(init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, 0.05)))
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)



    cam_cfg = CameraCfg(
        prim_path="/World/Crazyflie/body/front_cam",
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.5, -0.5, -0.5, 0.5),
            convention="ros",
        ),
    )
    camera = Camera(cam_cfg)

    sim.reset()
    return sim, robot, camera, stage


# ── Helper ────────────────────────────────────────────────────────────────────
def get_target_altitude(sim_time: float) -> float:
    z = Cfg.ALT_STEPS[0][1]
    for t_start, alt in Cfg.ALT_STEPS:
        if sim_time >= t_start:
            z = alt
    return z


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HOVER PID — Warehouse + Người đi lại + FPV Camera")
    print(f"  Người tại X={PERSON_X}m, đi dọc Y ± {WALK_Y_HALF}m, T={WALK_PERIOD}s")
    print("=" * 60)

    sim, robot, camera, stage = setup_simulation()
    prop_ids  = robot.find_bodies("m.*_prop")[0]
    A_inv     = make_alloc_inv(device=sim.device)
    dt        = sim.get_physics_dt()
    SIM_HZ    = int(round(1.0 / dt))
    CAM_EVERY = max(1, SIM_HZ // Cfg.CAM_UPDATE_HZ)

    alt_pid   = PIDController(Cfg.ALT_KP,   Cfg.ALT_KI,   Cfg.ALT_KD,   Cfg.ALT_ILIM)
    roll_pid  = PIDController(Cfg.ROLL_KP,  Cfg.ROLL_KI,  Cfg.ROLL_KD,  Cfg.ROLL_ILIM)
    pitch_pid = PIDController(Cfg.PITCH_KP, Cfg.PITCH_KI, Cfg.PITCH_KD, Cfg.PITCH_ILIM)
    yaw_pid   = PIDController(Cfg.YAW_KP,   Cfg.YAW_KI,   Cfg.YAW_KD,   Cfg.YAW_ILIM)

    yolo_model = YOLO(Cfg.YOLO_WEIGHTS)
    print(f"[YOLO] Loaded: {Cfg.YOLO_WEIGHTS}")

    fig_pid, ax_pid, lz, ltgt, fig_cam, im_rgb, im_dep = build_figures()
    rgb_buf = collections.deque(maxlen=max(1, Cfg.CAM_DELAY_FRAMES))

    maxlen = int(Cfg.WINDOW_S / dt) + 50
    times  = collections.deque(maxlen=maxlen)
    zs     = collections.deque(maxlen=maxlen)
    tgts   = collections.deque(maxlen=maxlen)

    sim_time = 0.0
    step     = 0

    while simulation_app.is_running():
        target_z = get_target_altitude(sim_time)

        pos  = robot.data.root_pos_w[0]
        quat = robot.data.root_quat_w[0]
        roll, pitch, yaw = [x[0].item() for x in euler_xyz_from_quat(quat.unsqueeze(0))]

        thrust  = max(0.0, alt_pid.update(target_z - pos[2].item(), dt))
        m_roll  = max(-Cfg.MAX_MOMENT,  min(Cfg.MAX_MOMENT,  roll_pid.update(-roll,           dt)))
        m_pitch = max(-Cfg.MAX_MOMENT,  min(Cfg.MAX_MOMENT,  pitch_pid.update(-pitch,         dt)))
        m_yaw   = max(-Cfg.MAX_YAW_MOM, min(Cfg.MAX_YAW_MOM, yaw_pid.update(wrap_angle(-yaw), dt)))

        apply_wrench(robot, sim, prop_ids, A_inv, thrust, m_roll, m_pitch, m_yaw)
        robot.write_data_to_sim()
        sim.step()
        sim_time += dt
        step     += 1
        robot.update(dt)
        camera.update(dt)

        update_person(stage, sim_time)

        if step % CAM_EVERY == 0:
            update_cam_plot(fig_cam, im_rgb, im_dep, camera, rgb_buf, yolo_model)

        times.append(sim_time); zs.append(pos[2].item()); tgts.append(target_z)
        if step % Cfg.PLOT_EVERY == 0:
            update_pid_plot(fig_pid, ax_pid, lz, ltgt, times, zs, tgts)

        if step % SIM_HZ == 0:
            py = WALK_Y_HALF * math.sin(2.0 * math.pi * sim_time / WALK_PERIOD)
            print(f"t={sim_time:5.1f}s | z={pos[2].item():+.3f}m (tgt={target_z:.1f}m) | "
                  f"người=({PERSON_X:.1f}, {py:+.2f})m")

        # Viewport theo drone, nhìn về phía người (-X)
        p = pos.cpu().numpy()
        sim.set_camera_view(
            eye=[p[0] + 3.0, p[1], p[2] + 1.5],
            target=[PERSON_X, p[1], 0.8],
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
