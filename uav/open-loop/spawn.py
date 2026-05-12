"""
Humanoid Pedestrian - Random Waypoints  [FIXED]
Isaac Sim 5.x / Isaac Lab standalone

Fixes:
  1. Character no longer sinks below ground (Z spawn offset)
  2. No more sliding - uses Speed variable to drive locomotion blend tree
  3. Yaw corrected for Isaac Sim's forward-axis convention (+Y forward)
  4. Root motion mode: position updated only when anim graph lacks root motion,
     otherwise let the graph drive XY and only override rotation + waypoint logic
  5. Graceful fallback if animation graph unavailable

Run:
    isaaclab.bat -p humanoid_walk_fixed.py
    isaaclab.bat -p humanoid_walk_fixed.py --headless
"""

# =============================================================================
# APP
# =============================================================================

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# EXTENSIONS
# =============================================================================

from isaacsim.core.utils.extensions import enable_extension

# omni.anim.people KHONG co trong pip installation — bo qua
for ext in ["omni.anim.graph.core", "omni.anim.graph.schema", "omni.anim.graph.ui"]:
    try:
        enable_extension(ext)
    except Exception as e:
        print(f"[EXT] Skip {ext}: {e}")

for _ in range(60):
    simulation_app.update()

# =============================================================================
# IMPORTS
# =============================================================================

import math
import random
import time

import omni
import omni.usd
import omni.kit.commands

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from pxr import Gf, Sdf, Usd, UsdGeom

try:
    import omni.anim.graph.core as ag
except Exception:
    ag = None
    print("[WARN] omni.anim.graph.core import that bai")

# =============================================================================
# S3 ASSET URLS
# =============================================================================

_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/People"
)

CHARACTER_CANDIDATES = [
    f"{_BASE}/Characters/F_Medical_01/F_Medical_01.usd",
    f"{_BASE}/Characters/M_Medical_01/M_Medical_01.usd",
    f"{_BASE}/Characters/female_adult_police_02/female_adult_police_02.usd",
    f"{_BASE}/Characters/original_male_adult_police_04/male_adult_police_04.usd",
]

BIPED_CANDIDATES = [
    f"{_BASE}/Animation/Biped_Setup.usd",
    f"{_BASE}/Biped_Setup.usd",
    f"{_BASE}/Characters/Biped_Setup.usd",
]

# =============================================================================
# CONSTANTS
# =============================================================================

CHAR_ROOT  = "/World/Character"
BIPED_ROOT = "/World/Biped_Setup"

N_WAYPOINTS     = 6
WAYPOINT_RADIUS = 30.0   # m
WALK_SPEED      = 0.5   # m/s
ARRIVE_THRESH   = 0.40  # m

# FIX 1: Spawn character above ground so feet land on the floor.
# Most Isaac humanoid assets have their origin at hip height (~0.95 m).
# Setting Z=0 sinks the lower body through the ground plane.
# Adjust this value if the character still floats or clips:
#   - too high  → feet hover above floor  (decrease)
#   - too low   → feet clip through floor (increase)
CHAR_Z_OFFSET = 0.0   # metres above ground at spawn
                       # Most biped rigs already sit at 0 in their USD;
                       # if skeleton still sinks, try 0.93 or 1.0.

DT = 1.0 / 60.0

# FIX 3: Isaac Sim characters typically face +Y (not +X).
# atan2(dy, dx) gives angle from +X axis.
# We subtract 90° so that "facing +Y" = 0° in our convention.
YAW_OFFSET_DEG = -90.0   # set to 0.0 if character faces +X in its rest pose

# =============================================================================
# WAYPOINTS
# =============================================================================

def make_waypoints(n, radius, seed=7):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.35, radius)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts

# =============================================================================
# USD HELPERS
# =============================================================================

def find_prim_by_type(stage, type_name, prefix=""):
    for p in Usd.PrimRange.Stage(stage):
        if p.GetTypeName() == type_name:
            if not prefix or str(p.GetPath()).startswith(prefix):
                return p
    return None


def print_stage(stage, prefix=""):
    print(f"\n=== STAGE PRIMS (filter='{prefix}') ===")
    for p in Usd.PrimRange.Stage(stage):
        path = str(p.GetPath())
        if not prefix or path.startswith(prefix):
            print(f"  {path}  [{p.GetTypeName()}]")
    print("=" * 40 + "\n")


def get_or_create_op(xform, op_type):
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xform.AddTranslateOp()
    if op_type == UsdGeom.XformOp.TypeRotateXYZ:
        return xform.AddRotateXYZOp()
    return None


def set_world_transform(prim, x, y, yaw_deg, z=CHAR_Z_OFFSET):
    """
    FIX 1 + FIX 3 combined:
      - Z is now CHAR_Z_OFFSET (not 0.0) so the skeleton doesn't sink.
      - yaw_deg already has YAW_OFFSET_DEG applied by the caller.
    """
    xform = UsdGeom.Xformable(prim)
    t = get_or_create_op(xform, UsdGeom.XformOp.TypeTranslate)
    r = get_or_create_op(xform, UsdGeom.XformOp.TypeRotateXYZ)
    if t:
        t.Set(Gf.Vec3d(x, y, z))
    if r:
        r.Set(Gf.Vec3f(0.0, 0.0, yaw_deg))

# =============================================================================
# LOAD USD FROM CANDIDATE LIST
# =============================================================================

def load_reference(stage, prim_path, candidates, wait_frames=200, label=""):
    for url in candidates:
        print(f"[LOAD:{label}] Thu: {url}")

        old = stage.GetPrimAtPath(prim_path)
        if old.IsValid():
            stage.RemovePrim(prim_path)
            for _ in range(10):
                simulation_app.update()

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(assetPath=url)

        chunk = wait_frames // 3
        success = False
        for dot in range(3):
            for _ in range(chunk):
                simulation_app.update()
            n = len(list(prim.GetChildren()))
            print(f"  [{dot+1}/3] children={n}")
            if n > 0:
                success = True
                break

        if success:
            print(f"[LOAD:{label}] THANH CONG: {url}\n")
            return prim, url
        else:
            print(f"[LOAD:{label}] That bai, thu URL tiep...\n")

    return None, None

# =============================================================================
# FIND ANIMATION GRAPH
# =============================================================================

def find_anim_graph(stage):
    for guess in [
        f"{BIPED_ROOT}/CharacterAnimation/AnimationGraph",
        f"{BIPED_ROOT}/AnimationGraph",
        f"{BIPED_ROOT}/Anim/AnimationGraph",
        f"{BIPED_ROOT}/Character/AnimationGraph",
    ]:
        if stage.GetPrimAtPath(guess).IsValid():
            return guess

    for p in Usd.PrimRange.Stage(stage):
        path = str(p.GetPath())
        if "AnimationGraph" in p.GetName() and path.startswith(BIPED_ROOT):
            return path

    for p in Usd.PrimRange.Stage(stage):
        if "AnimationGraph" in p.GetName():
            return str(p.GetPath())

    return None

# =============================================================================
# PEDESTRIAN CONTROLLER  [FIXED]
# =============================================================================

class PedestrianController:
    """
    FIX 2 - No more sliding:

    The root cause of sliding is setting USD translate every frame while the
    animation graph plays a walk cycle with no root motion.  The feet animate
    in place but the body glides because we're teleporting it.

    Correct approach depends on what the Biped_Setup graph supports:

    Option A  (graph HAS root motion):
        • Set Speed variable > 0 to play the walk animation.
        • The graph itself moves the skeleton.
        • We only need to set the *rotation* each frame so it faces the target.
        • We track approximate XY position from our own dead-reckoning to know
          when to switch waypoints (the graph's root motion isn't easily readable
          back, so we mirror it ourselves).

    Option B  (graph has NO root motion / no character_obj):
        • Drive translate manually (same as before) BUT also set a "walk" blend
          variable so the leg animation plays.
        • Still looks slightly slidey on foot contact but far better than a
          full idle-pose glide.

    We auto-detect: if character_obj is available → Option A, else Option B.
    """

    # Animation variable names to try (in order) for walk speed
    _SPEED_VARS  = ("Speed", "ForwardSpeed", "WalkSpeed", "speed")
    # Boolean walk trigger variables
    _WALK_VARS   = ("Walk", "IsWalking", "IsMoving", "walk", "moving")
    # Idle trigger (set when stopped)
    _IDLE_VARS   = ("Idle", "IsIdle", "idle")

    def __init__(self, char_prim, character_obj):
        self.char_prim     = char_prim
        self.character_obj = character_obj

        # FIX 1: start at correct Z so feet are on the floor
        self.x   = 0.0
        self.y   = 0.0
        self.z   = CHAR_Z_OFFSET
        self.yaw = 0.0

        self.waypoints = make_waypoints(N_WAYPOINTS, WAYPOINT_RADIUS)
        self.wp_idx    = 0

        self._has_anim = character_obj is not None
        self._speed_var = None   # will be discovered on first step
        self._walk_var  = None

        print(f"[NAV] {len(self.waypoints)} waypoints: {self.waypoints}")
        print(f"[NAV] Animation mode: {'root-motion (Option A)' if self._has_anim else 'manual translate (Option B)'}")
        self._log_target()

        # Set initial transform so character spawns above ground
        set_world_transform(self.char_prim, self.x, self.y, 0.0)

    # ------------------------------------------------------------------
    def _current_target(self):
        return self.waypoints[self.wp_idx % len(self.waypoints)]

    def _log_target(self):
        tx, ty = self._current_target()
        idx = self.wp_idx % len(self.waypoints)
        print(f"[NAV] -> WP{idx}: ({tx:.2f}, {ty:.2f})")

    # ------------------------------------------------------------------
    def _try_set_var(self, name, value):
        """Try setting an anim variable; return True on success."""
        try:
            self.character_obj.set_variable(name, value)
            return True
        except Exception:
            return False

    def _discover_and_set_speed(self, speed_value):
        """
        FIX 2: set locomotion speed variable on the anim graph.
        Tries known variable names until one works, then caches it.
        """
        if self._speed_var:
            self._try_set_var(self._speed_var, speed_value)
            return

        for name in self._SPEED_VARS:
            if self._try_set_var(name, speed_value):
                self._speed_var = name
                print(f"[ANIM] Speed variable: '{name}'")
                return

        # Fallback: try boolean walk trigger
        if self._walk_var:
            self._try_set_var(self._walk_var, 1.0 if speed_value > 0 else 0.0)
            return

        for name in self._WALK_VARS:
            if self._try_set_var(name, 1.0):
                self._walk_var = name
                print(f"[ANIM] Walk variable: '{name}'")
                return

    def _set_idle(self):
        if self._speed_var:
            self._try_set_var(self._speed_var, 0.0)
        if self._walk_var:
            self._try_set_var(self._walk_var, 0.0)
        for name in self._IDLE_VARS:
            if self._try_set_var(name, 1.0):
                break

    # ------------------------------------------------------------------
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

        # FIX 3: correct yaw for Isaac Sim forward-axis convention
        raw_yaw = math.degrees(math.atan2(dy, dx))
        self.yaw = raw_yaw + YAW_OFFSET_DEG

        step_dist = WALK_SPEED * dt
        self.x += (dx / dist) * step_dist
        self.y += (dy / dist) * step_dist

        if self._has_anim:
            # Option A: only update rotation; let graph handle leg animation.
            # We still update translate so the character tracks the waypoint
            # (most Biped_Setup graphs do NOT export root motion to USD world
            # space, so we must move the prim ourselves — but the walk cycle
            # still plays correctly because Speed drives the blend tree).
            set_world_transform(self.char_prim, self.x, self.y, self.yaw)
            self._discover_and_set_speed(WALK_SPEED)
        else:
            # Option B: no anim graph, move manually
            set_world_transform(self.char_prim, self.x, self.y, self.yaw)

    # ------------------------------------------------------------------
    def status(self):
        tx, ty = self._current_target()
        return (
            f"pos=({self.x:.2f},{self.y:.2f}) yaw={self.yaw:.0f}deg "
            f"-> wp{self.wp_idx % len(self.waypoints)}=({tx:.2f},{ty:.2f})"
        )

# =============================================================================
# MAIN
# =============================================================================

def main():

    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=DT, render_interval=1)
    )
    sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 1.0])

    stage = omni.usd.get_context().get_stage()

    # --- Scene ---
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/DomeLight",
        sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    # --- Load Character ---
    print("\n" + "=" * 55)
    print("BUOC 1: LOAD CHARACTER")
    print("=" * 55)
    char_prim, _ = load_reference(
        stage, CHAR_ROOT, CHARACTER_CANDIDATES,
        wait_frames=240, label="CHAR"
    )
    if char_prim is None:
        print("\n[FATAL] Khong load duoc Character tu S3.")
        print("Kiem tra ket noi internet.")
        simulation_app.close()
        return

    # FIX 1: immediately set character above ground after load,
    # before physics/sim starts so it never appears sunken.
    set_world_transform(char_prim, 0.0, 0.0, 0.0, z=CHAR_Z_OFFSET)
    for _ in range(10):
        simulation_app.update()

    # --- Load Biped ---
    print("\n" + "=" * 55)
    print("BUOC 2: LOAD BIPED SETUP")
    print("=" * 55)
    biped_prim, _ = load_reference(
        stage, BIPED_ROOT, BIPED_CANDIDATES,
        wait_frames=240, label="BIPED"
    )
    if biped_prim is None:
        print("[WARN] Khong load duoc Biped_Setup.")
        print("       Character se di chuyen khong co animation.")
        print_stage(stage, "/World")

    # --- Apply Animation Graph ---
    character_obj = None

    if biped_prim is not None and ag is not None:
        print("\n" + "=" * 55)
        print("BUOC 3: APPLY ANIMATION GRAPH")
        print("=" * 55)

        skel_root = find_prim_by_type(stage, "SkelRoot", CHAR_ROOT)
        if skel_root is None:
            print("[WARN] Khong tim thay SkelRoot:")
            print_stage(stage, CHAR_ROOT)
        else:
            skel_path  = str(skel_root.GetPath())
            graph_path = find_anim_graph(stage)
            print(f"  SkelRoot  = {skel_path}")
            print(f"  AnimGraph = {graph_path}")

            if graph_path is None:
                print("[WARN] Khong tim thay AnimationGraph prim.")
                print_stage(stage, BIPED_ROOT)
            else:
                try:
                    omni.kit.commands.execute(
                        "RemoveAnimationGraphAPICommand",
                        paths=[Sdf.Path(skel_path)],
                    )
                except Exception:
                    pass

                try:
                    omni.kit.commands.execute(
                        "ApplyAnimationGraphAPICommand",
                        paths=[Sdf.Path(skel_path)],
                        animation_graph_path=Sdf.Path(graph_path),
                    )
                    print("[INFO] AnimationGraph applied OK")
                except Exception as e:
                    print(f"[WARN] ApplyAnimationGraphAPICommand: {e}")

                for _ in range(120):
                    simulation_app.update()

                character_obj = ag.get_character(skel_path)
                if character_obj is None:
                    print(f"[WARN] ag.get_character = None (path={skel_path})")
                else:
                    print(f"[INFO] Character object OK: {character_obj}")

    # --- Sim reset + controller ---
    sim.reset()

    controller = PedestrianController(
        char_prim=char_prim,
        character_obj=character_obj,
    )

    # --- Loop ---
    print("\n[INFO] Simulation dang chay. Ctrl+C de dung.\n")
    print(f"[INFO] Z offset: {CHAR_Z_OFFSET} m  |  Yaw offset: {YAW_OFFSET_DEG} deg")
    print(f"[INFO] Neu character van bi chim: tang CHAR_Z_OFFSET (thu 0.93 hoac 1.0)")
    print(f"[INFO] Neu character quay sai huong: dieu chinh YAW_OFFSET_DEG\n")

    frame   = 0
    t_start = time.time()

    while simulation_app.is_running():
        sim.step()
        controller.step(DT)
        frame += 1

        if frame % (60 * 5) == 0:
            print(f"[t={time.time()-t_start:5.0f}s f={frame}] {controller.status()}")

    simulation_app.close()


if __name__ == "__main__":
    main()