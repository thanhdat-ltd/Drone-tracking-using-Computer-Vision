"""
Walking human pedestrian — IRA (Isaacsim.Replicator.Agent) pipeline
====================================================================
Key fix vs previous attempts:
  - Characters must be under /World/Characters/ (IRA hard-coded path)
  - Biped_Setup must be at /World/Characters/Biped_Setup (populate_anim_graph hardcodes this)
  - populate_anim_graph() from omni.anim.people must be called after loading Biped_Setup
    to wire up the AnimationGraph nodes that CharacterBehavior needs at runtime

Run:
    bash source/isaaclab_assets/isaaclab_assets/uav/open-loop/run_human.sh
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

enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.graph.schema")
enable_extension("omni.anim.graph.ui")
enable_extension("omni.anim.navigation.schema")
enable_extension("omni.anim.people")
enable_extension("omni.kit.scripting")

for _ in range(90):
    simulation_app.update()

# =============================================================================
# IMPORTS
# =============================================================================

import math
import os
import random
import tempfile
import time

import carb
import omni
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaacsim.core.utils import prims

from pxr import Sdf, Usd

from omni.anim.people.scripts.custom_command.populate_anim_graph import populate_anim_graph

# =============================================================================
# CONSTANTS
# =============================================================================

_BASE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/People"
)

CHARACTER_URL = f"{_BASE}/Characters/F_Medical_01/F_Medical_01.usd"
BIPED_URL     = f"{_BASE}/Characters/Biped_Setup.usd"

# IRA convention: all characters and biped live under /World/Characters/
CHAR_PARENT = "/World/Characters"
BIPED_PATH  = "/World/Characters/Biped_Setup"
CHAR_PATH   = "/World/Characters/Character"
CHAR_NAME   = "Character"  # name after CHAR_PARENT — used in command file

N_WAYPOINTS     = 6
WAYPOINT_RADIUS = 4.0
IDLE_DURATION   = 2.0
DT              = 1.0 / 60.0

# =============================================================================
# HELPERS
# =============================================================================

def make_waypoints(n, radius, seed=7):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(radius * 0.35, radius)
        pts.append((r * math.cos(a), r * math.sin(a), 0.0))
    return pts


def write_command_file(char_name, waypoints):
    lines = []
    for x, y, z in waypoints:
        lines.append(f"{char_name} GoTo {x:.3f} {y:.3f} {z:.3f} _")
        lines.append(f"{char_name} Idle {IDLE_DURATION:.1f}")
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="people_cmd_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    print(f"[CMD] {len(lines)} commands -> {path}")
    return path


def load_usd(stage, prim_path, url, label="", n_wait=300):
    """Load a USD asset as payload via isaacsim.core.utils.prims (same as IRA does)."""
    print(f"[LOAD:{label}] {url}")
    old = stage.GetPrimAtPath(prim_path)
    if old.IsValid():
        stage.RemovePrim(prim_path)
        for _ in range(5):
            simulation_app.update()
    prim = prims.create_prim(prim_path, "Xform", usd_path=url)
    chunk = n_wait // 3
    for dot in range(3):
        for _ in range(chunk):
            simulation_app.update()
        n = len(list(prim.GetChildren()))
        print(f"  [{dot+1}/3] children={n}")
        if n > 0:
            print(f"[LOAD:{label}] OK\n")
            return prim
    print(f"[LOAD:{label}] FAIL\n")
    return None


def find_skel_root(stage, prefix):
    for p in Usd.PrimRange.Stage(stage):
        if p.GetTypeName() == "SkelRoot" and str(p.GetPath()).startswith(prefix):
            return p
    return None


def find_anim_graph(stage, prefix):
    for p in Usd.PrimRange.Stage(stage):
        if p.GetTypeName() == "AnimationGraph" and str(p.GetPath()).startswith(prefix):
            return p
    return None


def get_behavior_script_path():
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    people_ext_path = ext_manager.get_extension_path_by_module("omni.anim.people")
    return os.path.join(
        people_ext_path, "omni", "anim", "people", "scripts", "character_behavior.py"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, render_interval=1))
    sim.set_camera_view(eye=[8.0, 8.0, 5.0], target=[0.0, 0.0, 1.0])
    stage = omni.usd.get_context().get_stage()

    # Scene
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/DomeLight",
        sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    # omni.anim.people carb settings
    settings = carb.settings.get_settings()
    settings.set("/persistent/exts/omni.anim.people/character_prim_path", CHAR_PARENT)
    settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
    settings.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
    settings.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")

    # Ensure /World/Characters parent prim exists
    if not stage.GetPrimAtPath(CHAR_PARENT).IsValid():
        prims.create_prim(CHAR_PARENT, "Xform")
        print(f"[INFO] Created {CHAR_PARENT}")

    # ------------------------------------------------------------------
    # STEP 1: Load Biped_Setup to /World/Characters/Biped_Setup
    # IRA loads it here so populate_anim_graph() can find it at the
    # hardcoded path /World/Characters/Biped_Setup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: LOAD BIPED_SETUP -> /World/Characters/Biped_Setup")
    print("=" * 60)

    biped_prim = load_usd(stage, BIPED_PATH, BIPED_URL, "BIPED", 300)
    if biped_prim is None:
        print("[FATAL] Biped_Setup load failed")
        simulation_app.close()
        return

    # Make it invisible (same as IRA does)
    biped_prim.GetAttribute("visibility").Set("invisible")

    # ------------------------------------------------------------------
    # STEP 2: populate_anim_graph()
    # This is the critical IRA function that wires up AnimationGraph nodes
    # (MotionMatching, BlendTree, etc.) so CharacterBehavior can drive them.
    # It hard-codes /World/Characters/Biped_Setup, which is why BIPED_PATH
    # must match exactly.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: POPULATE ANIM GRAPH (IRA)")
    print("=" * 60)

    try:
        populate_anim_graph()
        print("[OK] populate_anim_graph() completed")
    except Exception as e:
        print(f"[WARN] populate_anim_graph: {e}")

    for _ in range(60):
        simulation_app.update()

    # ------------------------------------------------------------------
    # STEP 3: Load character to /World/Characters/Character
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: LOAD CHARACTER -> /World/Characters/Character")
    print("=" * 60)

    char_prim = load_usd(stage, CHAR_PATH, CHARACTER_URL, "CHAR", 300)
    if char_prim is None:
        print("[FATAL] Character load failed")
        simulation_app.close()
        return

    for _ in range(60):
        simulation_app.update()

    # ------------------------------------------------------------------
    # STEP 4: Apply AnimationGraph to character SkelRoot
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: APPLY ANIMATION GRAPH")
    print("=" * 60)

    skel_root = find_skel_root(stage, CHAR_PATH)
    anim_graph = find_anim_graph(stage, BIPED_PATH)
    print(f"  SkelRoot  = {skel_root.GetPath() if skel_root else None}")
    print(f"  AnimGraph = {anim_graph.GetPath() if anim_graph else None}")

    skel_path = None
    if skel_root and anim_graph:
        skel_path = str(skel_root.GetPath())
        graph_path = str(anim_graph.GetPath())
        paths = [Sdf.Path(skel_path)]
        try:
            omni.kit.commands.execute("RemoveAnimationGraphAPICommand", paths=paths)
        except Exception:
            pass
        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=paths,
            animation_graph_path=Sdf.Path(graph_path),
        )
        print("[OK] ApplyAnimationGraphAPICommand done")

        for _ in range(120):
            simulation_app.update()

        try:
            import omni.anim.graph.core as ag
            char_obj = ag.get_character(skel_path)
            print(f"  ag.get_character = {char_obj}  (None = still not wired; non-None = success)")
        except Exception as e:
            print(f"  ag.get_character error: {e}")
    else:
        print("[WARN] SkelRoot or AnimGraph not found, skipping ApplyAnimationGraph")

    # ------------------------------------------------------------------
    # STEP 5: Attach CharacterBehavior script to SkelRoot
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: ATTACH CHARACTER BEHAVIOR SCRIPT")
    print("=" * 60)

    if skel_root:
        script_path = get_behavior_script_path()
        print(f"  script = {script_path}")
        paths = [Sdf.Path(str(skel_root.GetPath()))]
        try:
            omni.kit.commands.execute("RemoveScriptingAPICommand", paths=paths)
        except Exception:
            pass
        omni.kit.commands.execute("ApplyScriptingAPICommand", paths=paths)
        attr = skel_root.GetAttribute("omni:scripting:scripts")
        attr.Set([script_path])
        print("[OK] Script attached")
    else:
        print("[WARN] No SkelRoot — skipping behavior script")

    # ------------------------------------------------------------------
    # STEP 6: Write command file and point omni.anim.people to it
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: WRITE COMMAND FILE")
    print("=" * 60)

    waypoints  = make_waypoints(N_WAYPOINTS, WAYPOINT_RADIUS)
    cmd_file   = write_command_file(CHAR_NAME, waypoints)
    settings.set("/exts/omni.anim.people/command_settings/command_file_path", cmd_file)
    print(f"  character_prim_path = {settings.get('/persistent/exts/omni.anim.people/character_prim_path')}")
    print(f"  command_file_path   = {cmd_file}")

    for _ in range(30):
        simulation_app.update()

    # ------------------------------------------------------------------
    # STEP 7: Start simulation + timeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: START SIMULATION")
    print("=" * 60)

    sim.reset()

    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()
        print("[INFO] Timeline started")

    for _ in range(120):
        simulation_app.update()

    print("\n[INFO] Running. Ctrl+C to stop.")
    print(f"[INFO] Watch /World/Characters/Character for movement.\n")

    frame   = 0
    t_start = time.time()

    while simulation_app.is_running():
        sim.step()
        frame += 1

        if frame % (60 * 5) == 0:
            elapsed = time.time() - t_start
            # Check character position as a sanity check
            try:
                char = stage.GetPrimAtPath(CHAR_PATH)
                xf   = omni.usd.get_world_transform_matrix(char)
                pos  = xf.ExtractTranslation()
                print(f"[t={elapsed:5.0f}s f={frame}] char_pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
            except Exception:
                print(f"[t={elapsed:5.0f}s f={frame}] running")

    simulation_app.close()


if __name__ == "__main__":
    main()
