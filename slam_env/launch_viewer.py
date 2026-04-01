"""
launch_viewer.py
================
Launches an interactive MuJoCo viewer for the SLAM environment.
You can freely orbit, zoom, and inspect the scene.

Controls (built-in MuJoCo viewer):
  Left-drag       — rotate camera
  Right-drag      — pan camera
  Scroll          — zoom
  Double-click    — select body / show info
  F               — toggle fullscreen
  H               — toggle help overlay
  Esc             — quit

Usage:
    cd NYU_ROB_GY_6213/
    python slam_env/launch_viewer.py
    python slam_env/launch_viewer.py --map maze
    python slam_env/launch_viewer.py --map l_shaped
"""

import sys
import os
import argparse
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from slam_env.maps.map_loader  import MapLoader
from slam_env.maps.simple_room import SIMPLE_ROOM
from slam_env.maps.l_shaped    import L_SHAPED_ROOM
from slam_env.maps.maze        import MAZE_MAP

MAP_REGISTRY = {
    "simple_room": SIMPLE_ROOM,
    "l_shaped":    L_SHAPED_ROOM,
    "maze":        MAZE_MAP,
}


def parse_args():
    p = argparse.ArgumentParser(description="Interactive MuJoCo SLAM viewer")
    p.add_argument("--map", default="simple_room",
                   choices=list(MAP_REGISTRY.keys()),
                   help="Which map to load (default: simple_room)")
    return p.parse_args()


def main():
    args = parse_args()
    map_dict = MAP_REGISTRY[args.map]

    print("=" * 60)
    print(f"  SLAM Env — Interactive MuJoCo Viewer")
    print(f"  Map : {map_dict['name']}")
    print(f"  Desc: {map_dict.get('description', '')}")
    print("=" * 60)
    print("\n  MuJoCo viewer controls:")
    print("    Left-drag   → rotate camera")
    print("    Right-drag  → pan camera")
    print("    Scroll      → zoom")
    print("    F           → fullscreen")
    print("    H           → help overlay")
    print("    Esc         → quit\n")

    # Build model
    loader = MapLoader(map_dict)
    xml    = loader.get_xml()
    model  = mujoco.MjModel.from_xml_string(xml)
    data   = mujoco.MjData(model)

    # Set robot to start pose
    x, y, theta = loader.robot_start
    import math
    qw = math.cos(theta / 2)
    qz = math.sin(theta / 2)
    robot_jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_joint")
    qpos_adr   = model.jnt_qposadr[robot_jid]
    data.qpos[qpos_adr:qpos_adr + 7] = [x, y, 0.03, qw, 0, 0, qz]
    mujoco.mj_forward(model, data)

    print(f"  Robot start: x={x:.3f}  y={y:.3f}  θ={math.degrees(theta):.1f}°")
    print("  Close the viewer window or press Esc to exit.\n")

    # Launch blocking interactive viewer
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
