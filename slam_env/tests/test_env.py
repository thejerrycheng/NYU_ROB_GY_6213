"""
test_env.py  —  Headless smoke-test (no display needed)
Runs one episode with the default map and prints observations/info.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from slam_env.envs.slam_env import SLAMEnv
from slam_env.maps.simple_room import SIMPLE_ROOM
from slam_env.maps.l_shaped    import L_SHAPED_ROOM
from slam_env.maps.maze        import MAZE_MAP


def test_basic():
    print("=" * 60)
    print("  SLAMEnv  —  Basic smoke test")
    print("=" * 60)

    # ── Test 1: Simple room ───────────────────────────────────────────────
    env = SLAMEnv(map_dict=SIMPLE_ROOM, noise=True, render_mode=None)
    obs, info = env.reset()

    print(f"\n[MAP] {info['map_name']}")
    print(f"  obs shape  : {obs.shape}")
    print(f"  obs dtype  : {obs.dtype}")
    print(f"  lidar[0:5] : {obs[:5].round(3)}")
    print(f"  pose       : {info['pose']}")
    assert obs.shape == (363,), f"Expected (363,) got {obs.shape}"

    for step in range(30):
        v = 40.0 if step < 15 else 0.0
        a = 0.0  if step < 15 else 30.0
        obs, rew, term, trunc, info = env.step(np.array([v, a]))

    print(f"  After 30 steps pose: {info['pose']}")
    env.close()

    # ── Test 2: hot-swap map ──────────────────────────────────────────────
    env2 = SLAMEnv(map_dict=SIMPLE_ROOM, noise=False, render_mode=None)
    obs2, _ = env2.reset()
    print(f"\n[MAP SWAP] Loading L_SHAPED_ROOM…")
    env2.load_map(L_SHAPED_ROOM)
    obs3, info3 = env2.reset()
    print(f"  new map: {info3['map_name']}")
    env2.close()

    # ── Test 3: Lidar helper ──────────────────────────────────────────────
    env3 = SLAMEnv(map_dict=MAZE_MAP, noise=False, render_mode=None)
    env3.reset()
    angles, dists = env3.get_lidar_scan()
    hx, hy = env3.get_lidar_points_world()
    print(f"\n[LIDAR]  rays={len(dists)}  hits={len(hx)}")
    print(f"  min_dist={min(dists):.3f}m  max_dist={max(dists):.3f}m")
    env3.close()

    # ── Test 4: rgb_array render (skipped in headless/no-GPU environments) ──
    import os
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("MUJOCO_GL"))
    if has_display:
        env4 = SLAMEnv(map_dict=SIMPLE_ROOM, noise=False, render_mode="rgb_array")
        env4.reset()
        rgb = env4.render()
        print(f"\n[RENDER]  rgb shape={rgb.shape}  dtype={rgb.dtype}")
        assert rgb.ndim == 3 and rgb.shape[2] == 3, "Expected H×W×3 RGB image"
        env4.close()
    else:
        print("\n[RENDER]  ⚠  Skipped (no display / GPU). Set DISPLAY or"
              " MUJOCO_GL=egl to enable.")

    print("\n✓  All tests passed!\n")


if __name__ == "__main__":
    test_basic()
