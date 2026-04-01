"""
visualize_sim.py
================
Runs the SLAMEnv with your original control_sequence and renders:
  - Left panel  : 2-D top-down lidar / trajectory view (matplotlib)
  - Right panel : MuJoCo off-screen RGB render (top-down camera)

Run:
    python visualize_sim.py
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # Changed from TkAgg to fix the macOS crash
import matplotlib.pyplot as plt    # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.gridspec as gridspec

from slam_env.envs.slam_env import SLAMEnv
from slam_env.maps.simple_room import SIMPLE_ROOM
from slam_env.maps.l_shaped    import L_SHAPED_ROOM
from slam_env.maps.maze        import MAZE_MAP


# ── User-editable parameters ──────────────────────────────────────────────────
CHOSEN_MAP = SIMPLE_ROOM      # ← swap to L_SHAPED_ROOM or MAZE_MAP
TOTAL_STEPS = 120
DT          = 0.1

# Same control_sequence as your original simulator
CONTROL_SEQUENCE = [
    (30,  40.0,  0.0),    # Phase 1: straight ahead
    (80,  40.0, 50.0),    # Phase 2: sharp right
    (100, 50.0, -10.0),   # Phase 3: slight left, faster
]
# ─────────────────────────────────────────────────────────────────────────────


def get_cmd(step):
    for end_step, v, alpha in CONTROL_SEQUENCE:
        if step < end_step:
            return v, alpha
    return 0.0, 0.0


def draw_lidar_panel(ax, env, step, history_x, history_y):
    ax.clear()

    # Map walls
    for wall in env._map_loader.walls:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)

    # Trajectory
    ax.plot(history_x, history_y, 'b--', linewidth=1.5, alpha=0.6,
            label="Path")

    # Lidar beams & hits
    angles, dists = env.get_lidar_scan()
    state = env._state
    ray_x, ray_y, hit_x, hit_y = [], [], [], []

    for a, d in zip(angles, dists):
        if d < env.max_range - 0.01:
            ga = state.theta + a
            hx = state.x + d * math.cos(ga)
            hy = state.y + d * math.sin(ga)
            ray_x += [state.x, hx, None]
            ray_y += [state.y, hy, None]
            hit_x.append(hx)
            hit_y.append(hy)

    ax.plot(ray_x, ray_y, color="lightblue", linewidth=0.4, zorder=1)
    ax.plot(hit_x, hit_y, 'r.', markersize=1.5, zorder=2)

    # Robot
    ax.plot(state.x, state.y, 'go', markersize=8, zorder=3, label="Robot")
    al = 0.15
    ax.arrow(state.x, state.y,
             al * math.cos(state.theta), al * math.sin(state.theta),
             head_width=0.05, head_length=0.05,
             fc='green', ec='green', zorder=4)

    xmin, xmax, ymin, ymax = env._map_loader.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"2-D Lidar View  |  Step {step}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)


def run():
    env = SLAMEnv(
        map_dict    = CHOSEN_MAP,
        dt          = DT,
        num_rays    = 360,
        noise       = True,
        render_mode = "rgb_array",
    )
    obs, _ = env.reset()

    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    ax_lidar  = fig.add_subplot(gs[0, 0])
    ax_mujoco = fig.add_subplot(gs[0, 1])

    history_x = [env._state.x]
    history_y = [env._state.y]

    for step in range(TOTAL_STEPS):
        v_cmd, alpha_cmd = get_cmd(step)
        action = np.array([v_cmd, alpha_cmd], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        history_x.append(env._state.x)
        history_y.append(env._state.y)

        # ── Left: 2-D lidar plot ──────────────────────────────────────────
        draw_lidar_panel(ax_lidar, env, step, history_x, history_y)

        # ── Right: MuJoCo top-down render ─────────────────────────────────
        rgb = env.render()
        ax_mujoco.clear()
        ax_mujoco.imshow(rgb, origin='upper')
        ax_mujoco.set_title(f"MuJoCo 3-D View  |  Map: {info['map_name']}")
        ax_mujoco.axis('off')

        fig.suptitle(f"Active SLAM Sim  —  step {step+1}/{TOTAL_STEPS}  "
                     f"|  v={v_cmd:.0f}  α={alpha_cmd:.0f}",
                     fontsize=12)
        plt.tight_layout()
        plt.pause(0.05)

        if terminated or truncated:
            break

    plt.ioff()
    plt.show()
    env.close()


if __name__ == "__main__":
    run()
