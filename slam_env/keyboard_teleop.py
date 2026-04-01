"""
keyboard_teleop.py
==================
Drive the 4-wheel Ackermann SLAM robot with your keyboard in MuJoCo.

HOW TO RUN
----------
  macOS  →  mjpython slam_env/keyboard_teleop.py  [--map maze] [--noise]
  Linux  →  python   slam_env/keyboard_teleop.py  [--map maze] [--noise]

  Find mjpython:
    python -c "import mujoco,os; print(os.path.dirname(mujoco.__file__))"

Key bindings  (click inside the MuJoCo viewer window first)
-----------------------------------------------------------
  ↑ / ↓    v_cmd  ±5 per tap   (positive = forward, negative = reverse)
  ← / →    α_cmd  ±5 per tap   (negative = steer left, positive = steer right)
  X         STOP — zero both commands
  R         RESET to start pose
  Q / Esc   quit

Mouse: orbit / pan / zoom freely at all times.
"""

import sys, os, argparse, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco, mujoco.viewer
import numpy as np

from slam_env.maps.map_loader import MapLoader, COL_HALF_X, COL_HALF_Y, ROBOT_JOINT_Z, CHASSIS_CX
from slam_env.maps.simple_room   import SIMPLE_ROOM
from slam_env.maps.l_shaped      import L_SHAPED_ROOM
from slam_env.maps.maze          import MAZE_MAP
from slam_env.utils.motion_model import MotionModel, VehicleDynamics, RobotState, V_M, V_C, L, DELTA_COEFFS
from slam_env.utils.lidar_sensor  import LidarSensor

MAP_REGISTRY = {
    "simple_room": SIMPLE_ROOM,
    "l_shaped":    L_SHAPED_ROOM,
    "maze":        MAZE_MAP,
}

KEY_UP=265; KEY_DOWN=264; KEY_LEFT=263; KEY_RIGHT=262
KEY_X=88;   KEY_R=82;     KEY_Q=81;    KEY_ESCAPE=256

V_STEP=5.0; ALPHA_STEP=5.0
V_MAX=100.0; V_MIN=-100.0
ALPHA_MAX=100.0; ALPHA_MIN=-100.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="simple_room", choices=list(MAP_REGISTRY.keys()))
    p.add_argument("--dt",  type=float, default=0.05)
    p.add_argument("--noise", action="store_true")
    p.add_argument("--log-interval", type=int, default=20)
    return p.parse_args()


# ── Command state ─────────────────────────────────────────────────────────────
class Cmd:
    def __init__(self):
        self.v_cmd=0.0; self.alpha_cmd=0.0
        self.do_stop=False; self.do_reset=False; self.do_quit=False


def make_key_callback(cmd):
    def key_callback(keycode: int):
        if   keycode == KEY_UP:    cmd.v_cmd     = float(np.clip(cmd.v_cmd+V_STEP,       V_MIN,     V_MAX))
        elif keycode == KEY_DOWN:  cmd.v_cmd     = float(np.clip(cmd.v_cmd-V_STEP,       V_MIN,     V_MAX))
        elif keycode == KEY_LEFT:  cmd.alpha_cmd = float(np.clip(cmd.alpha_cmd-ALPHA_STEP, ALPHA_MIN, ALPHA_MAX))
        elif keycode == KEY_RIGHT: cmd.alpha_cmd = float(np.clip(cmd.alpha_cmd+ALPHA_STEP, ALPHA_MIN, ALPHA_MAX))
        elif keycode == KEY_X:
            cmd.v_cmd=0.0; cmd.alpha_cmd=0.0; cmd.do_stop=True
            print("\n  [X] STOP", flush=True)
        elif keycode == KEY_R:
            cmd.v_cmd=0.0; cmd.alpha_cmd=0.0; cmd.do_reset=True
            print("\n  [R] RESET", flush=True)
        elif keycode in (KEY_Q, KEY_ESCAPE):
            cmd.do_quit=True; print("\n  [Q] Quit", flush=True)
        _print_cmd(cmd)
    return key_callback


def _print_cmd(cmd):
    vc=cmd.v_cmd
    vp = max(0., V_M*vc+V_C) if vc>0 else (min(0.,-(V_M*abs(vc)+V_C)) if vc<0 else 0.)
    delta_deg = math.degrees(
        DELTA_COEFFS[0]*cmd.alpha_cmd**2 +
        DELTA_COEFFS[1]*cmd.alpha_cmd +
        DELTA_COEFFS[2])
    print(f"\r  v_cmd={cmd.v_cmd:+6.1f}  α={cmd.alpha_cmd:+6.1f}"
          f"  →  v_phys={vp:+.4f}m/s  δ={delta_deg:+.1f}°          ",
          end="", flush=True)


# ── OBB wall collision (axis-aligned box robot vs line-segment walls) ─────────
def clamp_obb_to_walls(walls, x, y, theta, half_x, half_y):
    """
    Separating Axis Test between a rotated rectangle (robot) and each wall
    segment (treated as a thin capsule with radius = WALL_THICKNESS/2).

    For simplicity and robustness we use the robot's AABB in world space
    (conservative but always correct — avoids ghost penetrations from the
    pure-circle approximation that didn't match the box visual).

    Returns (new_x, new_y, hit).
    """
    ct, st = math.cos(theta), math.sin(theta)
    # World-frame AABB of the rotated robot box
    # Half-extents after rotation
    wx = abs(half_x*ct) + abs(half_y*st)
    wy = abs(half_x*st) + abs(half_y*ct)

    # Robot AABB centre in world = (x, y) + chassis offset along heading
    # The freejoint origin is at the REAR AXLE; chassis centre is CHASSIS_CX ahead
    from slam_env.maps.map_loader import CHASSIS_CX
    cx_world = x + CHASSIS_CX * ct
    cy_world = y + CHASSIS_CX * st

    hit = False
    WALL_T = 0.05 / 2   # half wall thickness

    for (x1, y1, x2, y2) in walls:
        dx, dy = x2-x1, y2-y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        nx, ny = -dy/seg_len, dx/seg_len   # wall outward normal (left side)

        # Project robot AABB centre onto wall normal
        # Distance from wall line to robot centre
        dist_n = (cx_world-x1)*nx + (cy_world-y1)*ny
        # Project robot half-extents onto normal
        r_n = wx*abs(nx) + wy*abs(ny)
        # Total separation needed = r_n + wall_half_thickness
        sep = r_n + WALL_T

        if abs(dist_n) >= sep:
            continue   # no overlap in normal direction

        # Check overlap along wall tangent
        tx, ty = dx/seg_len, dy/seg_len
        dist_t = (cx_world-x1)*tx + (cy_world-y1)*ty
        r_t    = wx*abs(tx) + wy*abs(ty)
        seg_half = seg_len / 2
        closest_t = max(-seg_half, min(seg_half, dist_t - seg_half)) + seg_half
        if abs(dist_t - seg_half) > seg_half + r_t:
            continue   # no overlap along wall length

        # Penetration depth and push-out direction
        overlap = sep - abs(dist_n)
        sign    = 1.0 if dist_n >= 0 else -1.0
        push_x  = sign * nx * overlap
        push_y  = sign * ny * overlap

        # Apply push to rear-axle origin (reverse chassis offset)
        cx_world += push_x
        cy_world += push_y
        hit = True

    # Convert back to rear-axle world position
    new_x = cx_world - CHASSIS_CX * ct
    new_y = cy_world - CHASSIS_CX * st
    return new_x, new_y, hit


# ── MuJoCo helpers ────────────────────────────────────────────────────────────
def _joint_addrs(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_joint")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def set_pose(model, data, state: RobotState):
    """Teleport to exact pose (reset)."""
    qadr, dadr = _joint_addrs(model)
    qw = math.cos(state.theta/2); qz = math.sin(state.theta/2)
    data.qpos[qadr:qadr+7] = [state.x, state.y, ROBOT_JOINT_Z, qw, 0., 0., qz]
    data.qvel[dadr:dadr+6] = 0.


def write_pose(model, data, x, y, theta):
    """Write wall-clamped pose."""
    qadr, dadr = _joint_addrs(model)
    qw = math.cos(theta/2); qz = math.sin(theta/2)
    data.qpos[qadr:qadr+7] = [x, y, ROBOT_JOINT_Z, qw, 0., 0., qz]
    data.qvel[dadr:dadr+6] = 0.


def set_steer_visual(model, data, alpha_cmd):
    """
    Drive the steer_joint to the physical steering angle (in radians)
    so the front wheels rotate to match alpha_cmd visually.
    """
    delta_phys = (DELTA_COEFFS[0]*alpha_cmd**2
                + DELTA_COEFFS[1]*alpha_cmd
                + DELTA_COEFFS[2])  # radians
    # Find steer_joint qpos index
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_joint")
    if jid >= 0:
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = delta_phys


def log_state(step, state, cmd, v_phys, lidar_min, lidar_hits):
    theta_deg = math.degrees(state.theta) % 360
    delta_deg = math.degrees(
        DELTA_COEFFS[0]*cmd.alpha_cmd**2 +
        DELTA_COEFFS[1]*cmd.alpha_cmd +
        DELTA_COEFFS[2])
    print(f"\n  step={step:5d} | "
          f"x={state.x:6.3f}m  y={state.y:6.3f}m  θ={theta_deg:6.1f}° | "
          f"v_cmd={cmd.v_cmd:+6.1f}  α={cmd.alpha_cmd:+6.1f}  δ={delta_deg:+.1f}° | "
          f"v_phys={v_phys:+.4f}m/s | "
          f"lidar_min={lidar_min:.3f}m  hits={lidar_hits:3d}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    map_dict = MAP_REGISTRY[args.map]
    dt       = args.dt

    if sys.platform == "darwin":
        import mujoco.viewer as _v
        if not isinstance(getattr(_v,"_MJPYTHON",None), getattr(_v,"_MjPythonBase",type(None))):
            mj_dir = os.path.dirname(mujoco.__file__)
            print(f"\n  ERROR: macOS requires mjpython.\n"
                  f"  Run: {mj_dir}/mjpython slam_env/keyboard_teleop.py --map {args.map}\n")
            sys.exit(1)

    print("="*68)
    print(f"  SLAM Env — 4-Wheel Ackermann Teleop")
    print(f"  Map : {map_dict['name']}  —  {map_dict.get('description','')}")
    print(f"  dt  : {dt}s   noise: {args.noise}")
    print("="*68)
    print()
    print("  Controls (click the MuJoCo viewer window first):")
    print("    ↑ / ↓    v_cmd ±5 per tap   (+ forward  - reverse)")
    print("    ← / →    α_cmd ±5 per tap   (- left steer  + right steer)")
    print("    X        STOP    R  RESET    Q/Esc  quit")
    print("    Mouse    orbit / pan / zoom")
    print()

    loader = MapLoader(map_dict)
    model  = mujoco.MjModel.from_xml_string(loader.get_xml())
    data   = mujoco.MjData(model)

    dynamics   = VehicleDynamics(noise=args.noise)
    v_current  = 0.0          # actual physical speed [m/s], persists across steps
    lidar  = LidarSensor(walls=loader.walls, num_rays=360,
                         max_range=5.0, noise=args.noise)

    start       = RobotState(*loader.robot_start)
    robot_state = RobotState(*loader.robot_start)
    set_pose(model, data, robot_state)
    mujoco.mj_forward(model, data)

    cmd    = Cmd()
    key_cb = make_key_callback(cmd)

    xmin,xmax,ymin,ymax = map_dict["bounds"]
    map_cx = (xmin+xmax)/2; map_cy = (ymin+ymax)/2
    map_span = max(xmax-xmin, ymax-ymin)

    print(f"  Start: x={start.x:.3f}m  y={start.y:.3f}m  "
          f"θ={math.degrees(start.theta):.1f}°\n")

    step = 0

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb,
        show_left_ui=True, show_right_ui=False,
    ) as viewer:
        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance  = map_span * 0.9
        viewer.cam.elevation = -55.0
        viewer.cam.azimuth   = 180.0
        viewer.cam.lookat[:] = [map_cx, map_cy, 0.0]

        while viewer.is_running():
            t0 = time.perf_counter()

            if cmd.do_quit: break

            if cmd.do_reset:
                robot_state = RobotState(*loader.robot_start)
                v_current   = dynamics.reset_speed()
                set_pose(model, data, robot_state)
                mujoco.mj_forward(model, data)
                cmd.do_reset=False; cmd.do_stop=False
                print(f"  Reset → {robot_state.as_tuple()}", flush=True)
                continue

            if cmd.do_stop:
                cmd.do_stop=False

            # ── Vehicle dynamics step (inertia + rolling resistance) ────
            # v_current persists across steps — it is the actual speed state.
            # Even when v_cmd=0 the robot coasts until friction brings it to rest.
            proposed, v_current = dynamics.step(
                robot_state, v_current, cmd.v_cmd, cmd.alpha_cmd, dt)
            v_phys = v_current   # for logging

            # ── OBB wall collision ────────────────────────────────────────
            nx, ny, hit = clamp_obb_to_walls(
                loader.walls,
                proposed.x, proposed.y, proposed.theta,
                COL_HALF_X, COL_HALF_Y)
            robot_state = RobotState(nx, ny, proposed.theta)

            # ── Write pose + steer visual to MuJoCo ──────────────────────
            write_pose(model, data, robot_state.x, robot_state.y, robot_state.theta)
            set_steer_visual(model, data, cmd.alpha_cmd)
            mujoco.mj_forward(model, data)
            viewer.sync()

            # ── Lidar ─────────────────────────────────────────────────────
            _, dists   = lidar.scan(robot_state.x, robot_state.y, robot_state.theta)
            dists_arr  = np.array(dists)
            lidar_min  = float(dists_arr.min())
            lidar_hits = int((dists_arr < 4.99).sum())

            step += 1
            if step % args.log_interval == 0:
                log_state(step, robot_state, cmd, v_phys, lidar_min, lidar_hits)

            elapsed = time.perf_counter()-t0
            wait    = dt - elapsed
            if wait > 0: time.sleep(wait)

    print(f"\n  Session ended — {step} steps")
    print(f"  Final: x={robot_state.x:.4f}m  y={robot_state.y:.4f}m  "
          f"θ={math.degrees(robot_state.theta)%360:.2f}°\n")


if __name__ == "__main__":
    main()