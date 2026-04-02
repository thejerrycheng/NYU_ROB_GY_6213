"""
keyboard_teleop.py
==================
Drive the 4-wheel Ackermann SLAM robot with your keyboard.
Lidar scan is visualised live inside the MuJoCo viewer via user_scn.

HOW TO RUN
----------
  macOS  →  mjpython slam_env/keyboard_teleop.py  [--map maze] [--noise]
  Linux  →  python   slam_env/keyboard_teleop.py  [--map maze] [--noise]

  Find mjpython:
    python -c "import mujoco,os; print(os.path.dirname(mujoco.__file__))"

Key bindings  (click the MuJoCo viewer window first)
----------------------------------------------------
  ↑ / ↓    v_cmd  ±5 per tap   (+ forward, - reverse)
  ← / →    α_cmd  ±5 per tap   (- left steer, + right steer)
  X         STOP
  R         RESET to start pose
  Q / Esc   quit

  Mouse: orbit / pan / zoom freely at all times.
"""

import sys, os, argparse, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco, mujoco.viewer
import numpy as np

from slam_env.maps.map_loader    import (MapLoader, COL_HALF_X, COL_HALF_Y,
                                          ROBOT_JOINT_Z, CHASSIS_CX)
from slam_env.maps.simple_room   import SIMPLE_ROOM
from slam_env.maps.l_shaped      import L_SHAPED_ROOM
from slam_env.maps.maze          import MAZE_MAP
from slam_env.utils.motion_model import (MotionModel, VehicleDynamics,
                                          RobotState, V_M, V_C, L, DELTA_COEFFS)
from slam_env.utils.lidar_sensor  import LidarSensor, load_lidar_config

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
    p.add_argument("--map",    default="simple_room", choices=list(MAP_REGISTRY.keys()))
    p.add_argument("--dt",     type=float, default=0.05)
    p.add_argument("--noise",  action="store_true", help="Enable all lidar noise")
    p.add_argument("--no-lidar-vis", action="store_true",
                   help="Disable lidar visualisation (faster)")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--lidar-cfg", default=None,
                   help="Path to a custom lidar_config.yaml")
    return p.parse_args()


# ── Command state ─────────────────────────────────────────────────────────────
class Cmd:
    def __init__(self):
        self.v_cmd=0.0; self.alpha_cmd=0.0
        self.do_stop=False; self.do_reset=False; self.do_quit=False


def make_key_callback(cmd):
    def key_callback(keycode: int):
        if   keycode == KEY_UP:    cmd.v_cmd     = float(np.clip(cmd.v_cmd+V_STEP,     V_MIN,    V_MAX))
        elif keycode == KEY_DOWN:  cmd.v_cmd     = float(np.clip(cmd.v_cmd-V_STEP,     V_MIN,    V_MAX))
        elif keycode == KEY_LEFT:  cmd.alpha_cmd = float(np.clip(cmd.alpha_cmd-ALPHA_STEP, ALPHA_MIN, ALPHA_MAX))
        elif keycode == KEY_RIGHT: cmd.alpha_cmd = float(np.clip(cmd.alpha_cmd+ALPHA_STEP, ALPHA_MIN, ALPHA_MAX))
        elif keycode == KEY_X:
            cmd.v_cmd=0.; cmd.alpha_cmd=0.; cmd.do_stop=True
            print("\n  [X] STOP", flush=True)
        elif keycode == KEY_R:
            cmd.v_cmd=0.; cmd.alpha_cmd=0.; cmd.do_reset=True
            print("\n  [R] RESET", flush=True)
        elif keycode in (KEY_Q, KEY_ESCAPE):
            cmd.do_quit=True; print("\n  [Q] Quit", flush=True)
        _print_cmd(cmd)
    return key_callback


def _print_cmd(cmd):
    vc=cmd.v_cmd
    vp = max(0., V_M*vc+V_C) if vc>0 else (min(0.,-(V_M*abs(vc)+V_C)) if vc<0 else 0.)
    d  = math.degrees(DELTA_COEFFS[0]*cmd.alpha_cmd**2 +
                       DELTA_COEFFS[1]*cmd.alpha_cmd +
                       DELTA_COEFFS[2])
    print(f"\r  v_cmd={cmd.v_cmd:+6.1f}  α={cmd.alpha_cmd:+6.1f}"
          f"  →  v_phys={vp:+.4f}m/s  δ={d:+.1f}°          ",
          end="", flush=True)


# ── OBB wall collision ────────────────────────────────────────────────────────
def clamp_obb_to_walls(walls, x, y, theta, half_x, half_y):
    ct, st = math.cos(theta), math.sin(theta)
    wx = abs(half_x*ct) + abs(half_y*st)
    wy = abs(half_x*st) + abs(half_y*ct)
    cx_w = x + CHASSIS_CX*ct
    cy_w = y + CHASSIS_CX*st
    hit  = False
    WALL_T = 0.05 / 2

    for (x1, y1, x2, y2) in walls:
        dx, dy  = x2-x1, y2-y1
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9: continue
        nx_, ny_ = -dy/seg_len,  dx/seg_len
        dist_n   = (cx_w-x1)*nx_ + (cy_w-y1)*ny_
        r_n      = wx*abs(nx_) + wy*abs(ny_)
        sep      = r_n + WALL_T
        if abs(dist_n) >= sep: continue
        tx_, ty_ = dx/seg_len, dy/seg_len
        dist_t   = (cx_w-x1)*tx_ + (cy_w-y1)*ty_
        r_t      = wx*abs(tx_) + wy*abs(ty_)
        if abs(dist_t - seg_len/2) > seg_len/2 + r_t: continue
        overlap  = sep - abs(dist_n)
        sign     = 1. if dist_n >= 0 else -1.
        cx_w    += sign * nx_ * overlap
        cy_w    += sign * ny_ * overlap
        hit      = True

    return cx_w - CHASSIS_CX*ct, cy_w - CHASSIS_CX*st, hit


# ── MuJoCo pose helpers ───────────────────────────────────────────────────────
def _joint_addrs(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_joint")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def set_pose(model, data, state: RobotState):
    qadr, dadr = _joint_addrs(model)
    qw = math.cos(state.theta/2); qz = math.sin(state.theta/2)
    data.qpos[qadr:qadr+7] = [state.x, state.y, ROBOT_JOINT_Z, qw, 0., 0., qz]
    data.qvel[dadr:dadr+6] = 0.


def write_pose(model, data, x, y, theta):
    qadr, dadr = _joint_addrs(model)
    qw = math.cos(theta/2); qz = math.sin(theta/2)
    data.qpos[qadr:qadr+7] = [x, y, ROBOT_JOINT_Z, qw, 0., 0., qz]
    data.qvel[dadr:dadr+6] = 0.


def set_steer_visual(model, data, alpha_cmd):
    """
    Write steering angle to the MuJoCo steer_joint.
    Sign convention:
      alpha_cmd > 0  →  steer RIGHT  →  delta_phys > 0  →  robot turns right
      MuJoCo hinge +Z = CCW = wheels point LEFT, so we NEGATE delta_phys.
    """
    from slam_env.utils.motion_model import DELTA_COEFFS as DC, MAX_STEER
    import math as _m
    delta = DC[0]*alpha_cmd**2 + DC[1]*alpha_cmd + DC[2]
    delta = max(-MAX_STEER, min(MAX_STEER, delta))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_joint")
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = -delta   # negated: fixes visual reversal


# ── Lidar visualisation ───────────────────────────────────────────────────────
def draw_lidar(user_scn, robot_x, robot_y, robot_theta,
               rel_angles, distances, flags, cfg):
    """
    Write lidar beams and hit points into MuJoCo user_scn.
    Called every frame; resets ngeom from the lidar base index each time.
    """
    vis  = cfg["visualization"]
    if not vis["enabled"]:
        user_scn.ngeom = 0
        return

    beam_rgba    = np.array(vis["beam_rgba"],    dtype=np.float32)
    hit_rgba     = np.array(vis["hit_rgba"],     dtype=np.float32)
    miss_rgba    = np.array(vis["miss_rgba"],    dtype=np.float32)
    outlier_rgba = np.array(vis["outlier_rgba"], dtype=np.float32)
    beam_r       = float(vis["beam_radius"])
    hit_r        = float(vis["hit_radius"])
    z            = float(vis["beam_height"])
    show_beams   = bool(vis["show_beams"])
    show_hits    = bool(vis["show_hits"])
    show_misses  = bool(vis["show_misses"])
    max_vis      = int(vis["max_vis_beams"])
    max_range    = float(cfg["geometry"]["max_range"])

    angles  = np.asarray(rel_angles)
    dists   = np.asarray(distances)
    fl      = np.asarray(flags, dtype=np.uint8)
    N       = min(len(angles), max_vis)

    geom_idx = 0   # index into user_scn.geoms

    for i in range(N):
        if geom_idx + 2 >= user_scn.maxgeom:
            break

        a     = robot_theta + angles[i]
        d     = dists[i]
        flag  = fl[i]
        is_hit    = d < max_range * 0.999
        is_out    = flag != 0

        # End-point of beam
        ex = robot_x + d * math.cos(a)
        ey = robot_y + d * math.sin(a)

        # ── Beam line ─────────────────────────────────────────────────────
        if show_beams and (is_hit or show_misses):
            g = user_scn.geoms[geom_idx]
            rgba = outlier_rgba if is_out else (beam_rgba if is_hit else miss_rgba)
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, beam_r,
                np.array([robot_x, robot_y, z]),
                np.array([ex, ey, z]))
            g.rgba[:] = rgba
            geom_idx += 1

        # ── Hit-point sphere ──────────────────────────────────────────────
        if show_hits and is_hit:
            g = user_scn.geoms[geom_idx]
            rgba = outlier_rgba if is_out else hit_rgba
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([hit_r, hit_r, hit_r]),
                np.array([ex, ey, z]),
                np.eye(3).flatten().astype(np.float64),
                rgba.astype(np.float64))
            geom_idx += 1

    user_scn.ngeom = geom_idx


# ── Logger ────────────────────────────────────────────────────────────────────
def log_state(step, state, cmd, v_phys, dists, flags):
    d_arr    = np.array(dists)
    fl_arr   = np.array(flags)
    valid    = d_arr[fl_arr == 0]
    lidar_min= float(valid.min()) if len(valid) else float(d_arr.min())
    hits     = int((d_arr < 4.99).sum())
    outliers = int((fl_arr != 0).sum())
    theta    = math.degrees(state.theta) % 360
    delta    = math.degrees(DELTA_COEFFS[0]*cmd.alpha_cmd**2 +
                             DELTA_COEFFS[1]*cmd.alpha_cmd +
                             DELTA_COEFFS[2])
    print(f"\n  step={step:5d} | "
          f"x={state.x:6.3f}  y={state.y:6.3f}  θ={theta:6.1f}° | "
          f"v_cmd={cmd.v_cmd:+6.1f}  α={cmd.alpha_cmd:+5.1f}  δ={delta:+.1f}° | "
          f"v={v_phys:+.4f}m/s | "
          f"lidar_min={lidar_min:.3f}m  hits={hits}  outliers={outliers}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    map_dict = MAP_REGISTRY[args.map]
    dt       = args.dt

    # ── macOS guard ───────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        import mujoco.viewer as _v
        if not isinstance(getattr(_v,"_MJPYTHON",None),
                          getattr(_v,"_MjPythonBase",type(None))):
            mj_dir = os.path.dirname(mujoco.__file__)
            print(f"\n  ERROR: macOS requires mjpython.\n"
                  f"  Run: {mj_dir}/mjpython slam_env/keyboard_teleop.py"
                  f" --map {args.map}\n")
            sys.exit(1)

    # ── Load lidar config ─────────────────────────────────────────────────────
    lidar_cfg = load_lidar_config(args.lidar_cfg)
    if args.no_lidar_vis:
        lidar_cfg["visualization"]["enabled"] = False

    print("=" * 70)
    print(f"  SLAM Env — 4-Wheel Ackermann Teleop  |  Lidar visualisation ON")
    print(f"  Map    : {map_dict['name']}  —  {map_dict.get('description','')}")
    print(f"  dt     : {dt}s   noise: {args.noise}")
    print(f"  Lidar  : {lidar_cfg['geometry']['num_rays']} rays  "
          f"max={lidar_cfg['geometry']['max_range']}m  "
          f"noise={'ON' if args.noise else 'OFF'}")
    print("=" * 70)
    print()
    print("  Controls (click the MuJoCo viewer first):")
    print("    ↑/↓  v_cmd ±5      ←/→  α_cmd ±5")
    print("    X  STOP   R  RESET   Q/Esc  quit")
    print("    Mouse  orbit/pan/zoom")
    print()

    # ── Build MuJoCo model ────────────────────────────────────────────────────
    loader = MapLoader(map_dict)
    model  = mujoco.MjModel.from_xml_string(loader.get_xml())
    data   = mujoco.MjData(model)

    # ── Lidar sensor ──────────────────────────────────────────────────────────
    lidar = LidarSensor(walls=loader.walls, config=lidar_cfg, noise=args.noise)

    # ── Dynamics ──────────────────────────────────────────────────────────────
    dynamics  = VehicleDynamics(noise=args.noise)
    v_current = 0.0

    # ── Initial pose ──────────────────────────────────────────────────────────
    start       = RobotState(*loader.robot_start)
    robot_state = RobotState(*loader.robot_start)
    set_pose(model, data, robot_state)
    mujoco.mj_forward(model, data)

    cmd    = Cmd()
    key_cb = make_key_callback(cmd)

    xmin,xmax,ymin,ymax = map_dict["bounds"]
    map_cx   = (xmin+xmax)/2; map_cy = (ymin+ymax)/2
    map_span = max(xmax-xmin, ymax-ymin)

    print(f"  Start: x={start.x:.3f}m  y={start.y:.3f}m  θ={math.degrees(start.theta):.1f}°")
    print(f"  Lidar will draw into viewer.user_scn (MAX_GEOM=100000)\n")

    step         = 0
    last_angles  = []
    last_dists   = []
    last_flags   = np.array([], dtype=np.uint8)

    # ── Launch viewer with user_scn ───────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=key_cb,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:

        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance  = map_span * 0.9
        viewer.cam.elevation = -55.0
        viewer.cam.azimuth   = 180.0
        viewer.cam.lookat[:] = [map_cx, map_cy, 0.0]

        while viewer.is_running():
            t0 = time.perf_counter()

            # ── Flags ─────────────────────────────────────────────────────
            if cmd.do_quit: break

            if cmd.do_reset:
                robot_state = RobotState(*loader.robot_start)
                v_current   = dynamics.reset_speed()
                set_pose(model, data, robot_state)
                mujoco.mj_forward(model, data)
                cmd.do_reset=False; cmd.do_stop=False
                print(f"\n  Reset → {robot_state.as_tuple()}", flush=True)
                continue

            if cmd.do_stop:
                cmd.do_stop = False

            # ── Vehicle dynamics ──────────────────────────────────────────
            proposed, v_current = dynamics.step(
                robot_state, v_current, cmd.v_cmd, cmd.alpha_cmd, dt)

            # ── Wall collision ────────────────────────────────────────────
            nx, ny, _ = clamp_obb_to_walls(
                loader.walls, proposed.x, proposed.y, proposed.theta,
                COL_HALF_X, COL_HALF_Y)
            robot_state = RobotState(nx, ny, proposed.theta)

            # ── MuJoCo pose sync ──────────────────────────────────────────
            write_pose(model, data, robot_state.x, robot_state.y, robot_state.theta)
            set_steer_visual(model, data, cmd.alpha_cmd)
            mujoco.mj_forward(model, data)

            # ── Lidar scan ────────────────────────────────────────────────
            rel_angles, dists, flags = lidar.scan(
                robot_state.x, robot_state.y, robot_state.theta)
            last_angles = rel_angles
            last_dists  = dists
            last_flags  = flags

            # ── Draw lidar in user_scn ────────────────────────────────────
            draw_lidar(viewer.user_scn,
                       robot_state.x, robot_state.y, robot_state.theta,
                       rel_angles, dists, flags, lidar_cfg)

            viewer.sync()

            # ── Log ───────────────────────────────────────────────────────
            step += 1
            if step % args.log_interval == 0:
                log_state(step, robot_state, cmd, v_current, dists, flags)

            # ── Pace ──────────────────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            wait    = dt - elapsed
            if wait > 0:
                time.sleep(wait)

    print(f"\n  Session ended — {step} steps")
    print(f"  Final: x={robot_state.x:.4f}m  y={robot_state.y:.4f}m  "
          f"θ={math.degrees(robot_state.theta)%360:.2f}°\n")


if __name__ == "__main__":
    main()