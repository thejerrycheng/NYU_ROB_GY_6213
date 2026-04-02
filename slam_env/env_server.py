"""
env_server.py  —  SLAM Environment Server
==========================================
The canonical simulation process. Run this with mjpython.
Everything else (visualization, controllers, SLAM) subscribes to it.

Architecture
------------

  ┌─────────────────────────────────────────────────────────┐
  │                    env_server.py                        │
  │                   (mjpython)                            │
  │                                                         │
  │  MuJoCo physics  ──►  robot state                      │
  │  VehicleDynamics ──►  true pose, velocity              │
  │  LidarSensor     ──►  scan (noisy + noiseless)         │
  │                                                         │
  │  Publishes on named topics (FIFO pipes):                │
  │    "state"  → {x, y, theta, v, step, ...}              │
  │    "lidar"  → {angles, dists, flags, ref_dists, ...}   │
  │    "map"    → {walls, bounds, robot_start}  (once)     │
  │                                                         │
  │  Listens for control commands:                          │
  │    "cmd"    ← {v_cmd, alpha_cmd}                       │
  │              from keyboard, controller, or anything     │
  └─────────────────────────────────────────────────────────┘
         │                           ▲
         ▼ publish                   │ subscribe / send cmd
  ┌──────────────┐    ┌──────────────────────────────────┐
  │ lidar_viz.py │    │  keyboard_client.py (optional)   │
  │ slam_viz.py  │    │  your_controller.py              │
  │ ekf_slam.py  │    │  (any process, regular python)   │
  └──────────────┘    └──────────────────────────────────┘

HOW TO RUN
----------
  macOS:   mjpython env_server.py --map simple_room --noise
  Linux:   python   env_server.py --map simple_room --noise

  Then in separate terminals (regular python, no mjpython needed):
    python visualization/lidar_viz.py          ← subscribe to lidar
    python keyboard_client.py                  ← send keyboard commands
    python slam_teleop.py                      ← subscribe + send cmd

  You can run any combination. Each subscriber is independent.
  The env runs at its own rate regardless of how many clients connect.

Controls via keyboard_client.py OR the MuJoCo window (arrow keys):
  ↑/↓   v_cmd ±5    ←/→   α_cmd ±5
  X  STOP    R  RESET    Q/Esc  quit
"""

import sys, os, argparse, math, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco, mujoco.viewer

from slam_env.maps.map_registry  import load_map, list_maps
from slam_env.maps.map_loader    import (MapLoader, COL_HALF_X, COL_HALF_Y,
                                          ROBOT_JOINT_Z, CHASSIS_CX)
from slam_env.utils.motion_model import (VehicleDynamics, RobotState,
                                          DELTA_COEFFS, MAX_STEER, V_M, V_C)
from slam_env.utils.lidar_sensor  import LidarSensor, load_lidar_config
from slam_env.env_ipc             import EnvPublisher, EnvSubscriber

# ── GLFW key codes ────────────────────────────────────────────────────────────
KEY_UP=265; KEY_DOWN=264; KEY_LEFT=263; KEY_RIGHT=262
KEY_X=88;   KEY_R=82;     KEY_Q=81;    KEY_ESCAPE=256
V_STEP=5.0; ALPHA_STEP=5.0
V_MAX=100.; V_MIN=-100.; ALPHA_MAX=100.; ALPHA_MIN=-100.


def parse_args():
    p = argparse.ArgumentParser(description="SLAM Env Server (mjpython)")
    p.add_argument("--map",       default="simple_room", choices=list_maps())
    p.add_argument("--dt",        type=float, default=0.05,
                   help="Control timestep [s]")
    p.add_argument("--noise",     action="store_true",
                   help="Enable lidar noise model")
    p.add_argument("--lidar-cfg", default=None,
                   help="Path to custom lidar_config.yaml")
    return p.parse_args()


# ── Command state (written by key_callback + external cmd subscriber) ─────────
class CmdState:
    """Thread-safe command state. Written by MuJoCo key_callback OR cmd subscriber."""
    def __init__(self):
        self._lock      = threading.Lock()
        self.v_cmd      = 0.0
        self.alpha_cmd  = 0.0
        self.do_stop    = False
        self.do_reset   = False
        self.do_quit    = False

    def apply(self, v_cmd=None, alpha_cmd=None, stop=False, reset=False, quit_=False):
        with self._lock:
            if v_cmd     is not None: self.v_cmd     = float(np.clip(v_cmd,     V_MIN, V_MAX))
            if alpha_cmd is not None: self.alpha_cmd = float(np.clip(alpha_cmd, ALPHA_MIN, ALPHA_MAX))
            if stop:  self.v_cmd=0.; self.alpha_cmd=0.; self.do_stop=True
            if reset: self.v_cmd=0.; self.alpha_cmd=0.; self.do_reset=True
            if quit_: self.do_quit=True

    def get(self):
        with self._lock:
            return (self.v_cmd, self.alpha_cmd,
                    self.do_stop, self.do_reset, self.do_quit)

    def clear_flags(self):
        with self._lock:
            self.do_stop=False; self.do_reset=False


def make_key_callback(cmd: CmdState):
    """MuJoCo viewer key callback — single int keycode."""
    def cb(keycode: int):
        with cmd._lock:
            vc = cmd.v_cmd; ac = cmd.alpha_cmd
        if   keycode==KEY_UP:    cmd.apply(v_cmd    =vc+V_STEP)
        elif keycode==KEY_DOWN:  cmd.apply(v_cmd    =vc-V_STEP)
        elif keycode==KEY_LEFT:  cmd.apply(alpha_cmd=ac-ALPHA_STEP)
        elif keycode==KEY_RIGHT: cmd.apply(alpha_cmd=ac+ALPHA_STEP)
        elif keycode==KEY_X:     cmd.apply(stop=True);  print("\n  [X] STOP",  flush=True)
        elif keycode==KEY_R:     cmd.apply(reset=True); print("\n  [R] RESET", flush=True)
        elif keycode in (KEY_Q, KEY_ESCAPE): cmd.apply(quit_=True)
        with cmd._lock:
            vc2=cmd.v_cmd; ac2=cmd.alpha_cmd
        vp = max(0.,V_M*vc2+V_C) if vc2>0 else (min(0.,-(V_M*abs(vc2)+V_C)) if vc2<0 else 0.)
        print(f"\r  v_cmd={vc2:+6.1f}  α={ac2:+6.1f}  v_phys={vp:+.4f}m/s    ",
              end="", flush=True)
    return cb


def _cmd_listener(cmd: CmdState, sub: EnvSubscriber):
    """Background thread: reads external control commands from env_ipc."""
    while True:
        msg = sub.recv("cmd")
        if msg is None:
            break
        if msg.get("quit"):
            cmd.apply(quit_=True); break
        if msg.get("reset"):
            cmd.apply(reset=True); continue
        if msg.get("stop"):
            cmd.apply(stop=True); continue
        cmd.apply(v_cmd=msg.get("v_cmd"), alpha_cmd=msg.get("alpha_cmd"))


# ── Physics helpers ───────────────────────────────────────────────────────────
def _jaddrs(model):
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"robot_joint")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

def set_pose(model, data, s: RobotState):
    qa,da=_jaddrs(model)
    qw=math.cos(s.theta/2); qz=math.sin(s.theta/2)
    data.qpos[qa:qa+7]=[s.x,s.y,ROBOT_JOINT_Z,qw,0.,0.,qz]; data.qvel[da:da+6]=0.

def write_pose(model, data, x, y, th):
    qa,da=_jaddrs(model)
    qw=math.cos(th/2); qz=math.sin(th/2)
    data.qpos[qa:qa+7]=[x,y,ROBOT_JOINT_Z,qw,0.,0.,qz]; data.qvel[da:da+6]=0.

def set_steer(model, data, alpha):
    c=DELTA_COEFFS; d=float(np.clip(c[0]*alpha**2+c[1]*alpha+c[2],-MAX_STEER,MAX_STEER))
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"steer_joint")
    if jid>=0: data.qpos[model.jnt_qposadr[jid]]=-d

def clamp_obb(walls, x, y, theta):
    ct,st=math.cos(theta),math.sin(theta)
    wx=abs(COL_HALF_X*ct)+abs(COL_HALF_Y*st)
    wy=abs(COL_HALF_X*st)+abs(COL_HALF_Y*ct)
    cxw=x+CHASSIS_CX*ct; cyw=y+CHASSIS_CX*st
    WT=0.025
    for (x1,y1,x2,y2) in walls:
        dx,dy=x2-x1,y2-y1; sl=math.hypot(dx,dy)
        if sl<1e-9: continue
        nx_,ny_=-dy/sl,dx/sl
        dn=(cxw-x1)*nx_+(cyw-y1)*ny_
        rn=wx*abs(nx_)+wy*abs(ny_); sep=rn+WT
        if abs(dn)>=sep: continue
        tx_,ty_=dx/sl,dy/sl
        dt_=(cxw-x1)*tx_+(cyw-y1)*ty_
        rt=wx*abs(tx_)+wy*abs(ty_)
        if abs(dt_-sl/2)>sl/2+rt: continue
        ov=sep-abs(dn); sg=1. if dn>=0 else -1.
        cxw+=sg*nx_*ov; cyw+=sg*ny_*ov
    return cxw-CHASSIS_CX*ct, cyw-CHASSIS_CX*st


# ── Lidar dots in MuJoCo viewer ───────────────────────────────────────────────
def draw_lidar_mujoco(user_scn, rx, ry, rth, rel_angles, dists, max_range):
    """Large red dots at every lidar hit — raw sensor points, no lines."""
    HIT_R = 0.04                                          # 4 cm — visible from above
    Z     = 0.15                                          # above chassis
    RGBA  = np.array([1.0, 0.0, 0.0, 1.0], np.float64)  # solid red
    SIZE  = np.array([HIT_R, HIT_R, HIT_R], np.float64)
    EYE3  = np.eye(3).flatten().astype(np.float64)
    ra    = np.asarray(rel_angles, np.float64)
    d     = np.asarray(dists,      np.float64)
    gi    = 0
    for i in range(len(ra)):
        if gi >= user_scn.maxgeom: break
        if d[i] >= max_range * 0.999: continue
        a  = rth + ra[i]
        ep = np.array([rx + d[i]*math.cos(a), ry + d[i]*math.sin(a), Z], np.float64)
        mujoco.mjv_initGeom(user_scn.geoms[gi],
                            mujoco.mjtGeom.mjGEOM_SPHERE, SIZE, ep, EYE3, RGBA)
        gi += 1
    user_scn.ngeom = gi


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # macOS guard
    if sys.platform == "darwin":
        import mujoco.viewer as _v
        if not isinstance(getattr(_v,"_MJPYTHON",None),
                          getattr(_v,"_MjPythonBase",type(None))):
            mj_dir = os.path.dirname(mujoco.__file__)
            print(f"\n  ERROR: macOS requires mjpython.\n"
                  f"  Run: {mj_dir}/mjpython env_server.py --map {args.map}\n")
            sys.exit(1)

    map_dict  = load_map(args.map)
    lidar_cfg = load_lidar_config(args.lidar_cfg)
    dt        = args.dt
    max_range = float(lidar_cfg["geometry"]["max_range"])

    print("=" * 68)
    print(f"  SLAM Env Server  |  map: {map_dict['name']}  noise: {args.noise}")
    print(f"  dt={dt}s  rays={lidar_cfg['geometry']['num_rays']}")
    print()
    print("  Topics published:")
    print("    state  → true robot pose, velocity, step")
    print("    lidar  → scan (noisy + noiseless ref)")
    print("    map    → walls, bounds (sent once on connect)")
    print()
    print("  Topic consumed:")
    print("    cmd    ← v_cmd, alpha_cmd from any client")
    print()
    print("  Connect subscribers BEFORE starting (or they'll miss map msg):")
    print("    python visualization/lidar_viz.py")
    print("    python keyboard_client.py")
    print("=" * 68)
    print()

    # ── Build model ───────────────────────────────────────────────────────────
    loader = MapLoader(map_dict)
    model  = mujoco.MjModel.from_xml_string(loader.get_xml())
    data   = mujoco.MjData(model)

    lidar     = LidarSensor(walls=loader.walls, config=lidar_cfg, noise=args.noise)
    lidar_ref = LidarSensor(walls=loader.walls, config=lidar_cfg, noise=False)
    dynamics  = VehicleDynamics(noise=False)
    v_curr    = 0.0

    robot_state = RobotState(*map_dict["robot_start"])
    set_pose(model, data, robot_state)
    mujoco.mj_forward(model, data)

    # ── IPC setup ─────────────────────────────────────────────────────────────
    pub = EnvPublisher()          # publishes state, lidar, map
    sub = EnvSubscriber("cmd")    # receives control commands

    cmd = CmdState()

    # Background thread: drain incoming cmd messages
    t_cmd = threading.Thread(target=_cmd_listener, args=(cmd, sub),
                             daemon=True, name="cmd-listener")
    t_cmd.start()

    key_cb = make_key_callback(cmd)

    xmin,xmax,ymin,ymax = map_dict["bounds"]
    map_cx=(xmin+xmax)/2; map_cy=(ymin+ymax)/2
    span=max(xmax-xmin,ymax-ymin)

    # Publish map once (static info — subscribers cache it)
    map_msg = dict(
        name        = map_dict["name"],
        walls       = map_dict["walls"],
        bounds      = map_dict["bounds"],
        robot_start = map_dict["robot_start"],
        obstacles   = map_dict.get("obstacles", []),
        max_range   = max_range,
    )
    pub.publish("map", map_msg)

    step = 0
    print(f"  Map published. Starting sim loop.\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb,
        show_left_ui=True, show_right_ui=False,
    ) as viewer:

        viewer.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance  = span * 0.9
        viewer.cam.elevation = -55.0
        viewer.cam.azimuth   = 180.0
        viewer.cam.lookat[:] = [map_cx, map_cy, 0.0]

        while viewer.is_running():
            t0 = time.perf_counter()

            v_cmd, alpha_cmd, do_stop, do_reset, do_quit = cmd.get()

            if do_quit: break

            if do_reset:
                robot_state = RobotState(*map_dict["robot_start"])
                v_curr      = dynamics.reset_speed()
                set_pose(model, data, robot_state)
                mujoco.mj_forward(model, data)
                cmd.clear_flags()
                print("\n  [RESET]", flush=True)
                continue

            if do_stop:
                cmd.clear_flags()

            # ── Physics step ──────────────────────────────────────────────
            proposed, v_curr = dynamics.step(
                robot_state, v_curr, v_cmd, alpha_cmd, dt)
            nx, ny = clamp_obb(loader.walls,
                               proposed.x, proposed.y, proposed.theta)
            robot_state = RobotState(nx, ny, proposed.theta)

            # ── MuJoCo sync ───────────────────────────────────────────────
            write_pose(model, data, robot_state.x, robot_state.y, robot_state.theta)
            set_steer(model, data, alpha_cmd)
            mujoco.mj_forward(model, data)

            # ── Lidar ─────────────────────────────────────────────────────
            ra,   d,   f   = lidar.scan(
                robot_state.x, robot_state.y, robot_state.theta)
            ra_r, d_r      = lidar_ref.scan_noiseless(
                robot_state.x, robot_state.y, robot_state.theta)

            step += 1

            # ── Draw lidar in MuJoCo viewer (red dots only) ───────────────
            draw_lidar_mujoco(viewer.user_scn,
                              robot_state.x, robot_state.y, robot_state.theta,
                              ra, d, max_range)
            viewer.sync()

            # ── Publish state ─────────────────────────────────────────────
            pub.publish("state", dict(
                step      = step,
                x         = robot_state.x,
                y         = robot_state.y,
                theta     = robot_state.theta,
                v_phys    = v_curr,
                v_cmd     = v_cmd,
                alpha_cmd = alpha_cmd,
                dt        = dt,
            ))

            # ── Publish lidar ─────────────────────────────────────────────
            pub.publish("lidar", dict(
                step        = step,
                rel_angles  = np.asarray(ra,   np.float32),
                dists       = np.asarray(d,    np.float32),
                flags       = np.asarray(f,    np.uint8),
                ref_angles  = np.asarray(ra_r, np.float32),
                ref_dists   = np.asarray(d_r,  np.float32),
                max_range   = max_range,
                rx          = robot_state.x,
                ry          = robot_state.y,
                rth         = robot_state.theta,
            ))

            # ── Terminal log ──────────────────────────────────────────────
            d_arr=np.asarray(d); f_arr=np.asarray(f)
            vm=d_arr[(f_arr==0)&(d_arr<max_range*0.99)]
            print(f"\r  step={step:5d} | "
                  f"x={robot_state.x:.3f}  y={robot_state.y:.3f}  "
                  f"θ={math.degrees(robot_state.theta)%360:.1f}° | "
                  f"v={v_curr:+.4f}m/s | "
                  f"hits={(f_arr==0).sum():3d}  "
                  f"min={vm.min() if len(vm) else 0:.3f}m",
                  end="", flush=True)

            elapsed = time.perf_counter()-t0
            wait    = dt - elapsed
            if wait > 0: time.sleep(wait)

    pub.close()
    print(f"\n\n  Env server stopped — {step} steps")
    print(f"  Final: x={robot_state.x:.4f}  y={robot_state.y:.4f}  "
          f"θ={math.degrees(robot_state.theta)%360:.2f}°\n")


if __name__ == "__main__":
    main()
