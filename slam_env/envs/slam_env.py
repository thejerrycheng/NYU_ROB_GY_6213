"""
SLAMEnv
=======
A MuJoCo-backed OpenAI-gym–style environment for Active SLAM research.

Key design goals
----------------
1. **Swappable maps** — pass any map dict at construction time.
2. **Calibrated motion model** — your V_M / V_C / DELTA_COEFFS are used directly.
3. **2-D ray-cast lidar** — same algorithm as your original sim, vectorised with numpy.
4. **MuJoCo for 3-D visualisation** — the physics sim is used for rendering only;
   robot pose is driven by direct qpos writes (kinematic control), so the gym
   integrates with any SLAM or RL algorithm without needing MuJoCo actuators.
5. **Gym-compatible API** — reset() / step() / render() / close().

Observation space
-----------------
Box(num_rays + 3,)  →  [lidar_distances..., x, y, theta]
(You can subclass and override _build_obs() to return only lidar if preferred.)

Action space
------------
Box(2,) continuous  →  [v_cmd, alpha_cmd]
Bounds match your typical command range (±100, ±100).
"""

import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

from slam_env.maps.map_loader import MapLoader
from slam_env.utils.motion_model import MotionModel, RobotState
from slam_env.utils.lidar_sensor import LidarSensor
from slam_env.maps.simple_room import SIMPLE_ROOM

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DT        = 0.1      # seconds per step
NUM_LIDAR_RAYS    = 360
MAX_LIDAR_RANGE   = 5.0
MAX_STEPS         = 1000     # episode length before truncation


class SLAMEnv(gym.Env):
    """
    MuJoCo-backed SLAM gym environment.

    Parameters
    ----------
    map_dict    : map definition dict (default: SIMPLE_ROOM)
    dt          : simulation timestep [s]
    num_rays    : number of lidar beams
    max_range   : lidar max range [m]
    noise       : enable process + sensor noise
    render_mode : 'human' (interactive MuJoCo viewer) |
                  'rgb_array' (off-screen) |
                  None (no rendering)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self,
                 map_dict:    dict  = None,
                 dt:          float = DEFAULT_DT,
                 num_rays:    int   = NUM_LIDAR_RAYS,
                 max_range:   float = MAX_LIDAR_RANGE,
                 noise:       bool  = True,
                 render_mode: str   = None):

        super().__init__()

        self.map_dict    = map_dict or SIMPLE_ROOM
        self.dt          = dt
        self.num_rays    = num_rays
        self.max_range   = max_range
        self.noise       = noise
        self.render_mode = render_mode

        # Build map & MuJoCo model
        self._map_loader  = MapLoader(self.map_dict)
        self._init_mujoco()

        # Motion model & lidar
        self._motion_model = MotionModel(noise=self.noise)
        self._lidar        = LidarSensor(
            walls     = self._map_loader.walls,
            num_rays  = self.num_rays,
            max_range = self.max_range,
            noise     = self.noise,
        )

        # State
        self._state   : RobotState = RobotState(*self._map_loader.robot_start)
        self._step_cnt: int        = 0

        # ── Observation space ────────────────────────────────────────────────
        # [lidar_d_0 … lidar_d_N-1,  x,  y,  theta]
        obs_dim = self.num_rays + 3
        low  = np.concatenate([np.zeros(self.num_rays),
                               np.array([-np.inf, -np.inf, -math.pi])])
        high = np.concatenate([np.full(self.num_rays, max_range),
                               np.array([ np.inf,  np.inf,  math.pi])])
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

        # ── Action space ─────────────────────────────────────────────────────
        # [v_cmd, alpha_cmd]  — same scale as your original simulator
        self.action_space = spaces.Box(
            low  = np.array([-100.0, -100.0], dtype=np.float32),
            high = np.array([ 100.0,  100.0], dtype=np.float32),
        )

        # Viewer handle (lazy init)
        self._viewer = None
        self._renderer: mujoco.Renderer | None = None

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Optionally swap map at reset time via options dict
        if options and "map_dict" in options:
            self._load_map(options["map_dict"])

        start = self._map_loader.robot_start
        self._state    = RobotState(*start)
        self._step_cnt = 0

        self._sync_mujoco_pose()
        obs  = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(self, action):
        v_cmd, alpha_cmd = float(action[0]), float(action[1])

        # 1. Motion model prediction
        self._state = self._motion_model.step(
            self._state, v_cmd, alpha_cmd, self.dt)

        # 2. Sync pose into MuJoCo (kinematic drive)
        self._sync_mujoco_pose()
        mujoco.mj_forward(self._model, self._data)

        self._step_cnt += 1

        # 3. Observations
        obs  = self._build_obs()
        info = self._build_info()

        # 4. Termination / truncation
        terminated = False                         # no terminal goal by default
        truncated  = self._step_cnt >= MAX_STEPS

        # Reward: override in subclasses for your SLAM objective
        reward = 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Map swapping ─────────────────────────────────────────────────────────

    def load_map(self, map_dict: dict):
        """
        Hot-swap the map at any time (rebuilds the MuJoCo model).
        Call reset() afterwards to start a new episode.
        """
        self._load_map(map_dict)

    # ── Lidar access ─────────────────────────────────────────────────────────

    def get_lidar_scan(self):
        """Returns (angles, distances) for the current pose."""
        return self._lidar.scan(
            self._state.x, self._state.y, self._state.theta)

    def get_lidar_points_world(self):
        """Returns (hit_x, hit_y) arrays in world frame."""
        return self._lidar.scan_as_points(
            self._state.x, self._state.y, self._state.theta)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_mujoco(self):
        xml = self._map_loader.get_xml()
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data  = mujoco.MjData(self._model)
        self._model.opt.timestep = self.dt

    def _load_map(self, map_dict: dict):
        self.map_dict    = map_dict
        self._map_loader = MapLoader(map_dict)
        self._lidar      = LidarSensor(
            walls     = self._map_loader.walls,
            num_rays  = self.num_rays,
            max_range = self.max_range,
            noise     = self.noise,
        )
        self._init_mujoco()
        # Re-init viewer if open
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _sync_mujoco_pose(self):
        """Write robot state into MuJoCo qpos (freejoint = 7-DOF)."""
        robot_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "robot_joint")
        qpos_adr = self._model.jnt_qposadr[robot_jid]

        x, y, theta = self._state.x, self._state.y, self._state.theta
        z = 0.03   # chassis half-height above floor

        # Quaternion from Z-rotation
        qw = math.cos(theta / 2)
        qx = 0.0
        qy = 0.0
        qz = math.sin(theta / 2)

        self._data.qpos[qpos_adr:qpos_adr + 7] = [x, y, z, qw, qx, qy, qz]
        self._data.qvel[qpos_adr:qpos_adr + 6] = 0  # zero velocity

    def _build_obs(self) -> np.ndarray:
        _, dists = self._lidar.scan(
            self._state.x, self._state.y, self._state.theta)
        lidar_obs = np.array(dists, dtype=np.float32)
        pose_obs  = np.array([self._state.x, self._state.y, self._state.theta],
                             dtype=np.float32)
        return np.concatenate([lidar_obs, pose_obs])

    def _build_info(self) -> dict:
        return {
            "pose":      self._state.as_tuple(),
            "step":      self._step_cnt,
            "map_name":  self.map_dict.get("name", "unknown"),
        }

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_human(self):
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo is not installed.")
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            # Set top-down camera
            self._viewer.cam.type     = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, "top_down")
        self._viewer.sync()

    def _render_rgb_array(self) -> np.ndarray:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self._model, height=720, width=960)
            except Exception:
                import os
                os.environ.setdefault("MUJOCO_GL", "egl")
                try:
                    self._renderer = mujoco.Renderer(self._model, height=720, width=960)
                except Exception:
                    return np.zeros((720, 960, 3), dtype=np.uint8)
        self._renderer.update_scene(self._data, camera="top_down")
        return self._renderer.render()
