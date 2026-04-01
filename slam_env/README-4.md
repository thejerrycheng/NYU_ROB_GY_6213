# Active SLAM — MuJoCo Gym Environment

A modular, OpenAI-gym–compatible simulation environment for Active SLAM research,
built on **MuJoCo 3** and **Gymnasium**. The robot's motion model uses your
exact calibrated constants (V_M, V_C, DELTA_COEFFS …), and the 2-D lidar
is a vectorised numpy ray-caster with empirical Gaussian noise.

---

## Folder structure

```
slam_env/
│
├── __init__.py                  # Exports SLAMEnv
│
├── envs/
│   ├── __init__.py
│   └── slam_env.py              ← Main gym environment (reset / step / render)
│
├── maps/
│   ├── __init__.py
│   ├── map_loader.py            ← dict → MuJoCo XML builder
│   ├── simple_room.py           ← 4 m × 3 m rectangular room
│   ├── l_shaped.py              ← L-shaped room with a corner
│   └── maze.py                  ← Maze with corridors and obstacles
│
├── utils/
│   ├── __init__.py
│   ├── motion_model.py          ← Calibrated bicycle model (your constants)
│   └── lidar_sensor.py          ← Vectorised 2-D ray-cast lidar
│
├── tests/
│   └── test_env.py              ← Headless smoke-tests
│
├── visualize_sim.py             ← Side-by-side matplotlib + MuJoCo render
└── README.md
```

---

## Quick start

### Install

```bash
pip install mujoco gymnasium numpy matplotlib
```

### Run the visualiser

```bash
cd slam_env/
python visualize_sim.py
```

This opens a dual-panel window:
- **Left** — 2-D lidar beams + trajectory trail (matplotlib, identical to your original sim)
- **Right** — MuJoCo top-down 3-D render

### Run headless tests

```bash
PYTHONPATH=. python slam_env/tests/test_env.py
```

---

## Environment API

```python
from slam_env import SLAMEnv
from slam_env.maps.maze import MAZE_MAP

env = SLAMEnv(
    map_dict    = MAZE_MAP,   # swap any map dict here
    dt          = 0.1,        # seconds per step
    num_rays    = 360,        # lidar beams
    noise       = True,       # process + sensor noise
    render_mode = "human",    # "human" | "rgb_array" | None
)

obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()   # [v_cmd, alpha_cmd]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
```

### Observation space

`Box(363,)` = `[lidar_d_0 … lidar_d_359, x, y, theta]`

| Slice      | Content                        |
|------------|--------------------------------|
| `obs[:360]`| lidar distances [0, max_range] |
| `obs[360]` | robot x [m]                    |
| `obs[361]` | robot y [m]                    |
| `obs[362]` | robot heading θ [−π, π]       |

### Action space

`Box(2,)` = `[v_cmd, alpha_cmd]` in range `[−100, 100]`

These are the same raw command units as your original simulator. The motion
model converts them to physical velocity and steering angle internally.

---

## Swapping maps

### At construction time

```python
env = SLAMEnv(map_dict=L_SHAPED_ROOM)
```

### At reset time (via options)

```python
obs, info = env.reset(options={"map_dict": MAZE_MAP})
```

### Hot-swap mid-episode

```python
env.load_map(MAZE_MAP)
obs, info = env.reset()
```

### Adding a custom map

A map is just a Python dict. Copy any existing map file and edit:

```python
MY_MAP = {
    "name": "my_map",
    "description": "Custom environment",
    "robot_start": [0.5, 0.5, 0.0],   # [x, y, theta_rad]
    "bounds": (-0.1, 5.1, -0.1, 5.1), # for renderer axis limits
    "walls": [
        (0.0, 0.0, 5.0, 0.0),  # (x1, y1, x2, y2)
        (5.0, 0.0, 5.0, 5.0),
        (5.0, 5.0, 0.0, 5.0),
        (0.0, 5.0, 0.0, 0.0),
    ],
    "obstacles": [
        # (centre_x, centre_y, half_width, half_height, height)
        (2.5, 2.5, 0.3, 0.3, 0.4),
    ]
}
```

---

## Motion model

`utils/motion_model.py` exactly mirrors your calibrated model:

| Constant       | Value       | Meaning                         |
|----------------|-------------|----------------------------------|
| `V_M`          | 0.004808    | Velocity gain                    |
| `V_C`          | −0.045557   | Velocity offset                  |
| `VAR_V`        | 0.00057829  | Process noise σ² for velocity    |
| `DELTA_COEFFS` | [0.000027, 0.007798, 0.029847] | Steering polynomial |
| `VAR_DELTA`    | 0.00023134  | Process noise σ² for steering    |
| `L`            | 0.145 m     | Wheelbase                        |

The `MotionModel` class also exposes `jacobian_F()` and `process_noise_Q()`
for EKF/UKF integration.

---

## Lidar sensor

`utils/lidar_sensor.py` — vectorised numpy ray-caster:

| Parameter    | Default | Meaning                      |
|--------------|---------|-------------------------------|
| `num_rays`   | 360     | Angular resolution (1°/ray)  |
| `max_range`  | 5.0 m   | Max sensing distance          |
| `VAR_LIDAR`  | 0.000363| Range noise variance σ²_z    |

Helper methods:
- `scan(x, y, θ)` → `(angles, distances)` — same return format as your original
- `scan_as_points(x, y, θ)` → `(hit_x, hit_y)` in world frame
- `observation_vector(x, y, θ)` → `np.ndarray` ready for RL obs stacking
- `expected_range(x, y, θ, ray_idx)` → noiseless h(x) for EKF

---

## MuJoCo visualisation

The MuJoCo XML is auto-generated from the map dict by `MapLoader`.
Two cameras are always present:

| Camera       | Position        | Use                              |
|--------------|-----------------|-----------------------------------|
| `top_down`   | Directly above  | Default render view              |
| `follow_cam` | Offset above    | Can be aimed at robot in viewer  |

In **`human`** mode (`render_mode="human"`), `mujoco.viewer.launch_passive()`
opens an interactive window you can rotate/zoom freely.

In **`rgb_array`** mode, `mujoco.Renderer` renders off-screen (requires OpenGL / EGL):

```bash
# On headless servers with a GPU:
MUJOCO_GL=egl python visualize_sim.py
```

---

## Real-robot bridge (next step)

The environment is structured so that `SLAMEnv.step()` calls:
1. `MotionModel.step()` — prediction
2. `LidarSensor.scan()` — simulated measurement

For a real robot, subclass `SLAMEnv` and override those two calls to read
from your actual hardware interfaces instead.

---

## Roadmap

- [ ] EKF / Particle Filter SLAM plug-in interface
- [ ] Occupancy-grid map building helper
- [ ] Goal-conditioned reward for Active SLAM exploration
- [ ] ROS 2 bridge for real-robot deployment
- [ ] Gymnasium `register()` entry-points for `gym.make("SLAM-v0")`
