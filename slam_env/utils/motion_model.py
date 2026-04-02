"""
motion_model.py
===============
Calibrated bicycle / Ackermann motion model + vehicle dynamics.
All hyperparameters are loaded from  config/vehicle_config.yaml.

Two classes:
  MotionModel     — pure kinematic bicycle model (for SLAM estimators / EKF)
  VehicleDynamics — adds longitudinal inertia + rolling resistance (for teleop)
"""

import math, random, yaml
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CFG = Path(__file__).parent.parent / "config" / "vehicle_config.yaml"


def load_vehicle_config(path=None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_CFG
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Module-level constants (populated on first import, can be refreshed) ──────
def _load_globals(cfg=None):
    c = cfg or load_vehicle_config()
    k = c["kinematics"]
    d = c["dynamics"]
    return {
        "L":            float(k["wheelbase_m"]),
        "V_M":          float(k["velocity_gain"]),
        "V_C":          float(k["velocity_offset"]),
        "DELTA_COEFFS": list(k["steering_poly"]),
        "MAX_STEER":    math.radians(float(k["max_steer_deg"])),
        "VAR_V":        float(k["noise_sigma_v"])**2,
        "VAR_DELTA":    float(k["noise_sigma_delta"])**2,
        "MASS":         float(d["mass_kg"]),
        "TAU_ACCEL":    float(d["tau_accel_s"]),
        "TAU_BRAKE":    float(d["tau_brake_s"]),
        "MU_ROLL":      float(d["mu_rolling"]),
        "V_STOP_EPS":   float(d["v_stop_eps"]),
    }

_G = _load_globals()

# Expose as module-level names for backward-compatibility
L            = _G["L"]
V_M          = _G["V_M"]
V_C          = _G["V_C"]
DELTA_COEFFS = _G["DELTA_COEFFS"]
MAX_STEER    = _G["MAX_STEER"]
VAR_V        = _G["VAR_V"]
VAR_DELTA    = _G["VAR_DELTA"]
MASS         = _G["MASS"]
TAU_ACCEL    = _G["TAU_ACCEL"]
TAU_BRAKE    = _G["TAU_BRAKE"]
MU_ROLL      = _G["MU_ROLL"]
V_STOP_EPS   = _G["V_STOP_EPS"]
G            = 9.81


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    def as_list(self):  return [self.x, self.y, self.theta]
    def as_tuple(self): return (self.x, self.y, self.theta)


# ─────────────────────────────────────────────────────────────────────────────
class MotionModel:
    """
    Pure kinematic Ackermann / bicycle model.
    No inertia — command maps directly to physical speed/steer.
    Used by SLAM estimators (EKF, particle filter).
    """

    def __init__(self, noise: bool = True, config=None):
        cfg = load_vehicle_config(config) if isinstance(config, (str, Path)) else \
              config if isinstance(config, dict) else load_vehicle_config()
        g = cfg["kinematics"]
        self.noise        = noise
        self.L            = float(g["wheelbase_m"])
        self.V_M          = float(g["velocity_gain"])
        self.V_C          = float(g["velocity_offset"])
        self.DELTA_COEFFS = list(g["steering_poly"])
        self.MAX_STEER    = math.radians(float(g["max_steer_deg"]))
        self.sigma_v      = float(g["noise_sigma_v"])
        self.sigma_d      = float(g["noise_sigma_delta"])

    def step(self, state: RobotState, v_cmd: float, alpha_cmd: float,
             dt: float = 0.1) -> RobotState:
        v_phys, delta = self._map_commands(v_cmd, alpha_cmd)
        if self.noise and abs(v_phys) > 0:
            v_phys += random.gauss(0, self.sigma_v)
            delta  += random.gauss(0, self.sigma_d)
        return self._integrate(state, v_phys, delta, dt)

    def step_noiseless(self, state: RobotState, v_cmd: float, alpha_cmd: float,
                       dt: float = 0.1) -> RobotState:
        v_phys, delta = self._map_commands(v_cmd, alpha_cmd)
        return self._integrate(state, v_phys, delta, dt)

    def jacobian_F(self, state: RobotState, v_cmd: float, alpha_cmd: float,
                   dt: float = 0.1):
        v_phys, _ = self._map_commands(v_cmd, alpha_cmd)
        return [
            [1, 0, -v_phys * math.sin(state.theta) * dt],
            [0, 1,  v_phys * math.cos(state.theta) * dt],
            [0, 0,  1],
        ]

    def process_noise_Q(self, dt: float = 0.1):
        sv = self.sigma_v * dt; sd = self.sigma_d * dt
        return [[sv**2,0,0],[0,sv**2,0],[0,0,sd**2]]

    def _map_commands(self, v_cmd: float, alpha_cmd: float):
        """
        v_cmd  → v_phys  [m/s]  (symmetric deadband, supports reverse)
        alpha_cmd → delta [rad] (polynomial, clamped to ±MAX_STEER)
        """
        if v_cmd == 0.0:
            v_phys = 0.0
        elif v_cmd > 0.0:
            v_phys = self.V_M * v_cmd + self.V_C
            if v_phys < 0: v_phys = 0.0
        else:
            v_phys = -(self.V_M * abs(v_cmd) + self.V_C)
            if v_phys > 0: v_phys = 0.0

        c = self.DELTA_COEFFS
        delta = c[0]*alpha_cmd**2 + c[1]*alpha_cmd + c[2]
        # Clamp to maximum mechanical steering angle
        delta = max(-self.MAX_STEER, min(self.MAX_STEER, delta))
        return v_phys, delta

    def _integrate(self, state: RobotState, v: float, delta: float,
                   dt: float) -> RobotState:
        w  = (v * math.tan(delta)) / self.L if self.L > 0 else 0.0
        nx = state.x + v * math.cos(state.theta) * dt
        ny = state.y + v * math.sin(state.theta) * dt
        nt = _wrap(state.theta - w * dt)
        return RobotState(nx, ny, nt)


# ─────────────────────────────────────────────────────────────────────────────
class VehicleDynamics:
    """
    Adds longitudinal inertia + rolling resistance on top of MotionModel.
    v_current is a persistent speed state (maintained by the caller).

    Usage:
        dyn = VehicleDynamics()
        v = 0.0
        while running:
            state, v = dyn.step(state, v, v_cmd, alpha_cmd, dt)
    """

    def __init__(self, noise: bool = False, config=None):
        cfg = load_vehicle_config(config) if isinstance(config, (str, Path)) else \
              config if isinstance(config, dict) else load_vehicle_config()
        d = cfg["dynamics"]
        self.tau_a   = float(d["tau_accel_s"])
        self.tau_b   = float(d["tau_brake_s"])
        self.mu      = float(d["mu_rolling"])
        self.eps     = float(d["v_stop_eps"])
        self._km     = MotionModel(noise=False, config=cfg)
        self.noise   = noise

    def step(self, state: RobotState, v_current: float,
             v_cmd: float, alpha_cmd: float, dt: float = 0.05):
        v_target, delta = self._km._map_commands(v_cmd, alpha_cmd)
        if self.noise:
            delta += random.gauss(0, self._km.sigma_d)
        v_new = self._speed_step(v_current, v_target, v_cmd, dt)
        new_state = self._km._integrate(state, v_new, delta, dt)
        return new_state, v_new

    def reset_speed(self) -> float:
        return 0.0

    def _speed_step(self, v, v_t, v_cmd, dt):
        if abs(v) <= self.eps and abs(v_t) < self.eps:
            return 0.0
        speeding = abs(v_t) >= abs(v)
        tau = self.tau_a if speeding else self.tau_b
        a_m = (v_t - v) / tau
        a_r = -self.mu * G * (1 if v > self.eps else (-1 if v < -self.eps else 0))
        v_new = v + (a_m + a_r) * dt
        if v_cmd == 0.0:
            if v > 0 and v_new < 0: return 0.0
            if v < 0 and v_new > 0: return 0.0
        return v_new


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi