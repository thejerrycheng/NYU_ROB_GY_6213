"""
motion_model.py
===============
Calibrated bicycle / Ackermann motion model with vehicle dynamics (inertia).

Two classes:

  MotionModel       — pure kinematic bicycle model (original, used by SLAM estimators)
  VehicleDynamics   — longitudinal + angular inertia layer on top of MotionModel
                      (used by the teleop / sim loop for realistic feel)

When v_cmd drops to 0, VehicleDynamics coasts and decelerates under
rolling resistance rather than stopping instantly.
"""

import math
import random
from dataclasses import dataclass, field


# ── Calibrated constants ──────────────────────────────────────────────────────
L            = 0.145         # wheelbase [m]
V_M          = 0.004808      # velocity gain
V_C          = -0.045557     # velocity offset
VAR_V        = 0.00057829    # process noise σ² for velocity
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]   # steering polynomial coeffs
VAR_DELTA    = 0.00023134    # process noise σ² for steering angle

# ── Vehicle dynamics constants ────────────────────────────────────────────────
MASS         = 1.5           # kg   total vehicle mass
TAU_ACCEL    = 0.50          # s    motor time-constant while accelerating
TAU_BRAKE    = 0.25          # s    motor time-constant while braking / stopping
MU_ROLL      = 0.02          # rolling-resistance coefficient (wheels on floor)
G            = 9.81          # m/s²
V_STOP_EPS   = 5e-4          # m/s  speed below which static friction holds


@dataclass
class RobotState:
    x:     float = 0.0
    y:     float = 0.0
    theta: float = 0.0

    def as_list(self):  return [self.x, self.y, self.theta]
    def as_tuple(self): return (self.x, self.y, self.theta)


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic bicycle model  (no inertia — used by SLAM estimators)
# ─────────────────────────────────────────────────────────────────────────────

class MotionModel:
    """
    Pure kinematic bicycle model.
    v_cmd → v_phys instantly (no lag).  Used by EKF/particle-filter.
    """

    def __init__(self, noise: bool = True, wheelbase: float = L):
        self.noise = noise
        self.L     = wheelbase

    def step(self, state: RobotState, v_cmd: float, alpha_cmd: float,
             dt: float = 0.1) -> RobotState:
        v_phys, delta_phys = self._map_commands(v_cmd, alpha_cmd)
        if self.noise and abs(v_phys) > 0:
            v_phys     += random.gauss(0, math.sqrt(VAR_V))
            delta_phys += random.gauss(0, math.sqrt(VAR_DELTA))
        return self._integrate(state, v_phys, delta_phys, dt)

    def step_noiseless(self, state: RobotState, v_cmd: float, alpha_cmd: float,
                       dt: float = 0.1) -> RobotState:
        v_phys, delta_phys = self._map_commands(v_cmd, alpha_cmd)
        return self._integrate(state, v_phys, delta_phys, dt)

    def jacobian_F(self, state: RobotState, v_cmd: float, alpha_cmd: float,
                   dt: float = 0.1):
        v_phys, _ = self._map_commands(v_cmd, alpha_cmd)
        return [
            [1, 0, -v_phys * math.sin(state.theta) * dt],
            [0, 1,  v_phys * math.cos(state.theta) * dt],
            [0, 0,  1],
        ]

    def process_noise_Q(self, dt: float = 0.1):
        sv = math.sqrt(VAR_V) * dt
        sd = math.sqrt(VAR_DELTA) * dt
        return [[sv**2, 0, 0], [0, sv**2, 0], [0, 0, sd**2]]

    def _map_commands(self, v_cmd: float, alpha_cmd: float):
        """
        Map raw commands → (v_phys [m/s], delta_phys [rad]).
        Symmetric deadband: commands inside ±9.5 produce zero physical speed.
        """
        if v_cmd == 0.0:
            v_phys = 0.0
        elif v_cmd > 0.0:
            v_phys = V_M * v_cmd + V_C
            if v_phys < 0.0: v_phys = 0.0
        else:
            v_phys = -(V_M * abs(v_cmd) + V_C)
            if v_phys > 0.0: v_phys = 0.0

        delta_phys = (DELTA_COEFFS[0] * alpha_cmd ** 2
                      + DELTA_COEFFS[1] * alpha_cmd
                      + DELTA_COEFFS[2])
        return v_phys, delta_phys

    def _integrate(self, state: RobotState, v_phys: float,
                   delta_phys: float, dt: float) -> RobotState:
        w  = (v_phys * math.tan(delta_phys)) / self.L if self.L > 0 else 0.0
        nx = state.x + v_phys * math.cos(state.theta) * dt
        ny = state.y + v_phys * math.sin(state.theta) * dt
        nt = _angle_wrap(state.theta - w * dt)
        return RobotState(nx, ny, nt)


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle dynamics  (inertia + rolling resistance — used by teleop)
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDynamics:
    """
    First-order longitudinal dynamics with rolling resistance.

    Wraps MotionModel so the bicycle kinematics are still used for
    position / heading integration — but the speed fed in is the
    *actual* vehicle speed evolving under inertia, not the raw target.

    Physics
    -------
    v_target = _map_commands(v_cmd)            ← what the motor is aiming for
    a_motor  = (v_target − v_current) / tau    ← first-order motor lag
    a_roll   = −μ·g·sign(v)                    ← rolling resistance
    v_dot    = a_motor + a_roll

    When |v| < V_STOP_EPS and v_cmd==0 → static friction holds (v=0).

    Parameters
    ----------
    tau_accel  : motor time constant while speeding up  [s]
    tau_brake  : motor time constant while slowing down [s]
    mu_roll    : rolling-resistance coefficient
    noise      : add Gaussian noise to steering angle

    Usage
    -----
    dyn = VehicleDynamics()
    # each control step:
    state, v_actual = dyn.step(state, v_actual, v_cmd, alpha_cmd, dt)
    """

    def __init__(self,
                 tau_accel: float = TAU_ACCEL,
                 tau_brake: float = TAU_BRAKE,
                 mu_roll:   float = MU_ROLL,
                 noise:     bool  = False,
                 wheelbase: float = L):
        self.tau_accel = tau_accel
        self.tau_brake = tau_brake
        self.mu_roll   = mu_roll
        self.noise     = noise
        self.L         = wheelbase
        self._km = MotionModel(noise=False, wheelbase=wheelbase)

    def step(self,
             state:     RobotState,
             v_current: float,
             v_cmd:     float,
             alpha_cmd: float,
             dt:        float = 0.05) -> tuple[RobotState, float]:
        """
        Advance one control timestep.

        Parameters
        ----------
        state      : current robot pose
        v_current  : current physical speed [m/s]  (maintained across steps)
        v_cmd      : raw velocity command
        alpha_cmd  : raw steering command
        dt         : timestep [s]

        Returns
        -------
        (new_state, new_v_current)
        """
        v_target, delta_phys = self._km._map_commands(v_cmd, alpha_cmd)

        if self.noise:
            delta_phys += random.gauss(0, math.sqrt(VAR_DELTA))

        new_v = self._update_speed(v_current, v_target, v_cmd, dt)

        # Integrate pose with the actual (inertia-filtered) speed
        new_state = self._km._integrate(state, new_v, delta_phys, dt)
        return new_state, new_v

    def reset_speed(self) -> float:
        """Call this on robot reset to clear velocity state."""
        return 0.0

    # ── Private ───────────────────────────────────────────────────────────────

    def _update_speed(self, v: float, v_target: float,
                      v_cmd: float, dt: float) -> float:
        """Longitudinal dynamics integration (Euler)."""

        # Static friction: hold at rest when commanded to stop
        if abs(v) <= V_STOP_EPS and abs(v_target) < V_STOP_EPS:
            return 0.0

        # Motor acceleration (first-order lag toward target)
        speeding_up = abs(v_target) >= abs(v)
        tau     = self.tau_accel if speeding_up else self.tau_brake
        a_motor = (v_target - v) / tau

        # Rolling resistance (always opposes motion)
        if v > V_STOP_EPS:
            a_roll = -self.mu_roll * G
        elif v < -V_STOP_EPS:
            a_roll =  self.mu_roll * G
        else:
            a_roll = 0.0

        v_new = v + (a_motor + a_roll) * dt

        # Prevent rolling resistance from reversing direction when stopping
        if v_cmd == 0.0:
            if v > 0 and v_new < 0:
                return 0.0
            if v < 0 and v_new > 0:
                return 0.0

        return v_new

    # ── Accessors for logging ─────────────────────────────────────────────────

    @staticmethod
    def acceleration(v_current: float, v_target: float,
                     tau: float, mu: float = MU_ROLL) -> float:
        """Instantaneous acceleration [m/s²] — useful for logging."""
        a_motor = (v_target - v_current) / tau
        a_roll  = -mu * G * (1 if v_current > 0 else -1 if v_current < 0 else 0)
        return a_motor + a_roll


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi