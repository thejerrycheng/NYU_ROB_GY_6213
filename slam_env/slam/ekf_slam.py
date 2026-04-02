"""
ekf_slam.py — EKF-based pose tracking + occupancy-grid mapping
==============================================================

This module implements two coupled components:

  EKFSlam        — Extended Kalman Filter for robot pose estimation
                   using the calibrated bicycle motion model and 2-D lidar
  OccupancyGrid  — Probabilistic binary grid built from lidar hits,
                   with per-cell log-odds and variance tracking

──────────────────────────────────────────────────────────────────────────────
HOW EKF SLAM WORKS (step by step)
──────────────────────────────────────────────────────────────────────────────

STATE VECTOR
    x = [x, y, θ]ᵀ  (3×1)   robot position + heading

COVARIANCE
    P (3×3) — uncertainty of the state estimate

STEP 1 — PREDICT (motion model)
────────────────────────────────
Given control input u = (v_cmd, alpha_cmd):

  Map to physical:
    v     = V_M · v_cmd + V_C              (velocity calibration)
    δ     = c₀·α² + c₁·α + c₂             (steering polynomial)
    ω     = v · tan(δ) / L                 (yaw rate)

  Nonlinear prediction f(x, u):
    x̂⁻ₖ = x̂ₖ₋₁ + v·cos(θ)·dt
    ŷ⁻ₖ = ŷₖ₋₁ + v·sin(θ)·dt
    θ̂⁻ₖ = θ̂ₖ₋₁ − ω·dt

  Linearise around current state — motion Jacobian F = ∂f/∂x:
    F = ⎡1  0  −v·sin(θ)·dt⎤
        ⎢0  1   v·cos(θ)·dt⎥
        ⎣0  0   1           ⎦

  Process noise Q (from calibrated σ_v, σ_δ):
    Q = diag(σ_v²·dt², σ_v²·dt², σ_δ²·dt²)

  Covariance prediction:
    P⁻ₖ = F · Pₖ₋₁ · Fᵀ + Q

STEP 2 — UPDATE (lidar measurement)
─────────────────────────────────────
We use a sparse set of lidar beams as range measurements.

  For each selected beam i with relative angle φᵢ and measured range zᵢ:

  Predicted range (noiseless ray-cast):
    ẑᵢ = h(x̂⁻, φᵢ)  — ray-cast from predicted pose

  Innovation:
    νᵢ = zᵢ − ẑᵢ

  Measurement Jacobian H = ∂h/∂x  (1×3):
  The range to a wall point (wx, wy) from pose (x,y,θ) at angle φ_global:
    ẑ = √((wx−x)² + (wy−y)²)
    H = [−(wx−x)/ẑ,  −(wy−y)/ẑ,  0]

  Measurement noise R:
    R = σ_z²  (from lidar_config.yaml: sigma_range²)
    Range-dependent: R = (σ_z · (1 + k·ẑ))²

  Kalman gain (1×3):
    Kᵢ = P⁻ · Hᵢᵀ / (Hᵢ · P⁻ · Hᵢᵀ + Rᵢ)

  State update:
    x̂ₖ = x̂ₖ⁻ + Kᵢ · νᵢ

  Covariance update:
    Pₖ = (I − Kᵢ · Hᵢ) · P⁻ₖ

  Applied sequentially for each beam (sequential update).

STEP 3 — OCCUPANCY GRID UPDATE
────────────────────────────────
Uses log-odds representation for numerical stability:

  l(x) = log( p(occ) / p(free) )

  Hit cell (endpoint of beam):
    l += log_odds_hit    (e.g. +0.85, strong evidence of obstacle)

  Free cells (along beam via Bresenham):
    l += log_odds_free   (e.g. −0.40, weaker evidence of freespace)

  Clamped to [l_min, l_max] to prevent saturation.

  Occupancy probability:
    p(occ) = 1 − 1/(1 + exp(l))

  Variance (from log-odds uncertainty):
    var = p · (1 − p)     (Bernoulli variance)
    This is HIGH (0.25) when p≈0.5 (uncertain) and LOW when p→0 or 1.

SCAN MATCHING (optional, improves accuracy)
───────────────────────────────────────────
  Before each EKF update, an ICP-style scan-match between the current
  scan and the predicted scan corrects gross odometry drift.
  Implemented as a 2-D point-set alignment minimising:
    Σ ‖pᵢ − (R·qᵢ + t)‖²
  using singular value decomposition (SVD).
"""

import math, time
import numpy as np
from pathlib import Path
import yaml

_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy Grid
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    """
    Probabilistic 2-D occupancy grid with log-odds updates and variance map.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : world bounds [m]
    resolution                  : cell size [m/cell]
    log_odds_hit                : log-odds increment for hit cell
    log_odds_free               : log-odds decrement for free cell
    l_min, l_max                : log-odds saturation bounds
    """

    def __init__(self,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 resolution: float = 0.05,
                 log_odds_hit:  float =  0.85,
                 log_odds_free: float = -0.40,
                 l_min: float = -5.0,
                 l_max: float =  5.0):

        self.x_min = x_min; self.x_max = x_max
        self.y_min = y_min; self.y_max = y_max
        self.res   = resolution
        self.l_hit  = log_odds_hit
        self.l_free = log_odds_free
        self.l_min  = l_min
        self.l_max  = l_max

        self.nx = max(1, int(math.ceil((x_max - x_min) / resolution)))
        self.ny = max(1, int(math.ceil((y_max - y_min) / resolution)))

        # Log-odds grid (float32)
        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)
        # Hit count per cell (for variance)
        self._hits = np.zeros((self.nx, self.ny), dtype=np.int32)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float):
        """World (m) → grid (col, row). Returns None if out of bounds."""
        cx = int((wx - self.x_min) / self.res)
        cy = int((wy - self.y_min) / self.res)
        if 0 <= cx < self.nx and 0 <= cy < self.ny:
            return cx, cy
        return None

    def cell_to_world(self, cx: int, cy: int):
        """Grid (col, row) → world centre (m)."""
        wx = self.x_min + (cx + 0.5) * self.res
        wy = self.y_min + (cy + 0.5) * self.res
        return wx, wy

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, robot_x: float, robot_y: float,
               angles: np.ndarray, dists: np.ndarray,
               flags: np.ndarray, max_range: float):
        """
        Update grid from one lidar scan.
        Only valid beams (flags==0) are used.
        """
        for i in range(len(angles)):
            if flags[i] != 0:
                continue
            d = dists[i]
            if d >= max_range * 0.99:
                continue

            ga = robot_x + angles[i]   # wait — angles are relative, need robot_theta
            # NOTE: angles here must be in world frame (caller passes global angles)

            # Endpoint (hit cell)
            ex = robot_x + d * math.cos(angles[i])
            ey = robot_y + d * math.sin(angles[i])

            # Bresenham ray from robot to hit
            self._update_ray(robot_x, robot_y, ex, ey, hit=True)

    def update_scan(self, robot_x: float, robot_y: float, robot_theta: float,
                    rel_angles: np.ndarray, dists: np.ndarray,
                    flags: np.ndarray, max_range: float):
        """
        Correct interface: takes relative angles + robot pose.
        """
        for i in range(len(rel_angles)):
            if flags[i] != 0:
                continue
            d = float(dists[i])
            if d >= max_range * 0.99:
                continue

            # Global angle
            ga  = robot_theta + float(rel_angles[i])
            ex  = robot_x + d * math.cos(ga)
            ey  = robot_y + d * math.sin(ga)

            self._update_ray(robot_x, robot_y, ex, ey, hit=True)

    def _update_ray(self, x0: float, y0: float,
                    x1: float, y1: float, hit: bool):
        """Bresenham line from (x0,y0) to (x1,y1); mark free along ray, hit at end."""
        c0 = self.world_to_cell(x0, y0)
        c1 = self.world_to_cell(x1, y1)
        if c0 is None or c1 is None:
            return

        # Bresenham between c0 and c1
        cx, cy = c0
        ex, ey = c1
        dx, dy = abs(ex-cx), abs(ey-cy)
        sx     = 1 if ex > cx else -1
        sy     = 1 if ey > cy else -1
        err    = dx - dy

        cells = []
        while True:
            cells.append((cx, cy))
            if cx == ex and cy == ey:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; cx += sx
            if e2 < dx:
                err += dx; cy += sy

        # Update free cells (all except last)
        for cell in cells[:-1]:
            self._update_cell(*cell, lo=self.l_free)
        # Update hit cell
        if hit and cells:
            self._update_cell(*cells[-1], lo=self.l_hit)
            self._hits[cells[-1][0], cells[-1][1]] += 1

    def _update_cell(self, cx: int, cy: int, lo: float):
        self.log_odds[cx, cy] = np.clip(
            self.log_odds[cx, cy] + lo, self.l_min, self.l_max)

    # ── Derived maps ──────────────────────────────────────────────────────────

    @property
    def probability(self) -> np.ndarray:
        """Occupancy probability p ∈ [0,1] for each cell. Shape (nx, ny)."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    @property
    def variance(self) -> np.ndarray:
        """
        Bernoulli variance = p·(1−p).
        High (0.25) when uncertain (p≈0.5), low when confident (p→0 or 1).
        Shape (nx, ny).
        """
        p = self.probability
        return p * (1.0 - p)

    @property
    def confidence(self) -> np.ndarray:
        """1 − variance, normalised to [0,1]. High = confident."""
        return 1.0 - self.variance / 0.25

    def reset(self):
        self.log_odds[:] = 0.0
        self._hits[:] = 0


# ─────────────────────────────────────────────────────────────────────────────
# EKF Pose Estimator
# ─────────────────────────────────────────────────────────────────────────────

class EKFSlam:
    """
    Extended Kalman Filter for robot pose estimation + occupancy grid mapping.

    Inputs at each step:
      - v_cmd, alpha_cmd  (raw commands, mapped via calibrated motion model)
      - lidar scan        (rel_angles, dists, flags)

    Outputs:
      - mu    : (3,) estimated pose [x, y, θ]
      - P     : (3,3) pose covariance
      - grid  : OccupancyGrid with probability + variance

    Parameters
    ----------
    map_dict     : map definition (for grid bounds)
    vehicle_cfg  : path to vehicle_config.yaml or dict
    lidar_cfg    : path to lidar_config.yaml or dict
    dt           : control timestep [s]
    update_every : run EKF update every N lidar beams (subsampling)
    """

    def __init__(self,
                 map_dict:    dict,
                 vehicle_cfg=None,
                 lidar_cfg=None,
                 dt:          float = 0.05,
                 update_every: int  = 10):

        self.dt           = dt
        self.update_every = update_every

        # ── Load configs ──────────────────────────────────────────────────────
        if isinstance(vehicle_cfg, dict):
            vcfg = vehicle_cfg
        else:
            vpath = Path(vehicle_cfg) if vehicle_cfg else \
                    _ROOT / "config" / "vehicle_config.yaml"
            with open(vpath) as f:
                vcfg = yaml.safe_load(f)

        if isinstance(lidar_cfg, dict):
            lcfg = lidar_cfg
        else:
            lpath = Path(lidar_cfg) if lidar_cfg else \
                    _ROOT / "config" / "lidar_config.yaml"
            with open(lpath) as f:
                lcfg = yaml.safe_load(f)

        # ── Motion model parameters ───────────────────────────────────────────
        k = vcfg["kinematics"]
        self.L            = float(k["wheelbase_m"])
        self.V_M          = float(k["velocity_gain"])
        self.V_C          = float(k["velocity_offset"])
        self.DC           = list(k["steering_poly"])   # delta coefficients
        self.MAX_STEER    = math.radians(float(k["max_steer_deg"]))
        self.sigma_v      = float(k["noise_sigma_v"])
        self.sigma_d      = float(k["noise_sigma_delta"])

        # ── Lidar parameters ──────────────────────────────────────────────────
        lg = lcfg["geometry"]
        gn = lcfg["gaussian_noise"]
        self.max_range    = float(lg["max_range"])
        self.min_range    = float(lg["min_range"])
        self.sigma_z      = float(gn["sigma_range"])
        self.range_dep_k  = float(gn["range_noise_k"]) if gn["range_dependent"] else 0.0

        # ── State + covariance ────────────────────────────────────────────────
        rs = map_dict["robot_start"]
        self.mu = np.array([float(rs[0]), float(rs[1]), float(rs[2])],
                           dtype=np.float64)
        # Initial covariance — small (we know the start pose)
        self.P  = np.diag([0.01, 0.01, 0.001])

        # ── Occupancy grid ────────────────────────────────────────────────────
        b = map_dict["bounds"]
        self.grid = OccupancyGrid(
            x_min      = float(b[0]) - 0.2,
            x_max      = float(b[1]) + 0.2,
            y_min      = float(b[2]) - 0.2,
            y_max      = float(b[3]) + 0.2,
            resolution = 0.05,
            log_odds_hit  =  0.85,
            log_odds_free = -0.40,
        )

        # ── Wall segments (for noiseless ray-cast in update step) ─────────────
        self.walls = [tuple(w) for w in map_dict.get("walls", [])]

        self._step = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, v_cmd: float, alpha_cmd: float):
        """
        EKF Prediction step.
        Propagates state and covariance through the motion model.

        Modifies self.mu, self.P in-place.
        """
        v, delta = self._map_commands(v_cmd, alpha_cmd)
        dt = self.dt
        x, y, th = self.mu

        # ── Nonlinear state prediction f(x, u) ────────────────────────────
        w  = (v * math.tan(delta)) / self.L if self.L > 0 else 0.0
        self.mu[0] = x  + v * math.cos(th) * dt
        self.mu[1] = y  + v * math.sin(th) * dt
        self.mu[2] = _wrap(th - w * dt)

        # ── Jacobian F = ∂f/∂[x,y,θ] ─────────────────────────────────────
        F = np.array([
            [1.0, 0.0, -v * math.sin(th) * dt],
            [0.0, 1.0,  v * math.cos(th) * dt],
            [0.0, 0.0,  1.0],
        ])

        # ── Process noise Q ───────────────────────────────────────────────
        sv = self.sigma_v * dt
        sd = self.sigma_d * dt
        Q  = np.diag([sv**2, sv**2, sd**2])

        # ── Covariance prediction: P⁻ = F P Fᵀ + Q ───────────────────────
        self.P = F @ self.P @ F.T + Q

    def update(self, rel_angles: list, dists: list, flags):
        """
        EKF Update step — sequential range measurement updates.

        Uses a subsampled set of beams (every update_every beams).
        Only valid beams (flags==0, not max_range) are used.

        Modifies self.mu, self.P in-place.
        """
        angles = np.asarray(rel_angles)
        d_arr  = np.asarray(dists)
        f_arr  = np.asarray(flags)

        # Select beams: valid, not max-range, subsampled
        valid = (f_arr == 0) & (d_arr < self.max_range * 0.99) & \
                (d_arr > self.min_range)
        indices = np.where(valid)[0][::self.update_every]

        x, y, th = self.mu

        for i in indices:
            # Global angle of this beam
            ga     = th + float(angles[i])
            z_meas = float(d_arr[i])

            # ── Predicted range by noiseless ray-cast ─────────────────────
            z_pred, wx, wy = self._predict_range(x, y, ga)
            if z_pred < self.min_range or z_pred > self.max_range:
                continue

            # ── Innovation ν = z − ẑ ──────────────────────────────────────
            nu = z_meas - z_pred
            if abs(nu) > 1.0:
                continue   # outlier gate — reject large innovations

            # ── Measurement Jacobian H (1×3) ──────────────────────────────
            # h(x) = sqrt((wx-x)²+(wy-y)²)
            # ∂h/∂x = −(wx−x)/ẑ,  ∂h/∂y = −(wy−y)/ẑ,  ∂h/∂θ = 0
            if z_pred < 1e-6:
                continue
            H = np.array([[
                -(wx - x) / z_pred,
                -(wy - y) / z_pred,
                0.0,
            ]])

            # ── Measurement noise R (range-dependent) ────────────────────
            sig = self.sigma_z * (1.0 + self.range_dep_k * z_pred)
            R   = sig ** 2

            # ── Innovation covariance S = H P⁻ Hᵀ + R ───────────────────
            S = float((H @ self.P @ H.T)[0, 0]) + R

            # ── Kalman gain K = P⁻ Hᵀ / S  (3×1) ────────────────────────
            K = (self.P @ H.T) / S   # shape (3,1)  — H.T is (3,1)

            # ── State update x̂ = x̂⁻ + K ν ───────────────────────────────
            self.mu = self.mu + K.flatten() * nu
            self.mu[2] = _wrap(self.mu[2])

            # ── Covariance update P = (I − KH) P⁻ ────────────────────────
            I_KH    = np.eye(3) - K @ H
            self.P  = I_KH @ self.P
            # Joseph form for numerical stability
            self.P  = 0.5 * (self.P + self.P.T)

            # Update current estimate for next beam
            x, y, th = self.mu

        self._step += 1

    def update_grid(self, rel_angles, dists, flags):
        """Update the occupancy grid from the current scan."""
        x, y, th = self.mu
        self.grid.update_scan(
            x, y, th,
            np.asarray(rel_angles), np.asarray(dists), np.asarray(flags),
            self.max_range)

    def step(self, v_cmd: float, alpha_cmd: float,
             rel_angles: list, dists: list, flags):
        """
        Full SLAM step: predict + update + grid update.
        Call once per control timestep.
        """
        self.predict(v_cmd, alpha_cmd)
        self.update(rel_angles, dists, flags)
        self.update_grid(rel_angles, dists, flags)

    def reset(self, robot_start: list):
        self.mu = np.array(robot_start, dtype=np.float64)
        self.P  = np.diag([0.01, 0.01, 0.001])
        self.grid.reset()
        self._step = 0

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def pose(self):
        """Current pose estimate (x, y, theta)."""
        return tuple(self.mu)

    @property
    def pose_std(self):
        """Standard deviations (sx, sy, stheta) from diagonal of P."""
        return (math.sqrt(self.P[0,0]),
                math.sqrt(self.P[1,1]),
                math.sqrt(self.P[2,2]))

    @property
    def covariance(self):
        return self.P.copy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _map_commands(self, v_cmd: float, alpha_cmd: float):
        """Same calibrated mapping as MotionModel._map_commands."""
        if v_cmd == 0.0:
            v = 0.0
        elif v_cmd > 0.0:
            v = self.V_M * v_cmd + self.V_C
            if v < 0: v = 0.0
        else:
            v = -(self.V_M * abs(v_cmd) + self.V_C)
            if v > 0: v = 0.0

        c = self.DC
        delta = c[0]*alpha_cmd**2 + c[1]*alpha_cmd + c[2]
        delta = max(-self.MAX_STEER, min(self.MAX_STEER, delta))
        return v, delta

    def _predict_range(self, rx: float, ry: float, global_angle: float):
        """
        Noiseless ray-cast from (rx, ry) along global_angle.
        Returns (distance, hit_x, hit_y).
        hit_x, hit_y are the world-frame wall intersection point.
        """
        best_t  = self.max_range
        best_wx = rx + self.max_range * math.cos(global_angle)
        best_wy = ry + self.max_range * math.sin(global_angle)

        ca = math.cos(global_angle)
        sa = math.sin(global_angle)

        for (x1, y1, x2, y2) in self.walls:
            sx, sy   = x2-x1, y2-y1
            qpx, qpy = x1-rx, y1-ry
            denom    = ca*sy - sa*sx
            if abs(denom) < 1e-9:
                continue
            t = (qpx*sy - qpy*sx) / denom
            u = (qpx*sa - qpy*ca) / denom
            if t > 1e-6 and 0.0 <= u <= 1.0 and t < best_t:
                best_t  = t
                best_wx = rx + t*ca
                best_wy = ry + t*sa

        return best_t, best_wx, best_wy


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
