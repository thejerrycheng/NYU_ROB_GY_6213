"""
lidar_sensor.py
===============
Realistic 2-D ray-casting lidar simulation.

All noise parameters are loaded from  config/lidar_config.yaml  so you can
tune everything without touching code.

Noise model (applied in order):
  1. Angular jitter      — each beam direction has a small random deviation
  2. Gaussian range noise— additive, range-dependent (grows with distance)
  3. Dropout             — beam returns max_range (no detection)
  4. Short outliers      — spurious short return (glass, dust, cross-talk)
  5. Long outliers       — missed detection → max_range
  6. Mixed-pixel         — range-jump blending at depth discontinuities
  7. Min-range clamp     — values below min_range set to min_range

All are individually enable/disable-able in the YAML.

scan() returns (angles, distances, flags) where flags is a uint8 array:
  0 = valid hit
  1 = short outlier
  2 = long outlier / dropout
  3 = mixed pixel
  4 = min-range clamp
"""

import math
import os
import numpy as np
import yaml
from pathlib import Path


# ── Default config path ───────────────────────────────────────────────────────
_DEFAULT_CFG = Path(__file__).parent.parent / "config" / "lidar_config.yaml"


def load_lidar_config(path=None) -> dict:
    """Load and return the lidar config dict from YAML."""
    cfg_path = Path(path) if path else _DEFAULT_CFG
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class LidarSensor:
    """
    Realistic 2-D LiDAR sensor with full noise model.

    Parameters
    ----------
    walls    : list of (x1,y1,x2,y2) wall segments  (world frame)
    config   : path to lidar_config.yaml, or a pre-loaded dict.
               If None, uses the default config/lidar_config.yaml.
    noise    : global enable/disable for ALL noise (overrides YAML per-flag).
               Set False for a perfect noiseless sensor (e.g. for EKF h(x)).

    Usage
    -----
    lidar = LidarSensor(walls, noise=True)
    angles, dists, flags = lidar.scan(x, y, theta)
    """

    def __init__(self,
                 walls:  list,
                 config=None,
                 noise:  bool = True):

        self.walls = walls
        self.noise = noise

        # Load config
        if isinstance(config, dict):
            self.cfg = config
        else:
            self.cfg = load_lidar_config(config)

        self._apply_config()

    # ── Configuration ─────────────────────────────────────────────────────────

    def _apply_config(self):
        g   = self.cfg["geometry"]
        gn  = self.cfg["gaussian_noise"]
        an  = self.cfg["angular_noise"]
        out = self.cfg["outliers"]
        mp  = self.cfg["mixed_pixel"]
        dr  = self.cfg["dropout"]

        self.num_rays   = int(g["num_rays"])
        self.fov        = math.radians(float(g["fov_deg"]))
        self.start_angle= math.radians(float(g["start_angle_deg"]))
        self.min_range  = float(g["min_range"])
        self.max_range  = float(g["max_range"])

        self.gn_enabled  = bool(gn["enabled"])
        self.sigma_r     = float(gn["sigma_range"])
        self.range_dep   = bool(gn["range_dependent"])
        self.range_k     = float(gn["range_noise_k"])

        self.an_enabled  = bool(an["enabled"])
        self.sigma_a     = math.radians(float(an["sigma_angle_deg"]))

        self.out_enabled  = bool(out["enabled"])
        self.short_prob   = float(out["short_prob"])
        self.short_min_f  = float(out["short_min_frac"])
        self.long_prob    = float(out["long_prob"])

        self.mp_enabled   = bool(mp["enabled"])
        self.mp_jump      = float(mp["jump_threshold"])
        self.mp_blend     = float(mp["blend_fraction"])

        self.dr_enabled   = bool(dr["enabled"])
        self.dr_prob      = float(dr["prob"])

        # Pre-compute nominal ray angles (relative to robot heading)
        self._nom_angles = np.linspace(
            self.start_angle,
            self.start_angle + self.fov,
            self.num_rays,
            endpoint=False)

    def reload_config(self, path=None):
        """Hot-reload the YAML config during a running simulation."""
        self.cfg = load_lidar_config(path)
        self._apply_config()

    # ── Public scan API ───────────────────────────────────────────────────────

    def scan(self,
             robot_x: float,
             robot_y: float,
             robot_theta: float):
        """
        Fire all beams and return measurements.

        Returns
        -------
        angles    : (N,) float64  relative beam angles [rad]
        distances : (N,) float64  range measurements [m]
        flags     : (N,) uint8    0=valid,1=short_out,2=long_out/drop,3=mixed,4=minrange
        """
        N    = self.num_rays
        MISS = self.max_range

        # ── Step 1: angular jitter ────────────────────────────────────────────
        if self.noise and self.an_enabled:
            jitter = np.random.normal(0.0, self.sigma_a, N)
            rel_angles = self._nom_angles + jitter
        else:
            rel_angles = self._nom_angles.copy()

        global_angles = robot_theta + rel_angles

        # ── Step 2: ideal ray-cast ────────────────────────────────────────────
        dists_ideal = self._raycast(robot_x, robot_y, global_angles)

        # Working copy
        dists = dists_ideal.copy()
        flags = np.zeros(N, dtype=np.uint8)   # 0 = valid

        if not self.noise:
            dists = np.clip(dists, self.min_range, self.max_range)
            return list(rel_angles), list(dists), flags

        # ── Step 3: dropout ───────────────────────────────────────────────────
        if self.dr_enabled:
            drop_mask = (np.random.random(N) < self.dr_prob) & (dists < MISS)
            dists[drop_mask] = MISS
            flags[drop_mask] = 2

        # ── Step 4: Gaussian range noise (range-dependent σ) ─────────────────
        if self.gn_enabled:
            hit_mask = dists < MISS
            if self.range_dep:
                sigma_eff = self.sigma_r * (1.0 + self.range_k * dists)
            else:
                sigma_eff = np.full(N, self.sigma_r)
            noise_r = np.random.normal(0.0, sigma_eff, N)
            dists = np.where(hit_mask, dists + noise_r, dists)

        # ── Step 5: short outliers ────────────────────────────────────────────
        if self.out_enabled:
            # Only place short outliers on beams that had a real hit
            hit_mask = dists_ideal < MISS
            short_candidates = hit_mask & (np.random.random(N) < self.short_prob)
            if short_candidates.any():
                min_r = dists_ideal * self.short_min_f
                min_r = np.maximum(min_r, self.min_range)
                short_r = np.random.uniform(min_r, dists_ideal)
                dists[short_candidates] = short_r[short_candidates]
                flags[short_candidates] = 1

        # ── Step 6: long outliers ─────────────────────────────────────────────
            long_mask = (np.random.random(N) < self.long_prob) & (dists < MISS)
            dists[long_mask] = MISS
            flags[long_mask] = np.where(flags[long_mask] == 0, 2, flags[long_mask])

        # ── Step 7: mixed-pixel effect ────────────────────────────────────────
        if self.mp_enabled:
            # Find range jumps between adjacent beams
            d_roll = np.roll(dists, 1)
            jump   = np.abs(dists - d_roll) > self.mp_jump
            # Only on valid beams near the closer surface
            candidates = jump & (dists < MISS) & (flags == 0)
            blend_ev   = candidates & (np.random.random(N) < self.mp_blend)
            if blend_ev.any():
                alpha = np.random.uniform(0.0, 1.0, N)
                blended = alpha * dists + (1.0 - alpha) * d_roll
                dists[blend_ev] = blended[blend_ev]
                flags[blend_ev] = 3

        # ── Step 8: min-range clamp ───────────────────────────────────────────
        too_close = (dists < self.min_range) & (flags == 0)
        dists[too_close] = self.min_range
        flags[too_close] = 4

        # Final clamp
        dists = np.clip(dists, self.min_range, self.max_range)

        return list(rel_angles), list(dists), flags

    # ── Convenience helpers ───────────────────────────────────────────────────

    def scan_noiseless(self, robot_x, robot_y, robot_theta):
        """Pure ray-cast with no noise — for EKF h(x)."""
        global_angles = robot_theta + self._nom_angles
        d = self._raycast(robot_x, robot_y, global_angles)
        d = np.clip(d, self.min_range, self.max_range)
        return list(self._nom_angles), list(d)

    def scan_as_points(self, robot_x, robot_y, robot_theta):
        """
        Returns (hit_x, hit_y, out_x, out_y) in world frame.
        hit = valid returns;  out = outlier returns (flags != 0).
        """
        angles, dists, flags = self.scan(robot_x, robot_y, robot_theta)
        a   = np.array(angles) + robot_theta
        d   = np.array(dists)
        fl  = np.array(flags)

        valid   = (d < self.max_range) & (fl == 0)
        outlier = fl != 0

        hx = robot_x + d[valid]   * np.cos(a[valid])
        hy = robot_y + d[valid]   * np.sin(a[valid])
        ox = robot_x + d[outlier] * np.cos(a[outlier])
        oy = robot_y + d[outlier] * np.sin(a[outlier])

        return hx, hy, ox, oy

    def observation_vector(self, robot_x, robot_y, robot_theta) -> np.ndarray:
        """Flat distance array ready for RL obs stacking."""
        _, dists, _ = self.scan(robot_x, robot_y, robot_theta)
        return np.array(dists, dtype=np.float32)

    # ── Internal ray-cast ─────────────────────────────────────────────────────

    def _raycast(self, robot_x: float, robot_y: float,
                 global_angles: np.ndarray) -> np.ndarray:
        """
        Vectorised ray vs line-segment intersection.
        Returns ideal (noiseless) range array, max_range for misses.
        """
        N       = len(global_angles)
        MISS    = self.max_range
        cos_a   = np.cos(global_angles)
        sin_a   = np.sin(global_angles)
        dists   = np.full(N, MISS)

        for (x1, y1, x2, y2) in self.walls:
            sx, sy   = x2 - x1, y2 - y1
            qpx, qpy = x1 - robot_x, y1 - robot_y

            denom = cos_a * sy - sin_a * sx
            valid = np.abs(denom) > 1e-9

            t = np.where(valid, (qpx * sy - qpy * sx) / np.where(valid, denom, 1.0), np.inf)
            u = np.where(valid, (qpx * sin_a - qpy * cos_a) / np.where(valid, denom, 1.0), -1.0)

            hit = valid & (t > 1e-6) & (u >= 0.0) & (u <= 1.0) & (t < dists)
            dists = np.where(hit, t, dists)

        return dists