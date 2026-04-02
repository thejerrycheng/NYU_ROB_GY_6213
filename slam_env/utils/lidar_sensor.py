"""
lidar_sensor.py
===============
Realistic 2-D ray-casting LiDAR sensor.

All parameters loaded from config/lidar_config.yaml.

Noise pipeline (applied in order, all individually toggleable):
  1. Angular jitter      — beam direction error per scan (encoder/bearing noise)
  2. Gaussian range noise— additive, grows with distance (electronics + surface)
  3. Incidence scaling   — noise amplified at grazing angles (cos^-2 model)
  4. Dropout             — full beam loss → max_range (dark/transparent surfaces)
  5. Short outliers      — spurious near returns (dust, glass, cross-talk)
  6. Long outliers       — missed detections → max_range
  7. Mixed-pixel         — blended return at depth discontinuities (beam width)
  8. Motion blur         — range smearing from robot velocity (optional)
  9. Min-range clamp     — blind zone near sensor

scan() returns (angles, distances, flags):
  flags  0 = valid hit
         1 = short outlier
         2 = long outlier / dropout
         3 = mixed pixel
         4 = min-range clamp
"""

import math
import numpy as np
import yaml
from pathlib import Path

_DEFAULT_CFG = Path(__file__).parent.parent / "config" / "lidar_config.yaml"


def load_lidar_config(path=None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_CFG
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class LidarSensor:
    """
    Realistic 2-D LiDAR sensor with full noise + outlier model.

    Parameters
    ----------
    walls  : list of (x1,y1,x2,y2) wall segments in world frame
    config : path to lidar_config.yaml, or pre-loaded dict, or None (default)
    noise  : master switch — False gives a perfect noiseless ray-cast (EKF use)
    """

    def __init__(self, walls: list, config=None, noise: bool = True):
        self.walls = walls
        self.noise = noise
        self.cfg   = config if isinstance(config, dict) else load_lidar_config(config)
        self._apply_config()

    # ── Config ────────────────────────────────────────────────────────────────

    def _apply_config(self):
        g   = self.cfg["geometry"]
        gn  = self.cfg["gaussian_noise"]
        an  = self.cfg["angular_noise"]
        inc = self.cfg.get("incidence_noise", {})
        out = self.cfg["outliers"]
        mp  = self.cfg["mixed_pixel"]
        dr  = self.cfg["dropout"]
        mb  = self.cfg.get("motion_blur", {})

        self.num_rays    = int(g["num_rays"])
        self.fov         = math.radians(float(g["fov_deg"]))
        self.start_angle = math.radians(float(g["start_angle_deg"]))
        self.min_range   = float(g["min_range"])
        self.max_range   = float(g["max_range"])

        # Gaussian noise
        self.gn_en       = bool(gn["enabled"])
        self.sigma_r     = float(gn["sigma_range"])
        self.range_dep   = bool(gn["range_dependent"])
        self.range_k     = float(gn["range_noise_k"])

        # Angular jitter
        self.an_en       = bool(an["enabled"])
        self.sigma_a     = math.radians(float(an["sigma_angle_deg"]))

        # Incidence-angle scaling (grazing → more noise)
        self.inc_en      = bool(inc.get("enabled", False))
        self.inc_max     = float(inc.get("max_scale", 3.0))

        # Outliers
        self.out_en      = bool(out["enabled"])
        self.short_prob  = float(out["short_prob"])
        self.short_min_f = float(out["short_min_frac"])
        self.long_prob   = float(out["long_prob"])

        # Mixed pixel
        self.mp_en       = bool(mp["enabled"])
        self.mp_jump     = float(mp["jump_threshold"])
        self.mp_blend    = float(mp["blend_fraction"])

        # Dropout
        self.dr_en       = bool(dr["enabled"])
        self.dr_prob     = float(dr["prob"])

        # Motion blur
        self.mb_en       = bool(mb.get("enabled", False))
        self.mb_sigma    = float(mb.get("sigma_m", 0.005))

        # Pre-computed nominal angles
        self._nom_angles = np.linspace(
            self.start_angle,
            self.start_angle + self.fov,
            self.num_rays,
            endpoint=False)

    def reload_config(self, path=None):
        self.cfg = load_lidar_config(path)
        self._apply_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, robot_x: float, robot_y: float, robot_theta: float):
        """
        Fire all beams. Returns (angles, distances, flags).
        angles    : (N,) relative beam angles [rad]
        distances : (N,) measured ranges [m]
        flags     : (N,) uint8 — 0 valid, 1 short, 2 long/drop, 3 mixed, 4 minrange
        """
        N    = self.num_rays
        MISS = self.max_range

        # ── 1. Angular jitter ─────────────────────────────────────────────────
        if self.noise and self.an_en:
            rel_angles = self._nom_angles + np.random.normal(0.0, self.sigma_a, N)
        else:
            rel_angles = self._nom_angles.copy()

        global_angles = robot_theta + rel_angles

        # ── 2. Ideal ray-cast ─────────────────────────────────────────────────
        # Returns the true (noiseless) range AND the wall incidence angle cosine
        dists_ideal, cos_inc = self._raycast_with_incidence(
            robot_x, robot_y, global_angles)

        dists = dists_ideal.copy()
        flags = np.zeros(N, dtype=np.uint8)

        # Early exit for noiseless mode
        if not self.noise:
            dists = np.clip(dists, self.min_range, self.max_range)
            return list(rel_angles), list(dists), flags

        # Mask of beams that had a real geometric hit
        hit_ideal = dists_ideal < MISS

        # ── 3. Dropout (before Gaussian so lost beams don't get noise) ────────
        if self.dr_en:
            drop = hit_ideal & (np.random.random(N) < self.dr_prob)
            dists[drop] = MISS
            flags[drop] = 2

        # ── 4. Gaussian range noise ───────────────────────────────────────────
        if self.gn_en:
            # Only noise beams that still have a real hit (not dropped)
            hit_now = (flags == 0) & hit_ideal

            # Base sigma, scaled by distance
            if self.range_dep:
                sigma_eff = self.sigma_r * (1.0 + self.range_k * dists_ideal)
            else:
                sigma_eff = np.full(N, self.sigma_r)

            # Incidence scaling: grazing angles → larger noise
            # cos_inc ≈ 1 for perpendicular, ≈ 0 for grazing
            # scale = 1 / max(cos^2, 1/max_scale^2)  → bounded
            if self.inc_en:
                cos2 = np.clip(cos_inc ** 2, 1.0 / self.inc_max**2, 1.0)
                sigma_eff = sigma_eff / cos2
                sigma_eff = np.minimum(sigma_eff, self.sigma_r * self.inc_max)

            noise_r = np.random.normal(0.0, sigma_eff, N)
            dists = np.where(hit_now, dists + noise_r, dists)

        # ── 5. Motion blur ────────────────────────────────────────────────────
        if self.mb_en:
            hit_now = (flags == 0) & hit_ideal
            blur = np.random.normal(0.0, self.mb_sigma, N)
            dists = np.where(hit_now, dists + blur, dists)

        # ── 6. Short outliers ─────────────────────────────────────────────────
        if self.out_en:
            short_mask = hit_ideal & (flags == 0) & \
                         (np.random.random(N) < self.short_prob)
            if short_mask.any():
                lo = np.maximum(dists_ideal * self.short_min_f, self.min_range)
                hi = dists_ideal - 1e-4   # must be strictly less than true range
                # guard: if lo >= hi just put it at min_range
                valid_range = hi > lo
                short_r = np.where(
                    valid_range,
                    np.random.uniform(0, 1, N) * (hi - lo) + lo,
                    self.min_range)
                dists[short_mask] = short_r[short_mask]
                flags[short_mask] = 1

        # ── 7. Long outliers ──────────────────────────────────────────────────
            long_mask = (flags == 0) & (np.random.random(N) < self.long_prob)
            dists[long_mask] = MISS
            flags[long_mask] = 2

        # ── 8. Mixed-pixel (beam-width effect at depth edges) ─────────────────
        if self.mp_en:
            # Use the ideal range to detect jumps cleanly (not noisy range)
            d_roll = np.roll(dists_ideal, 1)
            jump   = np.abs(dists_ideal - d_roll) > self.mp_jump
            cands  = jump & (flags == 0) & hit_ideal & \
                     (np.random.random(N) < self.mp_blend)
            if cands.any():
                # Blend between current and adjacent beam range
                k             = cands.sum()
                alpha         = np.random.uniform(0.0, 1.0, k)
                blended       = alpha * dists[cands] + (1.0 - alpha) * d_roll[cands]
                dists[cands]  = blended
                flags[cands]  = 3

        # ── 9. Min-range clamp ────────────────────────────────────────────────
        too_close = (dists < self.min_range) & (flags == 0)
        dists[too_close] = self.min_range
        flags[too_close] = 4

        # Final hard clamp to valid range
        dists = np.clip(dists, self.min_range, self.max_range)

        return list(rel_angles), list(dists), flags

    def scan_noiseless(self, robot_x: float, robot_y: float, robot_theta: float):
        """Pure noiseless ray-cast. Returns (angles, distances). For EKF h(x)."""
        ga = robot_theta + self._nom_angles
        d, _ = self._raycast_with_incidence(robot_x, robot_y, ga)
        d = np.clip(d, self.min_range, self.max_range)
        return list(self._nom_angles), list(d)

    def scan_as_points(self, robot_x, robot_y, robot_theta):
        """Returns (valid_x, valid_y, outlier_x, outlier_y) in world frame."""
        ra, d, f = self.scan(robot_x, robot_y, robot_theta)
        a  = np.asarray(ra) + robot_theta
        d  = np.asarray(d);  f = np.asarray(f)
        v  = (d < self.max_range) & (f == 0)
        o  = f != 0
        return (robot_x + d[v]*np.cos(a[v]),   robot_y + d[v]*np.sin(a[v]),
                robot_x + d[o]*np.cos(a[o]),   robot_y + d[o]*np.sin(a[o]))

    def observation_vector(self, robot_x, robot_y, robot_theta) -> np.ndarray:
        """Flat distance array for RL observation stacking."""
        _, d, _ = self.scan(robot_x, robot_y, robot_theta)
        return np.asarray(d, dtype=np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raycast_with_incidence(self, robot_x, robot_y, global_angles):
        """
        Vectorised ray vs line-segment intersection.
        Returns (dists, cos_incidence):
          dists         — ideal range per beam (max_range for miss)
          cos_incidence — cosine of incidence angle at hit point
                          (1.0 = perpendicular, ~0 = grazing, 1.0 for miss)
        """
        N     = len(global_angles)
        MISS  = self.max_range
        ca    = np.cos(global_angles)
        sa    = np.sin(global_angles)
        dists = np.full(N, MISS)

        # Wall normal direction for each hit (used for incidence angle)
        # Store as (nx, ny) for best hit
        hit_nx = np.zeros(N)
        hit_ny = np.ones(N)   # default: perpendicular

        for (x1, y1, x2, y2) in self.walls:
            sx, sy   = x2 - x1, y2 - y1
            seg_len  = math.hypot(sx, sy)
            if seg_len < 1e-9:
                continue
            # Wall outward normal (left-hand side)
            nx_w = -sy / seg_len
            ny_w =  sx / seg_len

            qpx = x1 - robot_x
            qpy = y1 - robot_y

            denom = ca * sy - sa * sx
            valid = np.abs(denom) > 1e-9

            t = np.where(valid, (qpx*sy - qpy*sx) / np.where(valid, denom, 1.0), np.inf)
            u = np.where(valid, (qpx*sa - qpy*ca) / np.where(valid, denom, 1.0), -1.0)

            better = valid & (t > 1e-6) & (u >= 0.0) & (u <= 1.0) & (t < dists)
            dists  = np.where(better, t, dists)
            hit_nx = np.where(better, nx_w, hit_nx)
            hit_ny = np.where(better, ny_w, hit_ny)

        # Cosine of incidence angle = |ray_dir · wall_normal|
        # ray direction = (ca, sa), wall normal = (hit_nx, hit_ny)
        cos_inc = np.abs(ca * hit_nx + sa * hit_ny)
        # For miss beams, set cos_inc = 1 (no amplification)
        cos_inc = np.where(dists < MISS, cos_inc, 1.0)

        return dists, cos_inc