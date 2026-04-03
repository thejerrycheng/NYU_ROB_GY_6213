"""
lidar_sensor.py
===============
Simple, unbiased 2-D ray-casting LiDAR with Gaussian noise.

Design principle: keep it simple and correct.
  - Pure Gaussian additive range noise — zero mean, no clipping bias
  - Angular jitter per beam
  - Occasional dropouts (beam returns max_range)
  - Short outliers (spurious near returns, glass / dust)
  - No incidence-angle scaling (was the main source of asymmetric bias)
  - No mixed-pixel effect (adds complexity without physical justification)
  - No motion blur

All parameters from config/lidar_config.yaml.

scan()          -> (angles, dists, flags)  noisy measurement
scan_noiseless()-> (angles, dists)         ideal ray-cast  (for EKF h(x))

flags: 0=valid  1=short_outlier  2=dropout
"""

import math
import numpy as np
import yaml
from pathlib import Path

_DEFAULT_CFG = Path(__file__).parent.parent / "config" / "lidar_config.yaml"


def load_lidar_config(path=None) -> dict:
    p = Path(path) if path else _DEFAULT_CFG
    with open(p) as f:
        return yaml.safe_load(f)


class LidarSensor:
    """
    2-D LiDAR with simple Gaussian noise model.

    Parameters
    ----------
    walls  : list of (x1,y1,x2,y2) wall segments (world frame)
    config : lidar_config.yaml path, pre-loaded dict, or None (uses default)
    noise  : master on/off — False gives a perfect noiseless scan
    """

    def __init__(self, walls: list, config=None, noise: bool = True):
        self.walls = walls
        self.noise = noise
        self.cfg   = config if isinstance(config, dict) else load_lidar_config(config)
        self._load()

    def _load(self):
        g   = self.cfg["geometry"]
        gn  = self.cfg["gaussian_noise"]
        an  = self.cfg["angular_noise"]
        out = self.cfg["outliers"]
        dr  = self.cfg["dropout"]

        self.num_rays    = int(g["num_rays"])
        self.max_range   = float(g["max_range"])
        self.min_range   = float(g["min_range"])
        self.start_angle = math.radians(float(g["start_angle_deg"]))
        self.fov         = math.radians(float(g["fov_deg"]))

        self.sigma_r   = float(gn["sigma_range"])
        self.gn_en     = bool(gn["enabled"])
        self.range_dep = bool(gn["range_dependent"])
        self.range_k   = float(gn["range_noise_k"])

        self.sigma_a   = math.radians(float(an["sigma_angle_deg"]))
        self.an_en     = bool(an["enabled"])

        self.out_en      = bool(out["enabled"])
        self.short_prob  = float(out["short_prob"])
        self.short_min_f = float(out["short_min_frac"])

        self.dr_en   = bool(dr["enabled"])
        self.dr_prob = float(dr["prob"])

        self._nom = np.linspace(self.start_angle,
                                self.start_angle + self.fov,
                                self.num_rays, endpoint=False)

    def reload_config(self, path=None):
        self.cfg = load_lidar_config(path)
        self._load()

    # ── Public ────────────────────────────────────────────────────────────────

    def scan(self, rx: float, ry: float, rth: float):
        """
        Fire all beams from (rx, ry, rth).

        Returns
        -------
        angles : (N,)  relative beam angles [rad]
        dists  : (N,)  measured ranges [m]
        flags  : (N,)  uint8  0=valid  1=short_outlier  2=dropout
        """
        N = self.num_rays

        # 1. Angular jitter
        if self.noise and self.an_en:
            angles = self._nom + np.random.normal(0.0, self.sigma_a, N)
        else:
            angles = self._nom.copy()

        # 2. Ideal ray-cast (noiseless geometry)
        d_ideal = self._cast(rx, ry, rth + angles)
        d       = d_ideal.copy()
        flags   = np.zeros(N, dtype=np.uint8)

        if not self.noise:
            return angles, np.clip(d, self.min_range, self.max_range), flags

        hit = d_ideal < self.max_range  # beams that struck a wall

        # 3. Gaussian range noise — ONLY on hit beams, no clipping bias
        #    We add noise before any outlier logic so the distribution is
        #    symmetric (zero-mean) around the true range.
        if self.gn_en and hit.any():
            if self.range_dep:
                sigma = self.sigma_r * (1.0 + self.range_k * d_ideal)
            else:
                sigma = np.full(N, self.sigma_r)
            noise = np.random.normal(0.0, sigma)
            # Apply noise, then clamp to physical range
            # Key: clamp AFTER noise, not before — this keeps the distribution
            # symmetric for typical ranges (far from min/max_range boundaries)
            d = np.where(hit, d_ideal + noise, d)
            d = np.clip(d, self.min_range, self.max_range)

        # 4. Dropouts — lost returns -> max_range
        if self.dr_en:
            drop = hit & (np.random.random(N) < self.dr_prob)
            d[drop]     = self.max_range
            flags[drop] = 2

        # 5. Short outliers — spurious near returns
        if self.out_en:
            # Only place short returns on beams that had a real hit AND no dropout
            cands = hit & (flags == 0) & (np.random.random(N) < self.short_prob)
            if cands.any():
                lo = np.maximum(d_ideal * self.short_min_f, self.min_range)
                hi = d_ideal - 1e-3
                ok = hi > lo
                r  = np.random.uniform(0, 1, N) * np.where(ok, hi - lo, 0) + lo
                d    = np.where(cands & ok, r, d)
                flags = np.where(cands & ok, 1, flags).astype(np.uint8)

        return angles, d, flags

    def scan_noiseless(self, rx: float, ry: float, rth: float):
        """Pure geometry ray-cast. Returns (angles, dists). For EKF h(x)."""
        d = self._cast(rx, ry, rth + self._nom)
        return self._nom.copy(), np.clip(d, self.min_range, self.max_range)

    # ── Internal ray-cast ─────────────────────────────────────────────────────

    def _cast(self, rx: float, ry: float, global_angles: np.ndarray) -> np.ndarray:
        """Vectorised ray vs line-segment intersection. Returns ranges."""
        N   = len(global_angles)
        ca  = np.cos(global_angles)
        sa  = np.sin(global_angles)
        out = np.full(N, self.max_range)

        for (x1, y1, x2, y2) in self.walls:
            sx, sy   = x2 - x1, y2 - y1
            qpx, qpy = x1 - rx, y1 - ry
            denom    = ca * sy - sa * sx
            valid    = np.abs(denom) > 1e-9
            t = np.where(valid, (qpx*sy - qpy*sx) / np.where(valid, denom, 1.), np.inf)
            u = np.where(valid, (qpx*sa - qpy*ca) / np.where(valid, denom, 1.), -1.)
            hit = valid & (t > 1e-6) & (u >= 0.) & (u <= 1.) & (t < out)
            out = np.where(hit, t, out)

        return out