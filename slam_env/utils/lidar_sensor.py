"""
LidarSensor
===========
2-D ray-casting lidar sensor, ported from your original simulate_lidar_scan().

• Works purely in Python — no MuJoCo ray-cast API needed.
  (We cast rays against the map's wall list at Python speed; fast enough
  for 360 rays @ 10 Hz on any modern CPU.)
• Gaussian range noise with the same variance as your calibrated model.
• Returns angles + distances exactly as your original function did, so
  your existing visualiser / SLAM back-end code is fully compatible.
"""

import math
import random
import numpy as np

# Calibrated lidar noise variance (σ²_z) from your model
VAR_LIDAR = 0.000363


class LidarSensor:
    """
    Ray-casting 2-D lidar.

    Parameters
    ----------
    walls       : list of (x1,y1,x2,y2) wall segments
    num_rays    : number of beams (default 360 → 1° resolution)
    max_range   : max sensing distance in metres (default 5.0)
    noise       : whether to add Gaussian range noise
    var_lidar   : range noise variance (defaults to your calibrated value)
    fov         : field of view in radians (default 2π = full 360°)
    start_angle : first ray angle relative to robot heading (default 0)
    """

    def __init__(self,
                 walls:       list,
                 num_rays:    int   = 360,
                 max_range:   float = 5.0,
                 noise:       bool  = True,
                 var_lidar:   float = VAR_LIDAR,
                 fov:         float = 2 * math.pi,
                 start_angle: float = 0.0):

        self.walls       = walls
        self.num_rays    = num_rays
        self.max_range   = max_range
        self.noise       = noise
        self.sigma_z     = math.sqrt(var_lidar)
        self.fov         = fov
        self.start_angle = start_angle

        # Pre-compute relative ray angles (constant for this sensor config)
        self._rel_angles = np.linspace(start_angle,
                                       start_angle + fov,
                                       num_rays,
                                       endpoint=False)

    # ── Public ───────────────────────────────────────────────────────────────

    def scan(self, robot_x: float, robot_y: float, robot_theta: float):
        """
        Fire all rays and return (angles, distances).

        Parameters
        ----------
        robot_x, robot_y : robot position in world frame
        robot_theta      : robot heading in world frame [rad]

        Returns
        -------
        angles    : list of ray angles relative to robot heading [rad]
        distances : list of range measurements [m]  (≤ max_range)
        """
        global_angles = robot_theta + self._rel_angles
        distances = np.full(self.num_rays, self.max_range)

        # Pre-compute ray unit vectors (vectorised)
        cos_a = np.cos(global_angles)
        sin_a = np.sin(global_angles)

        for wall in self.walls:
            x1, y1, x2, y2 = wall
            sx = x2 - x1
            sy = y2 - y1

            # Segment → robot vectors
            qpx = x1 - robot_x
            qpy = y1 - robot_y

            # r × s  (denominator)
            r_cross_s = cos_a * sy - sin_a * sx   # (num_rays,)
            valid = np.abs(r_cross_s) > 1e-9

            t = np.where(valid,
                         (qpx * sy - qpy * sx) / np.where(valid, r_cross_s, 1),
                         np.inf)
            u = np.where(valid,
                         (qpx * sin_a - qpy * cos_a) / np.where(valid, r_cross_s, 1),
                         -1)

            hit = valid & (t > 1e-6) & (u >= 0) & (u <= 1) & (t < distances)
            distances = np.where(hit, t, distances)

        # Add Gaussian noise to all rays that actually hit something
        if self.noise:
            hit_mask = distances < self.max_range
            noise_arr = np.random.normal(0, self.sigma_z, self.num_rays)
            noisy = distances + noise_arr * hit_mask
            distances = np.maximum(0.0, noisy)

        return list(self._rel_angles), list(distances)

    def scan_as_points(self,
                       robot_x: float,
                       robot_y: float,
                       robot_theta: float):
        """
        Convenience: returns (hit_x, hit_y) arrays in the *world* frame.
        Useful for matplotlib scatter / MuJoCo debug markers.
        """
        angles, dists = self.scan(robot_x, robot_y, robot_theta)
        global_a = np.array(angles) + robot_theta
        d = np.array(dists)
        mask = d < self.max_range
        hx = robot_x + d[mask] * np.cos(global_a[mask])
        hy = robot_y + d[mask] * np.sin(global_a[mask])
        return hx, hy

    def observation_vector(self,
                           robot_x: float,
                           robot_y: float,
                           robot_theta: float) -> np.ndarray:
        """
        Returns distances as a flat numpy array — ready to stack into
        an RL observation vector.
        """
        _, dists = self.scan(robot_x, robot_y, robot_theta)
        return np.array(dists, dtype=np.float32)

    # ── Expected range (for EKF / UKF sensor model h(x)) ────────────────────

    def expected_range(self,
                       robot_x: float,
                       robot_y: float,
                       robot_theta: float,
                       ray_index: int) -> float:
        """
        Noiseless expected range for ray `ray_index`. Used by EKF h(x).
        """
        sensor = LidarSensor(self.walls, self.num_rays, self.max_range,
                             noise=False,
                             fov=self.fov,
                             start_angle=self.start_angle)
        _, dists = sensor.scan(robot_x, robot_y, robot_theta)
        return dists[ray_index]
