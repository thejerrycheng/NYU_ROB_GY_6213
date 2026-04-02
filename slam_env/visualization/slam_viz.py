"""
slam_viz.py
===========
Real-time matplotlib visualization for EKF-SLAM.

Shows four panels updated live as the robot is teleoperated:
  [0,0] Lidar scan     — raw beams, hit points, outliers, robot pose
  [0,1] EKF estimate   — estimated trajectory, uncertainty ellipse, true pose
  [1,0] Occupancy grid — probability map with confidence overlay
  [1,1] Grid variance  — per-cell Bernoulli variance (uncertainty map)

Usage (standalone test):
    python visualization/slam_viz.py

Typical usage (from teleop):
    from slam_env.visualization.slam_viz import SlamViz
    viz = SlamViz(map_dict)
    viz.update(robot_state, ekf, lidar_angles, lidar_dists, lidar_flags)
"""

import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to Qt5Agg / Agg if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Custom colourmap: white (free) → grey (unknown) → black (occupied)
_OCC_CMAP = LinearSegmentedColormap.from_list(
    "occ", ["white", "#dddddd", "#444444"])
# Variance colourmap: blue (low variance = confident) → red (high = uncertain)
_VAR_CMAP = LinearSegmentedColormap.from_list(
    "var", ["#1a6bab", "#f0f0f0", "#c0392b"])


class SlamViz:
    """
    Live SLAM visualization window.

    Parameters
    ----------
    map_dict   : map definition dict (for bounds and wall drawing)
    title      : window title
    figsize    : matplotlib figure size
    max_trail  : maximum number of trajectory points to keep
    """

    def __init__(self,
                 map_dict:  dict,
                 title:     str   = "EKF SLAM Visualization",
                 figsize:   tuple = (14, 10),
                 max_trail: int   = 500):

        self.map_dict  = map_dict
        self.max_trail = max_trail
        self.bounds    = map_dict["bounds"]   # (xmin, xmax, ymin, ymax)
        self.walls     = map_dict["walls"]

        # History buffers
        self._true_trail  = []   # [(x,y), ...]  ground-truth
        self._est_trail   = []   # [(x,y), ...]  EKF estimate
        self._step        = 0

        self._fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
        self._fig.canvas.manager.set_window_title(title)
        gs = GridSpec(2, 2, figure=self._fig,
                      hspace=0.38, wspace=0.32,
                      left=0.07, right=0.97, top=0.93, bottom=0.06)

        style = dict(facecolor="#16213e", aspect="equal")
        self._ax_lidar = self._fig.add_subplot(gs[0, 0], **style)
        self._ax_ekf   = self._fig.add_subplot(gs[0, 1], **style)
        self._ax_occ   = self._fig.add_subplot(gs[1, 0])
        self._ax_var   = self._fig.add_subplot(gs[1, 1])

        self._fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
        plt.ion()
        plt.show(block=False)

        # Image handles (for fast imshow updates)
        self._occ_im = None
        self._var_im = None

    # ── Main update call ──────────────────────────────────────────────────────

    def update(self,
               robot_x:  float, robot_y:  float, robot_theta: float,
               ekf,
               rel_angles: list, dists: list, flags,
               step: int = None):
        """
        Update all four panels. Call every simulation step.

        Parameters
        ----------
        robot_x/y/theta  : true robot pose (from MuJoCo)
        ekf              : EKFSlam instance
        rel_angles/dists : lidar scan
        flags            : lidar beam flags (0=valid,1=short,2=long,3=mixed,4=min)
        step             : current simulation step (for title)
        """
        self._step = step or self._step + 1

        # Append trails
        self._true_trail.append((robot_x, robot_y))
        ex, ey, _ = ekf.pose
        self._est_trail.append((ex, ey))
        if len(self._true_trail) > self.max_trail:
            self._true_trail = self._true_trail[-self.max_trail:]
            self._est_trail  = self._est_trail[-self.max_trail:]

        self._draw_lidar(robot_x, robot_y, robot_theta,
                         rel_angles, dists, flags, ekf)
        self._draw_ekf(robot_x, robot_y, robot_theta, ekf)
        self._draw_occ(ekf)
        self._draw_var(ekf)

        self._fig.suptitle(
            f"EKF SLAM  —  step {self._step}  |  "
            f"pose est: ({ex:.3f}, {ey:.3f})  "
            f"σ_x={ekf.pose_std[0]*100:.1f}cm  "
            f"σ_θ={math.degrees(ekf.pose_std[2]):.1f}°",
            color="white", fontsize=11, fontweight="bold")

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ── Panel 0: Lidar scan ───────────────────────────────────────────────────

    def _draw_lidar(self, rx, ry, rtheta,
                    rel_angles, dists, flags, ekf):
        ax = self._ax_lidar
        ax.clear()
        ax.set_facecolor("#16213e")

        xmin,xmax,ymin,ymax = self.bounds
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", "box")
        ax.set_title("Lidar Scan", color="white", fontsize=9, pad=3)
        ax.tick_params(colors="grey", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#334")

        # Map walls
        for (x1,y1,x2,y2) in self.walls:
            ax.plot([x1,x2],[y1,y2], color="#7f8c9a", linewidth=2.0, zorder=1)

        ra = np.asarray(rel_angles); d = np.asarray(dists); f = np.asarray(flags)
        max_r = ekf.max_range

        # Beams and hit points
        for i in range(len(ra)):
            ga = rtheta + ra[i]
            ex = rx + d[i]*math.cos(ga)
            ey = ry + d[i]*math.sin(ga)
            is_hit = d[i] < max_r * 0.99
            flag   = int(f[i])

            if is_hit:
                color = ("#e74c3c" if flag == 1 else
                         "#e67e22" if flag == 2 else
                         "#9b59b6" if flag == 3 else
                         "#2ecc71")
                ax.plot([rx,ex],[ry,ey], color="#1abc9c", alpha=0.18,
                        linewidth=0.5, zorder=2)
                ax.plot(ex, ey, ".", color=color, markersize=2.5, zorder=3)

        # Robot body
        ax.plot(rx, ry, "o", color="#f1c40f", markersize=8, zorder=5)
        ax.arrow(rx, ry, 0.18*math.cos(rtheta), 0.18*math.sin(rtheta),
                 head_width=0.06, head_length=0.06,
                 fc="#f1c40f", ec="#f1c40f", zorder=6)

        # Legend
        leg = [mpatches.Patch(color="#2ecc71", label="valid"),
               mpatches.Patch(color="#e74c3c", label="short"),
               mpatches.Patch(color="#e67e22", label="long/drop"),
               mpatches.Patch(color="#9b59b6", label="mixed")]
        ax.legend(handles=leg, loc="upper right", fontsize=6,
                  facecolor="#16213e", edgecolor="#334", labelcolor="white")

    # ── Panel 1: EKF estimate ─────────────────────────────────────────────────

    def _draw_ekf(self, rx, ry, rtheta, ekf):
        ax = self._ax_ekf
        ax.clear()
        ax.set_facecolor("#16213e")

        xmin,xmax,ymin,ymax = self.bounds
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", "box")
        ax.set_title("EKF Pose Estimate", color="white", fontsize=9, pad=3)
        ax.tick_params(colors="grey", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#334")

        # Map walls
        for (x1,y1,x2,y2) in self.walls:
            ax.plot([x1,x2],[y1,y2], color="#7f8c9a", linewidth=2.0)

        # True trajectory
        if len(self._true_trail) > 1:
            tx = [p[0] for p in self._true_trail]
            ty = [p[1] for p in self._true_trail]
            ax.plot(tx, ty, color="#3498db", alpha=0.6,
                    linewidth=1.2, label="True path")

        # EKF trajectory
        if len(self._est_trail) > 1:
            ex2 = [p[0] for p in self._est_trail]
            ey2 = [p[1] for p in self._est_trail]
            ax.plot(ex2, ey2, color="#e67e22", alpha=0.8,
                    linewidth=1.2, linestyle="--", label="EKF path")

        # Uncertainty ellipse (2-sigma) from P[0:2,0:2]
        ex, ey, eth = ekf.pose
        P2 = ekf.P[:2, :2]
        self._draw_ellipse(ax, ex, ey, P2, n_std=2.0,
                           color="#e67e22", alpha=0.25)

        # True robot
        ax.plot(rx, ry, "s", color="#3498db", markersize=7, zorder=5,
                label="True pose")
        ax.arrow(rx, ry, 0.15*math.cos(rtheta), 0.15*math.sin(rtheta),
                 head_width=0.05, fc="#3498db", ec="#3498db", zorder=6)

        # EKF robot
        ax.plot(ex, ey, "D", color="#e67e22", markersize=7, zorder=5,
                label="EKF pose")
        ax.arrow(ex, ey, 0.15*math.cos(eth), 0.15*math.sin(eth),
                 head_width=0.05, fc="#e67e22", ec="#e67e22", zorder=6)

        # Error
        err = math.hypot(rx-ex, ry-ey)
        ax.text(0.02, 0.97, f"pos err={err*100:.1f}cm\n"
                f"σ_x={ekf.pose_std[0]*100:.1f}cm  "
                f"σ_θ={math.degrees(ekf.pose_std[2]):.1f}°",
                transform=ax.transAxes, va="top", fontsize=7,
                color="#ecf0f1", bbox=dict(fc="#16213e", ec="none", alpha=0.7))

        ax.legend(loc="lower right", fontsize=6,
                  facecolor="#16213e", edgecolor="#334", labelcolor="white")

    @staticmethod
    def _draw_ellipse(ax, cx, cy, P2, n_std=2.0, color="orange", alpha=0.3):
        """Draw 2-D covariance ellipse from 2×2 submatrix."""
        try:
            vals, vecs = np.linalg.eigh(P2)
            vals = np.maximum(vals, 0)
            w = 2 * n_std * math.sqrt(vals[0])
            h = 2 * n_std * math.sqrt(vals[1])
            angle = math.degrees(math.atan2(vecs[1,0], vecs[0,0]))
            ell = Ellipse(xy=(cx,cy), width=w, height=h, angle=angle,
                          edgecolor=color, fc=color, alpha=alpha, lw=1.5)
            ax.add_patch(ell)
        except Exception:
            pass

    # ── Panel 2: Occupancy grid ───────────────────────────────────────────────

    def _draw_occ(self, ekf):
        ax = self._ax_occ
        g  = ekf.grid

        prob = g.probability.T   # transpose so x=col, y=row displays correctly
        conf = g.confidence.T

        extent = [g.x_min, g.x_max, g.y_min, g.y_max]

        if self._occ_im is None:
            self._occ_im = ax.imshow(
                prob, origin="lower", extent=extent,
                cmap=_OCC_CMAP, vmin=0.0, vmax=1.0,
                interpolation="nearest", aspect="equal")
            plt.colorbar(self._occ_im, ax=ax, shrink=0.8, label="P(occ)")
        else:
            self._occ_im.set_data(prob)

        ax.set_title("Occupancy Grid  (prob)", fontsize=9, pad=3)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=7)

        # EKF robot marker
        ex, ey, eth = ekf.pose
        ax.plot(ex, ey, "D", color="#e67e22", markersize=6, zorder=5)

    # ── Panel 3: Variance map ─────────────────────────────────────────────────

    def _draw_var(self, ekf):
        ax = self._ax_var
        g  = ekf.grid

        var = g.variance.T
        extent = [g.x_min, g.x_max, g.y_min, g.y_max]

        if self._var_im is None:
            self._var_im = ax.imshow(
                var, origin="lower", extent=extent,
                cmap=_VAR_CMAP, vmin=0.0, vmax=0.25,
                interpolation="nearest", aspect="equal")
            plt.colorbar(self._var_im, ax=ax, shrink=0.8,
                         label="Variance p(1−p)")
        else:
            self._var_im.set_data(var)

        ax.set_title("Map Variance  (uncertainty)", fontsize=9, pad=3)
        ax.set_xlabel("x [m]", fontsize=7)
        ax.set_ylabel("y [m]", fontsize=7)
        ax.tick_params(labelsize=7)

        ex, ey, eth = ekf.pose
        ax.plot(ex, ey, "D", color="#f1c40f", markersize=6, zorder=5)

    def close(self):
        plt.ioff()
        plt.close(self._fig)
