# External libraries
import os
import asyncio
import cv2
import math
import random
import socket
from matplotlib import pyplot as plt
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import strftime

# Local libraries
import robot_python_code
import parameters

# =======================================================
# Configuration & Kinematic Constants
# =======================================================
stream_video = True
DATA_DIR    = "online_dataset"
PD_DATA_DIR = "pd_control_dataset"
SL_DATA_DIR = "sl_control_dataset"
for d in [DATA_DIR, PD_DATA_DIR, SL_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

INITIAL_POSE = [0.1, 0.2, math.pi / 2]

L            = 0.145
V_M          = 0.004808
V_C          = -0.045557
VAR_V        = 0.057829
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA    = 0.023134

MAX_RANGE = 5.0
X_OFFSET  = 0.12
VAR_Z     = 0.0025

# --- THEME ---
CARD_BG    = 'bg-slate-900'
TEXT_COLOR = 'text-slate-200'
HEADER_BG  = 'bg-slate-950'

def angle_wrap(angle):
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


# =======================================================
# Motion Models
# =======================================================
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        if v_cmd == 0.0:
            v_expected = 0.0
            w_expected = 0.0
        else:
            v_expected = (V_M * v_cmd) + V_C
            if v_expected < 0: v_expected = 0.0
            alpha = steering_angle_command
            delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]
            w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0
        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2]  = angle_wrap(self.state[2] - delta_t * w_expected)


class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.weight = 1.0
        self.log_w  = 0.0

    def predict(self, v_cmd, alpha_cmd, dt):
        if v_cmd == 0.0:
            v_s = 0.0; w_s = 0.0
        else:
            v_expected     = (V_M * v_cmd) + V_C
            if v_expected < 0: v_expected = 0.0
            delta_expected = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
            v_s = v_expected + random.gauss(0, math.sqrt(VAR_V))
            if v_s < 0: v_s = 0.0
            d_s = delta_expected + random.gauss(0, math.sqrt(VAR_DELTA))
            w_s = (v_s * math.tan(d_s)) / L if L > 0 else 0.0
        self.x     += v_s * math.cos(self.theta) * dt
        self.y     += v_s * math.sin(self.theta) * dt
        self.theta  = angle_wrap(self.theta - w_s * dt)

    def update_weight(self, angles, distances):
        log_w = 0.0; ray_step = 10
        xs = self.x + X_OFFSET * math.cos(self.theta)
        ys = self.y + X_OFFSET * math.sin(self.theta)
        for i in range(0, len(angles), ray_step):
            raw_dist = distances[i]
            if raw_dist < 100 or raw_dist >= 4900: continue
            dist_m       = raw_dist / 1000.0
            angle_rad    = -(angles[i] * math.pi / 180.0)
            global_angle = angle_wrap(self.theta + angle_rad)
            rx, ry       = math.cos(global_angle), math.sin(global_angle)
            min_dist     = MAX_RANGE
            for wall in parameters.wall_corner_list:
                qx, qy, bx, by = wall
                sx, sy = bx - qx, by - qy
                denom  = rx * sy - ry * sx
                if abs(denom) > 1e-6:
                    t = ((qx - xs)*sy - (qy - ys)*sx) / denom
                    u = ((qx - xs)*ry - (qy - ys)*rx) / denom
                    if 0 <= u <= 1 and 0 < t < min_dist: min_dist = t
            if min_dist < MAX_RANGE:
                error  = min_dist - dist_m
                log_w -= min((error**2) / (2 * VAR_Z), 10.0)
            else:
                log_w -= 10.0
        self.log_w = log_w


class ParticleFilter:
    def __init__(self, num_particles, initial_pose):
        self.num_particles = num_particles
        self.particles     = []
        all_walls          = np.array(parameters.wall_corner_list)
        self.x_min, self.x_max = np.min(all_walls[:,[0,2]]), np.max(all_walls[:,[0,2]])
        self.y_min, self.y_max = np.min(all_walls[:,[1,3]]), np.max(all_walls[:,[1,3]])
        self.global_initialization(initial_pose)

    def global_initialization(self, pose):
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.gauss(pose[0], 0.1); y = random.gauss(pose[1], 0.1)
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max): continue
            self.particles.append(Particle(x, y, angle_wrap(random.gauss(pose[2], 0.1))))

    def predict(self, v_cmd, alpha_cmd, dt):
        for p in self.particles: p.predict(v_cmd, alpha_cmd, dt)

    def correct(self, angles, distances):
        if len(angles) < 10: return
        for p in self.particles: p.update_weight(angles, distances)
        max_log_w = max(p.log_w for p in self.particles)
        for p in self.particles: p.weight = math.exp(p.log_w - max_log_w)
        self.resample()

    def resample(self):
        weights = [p.weight for p in self.particles]
        new_particles = []; index = random.randint(0, self.num_particles - 1)
        beta = 0.0; max_w = max(weights)
        for _ in range(self.num_particles):
            beta += random.uniform(0, 2.0 * max_w)
            while beta > weights[index]:
                beta -= weights[index]; index = (index + 1) % self.num_particles
            p = self.particles[index]
            new_particles.append(Particle(p.x, p.y, p.theta))
        self.particles = new_particles

    def get_estimate(self):
        n = self.num_particles
        return [sum(p.x for p in self.particles)/n,
                sum(p.y for p in self.particles)/n,
                math.atan2(sum(math.sin(p.theta) for p in self.particles),
                           sum(math.cos(p.theta) for p in self.particles))]


# =======================================================
# PD Position Controller  ─  dual-loop design
# =======================================================
#
#  STEERING LOOP  →  controls heading to follow the correct curvature/path
#    error  :  e_h  = atan2(goal_y−y, goal_x−x) − robot_heading
#    output :  u_s  = Kp_steer·e_h + Kd_steer·(de_h/dt)
#              clamped to [−20, +20]
#
#  SPEED LOOP     →  controls forward progress; stops at the target distance
#    error  :  e_d  = Euclidean distance to goal
#    output :  u_v  = Kp_speed · e_d
#              clamped to [MIN_SPEED_CMD, MAX_SPEED_CMD]
#              zeroed when e_d < GOAL_THRESHOLD  (dead-band → stop)
#
#  Phase logic:
#    |e_h| > ALIGN_THRESHOLD  →  rotate in place (speed = 0, steer active)
#    |e_h| ≤ ALIGN_THRESHOLD  →  both loops run simultaneously
# =======================================================
class PDPositionController:
    # Steering PD gains
    KP_STEER      = 8.0
    KD_STEER      = 1.2
    # Speed P gain
    KP_SPEED      = 60.0
    MAX_SPEED_CMD = 80.0
    MIN_SPEED_CMD = 18.0
    # Thresholds
    ALIGN_THRESHOLD = 0.15   # rad
    GOAL_THRESHOLD  = 0.05   # m

    def __init__(self):
        self.reset()

    def reset(self):
        self.active              = False
        self.goal_x              = 0.0
        self.goal_y              = 0.0
        self._prev_heading_error = 0.0
        # logging buffers
        self.log_t          = []
        self.log_x          = []
        self.log_y          = []
        self.log_ex         = []   # x error
        self.log_ey         = []   # y error
        self.log_dist       = []   # Euclidean distance error
        self.log_heading_e  = []   # heading error [rad]
        self.log_speed_cmd  = []
        self.log_steer_cmd  = []
        self._start_time    = None

    def set_goal(self, gx, gy):
        self.goal_x = gx; self.goal_y = gy
        self._prev_heading_error = 0.0
        self._start_time         = time.time()
        self.active              = True

    def compute(self, pose, dt):
        """Returns (speed_cmd, steer_cmd, reached).  Appends one log row."""
        if not self.active:
            return 0, 0, False

        ex   = self.goal_x - pose[0]
        ey   = self.goal_y - pose[1]
        dist = math.hypot(ex, ey)

        # ── SPEED LOOP ──────────────────────────────────────────────────
        if dist < self.GOAL_THRESHOLD:
            self._append_log(pose, ex, ey, dist, 0.0, 0.0, 0.0)
            self.active = False
            return 0, 0, True

        speed_cmd = float(np.clip(self.KP_SPEED * dist,
                                  self.MIN_SPEED_CMD, self.MAX_SPEED_CMD))

        # ── STEERING LOOP ───────────────────────────────────────────────
        desired_heading = math.atan2(ey, ex)
        heading_error   = angle_wrap(desired_heading - pose[2])

        d_heading = (heading_error - self._prev_heading_error) / dt if dt > 0 else 0.0
        self._prev_heading_error = heading_error

        steer_cmd = float(np.clip(
            self.KP_STEER * heading_error + self.KD_STEER * d_heading,
            -20.0, 20.0
        ))

        # Speed zeroed while aligning in place
        if abs(heading_error) > self.ALIGN_THRESHOLD:
            speed_cmd = 0.0

        self._append_log(pose, ex, ey, dist, heading_error, speed_cmd, steer_cmd)
        return speed_cmd, steer_cmd, False

    def _append_log(self, pose, ex, ey, dist, he, spd, steer):
        t = time.time() - self._start_time if self._start_time else 0.0
        self.log_t.append(t);         self.log_x.append(pose[0])
        self.log_y.append(pose[1]);   self.log_ex.append(ex)
        self.log_ey.append(ey);       self.log_dist.append(dist)
        self.log_heading_e.append(he); self.log_speed_cmd.append(spd)
        self.log_steer_cmd.append(steer)

    def save_all(self, base_path, traj_history, map_walls, video_writer=None):
        """Write CSV, error plots, command plots, trajectory plot; release video."""
        if not self.log_t:
            return

        # 1 ── CSV ────────────────────────────────────────────────────────
        with open(base_path + "_commands.csv", 'w') as f:
            f.write("Time_s,X,Y,Error_X,Error_Y,Dist_Error_m,"
                    "Heading_Error_rad,Speed_Cmd,Steer_Cmd\n")
            for i in range(len(self.log_t)):
                f.write(f"{self.log_t[i]:.4f},{self.log_x[i]:.4f},{self.log_y[i]:.4f},"
                        f"{self.log_ex[i]:.4f},{self.log_ey[i]:.4f},"
                        f"{self.log_dist[i]:.4f},{self.log_heading_e[i]:.4f},"
                        f"{self.log_speed_cmd[i]:.2f},{self.log_steer_cmd[i]:.2f}\n")

        # 2 ── Error plots (X, Y, Distance) ──────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor='#0f172a')
        fig.suptitle('PD Position Controller — Tracking Errors', color='white', fontsize=13)
        for ax, data, lbl, col in zip(
                axes,
                [self.log_ex, self.log_ey, self.log_dist],
                ['X Error (m)', 'Y Error (m)', 'Distance Error (m)'],
                ['#3b82f6', '#f97316', '#22c55e']):
            ax.set_facecolor('#0f172a')
            ax.plot(self.log_t, data, color=col, linewidth=1.5)
            ax.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
            ax.set_ylabel(lbl, color='#94a3b8', fontsize=9)
            ax.tick_params(colors='#94a3b8')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        axes[-1].set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
        plt.tight_layout()
        fig.savefig(base_path + "_error_plots.png", dpi=150,
                    bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig)

        # 3 ── Command plots (speed & steering) ──────────────────────────
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0f172a')
        fig2.suptitle('PD Position Controller — Control Commands', color='white', fontsize=13)
        for ax in (ax1, ax2):
            ax.set_facecolor('#0f172a'); ax.tick_params(colors='#94a3b8')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        ax1.plot(self.log_t, self.log_speed_cmd, color='#facc15', linewidth=1.5)
        ax1.set_ylabel('Speed Command (0-100)', color='#94a3b8', fontsize=9)
        ax1.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
        ax2.plot(self.log_t, self.log_steer_cmd, color='#ec4899', linewidth=1.5)
        ax2.set_ylabel('Steering Command (−20 to +20)', color='#94a3b8', fontsize=9)
        ax2.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
        plt.tight_layout()
        fig2.savefig(base_path + "_command_plots.png", dpi=150,
                     bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig2)

        # 4 ── Trajectory plot ────────────────────────────────────────────
        fig3, ax = plt.subplots(figsize=(6, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a'); ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values(): sp.set_color('#334155')
        ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        for wall in map_walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='white', linewidth=1.5)
        if traj_history['est_x']:
            ax.plot(traj_history['est_x'], traj_history['est_y'],
                    color='#3b82f6', linewidth=1.5, label='PF Trajectory')
        ax.plot(self.log_x, self.log_y,
                color='#facc15', linewidth=1.5, linestyle='--', label='PD Run Path')
        ax.plot(self.goal_x, self.goal_y, '*', color='#facc15',
                markersize=12, zorder=9, label='Goal')
        if self.log_x:
            ax.plot(self.log_x[0], self.log_y[0], 'o', color='#22c55e',
                    markersize=8, zorder=9, label='Start')
        ax.set_aspect('equal')
        ax.legend(facecolor='#0f172a', edgecolor='#334155',
                  labelcolor='#94a3b8', fontsize=8)
        ax.set_title('PD Controller Trajectory', color='white', fontsize=11)
        fig3.savefig(base_path + "_trajectory.png", dpi=150,
                     bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig3)

        # 5 ── Release video ──────────────────────────────────────────────
        if video_writer is not None:
            video_writer.release()


# =======================================================
# Straight-Line Controller  ─  revised robust dual-loop design
# =======================================================
#
#  ROOT CAUSES of the observed oscillations & growing heading error:
#   1. Pure heading feedback ignores lateral drift — cross-track error
#      must be fed back into the steering demand so the robot returns to
#      the reference line, not just re-aligns its heading.
#   2. Kp_steer was too high relative to Kd causing limit-cycle oscillation
#      at 100 ms sample rate with mechanical steering lag.
#   3. PF heading noise caused derivative spikes → steering saturation.
#
#  STEERING LOOP  →  combined heading + cross-track feedback (Stanley-like)
#    heading_error :  e_h = reference_heading − robot_heading
#    cross-track   :  e_c = lateral deviation from the reference line (m)
#    combined error:  e   = e_h + atan(Kc · e_c / (v_normaliser))
#    PD output     :  u_s = Kp_steer·e + Kd_steer·(de/dt)
#                     derivative is low-pass filtered to suppress PF noise
#                     output clamped to [−20, +20]
#
#  SPEED LOOP     →  proportional on remaining along-track distance
#    error  :  e_d  = target_distance − distance_travelled
#    output :  u_v  = Kp_speed · e_d,  clamped [MIN_SPEED, MAX_SPEED]
#              zeroed when e_d < GOAL_THRESHOLD
# =======================================================
class StraightLineController:
    KP_STEER      = 3.5    # lowered — was causing oscillation at 6.0
    KD_STEER      = 2.0    # raised  — more damping needed at 100ms rate
    KC_CROSS      = 1.5    # cross-track gain (Stanley term); tune 0.5–3.0
    V_NORMALISER  = 0.15   # approx forward speed (m/s) for Stanley scaling
    D_FILTER      = 0.4    # low-pass coefficient for derivative (0=no filter, 1=frozen)
    KP_SPEED      = 60.0
    MAX_SPEED_CMD = 80.0
    MIN_SPEED_CMD = 18.0
    GOAL_THRESHOLD = 0.04   # m

    def __init__(self):
        self.reset()

    def reset(self):
        self.active              = False
        self.start_x             = 0.0; self.start_y = 0.0
        self.target_dist         = 0.0
        self.reference_heading   = 0.0
        self._prev_combined_error = 0.0
        self._filtered_d_error    = 0.0   # low-pass filtered derivative
        # logging
        self.log_t          = []; self.log_x         = []; self.log_y        = []
        self.log_lat_err    = []; self.log_remaining  = []; self.log_heading_e = []
        self.log_speed_cmd  = []; self.log_steer_cmd  = []
        self._start_time    = None

    def set_goal(self, current_pose, distance):
        self.start_x = current_pose[0]; self.start_y = current_pose[1]
        self.reference_heading    = current_pose[2]
        self.target_dist          = distance
        self._prev_combined_error = 0.0
        self._filtered_d_error    = 0.0
        self._start_time          = time.time()
        self.active               = True

    def compute(self, pose, dt):
        if not self.active:
            return 0, 0, False

        dx        = pose[0] - self.start_x
        dy        = pose[1] - self.start_y
        c, s      = math.cos(self.reference_heading), math.sin(self.reference_heading)
        travelled = dx * c + dy * s          # along-track progress
        lateral   = -dx * s + dy * c         # cross-track deviation (+ve = left of line)
        remaining = self.target_dist - travelled

        # ── SPEED LOOP ──────────────────────────────────────────────────
        if remaining < self.GOAL_THRESHOLD:
            self._append_log(pose, lateral, 0.0, 0.0, 0.0, 0.0)
            self.active = False
            return 0, 0, True

        speed_cmd = float(np.clip(self.KP_SPEED * remaining,
                                  self.MIN_SPEED_CMD, self.MAX_SPEED_CMD))

        # ── STEERING LOOP  (heading + cross-track) ───────────────────────
        heading_error = angle_wrap(self.reference_heading - pose[2])

        # Stanley cross-track correction term — pulls robot back onto the line
        # atan saturates naturally so it won't saturate the steer output alone
        cross_track_correction = math.atan2(
            self.KC_CROSS * lateral,
            self.V_NORMALISER
        )
        combined_error = angle_wrap(heading_error + cross_track_correction)

        # Derivative with low-pass filter to suppress PF heading noise
        if dt > 0:
            raw_d = (combined_error - self._prev_combined_error) / dt
            self._filtered_d_error = (self.D_FILTER * self._filtered_d_error
                                      + (1.0 - self.D_FILTER) * raw_d)
        self._prev_combined_error = combined_error

        steer_cmd = float(np.clip(
            self.KP_STEER * combined_error + self.KD_STEER * self._filtered_d_error,
            -20.0, 20.0
        ))

        self._append_log(pose, lateral, remaining, combined_error, speed_cmd, steer_cmd)
        return speed_cmd, steer_cmd, False

    def _append_log(self, pose, lat, rem, he, spd, steer):
        t = time.time() - self._start_time if self._start_time else 0.0
        self.log_t.append(t);          self.log_x.append(pose[0])
        self.log_y.append(pose[1]);    self.log_lat_err.append(lat)
        self.log_remaining.append(rem); self.log_heading_e.append(he)
        self.log_speed_cmd.append(spd); self.log_steer_cmd.append(steer)

    def save_all(self, base_path, map_walls, video_writer=None):
        if not self.log_t:
            return

        # 1 ── CSV ────────────────────────────────────────────────────────
        with open(base_path + "_commands.csv", 'w') as f:
            f.write("Time_s,X,Y,Lateral_Error_m,Remaining_Dist_m,"
                    "Heading_Error_rad,Speed_Cmd,Steer_Cmd\n")
            for i in range(len(self.log_t)):
                f.write(f"{self.log_t[i]:.4f},{self.log_x[i]:.4f},{self.log_y[i]:.4f},"
                        f"{self.log_lat_err[i]:.4f},{self.log_remaining[i]:.4f},"
                        f"{self.log_heading_e[i]:.4f},"
                        f"{self.log_speed_cmd[i]:.2f},{self.log_steer_cmd[i]:.2f}\n")

        # 2 ── Error plots ────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor='#0f172a')
        fig.suptitle('Straight-Line Controller — Tracking Errors', color='white', fontsize=13)
        for ax, data, lbl, col in zip(
                axes,
                [self.log_lat_err, self.log_remaining, self.log_heading_e],
                ['Lateral Error (m)', 'Remaining Distance (m)', 'Heading Error (rad)'],
                ['#3b82f6', '#f97316', '#22c55e']):
            ax.set_facecolor('#0f172a')
            ax.plot(self.log_t, data, color=col, linewidth=1.5)
            ax.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
            ax.set_ylabel(lbl, color='#94a3b8', fontsize=9)
            ax.tick_params(colors='#94a3b8')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        axes[-1].set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
        plt.tight_layout()
        fig.savefig(base_path + "_error_plots.png", dpi=150,
                    bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig)

        # 3 ── Command plots ──────────────────────────────────────────────
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0f172a')
        fig2.suptitle('Straight-Line Controller — Control Commands', color='white', fontsize=13)
        for ax in (ax1, ax2):
            ax.set_facecolor('#0f172a'); ax.tick_params(colors='#94a3b8')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        ax1.plot(self.log_t, self.log_speed_cmd, color='#a78bfa', linewidth=1.5)
        ax1.set_ylabel('Speed Command (0-100)', color='#94a3b8', fontsize=9)
        ax1.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
        ax2.plot(self.log_t, self.log_steer_cmd, color='#ec4899', linewidth=1.5)
        ax2.set_ylabel('Steering Command (−20 to +20)', color='#94a3b8', fontsize=9)
        ax2.axhline(0, color='#475569', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Time (s)', color='#94a3b8', fontsize=9)
        plt.tight_layout()
        fig2.savefig(base_path + "_command_plots.png", dpi=150,
                     bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig2)

        # 4 ── Trajectory plot ────────────────────────────────────────────
        fig3, ax = plt.subplots(figsize=(6, 6), facecolor='#0f172a')
        ax.set_facecolor('#0f172a'); ax.tick_params(colors='#94a3b8')
        for sp in ax.spines.values(): sp.set_color('#334155')
        ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
        for wall in map_walls:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='white', linewidth=1.5)
        ax.plot(self.log_x, self.log_y,
                color='#a78bfa', linewidth=1.5, label='SL Run Path')
        ex = self.start_x + self.target_dist * math.cos(self.reference_heading)
        ey = self.start_y + self.target_dist * math.sin(self.reference_heading)
        ax.plot([self.start_x, ex], [self.start_y, ey],
                color='white', linewidth=1.0, linestyle='--', label='Reference Line')
        ax.plot(ex, ey, 'D', color='#a78bfa', markersize=8, zorder=9, label='Goal')
        if self.log_x:
            ax.plot(self.log_x[0], self.log_y[0], 'o', color='#22c55e',
                    markersize=8, zorder=9, label='Start')
        ax.set_aspect('equal')
        ax.legend(facecolor='#0f172a', edgecolor='#334155',
                  labelcolor='#94a3b8', fontsize=8)
        ax.set_title('Straight-Line Controller Trajectory', color='white', fontsize=11)
        fig3.savefig(base_path + "_trajectory.png", dpi=150,
                     bbox_inches='tight', facecolor='#0f172a')
        plt.close(fig3)

        if video_writer is not None:
            video_writer.release()


# =======================================================
# Helpers
# =======================================================
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

def get_time_in_ms():
    return int(time.time() * 1000)


# =======================================================
# Main GUI Application
# =======================================================
@ui.page('/')
def main():
    dark = ui.dark_mode(); dark.value = True
    ui.add_head_html('''
        <style>
            .nicegui-content { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
            .q-card { border-radius: 16px; border: 1px solid #334155; }
            .q-slider__track-container { height: 6px; border-radius: 3px; }
            .controller-active { border: 1px solid #3b82f6 !important;
                                  box-shadow: 0 0 12px rgba(59,130,246,0.25); }
            .controller-done   { border: 1px solid #22c55e !important;
                                  box-shadow: 0 0 12px rgba(34,197,94,0.25); }
        </style>
    ''')

    # ── Application State ─────────────────────────────────────────────────
    state = {
        'connected':        False,
        'udp':              None, 'sender': None, 'receiver': None,
        'sensor_signal':    robot_python_code.RobotSensorSignal([0, 0, 0]),
        'running_trial':    False, 'trial_start_time': 0,
        'base_filename':    "", 'csv_file': None, 'video_writer': None,
        'pf_last_time':     time.time(),
        'sweep_angles':     [], 'sweep_distances': [], 'last_lidar_angle': None,
        'latest_frame':     None,
        'ctrl_mode':        None,   # None | 'pd_position' | 'straight_line'
        'pd_video_writer':  None,
        'sl_video_writer':  None,
        '_pd_base':         "",
        '_sl_base':         "",
    }

    pf = ParticleFilter(num_particles=800, initial_pose=INITIAL_POSE)
    dr = MyMotionModel(initial_state=INITIAL_POSE)

    all_walls          = np.array(parameters.wall_corner_list)
    x_min, x_max       = np.min(all_walls[:,[0,2]]), np.max(all_walls[:,[0,2]])
    y_min, y_max       = np.min(all_walls[:,[1,3]]), np.max(all_walls[:,[1,3]])
    x_pad, y_pad       = (x_max - x_min)*0.1, (y_max - y_min)*0.1

    history = {'est_x': [], 'est_y': [], 'dr_x': [], 'dr_y': []}

    pd_ctrl = PDPositionController()
    sl_ctrl = StraightLineController()

    if stream_video:
        try:
            video_capture = cv2.VideoCapture(parameters.camera_id)
        except:
            video_capture = cv2.VideoCapture(0)

    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not stream_video or not video_capture.isOpened():
            return Response(content=b'', media_type='image/jpeg')
        _, frame = await run.io_bound(video_capture.read)
        if frame is None: return Response(content=b'', media_type='image/jpeg')
        return Response(content=await run.cpu_bound(convert, frame), media_type='image/jpeg')

    def _open_video_writer(path: str):
        if not stream_video or not video_capture.isOpened(): return None
        w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (w, h))

    # ── Connection ────────────────────────────────────────────────────────
    def update_connection_to_robot():
        if udp_switch.value and not state['connected']:
            udp, success = robot_python_code.create_udp_communication(
                parameters.arduinoIP, parameters.localIP,
                parameters.arduinoPort, parameters.localPort, parameters.bufferSize)
            if success:
                udp.UDPServerSocket.settimeout(0.05)
                state.update(udp=udp, connected=True,
                             sender=robot_python_code.MsgSender(time.perf_counter(), 2, udp),
                             receiver=robot_python_code.MsgReceiver(time.perf_counter(), 3, udp))
                status_indicator.classes('bg-green-500', remove='bg-red-500')
                status_label.set_text('Connected')
            else:
                udp_switch.value = False
        elif not udp_switch.value and state['connected']:
            if state['sender']: state['sender'].send_control_signal([0, 0])
            state['connected'] = False; pd_ctrl.reset(); sl_ctrl.reset()
            state['ctrl_mode'] = None
            status_indicator.classes('bg-red-500', remove='bg-green-500')
            status_label.set_text('Disconnected')

    # ── Manual trial ──────────────────────────────────────────────────────
    def run_trial():
        if not state['connected']:
            ui.notify('Must connect to hardware first!', type='warning'); return
        state.update(trial_start_time=get_time_in_ms(), running_trial=True)
        steering_switch.value = speed_switch.value = True
        for k in ('est_x','est_y','dr_x','dr_y'): history[k].clear()
        state['sweep_angles'].clear(); state['sweep_distances'].clear()
        date_str = strftime("%Y_%m_%d_%H_%M_%S")
        state['base_filename'] = f"{DATA_DIR}/{slider_speed.value}_{slider_steering.value}_{date_str}"
        state['csv_file'] = open(f"{state['base_filename']}_dataset.csv", 'w')
        state['csv_file'].write("Time_s,Encoder_Counts,Steering,Lidar_Angles,Lidar_Distances\n")
        state['video_writer'] = _open_video_writer(f"{state['base_filename']}_video.mp4")
        ui.notify('5-Second Trial Started & Recording...', type='positive')

    def end_trial():
        state['running_trial'] = False
        speed_switch.value = steering_switch.value = False
        if state['sender']: state['sender'].send_control_signal([0, 0])
        if state['csv_file']:  state['csv_file'].close();  state['csv_file'] = None
        if state['video_writer']: state['video_writer'].release(); state['video_writer'] = None
        main_plot.fig.savefig(f"{state['base_filename']}_plot.png",
                              dpi=300, bbox_inches='tight', facecolor='#0f172a')
        ui.notify(f'Trial saved to {DATA_DIR}/', type='info')
        trial_timer_label.set_text('0.0s')

    # ── PD controller callbacks ───────────────────────────────────────────
    def start_pd_controller():
        if not state['connected']:
            ui.notify('Connect to hardware first!', type='warning'); return
        try:
            gx, gy = float(pd_goal_x.value), float(pd_goal_y.value)
        except (ValueError, TypeError):
            ui.notify('Enter valid X and Y goal coordinates.', type='negative'); return

        sl_ctrl.reset()
        if state['sl_video_writer']:
            state['sl_video_writer'].release(); state['sl_video_writer'] = None

        pd_ctrl.reset()
        pd_ctrl.KP_STEER    = float(pd_kp_steer.value or PDPositionController.KP_STEER)
        pd_ctrl.KD_STEER    = float(pd_kd_steer.value or PDPositionController.KD_STEER)
        pd_ctrl.KP_SPEED    = float(pd_kp_speed.value or PDPositionController.KP_SPEED)
        pd_ctrl.MIN_SPEED_CMD = float(pd_min_spd.value or PDPositionController.MIN_SPEED_CMD)
        pd_ctrl.set_goal(gx, gy)
        state['ctrl_mode'] = 'pd_position'

        date_str = strftime("%Y_%m_%d_%H_%M_%S")
        state['_pd_base']      = f"{PD_DATA_DIR}/pd_{gx:.2f}_{gy:.2f}_{date_str}"
        state['pd_video_writer'] = _open_video_writer(state['_pd_base'] + "_video.mp4")

        pd_status_label.set_text('▶ Running')
        pd_status_label.classes('text-blue-400', remove='text-slate-500 text-green-400')
        pd_card.classes('controller-active', remove='controller-done')
        ui.notify(f'PD Controller → Goal ({gx:.2f}, {gy:.2f})', type='positive')

    def finish_pd_controller(reached: bool):
        state['ctrl_mode'] = None
        pd_ctrl.save_all(state['_pd_base'], history,
                         parameters.wall_corner_list,
                         video_writer=state['pd_video_writer'])
        state['pd_video_writer'] = None
        if reached:
            pd_status_label.set_text('✓ Reached')
            pd_status_label.classes('text-green-400', remove='text-blue-400 text-slate-500')
            pd_card.classes('controller-done', remove='controller-active')
            pd_dist_label.set_text('0.000 m')
            ui.notify(f'PD Goal reached! Data saved → {PD_DATA_DIR}/', type='positive')
        else:
            pd_status_label.set_text('■ Stopped')
            pd_status_label.classes('text-slate-500', remove='text-blue-400 text-green-400')
            pd_card.classes(remove='controller-active controller-done')
            ui.notify(f'PD stopped. Data saved → {PD_DATA_DIR}/', type='info')

    def stop_pd_controller():
        if state['sender']: state['sender'].send_control_signal([0, 0])
        pd_ctrl.active = False
        finish_pd_controller(reached=False)

    # ── Straight-line callbacks ───────────────────────────────────────────
    def start_sl_controller():
        if not state['connected']:
            ui.notify('Connect to hardware first!', type='warning'); return
        try:
            dist = float(sl_distance.value)
            if dist <= 0: raise ValueError
        except (ValueError, TypeError):
            ui.notify('Enter a positive distance (metres).', type='negative'); return

        pd_ctrl.reset()
        if state['pd_video_writer']:
            state['pd_video_writer'].release(); state['pd_video_writer'] = None

        pose = pf.get_estimate()
        sl_ctrl.reset()
        sl_ctrl.KP_STEER      = float(sl_kp_steer.value or StraightLineController.KP_STEER)
        sl_ctrl.KD_STEER      = float(sl_kd_steer.value or StraightLineController.KD_STEER)
        sl_ctrl.KC_CROSS      = float(sl_kc.value      or StraightLineController.KC_CROSS)
        sl_ctrl.D_FILTER      = float(sl_d_filter.value if sl_d_filter.value is not None else StraightLineController.D_FILTER)
        sl_ctrl.KP_SPEED      = float(sl_kp_speed.value or StraightLineController.KP_SPEED)
        sl_ctrl.MIN_SPEED_CMD = float(sl_min_spd.value  or StraightLineController.MIN_SPEED_CMD)
        sl_ctrl.set_goal(pose, dist)
        state['ctrl_mode'] = 'straight_line'

        date_str = strftime("%Y_%m_%d_%H_%M_%S")
        state['_sl_base']      = f"{SL_DATA_DIR}/sl_{dist:.2f}m_{date_str}"
        state['sl_video_writer'] = _open_video_writer(state['_sl_base'] + "_video.mp4")

        sl_status_label.set_text('▶ Running')
        sl_status_label.classes('text-blue-400', remove='text-slate-500 text-green-400')
        sl_card.classes('controller-active', remove='controller-done')
        ui.notify(f'Straight-Line Controller → {dist:.2f} m', type='positive')

    def finish_sl_controller(reached: bool):
        state['ctrl_mode'] = None
        sl_ctrl.save_all(state['_sl_base'], parameters.wall_corner_list,
                         video_writer=state['sl_video_writer'])
        state['sl_video_writer'] = None
        if reached:
            sl_status_label.set_text('✓ Reached')
            sl_status_label.classes('text-green-400', remove='text-blue-400 text-slate-500')
            sl_card.classes('controller-done', remove='controller-active')
            sl_dist_label.set_text('0.000 m')
            ui.notify(f'SL Distance reached! Data saved → {SL_DATA_DIR}/', type='positive')
        else:
            sl_status_label.set_text('■ Stopped')
            sl_status_label.classes('text-slate-500', remove='text-blue-400 text-green-400')
            sl_card.classes(remove='controller-active controller-done')
            ui.notify(f'SL stopped. Data saved → {SL_DATA_DIR}/', type='info')

    def stop_sl_controller():
        if state['sender']: state['sender'].send_control_signal([0, 0])
        sl_ctrl.active = False
        finish_sl_controller(reached=False)

    # ── Localization plot ─────────────────────────────────────────────────
    def show_localization_plot():
        with main_plot:
            fig = main_plot.fig; fig.patch.set_facecolor('#0f172a'); plt.clf()
            ax  = plt.gca(); ax.set_facecolor('#0f172a')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.tick_params(axis='x', colors='#94a3b8'); ax.tick_params(axis='y', colors='#94a3b8')

            for wall in parameters.wall_corner_list:
                ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='white', linewidth=2, zorder=1)

            px = [p.x for p in pf.particles[::2]]; py = [p.y for p in pf.particles[::2]]
            ax.scatter(px, py, s=2, c='#22c55e', alpha=0.3, zorder=2)

            if history['dr_x']:
                ax.plot(history['dr_x'], history['dr_y'],
                        color='#f97316', linestyle='--', linewidth=1.5, alpha=0.8,
                        label='Dead Reckoning', zorder=3)
                ax.plot(history['est_x'], history['est_y'],
                        color='#3b82f6', linestyle='-', linewidth=1.5, alpha=0.6,
                        label='Particle Filter', zorder=4)

            est_pose = pf.get_estimate()
            xs = est_pose[0] + X_OFFSET * math.cos(est_pose[2])
            ys = est_pose[1] + X_OFFSET * math.sin(est_pose[2])
            for i in range(0, state['sensor_signal'].num_lidar_rays, 5):
                d = state['sensor_signal'].distances[i]
                if 100 < d < 4900:
                    ang = angle_wrap(est_pose[2] - state['sensor_signal'].angles[i]*math.pi/180.0)
                    ax.plot([xs, xs + d/1000*math.cos(ang)],
                            [ys, ys + d/1000*math.sin(ang)],
                            color='#ef4444', alpha=0.2, linewidth=0.5, zorder=5)

            ax.plot(dr.state[0], dr.state[1], 'o', color='#f97316', markersize=5, zorder=6)
            ax.plot(est_pose[0], est_pose[1], 'o', color='#3b82f6', markersize=5, zorder=7)
            ax.quiver(est_pose[0], est_pose[1],
                      math.cos(est_pose[2]), math.sin(est_pose[2]),
                      color='#ec4899', scale=15, width=0.007, zorder=8)

            if state['ctrl_mode'] == 'pd_position' and pd_ctrl.active:
                ax.plot(pd_ctrl.goal_x, pd_ctrl.goal_y, '*', color='#facc15',
                        markersize=12, zorder=9, label='PD Goal')
                ax.plot([est_pose[0], pd_ctrl.goal_x], [est_pose[1], pd_ctrl.goal_y],
                        color='#facc15', linestyle=':', linewidth=1.0, alpha=0.5, zorder=8)
            elif state['ctrl_mode'] == 'straight_line' and sl_ctrl.active:
                ex = sl_ctrl.start_x + sl_ctrl.target_dist*math.cos(sl_ctrl.reference_heading)
                ey = sl_ctrl.start_y + sl_ctrl.target_dist*math.sin(sl_ctrl.reference_heading)
                ax.plot(ex, ey, 'D', color='#a78bfa', markersize=8, zorder=9, label='SL Goal')
                ax.plot([sl_ctrl.start_x, ex], [sl_ctrl.start_y, ey],
                        color='#a78bfa', linestyle=':', linewidth=1.0, alpha=0.5, zorder=8)

            ax.grid(True, color='#1e293b', linestyle='--')
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect('equal')
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left',
                          facecolor='#0f172a', edgecolor='#334155',
                          labelcolor='#94a3b8', fontsize=7)

    # ======================================================================
    # GUI LAYOUT
    # ======================================================================
    with ui.header().classes(f'{HEADER_BG} shadow-md p-4 flex items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('smart_toy', size='32px', color='blue-400')
            ui.label('Robot Command Center').classes('text-xl font-bold tracking-wide text-white')
        with ui.row().classes('items-center gap-2 bg-slate-800 px-3 py-1 rounded-full'):
            status_indicator = ui.element('div').classes(
                'w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]')
            status_label = ui.label('Disconnected').classes('text-xs font-semibold text-slate-300')

    with ui.column().classes('w-full p-6 gap-6 items-center max-w-7xl mx-auto'):

        # ── Row 1 ─────────────────────────────────────────────────────────
        with ui.grid(columns=3).classes('w-full gap-6'):
            with ui.card().classes(f'w-full {CARD_BG} p-0 overflow-hidden relative'):
                ui.label('Camera Feed').classes(
                    'absolute top-3 left-4 z-10 text-xs font-bold text-white/70 '
                    'bg-black/50 px-2 py-1 rounded backdrop-blur-sm')
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full h-64 object-cover')
                else:
                    with ui.column().classes('w-full h-64 items-center justify-center bg-slate-800'):
                        ui.icon('videocam_off', size='48px', color='slate-600')
                        ui.label('No Video Stream').classes('text-slate-500 text-sm mt-2')
                    video_image = None

            with ui.card().classes(f'w-full {CARD_BG} items-center justify-center p-2'):
                main_plot = ui.pyplot(figsize=(3.5, 3.5), close=False)

            with ui.card().classes(f'w-full {CARD_BG} p-5 flex flex-col justify-between'):
                ui.label('Telemetry').classes('text-sm font-bold text-slate-400 mb-4 uppercase tracking-wider')
                with ui.row().classes('items-baseline justify-between w-full mb-2'):
                    ui.label('Encoder Count').classes(TEXT_COLOR)
                    encoder_count_label = ui.label('0').classes('text-2xl font-mono text-blue-400')
                ui.separator().classes('bg-slate-700 my-4')
                udp_switch = ui.switch('Hardware Connection').props('color=green keep-color').classes('text-slate-300')
                with ui.row().classes('w-full items-center justify-between mt-2'):
                    ui.button('START TRIAL', on_click=run_trial).props('unelevated').classes(
                        'bg-blue-600 hover:bg-blue-500 text-white w-2/3 rounded-lg font-bold')
                    trial_timer_label = ui.label('0.0s').classes('text-slate-400 font-mono text-sm')

        # ── Row 2: Manual Drive ───────────────────────────────────────────
        with ui.card().classes(f'w-full {CARD_BG} p-6'):
            ui.label('Drive Control').classes('text-sm font-bold text-slate-400 mb-6 uppercase tracking-wider')
            with ui.grid(columns=2).classes('w-full gap-12'):
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('speed', color='blue-400')
                            ui.label('Speed').classes('text-lg font-medium text-white')
                        speed_switch = ui.switch().props('color=blue dense')
                    slider_speed = ui.slider(min=-100, max=100, value=0).props(
                        'label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('directions_car', color='blue-400')
                            ui.label('Steering').classes('text-lg font-medium text-white')
                        steering_switch = ui.switch().props('color=blue dense')
                    slider_steering = ui.slider(min=-20, max=20, value=0).props(
                        'label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')

        # ── Row 3: Autonomous Controllers ─────────────────────────────────
        with ui.grid(columns=2).classes('w-full gap-6'):

            # ── PD Position Controller ─────────────────────────────────────
            with ui.card().classes(f'w-full {CARD_BG} p-5') as pd_card:
                with ui.row().classes('items-center gap-2 mb-3'):
                    ui.icon('my_location', color='yellow-400')
                    ui.label('PD Position Controller').classes(
                        'text-sm font-bold text-slate-300 uppercase tracking-wider')
                    ui.space()
                    pd_status_label = ui.label('■ Idle').classes('text-xs font-mono text-slate-500')

                ui.label('Target Position (metres)').classes('text-xs text-slate-500 mb-2')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    with ui.column().classes('gap-1'):
                        ui.label('Goal X').classes('text-xs text-slate-400')
                        pd_goal_x = ui.number(value=0.5, step=0.05, format='%.2f').props(
                            'dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('Goal Y').classes('text-xs text-slate-400')
                        pd_goal_y = ui.number(value=0.5, step=0.05, format='%.2f').props(
                            'dense outlined dark').classes('w-full')

                ui.separator().classes('bg-slate-700 my-2')

                # Steering loop gains
                with ui.row().classes('items-center gap-1 mb-2'):
                    ui.icon('turn_right', size='16px', color='pink-400')
                    ui.label('Steering PD  —  heading error → steer cmd').classes('text-xs text-slate-400')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    with ui.column().classes('gap-1'):
                        ui.label('Kp Steer').classes('text-xs text-slate-400')
                        pd_kp_steer = ui.number(value=PDPositionController.KP_STEER, step=0.5,
                                                format='%.1f').props('dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('Kd Steer').classes('text-xs text-slate-400')
                        pd_kd_steer = ui.number(value=PDPositionController.KD_STEER, step=0.1,
                                                format='%.1f').props('dense outlined dark').classes('w-full')

                # Speed loop gains
                with ui.row().classes('items-center gap-1 mb-2'):
                    ui.icon('speed', size='16px', color='yellow-400')
                    ui.label('Speed P  —  distance error → speed cmd').classes('text-xs text-slate-400')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    with ui.column().classes('gap-1'):
                        ui.label('Kp Speed').classes('text-xs text-slate-400')
                        pd_kp_speed = ui.number(value=PDPositionController.KP_SPEED, step=5.0,
                                                format='%.1f').props('dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('Min Speed Cmd').classes('text-xs text-slate-400')
                        pd_min_spd = ui.number(value=PDPositionController.MIN_SPEED_CMD, step=1.0,
                                               format='%.0f').props('dense outlined dark').classes('w-full')

                with ui.row().classes('w-full gap-3 mt-2'):
                    ui.button('▶  Go to Goal', on_click=start_pd_controller).props('unelevated').classes(
                        'bg-yellow-600 hover:bg-yellow-500 text-black font-bold flex-1 rounded-lg')
                    ui.button('■  Stop', on_click=stop_pd_controller).props('unelevated').classes(
                        'bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-4')

                with ui.row().classes('w-full justify-between mt-3 pt-3 border-t border-slate-700'):
                    ui.label('Distance to goal').classes('text-xs text-slate-500')
                    pd_dist_label = ui.label('—').classes('text-xs font-mono text-yellow-400')

            # ── Straight-Line Controller ───────────────────────────────────
            with ui.card().classes(f'w-full {CARD_BG} p-5') as sl_card:
                with ui.row().classes('items-center gap-2 mb-3'):
                    ui.icon('arrow_forward', color='purple-400')
                    ui.label('Straight-Line Controller').classes(
                        'text-sm font-bold text-slate-300 uppercase tracking-wider')
                    ui.space()
                    sl_status_label = ui.label('■ Idle').classes('text-xs font-mono text-slate-500')

                ui.label('Travel distance (metres)').classes('text-xs text-slate-500 mb-2')
                with ui.column().classes('w-full gap-1 mb-3'):
                    ui.label('Distance (m)').classes('text-xs text-slate-400')
                    sl_distance = ui.number(value=0.5, min=0.01, step=0.05, format='%.2f').props(
                        'dense outlined dark').classes('w-full')

                ui.separator().classes('bg-slate-700 my-2')

                with ui.row().classes('items-center gap-1 mb-2'):
                    ui.icon('turn_right', size='16px', color='pink-400')
                    ui.label('Steering PD  —  heading + cross-track → steer cmd').classes('text-xs text-slate-400')
                with ui.grid(columns=2).classes('w-full gap-3 mb-2'):
                    with ui.column().classes('gap-1'):
                        ui.label('Kp Steer').classes('text-xs text-slate-400')
                        sl_kp_steer = ui.number(value=StraightLineController.KP_STEER, step=0.5,
                                                format='%.1f').props('dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('Kd Steer').classes('text-xs text-slate-400')
                        sl_kd_steer = ui.number(value=StraightLineController.KD_STEER, step=0.1,
                                                format='%.1f').props('dense outlined dark').classes('w-full')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    with ui.column().classes('gap-1'):
                        ui.label('Kc Cross-Track').classes('text-xs text-slate-400')
                        ui.label('pulls back to reference line').classes('text-xs text-slate-600')
                        sl_kc = ui.number(value=StraightLineController.KC_CROSS, step=0.1,
                                          format='%.1f').props('dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('D-Filter (0=off, 1=frozen)').classes('text-xs text-slate-400')
                        ui.label('suppresses PF noise spikes').classes('text-xs text-slate-600')
                        sl_d_filter = ui.number(value=StraightLineController.D_FILTER, step=0.05,
                                                min=0.0, max=0.95, format='%.2f').props('dense outlined dark').classes('w-full')

                with ui.row().classes('items-center gap-1 mb-2'):
                    ui.icon('speed', size='16px', color='purple-400')
                    ui.label('Speed P  —  remaining dist → speed cmd').classes('text-xs text-slate-400')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    with ui.column().classes('gap-1'):
                        ui.label('Kp Speed').classes('text-xs text-slate-400')
                        sl_kp_speed = ui.number(value=StraightLineController.KP_SPEED, step=5.0,
                                                format='%.1f').props('dense outlined dark').classes('w-full')
                    with ui.column().classes('gap-1'):
                        ui.label('Min Speed Cmd').classes('text-xs text-slate-400')
                        sl_min_spd = ui.number(value=StraightLineController.MIN_SPEED_CMD, step=1.0,
                                               format='%.0f').props('dense outlined dark').classes('w-full')

                with ui.row().classes('w-full gap-3 mt-2'):
                    ui.button('▶  Drive Straight', on_click=start_sl_controller).props('unelevated').classes(
                        'bg-purple-600 hover:bg-purple-500 text-white font-bold flex-1 rounded-lg')
                    ui.button('■  Stop', on_click=stop_sl_controller).props('unelevated').classes(
                        'bg-slate-700 hover:bg-slate-600 text-white rounded-lg px-4')

                with ui.row().classes('w-full justify-between mt-3 pt-3 border-t border-slate-700'):
                    ui.label('Distance remaining').classes('text-xs text-slate-500')
                    sl_dist_label = ui.label('—').classes('text-xs font-mono text-purple-400')

    # ======================================================================
    # Control Loop  (100 ms)
    # ======================================================================
    async def control_loop():
        update_connection_to_robot()

        if state['running_trial']:
            dt_trial = get_time_in_ms() - state['trial_start_time']
            trial_timer_label.set_text(f"{dt_trial / 1000:.1f}s")
            if dt_trial > 5000: end_trial()

        current_time = time.time()
        dt_pf        = current_time - state['pf_last_time']
        state['pf_last_time'] = current_time

        est_pose  = pf.get_estimate()
        cmd_speed = 0
        cmd_steer = 0

        # ── Autonomous priority ───────────────────────────────────────────
        if state['ctrl_mode'] == 'pd_position' and pd_ctrl.active:
            cmd_speed, cmd_steer, reached = pd_ctrl.compute(est_pose, dt_pf)
            ex = pd_ctrl.goal_x - est_pose[0]; ey = pd_ctrl.goal_y - est_pose[1]
            pd_dist_label.set_text(f"{math.hypot(ex, ey):.3f} m")
            if reached: finish_pd_controller(reached=True)

        elif state['ctrl_mode'] == 'straight_line' and sl_ctrl.active:
            cmd_speed, cmd_steer, reached = sl_ctrl.compute(est_pose, dt_pf)
            dx = est_pose[0] - sl_ctrl.start_x; dy = est_pose[1] - sl_ctrl.start_y
            travelled = (dx*math.cos(sl_ctrl.reference_heading)
                        + dy*math.sin(sl_ctrl.reference_heading))
            sl_dist_label.set_text(f"{max(0.0, sl_ctrl.target_dist - travelled):.3f} m")
            if reached: finish_sl_controller(reached=True)

        else:
            if speed_switch.value:    cmd_speed = slider_speed.value
            if steering_switch.value: cmd_steer = slider_steering.value

        # ── Camera ───────────────────────────────────────────────────────
        if stream_video and video_capture.isOpened():
            read_result = await run.io_bound(video_capture.read)
            if read_result is not None:
                ret, state['latest_frame'] = read_result
                frame = state['latest_frame']
                if frame is not None:
                    if state['running_trial']       and state['video_writer']:    state['video_writer'].write(frame)
                    if state['ctrl_mode'] == 'pd_position'   and state['pd_video_writer']: state['pd_video_writer'].write(frame)
                    if state['ctrl_mode'] == 'straight_line' and state['sl_video_writer']: state['sl_video_writer'].write(frame)
                    if video_image: video_image.force_reload()

        # ── Hardware & PF ─────────────────────────────────────────────────
        if state['connected']:
            state['sender'].send_control_signal([cmd_speed, cmd_steer])
            try:
                state['sensor_signal'] = state['receiver'].receive_robot_sensor_signal(
                    state['sensor_signal'])
                encoder_count_label.set_text(str(state['sensor_signal'].encoder_counts))
            except socket.timeout:
                pass

            if state['running_trial'] and state['csv_file']:
                t_rel = (get_time_in_ms() - state['trial_start_time']) / 1000.0
                state['csv_file'].write(
                    f"{t_rel:.3f},{state['sensor_signal'].encoder_counts},{cmd_steer},"
                    f"\"{state['sensor_signal'].angles}\","
                    f"\"{state['sensor_signal'].distances}\"\n")

            if dt_pf > 0:
                pf.predict(cmd_speed, cmd_steer, dt_pf)
                dr.step_update(cmd_speed, cmd_steer, dt_pf)

            sweep_complete = False
            for i in range(state['sensor_signal'].num_lidar_rays):
                ang = state['sensor_signal'].angles[i]; dist = state['sensor_signal'].distances[i]
                if state['last_lidar_angle'] is not None and abs(ang - state['last_lidar_angle']) > 180:
                    sweep_complete = True
                state['sweep_angles'].append(ang); state['sweep_distances'].append(dist)
                state['last_lidar_angle'] = ang

            if sweep_complete:
                pf.correct(state['sweep_angles'], state['sweep_distances'])
                state['sweep_angles'] = []; state['sweep_distances'] = []

            est_pose = pf.get_estimate()
            history['est_x'].append(est_pose[0]); history['est_y'].append(est_pose[1])
            history['dr_x'].append(dr.state[0]);  history['dr_y'].append(dr.state[1])

        show_localization_plot()

    ui.timer(0.1, control_loop)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title='Robot Dashboard', dark=True)