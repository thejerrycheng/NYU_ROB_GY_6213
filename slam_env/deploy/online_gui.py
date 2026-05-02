# External libraries
import os
import asyncio
import cv2
import math
import random
import socket
import heapq
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from nicegui import ui, app, run
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

INITIAL_POSE = [0, 0, 0]

L            = 0.145
V_M          = 0.004808
V_C          = -0.045557
VAR_V        = 0.057829
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA    = 0.023134

MAX_RANGE = 5.0
X_OFFSET  = 0.12
VAR_Z     = 0.0025

# --- SLAM HYPERPARAMETERS ---
NUM_PARTICLES = 50  
PF_RESAMPLE_THRESHOLD = NUM_PARTICLES / 2.0
GRID_RESOLUTION = 0.05
L_0             = 0.0
L_OCC           = 0.85
L_FREE          = -0.4
MAX_LOG_ODDS    = 5.0
MIN_LOG_ODDS    = -5.0

ROBOT_RADIUS           = 0.15
PLANNER_WALL_CLEARANCE = 0.30  
LOOKAHEAD_DISTANCE = 0.4
PROB_FREE_THRESH  = 0.55
PROB_UNKNOWN_LOW  = 0.45
PROB_UNKNOWN_HIGH = 0.55
PROB_WALL_THRESH  = 0.10
RENDER_SKIP       = 5  # Renders GUI plot every 500ms to save CPU

# --- THEME ---
CARD_BG    = 'bg-slate-900'
TEXT_COLOR = 'text-slate-200'
HEADER_BG  = 'bg-slate-950'

# =======================================================
# Utilities
# =======================================================
def get_naive_frontier_mask(prob_grid):
    is_free    = prob_grid > PROB_FREE_THRESH
    is_unknown = ((prob_grid >= PROB_UNKNOWN_LOW) & (prob_grid <= PROB_UNKNOWN_HIGH))
    is_wall    = prob_grid < 0.45
    has_unknown_neighbor = (
        np.roll(is_unknown, 1, axis=0) | np.roll(is_unknown, -1, axis=0) |
        np.roll(is_unknown, 1, axis=1) | np.roll(is_unknown, -1, axis=1)
    )
    wall_buffer = binary_dilation(is_wall, iterations=4)
    return is_free & has_unknown_neighbor & ~wall_buffer

def angle_wrap(angle):
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def get_physical_commands(v_cmd, alpha_cmd):
    if v_cmd == 0.0: return 0.0, 0.0
    v_mag = (V_M * abs(v_cmd)) + V_C
    if v_mag < 0: v_mag = 0.0
    v_phys     = v_mag if v_cmd > 0 else -v_mag
    delta_phys = (DELTA_COEFFS[0] * (alpha_cmd ** 2) + DELTA_COEFFS[1] * alpha_cmd + DELTA_COEFFS[2])
    return v_phys, delta_phys

def predict_next_pose(current_pose, v_phys, delta_phys, delta_t=0.1):
    x, y, theta = current_pose
    w       = (v_phys * math.tan(delta_phys)) / L if L > 0 else 0.0
    next_x  = x + v_phys * math.cos(theta) * delta_t
    next_y  = y + v_phys * math.sin(theta) * delta_t
    next_th = angle_wrap(theta - w * delta_t)
    return np.array([next_x, next_y, next_th])


# =======================================================
# Old Motion Models (Kept for PD / SL plotting compatibility)
# =======================================================
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        if v_cmd == 0.0:
            v_expected = 0.0; w_expected = 0.0
        else:
            v_expected = (V_M * v_cmd) + V_C
            if v_expected < 0: v_expected = 0.0
            alpha = steering_angle_command
            delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]
            w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0
        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2]  = angle_wrap(self.state[2] - delta_t * w_expected)


# =======================================================
# FastSLAM & Active Exploration
# =======================================================
class FastSLAMParticle:
    def __init__(self, initial_pose, grid_shape):
        self.pose = np.array(initial_pose, dtype=float)
        self.weight = 1.0 / NUM_PARTICLES
        self.grid = np.full(grid_shape, L_0)

    def clone(self):
        new_p = FastSLAMParticle(self.pose, self.grid.shape)
        new_p.pose = np.copy(self.pose)
        new_p.weight = self.weight
        new_p.grid = np.copy(self.grid) 
        return new_p

class FastSLAM:
    def __init__(self, initial_pose, bounds):
        self.offset_x = bounds['min_x']
        self.offset_y = bounds['min_y']
        self.W = int((bounds['max_x'] - bounds['min_x']) / GRID_RESOLUTION)
        self.H = int((bounds['max_y'] - bounds['min_y']) / GRID_RESOLUTION)
        
        self.particles = [FastSLAMParticle(initial_pose, (self.W, self.H)) for _ in range(NUM_PARTICLES)]
        self.best_particle = self.particles[0]
        
        self.dist_since_resample = 0.0
        self.rot_since_resample = 0.0

    def world_to_grid(self, x, y):
        return (int((x - self.offset_x) / GRID_RESOLUTION), int((y - self.offset_y) / GRID_RESOLUTION))
        
    def grid_to_world(self, gx, gy):
        return ((gx * GRID_RESOLUTION) + self.offset_x, (gy * GRID_RESOLUTION) + self.offset_y)

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y   = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        points.append((x, y))
        return points

    def predict(self, v_phys, delta_phys, dt):
        self.dist_since_resample += abs(v_phys) * dt
        w = (v_phys * math.tan(delta_phys)) / L if L > 0 else 0.0
        self.rot_since_resample += abs(w) * dt
        
        for p in self.particles:
            v_noisy = v_phys + random.gauss(0, math.sqrt(VAR_V)*2.0) if v_phys != 0 else 0.0
            d_noisy = delta_phys + random.gauss(0, math.sqrt(VAR_DELTA)*2.0) if v_phys != 0 else 0.0
            p.pose = predict_next_pose(p.pose, v_noisy, d_noisy, dt)

    def update(self, angles, distances, max_range=5.0):
        slam_angles = angles[::4]
        slam_distances = distances[::4]
        weight_sum = 0.0
        
        search_offsets = [
            (0.0, 0.0, 0.0),
            (0.10, 0.0, 0.0), (-0.10, 0.0, 0.0),   
            (0.0, 0.10, 0.0), (0.0, -0.10, 0.0),   
            (0.0, 0.0, 0.10), (0.0, 0.0, -0.10)
        ]
        
        for p in self.particles:
            best_score = -999999.0
            best_pose = p.pose
            prob_grid = 1.0 / (1.0 + np.exp(-p.grid))
            
            for dx, dy, dtheta in search_offsets:
                test_pose = np.array([
                    p.pose[0] + dx * math.cos(p.pose[2]) - dy * math.sin(p.pose[2]),
                    p.pose[1] + dx * math.sin(p.pose[2]) + dy * math.cos(p.pose[2]),
                    angle_wrap(p.pose[2] + dtheta)
                ])
                
                score = -((math.hypot(dx, dy) * 40.0) + (abs(dtheta) * 20.0))
                rx, ry, rtheta = test_pose
                valid_hits = 0
                
                for i in range(len(slam_angles)):
                    dist = slam_distances[i]
                    if dist >= max_range - 0.1: continue 
                    
                    glob_angle = rtheta + slam_angles[i]
                    end_x = rx + dist * math.cos(glob_angle)
                    end_y = ry + dist * math.sin(glob_angle)
                    gx, gy = self.world_to_grid(end_x, end_y)
                    
                    if 0 <= gx < self.W and 0 <= gy < self.H:
                        score += (prob_grid[gx, gy] - 0.5) * 10.0
                        valid_hits += 1
                
                if valid_hits < 5 and dx != 0.0: continue
                if score > best_score:
                    best_score = score
                    best_pose = test_pose
            
            p.pose = best_pose
            p.weight *= math.exp(best_score / max(1, len(slam_angles)))
            weight_sum += p.weight
            
        if weight_sum > 0:
            for p in self.particles: p.weight /= weight_sum
        else:
            for p in self.particles: p.weight = 1.0 / NUM_PARTICLES

        n_eff = 1.0 / sum([p.weight**2 for p in self.particles])

        if n_eff < PF_RESAMPLE_THRESHOLD and (self.dist_since_resample > 0.3 or self.rot_since_resample > 0.3):
            new_particles = []
            r = random.uniform(0, 1.0 / NUM_PARTICLES)
            c = self.particles[0].weight
            i = 0
            for m in range(NUM_PARTICLES):
                U = r + m * (1.0 / NUM_PARTICLES)
                while U > c:
                    i += 1
                    c += self.particles[i].weight
                cloned_p = self.particles[i].clone()
                cloned_p.weight = 1.0 / NUM_PARTICLES
                new_particles.append(cloned_p)
            self.particles = new_particles
            self.dist_since_resample = 0.0
            self.rot_since_resample = 0.0

        self.best_particle = max(self.particles, key=lambda p: p.weight)

        for p in self.particles:
            rx, ry, rtheta = p.pose
            gx0, gy0 = self.world_to_grid(rx, ry)
            
            for i in range(len(slam_angles)):
                dist = slam_distances[i]
                glob_angle = rtheta + slam_angles[i]
                end_x = rx + dist * math.cos(glob_angle)
                end_y = ry + dist * math.sin(glob_angle)
                gx1, gy1 = self.world_to_grid(end_x, end_y)
                
                cells = self.bresenham_line(gx0, gy0, gx1, gy1)
                for j, (cx, cy) in enumerate(cells):
                    if 0 <= cx < self.W and 0 <= cy < self.H:
                        if j == len(cells) - 1 and dist < (max_range - 0.1):
                            p.grid[cx, cy] += L_OCC
                        else:
                            if p.grid[cx, cy] < 1.5: 
                                p.grid[cx, cy] += L_FREE
                        p.grid[cx, cy] = np.clip(p.grid[cx, cy], MIN_LOG_ODDS, MAX_LOG_ODDS)

    def get_best_probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.best_particle.grid))

class ActiveSLAMController:
    def __init__(self, slam):
        self.slam            = slam
        self.active          = False
        self.current_path    = []
        self.target_frontier = None
        self.step_counter    = 0
        self.cached_inflated = None
        self.blacklisted_frontiers = []
        self.recovery_steps  = 0

        self.KP_steer = 40.0; self.KD_steer = 6.0
        self.KP_speed = 100.0; self.MIN_V_CMD = 40.0   
        self.MAX_V_CMD = 80.0; self.MAX_ALPHA_CMD = 100.0
        self.GOAL_TOLERANCE = 0.20   
        self.STUCK_CHECK_STEPS = 30; self.STUCK_DIST_MIN = 0.05   
        self.stuck_check_pose = None; self.stuck_timer = 0; self.prev_heading_err = 0.0
        self.EMERGENCY_BRAKE_DIST = 0.22

    def reset(self):
        self.active = False
        self.current_path = []
        self.target_frontier = None
        self.step_counter = 0
        self.blacklisted_frontiers = []
        self._reset_pd()

    def start(self):
        self.reset()
        self.active = True

    def _reset_pd(self):
        self.prev_heading_err = 0.0; self.stuck_check_pose = None; self.stuck_timer = 0

    def _subsample_path(self, path, step_m=0.20):
        if len(path) <= 2: return path
        subsampled = [path[0]]; accumulated = 0.0
        for i in range(1, len(path)):
            dx, dy = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            accumulated += math.hypot(dx, dy)
            if accumulated >= step_m:
                subsampled.append(path[i]); accumulated = 0.0
        if subsampled[-1] != path[-1]: subsampled.append(path[-1])   
        return subsampled

    def get_inflated_obstacles(self):
        prob_grid = self.slam.get_best_probabilities()
        confirmed_wall = prob_grid < 0.20
        probable_wall  = prob_grid < 0.40
        confirmed_inflation = int((ROBOT_RADIUS + PLANNER_WALL_CLEARANCE) / GRID_RESOLUTION)
        probable_inflation  = int(ROBOT_RADIUS / GRID_RESOLUTION)
        return (binary_dilation(confirmed_wall, iterations=confirmed_inflation) | binary_dilation(probable_wall, iterations=probable_inflation))

    def find_frontiers(self, inflated_obstacles):
        prob_grid  = self.slam.get_best_probabilities()
        is_free    = prob_grid > PROB_FREE_THRESH
        is_unknown = ((prob_grid >= PROB_UNKNOWN_LOW) & (prob_grid <= PROB_UNKNOWN_HIGH))
        unknown_expanded = maximum_filter(is_unknown, size=5)
        frontier_grid    = is_free & unknown_expanded & ~inflated_obstacles
        frontier_pixels  = np.argwhere(frontier_grid)
        if len(frontier_pixels) == 0: return []
        sampled = frontier_pixels[::8]; candidates = []
        for px in sampled:
            gx, gy = int(px[0]), int(px[1])
            if not (0 <= gx < self.slam.W and 0 <= gy < self.slam.H): continue
            if inflated_obstacles[gx, gy]: continue
            wx, wy = self.slam.grid_to_world(gx, gy)
            candidates.append((wx, wy))
        return candidates

    def is_kinematically_reachable(self, robot_pose, goal_pos):
        rx, ry, rtheta = robot_pose; gx, gy = goal_pos
        if math.hypot(gx - rx, gy - ry) < self.GOAL_TOLERANCE: return False
        if abs(angle_wrap(math.atan2(gy - ry, gx - rx) - rtheta)) > math.radians(150): return False
        delta_max  = abs(DELTA_COEFFS[1] * self.MAX_ALPHA_CMD + DELTA_COEFFS[2])
        min_radius = (L / math.tan(delta_max) if delta_max > 1e-6 and math.tan(delta_max) > 1e-6 else 999.0)
        local_x =  (gx - rx) * math.cos(rtheta) + (gy - ry) * math.sin(rtheta)
        local_y = -(gx - rx) * math.sin(rtheta) + (gy - ry) * math.cos(rtheta)
        if abs(local_y) > 1e-6 and (local_x ** 2 + local_y ** 2) / (2.0 * abs(local_y)) < min_radius * 0.4: return False
        return True

    def a_star_plan(self, start_pose, goal_world, inflated_obstacles):
        sgx, sgy = self.slam.world_to_grid(start_pose[0], start_pose[1])
        ggx, ggy = self.slam.world_to_grid(goal_world[0], goal_world[1])
        if not (0 <= ggx < self.slam.W and 0 <= ggy < self.slam.H): return []
        
        safe = inflated_obstacles.copy()
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = sgx + dx, sgy + dy
                if 0 <= nx < self.slam.W and 0 <= ny < self.slam.H: safe[nx, ny] = False

        open_set = []; heapq.heappush(open_set, (0, (sgx, sgy)))
        came_from = {}; g_score = {(sgx, sgy): 0}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == (ggx, ggy):
                path = []
                while cur in came_from:
                    path.append(self.slam.grid_to_world(cur[0], cur[1]))
                    cur = came_from[cur]
                return path[::-1]
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0), (1,1),(-1,-1),(1,-1),(-1,1)]:
                nb = (cur[0] + dx, cur[1] + dy)
                if (0 <= nb[0] < self.slam.W and 0 <= nb[1] < self.slam.H and not safe[nb[0], nb[1]]):
                    tg = g_score[cur] + math.hypot(dx, dy)
                    if nb not in g_score or tg < g_score[nb]:
                        came_from[nb] = cur; g_score[nb] = tg
                        heapq.heappush(open_set, (tg + math.hypot(ggx - nb[0], ggy - nb[1]), nb))
        return []
        
    def local_planner_check(self, robot_pose, inflated_obstacles):
        if not self.current_path: return False
        check_wps = min(8, len(self.current_path))
        for wp in self.current_path[:check_wps]:
            gx, gy = self.slam.world_to_grid(wp[0], wp[1])
            if 0 <= gx < self.slam.W and 0 <= gy < self.slam.H and inflated_obstacles[gx, gy]:
                self.current_path = []
                self._reset_pd()
                return True
        return False

    def pd_controller(self, robot_pose, lidar_angles, lidar_distances, dt=0.1):
        if not self.current_path: return 0.0, 0.0
        rx, ry, rtheta = robot_pose

        self.stuck_timer += 1
        if self.stuck_check_pose is None: self.stuck_check_pose = [rx, ry]
        elif self.stuck_timer >= self.STUCK_CHECK_STEPS:
            if math.hypot(rx - self.stuck_check_pose[0], ry - self.stuck_check_pose[1]) < self.STUCK_DIST_MIN:
                if self.target_frontier: self.blacklisted_frontiers.append(self.target_frontier)
                self.current_path = []; self.target_frontier = None; self._reset_pd()
                self.recovery_steps = 10; return -self.MIN_V_CMD, 0.0
            self.stuck_check_pose = [rx, ry]; self.stuck_timer = 0

        dists = [math.hypot(p[0] - rx, p[1] - ry) for p in self.current_path]
        closest_idx = int(np.argmin(dists))
        if closest_idx > 0: self.current_path = self.current_path[closest_idx:]
        
        lookahead_idx = 0
        for i in range(len(self.current_path)):
            if math.hypot(self.current_path[i][0] - rx, self.current_path[i][1] - ry) > LOOKAHEAD_DISTANCE:
                lookahead_idx = i
                break
        
        if lookahead_idx == 0 and len(self.current_path) > 1:
            lookahead_idx = len(self.current_path) - 1
            
        wp_x, wp_y = self.current_path[lookahead_idx]

        heading_err = angle_wrap(math.atan2(wp_y - ry, wp_x - rx) - rtheta)
        
        repulsive_angular_force = 0.0
        if lidar_distances:
            for ang, dist in zip(lidar_angles, lidar_distances):
                rel_ang = angle_wrap(ang)
                if dist < 0.4 and abs(rel_ang) < math.radians(70): 
                    force_magnitude = (0.4 - dist) * 2.5 
                    repulsive_angular_force -= math.copysign(force_magnitude, rel_ang)

        blended_heading_err = heading_err + repulsive_angular_force

        d_heading = (blended_heading_err - self.prev_heading_err) / dt if dt > 0 else 0.0
        self.prev_heading_err = blended_heading_err

        alpha_cmd = float(np.clip(-(self.KP_steer * blended_heading_err + self.KD_steer * d_heading), -self.MAX_ALPHA_CMD, self.MAX_ALPHA_CMD))
        dist_to_final = math.hypot(self.current_path[-1][0] - rx, self.current_path[-1][1] - ry)
        base_v_cmd = self.MAX_V_CMD if dist_to_final > 0.5 else float(np.clip(self.KP_speed * dist_to_final, self.MIN_V_CMD, self.MAX_V_CMD))
        
        v_cmd = max(self.MIN_V_CMD, base_v_cmd * 0.4) if abs(blended_heading_err) > 0.4 else base_v_cmd
        return v_cmd, alpha_cmd

    def update(self, robot_pose, lidar_angles, lidar_distances):
        self.step_counter += 1
        
        if lidar_distances and self.recovery_steps == 0:
            for ang, dist in zip(lidar_angles, lidar_distances):
                if abs(angle_wrap(ang)) < math.radians(35) and dist < self.EMERGENCY_BRAKE_DIST:
                    if self.target_frontier: self.blacklisted_frontiers.append(self.target_frontier)
                    self.current_path = []; self.target_frontier = None; self._reset_pd()
                    self.recovery_steps = 10; break

        if self.recovery_steps > 0:
            self.recovery_steps -= 1; return -self.MIN_V_CMD, 0.0

        if self.cached_inflated is None or self.step_counter % 5 == 0:
            self.cached_inflated = self.get_inflated_obstacles()
        inflated_obstacles = self.cached_inflated

        self.local_planner_check(robot_pose, inflated_obstacles)

        if self.target_frontier is not None and math.hypot(self.target_frontier[0] - robot_pose[0], self.target_frontier[1] - robot_pose[1]) < self.GOAL_TOLERANCE * 2:
            if self.blacklisted_frontiers: self.blacklisted_frontiers.clear()
            self.target_frontier = None; self.current_path = []; self._reset_pd()

        if self.target_frontier is not None and not self.current_path:
            path = self.a_star_plan(robot_pose, self.target_frontier, inflated_obstacles)
            if path:
                self.current_path = self._subsample_path(path); self._reset_pd()
                return self.pd_controller(robot_pose, lidar_angles, lidar_distances)
            else: self.target_frontier = None

        if self.current_path: return self.pd_controller(robot_pose, lidar_angles, lidar_distances)

        candidates = self.find_frontiers(inflated_obstacles)
        if not candidates:
            return (self.MIN_V_CMD, 0.0) if self.step_counter < 30 else (None, None)

        feasible = [c for c in candidates if not any(math.hypot(c[0]-b[0], c[1]-b[1]) < 1.0 for b in self.blacklisted_frontiers) and self.is_kinematically_reachable(robot_pose, c)]
        if not feasible and candidates and self.blacklisted_frontiers:
            self.blacklisted_frontiers.clear()
            feasible = [c for c in candidates if self.is_kinematically_reachable(robot_pose, c)]
        if not feasible: feasible = candidates

        def frontier_cost(c):
            return math.hypot(c[0] - robot_pose[0], c[1] - robot_pose[1]) + 1.5 * abs(angle_wrap(math.atan2(c[1] - robot_pose[1], c[0] - robot_pose[0]) - robot_pose[2]))

        feasible_sorted = sorted(feasible, key=frontier_cost)
        for goal in feasible_sorted:
            path = self.a_star_plan(robot_pose, goal, inflated_obstacles)
            if path:
                self.target_frontier = goal; self.current_path = self._subsample_path(path); self._reset_pd()
                return self.pd_controller(robot_pose, lidar_angles, lidar_distances)
        return 0.0, 0.0


# =======================================================
# PD Controllers (Position / SL)
# =======================================================
class PDPositionController:
    KP_STEER      = 8.0
    KD_STEER      = 1.2
    KP_SPEED      = 60.0
    MAX_SPEED_CMD = 80.0
    MIN_SPEED_CMD = 18.0
    ALIGN_THRESHOLD = 0.15
    GOAL_THRESHOLD  = 0.05

    def __init__(self): self.reset()
    def reset(self):
        self.active              = False
        self.goal_x              = 0.0
        self.goal_y              = 0.0
        self._prev_heading_error = 0.0
        self.log_t          = []; self.log_x          = []; self.log_y          = []
        self.log_ex         = []; self.log_ey         = []; self.log_dist       = []
        self.log_heading_e  = []; self.log_speed_cmd  = []; self.log_steer_cmd  = []
        self._start_time    = None

    def set_goal(self, gx, gy):
        self.goal_x = gx; self.goal_y = gy
        self._prev_heading_error = 0.0
        self._start_time         = time.time()
        self.active              = True

    def compute(self, pose, dt):
        if not self.active: return 0, 0, False
        ex   = self.goal_x - pose[0]
        ey   = self.goal_y - pose[1]
        dist = math.hypot(ex, ey)

        if dist < self.GOAL_THRESHOLD:
            self._append_log(pose, ex, ey, dist, 0.0, 0.0, 0.0)
            self.active = False
            return 0, 0, True

        speed_cmd = float(np.clip(self.KP_SPEED * dist, self.MIN_SPEED_CMD, self.MAX_SPEED_CMD))
        desired_heading = math.atan2(ey, ex)
        heading_error   = angle_wrap(desired_heading - pose[2])

        d_heading = (heading_error - self._prev_heading_error) / dt if dt > 0 else 0.0
        self._prev_heading_error = heading_error

        steer_cmd = float(np.clip(self.KP_STEER * heading_error + self.KD_STEER * d_heading, -20.0, 20.0))
        if abs(heading_error) > self.ALIGN_THRESHOLD: speed_cmd = 0.0

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
        pass 


class StraightLineController:
    KP_STEER      = 3.5
    KD_STEER      = 2.0
    KC_CROSS      = 1.5
    V_NORMALISER  = 0.15
    D_FILTER      = 0.4
    KP_SPEED      = 60.0
    MAX_SPEED_CMD = 80.0
    MIN_SPEED_CMD = 18.0
    GOAL_THRESHOLD = 0.04

    def __init__(self): self.reset()
    def reset(self):
        self.active              = False
        self.start_x             = 0.0; self.start_y = 0.0
        self.target_dist         = 0.0
        self.reference_heading   = 0.0
        self._prev_combined_error = 0.0
        self._filtered_d_error    = 0.0
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
        if not self.active: return 0, 0, False
        dx        = pose[0] - self.start_x
        dy        = pose[1] - self.start_y
        c, s      = math.cos(self.reference_heading), math.sin(self.reference_heading)
        travelled = dx * c + dy * s
        lateral   = -dx * s + dy * c
        remaining = self.target_dist - travelled

        if remaining < self.GOAL_THRESHOLD:
            self._append_log(pose, lateral, 0.0, 0.0, 0.0, 0.0)
            self.active = False
            return 0, 0, True

        speed_cmd = float(np.clip(self.KP_SPEED * remaining, self.MIN_SPEED_CMD, self.MAX_SPEED_CMD))
        heading_error = angle_wrap(self.reference_heading - pose[2])

        cross_track_correction = math.atan2(self.KC_CROSS * lateral, self.V_NORMALISER)
        combined_error = angle_wrap(heading_error + cross_track_correction)

        if dt > 0:
            raw_d = (combined_error - self._prev_combined_error) / dt
            self._filtered_d_error = (self.D_FILTER * self._filtered_d_error + (1.0 - self.D_FILTER) * raw_d)
        self._prev_combined_error = combined_error

        steer_cmd = float(np.clip(self.KP_STEER * combined_error + self.KD_STEER * self._filtered_d_error, -20.0, 20.0))

        self._append_log(pose, lateral, remaining, combined_error, speed_cmd, steer_cmd)
        return speed_cmd, steer_cmd, False

    def _append_log(self, pose, lat, rem, he, spd, steer):
        t = time.time() - self._start_time if self._start_time else 0.0
        self.log_t.append(t);          self.log_x.append(pose[0])
        self.log_y.append(pose[1]);    self.log_lat_err.append(lat)
        self.log_remaining.append(rem); self.log_heading_e.append(he)
        self.log_speed_cmd.append(spd); self.log_steer_cmd.append(steer)

    def save_all(self, base_path, map_walls, video_writer=None):
        pass


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
            .controller-active { border: 1px solid #3b82f6 !important; box-shadow: 0 0 12px rgba(59,130,246,0.25); }
            .controller-done   { border: 1px solid #22c55e !important; box-shadow: 0 0 12px rgba(34,197,94,0.25); }
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
        'ctrl_mode':        None,   # None | 'pd_position' | 'straight_line' | 'active_slam'
        'pd_video_writer':  None,
        'sl_video_writer':  None,
        '_pd_base':         "",
        '_sl_base':         "",
        'render_step':      0
    }

    # Set up Dynamic 20x20m bounds for the real world around the origin
    slam_bounds = {
        'min_x': INITIAL_POSE[0] - 10.0, 'max_x': INITIAL_POSE[0] + 10.0,
        'min_y': INITIAL_POSE[1] - 10.0, 'max_y': INITIAL_POSE[1] + 10.0,
    }

    fast_slam = FastSLAM(INITIAL_POSE, slam_bounds)
    active_slam_ctrl = ActiveSLAMController(fast_slam)

    dr = MyMotionModel(initial_state=INITIAL_POSE)
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
            state['connected'] = False; pd_ctrl.reset(); sl_ctrl.reset(); active_slam_ctrl.reset()
            state['ctrl_mode'] = None
            status_indicator.classes('bg-red-500', remove='bg-green-500')
            status_label.set_text('Disconnected')

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
        main_plot.fig.savefig(f"{state['base_filename']}_plot.png", dpi=300, bbox_inches='tight', facecolor='#0f172a')
        ui.notify(f'Trial saved to {DATA_DIR}/', type='info')
        trial_timer_label.set_text('0.0s')

    # ── PD controller callbacks ───────────────────────────────────────────
    def start_pd_controller():
        if not state['connected']: ui.notify('Connect to hardware first!', type='warning'); return
        try: gx, gy = float(pd_goal_x.value), float(pd_goal_y.value)
        except (ValueError, TypeError): ui.notify('Enter valid X and Y goal coordinates.', type='negative'); return
        sl_ctrl.reset(); active_slam_ctrl.reset(); pd_ctrl.reset()
        pd_ctrl.set_goal(gx, gy)
        state['ctrl_mode'] = 'pd_position'
        pd_status_label.set_text('▶ Running'); pd_status_label.classes('text-blue-400', remove='text-slate-500 text-green-400')
        pd_card.classes('controller-active', remove='controller-done')

    def stop_pd_controller():
        if state['sender']: state['sender'].send_control_signal([0, 0])
        pd_ctrl.active = False
        state['ctrl_mode'] = None
        pd_status_label.set_text('■ Stopped'); pd_status_label.classes('text-slate-500', remove='text-blue-400 text-green-400')
        pd_card.classes(remove='controller-active controller-done')

    # ── Straight-line callbacks ───────────────────────────────────────────
    def start_sl_controller():
        if not state['connected']: ui.notify('Connect to hardware first!', type='warning'); return
        try: dist = float(sl_distance.value)
        except (ValueError, TypeError): ui.notify('Enter a positive distance (metres).', type='negative'); return
        pd_ctrl.reset(); active_slam_ctrl.reset(); sl_ctrl.reset()
        pose = fast_slam.best_particle.pose
        sl_ctrl.set_goal(pose, dist)
        state['ctrl_mode'] = 'straight_line'
        sl_status_label.set_text('▶ Running'); sl_status_label.classes('text-blue-400', remove='text-slate-500 text-green-400')
        sl_card.classes('controller-active', remove='controller-done')

    def stop_sl_controller():
        if state['sender']: state['sender'].send_control_signal([0, 0])
        sl_ctrl.active = False
        state['ctrl_mode'] = None
        sl_status_label.set_text('■ Stopped'); sl_status_label.classes('text-slate-500', remove='text-blue-400 text-green-400')
        sl_card.classes(remove='controller-active controller-done')

    # ── Active SLAM callbacks ─────────────────────────────────────────────
    def start_active_slam():
        if not state['connected']: ui.notify('Connect to hardware first!', type='warning'); return
        pd_ctrl.reset(); sl_ctrl.reset()
        state['ctrl_mode'] = 'active_slam'
        active_slam_ctrl.start()
        active_status_label.set_text('▶ Exploring'); active_status_label.classes('text-blue-400', remove='text-slate-500 text-green-400')
        active_card.classes('controller-active', remove='controller-done')
        ui.notify('Active SLAM Exploration Started', type='positive')

    def stop_active_slam():
        state['ctrl_mode'] = None
        active_slam_ctrl.active = False
        if state['sender']: state['sender'].send_control_signal([0, 0])
        active_status_label.set_text('■ Stopped'); active_status_label.classes('text-slate-500', remove='text-blue-400 text-green-400')
        active_card.classes(remove='controller-active controller-done')

    # ── Localization plot ─────────────────────────────────────────────────
    def show_localization_plot():
        state['render_step'] += 1
        if state['render_step'] % RENDER_SKIP != 0: return # Only render occasionally to save CPU
        
        with main_plot:
            fig = main_plot.fig; fig.patch.set_facecolor('#0f172a'); plt.clf()
            ax  = plt.gca(); ax.set_facecolor('#0f172a')
            for sp in ax.spines.values(): sp.set_color('#334155')
            ax.tick_params(axis='x', colors='#94a3b8'); ax.tick_params(axis='y', colors='#94a3b8')

            # 1. Base Map (Black/White/Grey)
            prob_grid = fast_slam.get_best_probabilities()
            cmap = LinearSegmentedColormap.from_list('grid_map', ['white', 'lightgrey', 'black'])
            ax.imshow(prob_grid.T, cmap=cmap, origin='lower', 
                      extent=[slam_bounds['min_x'], slam_bounds['max_x'], slam_bounds['min_y'], slam_bounds['max_y']], 
                      vmin=0, vmax=1, zorder=0)

            # 2. Frontier Overlay (Magenta pixels)
            frontier_mask = get_naive_frontier_mask(prob_grid)
            overlay = np.zeros((frontier_mask.shape[0], frontier_mask.shape[1], 4))
            overlay[frontier_mask] = [1, 0, 1, 0.6]  # Magenta w/ Alpha
            ax.imshow(overlay.swapaxes(0, 1), origin='lower', 
                      extent=[slam_bounds['min_x'], slam_bounds['max_x'], slam_bounds['min_y'], slam_bounds['max_y']], 
                      zorder=1)

            # 3. Particle Swarm (Yellow Dots)
            px = [p.pose[0] for p in fast_slam.particles]
            py = [p.pose[1] for p in fast_slam.particles]
            ax.plot(px, py, 'y.', markersize=4, alpha=0.5, zorder=2)

            # 4. Robot History (Blue Dashed Line)
            ax.plot(history['est_x'], history['est_y'], 'b--', linewidth=1.5, alpha=0.6, zorder=3)

            # 5. Best Estimated Robot Pose (Red Dot & Arrow)
            est_pose = fast_slam.best_particle.pose
            ax.plot(est_pose[0], est_pose[1], 'ro', markersize=6, zorder=7)
            ax.arrow(est_pose[0], est_pose[1], 0.2 * math.cos(est_pose[2]), 0.2 * math.sin(est_pose[2]), 
                     head_width=0.05, fc='r', ec='r', zorder=8)

            # 6. Navigation Paths & Targets
            if state['ctrl_mode'] == 'active_slam' and active_slam_ctrl.active:
                if active_slam_ctrl.current_path:
                    path_x = [p[0] for p in active_slam_ctrl.current_path]
                    path_y = [p[1] for p in active_slam_ctrl.current_path]
                    ax.plot(path_x, path_y, 'c-', linewidth=2, zorder=9)  # Cyan Path
                
                if active_slam_ctrl.target_frontier:
                    ax.plot(active_slam_ctrl.target_frontier[0], active_slam_ctrl.target_frontier[1], 
                            'm*', markersize=12, zorder=10) # Magenta Target

            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.5)
            # Dynamic camera framing: keep the robot roughly in the center
            ax.set_xlim(est_pose[0] - 2.5, est_pose[0] + 2.5)
            ax.set_ylim(est_pose[1] - 2.5, est_pose[1] + 2.5)
            ax.set_aspect('equal')

    # ======================================================================
    # GUI LAYOUT
    # ======================================================================
    with ui.header().classes(f'{HEADER_BG} shadow-md p-4 flex items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('smart_toy', size='32px', color='blue-400')
            ui.label('Robot Command Center').classes('text-xl font-bold tracking-wide text-white')
        with ui.row().classes('items-center gap-2 bg-slate-800 px-3 py-1 rounded-full'):
            status_indicator = ui.element('div').classes('w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]')
            status_label = ui.label('Disconnected').classes('text-xs font-semibold text-slate-300')

    with ui.column().classes('w-full p-6 gap-6 items-center max-w-7xl mx-auto'):
        with ui.grid(columns=3).classes('w-full gap-6'):
            with ui.card().classes(f'w-full {CARD_BG} p-0 overflow-hidden relative'):
                ui.label('Camera Feed').classes('absolute top-3 left-4 z-10 text-xs font-bold text-white/70 bg-black/50 px-2 py-1 rounded backdrop-blur-sm')
                if stream_video: video_image = ui.interactive_image('/video/frame').classes('w-full h-64 object-cover')
                else: video_image = None

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
                    ui.button('START TRIAL', on_click=run_trial).props('unelevated').classes('bg-blue-600 hover:bg-blue-500 text-white w-2/3 rounded-lg font-bold')
                    trial_timer_label = ui.label('0.0s').classes('text-slate-400 font-mono text-sm')

        with ui.card().classes(f'w-full {CARD_BG} p-6'):
            ui.label('Drive Control').classes('text-sm font-bold text-slate-400 mb-6 uppercase tracking-wider')
            with ui.grid(columns=2).classes('w-full gap-12'):
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('speed', color='blue-400'); ui.label('Speed').classes('text-lg font-medium text-white')
                        speed_switch = ui.switch().props('color=blue dense')
                    slider_speed = ui.slider(min=-100, max=100, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('directions_car', color='blue-400'); ui.label('Steering').classes('text-lg font-medium text-white')
                        steering_switch = ui.switch().props('color=blue dense')
                    slider_steering = ui.slider(min=-20, max=20, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')

        with ui.grid(columns=3).classes('w-full gap-6'):
            with ui.card().classes(f'w-full {CARD_BG} p-5') as pd_card:
                with ui.row().classes('items-center gap-2 mb-3'):
                    ui.icon('my_location', color='yellow-400')
                    ui.label('PD Controller').classes('text-sm font-bold text-slate-300 uppercase tracking-wider')
                    ui.space()
                    pd_status_label = ui.label('■ Idle').classes('text-xs font-mono text-slate-500')
                with ui.grid(columns=2).classes('w-full gap-3 mb-3'):
                    pd_goal_x = ui.number(value=0.5, step=0.05, label='Goal X').props('dense outlined dark').classes('w-full')
                    pd_goal_y = ui.number(value=0.5, step=0.05, label='Goal Y').props('dense outlined dark').classes('w-full')
                with ui.row().classes('w-full gap-3 mt-2'):
                    ui.button('▶ Go', on_click=start_pd_controller).classes('bg-yellow-600 text-black font-bold flex-1 rounded-lg')
                    ui.button('■ Stop', on_click=stop_pd_controller).classes('bg-slate-700 text-white rounded-lg px-4')

            with ui.card().classes(f'w-full {CARD_BG} p-5') as sl_card:
                with ui.row().classes('items-center gap-2 mb-3'):
                    ui.icon('arrow_forward', color='purple-400')
                    ui.label('Straight Line').classes('text-sm font-bold text-slate-300 uppercase tracking-wider')
                    ui.space()
                    sl_status_label = ui.label('■ Idle').classes('text-xs font-mono text-slate-500')
                sl_distance = ui.number(value=0.5, step=0.05, label='Distance (m)').props('dense outlined dark').classes('w-full mb-3')
                with ui.row().classes('w-full gap-3 mt-2'):
                    ui.button('▶ Drive', on_click=start_sl_controller).classes('bg-purple-600 text-white font-bold flex-1 rounded-lg')
                    ui.button('■ Stop', on_click=stop_sl_controller).classes('bg-slate-700 text-white rounded-lg px-4')

            with ui.card().classes(f'w-full {CARD_BG} p-5') as active_card:
                with ui.row().classes('items-center gap-2 mb-3'):
                    ui.icon('explore', color='green-400')
                    ui.label('Active SLAM').classes('text-sm font-bold text-slate-300 uppercase tracking-wider')
                    ui.space()
                    active_status_label = ui.label('■ Idle').classes('text-xs font-mono text-slate-500')
                ui.label('Autonomously maps room based on frontier detection.').classes('text-xs text-slate-400 mb-6')
                with ui.row().classes('w-full gap-3 mt-2'):
                    ui.button('▶ Explore', on_click=start_active_slam).classes('bg-green-600 text-white font-bold flex-1 rounded-lg')
                    ui.button('■ Stop', on_click=stop_active_slam).classes('bg-slate-700 text-white rounded-lg px-4')

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

        est_pose  = fast_slam.best_particle.pose
        cmd_speed = 0
        cmd_steer = 0

        if state['ctrl_mode'] == 'pd_position' and pd_ctrl.active:
            cmd_speed, cmd_steer, reached = pd_ctrl.compute(est_pose, dt_pf)
            if reached: stop_pd_controller()
        elif state['ctrl_mode'] == 'straight_line' and sl_ctrl.active:
            cmd_speed, cmd_steer, reached = sl_ctrl.compute(est_pose, dt_pf)
            if reached: stop_sl_controller()
        elif state['ctrl_mode'] == 'active_slam' and active_slam_ctrl.active:
            current_angles_rad = [-(a * math.pi / 180.0) for a in state['sensor_signal'].angles]
            current_dists_m = [d / 1000.0 for d in state['sensor_signal'].distances]
            cmd_speed, cmd_steer = active_slam_ctrl.update(est_pose, current_angles_rad, current_dists_m)
            if cmd_speed is None:
                ui.notify('Exploration Complete! Map fully mapped.', type='positive')
                stop_active_slam(); cmd_speed, cmd_steer = 0, 0
        else:
            if speed_switch.value:    cmd_speed = slider_speed.value
            if steering_switch.value: cmd_steer = slider_steering.value

        if stream_video and video_capture.isOpened():
            read_result = await run.io_bound(video_capture.read)
            if read_result is not None:
                ret, state['latest_frame'] = read_result
                if state['latest_frame'] is not None and video_image: video_image.force_reload()

        if state['connected']:
            state['sender'].send_control_signal([cmd_speed, cmd_steer])
            try:
                state['sensor_signal'] = state['receiver'].receive_robot_sensor_signal(state['sensor_signal'])
                encoder_count_label.set_text(str(state['sensor_signal'].encoder_counts))
            except socket.timeout: pass

            if dt_pf > 0:
                v_phys, delta_phys = get_physical_commands(cmd_speed, cmd_steer)
                fast_slam.predict(v_phys, delta_phys, dt_pf)

            sweep_complete = False
            for i in range(state['sensor_signal'].num_lidar_rays):
                ang = state['sensor_signal'].angles[i]
                dist = state['sensor_signal'].distances[i]
                if state['last_lidar_angle'] is not None and abs(ang - state['last_lidar_angle']) > 180:
                    sweep_complete = True
                state['sweep_angles'].append(ang); state['sweep_distances'].append(dist)
                state['last_lidar_angle'] = ang

            if sweep_complete:
                slam_angles_rad = [-(a * math.pi / 180.0) for a in state['sweep_angles']]
                slam_dists_m = [d / 1000.0 for d in state['sweep_distances']]
                valid_angles = []; valid_dists = []
                for a, d in zip(slam_angles_rad, slam_dists_m):
                    if 0.1 < d < 4.9:
                        valid_angles.append(a); valid_dists.append(d)
                
                fast_slam.update(valid_angles, valid_dists)
                state['sweep_angles'] = []; state['sweep_distances'] = []

                # Record pose history upon sweep completion
                history['est_x'].append(fast_slam.best_particle.pose[0])
                history['est_y'].append(fast_slam.best_particle.pose[1])

        show_localization_plot()

    ui.timer(0.1, control_loop)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title='Robot Dashboard', dark=True)