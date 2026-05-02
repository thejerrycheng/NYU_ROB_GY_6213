import math
import random
import argparse
import importlib
import heapq
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation, maximum_filter

# ==========================================
# HYPERPARAMETERS
# ==========================================
NUM_PARTICLES = 50  
PF_RESAMPLE_THRESHOLD = NUM_PARTICLES / 2.0

GRID_RESOLUTION = 0.05
L_0             = 0.0
L_OCC           = 0.85
L_FREE          = -0.4
MAX_LOG_ODDS    = 5.0
MIN_LOG_ODDS    = -5.0

L            = 0.145
V_M          = 0.004808
V_C          = -0.045557
VAR_V        = 0.00057829
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA    = 0.00023134
VAR_LIDAR    = 0.000363

ROBOT_RADIUS           = 0.15
PLANNER_WALL_CLEARANCE = 0.30  

LOOKAHEAD_DISTANCE = 0.4
GOAL_TOLERANCE     = 0.20
MAX_V_CMD          = 80.0
MAX_ALPHA_CMD      = 100.0

PROB_FREE_THRESH  = 0.55
PROB_UNKNOWN_LOW  = 0.45
PROB_UNKNOWN_HIGH = 0.55
PROB_WALL_THRESH  = 0.10

RENDER_SKIP = 5 
SLAM_SKIP   = 5   

# ==========================================
# UTILITIES & VISUALIZATION
# ==========================================

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
    return (angle + math.pi) % (2 * math.pi) - math.pi

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

def get_collision_info(target_x, target_y, walls, robot_radius):
    for wall in walls:
        qx, qy, bx, by = wall
        px, py  = bx - qx, by - qy
        norm_sq = px * px + py * py
        u = (((target_x - qx) * px + (target_y - qy) * py) / float(norm_sq) if norm_sq > 0 else 0)
        u  = max(min(u, 1.0), 0.0)
        cx, cy = qx + u * px, qy + u * py
        if math.hypot(target_x - cx, target_y - cy) <= robot_radius:
            return True, wall
    return False, None


# ==========================================
# GMAPPING: RAO-BLACKWELLIZED PARTICLE FILTER
# ==========================================

class Particle:
    def __init__(self, initial_pose, grid_shape):
        self.pose = np.array(initial_pose, dtype=float)
        self.weight = 1.0 / NUM_PARTICLES
        self.grid = np.full(grid_shape, L_0)

    def clone(self):
        new_p = Particle(self.pose, self.grid.shape)
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
        
        self.particles = [Particle(initial_pose, (self.W, self.H)) for _ in range(NUM_PARTICLES)]
        self.best_particle = self.particles[0]
        
        # FIX: Adaptive Resampling Counters
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
        
        # Approximate rotational velocity
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
            (0.0, 0.0, 0.10), (0.0, 0.0, -0.10),
            (0.05, 0.05, 0.0), (-0.05, -0.05, 0.0),
            (0.02, 0.0, 0.0), (-0.02, 0.0, 0.0),   
            (0.0, 0.02, 0.0), (0.0, -0.02, 0.0)    
        ]
        
        for p in self.particles:
            best_score = -999999.0
            best_pose = p.pose
            
            # Pre-compute the continuous probability field for this particle
            # This is significantly faster than calculating exp() for every cell individually
            prob_grid = 1.0 / (1.0 + np.exp(-p.grid))
            
            for dx, dy, dtheta in search_offsets:
                test_pose = np.array([
                    p.pose[0] + dx * math.cos(p.pose[2]) - dy * math.sin(p.pose[2]),
                    p.pose[1] + dx * math.sin(p.pose[2]) + dy * math.cos(p.pose[2]),
                    angle_wrap(p.pose[2] + dtheta)
                ])
                
                score = 0.0
                spatial_penalty = (math.hypot(dx, dy) * 40.0) + (abs(dtheta) * 20.0)
                score -= spatial_penalty
                
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
                        val = prob_grid[gx, gy]
                        score += (val - 0.5) * 10.0
                        valid_hits += 1
                
                if valid_hits < 5 and dx != 0.0:
                    continue
                            
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

        # --- B. ADAPTIVE RESAMPLING ---
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

        # --- C. MAP UPDATE ---
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
        return 1.0 / (1.0 + np.exp(self.best_particle.grid))

# ==========================================
# ACTIVE SLAM CONTROLLER
# ==========================================

class ActiveSLAMController:
    def __init__(self, slam):
        self.slam            = slam
        self.current_path    = []
        self.target_frontier = None
        self.step_counter    = 0
        self.cached_inflated = None
        self.blacklisted_frontiers = []
        self.recovery_steps  = 0

        self.KP_steer = 40.0; self.KD_steer = 6.0
        self.KP_speed = 100.0; self.MIN_V_CMD = 40.0   
        self.GOAL_TOLERANCE = 0.20   
        self.STUCK_CHECK_STEPS = 30; self.STUCK_DIST_MIN = 0.05   
        self.stuck_check_pose = None; self.stuck_timer = 0; self.prev_heading_err = 0.0
        
        self.EMERGENCY_BRAKE_DIST = 0.22

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
        delta_max  = abs(DELTA_COEFFS[1] * MAX_ALPHA_CMD + DELTA_COEFFS[2])
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
        """RESTORED: Checks if dynamic mapping has placed an obstacle on our current A* path"""
        if not self.current_path: return False
        check_wps = min(8, len(self.current_path))
        for wp in self.current_path[:check_wps]:
            gx, gy = self.slam.world_to_grid(wp[0], wp[1])
            if 0 <= gx < self.slam.W and 0 <= gy < self.slam.H and inflated_obstacles[gx, gy]:
                print("[Planner] Obstacle detected on path. Replanning.")
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
                print("[PD] Stuck! Blacklisting and reversing.")
                if self.target_frontier: self.blacklisted_frontiers.append(self.target_frontier)
                self.current_path = []; self.target_frontier = None; self._reset_pd()
                self.recovery_steps = 10; return -self.MIN_V_CMD, 0.0
            self.stuck_check_pose = [rx, ry]; self.stuck_timer = 0

        # Discard passed waypoints
        dists = [math.hypot(p[0] - rx, p[1] - ry) for p in self.current_path]
        closest_idx = int(np.argmin(dists))
        if closest_idx > 0: self.current_path = self.current_path[closest_idx:]
        
        # TRUE PURE PURSUIT LOOKAHEAD FIX
        # Find the first waypoint that is at least LOOKAHEAD_DISTANCE away
        lookahead_idx = 0
        for i in range(len(self.current_path)):
            if math.hypot(self.current_path[i][0] - rx, self.current_path[i][1] - ry) > LOOKAHEAD_DISTANCE:
                lookahead_idx = i
                break
        
        # If all points are closer than the lookahead, just aim for the very last one
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

        alpha_cmd = float(np.clip(-(self.KP_steer * blended_heading_err + self.KD_steer * d_heading), -MAX_ALPHA_CMD, MAX_ALPHA_CMD))
        dist_to_final = math.hypot(self.current_path[-1][0] - rx, self.current_path[-1][1] - ry)
        base_v_cmd = MAX_V_CMD if dist_to_final > 0.5 else float(np.clip(self.KP_speed * dist_to_final, self.MIN_V_CMD, MAX_V_CMD))
        
        # Braking logic
        v_cmd = max(self.MIN_V_CMD, base_v_cmd * 0.4) if abs(blended_heading_err) > 0.4 else base_v_cmd
        
        return v_cmd, alpha_cmd

    def update(self, robot_pose, lidar_angles, lidar_distances):
        self.step_counter += 1
        
        if lidar_distances and self.recovery_steps == 0:
            for ang, dist in zip(lidar_angles, lidar_distances):
                if abs(angle_wrap(ang)) < math.radians(35) and dist < self.EMERGENCY_BRAKE_DIST:
                    print("[Safety] Virtual Bumper! Reversing.")
                    if self.target_frontier: self.blacklisted_frontiers.append(self.target_frontier)
                    self.current_path = []; self.target_frontier = None; self._reset_pd()
                    self.recovery_steps = 10; break

        if self.recovery_steps > 0:
            self.recovery_steps -= 1; return -self.MIN_V_CMD, 0.0

        if self.cached_inflated is None or self.step_counter % SLAM_SKIP == 0:
            self.cached_inflated = self.get_inflated_obstacles()
        inflated_obstacles = self.cached_inflated

        # RESTORED LOCAL PLANNER CHECK
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

# ==========================================
# VECTORIZED SENSOR SIMULATION
# ==========================================

def simulate_lidar_scan(robot_x, robot_y, robot_theta, walls):
    num_rays  = 180; max_range = 5.0; sigma_z = math.sqrt(VAR_LIDAR)
    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    glob_angles = robot_theta + angles
    rx, ry = np.cos(glob_angles), np.sin(glob_angles)
    
    if not walls: return angles.tolist(), np.clip(np.full(num_rays, max_range) + np.random.normal(0, sigma_z, num_rays), 0, max_range).tolist()
        
    walls_arr = np.array(walls); qx, qy, bx, by = walls_arr.T
    sx, sy = bx - qx, by - qy
    r_cross_s = np.outer(rx, sy) - np.outer(ry, sx); valid = np.abs(r_cross_s) > 1e-6
    qpx, qpy = qx - robot_x, qy - robot_y
    t = np.divide(np.outer(np.ones(num_rays), qpx * sy - qpy * sx), r_cross_s, out=np.inf * np.ones_like(r_cross_s), where=valid)
    u = np.divide(qpx * ry[:, np.newaxis] - qpy * rx[:, np.newaxis], r_cross_s, out=np.inf * np.ones_like(r_cross_s), where=valid)
    
    hit = valid & (t > 0) & (u >= 0) & (u <= 1)
    min_t = np.min(np.where(hit, t, np.inf), axis=1)
    distances = np.minimum(min_t, max_range)
    hit_mask = distances < max_range
    distances[hit_mask] += np.random.normal(0, sigma_z, np.sum(hit_mask))
    return angles.tolist(), np.clip(distances, 0, max_range).tolist()

# ==========================================
# MAP LOADING & MAIN EXECUTION
# ==========================================

def load_map(map_name):
    try:
        map_module = importlib.import_module(f"maps.{map_name}")
        walls      = map_module.wall_corner_list
        start_pose = getattr(map_module, "start_pose", [0.0, 0.0, 0.0])
        all_x = [w[0] for w in walls] + [w[2] for w in walls]
        all_y = [w[1] for w in walls] + [w[3] for w in walls]
        bounds = {
            'min_x': min(all_x) - 1.5, 'max_x': max(all_x) + 1.5,
            'min_y': min(all_y) - 1.5, 'max_y': max(all_y) + 1.5,
        }
        return walls, start_pose, bounds
    except ModuleNotFoundError:
        print(f"Error: Map '{map_name}' not found.")
        exit(1)


def run_sim(map_name):
    walls, start_pose, bounds = load_map(map_name)
    delta_t   = 0.1
    true_pose = np.array(start_pose)

    slam = FastSLAM(true_pose, bounds)
    ai_controller = ActiveSLAMController(slam)

    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig1.canvas.manager.set_window_title(f'Ground Truth [{map_name}]')
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.canvas.manager.set_window_title('Gmapping: FastSLAM Exploration')

    cmap = LinearSegmentedColormap.from_list('grid_map', ['white', 'lightgrey', 'black'])
    history_x, history_y = [], []

    angles, distances = simulate_lidar_scan(true_pose[0], true_pose[1], true_pose[2], walls)

    step = 0
    while plt.fignum_exists(fig1.number) and plt.fignum_exists(fig2.number):

        best_pose = slam.best_particle.pose
        v_cmd, alpha_cmd = ai_controller.update(best_pose, angles, distances)
        
        if v_cmd is None:
            print("\n=======================================================")
            print(" EXPLORATION COMPLETE! ")
            print("=======================================================\n")
            break 

        v_phys, delta_phys = get_physical_commands(v_cmd, alpha_cmd)
        v_noisy = (v_phys + random.gauss(0, math.sqrt(VAR_V)) if v_phys != 0 else 0.0)
        d_noisy = (delta_phys + random.gauss(0, math.sqrt(VAR_DELTA)) if v_phys != 0 else 0.0)
        proposed = predict_next_pose(true_pose, v_noisy, d_noisy, delta_t)

        crashed, hit_wall = get_collision_info(proposed[0], proposed[1], walls, ROBOT_RADIUS)
        if crashed:
            qx, qy, bx, by = hit_wall
            wall_vec = np.array([bx - qx, by - qy])
            if np.linalg.norm(wall_vec) > 0:
                tangent  = wall_vec / np.linalg.norm(wall_vec)
                disp     = proposed[:2] - true_pose[:2]
                slide    = np.dot(disp, tangent) * tangent
                proposed[0] = true_pose[0] + slide[0]; proposed[1] = true_pose[1] + slide[1]
                still, _ = get_collision_info(proposed[0], proposed[1], walls, ROBOT_RADIUS)
                if still: proposed = true_pose.copy()
            else: proposed = true_pose.copy()
        true_pose = proposed

        angles, distances = simulate_lidar_scan(true_pose[0], true_pose[1], true_pose[2], walls)
        
        slam.predict(v_phys, delta_phys, delta_t)
        
        if step % SLAM_SKIP == 0:
            slam.update(angles, distances)
        
        best_pose = slam.best_particle.pose
        history_x.append(best_pose[0]); history_y.append(best_pose[1])

        if step % RENDER_SKIP == 0:
            ax1.clear()
            for wall in walls: ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
            ax1.plot(true_pose[0], true_pose[1], 'go', markersize=8)
            ax1.arrow(true_pose[0], true_pose[1], 0.2 * math.cos(true_pose[2]), 0.2 * math.sin(true_pose[2]), head_width=0.05, fc='g')
            ax1.set_xlim(bounds['min_x'], bounds['max_x']); ax1.set_ylim(bounds['min_y'], bounds['max_y'])
            ax1.grid(True, linestyle='--', alpha=0.3)

            ax2.clear()
            prob_grid = slam.get_best_probabilities()
            ax2.imshow(prob_grid.T, cmap=cmap, origin='lower', extent=[bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']], vmin=0, vmax=1)

            frontier_mask = get_naive_frontier_mask(prob_grid)
            overlay = np.zeros((frontier_mask.shape[0], frontier_mask.shape[1], 4))
            overlay[frontier_mask] = [1, 0, 1, 0.6]
            ax2.imshow(overlay.swapaxes(0, 1), origin='lower', extent=[bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']])

            px = [p.pose[0] for p in slam.particles]
            py = [p.pose[1] for p in slam.particles]
            ax2.plot(px, py, 'y.', markersize=4, alpha=0.5, label='Particle Swarm')

            ax2.plot(history_x, history_y, 'b--', linewidth=1, alpha=0.5)
            ax2.plot(best_pose[0], best_pose[1], 'ro', markersize=6)
            ax2.arrow(best_pose[0], best_pose[1], 0.2 * math.cos(best_pose[2]), 0.2 * math.sin(best_pose[2]), head_width=0.05, fc='r', ec='r')

            if ai_controller.current_path:
                path_x = [p[0] for p in ai_controller.current_path]
                path_y = [p[1] for p in ai_controller.current_path]
                ax2.plot(path_x, path_y, 'c-', linewidth=2)

            if ai_controller.target_frontier:
                ax2.plot(ai_controller.target_frontier[0], ai_controller.target_frontier[1], 'm*', markersize=12)

            ax2.set_title("Gmapping FastSLAM | Magenta = Frontier")
            ax2.set_xlim(bounds['min_x'], bounds['max_x']); ax2.set_ylim(bounds['min_y'], bounds['max_y'])
            plt.pause(0.001)

        step += 1

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='simple')
    args = parser.parse_args()
    print(f"Auto-Navigating Map: {args.map}")
    run_sim(args.map)