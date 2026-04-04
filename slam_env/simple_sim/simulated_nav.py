import math
import random
import argparse
import importlib
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation, label, maximum_filter

# ==========================================
# HYPERPARAMETERS
# ==========================================
EKF_PROCESS_NOISE     = np.diag([0.01, 0.01, math.radians(1.0)])**2
EKF_MEASUREMENT_NOISE = np.diag([0.05, 0.05, math.radians(2.0)])**2

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

# PHYSICAL & COLLISION BOUNDS
ROBOT_RADIUS           = 0.15
PLANNER_WALL_CLEARANCE = 0.25  

LOOKAHEAD_DISTANCE = 0.4
GOAL_TOLERANCE     = 0.20
MAX_V_CMD          = 80.0
MAX_ALPHA_CMD      = 100.0

PROB_FREE_THRESH  = 0.55
PROB_UNKNOWN_LOW  = 0.45
PROB_UNKNOWN_HIGH = 0.55
PROB_WALL_THRESH  = 0.10

RENDER_SKIP = 5 

# ==========================================
# MAP LOADING & MATH UTILS
# ==========================================

def get_naive_frontier_mask(mapper):
    prob_grid  = mapper.get_probabilities()
    is_free    = prob_grid > PROB_FREE_THRESH
    is_unknown = (prob_grid >= PROB_UNKNOWN_LOW) & (prob_grid <= PROB_UNKNOWN_HIGH)
    is_wall    = prob_grid < 0.45
    has_unknown_neighbor = (
        np.roll(is_unknown, 1, axis=0) | np.roll(is_unknown, -1, axis=0) |
        np.roll(is_unknown, 1, axis=1) | np.roll(is_unknown, -1, axis=1)
    )
    wall_buffer = binary_dilation(is_wall, iterations=4)
    return is_free & has_unknown_neighbor & ~wall_buffer


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


def angle_wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def get_physical_commands(v_cmd, alpha_cmd):
    if v_cmd == 0.0:
        return 0.0, 0.0
    v_mag = (V_M * abs(v_cmd)) + V_C
    if v_mag < 0:
        v_mag = 0.0
    v_phys     = v_mag if v_cmd > 0 else -v_mag
    delta_phys = (DELTA_COEFFS[0] * (alpha_cmd ** 2)
                + DELTA_COEFFS[1] * alpha_cmd
                + DELTA_COEFFS[2])
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
        u = (((target_x - qx) * px + (target_y - qy) * py) / float(norm_sq)
             if norm_sq > 0 else 0)
        u  = max(min(u, 1.0), 0.0)
        cx = qx + u * px
        cy = qy + u * py
        if math.hypot(target_x - cx, target_y - cy) <= robot_radius:
            return True, wall
    return False, None


# ==========================================
# EKF & OCCUPANCY GRID
# ==========================================

class EKFPoseTracker:
    def __init__(self, initial_pose):
        self.mu    = np.array(initial_pose, dtype=float)
        self.Sigma = np.eye(3) * 0.001

    def predict(self, v_phys, delta_phys, dt):
        x, y, theta = self.mu
        self.mu = predict_next_pose(self.mu, v_phys, delta_phys, dt)
        G_t = np.array([
            [1.0, 0.0, -v_phys * math.sin(theta) * dt],
            [0.0, 1.0,  v_phys * math.cos(theta) * dt],
            [0.0, 0.0,  1.0]
        ])
        self.Sigma = G_t @ self.Sigma @ G_t.T + EKF_PROCESS_NOISE

    def update(self, z):
        H_t = np.eye(3)
        S   = H_t @ self.Sigma @ H_t.T + EKF_MEASUREMENT_NOISE
        K   = self.Sigma @ H_t.T @ np.linalg.inv(S)
        inn = z - self.mu
        inn[2] = angle_wrap(inn[2])
        self.mu    = self.mu + K @ inn
        self.mu[2] = angle_wrap(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H_t) @ self.Sigma


class GridMapper:
    def __init__(self, bounds):
        self.offset_x = bounds['min_x']
        self.offset_y = bounds['min_y']
        self.W = int((bounds['max_x'] - bounds['min_x']) / GRID_RESOLUTION)
        self.H = int((bounds['max_y'] - bounds['min_y']) / GRID_RESOLUTION)
        self.grid = np.full((self.W, self.H), L_0)

    def world_to_grid(self, x, y):
        return (int((x - self.offset_x) / GRID_RESOLUTION),
                int((y - self.offset_y) / GRID_RESOLUTION))

    def grid_to_world(self, gx, gy):
        return ((gx * GRID_RESOLUTION) + self.offset_x,
                (gy * GRID_RESOLUTION) + self.offset_y)

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

    def update_map(self, ego_pose, angles, distances, max_range=5.0):
        rx, ry, rtheta = ego_pose
        gx0, gy0 = self.world_to_grid(rx, ry)
        PERSIST_THRESH = 1.5

        for i in range(len(angles)):
            dist       = distances[i]
            glob_angle = rtheta + angles[i]
            end_x      = rx + dist * math.cos(glob_angle)
            end_y      = ry + dist * math.sin(glob_angle)
            gx1, gy1   = self.world_to_grid(end_x, end_y)
            cells      = self.bresenham_line(gx0, gy0, gx1, gy1)
            for j, (cx, cy) in enumerate(cells):
                if 0 <= cx < self.W and 0 <= cy < self.H:
                    if j == len(cells) - 1 and dist < (max_range - 0.1):
                        self.grid[cx, cy] += L_OCC
                    else:
                        if self.grid[cx, cy] < PERSIST_THRESH:
                            self.grid[cx, cy] += L_FREE
                    self.grid[cx, cy] = np.clip(
                        self.grid[cx, cy], MIN_LOG_ODDS, MAX_LOG_ODDS)

    def get_probabilities(self):
        return 1.0 / (1.0 + np.exp(self.grid))


# ==========================================
# ACTIVE SLAM CONTROLLER
# ==========================================

class ActiveSLAMController:
    def __init__(self, mapper):
        self.mapper          = mapper
        self.current_path    = []
        self.target_frontier = None
        self.step_counter    = 0
        self.cached_inflated = None

        self.blacklisted_frontiers = []
        self.recovery_steps        = 0

        self.KP_steer = 40.0
        self.KD_steer =  6.0
        self.KP_speed  = 100.0  
        self.MIN_V_CMD = 40.0   

        self.ALIGN_THRESHOLD = 0.20          
        self.WP_REACH_DIST   = 0.30   

        self.prev_heading_err = 0.0
        self.stuck_check_pose = None
        self.stuck_timer      = 0      
        self.STUCK_CHECK_STEPS = 30     
        self.STUCK_DIST_MIN    = 0.05   

    def _reset_pd(self):
        self.prev_heading_err = 0.0
        self.stuck_check_pose = None
        self.stuck_timer      = 0

    def _subsample_path(self, path, step_m=0.25):
        if len(path) <= 2:
            return path
        subsampled = [path[0]]
        accumulated = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            accumulated += math.hypot(dx, dy)
            if accumulated >= step_m:
                subsampled.append(path[i])
                accumulated = 0.0
        if subsampled[-1] != path[-1]:
            subsampled.append(path[-1])   
        return subsampled

    def get_inflated_obstacles(self):
        prob_grid = self.mapper.get_probabilities()
        confirmed_wall = prob_grid < 0.20
        probable_wall  = prob_grid < 0.40

        confirmed_inflation = int((ROBOT_RADIUS + PLANNER_WALL_CLEARANCE) / GRID_RESOLUTION)
        probable_inflation  = int(ROBOT_RADIUS / GRID_RESOLUTION)

        return (binary_dilation(confirmed_wall, iterations=confirmed_inflation) |
                binary_dilation(probable_wall,  iterations=probable_inflation))

    def find_frontiers(self, inflated_obstacles):
        prob_grid  = self.mapper.get_probabilities()
        is_free    = prob_grid > PROB_FREE_THRESH
        is_unknown = ((prob_grid >= PROB_UNKNOWN_LOW) &
                      (prob_grid <= PROB_UNKNOWN_HIGH))
        unknown_expanded = maximum_filter(is_unknown, size=5)
        frontier_grid    = is_free & unknown_expanded & ~inflated_obstacles
        frontier_pixels  = np.argwhere(frontier_grid)

        if len(frontier_pixels) == 0:
            return []

        sampled    = frontier_pixels[::8]
        candidates = []
        for px in sampled:
            gx, gy = int(px[0]), int(px[1])
            if not (0 <= gx < self.mapper.W and 0 <= gy < self.mapper.H):
                continue
            if inflated_obstacles[gx, gy]:
                continue
            wx, wy = self.mapper.grid_to_world(gx, gy)
            candidates.append((wx, wy))

        return candidates

    def is_kinematically_reachable(self, robot_pose, goal_pos):
        rx, ry, rtheta = robot_pose
        gx, gy = goal_pos
        dist = math.hypot(gx - rx, gy - ry)
        if dist < GOAL_TOLERANCE:
            return False
        angle_to_goal = math.atan2(gy - ry, gx - rx)
        heading_diff  = abs(angle_wrap(angle_to_goal - rtheta))
        if heading_diff > math.radians(150):
            return False
        delta_max  = abs(DELTA_COEFFS[1] * MAX_ALPHA_CMD + DELTA_COEFFS[2])
        min_radius = (L / math.tan(delta_max)
                      if delta_max > 1e-6 and math.tan(delta_max) > 1e-6
                      else 999.0)
        dx      = gx - rx
        dy      = gy - ry
        local_x =  dx * math.cos(rtheta) + dy * math.sin(rtheta)
        local_y = -dx * math.sin(rtheta) + dy * math.cos(rtheta)
        if abs(local_y) > 1e-6:
            req_r = (local_x ** 2 + local_y ** 2) / (2.0 * abs(local_y))
            if req_r < min_radius * 0.4:
                return False
        return True

    def a_star_plan(self, start_pose, goal_world, inflated_obstacles):
        sgx, sgy = self.mapper.world_to_grid(start_pose[0], start_pose[1])
        ggx, ggy = self.mapper.world_to_grid(goal_world[0], goal_world[1])

        if not (0 <= ggx < self.mapper.W and 0 <= ggy < self.mapper.H):
            return []

        if inflated_obstacles[ggx, ggy]:
            found = False
            for radius in range(1, 20):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = ggx + dx, ggy + dy
                        if (0 <= nx < self.mapper.W and
                                0 <= ny < self.mapper.H and
                                not inflated_obstacles[nx, ny]):
                            ggx, ggy = nx, ny
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return []

        safe = inflated_obstacles.copy()
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                nx, ny = sgx + dx, sgy + dy
                if 0 <= nx < self.mapper.W and 0 <= ny < self.mapper.H:
                    safe[nx, ny] = False

        open_set = []
        heapq.heappush(open_set, (0, (sgx, sgy)))
        came_from = {}
        g_score   = {(sgx, sgy): 0}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == (ggx, ggy):
                path = []
                while cur in came_from:
                    path.append(self.mapper.grid_to_world(cur[0], cur[1]))
                    cur = came_from[cur]
                return path[::-1]
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),
                           (1,1),(-1,-1),(1,-1),(-1,1)]:
                nb = (cur[0] + dx, cur[1] + dy)
                if (0 <= nb[0] < self.mapper.W and
                        0 <= nb[1] < self.mapper.H and
                        not safe[nb[0], nb[1]]):
                    tg = g_score[cur] + math.hypot(dx, dy)
                    if nb not in g_score or tg < g_score[nb]:
                        came_from[nb] = cur
                        g_score[nb]   = tg
                        f = tg + math.hypot(ggx - nb[0], ggy - nb[1])
                        heapq.heappush(open_set, (f, nb))
        return []

    def local_planner_check(self, robot_pose, inflated_obstacles):
        rx, ry, rtheta = robot_pose

        CHECK_WPS = min(8, len(self.current_path))
        for wp in self.current_path[:CHECK_WPS]:
            gx, gy = self.mapper.world_to_grid(wp[0], wp[1])
            if (0 <= gx < self.mapper.W and 0 <= gy < self.mapper.H
                    and inflated_obstacles[gx, gy]):
                self.current_path    = []
                self.target_frontier = None
                self._reset_pd()
                return True

        NUM_PROBE = 8
        for k in range(1, NUM_PROBE + 1):
            probe_dist = (LOOKAHEAD_DISTANCE / NUM_PROBE) * k
            px = rx + probe_dist * math.cos(rtheta)
            py = ry + probe_dist * math.sin(rtheta)
            gx, gy = self.mapper.world_to_grid(px, py)
            if (0 <= gx < self.mapper.W and 0 <= gy < self.mapper.H
                    and inflated_obstacles[gx, gy]):
                self.current_path    = []
                self.target_frontier = None
                self._reset_pd()
                return True

        return False

    def pd_controller(self, robot_pose, dt=0.1):
        if not self.current_path:
            return 0.0, 0.0

        rx, ry, rtheta = robot_pose

        self.stuck_timer += 1
        if self.stuck_check_pose is None:
            self.stuck_check_pose = [rx, ry]
        elif self.stuck_timer >= self.STUCK_CHECK_STEPS:
            moved = math.hypot(rx - self.stuck_check_pose[0],
                               ry - self.stuck_check_pose[1])
            
            if moved < self.STUCK_DIST_MIN:
                print(f"[PD] Stuck! Blacklisting frontier and reversing.")
                if self.target_frontier:
                    self.blacklisted_frontiers.append(self.target_frontier)
                
                self.current_path    = []
                self.target_frontier = None
                self._reset_pd()
                
                self.recovery_steps = 8 
                return -self.MIN_V_CMD, 0.0

            self.stuck_check_pose = [rx, ry]
            self.stuck_timer      = 0

        dists = [math.hypot(p[0] - rx, p[1] - ry) for p in self.current_path]
        closest_idx = int(np.argmin(dists))

        if closest_idx > 0:
            self.current_path = self.current_path[closest_idx:]

        lookahead_idx = min(1, len(self.current_path) - 1)
        wp_x, wp_y = self.current_path[lookahead_idx]

        desired_heading = math.atan2(wp_y - ry, wp_x - rx)
        heading_err     = angle_wrap(desired_heading - rtheta)

        d_heading = (heading_err - self.prev_heading_err) / dt if dt > 0 else 0.0
        self.prev_heading_err = heading_err

        alpha_cmd = float(np.clip(
            -(self.KP_steer * heading_err + self.KD_steer * d_heading),
            -MAX_ALPHA_CMD, MAX_ALPHA_CMD
        ))

        dist_to_final = math.hypot(self.current_path[-1][0] - rx, self.current_path[-1][1] - ry)
        base_v_cmd = MAX_V_CMD if dist_to_final > 0.5 else float(np.clip(self.KP_speed * dist_to_final, self.MIN_V_CMD, MAX_V_CMD))
        
        if abs(heading_err) > 0.35:
            v_cmd = max(self.MIN_V_CMD, base_v_cmd * 0.5)
        else:
            v_cmd = base_v_cmd

        return v_cmd, alpha_cmd

    def update(self, robot_pose):
        self.step_counter += 1
        
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            return -self.MIN_V_CMD, 0.0

        if self.cached_inflated is None or self.step_counter % 5 == 0:
            self.cached_inflated = self.get_inflated_obstacles()
        inflated_obstacles = self.cached_inflated

        if self.target_frontier is not None:
            dist_to_goal = math.hypot(
                self.target_frontier[0] - robot_pose[0],
                self.target_frontier[1] - robot_pose[1])
            if dist_to_goal < GOAL_TOLERANCE * 2:
                if self.blacklisted_frontiers:
                    print("[Nav] Reached goal, clearing frontier blacklist.")
                    self.blacklisted_frontiers.clear()
                self.target_frontier = None
                self.current_path    = []
                self._reset_pd()

        if self.current_path and self.target_frontier is not None:
            if self.local_planner_check(robot_pose, inflated_obstacles):
                pass  
            else:
                return self.pd_controller(robot_pose)

        if self.target_frontier is not None and not self.current_path:
            path = self.a_star_plan(robot_pose, self.target_frontier, inflated_obstacles)
            if path:
                self.current_path = self._subsample_path(path)
                self._reset_pd()
                return self.pd_controller(robot_pose)
            else:
                self.target_frontier = None

        candidates = self.find_frontiers(inflated_obstacles)
        
        # ==============================================================
        # FIX: STARTUP GUARD
        # Prevents the robot from instantly terminating at tick 0 before 
        # the LiDAR has painted the initial map into the grid.
        # ==============================================================
        if not candidates:
            if self.step_counter < 30: # 3 seconds at 10Hz
                return self.MIN_V_CMD, 0.0 # Drive straight slightly to see
            else:
                return None, None # Truly finished

        feasible = []
        for c in candidates:
            is_blacklisted = any(math.hypot(c[0]-b[0], c[1]-b[1]) < 1.0 for b in self.blacklisted_frontiers)
            if is_blacklisted:
                continue
            if self.is_kinematically_reachable(robot_pose, c):
                feasible.append(c)

        # ==============================================================
        # FIX: BLACKLIST FORGIVENESS
        # If the only frontiers left are blacklisted, wipe the slate clean
        # and try them again as a final attempt to finish the map.
        # ==============================================================
        if not feasible and candidates and self.blacklisted_frontiers:
            print("[Nav] All remaining frontiers blacklisted. Clearing blacklist for final sweep.")
            self.blacklisted_frontiers.clear()
            feasible = [c for c in candidates if self.is_kinematically_reachable(robot_pose, c)]

        if not feasible:
            feasible = candidates

        def frontier_cost(c):
            dist = math.hypot(c[0] - robot_pose[0], c[1] - robot_pose[1])
            angle_to_goal = math.atan2(c[1] - robot_pose[1], c[0] - robot_pose[0])
            heading_diff  = abs(angle_wrap(angle_to_goal - robot_pose[2]))
            heading_penalty = 1.5 * heading_diff 
            return dist + heading_penalty

        feasible_sorted = sorted(feasible, key=frontier_cost)

        for goal in feasible_sorted:
            path = self.a_star_plan(robot_pose, goal, inflated_obstacles)
            if path:
                self.target_frontier = goal
                self.current_path    = self._subsample_path(path)
                self._reset_pd()
                return self.pd_controller(robot_pose)

        return 0.0, 0.0


# ==========================================
# VECTORIZED SENSOR SIMULATION
# ==========================================

def simulate_lidar_scan(robot_x, robot_y, robot_theta, walls):
    num_rays  = 180
    max_range = 5.0
    sigma_z   = math.sqrt(VAR_LIDAR)
    
    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    glob_angles = robot_theta + angles
    rx = np.cos(glob_angles)
    ry = np.sin(glob_angles)
    
    if not walls:
        return angles.tolist(), np.clip(np.full(num_rays, max_range) + np.random.normal(0, sigma_z, num_rays), 0, max_range).tolist()
        
    walls_arr = np.array(walls)
    qx, qy, bx, by = walls_arr.T
    sx, sy = bx - qx, by - qy
    
    r_cross_s = np.outer(rx, sy) - np.outer(ry, sx)
    valid = np.abs(r_cross_s) > 1e-6
    
    qpx = qx - robot_x
    qpy = qy - robot_y
    
    t_num = np.outer(np.ones(num_rays), qpx * sy - qpy * sx)
    t = np.divide(t_num, r_cross_s, out=np.inf * np.ones_like(r_cross_s), where=valid)
    
    u_num = qpx * ry[:, np.newaxis] - qpy * rx[:, np.newaxis]
    u = np.divide(u_num, r_cross_s, out=np.inf * np.ones_like(r_cross_s), where=valid)
    
    hit = valid & (t > 0) & (u >= 0) & (u <= 1)
    t_hit = np.where(hit, t, np.inf)
    min_t = np.min(t_hit, axis=1)
    
    distances = np.minimum(min_t, max_range)
    hit_mask = distances < max_range
    distances[hit_mask] += np.random.normal(0, sigma_z, np.sum(hit_mask))
    distances = np.clip(distances, 0, max_range)
    
    return angles.tolist(), distances.tolist()


def run_sim(map_name):
    walls, start_pose, bounds = load_map(map_name)
    delta_t   = 0.1
    true_pose = np.array(start_pose)

    ekf           = EKFPoseTracker(true_pose)
    mapper        = GridMapper(bounds)
    ai_controller = ActiveSLAMController(mapper)

    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig1.canvas.manager.set_window_title(f'Ground Truth [{map_name}]')
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.canvas.manager.set_window_title('Active SLAM & Autonomous Navigation')

    cmap = LinearSegmentedColormap.from_list(
        'grid_map', ['white', 'lightgrey', 'black'])
    ekf_history_x, ekf_history_y = [ekf.mu[0]], [ekf.mu[1]]

    step = 0
    while plt.fignum_exists(fig1.number) and plt.fignum_exists(fig2.number):

        # A. AI DECISION
        v_cmd, alpha_cmd = ai_controller.update(ekf.mu)
        
        if v_cmd is None:
            print("\n=======================================================")
            print(" EXPLORATION COMPLETE! ")
            print(" No frontier points left. Close the plot windows to exit.")
            print("=======================================================\n")
            break 

        # B. KINEMATICS
        v_phys, delta_phys = get_physical_commands(v_cmd, alpha_cmd)
        v_noisy = (v_phys + random.gauss(0, math.sqrt(VAR_V))
                   if v_phys != 0 else 0.0)
        d_noisy = (delta_phys + random.gauss(0, math.sqrt(VAR_DELTA))
                   if v_phys != 0 else 0.0)
        proposed = predict_next_pose(true_pose, v_noisy, d_noisy, delta_t)

        crashed, hit_wall = get_collision_info(
            proposed[0], proposed[1], walls, ROBOT_RADIUS)
        if crashed:
            qx, qy, bx, by = hit_wall
            wall_vec = np.array([bx - qx, by - qy])
            if np.linalg.norm(wall_vec) > 0:
                tangent  = wall_vec / np.linalg.norm(wall_vec)
                disp     = proposed[:2] - true_pose[:2]
                slide    = np.dot(disp, tangent) * tangent
                proposed[0] = true_pose[0] + slide[0]
                proposed[1] = true_pose[1] + slide[1]
                still, _ = get_collision_info(
                    proposed[0], proposed[1], walls, ROBOT_RADIUS)
                if still:
                    proposed = true_pose.copy()
            else:
                proposed = true_pose.copy()
        true_pose = proposed

        # C. SENSORS & EKF
        angles, distances = simulate_lidar_scan(
            true_pose[0], true_pose[1], true_pose[2], walls)
        ekf.predict(v_phys, delta_phys, delta_t)
        z = true_pose + np.random.multivariate_normal(
            [0, 0, 0], EKF_MEASUREMENT_NOISE)
        ekf.update(z)
        mapper.update_map(ekf.mu, angles, distances)
        ekf_history_x.append(ekf.mu[0])
        ekf_history_y.append(ekf.mu[1])

        if step % RENDER_SKIP == 0:
            # D. GROUND TRUTH WINDOW
            ax1.clear()
            for wall in walls:
                ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
            ax1.plot(true_pose[0], true_pose[1], 'go', markersize=8)
            ax1.arrow(true_pose[0], true_pose[1],
                      0.2 * math.cos(true_pose[2]), 0.2 * math.sin(true_pose[2]),
                      head_width=0.05, fc='g')
            ax1.set_title(f"Ground Truth | Step: {step}")
            ax1.set_xlim(bounds['min_x'], bounds['max_x'])
            ax1.set_ylim(bounds['min_y'], bounds['max_y'])
            ax1.grid(True, linestyle='--', alpha=0.3)

            # E. SLAM BELIEF WINDOW
            ax2.clear()
            prob_grid = mapper.get_probabilities()
            ax2.imshow(prob_grid.T, cmap=cmap, origin='lower',
                       extent=[bounds['min_x'], bounds['max_x'],
                               bounds['min_y'], bounds['max_y']],
                       vmin=0, vmax=1)

            frontier_mask = get_naive_frontier_mask(mapper)
            overlay = np.zeros((frontier_mask.shape[0], frontier_mask.shape[1], 4))
            overlay[frontier_mask] = [1, 0, 1, 0.6]
            ax2.imshow(overlay.swapaxes(0, 1), origin='lower',
                       extent=[bounds['min_x'], bounds['max_x'],
                               bounds['min_y'], bounds['max_y']])

            ax2.plot(ekf_history_x, ekf_history_y, 'b--', linewidth=1, alpha=0.5)
            ax2.plot(ekf.mu[0], ekf.mu[1], 'ro', markersize=6)
            ax2.arrow(ekf.mu[0], ekf.mu[1],
                      0.2 * math.cos(ekf.mu[2]), 0.2 * math.sin(ekf.mu[2]),
                      head_width=0.05, fc='r', ec='r')

            if ai_controller.current_path:
                px = [p[0] for p in ai_controller.current_path]
                py = [p[1] for p in ai_controller.current_path]
                ax2.plot(px, py, 'c-', linewidth=2)

            if ai_controller.target_frontier:
                ax2.plot(ai_controller.target_frontier[0],
                         ai_controller.target_frontier[1],
                         'm*', markersize=12)

            ax2.set_title("Active SLAM Belief | Magenta = Frontier")
            ax2.set_xlim(bounds['min_x'], bounds['max_x'])
            ax2.set_ylim(bounds['min_y'], bounds['max_y'])

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