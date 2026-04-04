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
ROBOT_RADIUS = 0.15

LOOKAHEAD_DISTANCE = 0.4
GOAL_TOLERANCE     = 0.20
MAX_V_CMD          = 80.0
MAX_ALPHA_CMD      = 100.0

# ==========================================
# PROBABILITY TRUTH TABLE
# get_probabilities() = 1 / (1 + exp(grid)) = sigmoid(-grid)
#
#   grid =  0.0  (never visited)  -> prob = 0.500  -> GREY
#   grid = -5.0  (free space)     -> prob = 0.993  -> BLACK
#   grid = +5.0  (wall)           -> prob = 0.007  -> WHITE
#
#   is_free    = prob > 0.55
#   is_unknown = 0.45 <= prob <= 0.55
#   is_wall    = prob < 0.10
# ==========================================

PROB_FREE_THRESH  = 0.55
PROB_UNKNOWN_LOW  = 0.45
PROB_UNKNOWN_HIGH = 0.55
PROB_WALL_THRESH  = 0.10


# ==========================================
# MAP LOADING & MATH UTILS
# ==========================================

def get_naive_frontier_mask(mapper):
    prob_grid = mapper.get_probabilities()
    is_free    = prob_grid > PROB_FREE_THRESH
    is_unknown = (prob_grid >= PROB_UNKNOWN_LOW) & (prob_grid <= PROB_UNKNOWN_HIGH)
    is_wall    = prob_grid < PROB_WALL_THRESH
    has_unknown_neighbor = (
        np.roll(is_unknown, 1, axis=0) | np.roll(is_unknown, -1, axis=0) |
        np.roll(is_unknown, 1, axis=1) | np.roll(is_unknown, -1, axis=1)
    )
    wall_buffer = binary_dilation(is_wall, iterations=3)
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
                        self.grid[cx, cy] += L_FREE
                    self.grid[cx, cy] = np.clip(
                        self.grid[cx, cy], MIN_LOG_ODDS, MAX_LOG_ODDS)

    def get_probabilities(self):
        # sigmoid(-grid): free(neg grid)->high prob, wall(pos)->low, unknown(0)->0.5
        return 1.0 / (1.0 + np.exp(self.grid))


# ==========================================
# ACTIVE SLAM CONTROLLER
# ==========================================

class ActiveSLAMController:
    def __init__(self, mapper):
        self.mapper          = mapper
        self.current_path    = []
        self.target_frontier = None

        # ------------------------------------------------------------------
        # Low-level PD controller — same dual-loop design as the proven
        # hardware PDPositionController.
        #
        # STEERING LOOP  →  alpha_cmd
        #   Measured state : EKF heading  rtheta
        #   Desired state  : angle to current waypoint  atan2(dy, dx)
        #   error          : heading_err = desired - measured  [rad, wrapped]
        #   output         : KP_steer * heading_err + KD_steer * d(err)/dt
        #   clamp          : [-MAX_ALPHA_CMD, +MAX_ALPHA_CMD]
        #
        # SPEED LOOP  →  v_cmd
        #   Measured state : EKF position
        #   Desired state  : current waypoint position
        #   error          : dist = hypot(dx, dy)  [m]
        #   output         : KP_speed * dist
        #   clamp          : [MIN_V_CMD, MAX_V_CMD]
        #   dead-band      : dist < WP_REACH_DIST  → advance to next waypoint
        #
        # PHASE LOGIC  (align-first, identical to hardware):
        #   |heading_err| > ALIGN_THRESHOLD  →  v_cmd = 0  (rotate in place)
        #   |heading_err| ≤ ALIGN_THRESHOLD  →  both loops active
        #
        # Gains are scaled from the hardware values
        # (hardware steer range [-20,20]; here [-100,100] → ×5):
        #   KP_steer  hardware=8.0   →  here=40.0
        #   KD_steer  hardware=1.2   →  here=6.0
        #   KP_speed  hardware=60.0  →  here=60.0  (same speed range 0-80)
        # ------------------------------------------------------------------

        # Steering PD
        self.KP_steer = 40.0
        self.KD_steer =  6.0

        # Speed P
        self.KP_speed  = 60.0
        self.MIN_V_CMD = 100.0   # hard floor — robot moves at speed

        # Thresholds
        self.ALIGN_THRESHOLD = 0.20          # rad  — rotate-in-place until within this
        self.WP_REACH_DIST   = 0.30   # m — matches subsampled waypoint spacing

        # Derivative state — reset whenever goal changes
        self.prev_heading_err = 0.0

        # Stuck detection
        self.stuck_check_pose  = None   # [x, y] snapshot
        self.stuck_timer       = 0      # steps since last snapshot
        self.STUCK_CHECK_STEPS = 30     # check every 3 s (at 10 Hz)
        self.STUCK_DIST_MIN    = 0.05   # m — must have moved this far

    # ------------------------------------------------------------------
    def _reset_pd(self):
        self.prev_heading_err = 0.0
        self.stuck_check_pose = None
        self.stuck_timer      = 0

    # ------------------------------------------------------------------
    def _subsample_path(self, path, step_m=0.25):
        """
        Reduce waypoint density to one point every step_m metres.
        A* on a 5cm grid produces waypoints every 5-7cm — far too many.
        Keeping only every ~5th waypoint (0.25m spacing) means:
          - fewer align-then-drive cycles per trip  →  much faster overall
          - WP_REACH_DIST of 0.30m still comfortably catches each waypoint
        The final waypoint (frontier goal) is always preserved.
        """
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
            subsampled.append(path[-1])   # always keep the goal
        return subsampled

    # ------------------------------------------------------------------
    def get_inflated_obstacles(self):
        prob_grid   = self.mapper.get_probabilities()
        is_obstacle = prob_grid < PROB_WALL_THRESH      # walls = LOW prob
        inflation_cells = int(ROBOT_RADIUS / GRID_RESOLUTION)
        return binary_dilation(is_obstacle, iterations=inflation_cells)

    # ------------------------------------------------------------------
    def find_frontiers(self, inflated_obstacles):
        prob_grid  = self.mapper.get_probabilities()
        is_free    = prob_grid > PROB_FREE_THRESH
        is_unknown = ((prob_grid >= PROB_UNKNOWN_LOW) &
                      (prob_grid <= PROB_UNKNOWN_HIGH))
        unknown_expanded = maximum_filter(is_unknown, size=5)
        frontier_grid    = is_free & unknown_expanded & ~inflated_obstacles
        frontier_pixels  = np.argwhere(frontier_grid)

        if len(frontier_pixels) == 0:
            print("[Frontiers] No frontier pixels found.")
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

        print(f"[Frontiers] {len(candidates)} candidates "
              f"from {len(frontier_pixels)} raw pixels")
        return candidates

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def a_star_plan(self, start_pose, goal_world, inflated_obstacles):
        sgx, sgy = self.mapper.world_to_grid(start_pose[0], start_pose[1])
        ggx, ggy = self.mapper.world_to_grid(goal_world[0], goal_world[1])

        if not (0 <= ggx < self.mapper.W and 0 <= ggy < self.mapper.H):
            return []

        # Snap blocked goal to nearest free cell
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

        # Clear start bubble so robot can always leave
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

    # ------------------------------------------------------------------
    def pd_controller(self, robot_pose, dt=0.1):
        """
        Dual-loop PD controller for Ackermann drive.

        Key design decision: NO align-first / stop-to-turn logic.
        Align-first causes the robot to stop whenever heading_err drifts
        above threshold (which happens constantly mid-path on curves),
        leading to oscillation and getting stuck.

        Instead: always drive forward at MIN_V_CMD floor, steer
        simultaneously.  Speed scales up with distance to waypoint so
        the robot naturally slows as it approaches each one.

        Measured state : EKF pose [x, y, theta]
        Desired state  : current subsampled A* waypoint
        """
        if not self.current_path:
            return 0.0, 0.0

        rx, ry, rtheta = robot_pose

        # ── Stuck detection ──────────────────────────────────────────────
        self.stuck_timer += 1
        if self.stuck_check_pose is None:
            self.stuck_check_pose = [rx, ry]
        elif self.stuck_timer >= self.STUCK_CHECK_STEPS:
            moved = math.hypot(rx - self.stuck_check_pose[0],
                               ry - self.stuck_check_pose[1])
            if moved < self.STUCK_DIST_MIN:
                print("[PD] Stuck detected — clearing path to force replan.")
                self.current_path    = []
                self.target_frontier = None
                self._reset_pd()
                return 0.0, 0.0
            self.stuck_check_pose = [rx, ry]
            self.stuck_timer      = 0

        # ── Smart waypoint selection ─────────────────────────────────────
        # 1. Find the index of the closest waypoint on the entire remaining path.
        # 2. Discard everything before it — robot cannot go backwards.
        # 3. Then look one step ahead of that closest point so the robot
        #    always has forward progress as its target, not a point it
        #    may already be alongside or past.
        # This means the robot never gets stuck trying to reverse-arc back
        # to a waypoint it has already passed.
        dists = [math.hypot(p[0] - rx, p[1] - ry) for p in self.current_path]
        closest_idx = int(np.argmin(dists))

        # Prune everything before the closest point
        if closest_idx > 0:
            self.current_path = self.current_path[closest_idx:]

        # Look one waypoint ahead of closest so we always aim forward
        lookahead_idx = min(1, len(self.current_path) - 1)
        wp_x, wp_y = self.current_path[lookahead_idx]

        # ── STEERING LOOP ────────────────────────────────────────────────
        desired_heading = math.atan2(wp_y - ry, wp_x - rx)
        heading_err     = angle_wrap(desired_heading - rtheta)

        d_heading = (heading_err - self.prev_heading_err) / dt if dt > 0 else 0.0
        self.prev_heading_err = heading_err

        # Sign convention:
        # predict_next_pose: theta -= w*dt, w = v*tan(delta)/L
        # So positive delta → robot turns RIGHT (theta decreases)
        # positive heading_err → goal is to the LEFT → need negative delta
        # → negate the PD output so positive error → negative alpha_cmd
        alpha_cmd = float(np.clip(
            -(self.KP_steer * heading_err + self.KD_steer * d_heading),
            -MAX_ALPHA_CMD, MAX_ALPHA_CMD
        ))

        # ── SPEED LOOP ───────────────────────────────────────────────────
        # Scale speed with distance — far away: fast, close: slow.
        # Hard floor of MIN_V_CMD (40) so robot always moves forward.
        # No stop-to-turn: Ackermann steers by driving, not spinning.
        dist = math.hypot(wp_x - rx, wp_y - ry)
        v_cmd = float(np.clip(
            self.KP_speed * dist,
            self.MIN_V_CMD, MAX_V_CMD
        ))

        return v_cmd, alpha_cmd

    # ------------------------------------------------------------------
    def update(self, robot_pose):
        inflated_obstacles = self.get_inflated_obstacles()

        # Check arrival at current frontier
        if self.target_frontier is not None:
            dist_to_goal = math.hypot(
                self.target_frontier[0] - robot_pose[0],
                self.target_frontier[1] - robot_pose[1])
            if dist_to_goal < GOAL_TOLERANCE * 2:
                print(f"[Nav] Arrived at {self.target_frontier} — selecting next frontier.")
                self.target_frontier = None
                self.current_path    = []
                self._reset_pd()

        # Follow existing path
        if self.current_path and self.target_frontier is not None:
            return self.pd_controller(robot_pose)

        # Path ran out but not arrived — replan to same frontier
        if self.target_frontier is not None and not self.current_path:
            path = self.a_star_plan(robot_pose, self.target_frontier, inflated_obstacles)
            if path:
                self.current_path = self._subsample_path(path)
                self._reset_pd()
                print(f"[Nav] Replanned to existing frontier {self.target_frontier} "
                      f"({len(self.current_path)} waypoints after subsampling)")
                return self.pd_controller(robot_pose)
            else:
                print("[Nav] Frontier unreachable — picking new one.")
                self.target_frontier = None

        # Pick a new frontier
        candidates = self.find_frontiers(inflated_obstacles)
        if not candidates:
            print("Exploration Complete: No frontier pixels found.")
            return 0.0, 0.0

        feasible = [c for c in candidates
                    if self.is_kinematically_reachable(robot_pose, c)]
        if not feasible:
            print("[Warn] Relaxing kinematic constraint — using all candidates")
            feasible = candidates

        # Always target the furthest reachable frontier
        feasible_sorted = sorted(
            feasible,
            key=lambda c: math.hypot(c[0] - robot_pose[0], c[1] - robot_pose[1]),
            reverse=True)

        for goal in feasible_sorted:
            path = self.a_star_plan(robot_pose, goal, inflated_obstacles)
            if path:
                self.target_frontier = goal
                self.current_path    = self._subsample_path(path)
                self._reset_pd()
                dist = math.hypot(goal[0] - robot_pose[0], goal[1] - robot_pose[1])
                print(f"[Nav] New frontier {goal}, dist={dist:.2f}m, "
                      f"{len(self.current_path)} waypoints (subsampled from {len(path)})")
                return self.pd_controller(robot_pose)

        print(f"[Warn] {len(feasible_sorted)} candidates but none reachable via A*")
        return 0.0, 0.0


# ==========================================
# SENSOR SIMULATION & MAIN LOOP
# ==========================================

def simulate_lidar_scan(robot_x, robot_y, robot_theta, walls):
    num_rays  = 180
    max_range = 5.0
    sigma_z   = math.sqrt(VAR_LIDAR)
    angles, distances = [], []
    ray_angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    for rel_ang in ray_angles:
        glob_ang = robot_theta + rel_ang
        rx, ry   = math.cos(glob_ang), math.sin(glob_ang)
        min_dist = max_range
        for wall in walls:
            qx, qy, bx, by = wall
            sx, sy    = bx - qx, by - qy
            r_cross_s = rx * sy - ry * sx
            if abs(r_cross_s) > 1e-6:
                qpx, qpy = qx - robot_x, qy - robot_y
                t = (qpx * sy - qpy * sx) / r_cross_s
                u = (qpx * ry - qpy * rx) / r_cross_s
                if t > 0 and 0 <= u <= 1 and t < min_dist:
                    min_dist = t
        if min_dist < max_range:
            min_dist = max(0.0, min_dist + random.gauss(0, sigma_z))
        angles.append(rel_ang)
        distances.append(min_dist)
    return angles, distances


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

        # B. KINEMATICS
        v_phys, delta_phys = get_physical_commands(v_cmd, alpha_cmd)
        v_noisy = (v_phys + random.gauss(0, math.sqrt(VAR_V))
                   if v_phys != 0 else 0.0)
        d_noisy = (delta_phys + random.gauss(0, math.sqrt(VAR_DELTA))
                   if v_phys != 0 else 0.0)
        proposed = predict_next_pose(true_pose, v_noisy, d_noisy, delta_t)

        # Collision + wall sliding
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

        plt.pause(0.01)
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='simple')
    args = parser.parse_args()
    print(f"Auto-Navigating Map: {args.map}")
    run_sim(args.map)