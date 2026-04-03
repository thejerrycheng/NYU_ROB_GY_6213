import math
import random
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 🟢 SLAM & KINEMATIC HYPERPARAMETERS
# ==========================================
# EKF Covariances
EKF_PROCESS_NOISE = np.diag([0.01, 0.01, math.radians(1.0)])**2 
EKF_MEASUREMENT_NOISE = np.diag([0.05, 0.05, math.radians(2.0)])**2 

# Occupancy Grid Parameters
GRID_RESOLUTION = 0.05     
L_0 = 0.0                  
L_OCC = 0.85               
L_FREE = -0.4              
MAX_LOG_ODDS = 5.0
MIN_LOG_ODDS = -5.0

# Calibrated Constants from Ackermann model
L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.00057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134
VAR_LIDAR = 0.000363

# Global Teleop State
current_v_cmd = 0.0
current_alpha_cmd = 0.0
V_STEP = 5.0          
ALPHA_STEP = 5.0      
MAX_V_CMD = 100.0     
MAX_ALPHA_CMD = 100.0

# ==========================================
# MAP LOADING LOGIC
# ==========================================
def load_map(map_name):
    try:
        map_module = importlib.import_module(f"maps.{map_name}")
        walls = map_module.wall_corner_list
        start_pose = getattr(map_module, "start_pose", [0.0, 0.0, 0.0])
        
        # Calculate dynamic bounds for grid and plotting
        all_x = [w[0] for w in walls] + [w[2] for w in walls]
        all_y = [w[1] for w in walls] + [w[3] for w in walls]
        
        # Add 1.5m padding around the map
        bounds = {
            'min_x': min(all_x) - 1.5,
            'max_x': max(all_x) + 1.5,
            'min_y': min(all_y) - 1.5,
            'max_y': max(all_y) + 1.5,
        }
        return walls, start_pose, bounds
    except ModuleNotFoundError:
        print(f"Error: Map '{map_name}' not found in maps/ directory.")
        exit(1)

# ==========================================
# 1. MOTION MODEL 
# ==========================================
def angle_wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_physical_commands(v_cmd, alpha_cmd):
    if v_cmd == 0.0:
        v_phys = 0.0
    else:
        v_mag = (V_M * abs(v_cmd)) + V_C
        if v_mag < 0: v_mag = 0.0 
        v_phys = v_mag if v_cmd > 0 else -v_mag
    delta_phys = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
    return v_phys, delta_phys

def predict_next_pose(current_pose, v_phys, delta_phys, delta_t=0.1):
    x, y, theta = current_pose
    w = (v_phys * math.tan(delta_phys)) / L if L > 0 else 0.0
    next_x = x + (v_phys * math.cos(theta) * delta_t)
    next_y = y + (v_phys * math.sin(theta) * delta_t)
    next_theta = angle_wrap(theta - (w * delta_t))
    return np.array([next_x, next_y, next_theta])


# ==========================================
# 2. EXTENDED KALMAN FILTER
# ==========================================
class EKFPoseTracker:
    def __init__(self, initial_pose):
        self.mu = np.array(initial_pose, dtype=float)
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
        S = H_t @ self.Sigma @ H_t.T + EKF_MEASUREMENT_NOISE
        K = self.Sigma @ H_t.T @ np.linalg.inv(S)
        
        innovation = z - self.mu
        innovation[2] = angle_wrap(innovation[2]) 
        
        self.mu = self.mu + K @ innovation
        self.mu[2] = angle_wrap(self.mu[2])
        self.Sigma = (np.eye(3) - K @ H_t) @ self.Sigma


# ==========================================
# 3. DYNAMIC OCCUPANCY GRID MAPPER
# ==========================================
class GridMapper:
    def __init__(self, bounds):
        self.offset_x = bounds['min_x']
        self.offset_y = bounds['min_y']
        
        size_x = bounds['max_x'] - bounds['min_x']
        size_y = bounds['max_y'] - bounds['min_y']
        
        self.W = int(size_x / GRID_RESOLUTION)
        self.H = int(size_y / GRID_RESOLUTION)
        self.grid = np.full((self.W, self.H), L_0)
        
    def world_to_grid(self, x, y):
        gx = int((x - self.offset_x) / GRID_RESOLUTION)
        gy = int((y - self.offset_y) / GRID_RESOLUTION)
        return gx, gy
    
    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    def update_map(self, ego_pose, angles, distances, max_range=5.0):
        rx, ry, rtheta = ego_pose
        gx0, gy0 = self.world_to_grid(rx, ry)
        
        for i in range(len(angles)):
            dist = distances[i]
            global_angle = rtheta + angles[i]
            end_x = rx + dist * math.cos(global_angle)
            end_y = ry + dist * math.sin(global_angle)
            gx1, gy1 = self.world_to_grid(end_x, end_y)
            cells = self.bresenham_line(gx0, gy0, gx1, gy1)
            
            for j, (cx, cy) in enumerate(cells):
                if 0 <= cx < self.W and 0 <= cy < self.H:
                    if j == len(cells) - 1 and dist < (max_range - 0.1):
                        self.grid[cx, cy] += L_OCC
                    else:
                        self.grid[cx, cy] += L_FREE
                    self.grid[cx, cy] = np.clip(self.grid[cx, cy], MIN_LOG_ODDS, MAX_LOG_ODDS)
                    
    def get_probabilities(self):
        return 1.0 - (1.0 - (1.0 / (1.0 + np.exp(self.grid))))


# ==========================================
# 4. SENSOR MODEL & TELEOP LOGIC
# ==========================================
def simulate_lidar_scan(robot_x, robot_y, robot_theta, walls):
    num_rays = 180 
    max_range = 5.0
    sigma_z = math.sqrt(VAR_LIDAR)
    
    angles, distances = [], []
    ray_angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    
    for rel_ang in ray_angles:
        glob_ang = robot_theta + rel_ang
        rx, ry = math.cos(glob_ang), math.sin(glob_ang)
        min_dist = max_range
        
        for wall in walls:
            qx, qy, bx, by = wall
            sx, sy = bx - qx, by - qy
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

def on_key_press(event):
    global current_v_cmd, current_alpha_cmd
    if event.key == 'up':
        current_v_cmd = min(current_v_cmd + V_STEP, MAX_V_CMD)
    elif event.key == 'down':
        current_v_cmd = max(current_v_cmd - V_STEP, -MAX_V_CMD)
    elif event.key == 'left':
        current_alpha_cmd = max(current_alpha_cmd - ALPHA_STEP, -MAX_ALPHA_CMD)
    elif event.key == 'right':
        current_alpha_cmd = min(current_alpha_cmd + ALPHA_STEP, MAX_ALPHA_CMD)
    elif event.key in [' ', 'x']:
        current_v_cmd = 0.0
        current_alpha_cmd = 0.0


# ==========================================
# 5. SIMULATION LOOP 
# ==========================================
def run_sim(map_name):
    walls, start_pose, bounds = load_map(map_name)
    delta_t = 0.1 
    true_pose = np.array(start_pose)
    
    ekf = EKFPoseTracker(true_pose)
    mapper = GridMapper(bounds)
    
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig1.canvas.manager.set_window_title(f'Ground Truth [{map_name}]')
    fig1.canvas.mpl_connect('key_press_event', on_key_press)
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.canvas.manager.set_window_title(f'Active SLAM [{map_name}]')
    fig2.canvas.mpl_connect('key_press_event', on_key_press) 
    
    cmap = LinearSegmentedColormap.from_list('grid_map', ['white', 'lightgrey', 'black'])
    
    ekf_history_x, ekf_history_y = [ekf.mu[0]], [ekf.mu[1]]
    
    step = 0
    while plt.fignum_exists(fig1.number) and plt.fignum_exists(fig2.number):
        
        v_phys, delta_phys = get_physical_commands(current_v_cmd, current_alpha_cmd)
        
        v_phys_noisy = v_phys + random.gauss(0, math.sqrt(VAR_V)) if v_phys != 0 else 0.0
        delta_phys_noisy = delta_phys + random.gauss(0, math.sqrt(VAR_DELTA)) if v_phys != 0 else 0.0
        true_pose = predict_next_pose(true_pose, v_phys_noisy, delta_phys_noisy, delta_t)
        
        angles, distances = simulate_lidar_scan(true_pose[0], true_pose[1], true_pose[2], walls)
        
        ekf.predict(v_phys, delta_phys, delta_t)
        simulated_icp_measurement = true_pose + np.random.multivariate_normal([0, 0, 0], EKF_MEASUREMENT_NOISE)
        ekf.update(simulated_icp_measurement)
        
        ekf_history_x.append(ekf.mu[0])
        ekf_history_y.append(ekf.mu[1])
        mapper.update_map(ekf.mu, angles, distances)
        
        ax1.clear()
        for wall in walls:
            ax1.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
        ax1.plot(true_pose[0], true_pose[1], 'go', markersize=8)
        arrow_len = 0.2
        ax1.arrow(true_pose[0], true_pose[1], arrow_len*math.cos(true_pose[2]), arrow_len*math.sin(true_pose[2]), head_width=0.05, fc='g')
        ax1.set_title(f"Ground Truth\nCmd V: {current_v_cmd} | Cmd Steer: {current_alpha_cmd}")
        ax1.set_xlim(bounds['min_x'], bounds['max_x'])
        ax1.set_ylim(bounds['min_y'], bounds['max_y'])
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.clear()
        prob_grid = mapper.get_probabilities()
        ax2.imshow(prob_grid.T, cmap=cmap, origin='lower', 
                   extent=[bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']], 
                   vmin=0, vmax=1)
        ax2.plot(ekf_history_x, ekf_history_y, 'b--', linewidth=1.5, label='EKF Trajectory')
        ax2.plot(ekf.mu[0], ekf.mu[1], 'ro', markersize=6, label='EKF Pose')
        ax2.arrow(ekf.mu[0], ekf.mu[1], arrow_len*math.cos(ekf.mu[2]), arrow_len*math.sin(ekf.mu[2]), head_width=0.05, fc='r', ec='r')
        ax2.set_title(f"EKF Pose + Occupancy Grid Map\nStep: {step}")
        ax2.set_xlim(bounds['min_x'], bounds['max_x'])
        ax2.set_ylim(bounds['min_y'], bounds['max_y'])
        
        plt.pause(0.01)
        step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active SLAM Teleop Simulator")
    parser.add_argument('--map', type=str, default='simple', 
                        help='Name of the map to load from the maps/ folder (e.g. simple, maze, corridors, circular)')
    args = parser.parse_args()
    
    print(f"🚗 Loading Map: {args.map}")
    print("UP/DOWN: Speed | LEFT/RIGHT: Steer | X: Stop")
    run_sim(args.map)