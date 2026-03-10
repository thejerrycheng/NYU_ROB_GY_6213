import os
import glob
import csv
import ast
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import your local map parameters
import parameters

# =======================================================
# Configuration & Kinematic Constants
# =======================================================
# --- STARTING POSE [x, y, theta] ---
INITIAL_POSE = [0.9, 0.2, math.pi / 2]

# --- Kinematic Constants ---
L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.023134

# --- Sensor Calibration ---
MAX_RANGE = 5.0
X_OFFSET = 0.12 
VAR_Z = 0.00025 

def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

def get_latest_csv(directory="offline_dataset"):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found!")
        return None
    list_of_files = glob.glob(f"{directory}/*.csv")
    if not list_of_files:
        print(f"No CSV files found in '{directory}'!")
        return None
    return max(list_of_files, key=os.path.getctime)

# =======================================================
# 1. Pure Dead Reckoning (Deterministic Motion Model)
# =======================================================
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: 
            v_expected = 0.0

        alpha = steering_angle_command
        delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]

        w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0

        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2] = angle_wrap(self.state[2] - delta_t * w_expected)

        return self.state

# =======================================================
# 2. Particle Filter (Stochastic Motion Model + Lidar)
# =======================================================
class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.weight = 1.0
        self.log_w = 0.0 

    def predict(self, v_cmd, alpha_cmd, dt):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: 
            v_expected = 0.0
            
        delta_expected = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]

        if v_expected > 0:
            v_s = v_expected + random.gauss(0, math.sqrt(VAR_V))
            d_s = delta_expected + random.gauss(0, math.sqrt(VAR_DELTA))
        else:
            v_s = 0.0
            d_s = delta_expected

        w_s = (v_s * math.tan(d_s)) / L if L > 0 else 0.0
        
        self.x += v_s * math.cos(self.theta) * dt
        self.y += v_s * math.sin(self.theta) * dt
        self.theta = angle_wrap(self.theta - w_s * dt) 

    def update_weight(self, angles, distances):
        log_w = 0.0
        ray_step = 10 
        
        xs = self.x + X_OFFSET * math.cos(self.theta)
        ys = self.y + X_OFFSET * math.sin(self.theta)

        for i in range(0, len(angles), ray_step):
            raw_dist = distances[i]
            
            if raw_dist < 100 or raw_dist >= 4900: 
                continue
            
            dist_m = raw_dist / 1000.0
            angle_rad = -(angles[i] * math.pi / 180.0)
            global_angle = angle_wrap(self.theta + angle_rad)
            
            rx, ry = math.cos(global_angle), math.sin(global_angle)
            min_dist = MAX_RANGE
            
            for wall in parameters.wall_corner_list:
                qx, qy, bx, by = wall
                sx, sy = bx - qx, by - qy
                denom = rx * sy - ry * sx
                if abs(denom) > 1e-6:
                    t = ((qx - xs) * sy - (qy - ys) * sx) / denom
                    u = ((qx - xs) * ry - (qy - ys) * rx) / denom
                    if 0 <= u <= 1 and 0 < t < min_dist:
                        min_dist = t
                        
            if min_dist < MAX_RANGE:
                error = min_dist - dist_m
                penalty = (error**2) / (2 * VAR_Z)
                log_w -= min(penalty, 10.0) 
            else:
                log_w -= 10.0 
            
        self.log_w = log_w

class ParticleFilter:
    def __init__(self, num_particles, initial_pose):
        self.num_particles = num_particles
        self.particles = []
        
        all_walls = np.array(parameters.wall_corner_list)
        self.x_min = np.min(all_walls[:, [0, 2]])
        self.x_max = np.max(all_walls[:, [0, 2]])
        self.y_min = np.min(all_walls[:, [1, 3]])
        self.y_max = np.max(all_walls[:, [1, 3]])
        
        self.global_initialization(initial_pose)

    def global_initialization(self, pose):
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.gauss(pose[0], 0.1)
            y = random.gauss(pose[1], 0.1)
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                continue
            theta = angle_wrap(random.gauss(pose[2], 0.1))
            self.particles.append(Particle(x, y, theta))

    def predict(self, v_cmd, alpha_cmd, dt):
        for p in self.particles: 
            p.predict(v_cmd, alpha_cmd, dt)

    def correct(self, angles, distances):
        if len(angles) < 10: return 

        for p in self.particles: 
            p.update_weight(angles, distances)
            
        max_log_w = max(p.log_w for p in self.particles)
        for p in self.particles:
            p.weight = math.exp(p.log_w - max_log_w)
            
        self.resample()

    def resample(self):
        weights = [p.weight for p in self.particles]
        new_particles = []
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        max_w = max(weights)
        
        for _ in range(self.num_particles):
            beta += random.uniform(0, 2.0 * max_w)
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % self.num_particles
            
            p = self.particles[index]
            new_particles.append(Particle(p.x, p.y, p.theta))
            
        self.particles = new_particles

    def get_estimate(self):
        sum_x = sum(p.x for p in self.particles)
        sum_y = sum(p.y for p in self.particles)
        sum_sin = sum(math.sin(p.theta) for p in self.particles)
        sum_cos = sum(math.cos(p.theta) for p in self.particles)
        return [sum_x / self.num_particles, sum_y / self.num_particles, math.atan2(sum_sin, sum_cos)]
    
# =======================================================
# 3. Main Loop
# =======================================================
def run_offline_pf(csv_file):
    if not csv_file: return
        
    try:
        v_cmd = float(os.path.basename(csv_file).split('_')[0])
    except ValueError:
        v_cmd = 0.0

    print(f"Loading dataset: {csv_file}")
    print(f"Using Commanded Speed for DR & PF: {v_cmd}")

    all_walls = np.array(parameters.wall_corner_list)
    x_min, x_max = np.min(all_walls[:, [0, 2]]), np.max(all_walls[:, [0, 2]])
    y_min, y_max = np.min(all_walls[:, [1, 3]]), np.max(all_walls[:, [1, 3]])
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    history = {
        'steps': [], 
        'est_x': [], 'est_y': [], 'est_theta': [],
        'dr_x': [], 'dr_y': [], 'dr_theta': []
    }

    # --- Directory creation for saving plots ---
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    update_count = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    pf = ParticleFilter(num_particles=1500, initial_pose=INITIAL_POSE)
    dr = MyMotionModel(initial_state=INITIAL_POSE)
    last_time = None
    
    sweep_angles = []
    sweep_distances = []
    last_lidar_angle = None
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for step, row in enumerate(reader):
            current_time = float(row['Time_s'])
            steering = float(row['Steering'])
            raw_angles = ast.literal_eval(row['Lidar_Angles'])
            raw_distances = ast.literal_eval(row['Lidar_Distances'])
            
            if last_time is None:
                last_time = current_time
                continue

            dt = current_time - last_time
            if dt > 0:
                # 1. Prediction Step
                pf.predict(v_cmd, steering, dt)
                dr.step_update(v_cmd, steering, dt)
                
                # 2. Accumulate Lidar until full 360 lap is detected
                sweep_complete = False
                for i in range(len(raw_angles)):
                    ang = raw_angles[i]
                    dist = raw_distances[i]
                    
                    if last_lidar_angle is not None and abs(ang - last_lidar_angle) > 180:
                        sweep_complete = True
                        
                    sweep_angles.append(ang)
                    sweep_distances.append(dist)
                    last_lidar_angle = ang
                
                # 3. Correction Step
                if sweep_complete:
                    pf.correct(sweep_angles, sweep_distances)

                # 4. Save States
                est_pose = pf.get_estimate()
                history['steps'].append(step)
                history['est_x'].append(est_pose[0])
                history['est_y'].append(est_pose[1])
                history['est_theta'].append(est_pose[2])
                history['dr_x'].append(dr.state[0])
                history['dr_y'].append(dr.state[1])
                history['dr_theta'].append(dr.state[2])
            
            last_time = current_time

            # 5. Only visualize and save WHEN A SWEEP IS COMPLETED
            if dt > 0 and sweep_complete:
                ax.clear()
                # Draw Map
                for wall in parameters.wall_corner_list:
                    ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2, zorder=1)
                
                # Draw Particles
                px, py = [p.x for p in pf.particles], [p.y for p in pf.particles]
                ax.scatter(px, py, s=2, c='green', alpha=0.15, zorder=2, label='Particles')
                
                # Draw Paths
                ax.plot(history['dr_x'], history['dr_y'], color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Motion Model (DR)', zorder=3)
                ax.plot(history['est_x'], history['est_y'], color='blue', linestyle='-', linewidth=1.5, alpha=0.6, label='PF Estimate', zorder=4)

                # Draw Lidar
                ray_x, ray_y = [], []
                xs = est_pose[0] + X_OFFSET * math.cos(est_pose[2])
                ys = est_pose[1] + X_OFFSET * math.sin(est_pose[2])
                
                for i in range(0, len(sweep_angles), 5): 
                    raw_dist = sweep_distances[i]
                    if 100 < raw_dist < 4900:
                        dist = raw_dist / 1000.0
                        angle = angle_wrap(-(sweep_angles[i] * math.pi / 180.0) + est_pose[2])
                        hx = xs + dist * math.cos(angle)
                        hy = ys + dist * math.sin(angle)
                        ray_x.extend([xs, hx, None])
                        ray_y.extend([ys, hy, None])
                
                ax.plot(ray_x, ray_y, color='red', alpha=0.15, linewidth=0.5, zorder=5)

                # Draw Robot Vectors
                ax.plot(dr.state[0], dr.state[1], 'o', color='orange', markersize=6, zorder=6)
                ax.plot(est_pose[0], est_pose[1], 'mo', markersize=6, zorder=7)
                ax.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), color='magenta', scale=15, width=0.007, zorder=8)

                ax.set_title(f"Offline Physical PF | Update: {update_count} | Step: {step}")
                ax.set_xlim(x_min - x_pad, x_max + x_pad); ax.set_ylim(y_min - y_pad, y_max + y_pad)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize='x-small')
                
                # --- Save Figure to disk ---
                file_path = os.path.join(plot_dir, f'update_{update_count:04d}.png')
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                
                plt.pause(0.001)
                
                # Increment update count and explicitly clear the sweep buffers
                update_count += 1
                sweep_angles = []
                sweep_distances = []

    plt.ioff()
    plt.close(fig) 

    # ==========================================
    # FINAL TRAJECTORY PLOT
    # ==========================================
    fig_traj, ax_traj = plt.subplots(figsize=(4, 4))
    for wall in parameters.wall_corner_list:
        ax_traj.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
    
    ax_traj.plot(history['dr_x'], history['dr_y'], color='orange', linestyle='--', linewidth=1.5, label='Motion Model (DR)')
    ax_traj.plot(history['est_x'], history['est_y'], color='blue', linestyle='-', linewidth=1.5, label='PF Estimate')
    
    ax_traj.set_title("Final Trajectory: PF vs Motion Model", fontsize=10)
    ax_traj.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_traj.set_aspect('equal')
    ax_traj.legend(loc='lower left', fontsize=8)
    
    plt.savefig('offline_dr_vs_pf_comparison.png', dpi=300, bbox_inches='tight')
    print("Trajectory plot saved: offline_dr_vs_pf_comparison.png")
    print(f"Individual Lidar updates saved to folder: '{plot_dir}/'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Offline Particle Filter")
    parser.add_argument("--data", type=str, help="Path to the CSV dataset file.", default=None)
    args = parser.parse_args()

    target_csv = args.data
    if not target_csv:
        target_csv = get_latest_csv()

    if target_csv and os.path.exists(target_csv):
        run_offline_pf(target_csv)
    else:
        print(f"Error: Could not find dataset '{target_csv}'")