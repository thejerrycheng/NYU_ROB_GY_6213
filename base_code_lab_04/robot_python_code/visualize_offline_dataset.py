import os
import glob
import csv
import ast
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Import your local map parameters
import parameters

# --- 1. Calibrated Constants ---
INITIAL_POSE = [0.1, 0.1, math.pi / 2]

L = 0.145  
V_M = 0.004808 
V_C = -0.045557 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]

def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

# --- 2. Motion Model (Pure Dead Reckoning) ---
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        """Updates robot state using the pure Ackermann motion model (No Encoders)"""
        # Velocity Model
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: 
            v_expected = 0.0

        # Steering Model
        alpha = steering_angle_command
        delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]

        # Kinematics
        w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0

        # Euler Integration
        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2] = angle_wrap(self.state[2] - delta_t * w_expected)

        return self.state

def get_latest_csv(directory="offline_dataset"):
    """Finds the most recently created CSV file in the dataset folder."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found!")
        return None
        
    list_of_files = glob.glob(f"{directory}/*.csv")
    if not list_of_files:
        print(f"No CSV files found in '{directory}'!")
        return None
        
    return max(list_of_files, key=os.path.getctime)

def main(csv_file):
    if not csv_file: return
    
    # Extract the commanded speed from the filename
    try:
        v_cmd = float(os.path.basename(csv_file).split('_')[0])
    except ValueError:
        v_cmd = 0.0
        print("[Warning] Could not parse v_cmd from filename. Defaulting to 0.0")

    print(f"Loading dataset: {csv_file}")
    print(f"Using Commanded Speed: {v_cmd}")
    
    # --- Robot State Initialization ---
    motion_model = MyMotionModel(INITIAL_POSE)
    last_time = None
    
    # --- Setup Plot ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    all_x = [w[0] for w in parameters.wall_corner_list] + [w[2] for w in parameters.wall_corner_list]
    all_y = [w[1] for w in parameters.wall_corner_list] + [w[3] for w in parameters.wall_corner_list]
    plot_xlim = (min(all_x) - 0.5, max(all_x) + 0.5)
    plot_ylim = (min(all_y) - 0.5, max(all_y) + 0.5)
    
    history_x, history_y = [], []
    
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for step, row in enumerate(reader):
            current_time = float(row['Time_s'])
            steering = float(row['Steering']) 
            
            raw_angles = ast.literal_eval(row['Lidar_Angles'])
            raw_distances = ast.literal_eval(row['Lidar_Distances'])
            
            # --- Dead Reckoning using the Motion Model ---
            if last_time is not None:
                dt = current_time - last_time
                if dt > 0:
                    motion_model.step_update(v_cmd, steering, dt)
            
            last_time = current_time
            
            curr_x = motion_model.state[0]
            curr_y = motion_model.state[1]
            curr_theta = motion_model.state[2]
            
            history_x.append(curr_x)
            history_y.append(curr_y)
            
            # --- Visualization (Subsample every 2nd frame to prevent lag) ---
            if step % 2 == 0:
                ax.clear()
                
                # Draw Map
                for wall in parameters.wall_corner_list:
                    ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2, zorder=1)
                
                # Draw Trajectory
                ax.plot(history_x, history_y, 'b--', linewidth=1.5, alpha=0.6, label="Motion Model Path", zorder=2)
                
                # --- FIX: Extrinsic Lidar Calibration ---
                X_OFFSET = 0
                xs = curr_x + X_OFFSET * math.cos(curr_theta)
                ys = curr_y + X_OFFSET * math.sin(curr_theta)

                ray_x, ray_y = [], []
                for i in range(0, len(raw_angles), 3): 
                    raw_dist = raw_distances[i]
                    
                    # FIX: Safely bound raw distances to remove hardware glitches
                    if 100 < raw_dist < 4900: 
                        dist_m = raw_dist / 1000.0
                        angle_rad = -(raw_angles[i] * math.pi / 180.0) 
                        
                        global_ray_angle = angle_wrap(curr_theta + angle_rad)
                        
                        # Project ray from sensor origin (xs, ys)
                        hit_x = xs + dist_m * math.cos(global_ray_angle)
                        hit_y = ys + dist_m * math.sin(global_ray_angle)
                        
                        ray_x.extend([xs, hit_x, None])
                        ray_y.extend([ys, hit_y, None])
                        
                ax.plot(ray_x, ray_y, color='red', alpha=0.3, linewidth=0.5, zorder=3)
                
                # Draw Robot Base
                ax.plot(curr_x, curr_y, 'mo', markersize=6, zorder=4, label="Robot Base")
                ax.quiver(curr_x, curr_y, math.cos(curr_theta), math.sin(curr_theta), color='magenta', scale=15, width=0.007, zorder=5)
                
                ax.set_title(f"Dataset Playback | Time: {current_time:.2f}s | Step: {step}")
                ax.set_xlim(plot_xlim)
                ax.set_ylim(plot_ylim)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize='x-small')
                
                plt.pause(0.1)

    print("\nPlayback complete! Close the plot window to exit.")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize offline dataset with dead reckoning.")
    parser.add_argument("--data", type=str, help="Path to the CSV dataset file.", default=None)
    args = parser.parse_args()

    target_csv = args.data
    if not target_csv:
        target_csv = get_latest_csv()

    if target_csv and os.path.exists(target_csv):
        main(target_csv)
    else:
        print(f"Error: Could not find dataset '{target_csv}'")