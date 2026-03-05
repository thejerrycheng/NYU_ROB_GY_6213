import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Import your local parameters file for the map
import parameters

# --- Calibrated Constants from your model ---
L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.00057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134
VAR_LIDAR = 0.000363  # Lidar measurement variance (sigma_z^2)

# ==========================================
# 1. MOTION MODEL (Prediction)
# ==========================================

def angle_wrap(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def predict_next_pose(current_pose, v_cmd, alpha_cmd, delta_t=0.1, add_noise=True):
    """Predicts the next robot pose given velocity and steering commands."""
    x, y, theta = current_pose
    
    # 1. Map Commands to Physical Values
    if v_cmd == 0.0:
        # Explicitly stop if commanded to stop
        v_physical = 0.0
    else:
        # Otherwise, apply the calibration model
        v_physical = (V_M * v_cmd) + V_C
        # Failsafe: Prevent backward drift if a small positive command can't overcome friction
        if v_physical < 0:
            v_physical = 0.0 

    delta_physical = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
    
    # 2. Add Noise for realism (Only if the robot is actually trying to move)
    if add_noise and v_physical > 0:
        v_physical += random.gauss(0, math.sqrt(VAR_V))
        delta_physical += random.gauss(0, math.sqrt(VAR_DELTA))
        
    # 3. Kinematics
    if L > 0:
        w = (v_physical * math.tan(delta_physical)) / L
    else:
        w = 0.0
        
    # 4. Euler Integration
    next_x = x + (v_physical * math.cos(theta) * delta_t)
    next_y = y + (v_physical * math.sin(theta) * delta_t)
    next_theta = angle_wrap(theta - (w * delta_t))
    
    return [next_x, next_y, next_theta]


# ==========================================
# 2. SENSOR MODEL (Correction / Measurement)
# ==========================================
def simulate_lidar_scan(robot_x, robot_y, robot_theta):
    """Simulates Lidar measurements and injects empirical Gaussian noise."""
    walls = parameters.wall_corner_list
    num_rays = 360
    max_range = 5.0
    
    sigma_z = math.sqrt(VAR_LIDAR)
    
    angles = []
    distances = []
    
    ray_angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    
    for relative_angle in ray_angles:
        global_ray_angle = robot_theta + relative_angle
        rx = math.cos(global_ray_angle)
        ry = math.sin(global_ray_angle)
        min_distance = max_range
        
        for wall in walls:
            qx, qy, bx, by = wall
            sx = bx - qx
            sy = by - qy
            r_cross_s = rx * sy - ry * sx
            if abs(r_cross_s) > 1e-6: 
                q_p_x = qx - robot_x
                q_p_y = qy - robot_y  
                t = (q_p_x * sy - q_p_y * sx) / r_cross_s  
                u = (q_p_x * ry - q_p_y * rx) / r_cross_s  
                if t > 0 and 0 <= u <= 1:
                    if t < min_distance:
                        min_distance = t
                        
        # --- ADD SENSOR NOISE ---
        if min_distance < max_range:
            noisy_distance = min_distance + random.gauss(0, sigma_z)
            min_distance = max(0.0, noisy_distance)
                        
        angles.append(relative_angle)
        distances.append(min_distance)
        
    return angles, distances


# ==========================================
# 3. VISUALIZATION & ANIMATION LOOP
# ==========================================

def visualize(ax, current_pose, history_x, history_y, angles, distances, step):
    """
    Purely handles drawing the state of the world to the screen.
    """
    ax.clear()
    
    # 1. Draw Map Walls
    for wall in parameters.wall_corner_list:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
        
    # 2. Draw Trajectory Trail
    ax.plot(history_x, history_y, 'b--', linewidth=1.5, alpha=0.6, label='Predicted Path')
    
    # 3. Fast Lidar Rendering Trick (Batched lines)
    ray_lines_x, ray_lines_y = [], []
    hit_points_x, hit_points_y = [], []
    
    for i in range(len(angles)):
        if distances[i] < 4.9: # Only draw rays that hit a wall
            global_angle = current_pose[2] + angles[i]
            hit_x = current_pose[0] + distances[i] * math.cos(global_angle)
            hit_y = current_pose[1] + distances[i] * math.sin(global_angle)
            
            ray_lines_x.extend([current_pose[0], hit_x, None])
            ray_lines_y.extend([current_pose[1], hit_y, None])
            
            hit_points_x.append(hit_x)
            hit_points_y.append(hit_y)
            
    # Draw Lidar beams and hit dots
    ax.plot(ray_lines_x, ray_lines_y, color='lightblue', linewidth=0.5, zorder=1)
    ax.plot(hit_points_x, hit_points_y, 'r.', markersize=2, zorder=2)
        
    # 4. Draw the Robot
    ax.plot(current_pose[0], current_pose[1], 'go', markersize=8, zorder=3, label='Robot')
    arrow_len = 0.15
    ax.arrow(current_pose[0], current_pose[1], 
             arrow_len * math.cos(current_pose[2]), arrow_len * math.sin(current_pose[2]), 
             head_width=0.05, head_length=0.05, fc='green', ec='green', zorder=4)

    # 5. Apply Plot Settings
    ax.set_title(f"Simultaneous Motion & Noisy Lidar Sim\nStep: {step}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    # Map Boundaries based on parameters
    all_x = [w[0] for w in parameters.wall_corner_list] + [w[2] for w in parameters.wall_corner_list]
    all_y = [w[1] for w in parameters.wall_corner_list] + [w[3] for w in parameters.wall_corner_list]
    ax.set_xlim(min(all_x) - 0.2, max(all_x) + 0.2)
    ax.set_ylim(min(all_y) - 0.2, max(all_y) + 0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)



# ==========================================
# 4. Simulation Loop
# ==========================================

def run_sim():
    """
    Handles simulation logic, user inputs, and calls the visualizer.
    """
    # --- 🟢 USER EDITABLE CONTROL PARAMETERS 🟢 ---
    total_steps = 120
    delta_t = 0.1 
    initial_pose = [0.3, 0.2, math.pi / 2]  # [x, y, theta]
    
    # Define driving phases: (end_step, v_cmd, alpha_cmd)
    # The robot will read this list and apply the commands in sequence.
    control_sequence = [
        (30,  40.0, 0.0),   # Phase 1: Drive straight until step 30
        (80,  40.0, 50.0),  # Phase 2: Sharp right turn until step 80
        (100, 50.0, -10.0)    # Phase 3: Drive straight slower until step 100
        # If the step goes above 100, it automatically stops (0.0, 0.0)
    ]
    # ----------------------------------------------

    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 8))
    
    current_pose = list(initial_pose)
    history_x = [current_pose[0]]
    history_y = [current_pose[1]]
    
    for step in range(total_steps):
        
        # 1. Determine current command based on sequence
        v_cmd = 0.0
        alpha_cmd = 0.0
        for end_step, v, alpha in control_sequence:
            if step < end_step:
                v_cmd = v
                alpha_cmd = alpha
                break # Found the active command, exit the inner loop

        # 2. Move the Robot (Prediction)
        current_pose = predict_next_pose(current_pose, v_cmd, alpha_cmd, delta_t, add_noise=True)
        history_x.append(current_pose[0])
        history_y.append(current_pose[1])
        
        # 3. Fire the Noisy Lidar (Measurement)
        angles, distances = simulate_lidar_scan(current_pose[0], current_pose[1], current_pose[2])
        
        # 4. Render everything to the screen
        visualize(ax, current_pose, history_x, history_y, angles, distances, step)
        
        plt.pause(0.05)
        
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_sim()