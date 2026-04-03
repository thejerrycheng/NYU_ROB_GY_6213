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
VAR_LIDAR = 0.000363

# --- Global Teleop State ---
current_v_cmd = 0.0
current_alpha_cmd = 0.0

# Teleop settings
V_STEP = 5.0          
ALPHA_STEP = 5.0      
MAX_V_CMD = 100.0     
MAX_ALPHA_CMD = 100.0

# ==========================================
# 1. MOTION MODEL (Pure Kinematic)
# ==========================================

def angle_wrap(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def predict_next_pose(current_pose, v_cmd, alpha_cmd, delta_t=0.1, add_noise=True):
    """Predicts the next robot pose instantly based on commands."""
    x, y, theta = current_pose
    
    # 1. Map Command to Physical Velocity (Fixed for zero and reverse)
    if v_cmd == 0.0:
        v_physical = 0.0
    else:
        # Calculate absolute speed magnitude using calibration
        v_mag = (V_M * abs(v_cmd)) + V_C
        
        # Prevent backward drift due to negative V_C intercept
        if v_mag < 0:
            v_mag = 0.0 
            
        # Re-apply the correct direction
        v_physical = v_mag if v_cmd > 0 else -v_mag

    # Calculate physical steering angle
    delta_physical = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
    
    # 2. Add Noise (Only if the robot is actually commanded to move)
    if add_noise and abs(v_physical) > 0.0001:
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
                        
        if min_distance < max_range:
            noisy_distance = min_distance + random.gauss(0, sigma_z)
            min_distance = max(0.0, noisy_distance)
                        
        angles.append(relative_angle)
        distances.append(min_distance)
        
    return angles, distances


# ==========================================
# 3. VISUALIZATION & KEYBOARD LOGIC
# ==========================================

def on_key_press(event):
    global current_v_cmd, current_alpha_cmd
    
    if event.key == 'up':
        current_v_cmd = min(current_v_cmd + V_STEP, MAX_V_CMD)
    elif event.key == 'down':
        current_v_cmd = max(current_v_cmd - V_STEP, -MAX_V_CMD)
    elif event.key == 'left':
        # Left steers left (decreases command to offset the negative Euler integration)
        current_alpha_cmd = max(current_alpha_cmd - ALPHA_STEP, -MAX_ALPHA_CMD)
    elif event.key == 'right':
        # Right steers right (increases command)
        current_alpha_cmd = min(current_alpha_cmd + ALPHA_STEP, MAX_ALPHA_CMD)
    elif event.key in [' ', 'x']:
        # Stop keys
        current_v_cmd = 0.0
        current_alpha_cmd = 0.0


def visualize(ax, current_pose, history_x, history_y, angles, distances, step):
    ax.clear()
    
    for wall in parameters.wall_corner_list:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
        
    ax.plot(history_x, history_y, 'b--', linewidth=1.5, alpha=0.6, label='Predicted Path')
    
    ray_lines_x, ray_lines_y = [], []
    hit_points_x, hit_points_y = [], []
    
    for i in range(len(angles)):
        if distances[i] < 4.9:
            global_angle = current_pose[2] + angles[i]
            hit_x = current_pose[0] + distances[i] * math.cos(global_angle)
            hit_y = current_pose[1] + distances[i] * math.sin(global_angle)
            
            ray_lines_x.extend([current_pose[0], hit_x, None])
            ray_lines_y.extend([current_pose[1], hit_y, None])
            
            hit_points_x.append(hit_x)
            hit_points_y.append(hit_y)
            
    ax.plot(ray_lines_x, ray_lines_y, color='lightblue', linewidth=0.5, zorder=1)
    ax.plot(hit_points_x, hit_points_y, 'r.', markersize=2, zorder=2)
        
    ax.plot(current_pose[0], current_pose[1], 'go', markersize=8, zorder=3, label='Robot')
    arrow_len = 0.15
    ax.arrow(current_pose[0], current_pose[1], 
             arrow_len * math.cos(current_pose[2]), arrow_len * math.sin(current_pose[2]), 
             head_width=0.05, head_length=0.05, fc='green', ec='green', zorder=4)

    ax.set_title(f"Teleop Sim | Step: {step}\nCmd V: {current_v_cmd:.1f} | Cmd Steer: {current_alpha_cmd:.1f}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
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
    delta_t = 0.1 
    initial_pose = [0.3, 0.2, math.pi / 2]  
    
    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    print("🚗 Teleop Started!")
    print("UP/DOWN: Change speed")
    print("LEFT/RIGHT: Steer")
    print("X or SPACE: Stop instantly")
    print("Close the window to exit.")

    current_pose = list(initial_pose)
    history_x = [current_pose[0]]
    history_y = [current_pose[1]]
    
    step = 0
    while plt.fignum_exists(fig.number):
        
        # 1. Move the Robot (Instantly applying commands)
        current_pose = predict_next_pose(
            current_pose, current_v_cmd, current_alpha_cmd, delta_t, add_noise=True
        )
        
        history_x.append(current_pose[0])
        history_y.append(current_pose[1])
        
        # 2. Fire Lidar
        angles, distances = simulate_lidar_scan(current_pose[0], current_pose[1], current_pose[2])
        
        # 3. Render
        visualize(ax, current_pose, history_x, history_y, angles, distances, step)
        
        plt.pause(delta_t)
        step += 1
        
    plt.ioff()
    print("Simulation closed.")

if __name__ == '__main__':
    run_sim()