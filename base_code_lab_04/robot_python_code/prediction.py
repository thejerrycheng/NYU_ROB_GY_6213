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

def predict_next_pose(current_pose, v_cmd, alpha_cmd, delta_t=0.1, add_noise=True):
    """
    Predicts the next robot pose given velocity and steering commands.
    """
    x, y, theta = current_pose
    
    # 1. Map Commands to Physical Values (Now strictly in meters/sec)
    v_physical = (V_M * v_cmd) + V_C
    
    # Check if the command is too low to overcome static friction (V_C)
    if v_physical < 0 and v_cmd > 0:
        v_physical = 0.0 

    # Map Steering (Ensure alpha is converted properly, assuming positive is right turn)
    delta_physical = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
    
    # 2. Add Noise for realism
    if add_noise and v_physical > 0:
        # NO DIVISION BY 1000 NEEDED ANYMORE: Variance is already correctly scaled
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
    next_theta = theta - (w * delta_t) # Negative because right turn decreases Theta
    
    return [next_x, next_y, next_theta]

def run_motion_simulation():
    # Start Interactive Plot Mode
    plt.ion() 
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Initial Robot Pose (Bottom left of the map, facing North/Up)
    current_pose = [0.3, 0.2, math.pi / 2]  
    
    # To store the trajectory history for drawing the path line
    history_x = [current_pose[0]]
    history_y = [current_pose[1]]
    
    delta_t = 0.1 # 10Hz update rate
    
    # Simulation Loop
    for step in range(120):
        # --- Control Strategy (Keep the robot inside the map) ---
        v_cmd = 0.0
        alpha_cmd = 0.0
        
        if step < 30:
            # Phase 1: Drive straight North (Moves ~0.45 meters)
            v_cmd = 40.0  
            alpha_cmd = 0.0
        elif step < 75:
            # Phase 2: Execute a tight Right Turn BEFORE hitting y=1.2m
            v_cmd = 40.0  
            alpha_cmd = 35.0 # Max safe steering for a tighter turn radius
        elif step < 100:
            # Phase 3: Drive straight East down the bottom hallway
            v_cmd = 40.0
            alpha_cmd = 0.0
        else:
            # Phase 4: Stop
            v_cmd = 0.0
            alpha_cmd = 0.0

        # --- Predict Next Pose ---
        current_pose = predict_next_pose(current_pose, v_cmd, alpha_cmd, delta_t, add_noise=True)
        
        # Save history for the trail
        history_x.append(current_pose[0])
        history_y.append(current_pose[1])
        
        # --- Visualization ---
        ax.clear()
        
        # 1. Redraw Map Walls from parameters.py
        for wall in parameters.wall_corner_list:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
            
        # 2. Draw the trajectory history (the path the robot took)
        ax.plot(history_x, history_y, 'b--', linewidth=1.5, alpha=0.6, label='Predicted Path')
            
        # 3. Draw the Robot (Base link)
        ax.plot(current_pose[0], current_pose[1], 'go', markersize=8, zorder=3, label='Robot')
        
        # 4. Draw the Robot Heading Indicator (Arrow)
        arrow_len = 0.15
        ax.arrow(current_pose[0], current_pose[1], 
                 arrow_len * math.cos(current_pose[2]), 
                 arrow_len * math.sin(current_pose[2]), 
                 head_width=0.05, head_length=0.05, fc='green', ec='green', zorder=4)

        # Plot Settings
        ax.set_title(f"Ackermann Kinematics Sim\nStep: {step} | v_cmd: {v_cmd} | alpha: {alpha_cmd}")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        
        # Scale axes based on the imported map boundaries
        all_x = [w[0] for w in parameters.wall_corner_list] + [w[2] for w in parameters.wall_corner_list]
        all_y = [w[1] for w in parameters.wall_corner_list] + [w[3] for w in parameters.wall_corner_list]
        ax.set_xlim(min(all_x) - 0.2, max(all_x) + 0.2)
        ax.set_ylim(min(all_y) - 0.2, max(all_y) + 0.2)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower left')
        
        # Pause to render the frame
        plt.pause(0.05)
        
    # Keep the window open when finished
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_motion_simulation()