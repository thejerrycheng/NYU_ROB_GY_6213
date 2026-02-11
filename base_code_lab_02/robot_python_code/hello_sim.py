import numpy as np
import matplotlib.pyplot as plt
import math
from motion_models import MyMotionModel, V_M, V_C

# --- Configuration ---
# Speed Command to use for moves (u)
SPEED_CMD = 75   # ~0.31 m/s (Brisk but controllable)

# Steering Commands (alpha)
# R_smallest (~0.74m) corresponds to Alpha = 20
ALPHA_SMALL = 20 
# 2 * R_smallest (~1.48m) corresponds to roughly Alpha = 10
ALPHA_LARGE = 10 

# Grid size
GRID = 1.48

def get_phys_velocity(v_cmd):
    """Returns physical velocity (m/s) for a given command."""
    # Using your calibration: v = 0.0048*u - 0.045
    # Assuming reverse (-u) gives negative velocity of same magnitude
    mag = 0.0048 * abs(v_cmd) - 0.0455
    if v_cmd < 0: return -mag
    return mag

def calc_dur_dist(dist, v_cmd):
    """Calculates duration needed to travel 'dist' meters."""
    v_phys = get_phys_velocity(v_cmd)
    return abs(dist / v_phys)

def calc_dur_turn(angle_deg, alpha, v_cmd):
    """Calculates duration for a turn of 'angle_deg'."""
    # 1. Estimate Radius R
    # Delta approx: 0.000027*a^2 + 0.0078*a + 0.03
    # We use the absolute alpha for radius calc
    a = abs(alpha)
    delta = 0.000027*(a**2) + 0.007798*a + 0.029847
    L = 0.145
    R = L / math.tan(delta)
    
    # 2. Arc Length
    angle_rad = math.radians(abs(angle_deg))
    arc_len = R * angle_rad
    
    # 3. Duration
    v_phys = get_phys_velocity(v_cmd)
    return abs(arc_len / v_phys)

# --- Constructing the Trajectory ---
# Format: (Description, v_cmd, alpha, duration)
traj_cmds = []

# 1. Move down 5.8 meters
dur = calc_dur_dist(5.8, SPEED_CMD)
traj_cmds.append(("Down 5.8m", -SPEED_CMD, 0, dur))

# 2. Up 1.45 meters
dur = calc_dur_dist(1.45, SPEED_CMD)
traj_cmds.append(("Up 1.45m", SPEED_CMD, 0, dur))

# 3. Turn Right 2R_smallest (Large) 90 deg
dur = calc_dur_turn(90, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Right 2R 90", SPEED_CMD, ALPHA_LARGE, dur))

# 4. Turn Left 2R 90 degrees
dur = calc_dur_turn(90, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Left 2R 90", SPEED_CMD, -ALPHA_LARGE, dur))

# 5. Straight 1.48 (1 grid)
dur = calc_dur_dist(1.48, SPEED_CMD)
traj_cmds.append(("Straight 1.48", SPEED_CMD, 0, dur))

# 6. Down 4 grids (5.92m)
dur = calc_dur_dist(4 * GRID, SPEED_CMD)
traj_cmds.append(("Down 4 Grids", -SPEED_CMD, 0, dur))

# 7. Right 2R 90 degrees
dur = calc_dur_turn(90, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Right 2R 90", SPEED_CMD, ALPHA_LARGE, dur))

# 8. Left R_smallest 180
dur = calc_dur_turn(180, ALPHA_SMALL, SPEED_CMD)
traj_cmds.append(("Left R_small 180", SPEED_CMD, -ALPHA_SMALL, dur))

# 9. Left 2R 270
dur = calc_dur_turn(270, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Left 2R 270", SPEED_CMD, -ALPHA_LARGE, dur))

# 10. Up 3 grids
dur = calc_dur_dist(3 * GRID, SPEED_CMD)
traj_cmds.append(("Up 3 Grids", SPEED_CMD, 0, dur))

# 11. Down 3 grids
dur = calc_dur_dist(3 * GRID, SPEED_CMD)
traj_cmds.append(("Down 3 Grids", -SPEED_CMD, 0, dur))

# 12. Right 2R 180
dur = calc_dur_turn(180, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Right 2R 180", SPEED_CMD, ALPHA_LARGE, dur))

# 13. Up 3 grids
dur = calc_dur_dist(3 * GRID, SPEED_CMD)
traj_cmds.append(("Up 3 Grids", SPEED_CMD, 0, dur))

# 14. Down 3 grids
dur = calc_dur_dist(3 * GRID, SPEED_CMD)
traj_cmds.append(("Down 3 Grids", -SPEED_CMD, 0, dur))

# 15. Left 2R 180
dur = calc_dur_turn(180, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Left 2R 180", SPEED_CMD, -ALPHA_LARGE, dur))

# 16. Right 2R 360
dur = calc_dur_turn(360, ALPHA_LARGE, SPEED_CMD)
traj_cmds.append(("Right 2R 360", SPEED_CMD, ALPHA_LARGE, dur))


def simulate_and_print():
    # 1. Print Code Block for Robot
    print("-" * 40)
    print("COPY THIS INTO YOUR ROBOT SCRIPT:")
    print("-" * 40)
    print("hard_coded_path = [")
    for desc, v, a, d in traj_cmds:
        print(f"    ({v}, {a}, {d:.2f}),  # {desc}")
    print("]")
    print("-" * 40)
    
    # 2. Simulate Visual Verification
    model = MyMotionModel(initial_state=[0, 0, math.pi/2], last_encoder_count=0)
    x_traj, y_traj = [0], [0]
    
    # Add Markers
    markers = [] # (x, y, label)
    
    for desc, v, a, d in traj_cmds:
        # Simulate step-by-step for the duration
        dt = 0.1
        steps = int(d / dt)
        
        # Calc physical inputs for simulation
        v_phys = get_phys_velocity(v)
        
        for _ in range(steps):
            ds = v_phys * dt
            enc = ds / 0.0001345 # approx KE
            
            # Reset encoder delta logic for sim
            model.last_encoder_count = 0
            model.step_update(enc, a, dt)
            
            x_traj.append(model.state[0])
            y_traj.append(model.state[1])
            
        markers.append((model.state[0], model.state[1], desc))

    # Plot
    plt.figure(figsize=(6, 8))
    plt.plot(x_traj, y_traj, linewidth=2)
    plt.scatter(0, 0, c='green', s=100, label='Start')
    
    # Plot segment end points
    for mx, my, mlabel in markers:
        plt.scatter(mx, my, c='red', s=10)
        
    plt.title("Hard Coded Trajectory Preview")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_and_print()