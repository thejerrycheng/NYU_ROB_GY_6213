import numpy as np
import matplotlib.pyplot as plt
import math
from motion_models import MyMotionModel, V_M, V_C

# --- Configuration ---
SPEED_CMD = 75       
ALPHA_TURN = 20      
STRAIGHT_OFFSET = -3.88 
LEFT_SCALE = 1.5   
RIGHT_SCALE = 1.0  

L = 0.145
delta_rad = 0.000027*(ALPHA_TURN**2) + 0.007798*ALPHA_TURN + 0.029847
RADIUS_R = L / math.tan(delta_rad)
GRID_SIZE = RADIUS_R 

def get_phys_velocity(v_cmd):
    mag = 0.004808 * abs(v_cmd) - 0.045557
    return mag if v_cmd > 0 else -mag

def calc_time_dist(grids, v_cmd):
    dist_m = grids * GRID_SIZE
    v_phys = get_phys_velocity(v_cmd)
    return abs(dist_m / v_phys)

def calc_time_turn(degrees, v_cmd, scale=1.0):
    angle_rad = math.radians(abs(degrees))
    arc_len = RADIUS_R * angle_rad
    v_phys = get_phys_velocity(v_cmd)
    return abs(arc_len / v_phys) * scale

# --- Define the Sequence ---
commands = []
commands.append(("Back 4 Grids", -SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(4, SPEED_CMD)))
commands.append(("Fwd 1 Grid", SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(1, SPEED_CMD)))
commands.append(("Right 90 (R)", SPEED_CMD, ALPHA_TURN, calc_time_turn(90, SPEED_CMD, scale=RIGHT_SCALE)))
commands.append(("Left 90 (R)", SPEED_CMD, -ALPHA_TURN, calc_time_turn(90, SPEED_CMD, scale=LEFT_SCALE)))
commands.append(("Fwd 1 Grid", SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(1, SPEED_CMD)))
commands.append(("Back 4 Grids", -SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(4, SPEED_CMD)))
commands.append(("Right 90 (R)", SPEED_CMD, ALPHA_TURN, calc_time_turn(90, SPEED_CMD, scale=RIGHT_SCALE)))
commands.append(("Left 90 (R)", SPEED_CMD, -ALPHA_TURN, calc_time_turn(90, SPEED_CMD, scale=LEFT_SCALE)))
commands.append(("Back 2 Grids", -SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(2, SPEED_CMD)))
commands.append(("Up 3 Grids", SPEED_CMD, STRAIGHT_OFFSET, calc_time_dist(3, SPEED_CMD)))

def plot_trajectory():
    # Set requested dimensions (7.16 / 2 = 3.58 inches wide)
    plt.figure(figsize=(3.58, 3.0))
    
    # IEEE academic plotting parameters
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'font.family': 'serif',
        'legend.fontsize': 7
    })

    model = MyMotionModel(initial_state=[0, 0, math.pi/2], last_encoder_count=0)
    x_traj, y_traj = [0], [0]
    dt = 0.05
    
    for _, v, a, dur in commands:
        steps = int(dur / dt)
        if steps == 0: continue
        v_phys = get_phys_velocity(v)
        # Using exact KE constant from your motion model
        d_enc = (v_phys * dt) / 0.0001345210 
        
        for _ in range(steps):
            model.last_encoder_count = 0
            model.step_update(d_enc, a, dt)
            x_traj.append(model.state[0])
            y_traj.append(model.state[1])

    # 
    plt.plot(x_traj, y_traj, color='blue', linewidth=1.2, label='Planned Path')
    plt.scatter(0, 0, color='green', marker='o', s=20, label='Start', zorder=5)
    plt.scatter(x_traj[-1], y_traj[-1], color='red', marker='x', s=20, label='End', zorder=5)
    
    plt.title("Tuned Fancy Trajectory Simulation")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, framealpha=0.9)
    
    plt.tight_layout(pad=0.2)
    plt.savefig('fancy_trajectory_tuned.pdf')
    plt.show()

if __name__ == "__main__":
    plot_trajectory()