import numpy as np
import matplotlib.pyplot as plt
import math
from motion_models import MyMotionModel, KE_VALUE, V_M, V_C

# --- Experimental Data ---
straight_measurements = {
    '100': [(-0.15, 3.15), (-0.10, 3.00), (0.093, 3.07)],
    '75':  [(0.4, 2.55), (0.15, 2.53), (0.10, 2.62), (0.25, 2.59)],
    '50':  [(-0.10, 1.52), (-0.02, 1.58), (-0.02, 1.58)],
    '40':  [(0.013, 0.95), (0.017, 0.98), (0.07, 0.87), (-0.15, 0.97)],
}

steering_measurements = {
    '20':  [(0.50, 0.66), (0.50, 0.75)],
    '15':  [(0.40, 0.83), (0.45, 0.81), (0.45, 0.78)],
    '10':  [(0.41, 0.93), (0.39, 0.85), (0.40, 0.95)],
    '5':   [(0.30, 0.97), (0.30, 0.96), (0.29, 1.00)],
    '-5':  [(-0.13, 1.05), (-0.13, 1.03), (-0.115, 1.10)],
    '-10': [(-0.23, 1.03), (-0.17, 1.04), (-0.15, 1.02)],
    '-15': [(-0.23, 0.98), (-0.26, 1.07), (-0.21, 1.02)],
    '-20': [(-0.42, 0.92), (-0.42, 0.95), (-0.41, 0.84)],
}

# --- Configuration ---
DT = 0.05
START_HEADING = math.pi / 2 
INITIAL_STATE = [0.0, 0.0, START_HEADING]

def simulate_path(v_cmd, alpha, duration, color, label=None):
    # Initialize Model
    model = MyMotionModel(initial_state=INITIAL_STATE, last_encoder_count=0)
    
    t_list = np.arange(0, duration, DT)
    x_traj, y_traj = [], []
    
    # Calculate expected physical velocity
    v_expected = V_M * v_cmd + V_C
    current_encoder_dist = 0
    
    for _ in t_list:
        ds = v_expected * DT
        current_encoder_dist += ds / KE_VALUE
        model.step_update(current_encoder_dist, alpha, DT)
        x_traj.append(model.state[0])
        y_traj.append(model.state[1])
        
    plt.plot(x_traj, y_traj, color=color, alpha=0.6, linewidth=1.5, label=label)

def run_simulation_overlay():
    # IEEE Constraints: Width 3.5, Max Height 3.0
    plt.figure(figsize=(3.5, 3.0))
    
    # Font Size 12 for readable text
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,  # Slightly smaller to prevent overlap on small fig
        'ytick.labelsize': 8,
        'legend.fontsize': 8,   # Legend must be small to fit in 3x3 box
        'font.family': 'serif'
    })
    
    # # 1. Simulate Straight Lines (7.5s)
    # speed_colors = {100: 'blue', 75: 'cyan', 50: 'green', 40: 'lime'}
    # for speed, points in straight_measurements.items():
    #     v_cmd = int(speed)
    #     simulate_path(v_cmd, 0, 7.5, speed_colors[v_cmd])
    #     px, py = zip(*points)
    #     plt.scatter(px, py, color=speed_colors[v_cmd], marker='x', s=30, zorder=5)

    # 2. Simulate Steering Curves (5.0s, v=50)
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(-25, 25)

    for alpha_str, points in steering_measurements.items():
        alpha = int(alpha_str)
        color = cmap(norm(alpha))
        
        simulate_path(50, alpha, 5.0, color)
        
        px, py = zip(*points)
        plt.scatter(px, py, color=color, marker='o', s=20, edgecolors='black', linewidth=0.5, zorder=5)

    # Formatting
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    # Short title to save vertical space
    plt.title("Model vs. Data")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Compact Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=5),
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='None', markersize=5)
    ]
    
    # 'loc' best will try to find empty space, avoiding data
    plt.legend(custom_lines, ['Sim Left', 'Sim Right', 'Meas Straight', 'Meas Turn'], 
               loc='best', framealpha=0.8, handlelength=1.5, borderpad=0.2, labelspacing=0.2)
    
    # Tight layout with small padding to maximize graph area
    plt.tight_layout(pad=0.3)
    plt.savefig('model_vs_data_overlay.pdf')
    plt.show()

if __name__ == "__main__":
    run_simulation_overlay()