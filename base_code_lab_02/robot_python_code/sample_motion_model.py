import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import random
from motion_models import MyMotionModel, KE_VALUE, V_M, V_C

# --- Configuration ---
NUM_SAMPLES = 100       # Monte Carlo Samples
TOTAL_DURATION = 100.0  # Seconds
SEGMENT_DURATION = 10.0 # Seconds per command change
DT = 0.1                # Integration step

# Generate a FIXED sequence of random commands for all samples to follow
random.seed(42) # Fixed seed for reproducibility
COMMAND_SEQUENCE = []
num_segments = int(TOTAL_DURATION / SEGMENT_DURATION)

for _ in range(num_segments):
    v_cmd = random.randint(20, 100)      # Velocity: 20-100
    alpha_cmd = random.randint(-20, 20)  # Steering: -20 to 20
    COMMAND_SEQUENCE.append((v_cmd, alpha_cmd))

def get_covariance_ellipse(x_points, y_points, n_std=3.0, color='red'):
    """Calculates the ellipse patch for a given set of (x,y) points."""
    if len(x_points) < 2: return None
    cov = np.cov(x_points, y_points)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    angle = np.rad2deg(np.arccos(v[0, 0]))
    
    ell = Ellipse(xy=(np.mean(x_points), np.mean(y_points)),
                  width=lambda_[0] * 2 * n_std,
                  height=lambda_[1] * 2 * n_std,
                  angle=angle, edgecolor=color, fc='none', lw=1.5, ls='--')
    return ell

def run_complex_simulation():
    # IEEE Column Width (3.5 inches)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    
    # 1. General Font Settings (Size 8)
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6,
        'font.family': 'serif'
    })

    # Storage for "Waypoints" to plot ellipses later
    # Format: waypoint_idx -> { 'x': [], 'y': [] }
    # Index 0 is Start, Index 1 is end of segment 1, etc.
    waypoints_data = {i: {'x': [], 'y': []} for i in range(num_segments + 1)}
    
    # Initialize Start Point (0,0) for all samples
    waypoints_data[0]['x'] = [0] * NUM_SAMPLES
    waypoints_data[0]['y'] = [0] * NUM_SAMPLES

    # --- Run Monte Carlo Samples ---
    for i in range(NUM_SAMPLES):
        # Initialize Model (Start facing +Y)
        model = MyMotionModel(initial_state=[0, 0, math.pi/2], last_encoder_count=0)
        
        x_traj, y_traj = [0], [0]
        
        for seg_idx, (v_cmd, alpha_cmd) in enumerate(COMMAND_SEQUENCE):
            v_expected = V_M * v_cmd + V_C
            
            for _ in np.arange(0, SEGMENT_DURATION, DT):
                # Fake encoder accumulation
                ds = v_expected * DT
                curr_enc = ds / KE_VALUE 
                model.last_encoder_count = 0 
                model.step_update(curr_enc, alpha_cmd, DT)
                
                x_traj.append(model.state[0])
                y_traj.append(model.state[1])
            
            # Store Waypoint (End of Segment)
            waypoints_data[seg_idx+1]['x'].append(model.state[0])
            waypoints_data[seg_idx+1]['y'].append(model.state[1])

        # Plot faint trajectory
        ax.plot(x_traj, y_traj, color='blue', alpha=0.05, linewidth=0.3)

    # --- Plot Ellipses and Labels ---
    # Start point dot
    ax.scatter(0, 0, color='black', s=5, zorder=5)

    for i in range(num_segments):
        # 1. Get Mean Position of the START of this segment
        start_x_mean = np.mean(waypoints_data[i]['x'])
        start_y_mean = np.mean(waypoints_data[i]['y'])
        
        # 2. Add Label for the UPCOMING command
        v_c, a_c = COMMAND_SEQUENCE[i]
        label_text = f"[{v_c}, {a_c}$^\circ$]"
        
        # Font Size 5 for Labels
        ax.text(start_x_mean, start_y_mean, label_text, 
                fontsize=5, color='black', 
                ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, pad=0.3, edgecolor='none', boxstyle='round'),
                zorder=20)

        # 3. Draw Ellipse at the END of this segment
        x_pts = waypoints_data[i+1]['x']
        y_pts = waypoints_data[i+1]['y']
        
        ellipse = get_covariance_ellipse(x_pts, y_pts, n_std=3.0, color='red')
        if ellipse: ax.add_patch(ellipse)
        
        # Plot mean point
        ax.scatter(np.mean(x_pts), np.mean(y_pts), color='black', s=5, zorder=5)

    # Formatting
    ax.set_title(f"Monte Carlo Propagation (N={NUM_SAMPLES})\n10s Intervals, Random Cmds")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.axis('equal')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', alpha=0.5, lw=1),
        Line2D([0], [0], color='red', linestyle='--', lw=1.5),
        Line2D([0], [0], marker='o', color='black', markersize=3, linestyle='None')
    ]
    ax.legend(custom_lines, ['Sample Traj', '3$\sigma$ Covariance', 'Mean Pos'], 
              loc='best', framealpha=0.9)
    
    plt.tight_layout(pad=0.2)
    plt.savefig('complex_trajectory_covariance.pdf')
    plt.show()

if __name__ == "__main__":
    run_complex_simulation()