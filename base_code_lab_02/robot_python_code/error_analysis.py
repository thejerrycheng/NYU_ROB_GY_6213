import numpy as np
import matplotlib.pyplot as plt
import math
import random
from motion_models import MyMotionModel, KE_VALUE, V_M, V_C, DELTA_COEFFS

# --- Configuration ---
NUM_SAMPLES = 100
TOTAL_DURATION = 100.0
SEGMENT_DURATION = 10.0
DT = 0.1

# Generate FIXED sequence (Same as before)
random.seed(42)
COMMAND_SEQUENCE = []
num_segments = int(TOTAL_DURATION / SEGMENT_DURATION)
for _ in range(num_segments):
    v_cmd = random.randint(20, 100)
    alpha_cmd = random.randint(-20, 20)
    COMMAND_SEQUENCE.append((v_cmd, alpha_cmd))

def run_deterministic_step(state, v_cmd, alpha_cmd, dt):
    """Calculates the noise-free update for Ground Truth."""
    x, y, theta = state
    
    # 1. Deterministic Physical Values
    v_det = V_M * v_cmd + V_C
    delta_det = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
    
    # 2. Kinematics (No Noise)
    L = 0.145
    w_det = (v_det * math.tan(delta_det)) / L
    
    # 3. Update
    x += dt * v_det * math.cos(theta)
    y += dt * v_det * math.sin(theta)
    theta -= dt * w_det # Right turn is negative theta
    
    return [x, y, theta]

def run_error_analysis():
    # Setup Time Axis
    time_steps = np.arange(0, TOTAL_DURATION, DT)
    num_steps = len(time_steps)
    
    # --- 1. Generate Ground Truth Trajectory ---
    gt_traj = np.zeros((num_steps, 3))
    current_state = [0, 0, math.pi/2]
    
    idx = 0
    for v_cmd, alpha_cmd in COMMAND_SEQUENCE:
        for _ in np.arange(0, SEGMENT_DURATION, DT):
            if idx >= num_steps: break
            gt_traj[idx] = current_state
            current_state = run_deterministic_step(current_state, v_cmd, alpha_cmd, DT)
            idx += 1
            
    # --- 2. Generate Monte Carlo Samples ---
    all_samples = np.zeros((NUM_SAMPLES, num_steps, 3))
    
    for i in range(NUM_SAMPLES):
        model = MyMotionModel(initial_state=[0, 0, math.pi/2], last_encoder_count=0)
        idx = 0
        for v_cmd, alpha_cmd in COMMAND_SEQUENCE:
            v_expected = V_M * v_cmd + V_C
            for _ in np.arange(0, SEGMENT_DURATION, DT):
                if idx >= num_steps: break
                
                all_samples[i, idx] = model.state
                
                ds = v_expected * DT
                curr_enc = ds / KE_VALUE 
                model.last_encoder_count = 0
                model.step_update(curr_enc, alpha_cmd, DT)
                idx += 1

    # --- 3. Calculate Errors ---
    error_x = all_samples[:, :, 0] - gt_traj[:, 0]
    error_y = all_samples[:, :, 1] - gt_traj[:, 1]
    error_th = all_samples[:, :, 2] - gt_traj[:, 2]

    # Calculate Statistics
    mu_x, std_x = np.mean(error_x, axis=0), np.std(error_x, axis=0)
    mu_y, std_y = np.mean(error_y, axis=0), np.std(error_y, axis=0)
    mu_th, std_th = np.mean(error_th, axis=0), np.std(error_th, axis=0)

    # --- 4. Plotting (IEEE Compact: 3.5 x 3.0) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 3.0), sharex=True)
    
    # Tight Font Settings
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'font.family': 'serif'
    })

    # Helper to plot bounds
    def plot_bound(ax, mean, std, color, label_text):
        ax.plot(time_steps, mean, color=color, linewidth=0.8, label='Mean')
        ax.fill_between(time_steps, mean - 3*std, mean + 3*std, 
                        color=color, alpha=0.2, label='3$\sigma$')
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(label_text, fontsize=6)
        
        # Segment Lines
        for t in range(0, int(TOTAL_DURATION), int(SEGMENT_DURATION)):
            ax.axvline(t, color='black', alpha=0.1, linewidth=0.5)

    # Plot X
    plot_bound(ax1, mu_x, std_x, 'blue', 'X Err (m)')
    ax1.set_title(f"State Error Propagation (N={NUM_SAMPLES})", pad=3)
    # Legend only on top plot to save space
    ax1.legend(loc='upper left', fontsize=5, framealpha=0.8, borderpad=0.2)

    # Plot Y
    plot_bound(ax2, mu_y, std_y, 'green', 'Y Err (m)')

    # Plot Theta
    plot_bound(ax3, mu_th, std_th, 'red', r'$\theta$ Err (rad)')
    ax3.set_xlabel("Time (s)", fontsize=7)

    # Adjust layout to remove vertical gaps
    plt.subplots_adjust(hspace=0.15)
    plt.tight_layout(pad=0.2)
    
    plt.savefig('error_evolution_compact.pdf')
    plt.show()

if __name__ == "__main__":
    run_error_analysis()