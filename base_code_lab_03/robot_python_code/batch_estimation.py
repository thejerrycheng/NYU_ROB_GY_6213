import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# Local libraries
import base_code_lab_03.robot_python_code.data_handling_old as data_handling_old
from extended_kalman_filter import ExtendedKalmanFilter

# --- Constants ---
L = 0.145
KE_VALUE = 0.0001345210
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]

# Variances (used as inverse weights for the optimizer)
VAR_V = 0.00057829
VAR_DELTA = 0.00023134
Q_DIAG = np.array([0.000350, 0.000320, 0.000770])
R_DIAG = np.array([VAR_V, VAR_V, VAR_DELTA])

# Pre-compute square root of information matrices (weights)
W_meas = 1.0 / np.sqrt(Q_DIAG)
W_motion = 1.0 / np.sqrt(R_DIAG)

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_motion_prediction(x_prev, u_t):
    """Calculates x_t given x_{t-1} and u_t."""
    s = u_t[0] * KE_VALUE
    alpha = u_t[1]
    delta = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]
    
    x_next = np.zeros(3)
    x_next[0] = x_prev[0] + s * math.cos(x_prev[2])
    x_next[1] = x_prev[1] + s * math.sin(x_prev[2])
    x_next[2] = normalize_angle(x_prev[2] - (s * math.tan(delta)) / L)
    return x_next

def batch_residuals(X_flat, num_steps, u_data, z_data, is_occluded, x_prior):
    """
    Cost function for Least Squares. Stacks all temporal and spatial errors.
    """
    X = X_flat.reshape((num_steps, 3))
    residuals = []
    
    # 1. Prior Residual (Anchor the start point)
    prior_err = (X[0] - x_prior) * 1000.0 # Heavy weight on known start
    prior_err[2] = normalize_angle(prior_err[2])
    residuals.append(prior_err)
    
    for t in range(1, num_steps):
        # 2. Motion Model Residuals (Process Noise)
        x_pred = get_motion_prediction(X[t-1], u_data[t])
        motion_err = X[t] - x_pred
        motion_err[2] = normalize_angle(motion_err[2])
        residuals.append(motion_err * W_motion)
        
        # 3. Measurement Residuals (Observation Noise)
        if not is_occluded[t]:
            meas_err = z_data[t] - X[t]
            meas_err[2] = normalize_angle(meas_err[2])
            residuals.append(meas_err * W_meas)
            
    return np.concatenate(residuals)

def run_batch_optimization(filename):
    print(f"Running Full Batch Optimization vs Iterative EKF: {os.path.basename(filename)}")
    ekf_data = data_handling_old.get_file_data_for_kf(filename)

    start_time = ekf_data[0][0]
    true_x0, true_y0, true_theta0 = ekf_data[0][3][0], ekf_data[0][3][1], ekf_data[0][3][5]
    x_0 = np.array([true_x0, true_y0, true_theta0])
    
    num_steps = len(ekf_data)
    
    # Storage for batch arrays
    u_data = np.zeros((num_steps, 2))
    z_data = np.zeros((num_steps, 3))
    is_occluded = np.zeros(num_steps, dtype=bool)
    times = np.zeros(num_steps)
    
    # EKF Setup for comparison
    Sigma_0 = np.diag([0.001, 0.001, 0.001])
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, ekf_data[0][2].encoder_counts)
    ekf_states = [x_0]
    
    # Parse data and run iterative EKF
    for t in range(1, num_steps):
        row = ekf_data[t]
        current_time = row[0]
        delta_t = current_time - ekf_data[t-1][0]
        times[t] = current_time
        
        de = row[2].encoder_counts - ekf_data[t-1][2].encoder_counts
        u_t = np.array([de, row[2].steering])
        u_data[t] = u_t
        
        raw_z = np.array([row[3][0], row[3][1], row[3][5]])
        z_data[t] = raw_z
        
        # Artificial 0.5s occlusion every 1.5s
        rel_time = current_time - start_time
        occ = (rel_time % 1.5) > 1.0
        if np.isnan(raw_z).any() or np.allclose(raw_z, [0.0, 0.0, 0.0]):
            occ = True
            
        is_occluded[t] = occ
        
        # Run EKF
        z_ekf = None if occ else raw_z
        ekf.update(np.array([row[2].encoder_counts, row[2].steering]), z_ekf, delta_t)
        ekf_states.append(ekf.state_mean.copy())

    ekf_states = np.array(ekf_states)

    # --- FULL BATCH OPTIMIZATION ---
    print("Solving Nonlinear Least Squares for all states...")
    
    # Use the iterative EKF output as the initial guess to speed up convergence
    X_init_flat = ekf_states.flatten()
    
    # Optimize
    res = least_squares(batch_residuals, X_init_flat, 
                        args=(num_steps, u_data, z_data, is_occluded, x_0),
                        method='trf', loss='huber', verbose=1)
                        
    batch_states = res.x.reshape((num_steps, 3))
    print("Optimization Complete.")

    # --- PLOTTING ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    
    ax0 = axes[0]
    ax0.plot(z_data[:, 0], z_data[:, 1], 'k--', alpha=0.5, label='Camera GT')
    ax0.plot(ekf_states[:, 0], ekf_states[:, 1], 'b-', alpha=0.7, linewidth=2, label='Iterative EKF')
    ax0.plot(batch_states[:, 0], batch_states[:, 1], 'g-', linewidth=2, label='Batch Optimization')
    
    ax0.set_title("Trajectory Comparison: Batch Optimization vs Iterative EKF")
    ax0.set_xlabel("X Position (m)")
    ax0.set_ylabel("Y Position (m)")
    ax0.legend()
    ax0.axis('equal')
    ax0.grid(True)

    def shade_occlusions(ax):
        active = False
        start_t = 0
        for t in range(num_steps):
            if is_occluded[t] and not active:
                active = True
                start_t = times[t]
            elif not is_occluded[t] and active:
                active = False
                ax.axvspan(start_t, times[t], color='gray', alpha=0.3)
        if active:
            ax.axvspan(start_t, times[-1], color='gray', alpha=0.3)

    err_x_ekf = ekf_states[:, 0] - z_data[:, 0]
    err_x_batch = batch_states[:, 0] - z_data[:, 0]
    
    err_y_ekf = ekf_states[:, 1] - z_data[:, 1]
    err_y_batch = batch_states[:, 1] - z_data[:, 1]
    
    err_th_ekf = [math.degrees(normalize_angle(ekf_states[i, 2] - z_data[i, 2])) for i in range(num_steps)]
    err_th_batch = [math.degrees(normalize_angle(batch_states[i, 2] - z_data[i, 2])) for i in range(num_steps)]

    error_data = [
        (axes[1], err_x_ekf, err_x_batch, 'X Error (m)'),
        (axes[2], err_y_ekf, err_y_batch, 'Y Error (m)'),
        (axes[3], err_th_ekf, err_th_batch, 'Theta Error (deg)')
    ]

    for ax, ekf_err, batch_err, label in error_data:
        ax.plot(times, ekf_err, 'b-', alpha=0.7, label='Iterative EKF')
        ax.plot(times, batch_err, 'g-', linewidth=2, label='Batch Optimizer')
        shade_occlusions(ax)
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True)
        
    axes[3].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_FILE = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    run_batch_optimization(DATA_FILE)