import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import math
import os
from scipy.optimize import least_squares

# --------------------------------------------------
# IEEE / CRV Publication Settings (8pt, 300dpi)
# --------------------------------------------------
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'lines.linewidth': 1.0,
    'lines.markersize': 2,
    'figure.dpi': 300
})

# Local libraries
import robot_python_code
from extended_kalman_filter import ExtendedKalmanFilter

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_ellipse_params(covariance_matrix, scale=3.0):
    pos_cov = covariance_matrix[0:2, 0:2]
    eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
    
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * scale * np.sqrt(abs(eigenvalues[0]))
    height = 2 * scale * np.sqrt(abs(eigenvalues[1]))
    
    return width, height, angle

# --------------------------------------------------
# Load Data
# --------------------------------------------------
def load_ekf_data_directly(filename):
    data_dict = robot_python_code.DataLoader(filename).load()
    times = data_dict['time']
    robot_sigs = data_dict['robot_sensor_signal']
    cam_sigs = data_dict['camera_sensor_signal']
    
    t0 = times[0]
    ekf_data = []
    
    for i in range(len(times)):
        t = times[i] - t0
        u_t = np.array([robot_sigs[i].encoder_counts, robot_sigs[i].steering])
        raw_cam = cam_sigs[i]
        
        is_occluded = False
        if raw_cam is None or len(raw_cam) < 6:
            is_occluded = True
        elif np.allclose(raw_cam, [0,0,0,0,0,0]):
            is_occluded = True
        elif i > 0 and np.allclose(raw_cam, cam_sigs[i-1]):
            is_occluded = True
            
        z_t = None if is_occluded else np.array([raw_cam[0], raw_cam[1], raw_cam[5]])
        ekf_data.append({'time': t, 'u_t': u_t, 'z_t': z_t, 'is_occluded': is_occluded})
        
    return ekf_data

# --------------------------------------------------
# Batch Optimization Residuals
# --------------------------------------------------
def compute_residuals(X_flat, ekf_data, times_full, Q_inv_sqrt, R_inv_sqrt, L0_inv_sqrt, x0_prior):
    num_states = len(times_full)
    X = X_flat.reshape((num_states, 3))
    res = []
    
    # Prior Residual
    diff_prior = X[0] - x0_prior
    diff_prior[2] = normalize_angle(diff_prior[2])
    res.append(L0_inv_sqrt @ diff_prior)
    
    for t in range(1, num_states):
        u_t = ekf_data[t]['u_t']
        u_t_prev = ekf_data[t-1]['u_t']
        
        # ==========================================================
        # TODO: Replace with your actual calibrated constants from Lab 2
        # ==========================================================
        L = 0.32          # Wheelbase (meters)
        K_e = 0.0015      # Encoder ticks to meters
        c0, c1, c2 = 0.0, 1.0, 0.0 # Steering calibration polynomial
        # ==========================================================
        
        delta_ticks = u_t[0] - u_t_prev[0]
        s_t = K_e * delta_ticks
        
        alpha_t = u_t[1]
        delta_t_steer = c0 * alpha_t**2 + c1 * alpha_t + c2
        
        # Motion Prediction
        x_prev = X[t-1]
        pred_x = x_prev[0] + s_t * math.cos(x_prev[2])
        pred_y = x_prev[1] + s_t * math.sin(x_prev[2])
        pred_th = x_prev[2] - (s_t / L) * math.tan(delta_t_steer)
        
        # Motion Residual
        diff_motion = X[t] - np.array([pred_x, pred_y, pred_th])
        diff_motion[2] = normalize_angle(diff_motion[2])
        res.append(R_inv_sqrt @ diff_motion)
        
        # Measurement Residual
        z_t = ekf_data[t]['z_t']
        if z_t is not None:
            diff_meas = z_t - X[t]
            diff_meas[2] = normalize_angle(diff_meas[2])
            res.append(Q_inv_sqrt @ diff_meas)
            
    return np.concatenate(res)

# --------------------------------------------------
# Main Analysis
# --------------------------------------------------
def run_offline_analysis(filename, save_plot=False, output_dir="./results"):
    print(f"\nProcessing: {os.path.basename(filename)}")
    ekf_data = load_ekf_data_directly(filename)

    true_x0, true_y0, true_theta0 = 0.0, 0.0, math.pi / 2 
    for row in ekf_data:
        if not row['is_occluded']:
            true_x0, true_y0, true_theta0 = row['z_t']
            break
            
    x_0 = [true_x0, true_y0, true_theta0]
    Sigma_0 = np.eye(3) * 0.1 
    encoder_counts_0 = ekf_data[0]['u_t'][0]
    
    # -------------------
    # ITERATIVE EKF PASS
    # -------------------
    print("   [+] Running Iterative EKF...")
    ekf = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)
    dr_filter = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)

    times = []
    est_x, est_y, est_theta = [], [], []
    dr_x, dr_y, dr_theta = [], [], []
    covariances, occlusion_flags, occlusion_times = [], [], []
    
    err_x, err_y, err_theta = [], [], []
    sig_x, sig_y, sig_theta = [], [], []
    last_z_t = x_0

    for i in range(1, len(ekf_data)):
        row = ekf_data[i]
        delta_t = row['time'] - ekf_data[i-1]['time']
        if delta_t <= 0: continue

        ekf.update(row['u_t'], row['z_t'], delta_t)
        dr_filter.update(row['u_t'], None, delta_t)

        times.append(row['time'])
        
        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        est_theta.append(ekf.state_mean[2])
        covariances.append(ekf.state_covariance.copy())
        occlusion_flags.append(row['is_occluded'])
        
        dr_x.append(dr_filter.state_mean[0])
        dr_y.append(dr_filter.state_mean[1])
        dr_theta.append(dr_filter.state_mean[2])
        
        if row['z_t'] is not None:
            last_z_t = row['z_t']
        else:
            occlusion_times.append(row['time'])
            
        err_x.append(ekf.state_mean[0] - last_z_t[0])
        err_y.append(ekf.state_mean[1] - last_z_t[1])
        err_theta.append(normalize_angle(ekf.state_mean[2] - last_z_t[2]))

        sig_x.append(3.0 * math.sqrt(abs(ekf.state_covariance[0, 0])))
        sig_y.append(3.0 * math.sqrt(abs(ekf.state_covariance[1, 1])))
        sig_theta.append(3.0 * math.sqrt(abs(ekf.state_covariance[2, 2])))

    # -------------------
    # BATCH OPTIMIZATION
    # -------------------
    print("   [+] Initializing Batch Optimizer using the Iterative EKF results...")
    
    Q_cov = np.diag([0.05**2, 0.05**2, np.deg2rad(2.0)**2])
    R_cov = np.diag([0.02**2, 0.02**2, np.deg2rad(1.0)**2])
    
    Q_inv_sqrt = np.linalg.cholesky(np.linalg.inv(Q_cov))
    R_inv_sqrt = np.linalg.cholesky(np.linalg.inv(R_cov))
    L0_inv_sqrt = np.linalg.cholesky(np.linalg.inv(Sigma_0))
    
    times_full = [ekf_data[0]['time']] + times
    
    # Explicitly build the initial guess matrix using the EKF state history
    ekf_trajectory_guess = np.column_stack((est_x, est_y, est_theta))
    X_init = np.vstack([x_0, ekf_trajectory_guess]).flatten()
    
    print("   [+] Starting Least Squares (Huber Loss)...")
    res = least_squares(
        compute_residuals, 
        X_init, 
        args=(ekf_data, times_full, Q_inv_sqrt, R_inv_sqrt, L0_inv_sqrt, x_0),
        method='trf',
        loss='huber',
        verbose=2
    )
    
    print(f"   [+] Optimization Finished. Status: {res.message}")
    
    batch_traj = res.x.reshape((len(times_full), 3))
    batch_x, batch_y, batch_theta = batch_traj[1:, 0], batch_traj[1:, 1], batch_traj[1:, 2]

    # Calculate Batch Errors relative to the last known valid ground truth
    batch_err_x, batch_err_y, batch_err_theta = [], [], []
    last_z_t_batch = x_0
    
    for i in range(1, len(ekf_data)):
        if ekf_data[i]['z_t'] is not None:
            last_z_t_batch = ekf_data[i]['z_t']
        
        batch_err_x.append(batch_x[i-1] - last_z_t_batch[0])
        batch_err_y.append(batch_y[i-1] - last_z_t_batch[1])
        batch_err_theta.append(normalize_angle(batch_theta[i-1] - last_z_t_batch[2]))

    # ==========================
    # 1. TRAJECTORY PLOT
    # ==========================
    fig_traj, ax_traj = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    
    ax_traj.plot(dr_x, dr_y, color='gray', linestyle='--', linewidth=1.2, label='DR')
    
    for i in range(1, len(times)):
        color = 'red' if occlusion_flags[i] else 'green'
        ax_traj.plot(est_x[i-1:i+1], est_y[i-1:i+1], color=color, linewidth=1.5, zorder=3)
        
    ax_traj.plot(batch_x, batch_y, color='blue', linewidth=1.5, linestyle='-.', label='Batch MAP', zorder=4)

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', lw=1.2, label='DR Traj'),
        Line2D([0], [0], color='green', lw=1.5, label='EKF (Corr)'),
        Line2D([0], [0], color='red', lw=1.5, label='EKF (Occ)'),
        Line2D([0], [0], color='blue', linestyle='-.', lw=1.5, label='Batch MAP')
    ]
    ax_traj.legend(handles=legend_elements, loc='best')

    ax_traj.set_title(f'Tracking vs DR vs Batch\n{os.path.basename(filename)}')
    ax_traj.set_xlabel('X Position (m)')
    ax_traj.set_ylabel('Y Position (m)')
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    ax_traj.axis('equal') 
    plt.tight_layout(pad=0.2)

    # ==========================
    # 2. ERROR PLOT
    # ==========================
    fig_err, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 3.0), sharex=True, dpi=300)

    def shade(ax):
        for t in occlusion_times:
            ax.axvspan(t, t + 0.1, color='gray', alpha=0.25)

    ax1.plot(times, err_x, 'k-', label='EKF Error')
    ax1.plot(times, batch_err_x, 'b-.', label='Batch Error')
    ax1.fill_between(times, sig_x, -np.array(sig_x), color='r', alpha=0.25)
    shade(ax1)
    ax1.set_ylabel("X Err (m)")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_title(f'EKF & Batch Errors - {os.path.basename(filename)[:15]}...')
    ax1.legend(loc='upper right')

    ax2.plot(times, err_y, 'k-')
    ax2.plot(times, batch_err_y, 'b-.')
    ax2.fill_between(times, sig_y, -np.array(sig_y), color='g', alpha=0.25)
    shade(ax2)
    ax2.set_ylabel("Y Err (m)")
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax3.plot(times, np.degrees(err_theta), 'k-')
    ax3.plot(times, np.degrees(batch_err_theta), 'b-.')
    ax3.fill_between(times, np.degrees(sig_theta), -np.degrees(sig_theta), color='b', alpha=0.25)
    shade(ax3)
    ax3.set_ylabel("θ Err (°)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(pad=0.2)

    if save_plot:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.basename(filename).replace('.pkl', '')
        fig_traj.savefig(os.path.join(output_dir, f"{base_name}_batch_traj.png"), dpi=300, bbox_inches='tight')
        fig_err.savefig(os.path.join(output_dir, f"{base_name}_batch_err.png"), dpi=300, bbox_inches='tight')
        plt.close('all')
    else:
        print("   [!] Close the plot windows to continue to the next file.")
        plt.show()

# --------------------------------------------------
if __name__ == "__main__":
    # Explicitly process only these two files
    target_files = [
        './data/robot_data_0_0_25_02_26_18_55_40.pkl',
        './data/robot_data_0_0_25_02_26_20_29_23.pkl'
    ]
    
    for file_path in target_files:
        if os.path.exists(file_path):
            run_offline_analysis(file_path, save_plot=False)
        else:
            print(f"Error: Could not find file {file_path}")
            
    print("\nBatch processing complete!")