import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import math
import glob
import os

# Local libraries
import robot_python_code
from extended_kalman_filter import ExtendedKalmanFilter

def get_ellipse_params(covariance_matrix, scale=3.0):
    """ Returns the width, height, and angle of the covariance ellipse """
    pos_cov = covariance_matrix[0:2, 0:2]
    eigenvalues, eigenvectors = np.linalg.eigh(pos_cov)
    
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * scale * np.sqrt(abs(eigenvalues[0]))
    height = 2 * scale * np.sqrt(abs(eigenvalues[1]))
    
    return width, height, angle

def load_ekf_data_directly(filename):
    """ Parses the .pkl file directly to avoid CSV timestamp mismatches """
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
        
        # Occlusion Detection: 
        is_occluded = False
        if raw_cam is None or len(raw_cam) < 6:
            is_occluded = True
        elif np.allclose(raw_cam, [0,0,0,0,0,0]):
            is_occluded = True
        elif i > 0 and np.allclose(raw_cam, cam_sigs[i-1]):
            is_occluded = True
            
        if is_occluded:
            z_t = None
        else:
            # Extract [X, Y, Yaw] from [X, Y, Z, Roll, Pitch, Yaw]
            z_t = np.array([raw_cam[0], raw_cam[1], raw_cam[5]])
            
        ekf_data.append({'time': t, 'u_t': u_t, 'z_t': z_t, 'is_occluded': is_occluded})
        
    return ekf_data

def run_offline_analysis(filename, save_plot=False, output_dir="./results"):
    print(f"Processing: {os.path.basename(filename)}")
    
    ekf_data = load_ekf_data_directly(filename)

    # Find the very first valid camera measurement to use as the starting pose
    true_x0, true_y0 = 0.0, 0.0
    true_theta0 = math.pi / 2  # Default to pointing +Y
    
    for row in ekf_data:
        if not row['is_occluded']:
            true_x0 = row['z_t'][0]
            true_y0 = row['z_t'][1]
            true_theta0 = row['z_t'][2]
            break
            
    x_0 = [true_x0, true_y0, true_theta0]
    
    # Increased Sigma_0 so the EKF allows itself to be corrected by the initial measurement
    Sigma_0 = np.eye(3) * 0.1 
    encoder_counts_0 = ekf_data[0]['u_t'][0]
    
    # Initialize both filters
    ekf = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)
    dr_filter = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)

    times = []
    est_x, est_y, est_theta = [], [], []
    dr_x, dr_y, dr_theta = [], [], []
    covariances = []
    dr_covariances = []
    occlusion_flags = []

    for i in range(1, len(ekf_data)):
        row = ekf_data[i]
        delta_t = row['time'] - ekf_data[i-1]['time']
        
        if delta_t <= 0:
            continue

        # Update both filters (Dead Reckoning forces z_t = None)
        ekf.update(row['u_t'], row['z_t'], delta_t)
        dr_filter.update(row['u_t'], None, delta_t)

        times.append(row['time'])
        
        # EKF States & Covariance
        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        est_theta.append(ekf.state_mean[2])
        covariances.append(ekf.state_covariance.copy())
        occlusion_flags.append(row['is_occluded'])
        
        # Dead Reckoning States & Covariance
        dr_x.append(dr_filter.state_mean[0])
        dr_y.append(dr_filter.state_mean[1])
        dr_theta.append(dr_filter.state_mean[2])
        dr_covariances.append(dr_filter.state_covariance.copy())

    # ==========================
    # DYNAMIC AESTHETIC SCALING
    # ==========================
    span_x = max(est_x) - min(est_x) if est_x else 1.0
    span_y = max(est_y) - min(est_y) if est_y else 1.0
    max_span = max(span_x, span_y, 0.5)

    dynamic_sigma = max(2.0, max_span * 1.5) 
    arrow_len = max_span * 0.05

    # ==========================
    # PLOTTING
    # ==========================
    # Set to exactly 3.5 inches wide, 3.0 inches tall
    fig, ax = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    
    # 1. Dead Reckoning Trajectory (Gray Dashed)
    ax.plot(dr_x, dr_y, color='gray', linestyle='--', linewidth=1.5, zorder=1, label='Dead Reckoning')

    # 2. Color-Coded EKF Trajectory Segments
    for i in range(1, len(times)):
        x_seg = [est_x[i-1], est_x[i]]
        y_seg = [est_y[i-1], est_y[i]]
        color = 'red' if occlusion_flags[i] else 'green'
        ax.plot(x_seg, y_seg, color=color, linewidth=1.5, zorder=3)

    # 3. Confidence Ellipses (Exactly Every 1.0 Seconds)
    last_ellipse_time = times[0] - 1.0 
    for i in range(len(times)):
        if times[i] - last_ellipse_time >= 1.0:
            
            # --- Dead Reckoning Ellipse ---
            w_dr, h_dr, angle_dr = get_ellipse_params(dr_covariances[i], scale=dynamic_sigma) 
            ell_dr = Ellipse(xy=(dr_x[i], dr_y[i]), width=w_dr, height=h_dr, angle=angle_dr, 
                             edgecolor='gray', linestyle='--', fc='None', lw=1.0, alpha=0.6, zorder=2)
            ax.add_patch(ell_dr)
            
            dx_dr = arrow_len * math.cos(dr_theta[i])
            dy_dr = arrow_len * math.sin(dr_theta[i])
            ax.arrow(dr_x[i], dr_y[i], dx_dr, dy_dr, color='gray', head_width=arrow_len*0.4, zorder=2)

            # --- EKF Ellipse ---
            w, h, angle = get_ellipse_params(covariances[i], scale=dynamic_sigma) 
            ell_color = 'red' if occlusion_flags[i] else 'green'
            ell = Ellipse(xy=(est_x[i], est_y[i]), width=w, height=h, angle=angle, 
                          edgecolor=ell_color, fc='None', lw=1.0, alpha=0.6, zorder=4)
            ax.add_patch(ell)
            
            dx = arrow_len * math.cos(est_theta[i])
            dy = arrow_len * math.sin(est_theta[i])
            ax.arrow(est_x[i], est_y[i], dx, dy, color=ell_color, head_width=arrow_len*0.4, zorder=5)
            
            last_ellipse_time = times[i]

    # Compact Legend for 3.5x3 plot
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='DR Traj'),
        Line2D([0], [0], color='green', lw=1.5, label='EKF (Corr)'),
        Line2D([0], [0], color='red', lw=1.5, label='EKF (Occ)'),
        Patch(edgecolor='gray', linestyle='--', facecolor='none', lw=1, label='DR Cov'),
        Patch(edgecolor='black', facecolor='none', lw=1, label=f'EKF {dynamic_sigma:.1f}$\sigma$')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=6)

    # Scaled down fonts
    plt.title(f'Tracking vs DR\n{os.path.basename(filename)}', fontsize=8)
    ax.set_xlabel('X Position (m)', fontsize=7)
    ax.set_ylabel('Y Position (m)', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal') 
    plt.tight_layout()
    
    if save_plot:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, f"{os.path.basename(filename).replace('.pkl', '')}_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"   Saved plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    pkl_files = glob.glob(os.path.join('./data/', '*.pkl'))
    
    if not pkl_files:
        print("No .pkl files found in ./data/")
    else:
        print(f"Found {len(pkl_files)} data files. Starting batch processing...\n")
        
        for file_path in pkl_files:
            run_offline_analysis(file_path, save_plot=True)
            
        print("\nBatch processing complete! All plots saved to ./results/")