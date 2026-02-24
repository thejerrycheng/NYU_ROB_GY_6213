import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import math
import glob
import os

# Local libraries
import data_handling
from extended_kalman_filter import ExtendedKalmanFilter

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

def run_offline_analysis(filename, save_plot=False, output_dir="./results"):
    print(f"Processing: {os.path.basename(filename)}")
    ekf_data = data_handling.get_file_data_for_kf(filename)

    true_x0 = ekf_data[0][3][0]
    true_y0 = ekf_data[0][3][1]
    true_theta0 = ekf_data[0][3][5]
    
    x_0 = [true_x0, true_y0, true_theta0]
    Sigma_0 = np.diag([0.001, 0.001, 0.001])
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    times, est_x, est_y, est_theta, meas_x, meas_y = [], [], [], [], [], []
    covariances, occlusion_times = [], []
    
    # Store coordinates specifically where occlusion happens
    occ_x, occ_y = [], []

    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        current_time = row[0]
        delta_t = current_time - ekf_data[t-1][0]
        
        if delta_t <= 0:
            continue
            
        u_t = np.array([row[2].encoder_counts, row[2].steering]) 
        raw_z = np.array([row[3][0], row[3][1], row[3][5]])
        
        is_occluded = False
        
        if np.isnan(raw_z).any() or raw_z is None:
            is_occluded = True
        elif np.allclose(raw_z, [0.0, 0.0, 0.0]):
            is_occluded = True
            
        if is_occluded:
            z_t = None
            occlusion_times.append(current_time)
            # Track the EKF's estimated spatial location during the outage
            occ_x.append(ekf.state_mean[0])
            occ_y.append(ekf.state_mean[1])
        else:
            z_t = raw_z

        ekf.update(u_t, z_t, delta_t)

        times.append(current_time)
        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        est_theta.append(ekf.state_mean[2])
        covariances.append(ekf.state_covariance.copy())
        
        meas_x.append(raw_z[0])
        meas_y.append(raw_z[1])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(meas_x, meas_y, 'k--', alpha=0.5)
    ax.plot(est_x, est_y, 'b-', linewidth=2, zorder=2)

    # Clearly mark the exact spatial coordinates of the occlusions
    if occ_x:
        ax.scatter(occ_x, occ_y, color='red', marker='x', s=40, zorder=4, label='Camera Occluded')

    step_size = max(1, len(times) // 20) 
    for i in range(0, len(times), step_size):
        w, h, angle = get_ellipse_params(covariances[i], scale=3.0) 
        ell_color = 'red' if times[i] in occlusion_times else 'green'
        
        ell = Ellipse(xy=(est_x[i], est_y[i]), width=w, height=h, angle=angle, 
                      edgecolor=ell_color, fc='None', lw=1.5, alpha=0.6, zorder=3)
        ax.add_patch(ell)
        
        dx = 0.05 * math.cos(est_theta[i])
        dy = 0.05 * math.sin(est_theta[i])
        ax.arrow(est_x[i], est_y[i], dx, dy, color='blue', head_width=0.01, zorder=3)

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Camera (Truth)'),
        Line2D([0], [0], color='blue', lw=2, label='EKF Trajectory'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markeredgecolor='red', markersize=8, label='Occlusion Zone'),
        Patch(edgecolor='green', facecolor='none', lw=1.5, label='3$\sigma$ Ellipse (Measured)')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.title(f'EKF Performance - {os.path.basename(filename)}')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)
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