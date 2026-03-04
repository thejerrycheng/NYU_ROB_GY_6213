import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
import os

import base_code_lab_03.robot_python_code.data_handling_old as data_handling_old
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

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def run_visualizer(filename):
    print(f"Loading data from {filename}...")
    ekf_data = data_handling_old.get_file_data_for_kf(filename)

    true_x0 = ekf_data[0][3][0]
    true_y0 = ekf_data[0][3][1]
    true_theta0 = ekf_data[0][3][5]
    
    x_0 = [true_x0, true_y0, true_theta0]
    Sigma_0 = np.diag([0.001, 0.001, 0.001])
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    times, est_x, est_y, est_theta = [], [], [], []
    true_x, true_y, true_theta = [], [], []
    err_x, err_y, err_theta = [], [], []
    sig_x, sig_y, sig_theta = [], [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        current_time = row[0]
        delta_t = current_time - ekf_data[t-1][0]
        
        if delta_t <= 0: continue
            
        u_t = np.array([row[2].encoder_counts, row[2].steering]) 
        raw_z = np.array([row[3][0], row[3][1], row[3][5]])
        
        if np.isnan(raw_z).any() or raw_z is None or np.allclose(raw_z, [0.0, 0.0, 0.0]):
            z_t = None
        else:
            z_t = raw_z

        ekf.update(u_t, z_t, delta_t)

        times.append(current_time)
        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        est_theta.append(ekf.state_mean[2])
        
        true_x.append(raw_z[0])
        true_y.append(raw_z[1])
        true_theta.append(raw_z[2])
        
        err_x.append(ekf.state_mean[0] - raw_z[0])
        err_y.append(ekf.state_mean[1] - raw_z[1])
        err_theta.append(normalize_angle(ekf.state_mean[2] - raw_z[2]))
        
        # Store 3-sigma bounds for the error plots
        sig_x.append(3.0 * math.sqrt(ekf.state_covariance[0, 0]))
        sig_y.append(3.0 * math.sqrt(ekf.state_covariance[1, 1]))
        sig_theta.append(3.0 * math.sqrt(ekf.state_covariance[2, 2]))

        if t % 2 == 0:
            ax.cla()
            ax.plot(true_x, true_y, 'k--', alpha=0.6, label='Ground Truth')
            ax.plot(est_x, est_y, 'b-', linewidth=2, label='EKF Estimate')
            ax.plot(true_x[-1], true_y[-1], 'ko')
            ax.plot(est_x[-1], est_y[-1], 'bo')
            
            w, h, angle = get_ellipse_params(ekf.state_covariance, scale=3.0)
            ell = Ellipse(xy=(est_x[-1], est_y[-1]), width=w, height=h, angle=angle, 
                          edgecolor='g', fc='None', lw=2, label='3$\sigma$ Confidence')
            ax.add_patch(ell)
            
            ax.set_title(f'Live EKF Tracking - Time: {current_time:.2f}s')
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.legend(loc='upper left')
            ax.grid(True)
            ax.axis('equal')
            
            # Dynamic window tracking the robot
            ax.set_xlim(est_x[-1] - 1.0, est_x[-1] + 1.0)
            ax.set_ylim(est_y[-1] - 1.0, est_y[-1] + 1.0)
            
            plt.pause(0.01)

    plt.ioff()
    plt.close(fig)

    print("Generating error plots with confidence bounds...")
    fig_err, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # X Error
    ax1.plot(times, err_x, 'r-', label='X Error')
    ax1.fill_between(times, np.array(sig_x), -np.array(sig_x), color='r', alpha=0.2, label='$\pm 3\sigma$ Bound')
    ax1.set_ylabel('X Error (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title(f'EKF Estimation Errors - {os.path.basename(filename)}')
    
    # Y Error
    ax2.plot(times, err_y, 'g-', label='Y Error')
    ax2.fill_between(times, np.array(sig_y), -np.array(sig_y), color='g', alpha=0.2, label='$\pm 3\sigma$ Bound')
    ax2.set_ylabel('Y Error (m)')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Theta Error (converted to degrees)
    err_theta_deg = [math.degrees(e) for e in err_theta]
    sig_theta_deg = [math.degrees(s) for s in sig_theta]
    ax3.plot(times, err_theta_deg, 'b-', label='Theta Error')
    ax3.fill_between(times, np.array(sig_theta_deg), -np.array(sig_theta_deg), color='b', alpha=0.2, label='$\pm 3\sigma$ Bound')
    ax3.set_ylabel('Theta Error (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    plt.tight_layout()
    
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    base_name = os.path.basename(filename).replace('.pkl', '')
    save_path = os.path.join(save_dir, f"{base_name}_errors.png")
    plt.savefig(save_path, dpi=300)
    print(f"Error plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    DATA_FILE = './data/robot_data_0_0_24_02_26_22_02_02.pkl'
    run_visualizer(DATA_FILE)