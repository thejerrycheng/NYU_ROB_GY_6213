import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import math
import os
import time

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

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def run_simulated_occlusion_visualizer(filename):
    print(f"Loading and simulating occlusions for: {os.path.basename(filename)}")
    ekf_data = data_handling.get_file_data_for_kf(filename)

    true_x0 = ekf_data[0][3][0]
    true_y0 = ekf_data[0][3][1]
    true_theta0 = ekf_data[0][3][5]
    start_data_time = ekf_data[0][0]
    
    x_0 = [true_x0, true_y0, true_theta0]
    Sigma_0 = np.diag([0.001, 0.001, 0.001])
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    times, est_x, est_y, est_theta = [], [], [], []
    true_x, true_y, true_theta = [], [], []
    err_x, err_y, err_theta = [], [], []
    sig_x, sig_y, sig_theta = [], [], []
    occlusion_times = []

    plt.ion()
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
    
    # Track the real-world start time for synchronization
    start_wall_time = time.time()
    
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        current_time = row[0]
        delta_t = current_time - ekf_data[t-1][0]
        
        if delta_t <= 0: continue
            
        u_t = np.array([row[2].encoder_counts, row[2].steering]) 
        raw_z = np.array([row[3][0], row[3][1], row[3][5]])
        
        rel_time = current_time - start_data_time
        cycle_time = rel_time % 1.5
        
        is_occluded = cycle_time > 1.0
        
        if np.isnan(raw_z).any() or raw_z is None or np.allclose(raw_z, [0.0, 0.0, 0.0]):
            is_occluded = True

        if is_occluded:
            z_t = None
            occlusion_times.append(current_time)
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
        
        sig_x.append(3.0 * math.sqrt(ekf.state_covariance[0, 0]))
        sig_y.append(3.0 * math.sqrt(ekf.state_covariance[1, 1]))
        sig_theta.append(3.0 * math.sqrt(ekf.state_covariance[2, 2]))

        # Render animation every 2 frames to prevent Matplotlib GUI freezing
        if t % 2 == 0:
            ax_anim.cla()
            ax_anim.plot(true_x, true_y, 'k--', alpha=0.6, label='Ground Truth')
            ax_anim.plot(est_x, est_y, 'b-', linewidth=2, label='EKF Estimate')
            
            ax_anim.plot(true_x[-1], true_y[-1], 'ko', markersize=6)
            ax_anim.plot(est_x[-1], est_y[-1], 'bo', markersize=6)
            
            w, h, angle = get_ellipse_params(ekf.state_covariance, scale=3.0)
            ell_color = 'red' if is_occluded else 'green'
            ell = Ellipse(xy=(est_x[-1], est_y[-1]), width=w, height=h, angle=angle, 
                          edgecolor=ell_color, fc='None', lw=2)
            ax_anim.add_patch(ell)
            
            dx = 0.1 * math.cos(est_theta[-1])
            dy = 0.1 * math.sin(est_theta[-1])
            ax_anim.arrow(est_x[-1], est_y[-1], dx, dy, color='blue', head_width=0.02)
            
            status = "OCCLUDED (Dead Reckoning)" if is_occluded else "TRACKING (Sensor Fusion)"
            ax_anim.set_title(f'Time: {rel_time:.2f}s | Status: {status}')
            ax_anim.set_xlabel('X Position (m)')
            ax_anim.set_ylabel('Y Position (m)')
            
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='--', alpha=0.6, label='Camera GT'),
                Line2D([0], [0], color='blue', lw=2, label='EKF Trajectory'),
                Patch(edgecolor='green', facecolor='none', lw=2, label='3$\sigma$ Ellipse (Visible)'),
                Patch(edgecolor='red', facecolor='none', lw=2, label='3$\sigma$ Ellipse (Occluded)')
            ]
            ax_anim.legend(handles=legend_elements, loc='upper left')
            ax_anim.grid(True)
            ax_anim.axis('equal')
            
            ax_anim.set_xlim(est_x[-1] - 0.8, est_x[-1] + 0.8)
            ax_anim.set_ylim(est_y[-1] - 0.8, est_y[-1] + 0.8)
            
            # --- REAL-TIME SYNCHRONIZATION ---
            target_wall_time = start_wall_time + rel_time
            sleep_time = target_wall_time - time.time()
            
            if sleep_time > 0:
                plt.pause(sleep_time)
            else:
                plt.pause(0.001) # Minimum yield to keep GUI responsive

    plt.ioff()
    plt.close(fig_anim)

    print("Generating error plots with confidence bounds...")
    fig_err, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    def shade_occlusions(ax):
        if not occlusion_times: return
        blocks, current = [], [occlusion_times[0]]
        for i in range(1, len(occlusion_times)):
            if occlusion_times[i] - occlusion_times[i-1] < 0.2:
                current.append(occlusion_times[i])
            else:
                blocks.append(current)
                current = [occlusion_times[i]]
        blocks.append(current)
        for b in blocks:
            ax.axvspan(b[0], b[-1], color='gray', alpha=0.3)

    ax1.plot(times, err_x, 'k-', label='X Error')
    ax1.fill_between(times, np.array(sig_x), -np.array(sig_x), color='r', alpha=0.3, label='$\pm 3\sigma$ Bound')
    shade_occlusions(ax1)
    ax1.set_ylabel('X Error (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title(f'EKF Errors (1.0s On / 0.5s Off) - {os.path.basename(filename)}')
    
    ax2.plot(times, err_y, 'k-', label='Y Error')
    ax2.fill_between(times, np.array(sig_y), -np.array(sig_y), color='g', alpha=0.3, label='$\pm 3\sigma$ Bound')
    shade_occlusions(ax2)
    ax2.set_ylabel('Y Error (m)')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    err_theta_deg = [math.degrees(e) for e in err_theta]
    sig_theta_deg = [math.degrees(s) for s in sig_theta]
    ax3.plot(times, err_theta_deg, 'k-', label='Theta Error')
    ax3.fill_between(times, np.array(sig_theta_deg), -np.array(sig_theta_deg), color='b', alpha=0.3, label='$\pm 3\sigma$ Bound')
    shade_occlusions(ax3)
    ax3.set_ylabel('Theta Error (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    
    plt.tight_layout()
    
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "simulated_occlusion_errors.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    DATA_FILE = './data/robot_data_0_0_24_02_26_22_02_02.pkl'
    run_simulated_occlusion_visualizer(DATA_FILE)