import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import math
import os
import time

# Set global matplotlib parameters for academic IEEE/CRV format
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5,
    'lines.linewidth': 1.0,
    'lines.markersize': 3
})

# Local libraries
try:
    import data_handling as data_handling_old
except ImportError:
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

def load_ekf_data_directly(filename):
    import robot_python_code
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
        
        # Occlusion Detection
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
            z_t = np.array([raw_cam[0], raw_cam[1], raw_cam[5]])
            
        ekf_data.append({'time': t, 'u_t': u_t, 'z_t': z_t, 'is_occluded': is_occluded})
        
    return ekf_data

def run_simulated_occlusion_visualizer(filename):
    print(f"Loading and simulating occlusions for: {os.path.basename(filename)}")
    
    ekf_data = load_ekf_data_directly(filename)

    # 1. Safely initialize starting position (Defaults to pi/2 facing +Y)
    true_x0, true_y0 = 0.0, 0.0
    true_theta0 = math.pi / 2 
    
    for row in ekf_data:
        if not row['is_occluded']:
            true_x0 = row['z_t'][0]
            true_y0 = row['z_t'][1]
            true_theta0 = row['z_t'][2]
            break
            
    x_0 = [true_x0, true_y0, true_theta0]
    
    # 2. Allow initial corrections
    Sigma_0 = np.eye(3) * 0.1 
    encoder_counts_0 = ekf_data[0]['u_t'][0]
    
    # 3. Initialize BOTH filters
    ekf = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)
    dr_filter = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)

    times = []
    est_x, est_y, est_theta = [], [], []
    dr_x, dr_y, dr_theta = [], [], []
    true_x, true_y, true_theta = [], [], []
    err_x, err_y, err_theta = [], [], []
    sig_x, sig_y, sig_theta = [], [], []
    occlusion_flags = [] # Track status for segmented coloring
    occlusion_times = []

    plt.ion()
    # Scaled Animation Plot to 3.5x3 inches
    fig_anim, ax_anim = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    
    start_wall_time = time.time()
    last_valid_z = [true_x0, true_y0, true_theta0]
    
    for i in range(1, len(ekf_data)):
        row = ekf_data[i]
        delta_t = row['time'] - ekf_data[i-1]['time']
        
        if delta_t <= 0: continue
            
        u_t = row['u_t']
        raw_z = row['z_t']
        
        file_occluded = row['is_occluded']
        if file_occluded:
            raw_z = last_valid_z 
        else:
            last_valid_z = raw_z
        
        cycle_time = row['time'] % 1.5
        sim_occluded = cycle_time > 1.0
        
        is_occluded = file_occluded or sim_occluded

        if is_occluded:
            z_t = None
            occlusion_times.append(row['time'])
        else:
            z_t = raw_z

        ekf.update(u_t, z_t, delta_t)
        dr_filter.update(u_t, None, delta_t)

        times.append(row['time'])
        occlusion_flags.append(is_occluded)
        
        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        est_theta.append(ekf.state_mean[2])
        
        dr_x.append(dr_filter.state_mean[0])
        dr_y.append(dr_filter.state_mean[1])
        dr_theta.append(dr_filter.state_mean[2])
        
        true_x.append(raw_z[0])
        true_y.append(raw_z[1])
        true_theta.append(raw_z[2])
        
        err_x.append(ekf.state_mean[0] - raw_z[0])
        err_y.append(ekf.state_mean[1] - raw_z[1])
        err_theta.append(normalize_angle(ekf.state_mean[2] - raw_z[2]))
        
        sig_x.append(3.0 * math.sqrt(abs(ekf.state_covariance[0, 0])))
        sig_y.append(3.0 * math.sqrt(abs(ekf.state_covariance[1, 1])))
        sig_theta.append(3.0 * math.sqrt(abs(ekf.state_covariance[2, 2])))

        # Render animation
        if i % 2 == 0:
            ax_anim.cla()
            
            ax_anim.plot(dr_x, dr_y, color='gray', linestyle='--', lw=1.0, label='DR')
            ax_anim.plot(true_x, true_y, 'k--', alpha=0.6, lw=1.0, label='Camera GT')
            ax_anim.plot(est_x, est_y, 'b-', lw=1.5, label='EKF Est')
            
            ax_anim.plot(true_x[-1], true_y[-1], 'ko', markersize=3)
            ax_anim.plot(est_x[-1], est_y[-1], 'bo', markersize=3)
            
            w, h, angle = get_ellipse_params(ekf.state_covariance, scale=3.0)
            ell_color = 'red' if is_occluded else 'green'
            ell = Ellipse(xy=(est_x[-1], est_y[-1]), width=w, height=h, angle=angle, 
                          edgecolor=ell_color, fc='None', lw=1.0)
            ax_anim.add_patch(ell)
            
            dx = 0.1 * math.cos(est_theta[-1])
            dy = 0.1 * math.sin(est_theta[-1])
            ax_anim.arrow(est_x[-1], est_y[-1], dx, dy, color='blue', head_width=0.015)
            
            status = "OCCLUDED" if is_occluded else "TRACKING"
            ax_anim.set_title(f'Time: {row["time"]:.1f}s | {status}')
            ax_anim.set_xlabel('X Position (m)')
            ax_anim.set_ylabel('Y Position (m)')
            
            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='--', lw=1.0, label='DR'),
                Line2D([0], [0], color='black', linestyle='--', alpha=0.6, lw=1.0, label='GT'),
                Line2D([0], [0], color='blue', lw=1.5, label='EKF'),
                Patch(edgecolor='green', facecolor='none', lw=1.0, label='Vis Cov'),
                Patch(edgecolor='red', facecolor='none', lw=1.0, label='Occ Cov')
            ]
            ax_anim.legend(handles=legend_elements, loc='upper left')
            ax_anim.grid(True, linestyle=':', alpha=0.6)
            ax_anim.axis('equal')
            
            ax_anim.set_xlim(est_x[-1] - 0.8, est_x[-1] + 0.8)
            ax_anim.set_ylim(est_y[-1] - 0.8, est_y[-1] + 0.8)
            
            plt.tight_layout(pad=0.5)
            
            # Sync
            target_wall_time = start_wall_time + row['time']
            sleep_time = target_wall_time - time.time()
            if sleep_time > 0:
                plt.pause(sleep_time)
            else:
                plt.pause(0.001) 

    plt.ioff()
    plt.close(fig_anim)

    # --- NEW: SAVE FULL TRAJECTORY PLOT ---
    print("Saving final trajectory plot...")
    fig_final, ax_final = plt.subplots(figsize=(3.5, 3.0), dpi=300)
    
    # 1. Dead Reckoning Trajectory
    ax_final.plot(dr_x, dr_y, color='gray', linestyle='--', lw=0.8, label='DR')
    
    # 2. Segmented EKF Trajectory (Green for Corrected, Red for Occluded)
    for j in range(1, len(times)):
        color = 'red' if occlusion_flags[j] else 'green'
        ax_final.plot(est_x[j-1:j+1], est_y[j-1:j+1], color=color, lw=1.2)
    
    ax_final.set_title(f"Full Trajectory: {os.path.basename(filename)[:15]}...", fontsize=6)
    ax_final.set_xlabel("X Position (m)", fontsize=6)
    ax_final.set_ylabel("Y Position (m)", fontsize=6)
    ax_final.tick_params(axis='both', which='major', labelsize=6)
    ax_final.grid(True, ls=':', alpha=0.5)
    ax_final.axis('equal')
    
    # Custom Legend
    leg_el = [Line2D([0], [0], color='green', lw=1.2, label='EKF (Vis)'),
              Line2D([0], [0], color='red', lw=1.2, label='EKF (Occ)'),
              Line2D([0], [0], color='gray', ls='--', lw=0.8, label='DR')]
    ax_final.legend(handles=leg_el, loc='best', fontsize=5)
    
    plt.tight_layout(pad=0.5)
    
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig_final.savefig(os.path.join(save_dir, "final_trajectory_summary.png"), dpi=300)
    plt.close(fig_final)
    # --- ERROR PLOTS (Scaled to 3.5x3 inches) ---
    print("Generating error plots with confidence bounds...")
    fig_err, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 3.0), sharex=True, dpi=300)
    
    def shade_occlusions(ax):
        if not occlusion_times: return
        blocks, current = [], [occlusion_times[0]]
        for j in range(1, len(occlusion_times)):
            if occlusion_times[j] - occlusion_times[j-1] < 0.2:
                current.append(occlusion_times[j])
            else:
                blocks.append(current)
                current = [occlusion_times[j]]
        blocks.append(current)
        for b in blocks:
            ax.axvspan(b[0], b[-1], color='gray', alpha=0.3)

    ax1.plot(times, err_x, 'k-', lw=1.0, label='X Error')
    ax1.fill_between(times, np.array(sig_x), -np.array(sig_x), color='r', alpha=0.3, label='$\pm 3\sigma$')
    shade_occlusions(ax1)
    ax1.set_ylabel('X Err (m)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title(f'EKF Errors (1.0s On/0.5s Off) - {os.path.basename(filename)[:10]}...')
    
    ax2.plot(times, err_y, 'k-', lw=1.0, label='Y Error')
    ax2.fill_between(times, np.array(sig_y), -np.array(sig_y), color='g', alpha=0.3, label='$\pm 3\sigma$')
    shade_occlusions(ax2)
    ax2.set_ylabel('Y Err (m)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    err_theta_deg = [math.degrees(e) for e in err_theta]
    sig_theta_deg = [math.degrees(s) for s in sig_theta]
    ax3.plot(times, err_theta_deg, 'k-', lw=1.0, label='Th Error')
    ax3.fill_between(times, np.array(sig_theta_deg), -np.array(sig_theta_deg), color='b', alpha=0.3, label='$\pm 3\sigma$')
    shade_occlusions(ax3)
    ax3.set_ylabel('Th Err (°)')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(pad=0.5)
    
    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "simulated_occlusion_errors.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    DATA_FILE = './data/robot_data_40_-15_25_02_26_19_03_42.pkl'
    run_simulated_occlusion_visualizer(DATA_FILE)