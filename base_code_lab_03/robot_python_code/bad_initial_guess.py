import numpy as np
import matplotlib.pyplot as plt
import math
import os

import data_handling
from extended_kalman_filter import ExtendedKalmanFilter

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def run_unknown_start_test(filename):
    print(f"Testing Initial Conditions on: {os.path.basename(filename)}")
    ekf_data = data_handling.get_file_data_for_kf(filename)

    true_x0 = ekf_data[0][3][0]
    true_y0 = ekf_data[0][3][1]
    true_theta0 = ekf_data[0][3][5]
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    # 1. Define Initialization Scenarios
    test_cases = {
        "Known Start (Truth)": {
            "x_0": np.array([true_x0, true_y0, true_theta0]), 
            "Sigma_0": np.diag([0.001, 0.001, 0.001]),
            "color": 'blue'
        },
        "Small Offset": {
            "x_0": np.array([true_x0 + 0.3, true_y0 - 0.3, true_theta0 + 0.5]), 
            "Sigma_0": np.diag([0.5, 0.5, 0.5]),
            "color": 'green'
        },
        "Large Offset": {
            "x_0": np.array([true_x0 - 1.0, true_y0 + 1.0, true_theta0 - 1.5]), 
            "Sigma_0": np.diag([2.0, 2.0, 2.0]),
            "color": 'orange'
        },
        "Very Bad Guess (Opposite & Far)": {
            "x_0": np.array([true_x0 + 3.0, true_y0 - 2.0, true_theta0 + 3.14]), 
            "Sigma_0": np.diag([10.0, 10.0, 10.0]), # Massive uncertainty accommodates bad guess
            "color": 'red'
        }
    }

    results = {}
    
    # 2. Run EKF for each scenario
    for case_name, params in test_cases.items():
        ekf = ExtendedKalmanFilter(params["x_0"], params["Sigma_0"], encoder_counts_0)
        
        times, est_x, est_y, est_th = [], [], [], []
        
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
            est_th.append(ekf.state_mean[2])
            
        results[case_name] = {
            'times': times, 'x': est_x, 'y': est_y, 'theta': est_th, 'color': params["color"]
        }

    # 3. Extract Ground Truth
    gt_times, gt_x, gt_y, gt_th = [], [], [], []
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        raw_z = np.array([row[3][0], row[3][1], row[3][5]])
        gt_times.append(row[0])
        gt_x.append(raw_z[0])
        gt_y.append(raw_z[1])
        gt_th.append(raw_z[2])

    # 4. Plotting
    fig = plt.figure(figsize=(12, 14))
    
    # Top Panel: Trajectory Map
    ax_traj = plt.subplot(4, 1, 1)
    ax_traj.plot(gt_x, gt_y, 'k--', alpha=0.6, linewidth=2, label='Ground Truth')
    
    for case_name, data in results.items():
        ax_traj.plot(data['x'], data['y'], color=data['color'], alpha=0.7, linewidth=2, label=case_name)
        # Mark the starting guess
        ax_traj.plot(data['x'][0], data['y'][0], color=data['color'], marker='X', markersize=10)
        
    ax_traj.set_title("Trajectory Convergence from Unknown Starts")
    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.legend()
    ax_traj.axis('equal')
    ax_traj.grid(True)

    # Bottom Panels: Error over time
    axes = [plt.subplot(4, 1, 2), plt.subplot(4, 1, 3), plt.subplot(4, 1, 4)]
    ylabels = ['X Error (m)', 'Y Error (m)', 'Theta Error (deg)']
    
    for case_name, data in results.items():
        err_x = np.array(data['x']) - np.array(gt_x)
        err_y = np.array(data['y']) - np.array(gt_y)
        err_th = [math.degrees(normalize_angle(data['theta'][i] - gt_th[i])) for i in range(len(gt_th))]
        
        axes[0].plot(data['times'], err_x, color=data['color'], alpha=0.8)
        axes[1].plot(data['times'], err_y, color=data['color'], alpha=0.8)
        axes[2].plot(data['times'], err_th, color=data['color'], alpha=0.8)

    for ax, label in zip(axes, ylabels):
        ax.set_ylabel(label)
        ax.grid(True)
        # Zoom in on the Y-axis to see the steady-state after the initial massive spike
        ax.set_ylim(-0.5, 0.5) if 'deg' not in label else ax.set_ylim(-30, 30)
        
    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_FILE = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    run_unknown_start_test(DATA_FILE)