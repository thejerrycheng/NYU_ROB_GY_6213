import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import math
import os
import time

# --------------------------------------------------
# IEEE / CRV Publication Settings (8pt, 300dpi)
# --------------------------------------------------
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.2,
    'figure.dpi': 300
})

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
    import robot_python_code

    data_dict = robot_python_code.DataLoader(filename).load()

    times = data_dict['time']
    robot_sigs = data_dict['robot_sensor_signal']
    cam_sigs = data_dict['camera_sensor_signal']

    t0 = times[0]
    ekf_data = []

    for i in range(len(times)):
        t = times[i] - t0
        u_t = np.array([robot_sigs[i].encoder_counts,
                        robot_sigs[i].steering])

        raw_cam = cam_sigs[i]

        is_occluded = (
            raw_cam is None or
            len(raw_cam) < 6 or
            np.allclose(raw_cam, [0, 0, 0, 0, 0, 0])
        )

        if is_occluded:
            z_t = None
        else:
            z_t = np.array([raw_cam[0], raw_cam[1], raw_cam[5]])

        ekf_data.append({
            'time': t,
            'u_t': u_t,
            'z_t': z_t,
            'is_occluded': is_occluded
        })

    return ekf_data


# --------------------------------------------------
# Main
# --------------------------------------------------
def run_simulated_occlusion_visualizer(filename):

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Running occlusion test on: {os.path.basename(filename)}")

    ekf_data = load_ekf_data_directly(filename)

    # Initial state from first valid measurement
    for row in ekf_data:
        if row['z_t'] is not None:
            x_0 = row['z_t']
            break

    Sigma_0 = np.eye(3) * 0.1
    encoder_counts_0 = ekf_data[0]['u_t'][0]

    ekf = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)
    dr_filter = ExtendedKalmanFilter(x_0, Sigma_0.copy(), encoder_counts_0)

    times = []
    est_x, est_y = [], []
    dr_x, dr_y = [], []
    true_x, true_y = [], []
    err_x, err_y, err_theta = [], [], []
    sig_x, sig_y, sig_theta = [], [], []
    occlusion_flags = []
    occlusion_times = []

    for i in range(1, len(ekf_data)):

        row = ekf_data[i]
        delta_t = row['time'] - ekf_data[i-1]['time']
        if delta_t <= 0:
            continue

        # Simulated periodic occlusion
        cycle_time = row['time'] % 1.5
        sim_occluded = cycle_time > 1.0

        z_t = None if sim_occluded else row['z_t']
        if sim_occluded:
            occlusion_times.append(row['time'])

        ekf.update(row['u_t'], z_t, delta_t)
        dr_filter.update(row['u_t'], None, delta_t)

        times.append(row['time'])
        occlusion_flags.append(sim_occluded)

        est_x.append(ekf.state_mean[0])
        est_y.append(ekf.state_mean[1])
        dr_x.append(dr_filter.state_mean[0])
        dr_y.append(dr_filter.state_mean[1])

        if row['z_t'] is not None:
            true_x.append(row['z_t'][0])
            true_y.append(row['z_t'][1])

            err_x.append(ekf.state_mean[0] - row['z_t'][0])
            err_y.append(ekf.state_mean[1] - row['z_t'][1])
            err_theta.append(normalize_angle(
                ekf.state_mean[2] - row['z_t'][2]))

            sig_x.append(3.0 * math.sqrt(abs(ekf.state_covariance[0, 0])))
            sig_y.append(3.0 * math.sqrt(abs(ekf.state_covariance[1, 1])))
            sig_theta.append(3.0 * math.sqrt(abs(ekf.state_covariance[2, 2])))
        else:
            true_x.append(est_x[-1])
            true_y.append(est_y[-1])

    # --------------------------------------------------
    # Trajectory Plot (3.5 × 2.7)
    # --------------------------------------------------
    fig_traj, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)

    ax.plot(dr_x, dr_y, '--', color='gray', label='Dead Reckoning')

    for j in range(1, len(est_x)):
        color = 'red' if occlusion_flags[j] else 'green'
        ax.plot(est_x[j-1:j+1],
                est_y[j-1:j+1],
                color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("EKF Under Periodic Occlusion")
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], color='green', lw=1.2, label='EKF (Visible)'),
        Line2D([0], [0], color='red', lw=1.2, label='EKF (Occluded)'),
        Line2D([0], [0], color='gray', ls='--', label='Dead Reckoning')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout(pad=0.2)

    fig_traj.savefig(
        os.path.join(save_dir, "trajectory_occlusion.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig_traj)

    # --------------------------------------------------
    # Error Plot (3.5 × 2.7)
    # --------------------------------------------------
    fig_err, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(3.5, 2.7),
        sharex=True,
        dpi=300
    )

    def shade(ax):
        for t in occlusion_times:
            ax.axvspan(t, t + 0.2, color='gray', alpha=0.25)

    ax1.plot(times, err_x, 'k-')
    ax1.fill_between(times, sig_x, -np.array(sig_x),
                     color='r', alpha=0.25)
    shade(ax1)
    ax1.set_ylabel("X Err (m)")
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2.plot(times, err_y, 'k-')
    ax2.fill_between(times, sig_y, -np.array(sig_y),
                     color='g', alpha=0.25)
    shade(ax2)
    ax2.set_ylabel("Y Err (m)")
    ax2.grid(True, linestyle=':', alpha=0.5)

    ax3.plot(times, np.degrees(err_theta), 'k-')
    ax3.fill_between(times,
                     np.degrees(sig_theta),
                     -np.degrees(sig_theta),
                     color='b', alpha=0.25)
    shade(ax3)
    ax3.set_ylabel("θ Err (°)")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(pad=0.2)

    fig_err.savefig(
        os.path.join(save_dir, "ekf_occlusion_errors.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig_err)

    print("Saved trajectory and error plots to ./results/")


# --------------------------------------------------
if __name__ == "__main__":
    DATA_FILE = './data/robot_data_40_-15_25_02_26_19_03_42.pkl'
    run_simulated_occlusion_visualizer(DATA_FILE)