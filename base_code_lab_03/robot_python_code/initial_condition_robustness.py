import numpy as np
import matplotlib.pyplot as plt
import math

import data_handling
from extended_kalman_filter import ExtendedKalmanFilter


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_error_norm(x_est, x_true):
    dx = x_est[0] - x_true[0]
    dy = x_est[1] - x_true[1]
    dtheta = normalize_angle(x_est[2] - x_true[2])
    return np.sqrt(dx**2 + dy**2 + dtheta**2)


def run_robustness_test(filename):

    print("Running EKF Robustness Test (First 1s)")

    ekf_data, _ = data_handling.get_file_data_for_kf(filename)

    # ---------------------------------------
    # Use first valid measurement as ground truth
    # ---------------------------------------
    start_index = None
    for i in range(len(ekf_data)):
        if ekf_data[i][3] is not None:
            start_index = i
            true_state0 = np.array(ekf_data[i][3])
            break

    if start_index is None:
        raise RuntimeError("No valid camera measurements found.")

    # Progressive initial error magnitudes
    error_levels = [0.25, 0.5, 1.0, 2.0]
    # -----------------------------------------
    # Publication-quality figure (IEEE column)
    # -----------------------------------------

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig = plt.figure(figsize=(3.5, 2.7))

    for err_mag in error_levels:

        x_0 = true_state0 + np.array([
            err_mag,
            -err_mag,
            np.deg2rad(20 * err_mag)
        ])

        Sigma_0 = np.diag([1.0, 1.0, np.deg2rad(45)**2])
        encoder_counts_0 = ekf_data[start_index][2].encoder_counts
        ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

        error_norm_history = []
        time_history = []

        t0 = ekf_data[start_index][0]

        for t in range(start_index + 1, len(ekf_data)):

            row = ekf_data[t]
            prev_row = ekf_data[t - 1]
            delta_t = row[0] - prev_row[0]

            if delta_t <= 0:
                continue

            current_time = row[0] - t0
            if current_time > 1.0:
                break

            u_t = np.array([
                row[2].encoder_counts,
                row[2].steering
            ])

            z_t = row[3]
            ekf.update(u_t, z_t, delta_t)

            if z_t is not None:
                true_state = np.array(z_t)
                error_norm = compute_error_norm(
                    ekf.state_mean,
                    true_state
                )
                error_norm_history.append(error_norm)
                time_history.append(current_time)

        plt.plot(time_history,
                error_norm_history,
                linewidth=1.4,
                label=f"{err_mag:.2f} m")

    plt.xlim([0, 1.0])
    plt.xlabel("Time (s)")
    plt.ylabel("State Error Norm")
    plt.title("EKF Convergence from Poor Initialization")
    plt.grid(True)
    plt.legend(frameon=False)

    plt.tight_layout(pad=0.2)

    save_path = "ekf_initial_condition_robustness.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    plt.show()

    # -----------------------------------------
    # Save high-resolution figure
    # -----------------------------------------
    save_path = "ekf_initial_condition_robustness.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    DATA_FILE = './data/robot_data_40_-15_25_02_26_19_03_42.pkl'
    run_robustness_test(DATA_FILE)