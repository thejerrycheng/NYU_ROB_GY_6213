import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

import data_handling as data_handling_old 
from extended_kalman_filter import ExtendedKalmanFilter


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def run_unknown_start_test(filename):

    print(f"Testing Initial Conditions on: {os.path.basename(filename)}")

    # --- CORRECT UNPACKING ---
    ekf_data, cam_pos_world = data_handling_old.get_file_data_for_kf(filename)

    # --------------------------------------------------
    # Find first valid camera measurement
    # --------------------------------------------------
    first_valid_idx = None
    for i in range(len(ekf_data)):
        if ekf_data[i][3] is not None:
            first_valid_idx = i
            break

    if first_valid_idx is None:
        raise RuntimeError("No valid camera measurements found in dataset.")

    # Extract initial pose from first valid measurement
    true_x0, true_y0, true_theta0 = ekf_data[first_valid_idx][3]
    start_time = ekf_data[first_valid_idx][0]

    x_0 = np.array([true_x0, true_y0, true_theta0])

    print("Initialized from first valid camera measurement:")
    print("x0 =", x_0)

    # --------------------------------------------------
    # EKF Setup
    # --------------------------------------------------
    Sigma_0 = np.diag([0.1, 0.1, 0.1])
    encoder_counts_0 = ekf_data[first_valid_idx][2].encoder_counts

    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    states = [x_0]

    # --------------------------------------------------
    # Run EKF
    # --------------------------------------------------
    for t in range(first_valid_idx + 1, len(ekf_data)):

        current_row = ekf_data[t]
        previous_row = ekf_data[t - 1]

        delta_t = current_row[0] - previous_row[0]
        if delta_t <= 0:
            continue

        u_t = np.array([
            current_row[2].encoder_counts,
            current_row[2].steering
        ])

        z_t = current_row[3]  # already [x, y, yaw] or None

        ekf.update(u_t, z_t, delta_t)
        states.append(ekf.state_mean.copy())

    states = np.array(states)

    # --------------------------------------------------
    # Plot Results
    # --------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'b-', label='EKF Estimate')

    if cam_pos_world is not None:
        plt.plot(cam_pos_world[0], cam_pos_world[1],
                 'rs', markersize=10, label='Camera')

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("EKF with Unknown Initial Guess")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    DATA_FILE = './data/robot_data_40_-15_25_02_26_19_03_42.pkl'
    run_unknown_start_test(DATA_FILE)