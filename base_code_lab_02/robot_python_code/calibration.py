import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Section 4: Straight Line Measurements (Steering = 0)
# Format: 'filename': (measured_x, measured_y)
# Note: Euclidean distance sqrt(x^2 + y^2) is used for Ke calibration.
# --- FULL INPUT DATASET ---

# Section 4: Straight Line Measurements (Steering = 0)
# Format: 'filename': (measured_x, measured_y)
# Calculated as s = sqrt(x^2 + y^2)
straight_measurements = {
    # Speed 100 Group (v = 100, alpha = 0)
    # 'robot_data_100_0_09_02_26_22_04_39': (0.02, 1.15),
    # 'robot_data_100_0_09_02_26_22_05_38': (-0.01, 1.14),
    # 'robot_data_100_0_09_02_26_22_07_32': (0.01, 1.16),
    # 'robot_data_100_0_09_02_26_22_12_05': (0.03, 1.13),
    # 'robot_data_100_0_09_02_26_22_13_36': (-0.02, 1.15),
    'robot_data_100_0_09_02_26_22_16_06': (-0.15, 3.15),
    'robot_data_100_0_09_02_26_22_16_50': (-0.10, 3.00),
    'robot_data_100_0_09_02_26_22_17_53': (0.093, 3.07),
    
    # Speed 75 Group (v = 75, alpha = 0)
    'robot_data_75_0_09_02_26_22_18_20':  (0.4, 2.55),
    'robot_data_75_0_09_02_26_22_19_08':  (0.15, 2.53),
    'robot_data_75_0_09_02_26_22_19_41':  (0.10, 2.62),
    'robot_data_75_0_09_02_26_22_20_29':  (0.25, 2.59),
    
    # Speed 50 Group (v = 50, alpha = 0)
    'robot_data_50_0_09_02_26_22_21_11':  (-0.10, 1.52),
    'robot_data_50_0_09_02_26_22_22_09':  (-0.02, 1.58),
    'robot_data_50_0_09_02_26_22_23_00':  (-0.02, 1.58),
    
    # Speed 40 Group (v = 40, alpha = 0)
    'robot_data_40_0_09_02_26_22_25_13':  (0.013, 0.95),
    'robot_data_40_0_09_02_26_22_25_33':  (0.017, 0.98),
    'robot_data_40_0_09_02_26_22_26_14':  (0.07, 0.87),
    'robot_data_40_0_09_02_26_22_27_11':  (-0.15, 0.97),
    # 'robot_data_40_0_09_02_26_22_27_42':  (0.00, 0.45),
    # 'robot_data_40_0_09_02_26_22_28_21':  (-0.01, 0.44),
}

# Section 5: Steering Measurements
# Format: 'filename': (measured_final_x, measured_final_y)
# R = (x^2 + y^2) / 2x
steering_measurements = {
    # --- POSITIVE STEERING (Left Turns) ---
    
    # v = 50, alpha = 20 deg (Hard Left)
    'robot_data_50_20_09_02_26_22_48_25': (0.50, 0.66),
    'robot_data_50_20_09_02_26_22_48_34': (0.50, 0.75),
    
    # v = 50, alpha = 15 deg (Left)
    'robot_data_50_15_09_02_26_22_51_17': (0.40, 0.83),
    'robot_data_50_15_09_02_26_22_52_01': (0.45, 0.81),
    'robot_data_50_15_09_02_26_22_52_53': (0.45, 0.78),
    
    # v = 50, alpha = 10 deg (Soft Left)
    'robot_data_50_10_09_02_26_22_53_50': (0.41, 0.93),
    'robot_data_50_10_09_02_26_22_54_33': (0.39, 0.85),
    'robot_data_50_10_09_02_26_22_55_12': (0.40, 0.95),
    
    # v = 50, alpha = 5 deg (Slight Left)
    'robot_data_50_5_09_02_26_22_55_57':  (0.30, 0.97),
    'robot_data_50_5_09_02_26_22_56_39':  (0.30, 0.96),
    'robot_data_50_5_09_02_26_22_57_25':  (0.29, 1.00),
    
    # --- NEGATIVE STEERING (Right Turns) ---
    
    # v = 50, alpha = -5 deg (Slight Right)
    'robot_data_50_-5_09_02_26_23_03_41':  (-0.42, 0.98),
    'robot_data_50_-5_09_02_26_23_03_52':  (-0.42, 0.95),
    'robot_data_50_-5_09_02_26_23_04_05':  (-0.41, 0.94), # Added from file list
    
    # v = 50, alpha = -10 deg (Soft Right)
    'robot_data_50_-10_09_02_26_23_10_01': (-0.23, 0.98),
    'robot_data_50_-10_09_02_26_23_10_36': (-0.26, 1.07),
    'robot_data_50_-10_09_02_26_23_11_20': (-0.21, 1.02), # Added from file list
    
    # v = 50, alpha = -15 deg (Right)
    'robot_data_50_-15_09_02_26_23_06_49': (-0.23, 1.03),
    'robot_data_50_-15_09_02_26_23_07_25': (-0.17, 1.04),
    'robot_data_50_-15_09_02_26_23_08_02': (-0.15, 1.02), # Added from file list
    
    # v = 50, alpha = -20 deg (Hard Right)
    'robot_data_50_-20_09_02_26_23_04_17': (-0.13, 1.10),
    'robot_data_50_-20_09_02_26_23_04_57': (-0.13, 1.03),
    'robot_data_50_-20_09_02_26_23_05_38': (-0.115, 1.05), # Added from file list
}

def run_system_id(data_directory='./data'):
    # Seting up lists to store calibration data
    e_calib, s_calib = [], []
    steer_cmds, radii, omegas = [], [], []
    base_path = Path(data_directory).resolve()

    # 1. Section 4: Encoder to Distance (Ke remains the same)
    for filename, (xm, ym) in straight_measurements.items():
        matching_files = list(base_path.glob(f"{filename}.pkl"))
        if not matching_files: continue
        with open(matching_files[0], 'rb') as f:
            data = pickle.load(f)
        encs = [row.encoder_counts for row in data['robot_sensor_signal']]
        delta_e = abs(encs[-1] - encs[0])
        s_actual = math.sqrt(xm**2 + ym**2)
        e_calib.append(delta_e)
        s_calib.append(s_actual)

    Ke = np.linalg.lstsq(np.array(e_calib)[:, np.newaxis], np.array(s_calib), rcond=None)[0][0]

    # 2. Section 5: Steering to Turning Radius with 2nd Order Fit
    for filename, (xm, ym) in steering_measurements.items():
        matching_files = list(base_path.glob(f"{filename}.pkl"))
        if not matching_files: continue
        with open(matching_files[0], 'rb') as f:
            data = pickle.load(f)
        
        steer_cmd = data['control_signal'][0][1] 
        theta = 0
        if abs(xm) > 1e-4:
            theta = math.atan2(xm, ym)
            # theta = thetat if steer_cmd > 0 else -thetat
            omega = 2*theta / 5  # Assuming time to reach final position is 5s for estimation
            R = (xm**2 + ym**2) / (2 * xm)
            steer_cmds.append(steer_cmd)
            radii.append(R)
            omegas.append(omega)

    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # Plot 1: Encoder to Distance
    ax1.scatter(e_calib, s_calib, color='blue', alpha=0.7)
    e_plot = np.linspace(0, max(e_calib)*1.1, 100)
    ax1.plot(e_plot, Ke * e_plot, 'r--', label=f'Ke={Ke:.8f}')
    ax1.set_title('Section 4: Encoder Count to Distance')
    ax1.set_xlabel('Encoder Counts'); ax1.set_ylabel('Distance (m)'); ax1.grid(True)

    # Plot 2: Steering to Turning Radius with 2nd Order Fit
    s_cmds_arr = np.array(steer_cmds)
    radii_arr = np.array(radii)
    
    # Second-order polynomial fit: R = p[0]*alpha^2 + p[1]*alpha + p[2]
    p_coeffs = np.polyfit(s_cmds_arr, radii_arr, 2)

    
    # Generate points for the smooth curve
    alpha_range = np.linspace(min(s_cmds_arr), max(s_cmds_arr), 100)
    r_fit = np.polyval(p_coeffs, alpha_range)

    ax2.scatter(s_cmds_arr, radii_arr, color='green', label='Measured Data')
    ax2.plot(alpha_range, r_fit, 'r-', label=f'2nd Order Fit')
    
    ax2.set_title('Section 5: Steering Command to Turning Radius (R)')
    ax2.set_xlabel('Steering Command (alpha)')
    ax2.set_ylabel('Turning Radius R (m)')
    ax2.set_ylim([-5, 5]) 
    ax2.legend()
    ax2.grid(True)

    # Plot 3. Steering to Angular Velocity (Omega) with 2nd Order Fit
    omegas_arr = np.array(omegas)
    p_omega_coeffs = np.polyfit(s_cmds_arr, omegas_arr, 2)
    omega_fit = np.polyval(p_omega_coeffs, alpha_range)
    ax3.scatter(s_cmds_arr, omegas_arr, color='purple', label='Measured Data')
    ax3.plot(alpha_range, omega_fit, 'r-', label=f'2nd Order Fit')
    ax3.set_title('Steering Command to Angular Velocity (Omega)')
    ax3.set_xlabel('Steering Command (alpha)')
    ax3.set_ylabel('Angular Velocity Omega (rad/s)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Ke: {Ke:.10f}")
    print(f"Radius Fit Coeffs (2nd order): {p_coeffs}")

if __name__ == "__main__":
    # Pointing to './data' because you are in 'robot_python_code' and 'data' is a folder inside it
    run_system_id(data_directory='./data')