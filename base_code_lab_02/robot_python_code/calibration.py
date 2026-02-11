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
    'robot_data_50_-5_09_02_26_23_03_41':  (-0.13, 1.05),
    'robot_data_50_-5_09_02_26_23_03_52':  (-0.13, 1.03),
    'robot_data_50_-5_09_02_26_23_04_05':  (-0.115, 1.10), # Added from file list
    
    # v = 50, alpha = -10 deg (Soft Right)
    'robot_data_50_-10_09_02_26_23_10_01': (-0.23, 1.03),
    'robot_data_50_-10_09_02_26_23_10_36': (-0.17, 1.04),
    'robot_data_50_-10_09_02_26_23_11_20': (-0.15, 1.02), # Added from file list
    
    # v = 50, alpha = -15 deg (Right)
    'robot_data_50_-15_09_02_26_23_06_49': (-0.23, 0.98),
    'robot_data_50_-15_09_02_26_23_07_25': (-0.26, 1.07),
    'robot_data_50_-15_09_02_26_23_08_02': (-0.21, 1.02), # Added from file list
    
    # v = 50, alpha = -20 deg (Hard Right)
    'robot_data_50_-20_09_02_26_23_04_17': (-0.42, 0.92),
    'robot_data_50_-20_09_02_26_23_04_57': (-0.42, 0.95),
    'robot_data_50_-20_09_02_26_23_05_38': (-0.41, 0.84), # Added from file list
}


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import pickle

def run_system_id(data_directory='./data'):
    e_calib, s_calib = [], []
    v_cmds, v_actuals = [], []
    steer_cmds, delta_angles = [], []
    
    base_path = Path(data_directory).resolve()
    L = 0.145  # Wheelbase: 14.5cm
    T_STRAIGHT = 7.5  # Duration for straight trials

    # 1. PROCESS DATA
    pathlist = list(base_path.glob('*.pkl'))
    
    # Process Straight Trials
    for filename, (xm, ym) in straight_measurements.items():
        matching = [p for p in pathlist if filename in p.name]
        if matching:
            with open(matching[0], 'rb') as f:
                data = pickle.load(f)
            s_actual = math.sqrt(xm**2 + ym**2)
            v_cmd = data['control_signal'][0][0]
            v_cmds.append(v_cmd)
            v_actuals.append(s_actual / T_STRAIGHT)

    # Process Steering Trials
    for filename, (xm, ym) in steering_measurements.items():
        matching = [p for p in pathlist if filename in p.name]
        if matching:
            with open(matching[0], 'rb') as f:
                data = pickle.load(f)
            steer_cmd = data['control_signal'][0][1] 
            if abs(xm) > 1e-4:
                R = (xm**2 + ym**2) / (2 * xm)
                delta = math.atan(L / R)
                steer_cmds.append(steer_cmd)
                delta_angles.append(delta)

    # 2. DYNAMIC LEAST SQUARES FITTING
    v_cmds_arr, v_act_arr = np.array(v_cmds), np.array(v_actuals)
    s_arr, d_arr = np.array(steer_cmds), np.array(delta_angles)

    # Velocity Fit: v = m*v_cmd + c
    A_v = np.column_stack((v_cmds_arr, np.ones(len(v_cmds_arr))))
    m_v, c_v = np.linalg.inv(A_v.T @ A_v) @ A_v.T @ v_act_arr

    # Steering Fit (2nd Order): delta = a*alpha^2 + b*alpha + c
    X_d = np.column_stack((s_arr**2, s_arr, np.ones(len(s_arr))))
    a, b, c = np.linalg.inv(X_d.T @ X_d) @ X_d.T @ d_arr

    # 3. DYNAMIC VARIANCE CALCULATION (Residual Analysis)
    # Velocity Variance
    v_predictions = m_v * v_cmds_arr + c_v
    v_residuals = v_act_arr - v_predictions
    var_v = np.mean(v_residuals**2)

    # Steering Variance
    d_predictions = a * s_arr**2 + b * s_arr + c
    d_residuals = d_arr - d_predictions
    var_delta = np.mean(d_residuals**2)

    # --- PLOTTING ---
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Plot 1: Velocity
    ax1.scatter(v_cmds_arr, v_act_arr, color='black', s=15)
    ax1.plot(v_cmds_arr, v_predictions, 'r-', label=f'var={var_v:.2e}')
    ax1.set_title('Velocity Fit'); ax1.legend(); ax1.grid(True, linestyle=':')

    # Plot 2: Steering
    alpha_range = np.linspace(min(s_arr), max(s_arr), 100)
    ax2.scatter(s_arr, d_arr, color='black', s=15)
    ax2.plot(alpha_range, a*alpha_range**2 + b*alpha_range + c, 'r-', label=f'var={var_delta:.2e}')
    ax2.set_title('Steering Fit'); ax2.legend(); ax2.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()

    print(f"Results:\nv = {m_v:.6f}*u + {c_v:.6f} | var_v = {var_v:.8f}")
    print(f"delta = {a:.6f}*a^2 + {b:.6f}*a + {c:.6f} | var_d = {var_delta:.8f}")
    
    return m_v, c_v, var_v, a, b, c, var_delta

# def run_system_id(data_directory='./data'):
#     e_calib, s_calib = [], []
#     v_cmds, v_actuals = [], []
#     steer_cmds, delta_angles = [], []
    
#     base_path = Path(data_directory).resolve()
#     L = 0.145  # Revised wheelbase: 14.5cm
#     T_STRAIGHT = 7.5  #

#     # 1. PROCESS STRAIGHT TRIALS
#     for filename, (xm, ym) in straight_measurements.items():
#         matching_files = list(base_path.glob(f"{filename}.pkl"))
#         if not matching_files: continue
#         with open(matching_files[0], 'rb') as f:
#             data = pickle.load(f)
        
#         s_actual = math.sqrt(xm**2 + ym**2) #
#         v_cmd = data['control_signal'][0][0]
#         v_actual = s_actual / T_STRAIGHT #
        
#         v_cmds.append(v_cmd)
#         v_actuals.append(v_actual)

#     # 2. PROCESS STEERING TRIALS
#     for filename, (xm, ym) in steering_measurements.items():
#         matching_files = list(base_path.glob(f"{filename}.pkl"))
#         if not matching_files: continue
#         with open(matching_files[0], 'rb') as f:
#             data = pickle.load(f)
        
#         steer_cmd = data['control_signal'][0][1] 
#         if abs(xm) > 1e-4:
#             R = (xm**2 + ym**2) / (2 * xm) #
#             delta = math.atan(L / R) #
#             steer_cmds.append(steer_cmd)
#             delta_angles.append(delta)

#     # --- LEAST SQUARES & VARIANCE ---
    
#     # Velocity: v = m*u + c
#     v_cmds_arr, v_act_arr = np.array(v_cmds), np.array(v_actuals)
#     A_v = np.column_stack((v_cmds_arr, np.ones(len(v_cmds_arr))))
#     m_v, c_v = np.linalg.inv(A_v.T @ A_v) @ A_v.T @ v_act_arr
#     # Variance of Velocity Residuals
#     v_preds = m_v * v_cmds_arr + c_v
#     var_v = np.var(v_act_arr - v_preds)

#     # Steering: delta = a*alpha^2 + b*alpha + c
#     s_arr, d_arr = np.array(steer_cmds), np.array(delta_angles)
#     X_d = np.column_stack((s_arr**2, s_arr, np.ones(len(s_arr))))
#     a, b, c = np.linalg.inv(X_d.T @ X_d) @ X_d.T @ d_arr
#     # Variance of Steering Residuals
#     d_preds = a * s_arr**2 + b * s_arr + c
#     var_d = np.var(d_arr - d_preds)

    


#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.titlesize': 12,
#         'axes.labelsize': 12,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 10,
#         'font.family': 'serif'  # Matches common LaTeX fonts
#     })


#     # 2. Setup Figure: 7 inches total width for two 3.5 inch plots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

#     # --- Plot 1: Velocity Command to Actual Velocity ---
#     v_cmds_arr = np.array(v_cmds)
#     v_act_arr = np.array(v_actuals)
    
#     # Manual Least Squares: v = m*v_cmd + c
#     A_v = np.column_stack((v_cmds_arr, np.ones(len(v_cmds_arr))))
#     m_v, c_v = np.linalg.inv(A_v.T @ A_v) @ A_v.T @ v_act_arr
    
#     ax1.scatter(v_cmds_arr, v_act_arr, color='black', s=15, label='Data')
#     ax1.plot(v_cmds_arr, m_v * v_cmds_arr + c_v, 'r-', linewidth=1.5, 
#              label=f'$v = {m_v:.4f}u + {c_v:.2f}$')
#     ax1.set_title('Velocity Calibration')
#     ax1.set_xlabel('Cmd Velocity ($v_{cmd}$)')
#     ax1.set_ylabel('Actual Velocity (m/s)')
#     ax1.grid(True, linestyle=':', alpha=0.7)
#     ax1.legend()

#     # --- Plot 2: Steering Command (alpha) to Steer Angle (delta) ---
#     s_arr = np.array(steer_cmds)
#     d_arr = np.array(delta_angles)
    
#     # Manual Least Squares (2nd Order): delta = a*alpha^2 + b*alpha + c
#     X_d = np.column_stack((s_arr**2, s_arr, np.ones(len(s_arr))))
#     a, b, c = np.linalg.inv(X_d.T @ X_d) @ X_d.T @ d_arr
    
#     alpha_range = np.linspace(min(s_arr), max(s_arr), 100)
#     delta_fit = a * alpha_range**2 + b * alpha_range + c
    
#     ax2.scatter(s_arr, d_arr, color='black', s=15, label='Data')
#     ax2.plot(alpha_range, delta_fit, 'r-', linewidth=1.5, label='2nd Order Fit')
#     ax2.set_title('Steering Calibration')
#     ax2.set_xlabel('Cmd Steering ($\\alpha$)')
#     ax2.set_ylabel('Steer Angle $\delta$ (rad)')
#     ax2.grid(True, linestyle=':', alpha=0.7)
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig('system_calibration_results.pdf', bbox_inches='tight')
#     plt.show()

#     print(f"v = {m_v:.6f}*v_cmd + {c_v:.6f}")
#     print(f"delta = {a:.6f}*alpha^2 + {b:.6f}*alpha + {c:.6f}")

if __name__ == "__main__":
    # Pointing to './data' because you are in 'robot_python_code' and 'data' is a folder inside it
    run_system_id(data_directory='./data')