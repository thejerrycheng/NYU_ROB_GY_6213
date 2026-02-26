# External Libraries
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import math
import numpy as np

# Internal Libraries
import parameters
import robot_python_code
import motion_models

# Open a file and return data in a form ready to plot
def get_file_data(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    camera_sensor_signal_list = data_dict['camera_sensor_signal']
    
    encoder_count_list = []
    velocity_list = []
    steering_angle_list = []
    x_camera_list = []
    y_camera_list = []
    z_camera_list = []
    yaw_camera_list = []
    
    for row in robot_sensor_signal_list:
        encoder_count_list.append(row.encoder_counts)
    for row in control_signal_list:
        velocity_list.append(row[0])
        steering_angle_list.append(row[1])
        
    T_cam_to_world = None
    
    for row in camera_sensor_signal_list:
        if row is not None and len(row) >= 6 and not np.allclose(row, 0):
            tvec = np.array(row[0:3], dtype=float)
            rvec = np.array(row[3:6], dtype=float)
            R_curr, _ = cv2.Rodrigues(rvec)
            
            T_cam_to_tag = np.eye(4)
            T_cam_to_tag[:3, :3] = R_curr
            T_cam_to_tag[:3, 3] = tvec.flatten()
            
            if T_cam_to_world is None:
                T_cam_to_world = np.linalg.inv(T_cam_to_tag)
                
            T_world_to_tag = T_cam_to_world @ T_cam_to_tag
            
            z_x = T_world_to_tag[0, 3]
            z_y = T_world_to_tag[1, 3]
            z_z = T_world_to_tag[2, 3]
            
            R_world_tag = T_world_to_tag[:3, :3]
            z_yaw = math.atan2(R_world_tag[1, 1], R_world_tag[0, 1])
            
            x_camera_list.append(z_x)
            y_camera_list.append(z_y)
            z_camera_list.append(z_z)
            yaw_camera_list.append(z_yaw)
        else:
            x_camera_list.append(0.0)
            y_camera_list.append(0.0)
            z_camera_list.append(0.0)
            yaw_camera_list.append(0.0)

    t0 = time_list[0]
    for i in range(len(time_list)):
        time_list[i] = time_list[i] - t0
    
    return time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list


# Open a file and return data in a form ready to plot
def get_file_data_for_kf(filename):
    data_loader = robot_python_code.DataLoader(filename)
    data_dict = data_loader.load()

    time_list = data_dict['time']
    control_signal_list = data_dict['control_signal']
    robot_sensor_signal_list = data_dict['robot_sensor_signal']
    camera_sensor_signal_list = data_dict['camera_sensor_signal']
    
    t0 = time_list[0]
    ekf_data = []
    
    T_cam_to_world = None
    cam_pos_world = None

    for i in range(len(time_list)):
        row_cam = camera_sensor_signal_list[i]
        z_t = None
        
        if row_cam is not None and len(row_cam) >= 6 and not np.allclose(row_cam, 0):
            tvec = np.array(row_cam[0:3], dtype=float)
            rvec = np.array(row_cam[3:6], dtype=float)
            R_curr, _ = cv2.Rodrigues(rvec)
            
            T_cam_to_tag = np.eye(4)
            T_cam_to_tag[:3, :3] = R_curr
            T_cam_to_tag[:3, 3] = tvec.flatten()
            
            if T_cam_to_world is None:
                T_cam_to_world = np.linalg.inv(T_cam_to_tag)
                cam_origin_world = T_cam_to_world @ np.array([0, 0, 0, 1])
                cam_pos_world = (cam_origin_world[0], cam_origin_world[1])
                
            T_world_to_tag = T_cam_to_world @ T_cam_to_tag
            
            z_x = T_world_to_tag[0, 3]
            z_y = T_world_to_tag[1, 3]
            
            R_world_tag = T_world_to_tag[:3, :3]
            z_yaw = math.atan2(R_world_tag[1, 1], R_world_tag[0, 1])
            z_t = [z_x, z_y, z_yaw]

        row = [time_list[i] - t0, control_signal_list[i], robot_sensor_signal_list[i], z_t]
        ekf_data.append(row)

    return ekf_data, cam_pos_world


def plot_trial_basics(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list = get_file_data(filename)
    
    plt.plot(time_list, encoder_count_list)
    plt.title('Encoder Values')
    plt.show()
    plt.plot(time_list, velocity_list)
    plt.title('Speed')
    plt.show()
    plt.plot(time_list, steering_angle_list)
    plt.title('Steering')
    plt.show()

def run_my_model_on_trial(filename, show_plot = True, plot_color = 'ko'):
    time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list = get_file_data(filename)
    
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)

    plt.plot(x_list, y_list,plot_color)
    plt.title('Motion Model Predicted XY Traj (m)')
    plt.axis([-0.5, 1.5, -1, 1])
    if show_plot:
        plt.show()

def plot_many_trial_predictions(directory):
    directory_path = Path(directory)
    plot_color_list = ['r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.','r.','k.','g.','c.', 'b.', 'r.','k.','g.','c.', 'b.']
    count = 0
    for item in directory_path.iterdir():
        filename = item.name
        plot_color = plot_color_list[count]
        run_my_model_on_trial(directory + filename, False, plot_color)
        count += 1
    plt.show()

def run_my_model_to_predict_distance(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, _, _ = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    distance = x_list[-30]
    return distance

def run_my_model_to_predict_state(filename):
    time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list = get_file_data(filename)
    motion_model = motion_models.MyMotionModel([0,0,0], 0)
    x_list, y_list, theta_list, distance_list = motion_model.traj_propagation(time_list, encoder_count_list, steering_angle_list)
    
    index_of_end = -30
    x = x_list[index_of_end]
    y = y_list[index_of_end]
    theta = theta_list[index_of_end]
    distance = distance_list[index_of_end]
    time_stamp = time_list[index_of_end] - time_list[0]
    return time_stamp, x, y, theta, distance

def get_diff_squared(m_list,p_list):
    diff_squared_list = []
    for i in range(len(m_list)):
        diff_squared = math.pow(m_list[i]-p_list[i],2)
        diff_squared_list.append(diff_squared)

    coefficients = np.polyfit(m_list, diff_squared_list, 2)
    p=np.poly1d(coefficients)

    plt.plot(m_list, diff_squared_list,'ko')
    plt.plot(m_list, p(m_list),'ro')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual - Predicted)^2 (m^2)')
    plt.show()
    return diff_squared_list

def get_diff_w_squared(dist_list, m_w_list, p_w_list):
    diff_squared_list = []
    for i in range(len(m_w_list)):
        diff_squared = math.pow(m_w_list[i]-p_w_list[i],2)
        diff_squared_list.append(diff_squared)

    plt.plot(dist_list, diff_squared_list,'ko')
    plt.plot([0], [0],'ko')
    plt.title("Error Squared (m^2)")
    plt.xlabel('Measured distance travelled (m)')
    plt.ylabel('(Actual w - Predicted w)^2 (m^2)')
    plt.show()
    return diff_squared_list

def process_files_and_plot(files_and_data, directory):
    predicted_distance_list = []
    measured_distance_list = []
    for row in files_and_data:
        filename = row[0]
        measured_distance = row[1]
        measured_distance_list.append(measured_distance)
        predicted_distance = run_my_model_to_predict_distance(directory + filename)
        predicted_distance_list.append(predicted_distance)

    plt.plot(measured_distance_list+[0], predicted_distance_list+[0], 'ko')
    plt.plot([0,1.7],[0,1.7])
    plt.title('Distance Trials')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()
    get_diff_squared(measured_distance_list, predicted_distance_list)

def process_files_and_plot_curve(files_and_data, directory):
    predicted_distance_list = []
    x_measured_list = []
    y_measured_list = []
    theta_measured_list = []
    x_predicted_list = []
    y_predicted_list = []
    theta_predicted_list = []
    w_measured_list = []
    w_predicted_list = []
    distance_predicted_list = []
    for row in files_and_data:
        filename = row[0]
        x_measured_distance = row[1]
        y_measured_distance = row[2]
        x_measured_list.append(x_measured_distance)
        y_measured_list.append(y_measured_distance)
        theta_measured = 2*math.atan2(y_measured_distance, x_measured_distance)
        theta_measured_list.append(theta_measured)

        time_stamp, x_predicted, y_predicted , theta_predicted, distance_predicted = run_my_model_to_predict_state(directory + filename)
        x_predicted_list.append(x_predicted)
        y_predicted_list.append(y_predicted)
        theta_predicted_list.append(theta_predicted)

        w_measured_list.append(theta_measured / time_stamp)
        w_predicted_list.append(theta_predicted / time_stamp)
        print("W:",w_measured_list[-1], w_predicted_list[-1])
        distance_predicted_list.append(distance_predicted)

    plt.plot(theta_measured_list+[0], theta_predicted_list+[0], 'ko')
    plt.plot([0, 2.1],[0, 2.1],'r')
    plt.title('Rotation Trials')
    plt.xlabel('Measured Theta (rad)')
    plt.ylabel('Predicted Theta (rad)')
    plt.legend(['Measured vs Predicted', 'Slope 1 Line'])
    plt.show()
    get_diff_w_squared(distance_predicted_list, w_measured_list, w_predicted_list)

def sample_model(num_samples):
    traj_duration = 10
    for i in range(num_samples):
        model = motion_models.MyMotionModel([0,0,0], 0)
        traj_x, traj_y, traj_theta = model.generate_simulated_traj(traj_duration)
        plt.plot(traj_x, traj_y, 'k.')

    plt.title('Sampling the model')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


class KalmanFilterPlot:
    def __init__(self, cam_pos=None):
        self.dir_length = 0.15
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.ekf_hist = []  
        self.dr_x, self.dr_y = [], []
        self.cam_pos = cam_pos 

    def update(self, ekf_mean, ekf_cov, dr_mean, z_t, is_occluded):
        self.ax.cla()

        self.ekf_hist.append((ekf_mean[0], ekf_mean[1], is_occluded))
        self.dr_x.append(dr_mean[0])
        self.dr_y.append(dr_mean[1])

        if self.cam_pos is not None:
            self.ax.plot(self.cam_pos[0], self.cam_pos[1], 'bs', markersize=12, label='Fixed Camera')
            self.ax.text(self.cam_pos[0] + 0.05, self.cam_pos[1] + 0.05, 'Camera', color='blue', fontsize=10, fontweight='bold')

        self.ax.plot(self.dr_x, self.dr_y, 'k--', alpha=0.6, label='Dead Reckoning')

        for i in range(1, len(self.ekf_hist)):
            x0, y0, occ0 = self.ekf_hist[i-1]
            x1, y1, occ1 = self.ekf_hist[i]
            color = 'red' if occ1 else 'green'
            self.ax.plot([x0, x1], [y0, y1], color=color, linewidth=2)

        self.ax.plot([], [], 'g-', linewidth=2, label='EKF (Corrected)')
        self.ax.plot([], [], 'r-', linewidth=2, label='EKF (Occluded)')

        if not is_occluded:
            self.ax.plot(z_t[0], z_t[1], 'gx', markersize=8, markeredgewidth=2, label='Measurement (Z_t)')

        eigenvalues, eigenvectors = np.linalg.eigh(ekf_cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * 3.0 * np.sqrt(abs(eigenvalues[0]))
        height = 2 * 3.0 * np.sqrt(abs(eigenvalues[1]))
        
        ell_color = 'red' if is_occluded else 'green'
        ell = Ellipse(xy=(ekf_mean[0], ekf_mean[1]), width=width, height=height, angle=angle, 
                      alpha=0.3, facecolor=ell_color)
        self.ax.add_patch(ell)

        self.ax.plot([dr_mean[0], dr_mean[0] + self.dir_length * math.cos(dr_mean[2])], 
                     [dr_mean[1], dr_mean[1] + self.dir_length * math.sin(dr_mean[2])], 'k')
        self.ax.plot([ekf_mean[0], ekf_mean[0] + self.dir_length * math.cos(ekf_mean[2])], 
                     [ekf_mean[1], ekf_mean[1] + self.dir_length * math.sin(ekf_mean[2])], color=ell_color)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Offline EKF Tracking vs Dead Reckoning (Occlusion Test)')
        
        self.ax.set_xlim(ekf_mean[0] - 1.5, ekf_mean[0] + 1.5)
        self.ax.set_ylim(ekf_mean[1] - 1.5, ekf_mean[1] + 1.5)
        
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        plt.draw()
        plt.pause(0.01)

def offline_efk():
    from online_ekf import ExtendedKalmanFilter 

    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    print(f"Loading data from {filename}...")
    
    ekf_data, cam_pos_world = get_file_data_for_kf(filename)

    x_0 = [0.0, 0.0, math.pi / 2]
    Sigma_0 = np.eye(3) * 0.1
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)
    dr_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    kalman_filter_plot = KalmanFilterPlot(cam_pos=cam_pos_world)

    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] 
        
        if delta_t <= 0:
            continue
            
        u_t = np.array([row[2].encoder_counts, row[2].steering]) 
        z_t = row[3] 
        
        is_occluded = (z_t is None)

        ekf.update(u_t, z_t, delta_t)
        dr_filter.update(u_t, None, delta_t) 

        kalman_filter_plot.update(ekf.state_mean, ekf.state_covariance[0:2, 0:2],
                                  dr_filter.state_mean, z_t, is_occluded)
        
    plt.ioff()
    plt.show()

######### MAIN ########

files_and_data = [
    ['robot_data_60_0_28_01_26_13_41_44.pkl', 67/100], 
    ['robot_data_60_0_28_01_26_13_43_41.pkl', 68/100],
    ['robot_data_60_0_28_01_26_13_37_15.pkl', 113/100],
    ['robot_data_60_0_28_01_26_13_35_18.pkl', 107/100],
    ['robot_data_60_0_28_01_26_13_41_10.pkl', 65/100],
    ['robot_data_60_0_28_01_26_13_42_55.pkl', 70/100],
    ['robot_data_60_0_28_01_26_13_39_36.pkl', 138/100],
    ['robot_data_60_0_28_01_26_13_42_19.pkl', 69/100],
    ['robot_data_60_0_28_01_26_13_36_10.pkl', 109/100],
    ['robot_data_60_0_28_01_26_13_33_20.pkl', 100/100],
    ['robot_data_60_0_28_01_26_13_34_28.pkl', 103/100],
]

files_and_data_curve = [
    ['robot_data_60_10_28_01_26_13_44_28.pkl', 61/100, 31/100],
    ['robot_data_60_10_28_01_26_13_45_14.pkl', 61/100, 32/100],
    ['robot_data_60_10_28_01_26_13_45_56.pkl', 61/100, 30/100],
    ['robot_data_60_10_28_01_26_13_46_26.pkl', 61/100, 31/100], 
    ['robot_data_60_10_28_01_26_13_47_10.pkl', 62/100, 29/100],
    ['robot_data_60_10_28_01_26_13_48_25.pkl', 70/100, 106/100],
    ['robot_data_60_10_28_01_26_13_49_08.pkl', 73/100, 106/100],
    ['robot_data_60_10_28_01_26_13_50_55.pkl', 73/100, 71/100],
    ['robot_data_60_10_28_01_26_13_51_34.pkl', 76/100, 69/100],
    ['robot_data_60_10_28_01_26_13_52_07.pkl', 78/100, 71/100],
    ['robot_data_60_10_28_01_26_13_52_35.pkl', 76/100, 70/100],
    ['robot_data_60_10_28_01_26_13_53_08.pkl', 76/100, 71/100],
]

# Plot the motion model predictions for a single trial
if False:
    filename = './data_straight/robot_data_60_0_28_01_26_13_36_10.pkl'
    run_my_model_on_trial(filename)

# Plot the motion model predictions for each trial in a folder
if False:
    directory = ('./data_straight/')
    plot_many_trial_predictions(directory)

# A list of files to open, process, and plot - for comparing predicted with actual distances
if False:
    directory = ('./data_straight/')    
    process_files_and_plot(files_and_data, directory)

if False:
    directory = ('./data_curve/')    
    process_files_and_plot_curve(files_and_data_curve, directory)

# Try to sample with the motion model
if False:
    sample_model(200)

# Try to load some camera data from a single trial
if False:
    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    time_list, encoder_count_list, velocity_list, steering_angle_list, x_camera_list, y_camera_list, z_camera_list, yaw_camera_list = get_file_data(filename)

    wheel_radius = 0.034 #cm
    encoder_counts_per_revolution = 152
    encoder_counts_to_distance = -2 * math.pi * wheel_radius/ encoder_counts_per_revolution

    plt.plot(time_list, ((np.array(encoder_count_list))-encoder_count_list[0]) * encoder_counts_to_distance + y_camera_list[0], 'k') 
    plt.plot(time_list, x_camera_list, 'g') 
    plt.plot(time_list, y_camera_list, 'b') 
    plt.legend(['Encoder Distance','Camera X','Camera Y'])
    plt.show()   

# Run the Offline EKF
if False:
    offline_efk()