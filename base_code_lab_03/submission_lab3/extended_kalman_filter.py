# External libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Local libraries
import parameters
import data_handling as data_handling_old

# --- Calibrated Constants from Motion Model ---
L = 0.145
KE_VALUE = 0.0001345210
V_M = 0.004808
V_C = -0.045557
VAR_V = 0.00057829
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134

class ExtendedKalmanFilter:
    def __init__(self, x_0, Sigma_0, encoder_counts_0):
        self.state_mean = np.array(x_0, dtype=float)
        self.state_covariance = np.array(Sigma_0, dtype=float)
        self.predicted_state_mean = np.zeros(3)
        self.predicted_state_covariance = np.eye(3)
        self.last_encoder_counts = encoder_counts_0

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update(self, u_t, z_t, delta_t):
        self.prediction_step(u_t, delta_t)
        
        # Correct only if measurement is valid and non-zero
        if z_t is not None and not np.isnan(z_t).any() and not np.allclose(z_t, [0.0, 0.0, 0.0]):
            self.correction_step(z_t)
        else:
            # If no measurement, the prediction becomes the new state
            self.state_mean = self.predicted_state_mean
            self.state_covariance = self.predicted_state_covariance

    def prediction_step(self, u_t, delta_t):
        s = self.distance_travelled_s(u_t[0])
        x_t, _ = self.g_function(self.state_mean, u_t, delta_t)
        
        G_x = self.get_G_x(self.state_mean, s)
        R_t = self.get_R(s)
        
        self.predicted_state_mean = x_t
        self.predicted_state_covariance = G_x @ self.state_covariance @ G_x.T + R_t
        self.last_encoder_counts = u_t[0]

    def correction_step(self, z_t):
        H = self.get_H()
        Q = self.get_Q() 
        
        z_pred = self.get_h_function(self.predicted_state_mean)
        y_t = z_t - z_pred
        y_t[2] = self.normalize_angle(y_t[2])
        
        S_t = H @ self.predicted_state_covariance @ H.T + Q
        K_t = self.predicted_state_covariance @ H.T @ np.linalg.inv(S_t)
        
        self.state_mean = self.predicted_state_mean + K_t @ y_t
        self.state_mean[2] = self.normalize_angle(self.state_mean[2])
        
        I = np.eye(3)
        self.state_covariance = (I - K_t @ H) @ self.predicted_state_covariance

    def distance_travelled_s(self, encoder_counts):
        de = encoder_counts - self.last_encoder_counts
        return de * KE_VALUE    
            
    def rotational_velocity_w(self, steering_angle_command):        
        alpha = steering_angle_command
        return DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]

    def g_function(self, x_tm1, u_t, delta_t):
        s = self.distance_travelled_s(u_t[0])
        delta = self.rotational_velocity_w(u_t[1])
        
        x_t = np.zeros(3)
        x_t[0] = x_tm1[0] + s * math.cos(x_tm1[2])
        x_t[1] = x_tm1[1] + s * math.sin(x_tm1[2])
        x_t[2] = self.normalize_angle(x_tm1[2] - (s * math.tan(delta)) / L)
        
        return x_t, s
    
    def get_h_function(self, x_t):
        return x_t
    
    def get_G_x(self, x_tm1, s):       
        theta = x_tm1[2]
        return np.array([
            [1.0, 0.0, -s * math.sin(theta)],
            [0.0, 1.0,  s * math.cos(theta)],
            [0.0, 0.0,  1.0]
        ])

    def get_H(self):
        return np.eye(3)
    
    def get_R(self, s):
        return np.diag([VAR_V * abs(s), VAR_V * abs(s), VAR_DELTA * abs(s)])

    def get_Q(self):
        """
        UPDATED: Empirical values from your External Camera Calibration.
        """
        var_x = 0.0036446932     
        var_y = 0.0169607627     
        var_theta = 0.0002150677  
        return np.diag([var_x, var_y, var_theta])

# class KalmanFilterPlot:

#     def __init__(self):
#         self.dir_length = 0.1
#         fig, ax = plt.subplots()
#         self.ax = ax
#         self.fig = fig

#     def update(self, state_mean, state_covaraiance):
#         plt.clf()

#         # Plot covariance ellipse
#         lambda_, v = np.linalg.eig(state_covaraiance)
#         lambda_ = np.sqrt(lambda_)
#         xy = (state_mean[0], state_mean[1])
#         angle=np.rad2deg(np.arctan2(*v[:,0][::-1]))
#         ell = Ellipse(xy, alpha=0.5, facecolor='red',width=lambda_[0], height=lambda_[1], angle = angle)
#         ax = self.fig.gca()
#         ax.add_artist(ell)
        
#         # Plot state estimate
#         plt.plot(state_mean[0], state_mean[1],'ro')
#         plt.plot([state_mean[0], state_mean[0]+ self.dir_length*math.cos(state_mean[2]) ], [state_mean[1], state_mean[1]+ self.dir_length*math.sin(state_mean[2]) ],'r')
#         plt.xlabel('X(m)')
#         plt.ylabel('Y(m)')
#         plt.axis([-0.25, 2, -1, 1])
#         plt.grid()
#         plt.draw()
#         plt.pause(0.1)

class KalmanFilterPlot:
    def __init__(self):
        self.dir_length = 0.1
        plt.ion() # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Store histories for drawing the trajectory trails
        self.ekf_x_hist, self.ekf_y_hist = [], []
        self.dr_x_hist, self.dr_y_hist = [], []

    def update(self, ekf_mean, ekf_cov, dr_mean, dr_cov, z_t):
        self.ax.cla()

        # Update histories
        self.ekf_x_hist.append(ekf_mean[0])
        self.ekf_y_hist.append(ekf_mean[1])
        self.dr_x_hist.append(dr_mean[0])
        self.dr_y_hist.append(dr_mean[1])

        # Plot trajectory trails
        self.ax.plot(self.dr_x_hist, self.dr_y_hist, 'k--', alpha=0.6, label='Dead Reckoning (Motion Only)')
        self.ax.plot(self.ekf_x_hist, self.ekf_y_hist, 'b-', linewidth=2, label='EKF (Motion + Camera)')
        if z_t is not None and not np.isnan(z_t).any() and not np.allclose(z_t, [0.0, 0.0, 0.0]):
            self.ax.plot(z_t[0], z_t[1], 'gx', markersize=8, label='Raw Camera Z_t')

        eigenvalues, eigenvectors = np.linalg.eigh(ekf_cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * 3.0 * np.sqrt(abs(eigenvalues[0]))
        height = 2 * 3.0 * np.sqrt(abs(eigenvalues[1]))
        
        ell = Ellipse(xy=(ekf_mean[0], ekf_mean[1]), width=width, height=height, angle=angle, 
                      alpha=0.3, facecolor='blue', label='EKF 3$\sigma$ Confidence')
        self.ax.add_patch(ell)
        
        # Plot current states & headings (Dead Reckoning = Black, EKF = Blue)
        self.ax.plot(dr_mean[0], dr_mean[1], 'ko')
        self.ax.plot([dr_mean[0], dr_mean[0] + self.dir_length * math.cos(dr_mean[2])], 
                     [dr_mean[1], dr_mean[1] + self.dir_length * math.sin(dr_mean[2])], 'k')
                     
        self.ax.plot(ekf_mean[0], ekf_mean[1], 'bo')
        self.ax.plot([ekf_mean[0], ekf_mean[0] + self.dir_length * math.cos(ekf_mean[2])], 
                     [ekf_mean[1], ekf_mean[1] + self.dir_length * math.sin(ekf_mean[2])], 'b')

        # Formatting
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Live EKF Tracking vs Dead Reckoning Drift')
        self.ax.set_xlim(ekf_mean[0] - 1.5, ekf_mean[0] + 1.5)
        self.ax.set_ylim(ekf_mean[1] - 1.5, ekf_mean[1] + 1.5)
        
        self.ax.legend(loc='upper left')
        self.ax.grid(True)
        self.ax.axis('equal') 
        
        plt.draw()
        plt.pause(0.01)


def offline_efk():
    # Get data to filter
    filename = './data/robot_data_68_0_06_02_26_17_12_19.pkl'
    print(f"Loading data from {filename}...")
    ekf_data = data_handling_old.get_file_data_for_kf(filename)

    # Initial States
    x_0 = [ekf_data[0][3][0]+.5, ekf_data[0][3][1], ekf_data[0][3][5]]
    Sigma_0 = np.eye(3) * 1.0
    encoder_counts_0 = ekf_data[0][2].encoder_counts
    
    # Instantiate the full EKF
    ekf = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)
    
    # Instantiate a second filter purely for Dead Reckoning
    dead_reckoning_filter = ExtendedKalmanFilter(x_0, Sigma_0, encoder_counts_0)

    # Create plotting tool
    kalman_filter_plot = KalmanFilterPlot()

    # Loop over sim data
    for t in range(1, len(ekf_data)):
        row = ekf_data[t]
        delta_t = ekf_data[t][0] - ekf_data[t-1][0] # time step size
        
        if delta_t <= 0:
            continue
            
        u_t = np.array([row[2].encoder_counts, row[2].steering]) # robot_sensor_signal
        z_t = np.array([row[3][0], row[3][1], row[3][5]]) # camera_sensor_signal

        # Run the full EKF with camera measurements
        ekf.update(u_t, z_t, delta_t)
        
        # Run the Dead Reckoning filter explicitly passing 'None' for camera data
        dead_reckoning_filter.update(u_t, None, delta_t)

        # Update the visualizer with both states
        kalman_filter_plot.update(
            ekf.state_mean, ekf.state_covariance[0:2, 0:2],
            dead_reckoning_filter.state_mean, dead_reckoning_filter.state_covariance[0:2, 0:2],
            z_t
        )
        
    # Prevent the plot from immediately closing when the loop finishes
    plt.ioff()
    plt.show()

####### MAIN #######
if False:
    offline_efk()