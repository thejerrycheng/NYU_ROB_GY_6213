import matplotlib
matplotlib.use('Agg') # Crucial fix: Renders in background to prevent macOS GIL crash

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

# Local libraries
import parameters

# ==========================================
# 1. EKF & STANDARD MOTION MODEL
# ==========================================
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
        
        if z_t is not None and not np.isnan(z_t).any() and not np.allclose(z_t, [0.0, 0.0, 0.0]):
            self.correction_step(z_t)
        else:
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
        var_x = 0.0036446932     
        var_y = 0.0169607627     
        var_theta = 0.0002150677  
        return np.diag([var_x, var_y, var_theta])


# ==========================================
# 2. CAMERA SETTINGS
# ==========================================
CAM_MATRIX = np.array([
    [1407.29350, 0.00000000, 975.201510],
    [0.00000000, 1405.70507, 534.098781], 
    [0.00000000, 0.00000000, 1.00000000]
])

DIST_COEFFS = np.array([
    [0.05904389, -0.03912268, -0.000681, 0.00122093, -0.09274777]
])

TAG_SIZE_M = 0.0762 
TARGET_TAG_ID = 5 
DICT_TYPE = aruco.DICT_APRILTAG_25h9

def get_camera_measurement(frame, T_cam_to_world):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or TARGET_TAG_ID not in ids:
        return None, None, frame
        
    idx = np.where(ids == TARGET_TAG_ID)[0][0]
    tag_corners = corners[idx][0]
    
    cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
    
    half_s = TAG_SIZE_M / 2.0
    obj_points = np.array([
        [-half_s,  half_s, 0], 
        [ half_s,  half_s, 0], 
        [ half_s, -half_s, 0], 
        [-half_s, -half_s, 0]  
    ], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(obj_points, tag_corners, CAM_MATRIX, DIST_COEFFS)
    
    if not success:
        return None, None, frame
        
    cv2.drawFrameAxes(frame, CAM_MATRIX, DIST_COEFFS, rvec, tvec, TAG_SIZE_M)
        
    R_curr, _ = cv2.Rodrigues(rvec)
    
    T_cam_to_tag = np.eye(4)
    T_cam_to_tag[:3, :3] = R_curr
    T_cam_to_tag[:3, 3] = tvec.flatten()
    
    z_t = None
    if T_cam_to_world is not None:
        T_world_to_tag = T_cam_to_world @ T_cam_to_tag
        
        z_x = T_world_to_tag[0, 3]
        z_y = T_world_to_tag[1, 3]
        
        R_world_tag = T_world_to_tag[:3, :3]
        front_vec_x = R_world_tag[0, 1]
        front_vec_y = R_world_tag[1, 1]
        z_yaw = math.atan2(front_vec_y, front_vec_x)
        
        z_t = np.array([z_x, z_y, z_yaw])
        
    return T_cam_to_tag, z_t, frame


# ==========================================
# 3. LIVE PLOTTER (OPENCV COMPATIBLE)
# ==========================================
class LiveTrackerPlot:
    def __init__(self):
        self.dir_length = 0.15
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.ekf_x, self.ekf_y = [], []
        self.dr_x, self.dr_y = [], []

    def update(self, ekf_mean, ekf_cov, dr_mean, dr_cov, z_t, vel_cmd, steer_cmd):
        self.ax.cla()

        self.ekf_x.append(ekf_mean[0])
        self.ekf_y.append(ekf_mean[1])
        self.dr_x.append(dr_mean[0])
        self.dr_y.append(dr_mean[1])

        self.ax.plot(self.dr_x, self.dr_y, 'k--', alpha=0.6, label='Dead Reckoning')
        self.ax.plot(self.ekf_x, self.ekf_y, 'b-', linewidth=2, label='EKF (Motion + Camera)')

        if z_t is not None:
            self.ax.plot(z_t[0], z_t[1], 'gx', markersize=10, markeredgewidth=2, label='Camera (Z_t)')

        eigenvalues, eigenvectors = np.linalg.eigh(ekf_cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * 3.0 * np.sqrt(abs(eigenvalues[0]))
        height = 2 * 3.0 * np.sqrt(abs(eigenvalues[1]))
        ell = Ellipse(xy=(ekf_mean[0], ekf_mean[1]), width=width, height=height, angle=angle, 
                      alpha=0.3, facecolor='blue')
        self.ax.add_patch(ell)

        self.ax.plot([dr_mean[0], dr_mean[0] + self.dir_length * math.cos(dr_mean[2])], 
                     [dr_mean[1], dr_mean[1] + self.dir_length * math.sin(dr_mean[2])], 'k')
        self.ax.plot(dr_mean[0], dr_mean[1], 'ko')
        
        self.ax.plot([ekf_mean[0], ekf_mean[0] + self.dir_length * math.cos(ekf_mean[2])], 
                     [ekf_mean[1], ekf_mean[1] + self.dir_length * math.sin(ekf_mean[2])], 'b')
        self.ax.plot(ekf_mean[0], ekf_mean[1], 'bo')

        info_text = (f"Speed Cmd: {vel_cmd:.1f}\n"
                     f"Steer Cmd: {steer_cmd:.1f}\n"
                     f"Pose X: {ekf_mean[0]:+.3f}m\n"
                     f"Pose Y: {ekf_mean[1]:+.3f}m\n"
                     f"Pose Th: {math.degrees(ekf_mean[2]):+.1f}deg")
        
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        self.ax.set_title("Live 2D Projection: EKF vs Dead Reckoning")
        self.ax.set_xlim(ekf_mean[0] - 1.0, ekf_mean[0] + 1.0)
        self.ax.set_ylim(ekf_mean[1] - 1.0, ekf_mean[1] + 1.0)
        self.ax.grid(True)
        self.ax.legend(loc='lower right')
        
        # Draw canvas to a NumPy array for OpenCV
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img


# ==========================================
# 4. MAIN LOOP (INTERACTIVE SIMULATION)
# ==========================================
def run_online_system():
    cap = cv2.VideoCapture(parameters.camera_id)
    plotter = LiveTrackerPlot()
    
    T_cam_to_world = None
    ekf = None
    dr_filter = None
    
    print("\n[!] Running in Pure Simulation Mode (Real Camera + W/A/S/D Controls)")
    print("[-] UDP Connection to real robot is disabled.")
    
    sim_encoder_ticks = 0
    velocity_cmd = 0.0      
    steering_angle = 0.0    
    
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        h, w = frame.shape[:2]
        new_mat, _ = cv2.getOptimalNewCameraMatrix(CAM_MATRIX, DIST_COEFFS, (w,h), 1, (w,h))
        frame = cv2.undistort(frame, CAM_MATRIX, DIST_COEFFS, None, new_mat)
        
        T_cam_to_tag, z_t, debug_frame = get_camera_measurement(frame, T_cam_to_world)
        
        curr_time = time.time()
        delta_t = curr_time - last_time
        last_time = curr_time
        
        key = cv2.waitKey(1)
        if key == 27: break
        elif key == ord('w'): velocity_cmd = min(100, velocity_cmd + 10)
        elif key == ord('s'): velocity_cmd = max(-100, velocity_cmd - 10)
        elif key == ord('a'): steering_angle = max(-20, steering_angle - 2)
        elif key == ord('d'): steering_angle = min(20, steering_angle + 2)
        elif key == ord('x'): velocity_cmd = 0; steering_angle = 0
        
        sim_encoder_ticks += int(velocity_cmd * delta_t)
        current_encoder = sim_encoder_ticks
            
        if key == 32 and T_cam_to_tag is not None and ekf is None:
            T_cam_to_world = np.linalg.inv(T_cam_to_tag)
            x_0 = [0.0, 0.0, math.pi / 2]
            Sigma_0 = np.eye(3) * 0.1
            ekf = ExtendedKalmanFilter(x_0, Sigma_0, current_encoder)
            dr_filter = ExtendedKalmanFilter(x_0, Sigma_0, current_encoder)
            print("[+] Origin Locked. EKF Started at 90 degrees (+Y).")

        if ekf is not None:
            u_t = np.array([current_encoder, steering_angle])
            
            ekf.update(u_t, z_t, delta_t)
            dr_filter.update(u_t, None, delta_t)
            
            # Fetch the generated plot image
            plot_img = plotter.update(ekf.state_mean, ekf.state_covariance[0:2, 0:2],
                                      dr_filter.state_mean, dr_filter.state_covariance[0:2, 0:2],
                                      z_t, velocity_cmd, steering_angle)
            
            cv2.putText(debug_frame, f"CMD: Spd {velocity_cmd:.0f} | Steer {steering_angle:.0f}", (10, 30), 1, 1.2, (255, 255, 0), 2)
            
            # Show the EKF plot in a separate OpenCV window
            cv2.imshow("EKF 2D Projection", plot_img)
        else:
            cv2.putText(debug_frame, "AIM TAG & PRESS SPACE TO START", (10, 30), 1, 1.5, (0, 0, 255), 2)

        # Show the Camera Feed
        cv2.imshow("Camera View", debug_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_online_system()