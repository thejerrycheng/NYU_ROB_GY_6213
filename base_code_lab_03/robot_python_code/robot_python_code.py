# External libraries
import serial
import time
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
import socket
from time import strftime
import math

# Local libraries
import parameters

# Function to try to connect to the robot via udp over wifi
def create_udp_communication(arduinoIP, localIP, arduinoPort, localPort, bufferSize):
    try:
        udp = UDPCommunication(arduinoIP, localIP, arduinoPort, localPort, bufferSize)
        print("Success in creating udp communication")
        return udp, True
    except:
        print("Failed to create udp communication!")
        return _, False
        
        
# Class to hold the UPD over wifi connection setup
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((localIP, localPort))
        
    def receive_msg(self):
        bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        clientMsg = "{}".format(message.decode())
        clientIP = "{}".format(address)
        return clientMsg
       
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))


# Class to hold the data logger that records data when needed
class DataLogger:
    def __init__(self, filename_start, data_name_list):
        self.filename_start = filename_start
        self.filename = filename_start
        self.line_count = 0
        self.dictionary = {}
        self.data_name_list = data_name_list
        for name in data_name_list:
            self.dictionary[name] = []
        self.currently_logging = False

    def reset_logfile(self, control_signal):
        self.filename = self.filename_start + "_"+str(control_signal[0])+"_"+str(control_signal[1]) + strftime("_%d_%m_%y_%H_%M_%S.pkl")
        self.dictionary = {}
        for name in self.data_name_list:
            self.dictionary[name] = []

    def log(self, logging_switch_on, time, control_signal, robot_sensor_signal, camera_sensor_signal, state_mean, state_covariance):
        if not logging_switch_on:
            if self.currently_logging:
                self.currently_logging = False
        else:
            if not self.currently_logging:
                self.currently_logging = True
                self.reset_logfile(control_signal)

        if self.currently_logging:
            self.dictionary['time'].append(time)
            self.dictionary['control_signal'].append(control_signal)
            self.dictionary['robot_sensor_signal'].append(robot_sensor_signal)
            self.dictionary['camera_sensor_signal'].append(camera_sensor_signal)
            self.dictionary['state_mean'].append(state_mean)
            self.dictionary['state_covariance'].append(state_covariance)

            self.line_count += 1
            if self.line_count > parameters.max_num_lines_before_write:
                self.line_count = 0
                with open(self.filename, 'wb') as file_handle:
                    pickle.dump(self.dictionary, file_handle)


# Utility for loading saved data
class DataLoader:
    def __init__(self, filename):
        self.filename = filename
        
    def load(self):
        with open(self.filename, 'rb') as file_handle:
            loaded_dict = pickle.load(file_handle)
        return loaded_dict


# Class to hold a message sender
class MsgSender:
    delta_send_time = 0.1

    def __init__(self, last_send_time, msg_size, udp_communication):
        self.last_send_time = last_send_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
        
    def send_control_signal(self, control_signal):
        packed_send_msg = self.pack_msg(control_signal)
        self.send(packed_send_msg)
    
    def send(self, msg):
        new_send_time = time.perf_counter()
        if new_send_time - self.last_send_time > self.delta_send_time:
            message = ""
            for data in msg:
                message = message + str(data)
            self.udp_communication.send_msg(message)
            self.last_send_time = new_send_time
      
    def pack_msg(self, msg):
        packed_msg = ""
        for data in msg:
            if packed_msg == "":
                packed_msg = packed_msg + str(data)
            else:
                packed_msg = packed_msg + ", "+ str(data)
        packed_msg = packed_msg + "\n"
        return packed_msg
        
        
# The robot's message receiver
class MsgReceiver:
    delta_receive_time = 0.05

    def __init__(self, last_receive_time, msg_size, udp_communication):
        self.last_receive_time = last_receive_time
        self.msg_size = msg_size
        self.udp_communication = udp_communication
      
    def receive(self):
        new_receive_time = time.perf_counter()
        if new_receive_time - self.last_receive_time > self.delta_receive_time:
            received_msg = self.udp_communication.receive_msg()
            self.last_receive_time = new_receive_time
            return True, received_msg
            
        return False, ""
    
    def unpack_msg(self, packed_msg):
        unpacked_msg = []
        msg_list = packed_msg.split(',')
        if len(msg_list) >= self.msg_size:
            for data in msg_list:
                unpacked_msg.append(float(data))
            return True, unpacked_msg

        return False, unpacked_msg
        
    def receive_robot_sensor_signal(self, last_robot_sensor_signal):
        robot_sensor_signal = last_robot_sensor_signal
        receive_ret, packed_receive_msg = self.receive()
        if receive_ret:
            unpack_ret, unpacked_receive_msg = self.unpack_msg(packed_receive_msg)
            if unpack_ret:
                robot_sensor_signal = RobotSensorSignal(unpacked_receive_msg)
            
        return robot_sensor_signal


# Class to hold a camera sensor data. 
class CameraSensor:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        # Updated to AprilTag dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_25h9)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # State variables for fixed-camera tracking
        self.T_cam_to_world = None
        self.target_tag_id = 5
        self.tag_size_m = 0.0762
        
    def get_signal(self, last_camera_signal):
        camera_signal = last_camera_signal
        ret, pose_estimate = self.get_pose_estimate()
        if ret:
            camera_signal = pose_estimate
        
        return camera_signal
        
    def get_pose_estimate(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, []
            
        h, w = frame.shape[:2]
        new_mat, _ = cv2.getOptimalNewCameraMatrix(parameters.camera_matrix, parameters.dist_coeffs, (w,h), 1, (w,h))
        frame = cv2.undistort(frame, parameters.camera_matrix, parameters.dist_coeffs, None, new_mat)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        
        if ids is not None and self.target_tag_id in ids:
            idx = np.where(ids == self.target_tag_id)[0][0]
            tag_corners = corners[idx][0]
            
            half_s = self.tag_size_m / 2.0
            obj_points = np.array([
                [-half_s,  half_s, 0], 
                [ half_s,  half_s, 0], 
                [ half_s, -half_s, 0], 
                [-half_s, -half_s, 0]  
            ], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(obj_points, tag_corners, parameters.camera_matrix, parameters.dist_coeffs)
            
            if not success:
                return False, []
                
            R_curr, _ = cv2.Rodrigues(rvec)
            
            T_cam_to_tag = np.eye(4)
            T_cam_to_tag[:3, :3] = R_curr
            T_cam_to_tag[:3, 3] = tvec.flatten()
            
            if self.T_cam_to_world is None:
                self.T_cam_to_world = np.linalg.inv(T_cam_to_tag)
                print("[+] Robot CameraSensor: Origin Locked. EKF Mapping Initialized.")
                
            T_world_to_tag = self.T_cam_to_world @ T_cam_to_tag
            
            z_x = T_world_to_tag[0, 3]
            z_y = T_world_to_tag[1, 3]
            z_z = T_world_to_tag[2, 3]
            
            R_world_tag = T_world_to_tag[:3, :3]
            front_vec_x = R_world_tag[0, 1]
            front_vec_y = R_world_tag[1, 1]
            z_yaw = math.atan2(front_vec_y, front_vec_x)
            
            pose_estimate = [z_x, z_y, z_z, 0.0, 0.0, z_yaw]
            return True, pose_estimate
            
        return False, []
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# A storage vessel for an instance of a robot signal
class RobotSensorSignal:
    def __init__(self, unpacked_msg):
        self.encoder_counts = int(unpacked_msg[0])
        self.steering = int(unpacked_msg[1])
        self.num_lidar_rays = int(unpacked_msg[2])
        self.angles = []
        self.distances = []
        for i in range(self.num_lidar_rays):
            index = 3 + i*2
            self.angles.append(unpacked_msg[index])
            self.distances.append(unpacked_msg[index+1])
    
    def print(self):
        print("Robot Sensor Signal")
        print(" encoder: ", self.encoder_counts)
        print(" steering:" , self.steering)
        print(" num_lidar_rays: ", self.num_lidar_rays)
        print(" angles: ",self.angles)
        print(" distances: ", self.distances)
    
    def to_list(self):
        sensor_data_list = []
        sensor_data_list.append(self.encoder_counts)
        sensor_data_list.append(self.steering)
        sensor_data_list.append(self.num_lidar_rays)
        for i in range(self.num_lidar_rays):
            sensor_data_list.append(self.angles[i])
            sensor_data_list.append(self.distances[i])
        
        return sensor_data_list