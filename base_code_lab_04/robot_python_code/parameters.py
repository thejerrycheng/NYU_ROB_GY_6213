# External libraries
import math
import numpy as np

# UDP parameters
localIP = "192.168.50.196" # "192.168.1.182" # Put your laptop computer's IP here 199
arduinoIP = "192.168.50.169" # "192.168.1.206" # Put your arduino's IP here 200
localPort = 4010
arduinoPort = 4010
bufferSize = 1024

# Camera parameters
camera_id = 0
marker_length = 0.071
camera_matrix = np.array([
    [1407.29350, 0.00000000, 975.201510],
    [0.00000000, 1405.70507, 534.098781], 
    [0.00000000, 0.00000000, 1.00000000]
])
dist_coeffs = np.array([
    [0.05904389, -0.03912268, -0.000681, 0.00122093, -0.09274777]
])

# Robot parameters
num_robot_sensors = 2 # encoder, steering
num_robot_control_signals = 2 # speed, steering

# Logging parameters
max_num_lines_before_write = 1
filename_start = './data/robot_data'
data_name_list = ['time', 'control_signal', 'robot_sensor_signal', 'camera_sensor_signal', 'state_mean', 'state_covariance']

# Experiment trial parameters
trial_time = 10000 # milliseconds
extra_trial_log_time = 2000 # milliseconds

# KF parameters
I3 = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
covariance_plot_scale = 100

# # PF parameters, modify the map and num particles as you see fit.
# num_particles = 1000
# wall_corner_list = [
#     [0, 0, 2.74, 0], 
#     [0, 0, 0, 3.78], 
#     [0, 3.78, 1.92, 3.78],
#     [1.03, 1.61, 1.03, 2.19],
#     [1.03, 2.19, 1.41, 2.19],
#     [1.92, 3.78, 1.92, 3.32],
#     [1.92, 3.32, 2.74, 3.32],
#     [2.74, 3.32, 2.74, 0]
#     ]

# PF parameters
num_particles = 1000

# Map: 1.2m x 1.8m room with a 0.6m x 0.6m obstacle in the top right.
# Coordinates are [x1, y1, x2, y2]
# wall_corner_list = [
#     [0.0, 0.0, 1.2, 0.0],       # Bottom wall
#     [0.0, 0.0, 0.0, 1.8],       # Left wall
#     [0.0, 1.8, 0.6, 1.8],       # Top wall (left half)
#     [0.6, 1.8, 0.6, 1.2],       # Obstacle left face
#     [0.6, 1.2, 1.2, 1.2],       # Obstacle bottom face
#     [1.2, 1.2, 1.2, 0.0]        # Right wall (bottom half)
# ]

distance_variance = 0.000363 # Lidar confidence


# --- Complicated Maze Map (4.0m x 4.0m) ---
# This layout includes an outer boundary, a central "H" divider, 
# and a disconnected inner pillar to create complex lidar returns.
wall_corner_list = [
    # Outer Boundary
    [0.0, 0.0, 4.0, 0.0],       # Bottom boundary
    [0.0, 0.0, 0.0, 4.0],       # Left boundary
    [0.0, 4.0, 4.0, 4.0],       # Top boundary
    [4.0, 4.0, 4.0, 0.0],       # Right boundary

    # Central Divider (Creates an L-corridor effect)
    [0.0, 2.0, 1.5, 2.0],       # Left horizontal divider
    [2.5, 2.0, 4.0, 2.0],       # Right horizontal divider
    
    # Internal Obstacles (Rooms)
    [1.0, 3.0, 1.0, 4.0],       # Vertical partition Top-Left
    [3.0, 0.0, 3.0, 1.0],       # Vertical partition Bottom-Right
    
    # The "Island" Obstacle (Center-Top)
    [1.8, 2.8, 2.2, 2.8],       # Bottom face
    [1.8, 3.2, 2.2, 3.2],       # Top face
    [1.8, 2.8, 1.8, 3.2],       # Left face
    [2.2, 2.8, 2.2, 3.2],       # Right face
    
    # The "Island" Obstacle (Center-Bottom)
    [1.8, 0.8, 2.2, 0.8],       # Bottom face
    [1.8, 1.2, 2.2, 1.2],       # Top face
    [1.8, 0.8, 1.8, 1.2],       # Left face
    [2.2, 0.8, 2.2, 1.2]        # Right face
]
