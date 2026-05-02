import math
import random
import argparse
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation, maximum_filter
import socket
import time

# ==========================================
# HYPERPARAMETERS (From your existing code)
# ==========================================
NUM_PARTICLES = 50  
PF_RESAMPLE_THRESHOLD = NUM_PARTICLES / 2.0
GRID_RESOLUTION = 0.05
L_0 = 0.0; L_OCC = 0.85; L_FREE = -0.4
MAX_LOG_ODDS = 5.0; MIN_LOG_ODDS = -5.0

L = 0.145
V_M = 0.004808; V_C = -0.045557
VAR_V = 0.00057829; VAR_DELTA = 0.00023134; VAR_LIDAR = 0.000363
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]

ROBOT_RADIUS = 0.15
PLANNER_WALL_CLEARANCE = 0.30  
LOOKAHEAD_DISTANCE = 0.4
GOAL_TOLERANCE = 0.20
MAX_V_CMD = 80.0; MAX_ALPHA_CMD = 100.0

PROB_FREE_THRESH = 0.55
PROB_UNKNOWN_LOW = 0.45; PROB_UNKNOWN_HIGH = 0.55
PROB_WALL_THRESH = 0.10

RENDER_SKIP = 5 
SLAM_SKIP = 5   

# ==========================================
# HARDWARE COMMUNICATION CLASSES
# ==========================================
class UDPCommunication:
    def __init__(self, arduinoIP, localIP, arduinoPort, localPort, bufferSize):
        self.arduinoIP = arduinoIP
        self.arduinoPort = arduinoPort
        self.localIP = localIP
        self.localPort = localPort
        self.bufferSize = bufferSize
        self.UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDPServerSocket.bind((localIP, localPort))
        self.UDPServerSocket.settimeout(0.1) # Prevent blocking forever on lost packets
        print(f"Listening for Robot on {localIP}:{localPort}")
        
    def receive_msg(self):
        try:
            bytesAddressPair = self.UDPServerSocket.recvfrom(self.bufferSize)
            return bytesAddressPair[0].decode()
        except socket.timeout:
            return None
       
    def send_msg(self, msg):
        bytesToSend = str.encode(msg)
        self.UDPServerSocket.sendto(bytesToSend, (self.arduinoIP, self.arduinoPort))

# Include your existing helper functions here...
# (angle_wrap, get_physical_commands, predict_next_pose, get_naive_frontier_mask)
# Include your existing FastSLAM, Particle, and ActiveSLAMController classes here...

# ==========================================
# HARDWARE PARSING UTILITIES
# ==========================================
def parse_robot_telemetry(msg):
    """
    Dummy parser: Adapt this to match exactly what your Arduino sends.
    Expected format example: "ODOM:v,w|LIDAR:dist1,dist2,...,distN"
    """
    try:
        parts = msg.split('|')
        v_real, w_real = map(float, parts[0].split(':')[1].split(','))
        
        # Assuming LIDAR sends 180 degrees of data (0 to 2*PI)
        lidar_str = parts[1].split(':')[1].split(',')
        lidar_distances = list(map(float, lidar_str))
        lidar_angles = np.linspace(0, 2 * math.pi, len(lidar_distances), endpoint=False).tolist()
        
        return v_real, w_real, lidar_angles, lidar_distances
    except Exception as e:
        print(f"Malformed UDP packet: {msg}")
        return None, None, None, None

# ==========================================
# REAL ROBOT MAIN EXECUTION LOOP
# ==========================================
def run_real_robot(arduino_ip, local_ip):
    # 1. Initialize UDP connection
    udp, connected = create_udp_communication(arduino_ip, local_ip, 8080, 8080, 4096)
    if not connected:
        return

    # 2. Define operational bounds (assume starting in a 10x10m unknown area)
    bounds = {
        'min_x': -5.0, 'max_x': 5.0,
        'min_y': -5.0, 'max_y': 5.0,
    }
    
    true_pose = np.array([0.0, 0.0, 0.0]) # Starting origin
    slam = FastSLAM(true_pose, bounds)
    ai_controller = ActiveSLAMController(slam)

    # 3. Setup Visualization (The "Middle Screen")
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Live Robot SLAM & Navigation')
    cmap = LinearSegmentedColormap.from_list('grid_map', ['white', 'lightgrey', 'black'])
    history_x, history_y = [], []

    step = 0
    delta_t = 0.1 # Approximate loop time

    print("\n=======================================================")
    print(" ROBOT DEPLOYMENT ACTIVE ")
    print("=======================================================\n")

    while plt.fignum_exists(fig.number):
        loop_start = time.time()

        # --- A. READ SENSORS ---
        msg = udp.receive_msg()
        if msg is None:
            continue # Wait for next packet
            
        v_real, w_real, angles, distances = parse_robot_telemetry(msg)
        if v_real is None: continue

        # --- B. SLAM PREDICT & UPDATE ---
        # Note: We use the *measured* velocity from the Arduino odometry for prediction
        delta_phys_est = math.atan((w_real * L) / v_real) if v_real != 0 else 0.0
        slam.predict(v_real, delta_phys_est, delta_t)
        
        if step % SLAM_SKIP == 0:
            slam.update(angles, distances)
            
        best_pose = slam.best_particle.pose
        history_x.append(best_pose[0]); history_y.append(best_pose[1])

        # --- C. PATH PLANNING & CONTROL ---
        v_cmd, alpha_cmd = ai_controller.update(best_pose, angles, distances)
        
        if v_cmd is None:
            print("Exploration Complete! Stopping robot.")
            udp.send_msg("CMD:0.0,0.0")
            break

        # Send command back to Arduino (Format: "CMD:v_cmd,alpha_cmd")
        command_string = f"CMD:{v_cmd:.2f},{alpha_cmd:.2f}"
        udp.send_msg(command_string)

        # --- D. VISUALIZATION ---
        if step % RENDER_SKIP == 0:
            ax.clear()
            prob_grid = slam.get_best_probabilities()
            
            # 1. Visualize Map Occupancy
            ax.imshow(prob_grid.T, cmap=cmap, origin='lower', extent=[bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']], vmin=0, vmax=1)

            # 2. Visualize Frontier (Magenta overlay)
            frontier_mask = get_naive_frontier_mask(prob_grid)
            overlay = np.zeros((frontier_mask.shape[0], frontier_mask.shape[1], 4))
            overlay[frontier_mask] = [1, 0, 1, 0.6]
            ax.imshow(overlay.swapaxes(0, 1), origin='lower', extent=[bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']])

            # 3. Visualize Robot Pose and Arrow
            ax.plot(history_x, history_y, 'b--', linewidth=1, alpha=0.5) # Trajectory
            ax.plot(best_pose[0], best_pose[1], 'ro', markersize=8) # Current pose point
            ax.arrow(best_pose[0], best_pose[1], 0.3 * math.cos(best_pose[2]), 0.3 * math.sin(best_pose[2]), 
                     head_width=0.1, head_length=0.1, fc='r', ec='r') # Heading arrow

            # 4. Visualize A* Path
            if ai_controller.current_path:
                path_x = [p[0] for p in ai_controller.current_path]
                path_y = [p[1] for p in ai_controller.current_path]
                ax.plot(path_x, path_y, 'c-', linewidth=2, label="A* Path")

            # 5. Visualize Target Frontier Goal
            if ai_controller.target_frontier:
                ax.plot(ai_controller.target_frontier[0], ai_controller.target_frontier[1], 'm*', markersize=15, label="Target Frontier")

            ax.set_title("Live FastSLAM | Magenta = Frontier | Cyan = A* Path")
            ax.set_xlim(bounds['min_x'], bounds['max_x']); ax.set_ylim(bounds['min_y'], bounds['max_y'])
            plt.pause(0.001)

        step += 1
        
        # Maintain loop timing consistency
        elapsed = time.time() - loop_start
        if elapsed < delta_t:
            time.sleep(delta_t - elapsed)

    plt.ioff()
    plt.show()

# Boilerplate execution
if __name__ == '__main__':
    # Define your IP addresses here
    ARDUINO_IP = "192.168.1.100" 
    LOCAL_IP = "192.168.1.50"
    run_real_robot(ARDUINO_IP, LOCAL_IP)