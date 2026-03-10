import os
import time
import math
import socket
import random
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import parameters
import robot_python_code

# =======================================================
# Configuration & Kinematic Constants
# =======================================================
INITIAL_POSE = [0.1, 0.1, math.pi / 2]

L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.023134

MAX_RANGE = 5.0
X_OFFSET = 0.12 
VAR_Z = 0.00025 

def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

# =======================================================
# 1. Pure Dead Reckoning (Deterministic Motion Model)
# =======================================================
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: 
            v_expected = 0.0

        alpha = steering_angle_command
        delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]

        w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0

        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2] = angle_wrap(self.state[2] - delta_t * w_expected)

        return self.state

# =======================================================
# 2. Particle Filter (Stochastic Motion Model + Lidar)
# =======================================================
class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.weight = 1.0
        self.log_w = 0.0 

    def predict(self, v_cmd, alpha_cmd, dt):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: 
            v_expected = 0.0
            
        delta_expected = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]

        if v_expected > 0:
            v_s = v_expected + random.gauss(0, math.sqrt(VAR_V))
            d_s = delta_expected + random.gauss(0, math.sqrt(VAR_DELTA))
        else:
            v_s = 0.0
            d_s = delta_expected

        w_s = (v_s * math.tan(d_s)) / L if L > 0 else 0.0
        
        self.x += v_s * math.cos(self.theta) * dt
        self.y += v_s * math.sin(self.theta) * dt
        self.theta = angle_wrap(self.theta - w_s * dt) 

    def update_weight(self, angles, distances):
        log_w = 0.0
        ray_step = 10 
        
        xs = self.x + X_OFFSET * math.cos(self.theta)
        ys = self.y + X_OFFSET * math.sin(self.theta)

        for i in range(0, len(angles), ray_step):
            raw_dist = distances[i]
            
            if raw_dist < 100 or raw_dist >= 4900: 
                continue
            
            dist_m = raw_dist / 1000.0
            angle_rad = -(angles[i] * math.pi / 180.0)
            global_angle = angle_wrap(self.theta + angle_rad)
            
            rx, ry = math.cos(global_angle), math.sin(global_angle)
            min_dist = MAX_RANGE
            
            for wall in parameters.wall_corner_list:
                qx, qy, bx, by = wall
                sx, sy = bx - qx, by - qy
                denom = rx * sy - ry * sx
                if abs(denom) > 1e-6:
                    t = ((qx - xs) * sy - (qy - ys) * sx) / denom
                    u = ((qx - xs) * ry - (qy - ys) * rx) / denom
                    if 0 <= u <= 1 and 0 < t < min_dist:
                        min_dist = t
                        
            if min_dist < MAX_RANGE:
                error = min_dist - dist_m
                penalty = (error**2) / (2 * VAR_Z)
                log_w -= min(penalty, 10.0) 
            else:
                log_w -= 10.0 
            
        self.log_w = log_w

class ParticleFilter:
    def __init__(self, num_particles, initial_pose):
        self.num_particles = num_particles
        self.particles = []
        
        all_walls = np.array(parameters.wall_corner_list)
        self.x_min = np.min(all_walls[:, [0, 2]])
        self.x_max = np.max(all_walls[:, [0, 2]])
        self.y_min = np.min(all_walls[:, [1, 3]])
        self.y_max = np.max(all_walls[:, [1, 3]])
        
        self.global_initialization(initial_pose)

    def global_initialization(self, pose):
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.gauss(pose[0], 0.1)
            y = random.gauss(pose[1], 0.1)
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max):
                continue
            theta = angle_wrap(random.gauss(pose[2], 0.1))
            self.particles.append(Particle(x, y, theta))

    def predict(self, v_cmd, alpha_cmd, dt):
        for p in self.particles: 
            p.predict(v_cmd, alpha_cmd, dt)

    def correct(self, angles, distances):
        if len(angles) < 10: return 

        for p in self.particles: 
            p.update_weight(angles, distances)
            
        max_log_w = max(p.log_w for p in self.particles)
        for p in self.particles:
            p.weight = math.exp(p.log_w - max_log_w)
            
        self.resample()

    def resample(self):
        weights = [p.weight for p in self.particles]
        new_particles = []
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        max_w = max(weights)
        
        for _ in range(self.num_particles):
            beta += random.uniform(0, 2.0 * max_w)
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % self.num_particles
            
            p = self.particles[index]
            new_particles.append(Particle(p.x, p.y, p.theta))
            
        self.particles = new_particles

    def get_estimate(self):
        sum_x = sum(p.x for p in self.particles)
        sum_y = sum(p.y for p in self.particles)
        sum_sin = sum(math.sin(p.theta) for p in self.particles)
        sum_cos = sum(math.cos(p.theta) for p in self.particles)
        return [sum_x / self.num_particles, sum_y / self.num_particles, math.atan2(sum_sin, sum_cos)]
    
# =======================================================
# 3. Live UDP Communication & Main Loop
# =======================================================
def main():
    # 1. Prompt user for commands
    print("=== Live Robot Particle Filter ===")
    try:
        speed = float(input("Enter desired speed: "))
        steering = float(input("Enter desired steering: "))
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return

    # 2. Setup UDP communication
    print("\nInitializing UDP socket...")
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, parameters.localIP, 
        parameters.arduinoPort, parameters.localPort, parameters.bufferSize
    )
    if not success:
        print("Failed to bind UDP socket. Check your network.")
        return

    udp.UDPServerSocket.settimeout(0.5)

    msg_sender = robot_python_code.MsgSender(time.perf_counter(), 2, udp)
    msg_receiver = robot_python_code.MsgReceiver(time.perf_counter(), 3, udp)
    sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])

    # 3. Connection Handshake
    print("Waiting for robot telemetry (5-second timeout)...")
    connected = False
    start_wait = time.perf_counter()
    
    while time.perf_counter() - start_wait < 5.0:
        try:
            msg_sender.send_control_signal([0.0, 0.0])
            raw_msg = udp.receive_msg()
            if raw_msg:
                print("Connection established successfully!")
                connected = True
                break
        except socket.timeout:
            pass 
            
    if not connected:
        print("Timeout: Failed to connect to the robot. Exiting.")
        return

    udp.UDPServerSocket.settimeout(0.05)

    # 4. Setup Filters and Visualization
    pf = ParticleFilter(num_particles=1500, initial_pose=INITIAL_POSE)
    dr = MyMotionModel(initial_state=INITIAL_POSE)
    
    all_walls = np.array(parameters.wall_corner_list)
    x_min, x_max = np.min(all_walls[:, [0, 2]]), np.max(all_walls[:, [0, 2]])
    y_min, y_max = np.min(all_walls[:, [1, 3]]), np.max(all_walls[:, [1, 3]])
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    history = {
        'est_x': [], 'est_y': [],
        'dr_x': [], 'dr_y': []
    }

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    sweep_angles = []
    sweep_distances = []
    last_lidar_angle = None
    update_count = 0

    print(f"\nStarting live filter for 5 seconds...")
    start_time = time.perf_counter()
    last_time = start_time
    last_print_time = start_time

    # 5. Live Loop
    try:
        while time.perf_counter() - start_time < 5.0:
            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time
            
            # Send Command
            msg_sender.send_control_signal([speed, steering])
            
            # Receive UDP Telemetry
            try:
                sensor_signal = msg_receiver.receive_robot_sensor_signal(sensor_signal)
                raw_angles = sensor_signal.angles
                raw_distances = sensor_signal.distances
            except socket.timeout:
                raw_angles = []
                raw_distances = []
            
            # Prediction Step
            if dt > 0:
                pf.predict(speed, steering, dt)
                dr.step_update(speed, steering, dt)
                
            # Accumulate Lidar
            sweep_complete = False
            for i in range(len(raw_angles)):
                ang = raw_angles[i]
                dist = raw_distances[i]
                
                if last_lidar_angle is not None and abs(ang - last_lidar_angle) > 180:
                    sweep_complete = True
                    
                sweep_angles.append(ang)
                sweep_distances.append(dist)
                last_lidar_angle = ang
            
            # Correction Step & Visualization Trigger
            if sweep_complete:
                pf.correct(sweep_angles, sweep_distances)
                
                est_pose = pf.get_estimate()
                history['est_x'].append(est_pose[0])
                history['est_y'].append(est_pose[1])
                history['dr_x'].append(dr.state[0])
                history['dr_y'].append(dr.state[1])
                
                # Render Real-Time Map
                ax.clear()
                for wall in parameters.wall_corner_list:
                    ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2, zorder=1)
                
                px, py = [p.x for p in pf.particles], [p.y for p in pf.particles]
                ax.scatter(px, py, s=2, c='green', alpha=0.15, zorder=2, label='Particles')
                
                ax.plot(history['dr_x'], history['dr_y'], color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Motion Model (DR)', zorder=3)
                ax.plot(history['est_x'], history['est_y'], color='blue', linestyle='-', linewidth=1.5, alpha=0.6, label='PF Estimate', zorder=4)

                ray_x, ray_y = [], []
                xs = est_pose[0] + X_OFFSET * math.cos(est_pose[2])
                ys = est_pose[1] + X_OFFSET * math.sin(est_pose[2])
                
                for i in range(0, len(sweep_angles), 5): 
                    raw_dist = sweep_distances[i]
                    if 100 < raw_dist < 4900:
                        dist = raw_dist / 1000.0
                        angle = angle_wrap(-(sweep_angles[i] * math.pi / 180.0) + est_pose[2])
                        hx = xs + dist * math.cos(angle)
                        hy = ys + dist * math.sin(angle)
                        ray_x.extend([xs, hx, None])
                        ray_y.extend([ys, hy, None])
                
                ax.plot(ray_x, ray_y, color='red', alpha=0.15, linewidth=0.5, zorder=5)

                ax.plot(dr.state[0], dr.state[1], 'o', color='orange', markersize=6, zorder=6)
                ax.plot(est_pose[0], est_pose[1], 'mo', markersize=6, zorder=7)
                ax.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), color='magenta', scale=15, width=0.007, zorder=8)

                time_left = 5.0 - (current_time - start_time)
                ax.set_title(f"LIVE Physical PF | Update: {update_count} | T-{time_left:.1f}s")
                ax.set_xlim(x_min - x_pad, x_max + x_pad); ax.set_ylim(y_min - y_pad, y_max + y_pad)
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize='x-small')
                
                plt.pause(0.001)
                
                update_count += 1
                sweep_angles = []
                sweep_distances = []

            # Print Countdown
            if current_time - last_print_time >= 1.0:
                time_left = 5.0 - (current_time - start_time)
                print(f"Time remaining: {time_left:.1f} seconds")
                last_print_time = current_time

            # Manage loop rate
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nLive run interrupted by user.")
        
    finally:
        # SAFETY: Stop the robot
        print("\nStopping the robot...")
        for _ in range(5): 
            try:
                msg_sender.send_control_signal([0.0, 0.0])
                time.sleep(0.05)
            except socket.timeout:
                pass
                
        plt.ioff()
        print("Run complete! Close the plot window to exit.")
        plt.show()

if __name__ == "__main__":
    main()