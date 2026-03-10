# External libraries
import asyncio
import cv2
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response

# Local libraries
from robot import Robot
import robot_python_code
import parameters

# Global variables
logging = False
stream_video = False

# =======================================================
# Particle Filter Configuration & Kinematic Constants
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
VAR_Z = 0.0025 # 5cm map artifact tolerance

def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: v_expected = 0.0
        alpha = steering_angle_command
        delta_expected = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]
        w_expected = (v_expected * math.tan(delta_expected)) / L if L > 0 else 0

        self.state[0] += delta_t * v_expected * math.cos(self.state[2])
        self.state[1] += delta_t * v_expected * math.sin(self.state[2])
        self.state[2] = angle_wrap(self.state[2] - delta_t * w_expected)

class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        self.weight = 1.0
        self.log_w = 0.0 

    def predict(self, v_cmd, alpha_cmd, dt):
        v_expected = (V_M * v_cmd) + V_C
        if v_expected < 0 and v_cmd > 0: v_expected = 0.0
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
            if raw_dist < 100 or raw_dist >= 4900: continue
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
                    if 0 <= u <= 1 and 0 < t < min_dist: min_dist = t
                        
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
        self.x_min, self.x_max = np.min(all_walls[:, [0, 2]]), np.max(all_walls[:, [0, 2]])
        self.y_min, self.y_max = np.min(all_walls[:, [1, 3]]), np.max(all_walls[:, [1, 3]])
        self.global_initialization(initial_pose)

    def global_initialization(self, pose):
        self.particles = []
        while len(self.particles) < self.num_particles:
            x = random.gauss(pose[0], 0.1)
            y = random.gauss(pose[1], 0.1)
            if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max): continue
            theta = angle_wrap(random.gauss(pose[2], 0.1))
            self.particles.append(Particle(x, y, theta))

    def predict(self, v_cmd, alpha_cmd, dt):
        for p in self.particles: p.predict(v_cmd, alpha_cmd, dt)

    def correct(self, angles, distances):
        if len(angles) < 10: return 
        for p in self.particles: p.update_weight(angles, distances)
        max_log_w = max(p.log_w for p in self.particles)
        for p in self.particles: p.weight = math.exp(p.log_w - max_log_w)
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


# Frame converter for the video stream
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()
    
def connect_with_camera():
    video_capture = cv2.VideoCapture(1)
    return video_capture
    
def update_video(video_image):
    if stream_video:
        video_image.force_reload()

def get_time_in_ms():
    return int(time.time()*1000)

# Create the gui page
@ui.page('/')
def main():
    # Robot variables
    robot = Robot()

    # --- Initialize Live Particle Filter State ---
    # Using 800 particles so the GUI thread doesn't bottleneck
    pf = ParticleFilter(num_particles=800, initial_pose=INITIAL_POSE)
    dr = MyMotionModel(initial_state=INITIAL_POSE)
    
    # Pre-compute map bounds for dynamic UI zooming
    all_walls = np.array(parameters.wall_corner_list)
    x_min, x_max = np.min(all_walls[:, [0, 2]]), np.max(all_walls[:, [0, 2]])
    y_min, y_max = np.min(all_walls[:, [1, 3]]), np.max(all_walls[:, [1, 3]])
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    history = {'est_x': [], 'est_y': [], 'dr_x': [], 'dr_y': []}
    
    pf_state = {
        'last_time': time.time(),
        'sweep_angles': [],
        'sweep_distances': [],
        'last_lidar_angle': None
    }

    # Set dark mode for gui
    dark = ui.dark_mode()
    dark.value = True
    
    if stream_video:
        video_capture = cv2.VideoCapture(parameters.camera_id)
    
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return placeholder
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    def update_commands():
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
                print("End Trial :", delta_time)
        
        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                logging_switch.value = False
                robot.extra_logging = False

        cmd_speed = slider_speed.value if speed_switch.value else 0
        cmd_steering_angle = slider_steering.value if steering_switch.value else 0
        return cmd_speed, cmd_steering_angle
        
    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(parameters.arduinoIP, parameters.localIP, parameters.arduinoPort, parameters.localPort, parameters.bufferSize)
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    print("Should be set for UDP!")
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False

    def enable_speed(): pass
    def enable_steering(): pass

    # --- Live Particle Filter Visualization ---
    def show_localization_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('black')
            plt.clf()
            plt.style.use('dark_background')
            plt.tick_params(axis='x', colors='lightgray')
            plt.tick_params(axis='y', colors='lightgray')

            # 1. Draw Map
            for wall in parameters.wall_corner_list:
                plt.plot([wall[0], wall[2]], [wall[1], wall[3]], color='white', linewidth=2, zorder=1)

            # 2. Draw Particles (Subsampled to keep GUI fast)
            px, py = [p.x for p in pf.particles[::2]], [p.y for p in pf.particles[::2]]
            plt.scatter(px, py, s=2, c='green', alpha=0.3, zorder=2)

            # 3. Draw Paths
            plt.plot(history['dr_x'], history['dr_y'], color='orange', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
            plt.plot(history['est_x'], history['est_y'], color='blue', linestyle='-', linewidth=1.5, alpha=0.6, zorder=4)

            # 4. Draw Current Lidar Rays (Using current live data buffer)
            est_pose = pf.get_estimate()
            xs = est_pose[0] + X_OFFSET * math.cos(est_pose[2])
            ys = est_pose[1] + X_OFFSET * math.sin(est_pose[2])
            
            # Show the raw rays hitting right now
            for i in range(0, robot.robot_sensor_signal.num_lidar_rays, 5):
                dist = robot.robot_sensor_signal.distances[i]
                if 100 < dist < 4900:
                    dist_m = dist / 1000.0
                    ang_rad = -(robot.robot_sensor_signal.angles[i] * math.pi / 180.0)
                    global_angle = angle_wrap(est_pose[2] + ang_rad)
                    hx = xs + dist_m * math.cos(global_angle)
                    hy = ys + dist_m * math.sin(global_angle)
                    plt.plot([xs, hx], [ys, hy], color='red', alpha=0.2, linewidth=0.5, zorder=5)

            # 5. Draw Robot States
            plt.plot(dr.state[0], dr.state[1], 'o', color='orange', markersize=5, zorder=6)
            plt.plot(est_pose[0], est_pose[1], 'mo', markersize=5, zorder=7)
            plt.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), color='magenta', scale=15, width=0.007, zorder=8)

            plt.grid(True, linestyle='--', alpha=0.3)
            plt.xlim(x_min - x_pad, x_max + x_pad)
            plt.ylim(y_min - y_pad, y_max + y_pad)
            plt.gca().set_aspect('equal', adjustable='box')

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        print("Start time:", robot.trial_start_time)

    # Create the gui title bar
    with ui.card().classes('w-full  items-center'):
        ui.label('ROB-GY - 6213: Robot Navigation & Localization').style('font-size: 24px;')
    
    with ui.card().classes('w-full'):
        with ui.grid(columns=3).classes('w-full items-center'):
            with ui.card().classes('w-full items-center h-60'):
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full h-full')
                else:
                    ui.image('./a_robot_image.jpg').props('height=2')
                    video_image = None
            
            # Matplotlib Visualizer Window
            with ui.card().classes('w-full items-center h-60'):
                main_plot = ui.pyplot(figsize=(3, 3))
                
            with ui.card().classes('items-center h-60'):
                ui.label('Encoder:').style('text-align: center;')
                encoder_count_label = ui.label('0')
                logging_switch = ui.switch('Data Logging ')
                udp_switch = ui.switch('Robot Connect')
                run_trial_button = ui.button('Run Trial', on_click=lambda:run_trial())
                
    # Sliders
    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('SPEED:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_speed = ui.slider(min=0, max=100, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_speed, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                speed_switch = ui.switch('Enable', on_change=lambda: enable_speed())

    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('STEER:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_steering = ui.slider(min=-20, max=20, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_steering, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                steering_switch = ui.switch('Enable', on_change=lambda: enable_steering())
        

    # --- Integrated Control Loop ---
    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
        
        # 1. Setup Time Delta
        current_time = time.time()
        dt = current_time - pf_state['last_time']
        pf_state['last_time'] = current_time

        # 2. Step the Filter (Only if Hardware is Active & Flowing)
        if robot.connected_to_hardware and dt > 0:
            
            # Predict
            pf.predict(cmd_speed, cmd_steering_angle, dt)
            dr.step_update(cmd_speed, cmd_steering_angle, dt)

            # Accumulate 360 Sweep
            sweep_complete = False
            for i in range(robot.robot_sensor_signal.num_lidar_rays):
                ang = robot.robot_sensor_signal.angles[i]
                dist = robot.robot_sensor_signal.distances[i]
                
                if pf_state['last_lidar_angle'] is not None and abs(ang - pf_state['last_lidar_angle']) > 180:
                    sweep_complete = True
                    
                pf_state['sweep_angles'].append(ang)
                pf_state['sweep_distances'].append(dist)
                pf_state['last_lidar_angle'] = ang
            
            # Correct
            if sweep_complete:
                pf.correct(pf_state['sweep_angles'], pf_state['sweep_distances'])
                pf_state['sweep_angles'] = []
                pf_state['sweep_distances'] = []

            # Save Live History
            est_pose = pf.get_estimate()
            history['est_x'].append(est_pose[0])
            history['est_y'].append(est_pose[1])
            history['dr_x'].append(dr.state[0])
            history['dr_y'].append(dr.state[1])

        # 3. Always refresh the UI Plot
        show_localization_plot()
        
    ui.timer(0.1, control_loop)

# Run the gui
ui.run(native=True)