# External libraries
import os
import asyncio
import cv2
import math
import random
import socket
from matplotlib import pyplot as plt
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import strftime

# Local libraries
import robot_python_code
import parameters

# =======================================================
# Configuration & Kinematic Constants
# =======================================================
stream_video = True
DATA_DIR = "online_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

INITIAL_POSE = [0.1, 0.2, math.pi / 2]

L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.023134

MAX_RANGE = 5.0
X_OFFSET = 0.12 
VAR_Z = 0.0025 

# --- THEME COLORS ---
CARD_BG = 'bg-slate-900'
TEXT_COLOR = 'text-slate-200'
HEADER_BG = 'bg-slate-950'

def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

# =======================================================
# Math Models (Dead Reckoning & Particle Filter)
# =======================================================
class MyMotionModel:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)

    def step_update(self, v_cmd, steering_angle_command, delta_t):
        if v_cmd == 0.0:
            v_expected = 0.0
            w_expected = 0.0
        else:
            v_expected = (V_M * v_cmd) + V_C
            if v_expected < 0: v_expected = 0.0
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
        if v_cmd == 0.0:
            v_s = 0.0
            w_s = 0.0
        else:
            v_expected = (V_M * v_cmd) + V_C
            if v_expected < 0: v_expected = 0.0
            delta_expected = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]

            v_s = v_expected + random.gauss(0, math.sqrt(VAR_V))
            if v_s < 0: v_s = 0.0
            d_s = delta_expected + random.gauss(0, math.sqrt(VAR_DELTA))
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


# =======================================================
# Helpers
# =======================================================
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

def get_time_in_ms():
    return int(time.time()*1000)

# =======================================================
# Main GUI Application
# =======================================================
@ui.page('/')
def main():
    # --- STYLING SETUP ---
    dark = ui.dark_mode()
    dark.value = True
    ui.add_head_html('''
        <style>
            .nicegui-content { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
            .q-card { border-radius: 16px; border: 1px solid #334155; }
            .q-slider__track-container { height: 6px; border-radius: 3px; }
        </style>
    ''')

    # --- Application State ---
    state = {
        'connected': False,
        'udp': None,
        'sender': None,
        'receiver': None,
        'sensor_signal': robot_python_code.RobotSensorSignal([0, 0, 0]),
        
        'running_trial': False,
        'trial_start_time': 0,
        'base_filename': "",
        'csv_file': None,
        'video_writer': None,
        
        'pf_last_time': time.time(),
        'sweep_angles': [],
        'sweep_distances': [],
        'last_lidar_angle': None,
        
        'latest_frame': None
    }

    # Initialize Filters & Paths
    pf = ParticleFilter(num_particles=800, initial_pose=INITIAL_POSE)
    dr = MyMotionModel(initial_state=INITIAL_POSE)
    
    all_walls = np.array(parameters.wall_corner_list)
    x_min, x_max = np.min(all_walls[:, [0, 2]]), np.max(all_walls[:, [0, 2]])
    y_min, y_max = np.min(all_walls[:, [1, 3]]), np.max(all_walls[:, [1, 3]])
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    history = {'est_x': [], 'est_y': [], 'dr_x': [], 'dr_y': []}
    
    # Camera Init
    if stream_video:
        try:
            video_capture = cv2.VideoCapture(parameters.camera_id)
        except:
            video_capture = cv2.VideoCapture(0) # Fallback
    
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not stream_video or not video_capture.isOpened():
            return Response(content=b'', media_type='image/jpeg')
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return Response(content=b'', media_type='image/jpeg')
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    def update_connection_to_robot():
        if udp_switch.value and not state['connected']:
            udp, success = robot_python_code.create_udp_communication(
                parameters.arduinoIP, parameters.localIP, 
                parameters.arduinoPort, parameters.localPort, parameters.bufferSize
            )
            if success:
                udp.UDPServerSocket.settimeout(0.05)
                state['udp'] = udp
                state['sender'] = robot_python_code.MsgSender(time.perf_counter(), 2, udp)
                state['receiver'] = robot_python_code.MsgReceiver(time.perf_counter(), 3, udp)
                state['connected'] = True
                status_indicator.classes('bg-green-500', remove='bg-red-500')
                status_label.set_text('Connected')
            else:
                udp_switch.value = False
        elif not udp_switch.value and state['connected']:
            if state['sender']: state['sender'].send_control_signal([0, 0])
            state['connected'] = False
            status_indicator.classes('bg-red-500', remove='bg-green-500')
            status_label.set_text('Disconnected')

    def run_trial():
        if not state['connected']:
            ui.notify('Must connect to hardware first!', type='warning')
            return
            
        cmd_s = slider_speed.value
        cmd_st = slider_steering.value
        
        state['trial_start_time'] = get_time_in_ms()
        state['running_trial'] = True
        steering_switch.value = True
        speed_switch.value = True
        
        # Reset Histories for clear plot
        history['est_x'].clear(); history['est_y'].clear()
        history['dr_x'].clear(); history['dr_y'].clear()
        state['sweep_angles'].clear()
        state['sweep_distances'].clear()
        
        # Start Data Logging
        date_str = strftime("%Y_%m_%d_%H_%M_%S")
        state['base_filename'] = f"{DATA_DIR}/{cmd_s}_{cmd_st}_{date_str}"
        
        # Open CSV
        state['csv_file'] = open(f"{state['base_filename']}_dataset.csv", 'w')
        state['csv_file'].write("Time_s,Encoder_Counts,Steering,Lidar_Angles,Lidar_Distances\n")
        
        # Open Video Writer
        if stream_video and video_capture.isOpened():
            w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            state['video_writer'] = cv2.VideoWriter(f"{state['base_filename']}_video.mp4", fourcc, 10.0, (w, h))

        ui.notify('5-Second Trial Started & Recording...', type='positive')

    def end_trial():
        state['running_trial'] = False
        speed_switch.value = False
        steering_switch.value = False
        if state['sender']: state['sender'].send_control_signal([0, 0])
        
        # Save CSV
        if state['csv_file'] is not None:


            state['csv_file'] = None
            
        # Save Video
        if state['video_writer'] is not None:
            state['video_writer'].release()
            state['video_writer'] = None
            
        # Save Plot Image
        fig = main_plot.fig
        fig.savefig(f"{state['base_filename']}_plot.png", dpi=300, bbox_inches='tight', facecolor='#0f172a')
        
        ui.notify(f'Trial Saved to {DATA_DIR}/!', type='info')
        trial_timer_label.set_text('0.0s')

    # --- Live Visualization Plot ---
    def show_localization_plot():
        with main_plot:
            fig = main_plot.fig
            fig.patch.set_facecolor('#0f172a')
            plt.clf()
            
            ax = plt.gca()
            ax.set_facecolor('#0f172a')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_color('#334155')
            ax.spines['right'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(axis='x', colors='#94a3b8')
            ax.tick_params(axis='y', colors='#94a3b8')

            # 1. Map
            for wall in parameters.wall_corner_list:
                ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='white', linewidth=2, zorder=1)

            # 2. Particles
            px, py = [p.x for p in pf.particles[::2]], [p.y for p in pf.particles[::2]]
            ax.scatter(px, py, s=2, c='#22c55e', alpha=0.3, zorder=2)

            # 3. Trajectories
            if len(history['dr_x']) > 0:
                ax.plot(history['dr_x'], history['dr_y'], color='#f97316', linestyle='--', linewidth=1.5, alpha=0.8, label='Dead Reckoning', zorder=3)
                ax.plot(history['est_x'], history['est_y'], color='#3b82f6', linestyle='-', linewidth=1.5, alpha=0.6, label='Particle Filter', zorder=4)

            # 4. Lidar Rays
            est_pose = pf.get_estimate()
            xs = est_pose[0] + X_OFFSET * math.cos(est_pose[2])
            ys = est_pose[1] + X_OFFSET * math.sin(est_pose[2])
            
            for i in range(0, state['sensor_signal'].num_lidar_rays, 5):
                dist = state['sensor_signal'].distances[i]
                if 100 < dist < 4900:
                    dist_m = dist / 1000.0
                    ang_rad = -(state['sensor_signal'].angles[i] * math.pi / 180.0)
                    global_angle = angle_wrap(est_pose[2] + ang_rad)
                    hx = xs + dist_m * math.cos(global_angle)
                    hy = ys + dist_m * math.sin(global_angle)
                    ax.plot([xs, hx], [ys, hy], color='#ef4444', alpha=0.2, linewidth=0.5, zorder=5)

            # 5. Robot Indicators
            dr_pose = dr.state if len(history['dr_x']) == 0 else [dr.state[0], dr.state[1]]
            ax.plot(dr_pose[0], dr_pose[1], 'o', color='#f97316', markersize=5, zorder=6)
            ax.plot(est_pose[0], est_pose[1], 'o', color='#3b82f6', markersize=5, zorder=7)
            ax.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), color='#ec4899', scale=15, width=0.007, zorder=8)

            ax.grid(True, color='#1e293b', linestyle='--')
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect('equal')
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left', facecolor='#0f172a', edgecolor='#334155', labelcolor='#94a3b8', fontsize=8)


    # --- GUI LAYOUT ---
    with ui.header().classes(f'{HEADER_BG} shadow-md p-4 flex items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('smart_toy', size='32px', color='blue-400')
            ui.label('Robot Command Center').classes('text-xl font-bold tracking-wide text-white')
        
        with ui.row().classes('items-center gap-2 bg-slate-800 px-3 py-1 rounded-full'):
            status_indicator = ui.element('div').classes('w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]')
            status_label = ui.label('Disconnected').classes('text-xs font-semibold text-slate-300')

    with ui.column().classes('w-full p-6 gap-6 items-center max-w-7xl mx-auto'):
        
        with ui.grid(columns=3).classes('w-full gap-6'):
            
            with ui.card().classes(f'w-full {CARD_BG} p-0 overflow-hidden relative group'):
                ui.label('Camera Feed').classes('absolute top-3 left-4 z-10 text-xs font-bold text-white/70 bg-black/50 px-2 py-1 rounded backdrop-blur-sm')
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full h-64 object-cover')
                else:
                    with ui.column().classes('w-full h-64 items-center justify-center bg-slate-800'):
                        ui.icon('videocam_off', size='48px', color='slate-600')
                        ui.label('No Video Stream').classes('text-slate-500 text-sm mt-2')
                    video_image = None
            
            with ui.card().classes(f'w-full {CARD_BG} items-center justify-center p-2'):
                main_plot = ui.pyplot(figsize=(3.5, 3.5), close=False) 
            
            with ui.card().classes(f'w-full {CARD_BG} p-5 flex flex-col justify-between h-full'):
                ui.label('Telemetry').classes('text-sm font-bold text-slate-400 mb-4 uppercase tracking-wider')
                with ui.row().classes('items-baseline justify-between w-full mb-2'):
                    ui.label('Encoder Count').classes(TEXT_COLOR)
                    encoder_count_label = ui.label('0').classes('text-2xl font-mono text-blue-400')
                ui.separator().classes('bg-slate-700 my-4')
                with ui.column().classes('w-full gap-3'):
                    udp_switch = ui.switch('Hardware Connection').props('color=green keep-color').classes('text-slate-300 w-full')
                    with ui.row().classes('w-full items-center justify-between mt-2'):
                        ui.button('START TRIAL', on_click=lambda:run_trial()).props('unelevated').classes('bg-blue-600 hover:bg-blue-500 text-white w-2/3 rounded-lg font-bold')
                        trial_timer_label = ui.label('0.0s').classes('text-slate-400 font-mono text-sm')

        with ui.card().classes(f'w-full {CARD_BG} p-6'):
            ui.label('Drive Control').classes('text-sm font-bold text-slate-400 mb-6 uppercase tracking-wider')
            with ui.grid(columns=2).classes('w-full gap-12'):
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('speed', color='blue-400')
                            ui.label('Speed').classes('text-lg font-medium text-white')
                        speed_switch = ui.switch().props('color=blue dense')
                    slider_speed = ui.slider(min=-100, max=100, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')

                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('directions_car', color='blue-400')
                            ui.label('Steering').classes('text-lg font-medium text-white')
                        steering_switch = ui.switch().props('color=blue dense')
                    slider_steering = ui.slider(min=-20, max=20, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')

    # --- Integrated Control Loop (Runs every 100ms) ---
    async def control_loop():
        update_connection_to_robot()
        
        # 1. Trial Timer Check
        cmd_speed, cmd_steer = 0, 0
        if state['running_trial']:
            dt_trial = get_time_in_ms() - state['trial_start_time']
            trial_timer_label.set_text(f"{dt_trial/1000:.1f}s")
            if dt_trial > 5000:
                end_trial()
            else:
                if speed_switch.value: cmd_speed = slider_speed.value
                if steering_switch.value: cmd_steer = slider_steering.value
        else:
            if speed_switch.value: cmd_speed = slider_speed.value
            if steering_switch.value: cmd_steer = slider_steering.value

        # 2. Camera Frame Processing
        if stream_video and video_capture.isOpened():
            read_result = await run.io_bound(video_capture.read)
            if read_result is not None:
                ret, state['latest_frame'] = read_result
                
                # Write to video file if trial is running
                if state['running_trial'] and state['video_writer'] is not None and state['latest_frame'] is not None:
                    state['video_writer'].write(state['latest_frame'])
                    
                if state['latest_frame'] is not None and video_image is not None:
                    video_image.force_reload()

        # 3. Hardware Comms & PF Update
        current_time = time.time()
        dt_pf = current_time - state['pf_last_time']
        state['pf_last_time'] = current_time

        if state['connected']:
            state['sender'].send_control_signal([cmd_speed, cmd_steer])
            
            try:
                state['sensor_signal'] = state['receiver'].receive_robot_sensor_signal(state['sensor_signal'])
                encoder_count_label.set_text(str(state['sensor_signal'].encoder_counts))
            except socket.timeout:
                pass

            # Write to CSV if trial is running
            if state['running_trial'] and state['csv_file'] is not None:
                t_rel = (get_time_in_ms() - state['trial_start_time']) / 1000.0
                enc = state['sensor_signal'].encoder_counts
                angs = str(state['sensor_signal'].angles)
                dsts = str(state['sensor_signal'].distances)
                state['csv_file'].write(f"{t_rel:.3f},{enc},{cmd_steer},\"{angs}\",\"{dsts}\"\n")

            # Predict Filters
            if dt_pf > 0:
                pf.predict(cmd_speed, cmd_steer, dt_pf)
                dr.step_update(cmd_speed, cmd_steer, dt_pf)

            # Accumulate Lidar Sweep
            sweep_complete = False
            for i in range(state['sensor_signal'].num_lidar_rays):
                ang = state['sensor_signal'].angles[i]
                dist = state['sensor_signal'].distances[i]
                if state['last_lidar_angle'] is not None and abs(ang - state['last_lidar_angle']) > 180:
                    sweep_complete = True
                state['sweep_angles'].append(ang)
                state['sweep_distances'].append(dist)
                state['last_lidar_angle'] = ang
            
            # Correct Filters
            if sweep_complete:
                pf.correct(state['sweep_angles'], state['sweep_distances'])
                state['sweep_angles'] = []
                state['sweep_distances'] = []

            # Update History
            est_pose = pf.get_estimate()
            history['est_x'].append(est_pose[0])
            history['est_y'].append(est_pose[1])
            history['dr_x'].append(dr.state[0])
            history['dr_y'].append(dr.state[1])

        show_localization_plot()
        
    ui.timer(0.1, control_loop)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title='Robot Dashboard', dark=True)