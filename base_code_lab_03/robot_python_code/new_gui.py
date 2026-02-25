# External libraries
import asyncio
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import time, strftime

# Local libraries
from robot import Robot
import robot_python_code
import parameters

# Global variables
logging = False
stream_video = True

# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

# Create the connection with a real camera.
def connect_with_camera():
    # UPDATED: Use the external camera defined in parameters.py
    video_capture = cv2.VideoCapture(parameters.camera_id)
    return video_capture

def update_video(video_image):
    if stream_video and video_image is not None:
        video_image.force_reload()

def get_time_in_ms():
    return int(time()*1000)

# --- THEME COLORS ---
CARD_BG = 'bg-slate-900'
ACCENT_COLOR = 'blue-500' 
TEXT_COLOR = 'text-slate-200'
HEADER_BG = 'bg-slate-950'

# Create the gui page
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

    # Robot variables
    robot = Robot()
    
    # Store latest frame globally for UI streaming and recording
    latest_frame = None
    
    # Manage video and measurement recording state entirely inside the GUI
    app_state = {
        'video_writer': None,
        'csv_file': None
    }

    # Lidar data setup
    max_lidar_range = 12
    lidar_angle_res = 2
    num_angles = int(360 / lidar_angle_res)
    lidar_distance_list = []
    lidar_cos_angle_list = []
    lidar_sin_angle_list = []
    for i in range(num_angles):
        lidar_distance_list.append(max_lidar_range)
        lidar_cos_angle_list.append(math.cos(i*lidar_angle_res/180*math.pi))
        lidar_sin_angle_list.append(math.sin(i*lidar_angle_res/180*math.pi))

    # Set up the video stream
    if stream_video:
        # UPDATED: Use the external camera defined in parameters.py
        video_capture = cv2.VideoCapture(parameters.camera_id)
    
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if latest_frame is None:
            return Response(content=b'', media_type='image/jpeg')
        jpeg = await run.cpu_bound(convert, latest_frame)
        return Response(content=jpeg, media_type='image/jpeg')

    # Logic Functions
    def update_lidar_data():
        for i in range(robot.robot_sensor_signal.num_lidar_rays):
            distance_in_mm = robot.robot_sensor_signal.distances[i]
            angle = 360-robot.robot_sensor_signal.angles[i]
            if distance_in_mm > 20 and abs(angle) < 360:
                index = max(0,min(int(360/lidar_angle_res-1),int((angle-(lidar_angle_res/2))/lidar_angle_res)))
                lidar_distance_list[index] = distance_in_mm/1000
               
    def update_commands():
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            trial_timer_label.set_text(f'{delta_time/1000:.1f}s')
            
            if delta_time > parameters.trial_time:
                robot.running_trial = False
                speed_switch.value = False
                steering_switch.value = False
                robot.extra_logging = True
                ui.notify('Trial Ended', type='positive')

        if robot.extra_logging:
            delta_time = get_time_in_ms() - robot.trial_start_time
            if delta_time > parameters.trial_time + parameters.extra_trial_log_time:
                logging_switch.value = False
                robot.extra_logging = False

        if speed_switch.value:
            cmd_speed = slider_speed.value
        else:
            cmd_speed = 0
        if steering_switch.value:
            cmd_steering_angle = slider_steering.value
        else:
            cmd_steering_angle = 0
        return cmd_speed, cmd_steering_angle
        
    def update_connection_to_robot():
        if udp_switch.value:
            if not robot.connected_to_hardware:
                udp, udp_success = robot_python_code.create_udp_communication(parameters.arduinoIP, parameters.localIP, parameters.arduinoPort, parameters.localPort, parameters.bufferSize)
                if udp_success:
                    robot.setup_udp_connection(udp)
                    robot.connected_to_hardware = True
                    status_indicator.classes('bg-green-500', remove='bg-red-500')
                    status_label.set_text('Connected')
                else:
                    udp_switch.value = False
                    robot.connected_to_hardware = False
                    ui.notify('Connection Failed', type='negative')
        else:
            if robot.connected_to_hardware:
                robot.eliminate_udp_connection()
                robot.connected_to_hardware = False
                status_indicator.classes('bg-red-500', remove='bg-green-500')
                status_label.set_text('Disconnected')
    
    def enable_speed():
        pass
        
    def enable_steering():
        pass

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
            
            # --- PLOT 1: EKF ESTIMATE (RED) ---
            sigma = 3
            covar_matrix = parameters.covariance_plot_scale * robot.extended_kalman_filter.state_covariance[0:2,0:2]
            x_est = robot.extended_kalman_filter.state_mean[0]
            y_est = robot.extended_kalman_filter.state_mean[1]
            lambda_, v = np.linalg.eig(covar_matrix)
            lambda_ = np.sqrt(lambda_)
            
            ell = Ellipse(xy=(x_est, y_est), alpha=0.5, facecolor='#ef4444', width=lambda_[0], height=lambda_[1], angle=np.rad2deg(np.arctan2(*v[:,0][::-1])))
            ax.add_artist(ell)
            plt.plot(x_est, y_est, 'ro', markersize=5, label='EKF Filtered')

            # --- PLOT 2: RAW CAMERA MEASUREMENT (BLUE) ---
            z_x = robot.camera_sensor_signal[0]
            z_y = robot.camera_sensor_signal[1]
            
            # Only plot if we have a non-zero measurement
            if z_x != 0.0 or z_y != 0.0:
                plt.plot(z_x, z_y, 'bx', markersize=6, label='Camera (Z_t)')

            # Add Legend
            plt.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#94a3b8', fontsize=8)

            plt.grid(True, color='#1e293b', linestyle='--')
            
            # Center the plot around the current position for better tracking
            plot_range = 1.0
            plt.xlim(x_est - plot_range, x_est + plot_range)
            plt.ylim(y_est - plot_range, y_est + plot_range)

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        ui.notify('Trial Started', type='info')
        
        # --- GUI RECORDING START (VIDEO & CSV) ---
        cmd_s = slider_speed.value
        cmd_st = slider_steering.value
        base_filename = parameters.filename_start + f"_{cmd_s}_{cmd_st}_" + strftime("%d_%m_%y_%H_%M_%S")
        
        # Open CSV logger for Measurements
        csv_filename = base_filename + "_measurements.csv"
        app_state['csv_file'] = open(csv_filename, 'w')
        app_state['csv_file'].write("timestamp_ms,z_x,z_y,z_theta\n")
        print(f"[+] GUI Started Recording Measurements: {csv_filename}")

        # Open Video Writer
        if stream_video and video_capture.isOpened():
            video_filename = base_filename + ".mp4"
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            app_state['video_writer'] = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height))
            print(f"[+] GUI Started Recording Video: {video_filename}")

    # --- UI LAYOUT ---
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
                    logging_switch = ui.switch('Data Logging').props('color=blue keep-color').classes('text-slate-300 w-full')
                    udp_switch = ui.switch('Hardware Connection').props('color=green keep-color').classes('text-slate-300 w-full')
                    
                    with ui.row().classes('w-full items-center justify-between mt-2'):
                        run_trial_button = ui.button('START TRIAL', on_click=lambda:run_trial()).props('unelevated').classes('bg-blue-600 hover:bg-blue-500 text-white w-2/3 rounded-lg')
                        trial_timer_label = ui.label('0.0s').classes('text-slate-400 font-mono text-sm')

        with ui.card().classes(f'w-full {CARD_BG} p-6'):
            ui.label('Drive Control').classes('text-sm font-bold text-slate-400 mb-6 uppercase tracking-wider')
            
            with ui.grid(columns=2).classes('w-full gap-12'):
                
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('speed', color='blue-400')
                            ui.label('Speed').classes('text-lg font-medium text-white')
                        speed_switch = ui.switch(on_change=lambda: enable_speed()).props('color=blue dense')
                    
                    slider_speed = ui.slider(min=-100, max=100, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                    ui.label('Throttle %').classes('text-xs text-slate-500 mt-1 self-end')

                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('directions_car', color='blue-400')
                            ui.label('Steering').classes('text-lg font-medium text-white')
                        steering_switch = ui.switch(on_change=lambda: enable_steering()).props('color=blue dense')
                    
                    slider_steering = ui.slider(min=-20, max=20, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                    
                    with ui.row().classes('w-full justify-between text-xs text-slate-500 mt-1'):
                        ui.label('Left')
                        ui.label('Center')
                        ui.label('Right')

    async def control_loop():
        nonlocal latest_frame
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        
        if stream_video and video_capture.isOpened():
            ret, latest_frame = await run.io_bound(video_capture.read)

        # --- GUI MEASUREMENT & VIDEO RECORDING TICK ---
        if logging_switch.value:
            # Write Video Frame
            if app_state['video_writer'] is not None and latest_frame is not None:
                app_state['video_writer'].write(latest_frame)
            
            # Write CSV Measurement
            if app_state['csv_file'] is not None:
                z_x = robot.camera_sensor_signal[0]
                z_y = robot.camera_sensor_signal[1]
                z_th = robot.camera_sensor_signal[5]
                app_state['csv_file'].write(f"{get_time_in_ms()},{z_x},{z_y},{z_th}\n")
            
        # --- GUI RECORDING STOP ---
        if not logging_switch.value:
            if app_state['video_writer'] is not None:
                app_state['video_writer'].release()
                app_state['video_writer'] = None
                print("[-] GUI Stopped Recording Video")
            
            if app_state['csv_file'] is not None:
                app_state['csv_file'].close()
                app_state['csv_file'] = None
                print("[-] GUI Stopped Recording Measurements CSV")

        # Original robot call (UNMODIFIED)
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        
        encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
        show_localization_plot() 
        
        if stream_video and latest_frame is not None:
            update_video(video_image)
        
    ui.timer(0.1, control_loop)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title='Robot Dashboard', dark=True)