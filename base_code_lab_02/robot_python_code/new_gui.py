# External libraries
import asyncio
import cv2
import math
from matplotlib import pyplot as plt
import matplotlib
from nicegui import ui, app, run
import numpy as np
import time
from fastapi import Response
from time import time

# Local libraries
import robot_python_code
import parameters

# Global variables
logging = False
stream_video = False

# Frame converter for the video stream, from OpenCV to a JPEG image
def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()

# Create the connection with a real camera.
def connect_with_camera():
    video_capture = cv2.VideoCapture(1)
    return video_capture

def update_video(video_image):
    if stream_video:
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
    # Set dark mode for gui and apply global style tweaks
    dark = ui.dark_mode()
    dark.value = True
    
    # Custom CSS for that "Apple" rounded feel
    ui.add_head_html('''
        <style>
            .nicegui-content { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
            .q-card { border-radius: 16px; border: 1px solid #334155; }
            .q-slider__track-container { height: 6px; border-radius: 3px; }
        </style>
    ''')

    # Robot variables
    robot = robot_python_code.Robot()

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
        video_capture = cv2.VideoCapture(1)
    
    @app.get('/video/frame')
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return placeholder
        jpeg = await run.cpu_bound(convert, frame)
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
        # Trial controls
        if robot.running_trial:
            delta_time = get_time_in_ms() - robot.trial_start_time
            # Update the trial timer label
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

        # Slider controls
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

    def show_lidar_plot():
        with main_plot:
            fig = main_plot.fig
            # Make the plot background transparent to blend with the card
            fig.patch.set_facecolor('#0f172a') # Matches slate-900
            plt.clf()
            
            # Setup Axes
            ax = plt.gca()
            ax.set_facecolor('#0f172a')
            
            # Custom styling for axes
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_color('#334155')
            ax.spines['right'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(axis='x', colors='#94a3b8')
            ax.tick_params(axis='y', colors='#94a3b8')
                
            for i in range(num_angles):
                distance = lidar_distance_list[i]
                cos_ang = lidar_cos_angle_list[i]
                sin_ang = lidar_sin_angle_list[i]
                x = [distance * cos_ang, max_lidar_range * cos_ang]
                y = [distance * sin_ang, max_lidar_range * sin_ang]
                plt.plot(x, y, color='#3b82f6', alpha=0.6, linewidth=1) # Blue lidar lines
            
            # Robot center point
            plt.plot(0, 0, 'ro', markersize=5) 
            
            plt.grid(True, color='#1e293b', linestyle='--')
            plt.xlim(-2,2)
            plt.ylim(-2,2)

    def run_trial():
        robot.trial_start_time = get_time_in_ms()
        robot.running_trial = True
        steering_switch.value = True
        speed_switch.value = True
        logging_switch.value = True
        ui.notify('Trial Started', type='info')

    # --- UI LAYOUT ---
    
    # 1. Header
    with ui.header().classes(f'{HEADER_BG} shadow-md p-4 flex items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('smart_toy', size='32px', color='blue-400')
            ui.label('Robot Command Center').classes('text-xl font-bold tracking-wide text-white')
        
        # Status Badge in Header
        with ui.row().classes('items-center gap-2 bg-slate-800 px-3 py-1 rounded-full'):
            status_indicator = ui.element('div').classes('w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]')
            status_label = ui.label('Disconnected').classes('text-xs font-semibold text-slate-300')

    # 2. Main Content Area
    with ui.column().classes('w-full p-6 gap-6 items-center max-w-7xl mx-auto'):
        
        # Row 1: Visualizations (Video + Lidar + Stats)
        with ui.grid(columns=3).classes('w-full gap-6'):
            
            # Card 1: Camera Feed
            with ui.card().classes(f'w-full {CARD_BG} p-0 overflow-hidden relative group'):
                ui.label('Camera Feed').classes('absolute top-3 left-4 z-10 text-xs font-bold text-white/70 bg-black/50 px-2 py-1 rounded backdrop-blur-sm')
                if stream_video:
                    video_image = ui.interactive_image('/video/frame').classes('w-full h-64 object-cover')
                else:
                    # Fallback if no video
                    with ui.column().classes('w-full h-64 items-center justify-center bg-slate-800'):
                        ui.icon('videocam_off', size='48px', color='slate-600')
                        ui.label('No Video Stream').classes('text-slate-500 text-sm mt-2')
                    video_image = None
            
            # Card 2: Lidar Plot
            with ui.card().classes(f'w-full {CARD_BG} items-center justify-center p-2'):
                 # Using negative margin to fit plot better
                main_plot = ui.pyplot(figsize=(3.5, 3.5), close=False) 
            
            # Card 3: Telemetry & Connection
            with ui.card().classes(f'w-full {CARD_BG} p-5 flex flex-col justify-between h-full'):
                ui.label('Telemetry').classes('text-sm font-bold text-slate-400 mb-4 uppercase tracking-wider')
                
                # Encoder Stat
                with ui.row().classes('items-baseline justify-between w-full mb-2'):
                    ui.label('Encoder Count').classes(TEXT_COLOR)
                    encoder_count_label = ui.label('0').classes('text-2xl font-mono text-blue-400')
                
                ui.separator().classes('bg-slate-700 my-4')
                
                # System Controls
                with ui.column().classes('w-full gap-3'):
                    logging_switch = ui.switch('Data Logging').props('color=blue keep-color').classes('text-slate-300 w-full')
                    udp_switch = ui.switch('Hardware Connection').props('color=green keep-color').classes('text-slate-300 w-full')
                    
                    with ui.row().classes('w-full items-center justify-between mt-2'):
                        run_trial_button = ui.button('START TRIAL', on_click=lambda:run_trial()).props('unelevated').classes('bg-blue-600 hover:bg-blue-500 text-white w-2/3 rounded-lg')
                        trial_timer_label = ui.label('0.0s').classes('text-slate-400 font-mono text-sm')

        # Row 2: Drive Controls
        with ui.card().classes(f'w-full {CARD_BG} p-6'):
            ui.label('Drive Control').classes('text-sm font-bold text-slate-400 mb-6 uppercase tracking-wider')
            
            with ui.grid(columns=2).classes('w-full gap-12'):
                
                # Speed Control Section
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('speed', color='blue-400')
                            ui.label('Speed').classes('text-lg font-medium text-white')
                        speed_switch = ui.switch(on_change=lambda: enable_speed()).props('color=blue dense')
                    
                    # Custom styled slider
                    slider_speed = ui.slider(min=0, max=100, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                    ui.label('Throttle %').classes('text-xs text-slate-500 mt-1 self-end')

                # Steering Control Section
                with ui.column().classes('w-full gap-2'):
                    with ui.row().classes('w-full justify-between items-center'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('directions_car', color='blue-400')
                            ui.label('Steering').classes('text-lg font-medium text-white')
                        steering_switch = ui.switch(on_change=lambda: enable_steering()).props('color=blue dense')
                    
                    # Custom styled slider
                    slider_steering = ui.slider(min=-20, max=20, value=0).props('label-always color=blue track-size=6px thumb-size=20px').classes('mt-4')
                    
                    with ui.row().classes('w-full justify-between text-xs text-slate-500 mt-1'):
                        ui.label('Left')
                        ui.label('Center')
                        ui.label('Right')

    # Loop setup
    async def control_loop():
        update_connection_to_robot()
        cmd_speed, cmd_steering_angle = update_commands()
        robot.control_loop(cmd_speed, cmd_steering_angle, logging_switch.value)
        encoder_count_label.set_text(robot.robot_sensor_signal.encoder_counts)
        update_lidar_data()
        show_lidar_plot()
        #update_video(video_image)
        
    ui.timer(0.1, control_loop)

# Run the gui with a dark theme specific title
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(native=True, title='Robot Dashboard', dark=True)