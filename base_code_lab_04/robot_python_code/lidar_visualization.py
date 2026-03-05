import math
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

# Local libraries
import parameters
import robot_python_code

# --- Configuration ---
FRONT_ANGLE_DEG = 0      # Angle to treat as "front" for characterization
TARGET_DISTANCES_M = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
CSV_FILENAME = "lidar_characterization_filtered.csv"
FILTER_ALPHA = 0.2       # EMA Smoothing factor (0.0 to 1.0). Lower = smoother but slower to update.
# ---------------------

# Global flag to signal when the user presses Enter in the plot
record_trigger = False

def on_key(event):
    """Matplotlib event handler for key presses."""
    global record_trigger
    if event.key == 'enter':
        record_trigger = True

def setup_plot():
    """Initializes the matplotlib plot with all required elements."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    lidar_scatter = ax.scatter([], [], s=5, c='blue', alpha=0.5, label='Raw Lidar Returns')
    front_point, = ax.plot([], [], 'go', markersize=10, markeredgecolor='black', label='Filtered Front Point')
    ax.plot(0, 0, 'r^', markersize=12, label='Robot Base Link')
    ax.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc='r', ec='r', label='Heading')
    
    info_text = ax.text(0.02, 0.98, 'Initializing...', transform=ax.transAxes, 
                        fontsize=11, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set(xlim=(-3, 3), ylim=(-3, 3))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    
    ax.set_title(f"Live Calibration (CLICK THIS WINDOW & PRESS ENTER)")
    ax.set_xlabel("X Distance (meters)")
    ax.set_ylabel("Y Distance (meters)")
    
    return fig, ax, lidar_scatter, front_point, info_text

def update_plot(fig, lidar_scatter, front_point, info_text, scan_buffer, filtered_front_m, status_msg, sensor_signal):
    """Updates the plot with new data."""
    acc_x, acc_y = [], []
    
    for deg in range(360):
        dist_mm = scan_buffer[deg]
        if not math.isinf(dist_mm) and dist_mm > 0:
            theta = sensor_signal.convert_hardware_angle(deg)
            dist_m = sensor_signal.convert_hardware_distance(dist_mm)
            acc_x.append(dist_m * math.cos(theta))
            acc_y.append(dist_m * math.sin(theta))
            
    lidar_scatter.set_offsets(np.c_[acc_x, acc_y] if acc_x else np.empty((0, 2)))

    if filtered_front_m is not None:
        front_theta = sensor_signal.convert_hardware_angle(FRONT_ANGLE_DEG)
        fx = filtered_front_m * math.cos(front_theta)
        fy = filtered_front_m * math.sin(front_theta)
        front_point.set_data([fx], [fy])
    else:
        front_point.set_data([], [])

    info_text.set_text(status_msg)
    fig.canvas.draw()
    fig.canvas.flush_events()

def main():
    global record_trigger
    
    # 1. Initialize UDP Connection
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, parameters.localIP, 
        parameters.arduinoPort, parameters.localPort, parameters.bufferSize
    )
    if not success: 
        print("Failed to initialize UDP. Exiting.")
        return

    receiver = robot_python_code.MsgReceiver(time.perf_counter(), parameters.num_robot_sensors, udp)
    sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])

    # 2. Setup Visualization
    fig, ax, lidar_scatter, front_point, info_text = setup_plot()
    
    all_results = []
    print("=========================================")
    print(" Live Lidar Calibration Started          ")
    print("=========================================\n")

    try:
        for gt_dist in TARGET_DISTANCES_M:
            record_trigger = False 
            filtered_front_m = None # Reset filter for each new target distance
            
            print(f"\n>>> Move robot to exactly {gt_dist:.2f}m. Use plot to align.")
            print(">>> IMPORTANT: Click on the plot window, then press [ENTER] to snap measurement...")
            
            scan_buffer = np.full(360, np.inf)
            last_angle = 0.0
            reading_saved = False

            # Flush stale UDP data
            for _ in range(10): receiver.receive_robot_sensor_signal(sensor_signal)

            while not reading_saved:
                sensor_signal = receiver.receive_robot_sensor_signal(sensor_signal)
                
                for i in range(sensor_signal.num_lidar_rays):
                    raw_dist = sensor_signal.distances[i]
                    raw_angle = sensor_signal.angles[i]
                    
                    if raw_dist > 20: 
                        if abs(raw_angle - last_angle) > 180:
                            # Full sweep complete
                            front_dist_mm = scan_buffer[FRONT_ANGLE_DEG]
                            is_valid_reading = not math.isinf(front_dist_mm) and front_dist_mm > 0
                            
                            # Update the Low-Pass Filter if we have a valid raw reading
                            if is_valid_reading:
                                raw_front_m = sensor_signal.convert_hardware_distance(front_dist_mm)
                                if filtered_front_m is None:
                                    filtered_front_m = raw_front_m
                                else:
                                    filtered_front_m = (FILTER_ALPHA * raw_front_m) + ((1 - FILTER_ALPHA) * filtered_front_m)

                            # Handle the Record Trigger
                            if record_trigger:
                                if is_valid_reading and filtered_front_m is not None:
                                    bias = filtered_front_m - gt_dist
                                    print(f"Recorded! Filtered Measured: {filtered_front_m:.4f}m | Bias: {bias:.4f}m")
                                    all_results.append([gt_dist, filtered_front_m, bias])
                                    
                                    update_plot(fig, lidar_scatter, front_point, info_text, scan_buffer, filtered_front_m, f"SAVED {filtered_front_m:.3f}m!", sensor_signal)
                                    plt.pause(1.0) 
                                    
                                    reading_saved = True
                                    break 
                                else:
                                    # Don't reset record_trigger. Just wait for the next valid sweep.
                                    status = f"Target: {gt_dist:.2f}m\n[ENTER PRESSED: WAITING FOR VALID DATA...]"
                                    update_plot(fig, lidar_scatter, front_point, info_text, scan_buffer, filtered_front_m, status, sensor_signal)
                            else:
                                # Standard live update
                                if filtered_front_m is not None:
                                    status = f"Target: {gt_dist:.2f}m | CLICK PLOT + PRESS ENTER\nFiltered Front: {filtered_front_m:.3f}m"
                                else:
                                    status = f"Target: {gt_dist:.2f}m | CLICK PLOT + PRESS ENTER\nFiltered Front: --"
                                update_plot(fig, lidar_scatter, front_point, info_text, scan_buffer, filtered_front_m, status, sensor_signal)

                            scan_buffer.fill(np.inf)

                        idx = int(round(raw_angle)) % 360
                        scan_buffer[idx] = raw_dist
                    last_angle = raw_angle
                
                time.sleep(0.002)
            
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
        
    finally:
        plt.ioff()
        plt.close()
        
        if all_results:
            print(f"\nWriting data to {CSV_FILENAME}...")
            with open(CSV_FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Ground_Truth_m", "Filtered_Measured_Distance_m", "Bias_m"])
                writer.writerows(all_results)
            print("Data saved successfully.")

if __name__ == '__main__':
    main()