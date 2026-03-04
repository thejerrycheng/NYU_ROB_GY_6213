import math
import matplotlib.pyplot as plt
import numpy as np
import time

# Local libraries
import parameters
import robot_python_code

def main():
    # 1. Initialize Connection
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, parameters.localIP, 
        parameters.arduinoPort, parameters.localPort, parameters.bufferSize
    )
    if not success: return

    receiver = robot_python_code.MsgReceiver(time.perf_counter(), parameters.num_robot_sensors, udp)
    sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])

    # 2. Setup Plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot elements
    scatter = ax.scatter([], [], s=10, c='blue', label='Lidar Returns')
    ax.plot(0, 0, 'r^', markersize=10, label='Robot (Base Link)') # Lidar position
    
    ax.set(xlim=(-5, 5), ylim=(-5, 5), title="Lidar Map (360° Fixed Buffer)")
    ax.grid(True)
    ax.legend(loc='upper right')

    # 3. Initialize 360-degree buffer with infinity
    scan_buffer = np.full(360, np.inf)
    last_angle = 0.0

    print("Running... (Ctrl+C to stop)")

    try:
        while True:
            sensor_signal = receiver.receive_robot_sensor_signal(sensor_signal)
            
            for i in range(sensor_signal.num_lidar_rays):
                raw_dist = sensor_signal.distances[i]
                raw_angle = sensor_signal.angles[i]
                
                if raw_dist > 20: 
                    # Detect 360-degree wrap-around (New Revolution)
                    if abs(raw_angle - last_angle) > 180:
                        acc_x, acc_y = [], []
                        
                        # Process the entire 360 buffer
                        for deg in range(360):
                            dist_mm = scan_buffer[deg]
                            if not math.isinf(dist_mm):
                                # Convert using your class functions
                                theta = sensor_signal.convert_hardware_angle(deg)
                                dist_m = sensor_signal.convert_hardware_distance(dist_mm)
                                
                                acc_x.append(dist_m * math.cos(theta))
                                acc_y.append(dist_m * math.sin(theta))
                        
                        # Update plot
                        if acc_x:
                            scatter.set_offsets(list(zip(acc_x, acc_y)))
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            
                        # Reset buffer to infinity for the next sweep
                        scan_buffer.fill(np.inf)

                    # Map angle to an integer index (0-359) and update buffer
                    idx = int(round(raw_angle)) % 360
                    scan_buffer[idx] = raw_dist
                
                last_angle = raw_angle
                
            time.sleep(0.005)

    except KeyboardInterrupt:
        plt.ioff()
        plt.close()

if __name__ == '__main__':
    main()