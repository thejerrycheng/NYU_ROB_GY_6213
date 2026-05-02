import time
import math
import socket
import numpy as np
import matplotlib.pyplot as plt

# Your local hardware libraries
import robot_python_code
import parameters

def main():
    print("Initializing UDP Connection to Robot...")
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, parameters.localIP,
        parameters.arduinoPort, parameters.localPort, parameters.bufferSize
    )
    
    if not success:
        print("Failed to bind UDP socket. Check IP addresses and ports in parameters.py.")
        return

    udp.UDPServerSocket.settimeout(0.05)  # Lower timeout helps keep the GUI responsive
    sender = robot_python_code.MsgSender(time.perf_counter(), 2, udp)
    receiver = robot_python_code.MsgReceiver(time.perf_counter(), 3, udp)
    
    # Initialize a dummy signal to start receiving
    sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])

    print("Connected! Opening visualization window...")

    # --- Setup Interactive Matplotlib Polar Plot ---
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')
        
    ax.set_rmax(5000)  # Max range in mm (5 meters)
    ax.set_rticks([1000, 2000, 3000, 4000, 5000])  
    ax.set_rlabel_position(-22.5)  
    ax.grid(True, color='#1e293b', linestyle='--')
    ax.set_title("Real-Time LiDAR Diagnostics", color='white', pad=20)

    # Initialize an empty scatter plot
    scatter = ax.scatter([], [], s=10, c='#00ff00', alpha=0.7, edgecolors='none')

    # Force the window to show up before entering the while loop
    plt.show(block=False)
    plt.pause(0.1)

    # Buffers to accumulate a full 360-degree sweep
    sweep_angles = []
    sweep_distances = []
    last_angle = None
    packets_received = 0
    
    print("Waiting for LiDAR data... Press Ctrl+C in the terminal to exit.")

    try:
        while True:
            # Send a zero-velocity command to keep the connection alive/robot stopped
            sender.send_control_signal([0, 0])
            
            try:
                # Poll the UDP socket
                sensor_signal = receiver.receive_robot_sensor_signal(sensor_signal)
                packets_received += 1
                
                # Process incoming data chunks
                for ang, dist in zip(sensor_signal.angles, sensor_signal.distances):
                    
                    # Detect sweep wrap-around (e.g., jumping from 359 degrees back to 0)
                    if last_angle is not None and abs(ang - last_angle) > 180:
                        # Sweep complete! Update the plot.
                        if len(sweep_angles) > 0:
                            # Convert angles to radians for the polar plot
                            rads = [math.radians(a) for a in sweep_angles]
                            
                            # Stack into Nx2 array for scatter.set_offsets
                            offsets = np.c_[rads, sweep_distances]
                            scatter.set_offsets(offsets)
                            
                            # Use plt.pause instead of canvas.draw to keep the GUI thread alive
                            plt.pause(0.001)
                        
                        # Clear buffers for the next sweep
                        sweep_angles = []
                        sweep_distances = []
                        print(f"Sweep rendered. Total UDP packets received: {packets_received}", end='\r')
                    
                    # Filter out dead zones or error readings (e.g., exactly 0 or > 5000)
                    if 100 < dist < 5000:
                        sweep_angles.append(ang)
                        sweep_distances.append(dist)
                        
                    last_angle = ang
                    
            except socket.timeout:
                # We still need to yield to the GUI even if no packet arrived
                plt.pause(0.001)
                
            except Exception as e:
                print(f"\nData parsing error: {e}")
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        # Clean up
        sender.send_control_signal([0, 0])
        plt.ioff()
        plt.show()  # Keep the final frame open until manually closed
        print("Hardware connection closed.")

if __name__ == '__main__':
    main()