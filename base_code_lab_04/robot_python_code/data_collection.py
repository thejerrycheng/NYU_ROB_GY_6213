import time
import os
import csv
import socket
import parameters
import robot_python_code

def main():
    # 1. Prompt user for commands
    print("=== Robot Data Collection ===")
    try:
        speed = float(input("Enter desired speed: "))
        steering = float(input("Enter desired steering: "))
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        return

    # 2. Setup directory and filename
    output_dir = "offline_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    # Format: {SPEED}_{Steering}_{DATE}_dataset.csv
    filename = f"{output_dir}/{speed}_{steering}_{date_str}_dataset.csv"

    # 3. Setup UDP communication
    print("\nInitializing UDP socket...")
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, parameters.localIP, 
        parameters.arduinoPort, parameters.localPort, parameters.bufferSize
    )
    if not success:
        print("Failed to bind UDP socket. Check your IP and network.")
        return

    # Apply a hardware-level timeout so the script doesn't freeze if the robot is off
    udp.UDPServerSocket.settimeout(0.5)

    msg_sender = robot_python_code.MsgSender(time.perf_counter(), 2, udp)
    msg_receiver = robot_python_code.MsgReceiver(time.perf_counter(), 3, udp)
    sensor_signal = robot_python_code.RobotSensorSignal([0, 0, 0])

    # 4. Connection Handshake (5-second timeout)
    print("Waiting for robot telemetry (5-second timeout)...")
    connected = False
    start_wait = time.perf_counter()
    
    while time.perf_counter() - start_wait < 5.0:
        try:
            # Send a zero-velocity "ping" to wake the robot
            msg_sender.send_control_signal([0.0, 0.0])
            # Attempt to read raw data directly from the socket
            raw_msg = udp.receive_msg()
            if raw_msg:
                print("Connection established successfully!")
                connected = True
                break
        except socket.timeout:
            # Expected behavior if robot hasn't responded yet
            pass 
            
    if not connected:
        print("Timeout: Failed to connect to the robot within 5 seconds. Exiting.")
        return

    # Lower the timeout for the active loop so a dropped packet doesn't lag the controller
    udp.UDPServerSocket.settimeout(0.05)

    # 5. Data collection loop
    print(f"\nStarting data collection for 5 seconds...")
    print(f"Command -> Speed: {speed}, Steering: {steering}")
    
    collected_data = []
    start_time = time.perf_counter()
    last_print_time = start_time

    try:
        while time.perf_counter() - start_time < 5.0:
            current_time = time.perf_counter()
            
            # Send the control command
            msg_sender.send_control_signal([speed, steering])
            
            # Safely receive sensor data
            try:
                sensor_signal = msg_receiver.receive_robot_sensor_signal(sensor_signal)
            except socket.timeout:
                # If a packet drops mid-run, just skip logging this exact cycle
                continue
            
            # Log data
            t_rel = current_time - start_time
            collected_data.append([
                round(t_rel, 4),
                sensor_signal.encoder_counts,
                sensor_signal.steering,
                str(sensor_signal.angles),    # Stringified list
                str(sensor_signal.distances)  # Stringified list
            ])

            # Print a 1-second countdown timer
            if current_time - last_print_time >= 1.0:
                time_left = 5.0 - (current_time - start_time)
                print(f"Time remaining: {time_left:.1f} seconds")
                last_print_time = current_time

            # Small sleep to regulate loop speed
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        
    finally:
        # 6. SAFETY: Stop the robot
        print("\nStopping the robot...")
        for _ in range(5): 
            try:
                msg_sender.send_control_signal([0.0, 0.0])
                time.sleep(0.05)
            except socket.timeout:
                pass
            
        # 7. Save Data to CSV
        if collected_data:
            print(f"Saving {len(collected_data)} rows of data to {filename}...")
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time_s", "Encoder_Counts", "Steering", "Lidar_Angles", "Lidar_Distances"])
                writer.writerows(collected_data)
            print("Data saved successfully!")
        else:
            print("No data was collected.")

if __name__ == "__main__":
    main()