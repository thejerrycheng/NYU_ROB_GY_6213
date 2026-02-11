import time
import sys
import parameters       # Your existing parameters file
import robot_python_code  # The library code you provided

# ==========================================
# 1. TRAJECTORY DEFINITION (Tuned Version)
# ==========================================
# Configuration used for calculation:
# Speed: 75
# Straight Offset: -3.88 deg
# Left Scale: 1.5x

trajectory_cmds = [
    # Format: (velocity, steering_angle, duration_seconds)
    # 1. Backwards 4 grids
    (-75, -1, 5),  # Back 4 Grids
    
    # 2. Forward 1 grid
    (75, 5, 1),  # Fwd 1 Grid
    
    # 3. Turn Right 90 (R)
    (75, 20.00, 3.66),  # Right 90 (R)
    
    # 4. Turn Left 90 (R) - Scaled 1.5x
    (75, -20.00, 3.8),  # Left 90 (R)
    
    # 5. Forward 1 grid
    (75, 5, 1),  # Fwd 1 Grid
    
    # 6. Backwards 4 grids
    (-75, 5, 7),  # Back 4 Grids
    
    # 7. Turn Right 90 (R)
    (75, 20.00, 3.66),  # Right 90 (R)
    
    # 8. Turn Left 90 (R) - Scaled 1.5x
    (75, -20.00, 3.86),  # Left 90 (R)
    
    # 9. Backwards 2 grids
    (-75, 5, 3.3),  # Back 2 Grids
    
    # 10. Up 3 grids
    (75, 5, 6),  # Up 3 Grids
    
    # Stop
    (0, -1, 0.5)     # Stop
]

# ==========================================
# 2. DEPLOYMENT SCRIPT
# ==========================================
def main():
    print("Initializing Robot...")
    
    # 1. Create Robot Instance
    robot = robot_python_code.Robot()
    
    # 2. Establish UDP Connection
    # Uses IPs and Ports from your parameters.py
    udp, success = robot_python_code.create_udp_communication(
        parameters.arduinoIP, 
        parameters.localIP, 
        parameters.arduinoPort, 
        parameters.localPort, 
        parameters.bufferSize
    )
    
    if not success:
        print("Error: Could not establish UDP connection.")
        return

    robot.setup_udp_connection(udp)
    
    # 3. Execution Loop
    print(f"Starting Trajectory execution ({len(trajectory_cmds)} segments)...")
    print("Make sure the robot battery is connected!")
    time.sleep(1) # Brief pause before start
    
    try:
        total_start_time = time.perf_counter()
        
        for i, (v_cmd, alpha_cmd, duration) in enumerate(trajectory_cmds):
            print(f"Seg {i+1}: Vel={v_cmd}, Steer={alpha_cmd:.2f}, Dur={duration:.2f}s")
            
            segment_start_time = time.perf_counter()
            
            # Keep sending the command for 'duration' seconds
            while (time.perf_counter() - segment_start_time) < duration:
                # The control_loop function handles sending UDP packets
                # We pass logging_switch_on=True to record the run
                robot.control_loop(cmd_speed=v_cmd, 
                                   cmd_steering_angle=alpha_cmd, 
                                   logging_switch_on=True)
                
                # Sleep briefly to prevent CPU hogging, but fast enough for smooth UDP
                # MsgSender has a built-in throttle (delta_send_time), 
                # so calling this frequently is safe.
                time.sleep(0.02) 
                
    except KeyboardInterrupt:
        print("\nEmergency Stop requested!")
    
    finally:
        # 4. Shutdown Sequence
        print("Stopping Robot...")
        # Send 0 command a few times to ensure it stops
        for _ in range(10):
            robot.control_loop(0, 0, logging_switch_on=False)
            time.sleep(0.05)
            
        robot.eliminate_udp_connection()
        print("Done.")

if __name__ == "__main__":
    main()