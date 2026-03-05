import math
import numpy as np
import matplotlib.pyplot as plt
import parameters 

def simulate_lidar_scan(robot_x, robot_y, robot_theta):
    """
    Simulates Lidar measurements by raycasting from the robot's pose.
    Outputs ONLY what a real Lidar outputs:
        angles: list of relative laser angles (in radians)
        distances: list of distances to the walls (in meters)
    """
    walls = parameters.wall_corner_list
    num_rays = 360
    max_range = 5.0
    angles = []
    distances = []
    
    ray_angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    
    for relative_angle in ray_angles:
        global_ray_angle = robot_theta + relative_angle
        rx = math.cos(global_ray_angle)
        ry = math.sin(global_ray_angle)
        min_distance = max_range
        
        for wall in walls:
            qx, qy, bx, by = wall
            sx = bx - qx
            sy = by - qy
            r_cross_s = rx * sy - ry * sx
            if abs(r_cross_s) > 1e-6: 
                q_p_x = qx - robot_x
                q_p_y = qy - robot_y  
                t = (q_p_x * sy - q_p_y * sx) / r_cross_s  # distance along ray
                u = (q_p_x * ry - q_p_y * rx) / r_cross_s  # distance along wall segment
                if t > 0 and 0 <= u <= 1:
                    if t < min_distance:
                        min_distance = t
                        
        angles.append(relative_angle)
        distances.append(min_distance)
        
    return angles, distances


def visualize_prediction(robot_x, robot_y, robot_theta):
    """
    Plots the map, the robot, and reconstructs the Lidar rays 
    using only the raw angle and distance data.
    """
    # Get the simulated raw sensor data
    angles, distances = simulate_lidar_scan(robot_x, robot_y, robot_theta)
    
    plt.figure(figsize=(6, 8))
    
    # 1. Plot the Map Walls from parameters.py
    for wall in parameters.wall_corner_list:
        plt.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
        
    # 2. Reconstruct and Plot the Lidar Rays
    for i in range(len(angles)):
        # Reconstruct the global angle of the laser
        global_angle = robot_theta + angles[i]
        
        # Calculate the X and Y hit points using trigonometry
        hit_x = robot_x + distances[i] * math.cos(global_angle)
        hit_y = robot_y + distances[i] * math.sin(global_angle)
        
        # Draw the laser beam
        plt.plot([robot_x, hit_x], [robot_y, hit_y], color='lightblue', linewidth=0.5, zorder=1)
        # Draw the red dot where it hits
        plt.plot(hit_x, hit_y, 'r.', markersize=3, zorder=2) 
        
    # 3. Plot the Robot
    plt.plot(robot_x, robot_y, 'go', markersize=8, zorder=3, label='Robot')
    
    # Draw heading indicator (arrow)
    arrow_len = 0.15
    plt.arrow(robot_x, robot_y, arrow_len * math.cos(robot_theta), arrow_len * math.sin(robot_theta), 
              head_width=0.05, head_length=0.05, fc='green', ec='green', zorder=4)

    plt.title(f"Expected Lidar Scan at x={robot_x:.2f}, y={robot_y:.2f}, θ={math.degrees(robot_theta):.0f}°")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    
    # 4. Dynamically set plot limits based on the map boundaries
    all_x = [w[0] for w in parameters.wall_corner_list] + [w[2] for w in parameters.wall_corner_list]
    all_y = [w[1] for w in parameters.wall_corner_list] + [w[3] for w in parameters.wall_corner_list]
    plt.xlim(min(all_x) - 0.2, max(all_x) + 0.2)
    plt.ylim(min(all_y) - 0.2, max(all_y) + 0.2)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == '__main__':
    # Test Pose
    test_x = 0.4
    test_y = 0.5
    test_theta = math.pi / 4  
    
    visualize_prediction(test_x, test_y, test_theta)