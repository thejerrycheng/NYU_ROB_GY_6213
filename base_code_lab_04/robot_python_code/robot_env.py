import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Import your local parameters file 
import parameters

# --- Kinematic & Sensor Constants (Specific to this robot) ---
L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.00057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134
NUM_RAYS = 360
MAX_RANGE = 5.0


def angle_wrap(angle):
    """Wraps an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

class RobotEnv:
    """
    OpenAI Gym-style wrapper for the Robot Simulation.
    Handles the hidden 'true' state of the robot and generates sensor data.
    """
    def __init__(self, initial_pose=[0.3, 0.2, math.pi/2], delta_t=0.1):
        self.dt = delta_t
        
        # Internal Hidden State
        self.state = initial_pose
        self.history_x = [self.state[0]]
        self.history_y = [self.state[1]]
        
        # Latest Sensor Data
        self.latest_angles = []
        self.latest_distances = []
        
        # Rendering tools
        self.fig = None
        self.ax = None
        self.step_count = 0

    def reset(self, pose=None):
        """Resets the environment to an initial state."""
        if pose is not None:
            self.state = pose
        
        self.history_x = [self.state[0]]
        self.history_y = [self.state[1]]
        self.step_count = 0
        self.latest_angles = []
        self.latest_distances = []
        
        return self.state

    def step(self, action):
        """
        Takes an action [v_cmd, alpha_cmd], advances the environment by dt, 
        and returns the new state and observation.
        """
        v_cmd, alpha_cmd = action
        
        # 1. Update True Robot State (Motion Model)
        self.state = self._predict_next_pose(self.state, v_cmd, alpha_cmd, self.dt)
        self.history_x.append(self.state[0])
        self.history_y.append(self.state[1])
        
        # 2. Generate Sensor Observation (Measurement Model)
        self.latest_angles, self.latest_distances = self._simulate_lidar_scan(
            self.state[0], self.state[1], self.state[2]
        )
        
        self.step_count += 1
        
        # Return the 'true' pose and the 'sensor observation' (angles, distances)
        observation = (self.latest_angles, self.latest_distances)
        return self.state, observation

    def _predict_next_pose(self, current_pose, v_cmd, alpha_cmd, delta_t):
        """Internal stochastic motion model based on calibration parameters."""
        x, y, theta = current_pose
        
        if v_cmd == 0.0:
            v_physical = 0.0
        else:
            v_physical = (V_M * v_cmd) + V_C
            if v_physical < 0: v_physical = 0.0 

        delta_physical = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
        
        if v_physical > 0:
            v_physical += random.gauss(0, math.sqrt(VAR_V))
            delta_physical += random.gauss(0, math.sqrt(VAR_DELTA))
            
        w = (v_physical * math.tan(delta_physical)) / L if L > 0 else 0.0
            
        next_x = x + (v_physical * math.cos(theta) * delta_t)
        next_y = y + (v_physical * math.sin(theta) * delta_t)
        next_theta = angle_wrap(theta - (w * delta_t))
        
        return [next_x, next_y, next_theta]

    def _simulate_lidar_scan(self, robot_x, robot_y, robot_theta):
        """Internal stochastic measurement model using parameters.py"""
        # Pulls the noise variance directly from your parameters file
        sigma_z = math.sqrt(parameters.distance_variance)
        
        angles, distances = [], []
        ray_angles = np.linspace(0, 2 * math.pi, NUM_RAYS, endpoint=False)
        
        for relative_angle in ray_angles:
            global_ray_angle = robot_theta + relative_angle
            rx = math.cos(global_ray_angle)
            ry = math.sin(global_ray_angle)
            min_distance = MAX_RANGE
            
            # Pulls the map geometry directly from your parameters file
            for wall in parameters.wall_corner_list:
                qx, qy, bx, by = wall
                sx, sy = bx - qx, by - qy
                r_cross_s = rx * sy - ry * sx
                if abs(r_cross_s) > 1e-6: 
                    q_p_x, q_p_y = qx - robot_x, qy - robot_y  
                    t = (q_p_x * sy - q_p_y * sx) / r_cross_s  
                    u = (q_p_x * ry - q_p_y * rx) / r_cross_s  
                    if t > 0 and 0 <= u <= 1 and t < min_distance:
                        min_distance = t
                            
            if min_distance < MAX_RANGE:
                min_distance = max(0.0, min_distance + random.gauss(0, sigma_z))
                            
            angles.append(relative_angle)
            distances.append(min_distance)
            
        return angles, distances

    def render(self):
        """Draws the current state of the environment."""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 8))
            
        self.ax.clear()
        
        # Draw Map Walls
        for wall in parameters.wall_corner_list:
            self.ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
            
        # Draw Trajectory
        self.ax.plot(self.history_x, self.history_y, 'b--', linewidth=1.5, alpha=0.6, label='True Path')
        
        # Draw Lidar
        ray_lines_x, ray_lines_y, hit_points_x, hit_points_y = [], [], [], []
        for i in range(len(self.latest_angles)):
            if self.latest_distances[i] < (MAX_RANGE - 0.1): 
                global_angle = self.state[2] + self.latest_angles[i]
                hit_x = self.state[0] + self.latest_distances[i] * math.cos(global_angle)
                hit_y = self.state[1] + self.latest_distances[i] * math.sin(global_angle)
                ray_lines_x.extend([self.state[0], hit_x, None])
                ray_lines_y.extend([self.state[1], hit_y, None])
                hit_points_x.append(hit_x)
                hit_points_y.append(hit_y)
                
        self.ax.plot(ray_lines_x, ray_lines_y, color='lightblue', linewidth=0.5, zorder=1)
        self.ax.plot(hit_points_x, hit_points_y, 'r.', markersize=2, zorder=2)
            
        # Draw Robot
        self.ax.plot(self.state[0], self.state[1], 'go', markersize=8, zorder=3, label='Robot')
        arrow_len = 0.15
        self.ax.arrow(self.state[0], self.state[1], 
                 arrow_len * math.cos(self.state[2]), arrow_len * math.sin(self.state[2]), 
                 head_width=0.05, head_length=0.05, fc='green', ec='green', zorder=4)

        # Apply Plot Settings
        self.ax.set_title(f"Environment Render | Step: {self.step_count}")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        
        all_x = [w[0] for w in parameters.wall_corner_list] + [w[2] for w in parameters.wall_corner_list]
        all_y = [w[1] for w in parameters.wall_corner_list] + [w[3] for w in parameters.wall_corner_list]
        self.ax.set_xlim(min(all_x) - 0.2, max(all_x) + 0.2)
        self.ax.set_ylim(min(all_y) - 0.2, max(all_y) + 0.2)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.pause(0.01)


# ==========================================
# Example Usage
# ==========================================
if __name__ == '__main__':
    # Initialize Environment
    env = RobotEnv(initial_pose=[0.3, 0.2, math.pi / 2], delta_t=0.1)
    
    control_sequence = [
        (30,  40.0, 0.0),   
        (80,  40.0, 50.0),  
        (100, 40.0, -10.0)    
    ]
    
    # Run Simulation Loop
    for step in range(120):
        action = [0.0, 0.0]
        for end_step, v, alpha in control_sequence:
            if step < end_step:
                action = [v, alpha]
                break 

        # Step the environment
        true_pose, sensor_data = env.step(action)
        
        # Render the environment
        env.render()
        
    plt.ioff()
    plt.show()