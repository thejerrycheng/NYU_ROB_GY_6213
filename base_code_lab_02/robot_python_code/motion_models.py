# External Libraries
import math
import random
import numpy as np

# Motion Model constants
L = 0.055  # Wheelbase in meters (Replace with your actual measurement)
ENCODER_RESOLUTION = 0.00012  # meters per count (Replace with your f_se(e) slope)

# A function for obtaining variance in distance travelled as a function of distance travelled
def variance_distance_travelled_s(distance):
    # Based on Step 4: sigma_s^2 = f_ss(e). 
    # Often modeled as a linear growth of variance over distance: sigma^2 = k * s
    k_s = 0.005 # Student needs to fit this from data
    var_s = k_s * abs(distance)
    return var_s

# Function to calculate distance from encoder counts
def distance_travelled_s(encoder_counts):
    # Based on Step 4: s = f_se(e)
    s = encoder_counts * ENCODER_RESOLUTION
    return s

# A function for obtaining variance in rotational velocity w
def variance_rotational_velocity_w(steering_angle_command):
    # Based on Step 5: sigma_w^2 = f_sw(alpha). 
    # Often constant for a specific servo/speed.
    var_w = 0.01 # Student needs to fit this from data
    return var_w

def rotational_velocity_w(steering_angle_command, velocity):
    # Based on MATLAB code: theta_dot = (v * tan(delta)) / L
    # steering_angle_command (delta) should be in radians here.
    if abs(L) < 1e-5: return 0
    w = (velocity * math.tan(steering_angle_command)) / L
    return w

# This class implements the motion model x_t = f(x_{t-1}, u_t)
class MyMotionModel:

    def __init__(self, initial_state, last_encoder_count):
        # state is [x, y, theta]
        self.state = np.array(initial_state, dtype=float)
        self.last_encoder_count = last_encoder_count

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        # 1. Calculate change in encoder counts and distance
        de = encoder_counts - self.last_encoder_count
        ds_det = distance_travelled_s(de)
        
        # 2. Add zero-mean Gaussian noise to distance
        # sigma = sqrt(variance)
        s_noise = random.gauss(0, math.sqrt(variance_distance_travelled_s(ds_det)))
        s_stochastic = ds_det + s_noise
        
        # 3. Calculate velocity and rotational velocity
        vel = s_stochastic / delta_t if delta_t > 0 else 0
        w_det = rotational_velocity_w(steering_angle_command, vel)
        
        # 4. Add noise to rotational velocity
        w_noise = random.gauss(0, math.sqrt(variance_rotational_velocity_w(steering_angle_command)))
        w_stochastic = w_det + w_noise
        
        # 5. Euler Integration update (Linear Matrix form logic)
        # x = x + dt * v * cos(theta)
        # y = y + dt * v * sin(theta)
        # theta = theta + dt * w
        self.state[0] += delta_t * vel * math.cos(self.state[2])
        self.state[1] += delta_t * vel * math.sin(self.state[2])
        self.state[2] += delta_t * w_stochastic
        
        self.last_encoder_count = encoder_counts
        return self.state
    
    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        x_list, y_list, theta_list = [self.state[0]], [self.state[1]], [self.state[2]]
        self.last_encoder_count = encoder_count_list[0]
        
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            if delta_t <= 0: continue
            
            new_state = self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            x_list.append(new_state[0])
            y_list.append(new_state[1])
            theta_list.append(new_state[2])

        return x_list, y_list, theta_list

    def generate_simulated_traj(self, duration):
        # Useful for Step 6: "Plot motion model predicted trajectories"
        delta_t = 0.01
        t_list, x_list, y_list, theta_list = [], [], [], []
        t = 0
        
        # Example: Constant velocity and steering
        v_sim = 0.5  # m/s
        delta_sim = math.radians(7) # 7 degrees as per your MATLAB script
        
        while t < duration:
            # Simulated encoder increment based on velocity
            # s = v * t -> encoder = s / res
            sim_enc = (v_sim * t) / ENCODER_RESOLUTION
            
            state = self.step_update(sim_enc, delta_sim, delta_t)
            
            t_list.append(t)
            x_list.append(state[0])
            y_list.append(state[1])
            theta_list.append(state[2])
            t += delta_t 
            
        return t_list, x_list, y_list, theta_list