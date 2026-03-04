import math
import random
import numpy as np

# --- 1. Calibrated Constants ---
L = 0.145  # Wheelbase: 14.5 cm
KE_VALUE = 0.0001345210  # Encoder resolution (Ke)

# Velocity Model: v_actual = m*v_cmd + c
V_M = 0.004808
V_C = -0.045557
VAR_V = 0.00057829  # Velocity Variance (sigma_v^2)

# Steering Model: delta = a*alpha^2 + b*alpha + c
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134  # Steering Variance (sigma_delta^2)

def distance_travelled_s(encoder_counts):
    """Maps encoder counts to distance using Ke."""
    return encoder_counts * KE_VALUE

def get_stochastic_inputs(v_cmd_derived, alpha):
    """Calculates stochastic v and delta based on identified models and noise."""
    # Note: v_cmd_derived is already in m/s from the encoders, 
    # so we apply noise directly, NOT the (V_M*u + C) map.
    v_stochastic = v_cmd_derived + random.gauss(0, math.sqrt(VAR_V))
    
    # Calculate Delta from Command Alpha
    delta_det = DELTA_COEFFS[0]*(alpha**2) + DELTA_COEFFS[1]*alpha + DELTA_COEFFS[2]
    delta_stochastic = delta_det + random.gauss(0, math.sqrt(VAR_DELTA))
    
    return v_stochastic, delta_stochastic

class MyMotionModel:
    def __init__(self, initial_state, last_encoder_count):
        # state is [x, y, theta]
        self.state = np.array(initial_state, dtype=float)
        self.last_encoder_count = last_encoder_count

    def step_update(self, encoder_counts, steering_angle_command, delta_t):
        """Updates robot state using stochastic Ackermann kinematics."""
        de = encoder_counts - self.last_encoder_count
        ds = distance_travelled_s(de)
        
        # 1. Determine physical velocity from encoder displacement
        v_physical = (ds / delta_t) if delta_t > 0 else 0
        
        # 2. Get noisy physical parameters
        v_s, d_s = get_stochastic_inputs(v_physical, steering_angle_command)
        
        # 3. Calculate angular velocity: w = (v * tan(delta)) / L
        # Note: We use absolute L to avoid division errors, direction is handled below
        if abs(L) > 0:
            w_s = (v_s * math.tan(d_s)) / L
        else:
            w_s = 0
        
        # 4. Euler Integration
        # Update X and Y
        self.state[0] += delta_t * v_s * math.cos(self.state[2])
        self.state[1] += delta_t * v_s * math.sin(self.state[2])
        
        # Update Theta
        # CRITICAL FIX: Positive Steering (alpha > 0) -> Positive Delta -> Right Turn
        # Right Turn = Clockwise = DECREASING Theta
        self.state[2] -= delta_t * w_s 
        
        self.last_encoder_count = encoder_counts
        return self.state

    def traj_propagation(self, time_list, encoder_count_list, steering_angle_list):
        """Propagates state through a list of logged measurements."""
        x_list = [self.state[0]]
        y_list = [self.state[1]]
        theta_list = [self.state[2]]
        
        self.last_encoder_count = encoder_count_list[0]
        
        for i in range(1, len(encoder_count_list)):
            delta_t = time_list[i] - time_list[i-1]
            if delta_t <= 0: continue
            
            self.step_update(encoder_count_list[i], steering_angle_list[i], delta_t)
            
            x_list.append(self.state[0])
            y_list.append(self.state[1])
            theta_list.append(self.state[2])

        return x_list, y_list, theta_list

    def generate_simulated_traj(self, duration, v_cmd=50.0, alpha_cmd=10.0):
        """Generates a predicted trajectory for report visualization."""
        delta_t = 0.05
        t_list, x_list, y_list, theta_list = [], [], [], []
        t = 0
        
        # Determine the expected physical velocity for this command
        # v_actual = m*v_cmd + c
        v_expected = V_M * v_cmd + V_C
        
        # Reset encoder count for simulation
        self.last_encoder_count = 0
        current_enc = 0
        
        while t < duration:
            # Simulate encoder accumulation
            ds = v_expected * delta_t
            current_enc += ds / KE_VALUE
            
            self.step_update(current_enc, alpha_cmd, delta_t)
            
            t_list.append(t)
            x_list.append(self.state[0])
            y_list.append(self.state[1])
            theta_list.append(self.state[2])
            t += delta_t 
            
        return t_list, x_list, y_list, theta_list