import math
import random
import numpy as np
import matplotlib.pyplot as plt
import parameters
from robot_env import RobotEnv, angle_wrap

# --- Kinematic Constants ---
L = 0.145
V_M = 0.004808 
V_C = -0.045557 
VAR_V = 0.00057829 
DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
VAR_DELTA = 0.00023134
MAX_RANGE = 5.0
KE_VALUE = 0.0001345210

class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta, self.weight = x, y, theta, 1.0

    def predict(self, v_physical, alpha_cmd, dt):
        """Uses physical velocity derived from encoders, just like MyMotionModel"""
        # 1. Apply stochastic noise to physical velocity
        v_s = v_physical + random.gauss(0, math.sqrt(VAR_V))
        
        # 2. Calculate deterministic steering delta, then add noise
        delta_det = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
        d_s = delta_det + random.gauss(0, math.sqrt(VAR_DELTA))

        # 3. Calculate angular velocity
        w_s = (v_s * math.tan(d_s)) / L if L > 0 else 0.0
        
        # 4. Euler Integration
        self.x += v_s * math.cos(self.theta) * dt
        self.y += v_s * math.sin(self.theta) * dt
        self.theta = angle_wrap(self.theta - w_s * dt) # Subtraction for correct Ackermann turning

    def update_weight(self, angles, distances):
        log_weight = 0.0
        ray_step = 1 
        for i in range(0, len(angles), ray_step):
            if distances[i] >= MAX_RANGE - 0.1: continue
            global_angle = self.theta + angles[i]
            rx, ry = math.cos(global_angle), math.sin(global_angle)
            min_dist = MAX_RANGE
            for wall in parameters.wall_corner_list:
                qx, qy, bx, by = wall
                sx, sy = bx - qx, by - qy
                denom = rx * sy - ry * sx
                if abs(denom) > 1e-6:
                    t = ((qx - self.x) * sy - (qy - self.y) * sx) / denom
                    u = ((qx - self.x) * ry - (qy - self.y) * rx) / denom
                    if 0 <= u <= 1 and 0 < t < min_dist:
                        min_dist = t
            error = min_dist - distances[i]
            log_weight += -(error**2) / (2 * parameters.distance_variance * 10.0)
        self.weight = math.exp(log_weight) + 1e-12

class ParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.particles = []
        self.last_encoder_count = 0.0 # Track encoders for Odometry
        self.global_initialization()

    def global_initialization(self):
        """Spawns particles ONLY in valid floor space (Rejection Sampling)."""
        self.particles = []
        while len(self.particles) < self.num_particles:
            x, y = 4, 0.1; 
            # if x > 0.6 and y > 1.2: continue 
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append(Particle(x, y, theta))

    def step(self, current_encoder, alpha_cmd, observation, dt):
        """Calculates v_physical from encoders before predicting"""
        de = current_encoder - self.last_encoder_count
        v_physical = (de * KE_VALUE) / dt if dt > 0 else 0.0
        self.last_encoder_count = current_encoder

        angles, distances = observation
        for p in self.particles: 
            p.predict(v_physical, alpha_cmd, dt)
        for p in self.particles: 
            p.update_weight(angles, distances)
        self.resample()

    def resample(self):
        weights = [p.weight for p in self.particles]
        total_w = sum(weights)
        max_w = max(weights)
        
        # --- Convergence Check ---
        is_lost = max_w < 1e-15 
        recovery_rate = 0.15 if is_lost else 0.02 
        
        new_particles = []
        
        # 1. Standard Resampling Wheel for most of the population
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        
        num_to_resample = int(self.num_particles * (1.0 - recovery_rate))
        
        for _ in range(num_to_resample):
            beta += random.uniform(0, 2.0 * max_w)
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % self.num_particles
            
            p = self.particles[index]
            new_particles.append(Particle(p.x, p.y, p.theta))
            
        # 2. Scholastic Recovery: Global Injection
        while len(new_particles) < self.num_particles:
            new_p = self.global_random_sample() 
            new_particles.append(new_p)
            
        self.particles = new_particles

    def global_random_sample(self):
        """Helper to spawn a particle anywhere in the valid maze floor."""
        while True:
            x = random.uniform(0.0, 6.0) 
            y = random.uniform(0.0, 6.0)
            
            is_valid = True
            if (2.7 < x < 3.3) and (2.7 < y < 3.3): is_valid = False 
            
            if is_valid:
                return Particle(x, y, random.uniform(-math.pi, math.pi))

    def get_estimate(self):
        sum_x = sum(p.x for p in self.particles)
        sum_y = sum(p.y for p in self.particles)
        sum_sin = sum(math.sin(p.theta) for p in self.particles)
        sum_cos = sum(math.cos(p.theta) for p in self.particles)
        return [sum_x / self.num_particles, sum_y / self.num_particles, math.atan2(sum_sin, sum_cos)]
    

def run_pf_simulation():
    # 1. Automatic Bound Calculation for Dynamic Scaling
    all_walls = np.array(parameters.wall_corner_list)
    x_coords = np.concatenate([all_walls[:, 0], all_walls[:, 2]])
    y_coords = np.concatenate([all_walls[:, 1], all_walls[:, 3]])
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    
    # 2. Setup Environment and Filter
    env = RobotEnv(initial_pose=[0.2, 0.2, math.pi / 2], delta_t=0.1)
    pf = ParticleFilter(num_particles=2000) 
    MAX_RANGE = 5.0 
    
    # Initialize history dictionary for final plots
    history = {
        'steps': [], 'true_x': [], 'true_y': [], 'true_theta': [],
        'est_x': [], 'est_y': [], 'est_theta': [],
        'err_x': [], 'err_y': [], 'err_theta': []
    }
    
    control_sequence = [
        (35, 40.0, 0.0), (75, 50.0, 50.0), (115, 50.0, -50.0),    
        (150, 50.0, 0.0), (200, 50.0, -30.0), (250, 50.0, -40.0), 
        (300, 50.0, 0.0), (370, 50.0, 40.0)                          
    ]
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # --- ADDED: Track simulated encoder counts ---
    simulated_encoder_count = 0.0
    
    for step in range(370):
        action = [0.0, 0.0]
        for end_step, v, alpha in control_sequence:
            if step < end_step:
                action = [v, alpha]; break 
                
        v_cmd = action[0]
        alpha_cmd = action[1]
        true_pose, sensor_data = env.step(action)
        angles, distances = sensor_data
        v_expected = (V_M * v_cmd) + V_C if v_cmd != 0.0 else 0.0
        ds = v_expected * env.dt
        simulated_encoder_count += (ds / KE_VALUE)
        pf.step(simulated_encoder_count, alpha_cmd, sensor_data, env.dt)
        est_pose = pf.get_estimate()

        # Update History for saving later
        history['steps'].append(step)
        history['true_x'].append(true_pose[0])
        history['true_y'].append(true_pose[1])
        history['true_theta'].append(true_pose[2])
        history['est_x'].append(est_pose[0])
        history['est_y'].append(est_pose[1])
        history['est_theta'].append(est_pose[2])
        history['err_x'].append(true_pose[0] - est_pose[0])
        history['err_y'].append(true_pose[1] - est_pose[1])
        history['err_theta'].append(angle_wrap(true_pose[2] - est_pose[2]))

        if step % 2 == 0:
            ax.clear()
            # Draw Map
            for wall in parameters.wall_corner_list:
                ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2, zorder=1)
            
            # Particles: GREEN DOTS
            px, py = [p.x for p in pf.particles], [p.y for p in pf.particles]
            ax.scatter(px, py, s=2, c='green', alpha=0.15, zorder=2, label='Particles')
            
            # Trajectories
            ax.plot(history['true_x'], history['true_y'], 'g-', linewidth=1.5, alpha=0.6, label='True Path (GT)', zorder=3)
            ax.plot(history['est_x'], history['est_y'], 'b-', linewidth=1.5, alpha=0.6, label='Est Path (PF)', zorder=4)

            # Lidar: RED LASER LINES
            for i in range(len(angles)): 
                if distances[i] < (MAX_RANGE - 0.1):
                    ga = est_pose[2] + angles[i]
                    hx = est_pose[0] + distances[i] * math.cos(ga)
                    hy = est_pose[1] + distances[i] * math.sin(ga)
                    ax.plot([est_pose[0], hx], [est_pose[1], hy], color='red', alpha=0.3, linewidth=0.5, zorder=5)

            # Current State Robots
            ax.plot(true_pose[0], true_pose[1], 'bo', markersize=6, label='True Robot', zorder=6)
            ax.plot(est_pose[0], est_pose[1], 'mo', markersize=6, label='PF Estimate', zorder=7)
            ax.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), 
                      color='magenta', scale=15, width=0.007, zorder=8)

            ax.set_title(f"PF Path Analysis | Step: {step}")
            ax.set_xlim(x_min - x_pad, x_max + x_pad); ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize='x-small', ncol=2)
            plt.pause(0.001)

    plt.ioff()
    plt.close(fig) 

    # ==========================================
    # 3. SAVE FINAL TRAJECTORY (3.5" x 3.5")
    # ==========================================
    fig_traj, ax_traj = plt.subplots(figsize=(3.5, 3.5))
    for wall in parameters.wall_corner_list:
        ax_traj.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=2)
    
    ax_traj.plot(history['true_x'], history['true_y'], 'g-', linewidth=1.5, label='Ground Truth')
    ax_traj.plot(history['est_x'], history['est_y'], 'b--', linewidth=1.5, label='PF Estimate')
    
    ax_traj.set_title("Final Trajectory Comparison", fontsize=9)
    ax_traj.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_traj.set_aspect('equal')
    ax_traj.legend(loc='lower left', fontsize=7)
    
    plt.savefig('final_trajectory_50.png', dpi=300, bbox_inches='tight')
    print("Trajectory plot saved: final_trajectory_50.png")

    # ==========================================
    # 4. SAVE ERROR PLOTS (x, y, theta)
    # ==========================================
    fig_err, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 6))
    
    ax1.plot(history['steps'], history['err_x'], 'r', label='Error X')
    ax1.set_ylabel('X Error (m)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['steps'], history['err_y'], 'b', label='Error Y')
    ax2.set_ylabel('Y Error (m)')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(history['steps'], np.degrees(history['err_theta']), 'g', label='Error Theta')
    ax3.set_ylabel('Theta Error (deg)')
    ax3.set_xlabel('Step')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis_50.png', dpi=300, bbox_inches='tight')
    print("Error plot saved: error_analysis_50.png")
    plt.show()

if __name__ == '__main__':
    run_pf_simulation()