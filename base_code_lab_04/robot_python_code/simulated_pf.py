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

class Particle:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0

    def predict(self, v_cmd, alpha_cmd, dt):
        v_phys = (V_M * v_cmd) + V_C if v_cmd != 0.0 else 0.0
        if v_phys < 0: v_phys = 0.0
        d_phys = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]

        if v_phys > 0:
            v_phys += random.gauss(0, math.sqrt(VAR_V) * 2.0)
            d_phys += random.gauss(0, math.sqrt(VAR_DELTA) * 2.0)

        w = (v_phys * math.tan(d_phys)) / L if L > 0 else 0.0
        self.x += v_phys * math.cos(self.theta) * dt
        self.y += v_phys * math.sin(self.theta) * dt
        self.theta = angle_wrap(self.theta - w * dt)

    def update_weight(self, angles, distances):
        log_weight = 0.0
        ray_step = 20 # Faster processing
        
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
            # Use 10.0 scale to keep the filter "open" during global search
            log_weight += -(error**2) / (2 * parameters.distance_variance * 10.0)

        self.weight = math.exp(log_weight) + 1e-12

class ParticleFilter:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.particles = []
        self.global_initialization()

    def global_initialization(self):
        """Spawns particles ONLY in valid floor space (Rejection Sampling)."""
        self.particles = []
        while len(self.particles) < self.num_particles:
            # 1. Spawn within the outer 1.2 x 1.8 bounding box
            x = random.uniform(0.0, 1.2)
            y = random.uniform(0.0, 1.8)
            
            # 2. REJECTION LOGIC: Check if it's inside the 0.6x0.6 top-right obstacle
            if x > 0.6 and y > 1.2:
                continue # Try again
                
            theta = random.uniform(-math.pi, math.pi)
            self.particles.append(Particle(x, y, theta))

    def step(self, action, observation, dt):
        v_cmd, alpha_cmd = action
        angles, distances = observation
        for p in self.particles: p.predict(v_cmd, alpha_cmd, dt)
        for p in self.particles: p.update_weight(angles, distances)
        self.resample()

    def resample(self):
        weights = [p.weight for p in self.particles]
        max_w = max(weights)
        new_particles = []
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        for _ in range(self.num_particles):
            beta += random.uniform(0, 2.0 * max_w)
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % self.num_particles
            chosen = self.particles[index]
            new_particles.append(Particle(chosen.x, chosen.y, chosen.theta))
        self.particles = new_particles

    def get_estimate(self):
        sum_x = sum(p.x for p in self.particles)
        sum_y = sum(p.y for p in self.particles)
        sum_sin = sum(math.sin(p.theta) for p in self.particles)
        sum_cos = sum(math.cos(p.theta) for p in self.particles)
        return [sum_x / self.num_particles, sum_y / self.num_particles, math.atan2(sum_sin, sum_cos)]

def run_pf_simulation():
    env = RobotEnv(initial_pose=[0.3, 0.2, math.pi / 2], delta_t=0.1)
    pf = ParticleFilter(num_particles=500)
    
    control_sequence = [
        (35,  40.0, 0.0),    
        (75,  25.0, 50.0),   
        (115, 20.0, -10.0),    
        (130, 0.0,  0.0)     
    ]
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))
    
    for step in range(130):
        action = [0.0, 0.0]
        for end_step, v, alpha in control_sequence:
            if step < end_step:
                action = [v, alpha]
                break 

        # 1. Step Environment & Filter
        true_pose, sensor_data = env.step(action)
        angles, distances = sensor_data
        pf.step(action, sensor_data, env.dt)
        est_pose = pf.get_estimate() # [x, y, theta]

        ax.clear()
        
        # --- Draw Map ---
        for wall in parameters.wall_corner_list:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
        
        # --- Draw Particles (Red Cloud) ---
        px = [p.x for p in pf.particles]
        py = [p.y for p in pf.particles]
        ax.scatter(px, py, s=2, c='red', alpha=0.2, zorder=2)

        # --- DEBUG: Overlay Lidar on Predicted Pose ---
        # We reconstruct where the Lidar hits WOULD be if the robot was at est_pose
        hit_x_list, hit_y_list = [], []
        for i in range(0, len(angles), 10): # Subsample for speed
            if distances[i] < (MAX_RANGE - 0.1):
                # Calculate global angle relative to the ESTIMATED heading
                global_angle = est_pose[2] + angles[i]
                
                # Project hits from the ESTIMATED position
                hit_x = est_pose[0] + distances[i] * math.cos(global_angle)
                hit_y = est_pose[1] + distances[i] * math.sin(global_angle)
                
                hit_x_list.append(hit_x)
                hit_y_list.append(hit_y)
        
        # Plot the Lidar "Ghost" (What the filter sees)
        ax.scatter(hit_x_list, hit_y_list, s=5, c='magenta', marker='x', alpha=0.6, label='Est. Lidar')

        # --- Draw Robot Comparison ---
        # True Robot (Green)
        ax.plot(true_pose[0], true_pose[1], 'go', markersize=8, zorder=5, label='True Robot')
        
        # PF Estimate (Blue)
        ax.plot(est_pose[0], est_pose[1], 'bo', markersize=7, zorder=6, label='PF Estimate')
        ax.arrow(est_pose[0], est_pose[1], 
                 0.15 * math.cos(est_pose[2]), 0.15 * math.sin(est_pose[2]), 
                 head_width=0.05, head_length=0.05, fc='blue', ec='blue', zorder=6)

        ax.set_title(f"PF Debug View | Step: {step}")
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.9)
        ax.set_aspect('equal')
        ax.legend(loc='lower left', fontsize='small')
        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_pf_simulation()