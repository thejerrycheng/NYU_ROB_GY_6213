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
KE_VALUE = 0.0001345210

class Particle:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta
        # No weight attribute needed for pure prediction

    def predict(self, v_physical, alpha_cmd, dt):
        """Pure prediction using calibrated physical parameters and noise"""
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
        self.theta = angle_wrap(self.theta - w_s * dt)

class ParticleFilter:
    def __init__(self, num_particles, initial_pose):
        self.num_particles = num_particles
        self.particles = []
        self.last_encoder_count = 0.0 
        
        # Initialize clustered around the start pose to watch uncertainty grow
        self.initialize_around_pose(initial_pose)

    def initialize_around_pose(self, pose):
        self.particles = []
        for _ in range(self.num_particles):
            # Add a tiny bit of initial noise so the particles aren't perfectly stacked
            x = pose[0] + 0.1
            y = pose[1] + 0.1
            theta = pose[2] + random.gauss(0, 0.01)
            self.particles.append(Particle(x, y, theta))

    def step(self, current_encoder, alpha_cmd, dt):
        """Only the prediction step. No observation, no weight update, no resampling."""
        de = current_encoder - self.last_encoder_count
        v_physical = (de * KE_VALUE) / dt if dt > 0 else 0.0
        self.last_encoder_count = current_encoder

        for p in self.particles: 
            p.predict(v_physical, alpha_cmd, dt)

    def get_estimate(self):
        sum_x = sum(p.x for p in self.particles)
        sum_y = sum(p.y for p in self.particles)
        sum_sin = sum(math.sin(p.theta) for p in self.particles)
        sum_cos = sum(math.cos(p.theta) for p in self.particles)
        return [sum_x / self.num_particles, sum_y / self.num_particles, math.atan2(sum_sin, sum_cos)]


def run_prediction_only():
    all_walls = np.array(parameters.wall_corner_list)
    x_coords = np.concatenate([all_walls[:, 0], all_walls[:, 2]])
    y_coords = np.concatenate([all_walls[:, 1], all_walls[:, 3]])
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_pad, y_pad = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
    
    # 1. Setup Environment and Filter
    start_pose = [0.2, 0.2, math.pi / 2]
    env = RobotEnv(initial_pose=start_pose, delta_t=0.1)
    
    # Pass start pose to PF so they begin together
    pf = ParticleFilter(num_particles=2000, initial_pose=start_pose) 
    
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
    
    simulated_encoder_count = 0.0
    
    for step in range(370):
        action = [0.0, 0.0]
        for end_step, v, alpha in control_sequence:
            if step < end_step:
                action = [v, alpha]; break 

        # Step the true environment
        true_pose, _ = env.step(action)
        v_cmd, alpha_cmd = action[0], action[1]
        
        # MOCK THE ENCODERS
        v_expected = (V_M * v_cmd) + V_C if v_cmd != 0.0 else 0.0
        ds = v_expected * env.dt
        simulated_encoder_count += (ds / KE_VALUE)

        # 2. STEP THE PF (PREDICTION ONLY - 3 arguments)
        pf.step(simulated_encoder_count, alpha_cmd, env.dt)
        est_pose = pf.get_estimate()

        # Update History
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
            ax.plot(history['est_x'], history['est_y'], 'b--', linewidth=1.5, alpha=0.6, label='Est Path (Dead Reckoning)', zorder=4)

            # Current State Robots
            ax.plot(true_pose[0], true_pose[1], 'bo', markersize=6, label='True Robot', zorder=5)
            ax.plot(est_pose[0], est_pose[1], 'mo', markersize=6, label='Estimate', zorder=6)
            ax.quiver(est_pose[0], est_pose[1], math.cos(est_pose[2]), math.sin(est_pose[2]), 
                      color='magenta', scale=15, width=0.007, zorder=7)

            ax.set_title(f"Pure Prediction (Dead Reckoning) | Step: {step}")
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
    ax_traj.plot(history['est_x'], history['est_y'], 'b--', linewidth=1.5, label='Dead Reckoning Est')
    
    ax_traj.set_title("Dead Reckoning Trajectory Drift", fontsize=9)
    ax_traj.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_traj.set_aspect('equal')
    ax_traj.legend(loc='lower left', fontsize=7)
    
    plt.savefig('dead_reckoning_trajectory.png', dpi=300, bbox_inches='tight')
    print("Trajectory plot saved: dead_reckoning_trajectory.png")

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
    plt.savefig('dead_reckoning_error.png', dpi=300, bbox_inches='tight')
    print("Error plot saved: dead_reckoning_error.png")
    plt.show()

if __name__ == '__main__':
    run_prediction_only()