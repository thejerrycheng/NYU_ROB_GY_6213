import math
import random
import numpy as np
import matplotlib.pyplot as plt
import parameters
from robot_env import RobotEnv, angle_wrap
from simulated_pf import ParticleFilter, Particle, MAX_RANGE

# Set global academic style
plt.rcParams.update({
    'font.size': 8, 
    'axes.labelsize': 8, 
    'legend.fontsize': 6,
    'axes.titlesize': 9
})

def run_experiment_trial(offset_m):
    gt_start = [0.1, 0.1, math.pi / 2]
    env = RobotEnv(initial_pose=list(gt_start), delta_t=0.1)
    
    # Initialize PF with the specific offset
    pf = ParticleFilter(num_particles=600)
    bias_pose = [gt_start[0] + offset_m, gt_start[1] + offset_m, gt_start[2]]
    
    # Manually cluster particles around the biased start for the experiment
    for p in pf.particles:
        p.x = random.gauss(bias_pose[0], 0.05)
        p.y = random.gauss(bias_pose[1], 0.05)
        p.theta = angle_wrap(random.gauss(bias_pose[2], 0.05))

    # Updated data dictionary to include 'std_t'
    data = {'steps': [], 'gt': [], 'est': [], 
            'err_x': [], 'err_y': [], 'err_t': [], 
            'std_x': [], 'std_y': [], 'std_t': [], 'total_err': []}
            
    control_sequence = [(35, 40.0, 0.0), (75, 20.0, 50.0), (120, 20.0, -50.0)]

    for step in range(120):
        action = [0, 0]
        for es, v, a in control_sequence:
            if step < es:
                action = [v, a]; break

        true_pose, sensor_data = env.step(action)
        pf.step(action, sensor_data, env.dt)
        
        # Calculate estimate and standard deviations
        est_pose = pf.get_estimate()
        # Calculate standard deviations for confidence bounds
        sx = np.std([p.x for p in pf.particles])
        sy = np.std([p.y for p in pf.particles])
        # Angular std (using sin/cos to handle wrapping safely)
        st = np.std([angle_wrap(p.theta - est_pose[2]) for p in pf.particles])
        
        data['steps'].append(step)
        data['gt'].append(list(true_pose))
        data['est'].append(list(est_pose))
        data['err_x'].append(true_pose[0] - est_pose[0])
        data['err_y'].append(true_pose[1] - est_pose[1])
        data['err_t'].append(angle_wrap(true_pose[2] - est_pose[2]))
        data['std_x'].append(sx)
        data['std_y'].append(sy)
        data['std_t'].append(st)
        data['total_err'].append(math.sqrt((true_pose[0]-est_pose[0])**2 + (true_pose[1]-est_pose[1])**2))

    return data

# --- Run All Trials ---
print("Running simulation trials...")
offsets = [0.0, 0.1, 0.2, 0.4, 0.8]
results = {f"{int(o*100)}cm": run_experiment_trial(o) for o in offsets}
colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))

# =========================================================
# 1. MERGED COMPARISON PLOT (7" x 4" Total)
# =========================================================
fig_merged, (ax_traj, ax_total) = plt.subplots(1, 2, figsize=(7, 4))

# Trajectory Overlay
for wall in parameters.wall_corner_list:
    ax_traj.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=1.5)
gt_path = np.array(results["0cm"]['gt'])
ax_traj.plot(gt_path[:,0], gt_path[:,1], 'g-', linewidth=2, label='GT', zorder=10)
for i, (name, data) in enumerate(results.items()):
    est = np.array(data['est'])
    ax_traj.plot(est[:,0], est[:,1], '--', color=colors[i], alpha=0.6, linewidth=0.8)
ax_traj.set_title("Trajectory Overlay")
ax_traj.set_aspect('equal')
ax_traj.set_xlim(-0.1, 1.3); ax_traj.set_ylim(-0.1, 1.9)

# Total Euclidean Error
for i, (name, data) in enumerate(results.items()):
    ax_total.plot(data['steps'], data['total_err'], color=colors[i], label=name, linewidth=1)
ax_total.set_title("Total Error Convergence")
ax_total.set_ylabel("Error (m)")
ax_total.set_xlabel("Step")
ax_total.grid(True, linestyle='--', alpha=0.4)
ax_total.legend(ncol=1, loc='upper right', fontsize='xx-small')

plt.tight_layout()
plt.savefig("fig_merged_overview.png", dpi=300)

# =========================================================
# 2. INDIVIDUAL TRIAL STACKS (3.5" Wide x 4" Total Height)
# =========================================================
for i, (name, data) in enumerate(results.items()):
    fig_stack, axs = plt.subplots(3, 1, figsize=(3.5, 4))
    t = data['steps']
    color = colors[i]
    
    # X-Error + Confidence
    axs[0].plot(t, data['err_x'], color=color, linewidth=1)
    axs[0].fill_between(t, 
                        np.array(data['err_x']) - 2*np.array(data['std_x']), 
                        np.array(data['err_x']) + 2*np.array(data['std_x']), 
                        color=color, alpha=0.2)
    axs[0].set_title(f"Trial: {name} | X Error", pad=2)
    axs[0].set_ylabel("Err (m)")
    axs[0].grid(True, alpha=0.3)
    
    # Y-Error + Confidence
    axs[1].plot(t, data['err_y'], color=color, linewidth=1)
    axs[1].fill_between(t, 
                        np.array(data['err_y']) - 2*np.array(data['std_y']), 
                        np.array(data['err_y']) + 2*np.array(data['std_y']), 
                        color=color, alpha=0.2)
    axs[1].set_title(f"Y Error", pad=2)
    axs[1].set_ylabel("Err (m)")
    axs[1].grid(True, alpha=0.3)
    
    # Theta-Error + Confidence
    err_t_deg = np.degrees(data['err_t'])
    std_t_deg = np.degrees(data['std_t'])
    axs[2].plot(t, err_t_deg, color=color, linewidth=1)
    axs[2].fill_between(t, 
                        err_t_deg - 2*std_t_deg, 
                        err_t_deg + 2*std_t_deg, 
                        color=color, alpha=0.2)
    axs[2].set_title(f"Theta Error", pad=2)
    axs[2].set_ylabel("Deg")
    axs[2].set_xlabel("Step")
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"report_stack_{name.replace(' ', '_')}.png", dpi=300)
    plt.close(fig_stack)

print("All plots generated successfully with Theta confidence bounds.")