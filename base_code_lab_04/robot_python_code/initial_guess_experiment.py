import math
import random
import numpy as np
import matplotlib.pyplot as plt
import parameters
from robot_env import RobotEnv, angle_wrap
from simulated_pf import ParticleFilter # Ensure simulated_pf.py is in the same folder

def run_experiment_trial(offset_m):
    """Runs a full simulation and returns detailed time-series data."""
    gt_start = [0.3, 0.2, math.pi / 2]
    env = RobotEnv(initial_pose=list(gt_start), delta_t=0.1)
    
    # Initialize PF with the specific offset
    pf = ParticleFilter(num_particles=600)
    bias_pose = [gt_start[0] + offset_m, gt_start[1] + offset_m, gt_start[2]]
    
    for p in pf.particles:
        p.x = random.gauss(bias_pose[0], 0.05)
        p.y = random.gauss(bias_pose[1], 0.05)
        p.theta = angle_wrap(random.gauss(bias_pose[2], 0.05))

    data = {'gt': [], 'est': [], 'err_x': [], 'err_y': [], 'err_t': [], 'std_x': [], 'std_y': []}
    control_sequence = [(35, 40.0, 0.0), (75, 25.0, 50.0), (120, 20.0, -10.0)]

    for step in range(120):
        action = [0, 0]
        for es, v, a in control_sequence:
            if step < es:
                action = [v, a]; break

        true_pose, sensor_data = env.step(action)
        pf.step(action, sensor_data, env.dt)
        est_pose = pf.get_estimate()
        
        # Calculate State-Specific Errors
        data['gt'].append(list(true_pose))
        data['est'].append(list(est_pose))
        data['err_x'].append(true_pose[0] - est_pose[0])
        data['err_y'].append(true_pose[1] - est_pose[1])
        data['err_t'].append(angle_wrap(true_pose[2] - est_pose[2]))
        
        # Calculate Confidence (Standard Deviation of Particles)
        data['std_x'].append(np.std([p.x for p in pf.particles]))
        data['std_y'].append(np.std([p.y for p in pf.particles]))

    return data

# --- Run All Trials ---
offsets = [0.0, 0.1, 0.2, 0.4, 0.8]
results = {f"{int(o*100)}cm": run_experiment_trial(o) for o in offsets}

# ==========================================
# VISUALIZATION SUITE
# ==========================================
plt.style.use('seaborn-v0_8-muted')
fig = plt.figure(figsize=(16, 12))

# 1. Trajectory Overlay Map (Top Left)
ax_map = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
for wall in parameters.wall_corner_list:
    ax_map.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=3)
    
for name, d in results.items():
    est = np.array(d['est'])
    ax_map.plot(est[:,0], est[:,1], '--', label=f'Est ({name})', alpha=0.8)
gt = np.array(results["0cm"]['gt'])
ax_map.plot(gt[:,0], gt[:,1], 'g-', linewidth=4, label='Ground Truth', zorder=10)
ax_map.set_title("Trajectory Recovery Comparison")
ax_map.legend(fontsize='small', loc='lower right')
ax_map.set_aspect('equal')

# 2. X-Error with Confidence Bounds (Top Right)
# We plot the 80cm case as it's the most challenging
ax_x = plt.subplot2grid((3, 2), (0, 1))
d80 = results["80cm"]
t = np.arange(120)
ax_x.plot(t, d80['err_x'], 'r', label='X Error (80cm)')
ax_x.fill_between(t, np.array(d80['err_x']) - 2*np.array(d80['std_x']), 
                  np.array(d80['err_x']) + 2*np.array(d80['std_x']), color='r', alpha=0.2, label='2σ Bound')
ax_x.set_title("X-Axis Error & Confidence (80cm Case)")
ax_x.grid(True)

# 3. Y-Error with Confidence Bounds (Middle Right)
ax_y = plt.subplot2grid((3, 2), (1, 1))
ax_y.plot(t, d80['err_y'], 'b', label='Y Error (80cm)')
ax_y.fill_between(t, np.array(d80['err_y']) - 2*np.array(d80['std_y']), 
                  np.array(d80['err_y']) + 2*np.array(d80['std_y']), color='b', alpha=0.2)
ax_y.set_title("Y-Axis Error & Confidence")
ax_y.grid(True)

# 4. Theta Error (Bottom Right)
ax_t = plt.subplot2grid((3, 2), (2, 1))
ax_t.plot(t, np.degrees(d80['err_t']), 'g', label='Heading Error (deg)')
ax_t.set_title("Heading Error (Degrees)")
ax_t.set_ylabel("Degrees")
ax_t.grid(True)

# 5. Global Error Convergence (Bottom Left)
ax_all = plt.subplot2grid((3, 2), (2, 0))
for name, d in results.items():
    total_err = np.sqrt(np.array(d['err_x'])**2 + np.array(d['err_y'])**2)
    ax_all.plot(total_err, label=name)
ax_all.set_title("Total Euclidean Error Convergence")
ax_all.set_ylabel("Meters")
ax_all.set_xlabel("Time Steps")
ax_all.legend(ncol=2, fontsize='x-small')

plt.tight_layout()
plt.show()