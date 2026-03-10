# External libraries
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import os
import cv2

# Local libraries
import parameters
import robot_python_code 

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi: angle -= 2*math.pi
    while angle < -math.pi: angle += 2*math.pi
    return angle

class State:
    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta

    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    def deepcopy(self):
        return copy.deepcopy(self)

class Wall:
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.length = self.corner1.distance_to(self.corner2)

class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = [Wall(w) for w in wall_corner_list]
            
        min_x = min(min(w.corner1.x, w.corner2.x) for w in self.wall_list)
        max_x = max(max(w.corner1.x, w.corner2.x) for w in self.wall_list)
        min_y = min(min(w.corner1.y, w.corner2.y) for w in self.wall_list)
        max_y = max(max(w.corner1.y, w.corner2.y) for w in self.wall_list)
        
        border = 0.5
        self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        self.particle_range = [min_x, max_x, min_y, max_y]

class Particle:
    def __init__(self):
        self.state = State(0, 0, 0)
        self.weight = 1.0
        
    def randomize_uniformly(self, xy_range):
        self.state = State(random.uniform(xy_range[0], xy_range[1]), 
                           random.uniform(xy_range[2], xy_range[3]), 
                           random.uniform(-math.pi, math.pi))
        self.weight = 1.0

    def randomize_around_initial_state(self, initial_state, state_stdev):
        self.state = State(random.gauss(initial_state.x, state_stdev.x),
                           random.gauss(initial_state.y, state_stdev.y),
                           angle_wrap(random.gauss(initial_state.theta, state_stdev.theta)))
        self.weight = 1.0
        
    def propagate_state(self, v_cmd, alpha_cmd, delta_t):
        """EXACT logic from your original prediction script"""
        L = 0.145
        V_M = 0.004808 
        V_C = -0.045557 
        VAR_V = 0.00057829 
        DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
        VAR_DELTA = 0.00023134

        # 1. Map Commands
        v_physical = (V_M * v_cmd) + V_C
        if v_physical < 0 and v_cmd > 0:
            v_physical = 0.0 

        delta_physical = DELTA_COEFFS[0]*(alpha_cmd**2) + DELTA_COEFFS[1]*alpha_cmd + DELTA_COEFFS[2]
        
        # 2. Add Noise
        if v_physical > 0:
            v_physical += random.gauss(0, math.sqrt(VAR_V))
            delta_physical += random.gauss(0, math.sqrt(VAR_DELTA))
            
        # 3. Kinematics
        w = (v_physical * math.tan(delta_physical)) / L if L > 0 else 0.0
            
        # 4. Integration
        self.state.x += (v_physical * math.cos(self.state.theta) * delta_t)
        self.state.y += (v_physical * math.sin(self.state.theta) * delta_t)
        self.state.theta = angle_wrap(self.state.theta - (w * delta_t))
        
    def calculate_weight(self, lidar_signal, map_obj):
        """EXACT cross-product line intersection using ALL 360 RAYS"""
        log_weight = 0.0
        max_range = 5.0
        
        # --- CHANGED: Process every single ray in the Lidar array ---
        ray_step = 1  
        
        for i in range(0, lidar_signal.num_lidar_rays, ray_step):
            raw_dist = lidar_signal.distances[i]
            if raw_dist < 4900: # Assuming 5000 is error/max
                measured_dist = lidar_signal.convert_hardware_distance(raw_dist)
                relative_angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
                
                # Reconstruct global ray angle
                global_ray_angle = self.state.theta + relative_angle
                rx = math.cos(global_ray_angle)
                ry = math.sin(global_ray_angle)
                min_distance = max_range
                
                # Raycast against all walls using cross product
                for wall in map_obj.wall_list:
                    qx, qy = wall.corner1.x, wall.corner1.y
                    bx, by = wall.corner2.x, wall.corner2.y
                    sx = bx - qx
                    sy = by - qy
                    
                    r_cross_s = rx * sy - ry * sx
                    if abs(r_cross_s) > 1e-6: 
                        q_p_x = qx - self.state.x
                        q_p_y = qy - self.state.y  
                        t = (q_p_x * sy - q_p_y * sx) / r_cross_s 
                        u = (q_p_x * ry - q_p_y * rx) / r_cross_s 
                        if t > 0 and 0 <= u <= 1:
                            if t < min_distance:
                                min_distance = t
                
                # Weight calculation
                if min_distance < max_range:
                    error = min_distance - measured_dist
                    var = getattr(parameters, 'distance_variance', 0.1)
                    # Scale penalty by 10 to keep particles alive with dense rays
                    log_weight += -(error**2) / (2 * var * 10.0)
                else:
                    log_weight -= 5.0 

        self.weight = math.exp(log_weight) + 1e-15

    def deepcopy(self):
        return copy.deepcopy(self)

class ParticleSet:
    def __init__(self, num_particles, xy_range, initial_state, state_stdev, known_start_state):
        self.num_particles = num_particles
        self.particle_range = xy_range
        if known_start_state:
            self.generate_initial_state_particles(initial_state, state_stdev)
        else:
            self.generate_uniform_random_particles(xy_range)
        self.mean_state = State(0, 0, 0)
        self.update_mean_state()
        
    def generate_uniform_random_particles(self, xy_range):
        self.particle_list = []
        for i in range(self.num_particles):
            p = Particle()
            p.randomize_uniformly(xy_range)
            self.particle_list.append(p)

    def generate_initial_state_particles(self, initial_state, state_stdev):
        self.particle_list = []
        for i in range(self.num_particles):
            p = Particle()
            p.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(p)

    def resample(self, max_weight):
        is_lost = max_weight < 1e-10
        recovery_rate = 0.15 if is_lost else 0.02
        
        new_particles = []
        num_to_resample = int(self.num_particles * (1.0 - recovery_rate))
        
        if max_weight == 0 or num_to_resample == 0:
            self.generate_uniform_random_particles(self.particle_range)
            return

        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        
        for i in range(num_to_resample):
            beta += random.uniform(0, 2.0 * max_weight)
            while beta > self.particle_list[index].weight:
                beta -= self.particle_list[index].weight
                index = (index + 1) % self.num_particles
            new_particles.append(self.particle_list[index].deepcopy())
            
        while len(new_particles) < self.num_particles:
            p = Particle()
            p.randomize_uniformly(self.particle_range)
            new_particles.append(p)
            
        self.particle_list = new_particles
            
    def update_mean_state(self):
        sum_x = sum(p.state.x for p in self.particle_list)
        sum_y = sum(p.state.y for p in self.particle_list)
        sum_sin = sum(math.sin(p.state.theta) for p in self.particle_list)
        sum_cos = sum(math.cos(p.state.theta) for p in self.particle_list)
            
        self.mean_state.x = sum_x / self.num_particles
        self.mean_state.y = sum_y / self.num_particles
        self.mean_state.theta = angle_wrap(math.atan2(sum_sin, sum_cos))

class ParticleFilter:
    def __init__(self, num_particles, map_obj, initial_state, state_stdev, known_start_state):
        self.map = map_obj
        self.particle_set = ParticleSet(num_particles, map_obj.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate = self.particle_set.mean_state

    def update(self, v_cmd, alpha_cmd, measurement_signal, delta_t):
        self.prediction(v_cmd, alpha_cmd, delta_t)
        if hasattr(measurement_signal, 'angles') and len(measurement_signal.angles) > 0:
            self.correction(measurement_signal)
        self.particle_set.update_mean_state()
        self.state_estimate = self.particle_set.mean_state.deepcopy()

    def prediction(self, v_cmd, alpha_cmd, delta_t):
        for particle in self.particle_set.particle_list:
            particle.propagate_state(v_cmd, alpha_cmd, delta_t)
        
    def correction(self, measurement_signal):
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)
        max_weight = max((p.weight for p in self.particle_set.particle_list), default=0)
        self.particle_set.resample(max_weight)

class ParticleFilterPlot:
    def __init__(self, map_obj, dataset_filepath):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.map = map_obj
        self.video_filename = f"{os.path.splitext(os.path.basename(dataset_filepath))[0]}_pf_run.mp4"
        self.video_writer = None 
        
        self.gt_hist_x, self.gt_hist_y = [], []
        self.est_hist_x, self.est_hist_y = [], []

    def update(self, state_mean, particle_set, lidar_signal, gt_state, hold_show_plot, step):
        self.ax.clear()
        
        self.gt_hist_x.append(gt_state.x)
        self.gt_hist_y.append(gt_state.y)
        self.est_hist_x.append(state_mean.x)
        self.est_hist_y.append(state_mean.y)
        
        for wall in self.map.wall_list:
            self.ax.plot([wall.corner1.x, wall.corner2.x], [wall.corner1.y, wall.corner2.y], 'k-', linewidth=2, zorder=1)

        x_particles = [p.state.x for p in particle_set.particle_list]
        y_particles = [p.state.y for p in particle_set.particle_list]
        self.ax.scatter(x_particles, y_particles, s=2, c='lime', alpha=0.15, zorder=2)

        self.ax.plot(self.gt_hist_x, self.gt_hist_y, 'b-', linewidth=1.5, alpha=0.6, label="Ground Truth", zorder=3)
        self.ax.plot(self.est_hist_x, self.est_hist_y, 'm--', linewidth=1.5, alpha=0.6, label="PF Estimate", zorder=4)

        # --- CHANGED: Visualize all Lidar rays ---
        
        ray_x, ray_y = [], []
        if hasattr(lidar_signal, 'angles'):
            for i in range(0, lidar_signal.num_lidar_rays, 1): 
                raw_dist = lidar_signal.distances[i]
                if raw_dist < 4900:
                    dist = lidar_signal.convert_hardware_distance(raw_dist)
                    angle = angle_wrap(lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta)
                    ray_x.extend([state_mean.x, state_mean.x + dist * math.cos(angle), None])
                    ray_y.extend([state_mean.y, state_mean.y + dist * math.sin(angle), None])
            self.ax.plot(ray_x, ray_y, color='red', alpha=0.15, linewidth=0.3, zorder=5)

        self.ax.plot(gt_state.x, gt_state.y, 'bo', markersize=6, zorder=6)
        self.ax.quiver(gt_state.x, gt_state.y, math.cos(gt_state.theta), math.sin(gt_state.theta), color='blue', scale=15, width=0.007, zorder=7)
        
        self.ax.plot(state_mean.x, state_mean.y, 'mo', markersize=6, zorder=8)
        self.ax.quiver(state_mean.x, state_mean.y, math.cos(state_mean.theta), math.sin(state_mean.theta), color='magenta', scale=15, width=0.007, zorder=9)
        
        self.ax.set_title(f"Offline PF Processing vs Ground Truth | Frame: {step}")
        self.ax.set_xlim(self.map.plot_range[0], self.map.plot_range[1])
        self.ax.set_ylim(self.map.plot_range[2], self.map.plot_range[3])
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend(loc='upper right', fontsize='x-small')

        self.fig.canvas.draw()
        img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        
        if self.video_writer is None:
            h, w, _ = img_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (w, h))
            
        self.video_writer.write(img_bgr)

        if hold_show_plot:
            if self.video_writer is not None:
                self.video_writer.release()
            plt.show()
        else:
            plt.pause(0.001)

def offline_pf():
    map_obj = Map(parameters.wall_corner_list)
    filename = './data/robot_data_0_0_25_02_26_21_41_33.pkl'
    
    # Load data directly using DataLoader to bypass any array mismatch errors
    data_dict = robot_python_code.DataLoader(filename).load()
    time_list = data_dict['time']
    control_list = data_dict['control_signal']
    sensor_list = data_dict['robot_sensor_signal']
    camera_list = data_dict.get('camera_sensor_signal', [])
    
    # Extract initial camera data for accurate PF initialization
    if len(camera_list) > 0 and len(camera_list[0]) >= 6:
        cam_0 = camera_list[0]
        initial_gt_state = State(cam_0[0], cam_0[1], cam_0[5])
    else:
        initial_gt_state = State(0.5, 0.5, 1.57) # Safe fallback
        
    particle_filter = ParticleFilter(
        parameters.num_particles, map_obj, 
        initial_state=initial_gt_state, 
        state_stdev=State(0.05, 0.05, 0.05),
        known_start_state=True
    )
    
    pf_plot = ParticleFilterPlot(map_obj, filename)

    plt.ion()
    last_gt_state = initial_gt_state 
    
    for t in range(1, len(time_list)):
        delta_t_s = (time_list[t] - time_list[t-1]) / 1000.0 
        if delta_t_s <= 0: continue
            
        # Extract Commands exactly as the prediction script expects
        v_cmd = control_list[t][0]
        alpha_cmd = control_list[t][1]
        z_t = sensor_list[t]
        
        # Safely extract Camera Ground Truth 
        if t < len(camera_list) and len(camera_list[t]) >= 6:
            cam_t = camera_list[t]
            last_gt_state = State(cam_t[0], cam_t[1], cam_t[5])

        # Step the filter using direct velocity and steering commands
        particle_filter.update(v_cmd, alpha_cmd, z_t, delta_t_s)
        
        # Subsample visual rendering to speed up offline processing
        if t % 3 == 0:
            pf_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, last_gt_state, False, t)

    plt.ioff()
    pf_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, last_gt_state, True, len(time_list))

if __name__ == '__main__':
    offline_pf()