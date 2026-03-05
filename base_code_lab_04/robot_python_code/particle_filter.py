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
import data_handling

# Helper function to make sure all angles are between -pi and pi
def angle_wrap(angle):
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

# Helper class to store and manipulate your states.
class State:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def distance_to(self, other_state):
        return math.sqrt(math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2))
        
    def distance_to_squared(self, other_state):
        return math.pow(self.x - other_state.x, 2) + math.pow(self.y - other_state.y, 2)

    def deepcopy(self):
        return copy.deepcopy(self)
        
    def print(self):
        print("State: ",self.x, self.y, self.theta)

# Class to store walls as objects
class Wall:
    def __init__(self, wall_corners):
        self.corner1 = State(wall_corners[0], wall_corners[1], 0)
        self.corner2 = State(wall_corners[2], wall_corners[3], 0)
        self.corner1_mm = State(wall_corners[0] * 1000, wall_corners[1] * 1000, 0)
        self.corner2_mm = State(wall_corners[2] * 1000, wall_corners[3] * 1000, 0)
        
        self.m = (wall_corners[3] - wall_corners[1])/(0.0001 + wall_corners[2] -  wall_corners[0])
        self.b = wall_corners[3] - self.m * wall_corners[2]
        self.b_mm =  wall_corners[3] * 1000 - self.m * wall_corners[2] * 1000
        self.length = self.corner1.distance_to(self.corner2)
        self.length_mm_squared = self.corner1_mm.distance_to_squared(self.corner2_mm)
        
        if self.m > 1000:
            self.vertical = True
        else:
            self.vertical = False
        if abs(self.m) < 0.1:
            self.horizontal = True
        else:
            self.horizontal = False

# A class to store 2D maps
class Map:
    def __init__(self, wall_corner_list):
        self.wall_list = []
        for wall_corners in wall_corner_list:
            self.wall_list.append(Wall(wall_corners))
        min_x = 999999
        max_x = -99999
        min_y = 999999
        max_y = -99999
        for wall in self.wall_list:
            min_x = min(min_x, min(wall.corner1.x, wall.corner2.x))
            max_x = max(max_x, max(wall.corner1.x, wall.corner2.x))
            min_y = min(min_y, min(wall.corner1.y, wall.corner2.y))
            max_y = max(max_y, max(wall.corner1.y, wall.corner2.y))
        border = 0.5
        self.plot_range = [min_x - border, max_x + border, min_y - border, max_y + border]
        self.particle_range = [min_x , max_x , min_y, max_y]

    def closest_distance_to_walls(self, state):
        closest_distance = 999999999999
        for wall in self.wall_list:
            closest_distance = self.get_distance_to_wall(state, wall, closest_distance)
        return closest_distance
        
    def get_distance_to_wall(self, state, wall, closest_distance):
        x0, y0 = state.x, state.y
        vx, vy = math.cos(state.theta), math.sin(state.theta)
        
        x1, y1 = wall.corner1.x, wall.corner1.y
        x2, y2 = wall.corner2.x, wall.corner2.y
        
        wx, wy = x2 - x1, y2 - y1
        denominator = wx * vy - wy * vx
        
        if abs(denominator) < 1e-6:
            return closest_distance 
            
        dx, dy = x1 - x0, y1 - y0
        t = (dx * vy - dy * vx) / denominator
        d = (dx * wy - dy * wx) / denominator
        
        if 0 <= t <= 1 and d > 0:
            if d < closest_distance:
                return d
        return closest_distance

# Class to hold a particle
class Particle:
    def __init__(self):
        self.state = State(0, 0, 0)
        self.weight = 1.0
        
    def randomize_uniformly(self, xy_range):
        x = random.uniform(xy_range[0], xy_range[1])
        y = random.uniform(xy_range[2], xy_range[3])
        theta = random.uniform(-math.pi, math.pi)
        self.state = State(x, y, theta)
        self.weight = 1.0

    def randomize_around_initial_state(self, initial_state, state_stdev):
        x = random.gauss(initial_state.x, state_stdev.x)
        y = random.gauss(initial_state.y, state_stdev.y)
        theta = angle_wrap(random.gauss(initial_state.theta, state_stdev.theta))
        self.state = State(x, y, theta)
        self.weight = 1.0
        
    def propagate_state(self, last_state, delta_encoder_counts, steering, delta_t):
        L = 0.145
        KE_VALUE = 0.001345210  
        DELTA_COEFFS = [0.000027, 0.007798, 0.029847]
        
        # Keep noise high enough to help the filter recover from small errors
        VAR_V = 0.00057829 * 1000
        VAR_DELTA = 0.00023134 * 1000

        # Force forward motion if encoders are reversed
        delta_encoder_counts = abs(delta_encoder_counts)

        ds = delta_encoder_counts * KE_VALUE
        v_physical = (ds / delta_t) if delta_t > 0 else 0
        
        # Stochastic inputs
        v_s = v_physical + random.gauss(0, math.sqrt(VAR_V))
        delta_det = DELTA_COEFFS[0]*(steering**2) + DELTA_COEFFS[1]*steering + DELTA_COEFFS[2]
        d_s = delta_det + random.gauss(0, math.sqrt(VAR_DELTA))
        
        # w_s is the angular velocity
        w_s = (v_s * math.tan(d_s)) / L if abs(L) > 0 else 0
        
        # Integration
        x = last_state.x + delta_t * v_s * math.cos(last_state.theta)
        y = last_state.y + delta_t * v_s * math.sin(last_state.theta)
        
        # --- ROTATION DIRECTION FIX ---
        # If your robot turns the WRONG way on screen, change this '+' to a '-' 
        # or vice versa. Standard Cartesian is '+'.
        theta = angle_wrap(last_state.theta + delta_t * w_s) 
        
        self.state = State(x, y, theta)
        
    def calculate_weight(self, lidar_signal, map):
        """
        Determines particle weight using Log-Likelihood to prevent math underflow.
        Includes extrinsic calibration for Lidar-to-Robot transform.
        """
        # --- 1. Extrinsic Calibration (ADJUST THESE) ---
        # Distance from robot center (rear axle) to Lidar turret in meters
        X_OFFSET = 0.12  
        # Angular offset if Lidar '0' isn't perfectly 'Front' (in Radians)
        THETA_OFFSET = 0.0 
        # 1.0 for Counter-Clockwise Lidar, -1.0 for Clockwise Lidar
        POLARITY = 1.0   

        log_weight = 0.0
        
        # --- 2. Calculate Sensor Origin in Global Space ---
        # The Lidar turret is not at the robot (x,y); it is offset along the heading.
        xs = self.state.x + X_OFFSET * math.cos(self.state.theta)
        ys = self.state.y + X_OFFSET * math.sin(self.state.theta)

        # --- 3. Process Lidar Rays ---
        # Subsample: Checking every 20th ray prevents "hyper-confidence" and saves CPU
        ray_step = 20 
        
        for i in range(0, lidar_signal.num_lidar_rays, ray_step):
            raw_dist = lidar_signal.distances[i]
            
            # Hardware filter: ignore invalid/too-close readings
            if raw_dist > 20: 
                measured_dist = lidar_signal.convert_hardware_distance(raw_dist)
                
                # Transform relative Lidar angle to Global Frame
                ray_angle_rel = lidar_signal.convert_hardware_angle(lidar_signal.angles[i])
                global_ray_angle = angle_wrap(
                    self.state.theta - (POLARITY * ray_angle_rel) + THETA_OFFSET
                )
                
                # Create a temporary state for the Raycaster starting at the SENSOR origin
                ray_origin_state = State(xs, ys, global_ray_angle)
                expected_dist = map.closest_distance_to_walls(ray_origin_state)
                
                # --- 4. Weight Calculation (Log Space) ---
                if expected_dist < 9999: # If ray hits a wall in the map
                    # Gaussian Error: we add negative penalties (Log-Likelihood)
                    # Instead of multiplying exp(), we add -(error^2 / 2*sigma^2)
                    error = expected_dist - measured_dist
                    
                    # distance_variance should be ~0.05 to 0.2 depending on sensor noise
                    var = getattr(parameters, 'distance_variance', 0.1)
                    log_weight += -(error**2) / (2 * var)
                else:
                    # Penalty for predicting "Infinity" when the sensor hit a wall
                    log_weight -= 5.0 

        # --- 5. Final Normalization ---
        # Convert back from Log space to [0, 1] weight
        # 1e-11 is a floor to ensure the particle doesn't literally "die" and break the resampler
        self.weight = math.exp(log_weight) + 1e-11

    def gaussian(self, expected_distance, distance):
        var = getattr(parameters, 'distance_variance', 0.05) 
        return math.exp(-math.pow(expected_distance - distance, 2) / (2 * var))

    def deepcopy(self):
        return copy.deepcopy(self)
        
    def print(self):
        print("Particle: ", self.state.x, self.state.y, self.state.theta, " w: ", self.weight)

class ParticleSet:
    def __init__(self, num_particles, xy_range, initial_state, state_stdev, known_start_state):
        self.num_particles = num_particles
        self.particle_list = []
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
            random_particle = Particle()
            random_particle.randomize_uniformly(xy_range)
            self.particle_list.append(random_particle)

    def generate_initial_state_particles(self, initial_state, state_stdev):
        self.particle_list = []
        for i in range(self.num_particles):
            random_particle = Particle()
            random_particle.randomize_around_initial_state(initial_state, state_stdev)
            self.particle_list.append(random_particle)

    def resample(self, max_weight):
        total_weight = sum(p.weight for p in self.particle_list)
        if total_weight == 0 or max_weight == 0:
            self.generate_uniform_random_particles(self.particle_range)
            return
            
        new_particles = []
        index = random.randint(0, self.num_particles - 1)
        beta = 0.0
        
        for i in range(self.num_particles):
            beta += random.uniform(0, 2.0 * max_weight)
            while beta > self.particle_list[index].weight:
                beta -= self.particle_list[index].weight
                index = (index + 1) % self.num_particles
            new_particles.append(self.particle_list[index].deepcopy())
            
        self.particle_list = new_particles
            
    def update_mean_state(self):
        sum_x, sum_y = 0, 0
        sum_sin_theta, sum_cos_theta = 0, 0
        
        for p in self.particle_list:
            sum_x += p.state.x
            sum_y += p.state.y
            sum_sin_theta += math.sin(p.state.theta)
            sum_cos_theta += math.cos(p.state.theta)
            
        self.mean_state.x = sum_x / self.num_particles
        self.mean_state.y = sum_y / self.num_particles
        self.mean_state.theta = angle_wrap(math.atan2(sum_sin_theta, sum_cos_theta))

class ParticleFilter:
    def __init__(self, num_particles, map, initial_state, state_stdev, known_start_state, encoder_counts_0):
        self.map = map
        self.particle_set = ParticleSet(num_particles, map.particle_range, initial_state, state_stdev, known_start_state)
        self.state_estimate = self.particle_set.mean_state
        self.state_estimate_list = []
        self.last_time = 0
        self.last_encoder_counts = encoder_counts_0

    def update(self, odometry_signal, measurement_signal, delta_t):
        self.prediction(odometry_signal, delta_t)
        
        # --- FIXED: CORRECTION STEP UNCOMMENTED AND ACTIVE ---
        if hasattr(measurement_signal, 'angles') and len(measurement_signal.angles) > 0:
            self.correction(measurement_signal)
            
        self.particle_set.update_mean_state()
        self.state_estimate_list.append(self.state_estimate.deepcopy())

    def prediction(self, odometry_signal, delta_t):
        encoder_counts = odometry_signal[0]
        steering = odometry_signal[1]
        
        delta_encoder_counts = encoder_counts - self.last_encoder_counts
        self.last_encoder_counts = encoder_counts
        
        for particle in self.particle_set.particle_list:
            last_state = particle.state.deepcopy()
            particle.propagate_state(last_state, delta_encoder_counts, steering, delta_t)
        
    def correction(self, measurement_signal):
        for particle in self.particle_set.particle_list:
            particle.calculate_weight(measurement_signal, self.map)
            
        max_weight = max((particle.weight for particle in self.particle_set.particle_list), default=0)
        self.particle_set.resample(max_weight)
        
    def print_state_estimate(self):
        print("Mean state: ", self.particle_set.mean_state.x, self.particle_set.mean_state.y, self.particle_set.mean_state.theta)

class ParticleFilterPlot:
    def __init__(self, map, dataset_filepath):
        self.dir_length = 0.1
        self.fig, self.ax = plt.subplots()
        self.map = map
        
        # Extract the dataset name to use as the video filename
        base_name = os.path.basename(dataset_filepath)
        name_without_ext = os.path.splitext(base_name)[0]
        self.video_filename = f"{name_without_ext}_pf_run.mp4"
        
        self.video_writer = None # Will be initialized on the first frame

    def update(self, state_mean, particle_set, lidar_signal, hold_show_plot):
        plt.clf()
        
        # 1. Plot Walls
        for wall in self.map.wall_list:
            plt.plot([wall.corner1.x, wall.corner2.x],[wall.corner1.y, wall.corner2.y],'k')

        # 2. Plot Lidar
        for i in range(len(lidar_signal.angles)):
            distance = lidar_signal.convert_hardware_distance(lidar_signal.distances[i])
            angle = lidar_signal.convert_hardware_angle(lidar_signal.angles[i]) + state_mean.theta
            x_ray = [state_mean.x, state_mean.x + distance * math.cos(angle)]
            y_ray = [state_mean.y, state_mean.y + distance * math.sin(angle)]
            plt.plot(x_ray, y_ray, 'r')

        # 3. Plot State Estimate and Particles
        plt.plot(state_mean.x, state_mean.y,'ro')
        plt.plot([state_mean.x, state_mean.x+ self.dir_length*math.cos(state_mean.theta) ], [state_mean.y, state_mean.y+ self.dir_length*math.sin(state_mean.theta) ],'r')
        x_particles, y_particles = self.to_plot_data(particle_set)
        plt.plot(x_particles, y_particles, 'g.')
        
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        plt.axis(self.map.plot_range)
        plt.grid()

        # --- VIDEO RECORDING LOGIC ---
        self.fig.canvas.draw()
        
        img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        h, w, _ = img_bgr.shape
        
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            fps = 20.0 
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, (w, h))
            print(f"[Video] Started recording to: {self.video_filename}")
            
        self.video_writer.write(img_bgr)
        # -----------------------------

        if hold_show_plot:
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"[Video] Successfully saved {self.video_filename}")
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)

    def to_plot_data(self, particle_set):
        x_list = []
        y_list = []
        for p in particle_set.particle_list:
            x_list.append(p.state.x)
            y_list.append(p.state.y)
        return x_list, y_list

def offline_pf():
    map = Map(parameters.wall_corner_list)
    filename = './data/robot_data_0_0_25_02_26_21_41_33.pkl'
    pf_data = data_handling.get_file_data_for_pf(filename)

    particle_filter = ParticleFilter(parameters.num_particles, map, initial_state = State(0.5, 2.0, 1.57), state_stdev = State(0.1,0.1,0.1), known_start_state=True, encoder_counts_0=pf_data[0][2].encoder_counts)
    particle_filter_plot = ParticleFilterPlot(map, filename)

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        
        delta_t_s = (pf_data[t][0] - pf_data[t-1][0]) / 1000.0 
        
        if delta_t_s <= 0:
            continue
            
        steering = getattr(row[2], 'steering', getattr(row[1], 'steering', 0))
        
        u_t = np.array([row[2].encoder_counts, steering])
        z_t = row[2] 

        particle_filter.update(u_t, z_t, delta_t_s)
        particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, False)

    particle_filter_plot.update(particle_filter.particle_set.mean_state, particle_filter.particle_set, z_t, True)

if __name__ == '__main__':
    offline_pf()