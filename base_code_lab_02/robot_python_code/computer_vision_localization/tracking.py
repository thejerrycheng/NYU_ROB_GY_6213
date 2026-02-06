import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import csv

# --- CONFIGURATION ---
TAG_SIZE_CAR = 0.10     # Size of Car Tag
TAG_SIZE_GROUND = 0.10  # Size of Floor Tags
ORIGIN_TAG_ID = 1       # We assume this tag is (0,0,0)
CAR_TAG_ID = 0          # The tag on the robot

# --- LOGGING SETUP ---
LOG_FILENAME = "trajectory_log.csv"
csv_file = open(LOG_FILENAME, 'w', newline='')
csv_writer = csv.writer(csv_file)
# Header
csv_writer.writerow([
    "Timestamp", 
    "Car_X", "Car_Y", "Car_Z", "Car_Theta",  # The Robot
    "Plane_A", "Plane_B", "Plane_C", "Plane_D", # The Floor (Camera Frame)
    "Opt_Error_Px", "Num_Ground_Tags" # Optimization Quality
])
print(f"Logging data to {LOG_FILENAME}...")

# --- STATE VARIABLES ---
ground_map = {} 
is_map_calibrated = False

def get_transform_matrix(rvec, tvec):
    mat = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    mat[:3, :3] = R
    mat[:3, 3] = tvec.flatten()
    return mat

def get_tag_corners_in_world(center_xyz, size):
    cx, cy, cz = center_xyz
    s = size / 2.0
    return np.array([
        [cx - s, cy + s, cz], # TL
        [cx + s, cy + s, cz], # TR
        [cx + s, cy - s, cz], # BR
        [cx - s, cy - s, cz]  # BL
    ], dtype=np.float32)

def compute_reprojection_error(obj_points, img_points, rvec, tvec, K, D):
    """Calculates how accurate the solvePnP result is (in pixels)"""
    projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    projected_points = projected_points.reshape(-1, 2)
    error = cv2.norm(img_points, projected_points, cv2.NORM_L2)
    return error / len(projected_points)

def main():
    global is_map_calibrated, ground_map

    # 1. Setup RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intrinsics.coeffs)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()

    # Mapping State
    map_samples = {}
    frames_mapped = 0
    MAPPING_FRAMES_NEEDED = 50
    T_world_to_cam = None 
    
    # Store previous car position to calculate Heading (Theta)
    prev_car_pos = None

    print(f"STEP 1: Place Tag {ORIGIN_TAG_ID} at origin.")
    print("STEP 2: Scatter other tags.")
    print("Press 'm' to MAP. Press 'q' to QUIT.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None:
                aruco.drawDetectedMarkers(img, corners, ids)
                ids = ids.flatten()

                # --- PHASE 1: MAPPING ---
                if not is_map_calibrated and frames_mapped > 0:
                    if ORIGIN_TAG_ID in ids:
                        idx_origin = list(ids).index(ORIGIN_TAG_ID)
                        rvec_o, tvec_o, _ = aruco.estimatePoseSingleMarkers(corners[idx_origin], TAG_SIZE_GROUND, camera_matrix, dist_coeffs)
                        T_cam_to_origin = get_transform_matrix(rvec_o[0], tvec_o[0])
                        T_origin_to_cam = np.linalg.inv(T_cam_to_origin)

                        for i, tag_id in enumerate(ids):
                            if tag_id == ORIGIN_TAG_ID or tag_id == CAR_TAG_ID: continue
                            
                            rvec_i, tvec_i, _ = aruco.estimatePoseSingleMarkers(corners[i], TAG_SIZE_GROUND, camera_matrix, dist_coeffs)
                            T_cam_to_tag = get_transform_matrix(rvec_i[0], tvec_i[0])
                            T_origin_to_tag = T_origin_to_cam @ T_cam_to_tag
                            
                            if tag_id not in map_samples: map_samples[tag_id] = []
                            map_samples[tag_id].append(T_origin_to_tag[:3, 3])
                        
                        frames_mapped += 1
                        cv2.putText(img, f"Mapping: {frames_mapped}/{MAPPING_FRAMES_NEEDED}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    if frames_mapped >= MAPPING_FRAMES_NEEDED:
                        ground_map[ORIGIN_TAG_ID] = np.array([0.0, 0.0, 0.0])
                        for tid, positions in map_samples.items():
                            avg_pos = np.mean(positions, axis=0)
                            avg_pos[2] = 0.0 # Force Flat
                            ground_map[tid] = avg_pos
                            print(f"Tag {tid} Map Pos: {avg_pos}")
                        is_map_calibrated = True
                        print("Mapping Complete.")

                # --- PHASE 2: TRACKING ---
                elif is_map_calibrated:
                    # 1. JOINT OPTIMIZATION
                    obj_points = []
                    img_points = []
                    
                    for i, tag_id in enumerate(ids):
                        if tag_id in ground_map:
                            obj_points.extend(get_tag_corners_in_world(ground_map[tag_id], TAG_SIZE_GROUND))
                            img_points.extend(corners[i][0])
                    
                    if len(obj_points) >= 4:
                        # Solve PnP
                        obj_pts_arr = np.array(obj_points, dtype=np.float32)
                        img_pts_arr = np.array(img_points, dtype=np.float32)
                        
                        ret, rvec_cam, tvec_cam = cv2.solvePnP(obj_pts_arr, img_pts_arr, camera_matrix, dist_coeffs)
                        
                        # Compute Optimization Error
                        opt_error = compute_reprojection_error(obj_pts_arr, img_pts_arr, rvec_cam, tvec_cam, camera_matrix, dist_coeffs)

                        # Compute Floor Plane Equation (Ax + By + Cz + D = 0)
                        # Relative to Camera Frame
                        R_mat, _ = cv2.Rodrigues(rvec_cam) # World->Cam rotation
                        
                        # Normal vector of floor in World is (0,0,1). 
                        # In Camera frame, it is the 3rd column of R (since R maps World basis to Cam basis)
                        normal_cam = R_mat[:, 2] 
                        A, B, C = normal_cam[0], normal_cam[1], normal_cam[2]
                        
                        # A point on the plane in Camera frame is the translation vector 'tvec_cam'
                        # (Because tvec is the origin of the world expressed in camera frame)
                        # D = - (n . p)
                        D = -np.dot(normal_cam, tvec_cam.flatten())
                        
                        # Matrix logic
                        T_world_to_cam = np.eye(4)
                        T_world_to_cam[:3, :3] = R_mat
                        T_world_to_cam[:3, 3] = tvec_cam.flatten()

                        # 2. TRACK CAR
                        if CAR_TAG_ID in ids:
                            idx_car = list(ids).index(CAR_TAG_ID)
                            rvec_car, tvec_car, _ = aruco.estimatePoseSingleMarkers(corners[idx_car], TAG_SIZE_CAR, camera_matrix, dist_coeffs)
                            
                            T_cam_to_car = get_transform_matrix(rvec_car[0], tvec_car[0])
                            T_cam_to_world = np.linalg.inv(T_world_to_cam)
                            T_world_to_car = T_cam_to_world @ T_cam_to_car
                            
                            cx, cy, cz = T_world_to_car[0,3], T_world_to_car[1,3], T_world_to_car[2,3]
                            
                            # Simple Heading Calculation (Theta)
                            theta = 0.0
                            if prev_car_pos is not None:
                                dx = cx - prev_car_pos[0]
                                dy = cy - prev_car_pos[1]
                                if np.sqrt(dx**2 + dy**2) > 0.01: # Only update if moved > 1cm
                                    theta = np.arctan2(dy, dx)
                            
                            prev_car_pos = [cx, cy]

                            # LOGGING
                            timestamp = time.time()
                            csv_writer.writerow([
                                timestamp, 
                                cx, cy, cz, theta, 
                                A, B, C, D, 
                                opt_error, 
                                len(obj_points)/4
                            ])
                            
                            # VISUALIZATION
                            text1 = f"Car: ({cx:.2f}, {cy:.2f}) H:{cz:.2f}"
                            text2 = f"Plane Err: {opt_error:.2f} px"
                            cv2.putText(img, text1, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            cv2.putText(img, text2, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('m') and not is_map_calibrated: frames_mapped = 1
            if key == ord('q'): break
            
            status = "TRACKING" if is_map_calibrated else "READY (Press 'm')"
            if frames_mapped > 0 and not is_map_calibrated: status = "MAPPING..."
            cv2.putText(img, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Logger", img)

    finally:
        pipeline.stop()
        csv_file.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()