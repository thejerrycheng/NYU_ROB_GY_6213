import cv2
import cv2.aruco as aruco
import numpy as np
import math

# ==========================================
# 1. CAMERA & TAG SETTINGS
# ==========================================
CAM_MATRIX = np.array([
    [1407.29350, 0.00000000, 975.201510],
    [0.00000000, 1405.70507, 534.098781],
    [0.00000000, 0.00000000, 1.00000000]
])

DIST_COEFFS = np.array([
    [0.05904389, -0.03912268, -0.000681, 0.00122093, -0.09274777]
])

TAG_SIZE_M = 0.0762  # 3 inches
TARGET_TAG_ID = 5
DICT_TYPE = aruco.DICT_APRILTAG_25h9

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_raw_matrices(frame):
    """Returns the raw Rotation Matrix and Translation Vector in the Camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or TARGET_TAG_ID not in ids:
        return None, None, frame
        
    idx = np.where(ids == TARGET_TAG_ID)[0][0]
    tag_corners = corners[idx][0]
    
    cv2.polylines(frame, [tag_corners.astype(int)], True, (0, 255, 0), 2)
    
    half_s = TAG_SIZE_M / 2.0
    obj_points = np.array([
        [-half_s,  half_s, 0], 
        [ half_s,  half_s, 0], 
        [ half_s, -half_s, 0], 
        [-half_s, -half_s, 0]  
    ], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(obj_points, tag_corners, CAM_MATRIX, DIST_COEFFS)
    
    if not success:
        return None, None, frame
        
    cv2.drawFrameAxes(frame, CAM_MATRIX, DIST_COEFFS, rvec, tvec, TAG_SIZE_M)
        
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, frame

# ==========================================
# 3. MAIN VISUALIZATION LOOP
# ==========================================
def live_2d_tracker():
    cap = cv2.VideoCapture(0)
    
    origin_R = None
    origin_t = None
    cam_pos_world = None
    
    # Map settings
    MAP_SIZE = 800
    PIXELS_PER_METER = 300  
    CENTER_X = MAP_SIZE // 2
    CENTER_Y = MAP_SIZE // 2
    
    print("\n" + "="*45)
    print(" WORLD FRAME PROJECTOR (TAG = 0,0,0) ")
    print("="*45)
    print("-> Place tag in starting position to set World Origin.")
    print("-> Camera is assumed STATIONARY.")
    print("-> Press 'r' to Reset the World Origin.")
    print("-> Press 'ESC' to Quit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        # Undistort frame
        h, w = frame.shape[:2]
        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(CAM_MATRIX, DIST_COEFFS, (w,h), 1, (w,h))
        frame = cv2.undistort(frame, CAM_MATRIX, DIST_COEFFS, None, new_cam_mat)
        
        R_current, t_current, debug_frame = get_raw_matrices(frame)
        
        # Create a blank black image for the 2D map
        map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        
        # Draw World Frame Axes (Origin = Center of map)
        cv2.line(map_img, (CENTER_X, 0), (CENTER_X, MAP_SIZE), (40, 40, 40), 1)
        cv2.line(map_img, (0, CENTER_Y), (MAP_SIZE, CENTER_Y), (40, 40, 40), 1)
        cv2.circle(map_img, (CENTER_X, CENTER_Y), int(0.5 * PIXELS_PER_METER), (40, 40, 40), 1) # 0.5m radius
        
        if R_current is not None and t_current is not None:
            # --- INITIALIZATION ---
            if origin_R is None:
                origin_R = R_current.copy()
                origin_t = t_current.copy()
                
                # Calculate Stationary Camera's position in the World Frame
                cam_pos_world = -np.dot(origin_R.T, origin_t)
                print("[+] World Frame Initialized at current tag position!")
            
            # --- COORDINATE TRANSFORMATION ---
            # 1. Transform current tag position into the World Frame
            tag_pos_world = np.dot(origin_R.T, t_current - origin_t)
            
            # Extract 2D projection (X and Y on the World plane)
            tag_world_x = tag_pos_world[0][0]
            tag_world_y = tag_pos_world[1][0]
            
            # 2. Transform current tag orientation into the World Frame
            R_world_tag = np.dot(origin_R.T, R_current)
            
            # --- EXTRACT +Y AXIS (CAR FRONT) ---
            # Instead of the X-axis (column 0), we use the Y-axis (column 1) of the rotation matrix
            front_vec_x = R_world_tag[0, 1]
            front_vec_y = R_world_tag[1, 1]
            tag_world_yaw = math.atan2(front_vec_y, front_vec_x)
            
            # --- DRAW ON 2D MAP ---
            # X is Right, Y is Up (Subtract Y because OpenCV Y goes down)
            px_tag = CENTER_X + int(tag_world_x * PIXELS_PER_METER)
            py_tag = CENTER_Y - int(tag_world_y * PIXELS_PER_METER) 
            
            px_cam = CENTER_X + int(cam_pos_world[0][0] * PIXELS_PER_METER)
            py_cam = CENTER_Y - int(cam_pos_world[1][0] * PIXELS_PER_METER)
            
            # Draw Stationary Camera (Blue Square)
            cv2.rectangle(map_img, (px_cam-6, py_cam-6), (px_cam+6, py_cam+6), (255, 100, 0), -1)
            cv2.putText(map_img, "Camera", (px_cam+10, py_cam), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

            # Draw World Origin (White Cross)
            cv2.drawMarker(map_img, (CENTER_X, CENTER_Y), (255, 255, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(map_img, "(0,0)", (CENTER_X+10, CENTER_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw Moving Tag (Green Dot)
            cv2.circle(map_img, (px_tag, py_tag), 6, (0, 255, 0), -1)
            
            # Draw Tag Heading Arrow (Now pointing along the +Y axis)
            arrow_len = 25
            end_x = px_tag + int(arrow_len * front_vec_x)
            end_y = py_tag - int(arrow_len * front_vec_y)
            cv2.arrowedLine(map_img, (px_tag, py_tag), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
            
            # --- LOGGING & OVERLAYS ---
            print(f"Tag World Pos -> X: {tag_world_x:+.3f}m | Y: {tag_world_y:+.3f}m | Theta: {math.degrees(tag_world_yaw):+06.1f}deg", end='\r')
            
            cv2.putText(map_img, f"Tag X: {tag_world_x:+.3f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(map_img, f"Tag Y: {tag_world_y:+.3f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(map_img, f"Tag Yaw: {math.degrees(tag_world_yaw):+.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            cv2.putText(map_img, "TAG NOT DETECTED", (CENTER_X - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Camera Feed", debug_frame)
        
        cv2.imshow("World Frame Projection", map_img)
        
        key = cv2.waitKey(1)
        if key == 27: 
            break
        elif key == ord('r'): 
            origin_R = None
            origin_t = None
            cam_pos_world = None
            print("\n[!] World Frame Reset. Place tag to recalibrate.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nExiting tracker...")

if __name__ == "__main__":
    live_2d_tracker()