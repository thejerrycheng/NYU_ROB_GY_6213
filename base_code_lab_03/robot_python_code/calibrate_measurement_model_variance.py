import cv2
import cv2.aruco as aruco
import numpy as np
import math

# ==========================================
# 1. EMPIRICAL GROUND TRUTH POSITIONS
# ==========================================
# [X_world, Y_world, Yaw_world]
GROUND_TRUTHS = [
    [0.00, 0.00,  1.57],   # Loc 1: Origin
    [0.00, 0.50,  1.57],   # Loc 2: 50cm Forward
    [0.00, 1.00,  1.57],   # Loc 3: 100cm Forward
    [0.00, 0.50,  3.14],   # Loc 4: Left Turn
    [0.00, 0.50,  3.14+1.57],   # Loc 5: U-Turn
    [0.00, 0.50,  1.57],   # Loc 2: 50cm Forward
    [0.50, 0.50,  1.57],   # Loc 7: Far Forward
    [-0.50, 0.50,  1.57],   # Loc 8: Far Left
]

# ==========================================
# 2. CAMERA & TAG SETTINGS
# ==========================================
CAM_MATRIX = np.array([
    [1407.29350, 0.00000000, 975.201510],
    [0.00000000, 1405.70507, 534.098781], 
    [0.00000000, 0.00000000, 1.00000000]
])

DIST_COEFFS = np.array([
    [0.05904389, -0.03912268, -0.000681, 0.00122093, -0.09274777]
])

TAG_SIZE_M = 0.0762 
TARGET_TAG_ID = 5 
DICT_TYPE = aruco.DICT_APRILTAG_25h9

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_raw_matrices(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(DICT_TYPE), aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or TARGET_TAG_ID not in ids:
        return None, None, frame
        
    idx = np.where(ids == TARGET_TAG_ID)[0][0]
    half_s = TAG_SIZE_M / 2.0
    obj_points = np.array([[-half_s,half_s,0],[half_s,half_s,0],[half_s,-half_s,0],[-half_s,-half_s,0]], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(obj_points, corners[idx][0], CAM_MATRIX, DIST_COEFFS)
    
    if not success: return None, None, frame
    cv2.drawFrameAxes(frame, CAM_MATRIX, DIST_COEFFS, rvec, tvec, TAG_SIZE_M)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, frame

def calculate_q_matrix():
    cap = cv2.VideoCapture(0)
    origin_R, origin_t = None, None
    errors_x, errors_y, errors_theta = [], [], []
    
    MAP_SIZE = 600
    PPM = 250 
    
    print("\n" + "="*45)
    print(" EXTERNAL CAMERA VARIANCE CALIBRATOR ")
    print("="*45)
    
    for i, truth in enumerate(GROUND_TRUTHS):
        x_true, y_true, th_true = truth
        captured = False
        
        while not captured:
            ret, frame = cap.read()
            if not ret: continue
            
            h, w = frame.shape[:2]
            new_mat, _ = cv2.getOptimalNewCameraMatrix(CAM_MATRIX, DIST_COEFFS, (w,h), 1, (w,h))
            frame = cv2.undistort(frame, CAM_MATRIX, DIST_COEFFS, None, new_mat)
            
            R_curr, t_curr, debug_frame = get_raw_matrices(frame)
            map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
            
            # Draw Grid
            cv2.line(map_img, (MAP_SIZE//2, 0), (MAP_SIZE//2, MAP_SIZE), (40,40,40), 1)
            cv2.line(map_img, (0, MAP_SIZE//2), (MAP_SIZE, MAP_SIZE//2), (40,40,40), 1)

            if R_curr is not None:
                if origin_R is None:
                    cv2.putText(debug_frame, "PLACE AT ORIGIN AND PRESS SPACE", (10, 35), 1, 1.5, (0,255,0), 2)
                else:
                    pos_w = np.dot(origin_R.T, t_curr - origin_t)
                    xw, yw = pos_w[0][0], pos_w[1][0]
                    R_wt = np.dot(origin_R.T, R_curr)
                    yww = math.atan2(R_wt[1, 1], R_wt[0, 1]) # Car Front = +Y
                    
                    # Current Errors
                    ex, ey = xw - x_true, yw - y_true
                    eth = math.degrees(normalize_angle(yww - th_true))
                    
                    # 2D Mapping
                    px, py = int(MAP_SIZE/2 + xw*PPM), int(MAP_SIZE/2 - yw*PPM)
                    cv2.circle(map_img, (px, py), 6, (0, 255, 0), -1)
                    arr_x, arr_y = int(px + 25*math.cos(yww)), int(py - 25*math.sin(yww))
                    cv2.arrowedLine(map_img, (px, py), (arr_x, arr_y), (0, 0, 255), 2)
                    
                    cv2.putText(debug_frame, f"STEP {i+1} ERR: X={ex:.2f} Y={ey:.2f} TH={eth:.1f}deg", 
                                (10,35), 1, 1.2, (255,255,0), 2)

            cv2.imshow("Calibration Feed", debug_frame)
            cv2.imshow("World Frame Map", map_img)
            
            key = cv2.waitKey(1)
            if key == 27: cap.release(); cv2.destroyAllWindows(); return
            if key == 32 and R_curr is not None:
                if origin_R is None:
                    origin_R, origin_t = R_curr.copy(), t_curr.copy()
                    print("[+] World Origin Locked.")
                else:
                    e_th_final = normalize_angle(yww - th_true)
                    errors_x.append(xw - x_true)
                    errors_y.append(yw - y_true)
                    errors_theta.append(e_th_final)
                    
                    print(f"Captured Loc {i+1} | Err X: {xw-x_true:.3f} | Err Y: {yw-y_true:.3f} | Err Theta: {math.degrees(e_th_final):.2f} deg")
                    captured = True

    cap.release(); cv2.destroyAllWindows()
    
    # Calculate Variance
    v_x = np.var(errors_x, ddof=1)
    v_y = np.var(errors_y, ddof=1)
    v_th = np.var(errors_theta, ddof=1)
    
    print("\n" + "="*45)
    print(" FINAL MEASUREMENT VARIANCES (Q_t) ")
    print("="*45)
    print(f"var_x:     {v_x:.10f}")
    print(f"var_y:     {v_y:.10f}")
    print(f"var_theta: {v_th:.10f}")
    print("\nQ_t = np.diag([" + f"{v_x:.10f}, {v_y:.10f}, {v_th:.10f}" + "])")

if __name__ == "__main__":
    calculate_q_matrix()