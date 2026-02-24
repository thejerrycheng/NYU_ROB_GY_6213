import cv2 as cv
import numpy as np
import math
from pupil_apriltags import Detector

# ==========================================
# 1. EMPIRICAL GROUND TRUTH POSITIONS
# ==========================================
# Format: [True_X, True_Y, True_Theta] in meters and radians.
# NOTE: Place your robot at the first location (0,0,0) to calibrate the World Origin.

GROUND_TRUTHS = [
    [0.00, 0.00,  0.00],   # Loc 1: Origin (Sets the World 0,0,0)
    [0.50, 0.00,  0.00],   # Loc 2: 50cm Forward
    [1.00, 0.00,  0.00],   # Loc 3: 100cm Forward
    [1.00, 0.50,  1.57],   # Loc 4: 100cm Forward, 50cm Left, facing Left (+90 deg)
    [0.50, 0.50,  3.14],   # Loc 5: 50cm Forward, 50cm Left, facing Backward (+180 deg)
    [0.00, 0.50, -1.57],   # Loc 6: 0cm Forward, 50cm Left, facing Right (-90 deg)
    [0.00, 1.00,  0.00],   # Loc 7: 0cm Forward, 100cm Left, facing Forward
    [0.50, 1.00,  1.57],   # Loc 8: 50cm Forward, 100cm Left, facing Left (+90 deg)
    [1.00, 1.00, -1.57],   # Loc 9: 100cm Forward, 100cm Left, facing Right (-90 deg)
    [1.00, 0.50,  0.00]    # Loc 10: 100cm Forward, 50cm Left, facing Forward
]

# ==========================================
# 2. CAMERA & TAG SETTINGS
# ==========================================
# Replace with your OpenCV calibration outputs
CAM_MATRIX = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
DIST_COEFFS = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
TAG_SIZE_M = 0.05 

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def get_raw_pose(detector, frame, cam_params):
    """Extracts raw planar pose from the camera frame."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=cam_params, tag_size=TAG_SIZE_M)
    
    if not tags:
        return None, frame
        
    tag = tags[0]
    x_cam = tag.pose_t[0][0]
    y_cam = tag.pose_t[1][0]
    theta_cam = -math.atan2(tag.pose_R[1,0], tag.pose_R[0,0])
    
    # Draw visualization
    cv.polylines(frame, [tag.corners.astype(int)], True, (0, 255, 0), 2)
    cv.circle(frame, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
    
    return np.array([x_cam, y_cam, theta_cam]), frame

def calculate_q_matrix():
    cam_params = (CAM_MATRIX[0,0], CAM_MATRIX[1,1], CAM_MATRIX[0,2], CAM_MATRIX[1,2])
    detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)
    
    cap = cv.VideoCapture(0)
    
    origin_raw = None
    errors_x, errors_y, errors_theta = [], [], []
    
    print("\n=== STARTING SENSOR CHARACTERIZATION ===")
    
    for i, truth in enumerate(GROUND_TRUTHS):
        x_true, y_true, theta_true = truth
        print(f"\nStep {i+1}/{len(GROUND_TRUTHS)}:")
        print(f"-> Place robot at True Pose: X={x_true:.2f}m, Y={y_true:.2f}m, Theta={math.degrees(theta_true):.0f}deg")
        print("-> Press SPACE to capture, ESC to quit.")
        
        captured = False
        while not captured:
            ret, frame = cap.read()
            if not ret: continue
            
            # Undistort frame
            h, w = frame.shape[:2]
            new_cam_mat, _ = cv.getOptimalNewCameraMatrix(CAM_MATRIX, DIST_COEFFS, (w,h), 1, (w,h))
            frame = cv.undistort(frame, CAM_MATRIX, DIST_COEFFS, None, new_cam_mat)
            
            raw_pose, debug_frame = get_raw_pose(detector, frame, cam_params)
            
            # Display Instructions
            cv.putText(debug_frame, f"Target: [{x_true:.2f}, {y_true:.2f}, {math.degrees(theta_true):.0f}deg]", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv.putText(debug_frame, "Press SPACE to record", (10, 60), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow("Variance Calibration", debug_frame)
            key = cv.waitKey(1)
            
            if key == 27: # ESC
                cap.release()
                cv.destroyAllWindows()
                return
                
            if key == 32 and raw_pose is not None: # SPACE
                if i == 0:
                    # Set the first measurement as the World Origin
                    origin_raw = raw_pose
                    x_w, y_w, theta_w = 0.0, 0.0, 0.0
                    print("   [+] Origin Calibrated!")
                else:
                    # Transform raw reading to World Frame based on Origin
                    dx = raw_pose[0] - origin_raw[0]
                    dy = raw_pose[1] - origin_raw[1]
                    th0 = origin_raw[2]
                    
                    x_w = dx * math.cos(-th0) - dy * math.sin(-th0)
                    y_w = dx * math.sin(-th0) + dy * math.cos(-th0)
                    theta_w = normalize_angle(raw_pose[2] - th0)
                
                # Calculate Error
                err_x = x_w - x_true
                err_y = y_w - y_true
                err_theta = normalize_angle(theta_w - theta_true)
                
                errors_x.append(err_x)
                errors_y.append(err_y)
                errors_theta.append(err_theta)
                
                print(f"   Recorded World Pose: [{x_w:.3f}, {y_w:.3f}, {math.degrees(theta_w):.1f}deg]")
                print(f"   Error:               [{err_x:.3f}, {err_y:.3f}, {math.degrees(err_theta):.1f}deg]")
                captured = True

    cap.release()
    cv.destroyAllWindows()

    # Calculate Sample Variance
    var_x = np.var(errors_x, ddof=1) if len(errors_x) > 1 else 0
    var_y = np.var(errors_y, ddof=1) if len(errors_y) > 1 else 0
    var_theta = np.var(errors_theta, ddof=1) if len(errors_theta) > 1 else 0
    
    print("\n" + "="*40)
    print(" EKF Q_t MATRIX VARIANCES READY ")
    print("="*40)
    print(f"var_x     = {var_x:.6f}")
    print(f"var_y     = {var_y:.6f}")
    print(f"var_theta = {var_theta:.6f}")
    print("\nPaste this into your get_Q() function:")
    print(f"return np.diag([{var_x:.6f}, {var_y:.6f}, {var_theta:.6f}])")

if __name__ == "__main__":
    calculate_q_matrix()