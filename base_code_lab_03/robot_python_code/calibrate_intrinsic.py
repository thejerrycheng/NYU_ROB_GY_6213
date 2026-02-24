import numpy as np
import cv2 as cv
import time

def live_calibrate_camera(rows, cols, square_size, camera_index=0):
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(cols-1,rows-1,0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size # Scale by actual physical square size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return None, None

    print("\n" + "="*50)
    print(" LIVE CAMERA CALIBRATION")
    print("="*50)
    print("-> Move the checkerboard around the frame.")
    print("-> It will auto-capture when detected (max 1 per second).")
    print("-> Press ENTER to finish and calculate intrinsics.")
    print("-> Press ESC to cancel without calculating.")
    print("="*50 + "\n")

    captured_frames = 0
    last_capture_time = 0
    capture_delay = 1.0  # Wait 1 second between auto-captures
    gray_shape = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        # Find the chess board corners
        ret_corners, corners = cv.findChessboardCorners(gray, (cols, rows), None)

        display_img = img.copy()

        if ret_corners:
            # Draw the corners to show it sees the board
            cv.drawChessboardCorners(display_img, (cols, rows), corners, ret_corners)
            
            current_time = time.time()
            if current_time - last_capture_time > capture_delay:
                # Refine corners and save
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                captured_frames += 1
                last_capture_time = current_time
                
                # Visual flash effect to indicate capture
                cv.rectangle(display_img, (0,0), (img.shape[1], img.shape[0]), (0, 255, 0), 10)

        # Overlay HUD
        cv.putText(display_img, f"Captured: {captured_frames}", (20, 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(display_img, "Press ENTER to compute | ESC to cancel", (20, 80), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv.imshow('Live Calibration', display_img)
        
        key = cv.waitKey(1)
        if key == 13:  # ENTER key
            break
        elif key == 27:  # ESC key
            cap.release()
            cv.destroyAllWindows()
            print("Calibration cancelled.")
            return None, None

    cap.release()
    cv.destroyAllWindows()

    if captured_frames < 5:
        print(f"Not enough frames captured ({captured_frames}). Need at least 5.")
        return None, None

    print(f"\nCalculating camera matrix using {captured_frames} frames...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    print("\n--- Calibration Results ---")
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    
    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\nTotal Reprojection Error: {mean_error/len(objpoints):.4f} px (Closer to 0 is better)")

    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    # USER CONFIGURATION
    CHESSBOARD_COLS = 10
    CHESSBOARD_ROWS = 7
    SQUARE_SIZE_METERS = 0.025 
    CAMERA_ID = 0

    live_calibrate_camera(CHESSBOARD_ROWS, CHESSBOARD_COLS, SQUARE_SIZE_METERS, CAMERA_ID)