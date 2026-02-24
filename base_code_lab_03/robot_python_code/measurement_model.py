import cv2 as cv
import numpy as np
import math
import time
from pupil_apriltags import Detector

class AprilTagPlanarLocalization:
    def __init__(self, camera_matrix, dist_coeffs, tag_size_meters, z_offset_meters=0.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size_meters
        
        # The Z-offset is the height of the tag from the floor.
        # Since pupil_apriltags calculates the true 3D metric position (X, Y, Z) 
        # relative to the lens, the planar (X, Y) coordinates are naturally scale-accurate 
        # regardless of how close the tag is to the camera, assuming the camera is looking straight down.
        self.z_offset = z_offset_meters 
        
        self.camera_params = (
            camera_matrix[0, 0], # fx
            camera_matrix[1, 1], # fy
            camera_matrix[0, 2], # cx
            camera_matrix[1, 2]  # cy
        )
        
        self.detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)
        
        # World Frame Origin
        self.origin_x = None
        self.origin_y = None
        self.origin_theta = None
        self.is_calibrated = False

    def normalize_angle(self, angle):
        """Keeps the angle wrapped between -PI and PI."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _get_raw_camera_pose(self, frame):
        """Extracts the raw X, Y, and Theta directly from the camera frame."""
        h, w = frame.shape[:2]
        new_cam_mat, _ = cv.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        undistorted = cv.cv.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_cam_mat)
        
        gray = cv.cvtColor(undistorted, cv.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size)
        
        if not tags:
            return None, undistorted
            
        tag = tags[0]
        
        # Extract Planar Translation (Camera X and Camera Y)
        # We ignore tvec[2] (Depth/Z) to strictly enforce planar 2D motion
        x_cam = tag.pose_t[0][0]
        y_cam = tag.pose_t[1][0]
        
        # Extract Yaw (Rotation around the Z-axis)
        R = tag.pose_R
        # Standard CV yaw extraction. The negative sign reverses the direction 
        # to ensure standard counter-clockwise = positive rotation mapping.
        theta_cam = -math.atan2(R[1,0], R[0,0]) 
        
        # Draw visualization
        for i in range(4):
            pt1 = tuple(tag.corners[i].astype(int))
            pt2 = tuple(tag.corners[(i+1)%4].astype(int))
            cv.line(undistorted, pt1, pt2, (0, 255, 0), 2)
        cv.circle(undistorted, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
        
        return np.array([x_cam, y_cam, theta_cam]), undistorted

    def calibrate_origin(self, cap, num_frames_to_average=20):
        """
        Runs a calibration loop. Place the robot at World (0,0) facing World +X.
        It averages multiple frames to get a noise-free initial transform.
        """
        print("\n--- STARTING CALIBRATION ---")
        print("Please place the robot at the desired (0,0) position.")
        print("Ensure the robot is facing your desired World +X direction.")
        print("Waiting for stable tag detection...\n")
        
        collected_poses = []
        
        while len(collected_poses) < num_frames_to_average:
            ret, frame = cap.read()
            if not ret: continue
            
            pose, debug_frame = self._get_raw_camera_pose(frame)
            
            if pose is not None:
                collected_poses.append(pose)
                cv.putText(debug_frame, f"Calibrating: {len(collected_poses)}/{num_frames_to_average}", 
                           (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                           
            cv.imshow("Calibration", debug_frame)
            cv.waitKey(30)
            
        # Average the raw readings to set the world origin
        poses_array = np.array(collected_poses)
        self.origin_x = np.mean(poses_array[:, 0])
        self.origin_y = np.mean(poses_array[:, 1])
        
        # Average angles requires care due to wrapping, but for a stationary robot, simple mean is usually fine
        self.origin_theta = np.mean(poses_array[:, 2])
        self.is_calibrated = True
        
        print("\n--- CALIBRATION COMPLETE ---")
        print(f"Origin X offset: {self.origin_x:.3f} m")
        print(f"Origin Y offset: {self.origin_y:.3f} m")
        print(f"Coordinate Rotation offset: {math.degrees(self.origin_theta):.1f} deg\n")

    def get_world_pose(self, frame):
        """
        Returns the robot's pose in the calibrated World Frame.
        """
        if not self.is_calibrated:
            raise ValueError("Must call calibrate_origin() before getting world poses.")
            
        raw_pose, debug_frame = self._get_raw_camera_pose(frame)
        if raw_pose is None:
            return None, debug_frame
            
        x_cam, y_cam, theta_cam = raw_pose
        
        # 1. Translate point so the initial position is at (0,0)
        dx = x_cam - self.origin_x
        dy = y_cam - self.origin_y
        
        # 2. Rotate the coordinates to align with the World +X axis
        # We rotate by negative origin_theta to cancel out the initial camera tilt
        cos_theta = math.cos(-self.origin_theta)
        sin_theta = math.sin(-self.origin_theta)
        
        x_world = dx * cos_theta - dy * sin_theta
        y_world = dx * sin_theta + dy * cos_theta
        
        # 3. Calculate relative world heading
        # This handles the "forward is tag's up" inherently because the initial 
        # tag orientation is subtracted out, leaving purely the robot's relative turn.
        theta_world = self.normalize_angle(theta_cam - self.origin_theta)
        
        return np.array([x_world, y_world, theta_world]), debug_frame

if __name__ == "__main__":
    # USER CONFIG: Replace with your actual calibration matrices
    CAM_MATRIX = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]])
    DIST_COEFFS = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    TAG_SIZE_M = 0.05 
    TAG_Z_OFFSET_M = 0.12 # Height of the tag off the ground
    
    # Initialize 
    localization = AprilTagPlanarLocalization(CAM_MATRIX, DIST_COEFFS, TAG_SIZE_M, TAG_Z_OFFSET_M)
    cap = cv.VideoCapture(0)
    
    # Run Calibration Sequence First
    localization.calibrate_origin(cap)
    
    # Main Measurement Loop
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        world_pose, debug_frame = localization.get_world_pose(frame)
        
        if world_pose is not None:
            x, y, theta = world_pose
            text = f"X: {x:.3f}m | Y: {y:.3f}m | Yaw: {math.degrees(theta):.1f} deg"
            cv.putText(debug_frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
        cv.imshow("World Frame Tracking", debug_frame)
        if cv.waitKey(1) == 27: # ESC
            break
            
    cap.release()
    cv.destroyAllWindows()