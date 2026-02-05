import cv2
import cv2.aruco as aruco
import numpy as np

# --- CONFIGURATION ---
TAG_ID = 0  # The ID number you want to generate (0, 1, 2, etc.)
TAG_SIZE_PX = 500  # Resolution of the image (pixels)
DICT_TYPE = aruco.DICT_APRILTAG_36h11 # Must match your detection script

# 1. Load the dictionary
aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)

# 2. Generate the marker
# The function returns the marker as a simple black/white numpy array
tag_image = np.zeros((TAG_SIZE_PX, TAG_SIZE_PX, 1), dtype="uint8")
aruco.generateImageMarker(aruco_dict, TAG_ID, TAG_SIZE_PX, tag_image, 1)

# 3. Save the file
filename = f"apriltag_36h11_id{TAG_ID}.png"
cv2.imwrite(filename, tag_image)

print(f"Generated {filename}")

# Optional: Show it
cv2.imshow("AprilTag", tag_image)
cv2.waitKey(0)
cv2.destroyAllWindows()