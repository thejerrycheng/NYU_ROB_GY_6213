# import cv2
# import cv2.aruco as aruco
# import numpy as np

# # --- CONFIGURATION ---
# TAGS_TO_GENERATE = [0, 1, 2, 3, 4, 5]
# DICT_TYPE = aruco.DICT_APRILTAG_36h11

# # A4 Paper Dimensions at 300 DPI
# A4_WIDTH_PX = 2480
# A4_HEIGHT_PX = 3508

# # Layout Settings
# ROWS = 3
# COLS = 2
# TAG_SIZE_PX = 700       # Size of the black part of the tag
# MARGIN_X = 400          # Horizontal space between tags
# MARGIN_Y = 400          # Vertical space between tags
# TEXT_OFFSET = 80        # Space between tag and label
# FONT_SCALE = 3.0
# THICKNESS = 5

# # Cutting Guide Settings
# CUT_PADDING = 60        # How much white space to leave inside the cut line
# CUT_COLOR = 180         # Light Gray (0=Black, 255=White)
# CUT_THICKNESS = 4       # Thin line for cutting

# def generate_a4_sheet():
#     aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
    
#     # Create white A4 canvas
#     sheet = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype="uint8") * 255

#     # Calculate grid start position to center everything
#     start_x = (A4_WIDTH_PX - (COLS * TAG_SIZE_PX) - ((COLS - 1) * MARGIN_X)) // 2
#     start_y = (A4_HEIGHT_PX - (ROWS * TAG_SIZE_PX) - ((ROWS - 1) * MARGIN_Y)) // 2

#     print(f"Generating sheet with IDs: {TAGS_TO_GENERATE}...")

#     for index, tag_id in enumerate(TAGS_TO_GENERATE):
#         if index >= ROWS * COLS: break

#         # Calculate Grid Position
#         r = index // COLS
#         c = index % COLS
        
#         # Top-Left corner of the TAG image
#         x = start_x + c * (TAG_SIZE_PX + MARGIN_X)
#         y = start_y + r * (TAG_SIZE_PX + MARGIN_Y)

#         # 1. Generate and Paste Tag
#         tag_img = np.zeros((TAG_SIZE_PX, TAG_SIZE_PX), dtype="uint8")
#         aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX, tag_img, 1)
#         sheet[y:y+TAG_SIZE_PX, x:x+TAG_SIZE_PX] = tag_img

#         # 2. Add Label
#         label = f"ID: {tag_id}"
#         (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        
#         text_x = x + (TAG_SIZE_PX - text_w) // 2
#         text_y = y - TEXT_OFFSET
        
#         cv2.putText(sheet, label, (text_x, text_y), 
#                     cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 0, THICKNESS)

#         # 3. Draw Cut Boarder (Dashed or Solid)
#         # We define a box that includes the label area + tag area + padding
        
#         # Top of the cut box (above the text)
#         cut_top = text_y - text_h - CUT_PADDING
#         # Bottom of the cut box (below the tag)
#         cut_bottom = y + TAG_SIZE_PX + CUT_PADDING
#         # Left of the cut box
#         cut_left = x - CUT_PADDING
#         # Right of the cut box
#         cut_right = x + TAG_SIZE_PX + CUT_PADDING

#         # Draw the rectangle (Solid Light Gray)
#         cv2.rectangle(sheet, (cut_left, cut_top), (cut_right, cut_bottom), CUT_COLOR, CUT_THICKNESS)

#     # Save
#     filename = "AprilTag_A4_With_CutLines.png"
#     cv2.imwrite(filename, sheet)
#     print(f"Success! Saved to {filename}")
    
#     # Preview
#     preview = cv2.resize(sheet, (0,0), fx=0.25, fy=0.25)
#     cv2.imshow("Preview", preview)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     generate_a4_sheet()


import cv2
import cv2.aruco as aruco
import numpy as np

# --- 1. Canvas & Print Settings (300 DPI) ---
DPI = 300
PPM = DPI / 25.4  # Pixels per millimeter (~11.811)

A4_WIDTH_PX = int(210 * PPM)   # 2480 pixels
A4_HEIGHT_PX = int(297 * PPM)  # 3508 pixels

# --- 2. Tag Physical Dimensions ---
TAG_SIZE_MM = 100
TAG_SIZE_PX = int(TAG_SIZE_MM * PPM)

# The white padding around the black square (Quiet Zone)
# AprilTags require at least 1 bit of white space. For a 100mm tag with 10x10 bits, 1 bit = 10mm.
QUIET_ZONE_MM = 10
QUIET_ZONE_PX = int(QUIET_ZONE_MM * PPM)

def generate_a4_tags():
    # Create a blank white A4 canvas
    canvas = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255
    
    # Initialize the tag36h11 dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    
    # We will generate ID 0 and ID 1
    tag_ids = [0, 1]
    
    # Vertical placement centers for the two tags on the A4 page
    y_centers = [A4_HEIGHT_PX // 4, (A4_HEIGHT_PX // 4) * 3]
    
    for i, tag_id in enumerate(tag_ids):
        # 1. Generate the raw black-and-white tag
        tag_img = aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX)
        
        # 2. Add the required white quiet zone
        padded_tag = cv2.copyMakeBorder(tag_img, 
                                        QUIET_ZONE_PX, QUIET_ZONE_PX, 
                                        QUIET_ZONE_PX, QUIET_ZONE_PX, 
                                        cv2.BORDER_CONSTANT, value=255)
        
        # 3. Add the grey cut-off border
        border_thickness = 3 # 3 pixels thick so it's visible on paper
        cut_ready_tag = cv2.copyMakeBorder(padded_tag, 
                                           border_thickness, border_thickness, 
                                           border_thickness, border_thickness, 
                                           cv2.BORDER_CONSTANT, value=150)
        
        # 4. Paste onto the A4 canvas
        h, w = cut_ready_tag.shape
        start_x = (A4_WIDTH_PX - w) // 2
        start_y = y_centers[i] - (h // 2)
        
        canvas[start_y:start_y+h, start_x:start_x+w] = cut_ready_tag
        
        # 5. Label the tag ID on the paper (above the cut line)
        label = f"tag36h11 - ID: {tag_id} (100mm)"
        cv2.putText(canvas, label, (start_x, start_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

    # Save the final A4 sheet
    output_filename = "apriltags_36h11_100mm_A4.png"
    cv2.imwrite(output_filename, canvas)
    print(f"Success! Saved printable sheet as: {output_filename}")

if __name__ == "__main__":
    generate_a4_tags()