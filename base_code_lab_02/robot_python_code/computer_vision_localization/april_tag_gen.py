import cv2
import cv2.aruco as aruco
import numpy as np

# --- CONFIGURATION ---
TAGS_TO_GENERATE = [0, 1, 2, 3, 4, 5]
DICT_TYPE = aruco.DICT_APRILTAG_36h11

# A4 Paper Dimensions at 300 DPI
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508

# Layout Settings
ROWS = 3
COLS = 2
TAG_SIZE_PX = 700       # Size of the black part of the tag
MARGIN_X = 400          # Horizontal space between tags
MARGIN_Y = 400          # Vertical space between tags
TEXT_OFFSET = 80        # Space between tag and label
FONT_SCALE = 3.0
THICKNESS = 5

# Cutting Guide Settings
CUT_PADDING = 60        # How much white space to leave inside the cut line
CUT_COLOR = 180         # Light Gray (0=Black, 255=White)
CUT_THICKNESS = 4       # Thin line for cutting

def generate_a4_sheet():
    aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
    
    # Create white A4 canvas
    sheet = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype="uint8") * 255

    # Calculate grid start position to center everything
    start_x = (A4_WIDTH_PX - (COLS * TAG_SIZE_PX) - ((COLS - 1) * MARGIN_X)) // 2
    start_y = (A4_HEIGHT_PX - (ROWS * TAG_SIZE_PX) - ((ROWS - 1) * MARGIN_Y)) // 2

    print(f"Generating sheet with IDs: {TAGS_TO_GENERATE}...")

    for index, tag_id in enumerate(TAGS_TO_GENERATE):
        if index >= ROWS * COLS: break

        # Calculate Grid Position
        r = index // COLS
        c = index % COLS
        
        # Top-Left corner of the TAG image
        x = start_x + c * (TAG_SIZE_PX + MARGIN_X)
        y = start_y + r * (TAG_SIZE_PX + MARGIN_Y)

        # 1. Generate and Paste Tag
        tag_img = np.zeros((TAG_SIZE_PX, TAG_SIZE_PX), dtype="uint8")
        aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX, tag_img, 1)
        sheet[y:y+TAG_SIZE_PX, x:x+TAG_SIZE_PX] = tag_img

        # 2. Add Label
        label = f"ID: {tag_id}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        
        text_x = x + (TAG_SIZE_PX - text_w) // 2
        text_y = y - TEXT_OFFSET
        
        cv2.putText(sheet, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 0, THICKNESS)

        # 3. Draw Cut Boarder (Dashed or Solid)
        # We define a box that includes the label area + tag area + padding
        
        # Top of the cut box (above the text)
        cut_top = text_y - text_h - CUT_PADDING
        # Bottom of the cut box (below the tag)
        cut_bottom = y + TAG_SIZE_PX + CUT_PADDING
        # Left of the cut box
        cut_left = x - CUT_PADDING
        # Right of the cut box
        cut_right = x + TAG_SIZE_PX + CUT_PADDING

        # Draw the rectangle (Solid Light Gray)
        cv2.rectangle(sheet, (cut_left, cut_top), (cut_right, cut_bottom), CUT_COLOR, CUT_THICKNESS)

    # Save
    filename = "AprilTag_A4_With_CutLines.png"
    cv2.imwrite(filename, sheet)
    print(f"Success! Saved to {filename}")
    
    # Preview
    preview = cv2.resize(sheet, (0,0), fx=0.25, fy=0.25)
    cv2.imshow("Preview", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_a4_sheet()