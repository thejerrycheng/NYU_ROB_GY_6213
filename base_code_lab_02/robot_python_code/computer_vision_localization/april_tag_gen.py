import cv2
import cv2.aruco as aruco
import numpy as np

# --- CONFIGURATION ---
TAGS_TO_GENERATE = [0, 1, 2, 3, 4, 5]  # IDs for the 6 tags
DICT_TYPE = aruco.DICT_APRILTAG_36h11  # Standard robust family

# A4 Paper Dimensions at 300 DPI
A4_WIDTH_PX = 2480
A4_HEIGHT_PX = 3508

# Layout Settings
ROWS = 3
COLS = 2
TAG_SIZE_PX = 600       # Size of the black part of the tag
MARGIN_X = 200          # Horizontal space between tags
MARGIN_Y = 250          # Vertical space between tags
TEXT_OFFSET = 80        # Space between tag and label
FONT_SCALE = 3.0
THICKNESS = 5

def generate_a4_sheet():
    # 1. Load Dictionary
    aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)

    # 2. Create a blank white A4 canvas (255 = White)
    sheet = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype="uint8") * 255

    # 3. Calculate grid spacing
    # This centers the grid on the page
    start_x = (A4_WIDTH_PX - (COLS * TAG_SIZE_PX) - ((COLS - 1) * MARGIN_X)) // 2
    start_y = (A4_HEIGHT_PX - (ROWS * TAG_SIZE_PX) - ((ROWS - 1) * MARGIN_Y)) // 2

    print(f"Generating sheet with IDs: {TAGS_TO_GENERATE}...")

    for index, tag_id in enumerate(TAGS_TO_GENERATE):
        if index >= ROWS * COLS:
            break # Stop if we exceed the grid

        # Calculate Row and Col index
        r = index // COLS
        c = index % COLS

        # Top-left corner for this specific tag
        x = start_x + c * (TAG_SIZE_PX + MARGIN_X)
        y = start_y + r * (TAG_SIZE_PX + MARGIN_Y)

        # A. Generate the raw marker (1 bit per pixel)
        # We generate it small (1x1 pixels per bit) then resize to ensure sharp edges
        # 36h11 is 8x8 bits (including border), so we generate essentially "1 pixel per bit"
        # but the function requires a size in pixels. Let's let OpenCV generate it 
        # directly at the target size to avoid aliasing issues.
        tag_img = np.zeros((TAG_SIZE_PX, TAG_SIZE_PX), dtype="uint8")
        aruco.generateImageMarker(aruco_dict, tag_id, TAG_SIZE_PX, tag_img, 1)

        # B. Paste tag onto the white sheet
        sheet[y:y+TAG_SIZE_PX, x:x+TAG_SIZE_PX] = tag_img

        # C. Add Label (ID Number)
        label = f"ID: {tag_id}"
        
        # Calculate text size to center it
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        text_x = x + (TAG_SIZE_PX - text_w) // 2
        text_y = y - TEXT_OFFSET # Position above the tag

        # Draw text (Black color = 0)
        cv2.putText(sheet, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 0, THICKNESS)
        
        # D. Draw a cutting guide (optional thin gray box) - Helps if you cut them out
        # cv2.rectangle(sheet, (x-20, y-150), (x+TAG_SIZE_PX+20, y+TAG_SIZE_PX+20), 200, 2)

    # 4. Save the result
    filename = "AprilTag_A4_Sheet.png"
    cv2.imwrite(filename, sheet)
    print(f"Success! Saved to {filename}")
    
    # Resize for preview (A4 is too big for screen)
    preview = cv2.resize(sheet, (0,0), fx=0.25, fy=0.25)
    cv2.imshow("Preview", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_a4_sheet()