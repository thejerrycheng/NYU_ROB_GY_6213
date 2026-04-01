"""
L-shaped room map — good for testing around corners.
"""

L_SHAPED_ROOM = {
    "name": "l_shaped",
    "description": "An L-shaped room with an inner corner obstacle",
    "robot_start": [0.3, 0.2, 1.5708],
    "bounds": (-0.1, 5.1, -0.1, 4.1),
    "walls": [
        # Outer L-shape boundary
        (0.0, 0.0, 5.0, 0.0),   # Bottom
        (5.0, 0.0, 5.0, 2.0),   # Right-lower
        (5.0, 2.0, 3.0, 2.0),   # Inner step horizontal
        (3.0, 2.0, 3.0, 4.0),   # Inner step vertical
        (3.0, 4.0, 0.0, 4.0),   # Top
        (0.0, 4.0, 0.0, 0.0),   # Left
    ],
    "obstacles": []
}
