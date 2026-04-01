"""
Simple maze map — multiple corridors and rooms.
"""

MAZE_MAP = {
    "name": "maze",
    "description": "A small maze with corridors and rooms",
    "robot_start": [0.3, 0.3, 0.0],
    "bounds": (-0.1, 6.1, -0.1, 6.1),
    "walls": [
        # Outer boundary
        (0.0, 0.0, 6.0, 0.0),
        (6.0, 0.0, 6.0, 6.0),
        (6.0, 6.0, 0.0, 6.0),
        (0.0, 6.0, 0.0, 0.0),
        # Inner walls forming corridors
        (2.0, 0.0, 2.0, 2.0),   # Vertical divider bottom-left
        (2.0, 3.0, 2.0, 6.0),   # Vertical divider top-left
        (4.0, 0.0, 4.0, 1.0),   # Vertical divider bottom-right
        (4.0, 2.0, 4.0, 4.0),   # Vertical divider mid-right
        (4.0, 5.0, 4.0, 6.0),   # Vertical divider top-right
        (1.0, 3.0, 3.0, 3.0),   # Horizontal mid corridor
        (3.0, 4.0, 6.0, 4.0),   # Horizontal upper corridor
    ],
    "obstacles": [
        # (cx, cy, half_w, half_h, height)
        (5.0, 2.0, 0.3, 0.3, 0.3),
    ]
}
