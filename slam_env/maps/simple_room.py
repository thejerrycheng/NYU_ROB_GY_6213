"""
Simple rectangular room map.

Wall format: (x1, y1, x2, y2) — line segment endpoints in meters.
robot_start: (x, y, theta) — initial pose.
bounds: (x_min, x_max, y_min, y_max) — for rendering limits.
"""

SIMPLE_ROOM = {
    "name": "simple_room",
    "description": "A plain rectangular room 4m x 3m",
    "robot_start": [0.3, 0.2, 1.5708],   # x, y, theta
    "bounds": (-0.1, 4.1, -0.1, 3.1),
    "walls": [
        # Outer boundary (x1, y1, x2, y2)
        (0.0, 0.0, 4.0, 0.0),   # Bottom wall
        (4.0, 0.0, 4.0, 3.0),   # Right wall
        (4.0, 3.0, 0.0, 3.0),   # Top wall
        (0.0, 3.0, 0.0, 0.0),   # Left wall
    ],
    # MuJoCo box obstacles: (cx, cy, half_width, half_height, height)
    "obstacles": []
}
