import math

start_pose = [0.0, 0.0, 0.0]
wall_corner_list = []

# Helper to generate polygons
def generate_polygon(cx, cy, radius, num_sides):
    walls = []
    angle_step = 2 * math.pi / num_sides
    for i in range(num_sides):
        x1 = cx + radius * math.cos(i * angle_step)
        y1 = cy + radius * math.sin(i * angle_step)
        x2 = cx + radius * math.cos((i + 1) * angle_step)
        y2 = cy + radius * math.sin((i + 1) * angle_step)
        walls.append([x1, y1, x2, y2])
    return walls

# 16-sided outer arena (radius 5m)
wall_corner_list.extend(generate_polygon(0, 0, 5.0, 16))

# Central circular pillar (radius 1m, 8-sided)
wall_corner_list.extend(generate_polygon(0, 0, 1.0, 8))

# Add a few internal walls to break symmetry
wall_corner_list.extend([
    [2.0, 2.0, 4.0, 2.0],
    [-2.0, -2.0, -2.0, -4.0]
])