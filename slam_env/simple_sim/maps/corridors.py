# A long hallway with rooms on the side
start_pose = [0.5, 0.5, 0.0]

wall_corner_list = [
    # Main hallway
    [0.0, 0.0, 8.0, 0.0],
    [0.0, 1.0, 8.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [8.0, 0.0, 8.0, 1.0],
    # Room 1
    [2.0, 1.0, 2.0, 3.0], [4.0, 1.0, 4.0, 3.0], [2.0, 3.0, 4.0, 3.0],
    # Doorway 1 opening (replace part of the hallway wall)
    [2.5, 1.0, 3.5, 1.0], # Note: Raycaster will overlap, but visually it acts as a room
]