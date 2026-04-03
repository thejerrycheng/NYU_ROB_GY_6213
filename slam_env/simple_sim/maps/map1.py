start_pose = [0.5, 0.5, 1.57]

wall_corner_list = [
    # Outer Boundary
    [0, 0, 10, 0], [0, 0, 0, 10], [0, 10, 10, 10], [10, 10, 10, 0],
    
    # Inner Labyrinth Walls
    [2, 0, 2, 8], [4, 10, 4, 2], [6, 0, 6, 8], [8, 10, 8, 2], # Vertical snake
    [0, 2, 1, 2], [3, 2, 4, 2], [8, 8, 9, 8], # Dead end traps
    [2, 4, 3, 4], [4, 6, 5, 6], [6, 4, 7, 4], # Horizontal blockers
    [1, 6, 2, 6], [8, 4, 9, 4]
]