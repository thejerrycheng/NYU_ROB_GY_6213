start_pose = [1.0, 1.0, 1.57]

wall_corner_list = [
    # Outer Boundary
    [0, 0, 12, 0], [0, 0, 0, 12], [0, 12, 12, 12], [12, 12, 12, 0],
    
    # Main Central Hallway (Runs horizontal at y=5 to y=7)
    [0, 5, 10, 5], [2, 7, 12, 7],
    
    # Bottom Rooms (Offices)
    [3, 0, 3, 5], [6, 0, 6, 5], [9, 0, 9, 5],
    # Doorway cutouts (represented by shortening the vertical walls, wait, we need walls to be solid except doors)
    # Let's make the top walls of the offices have gaps
    [0, 2.5, 3, 2.5], [3, 2.5, 5, 2.5], [7, 2.5, 9, 2.5], [9, 2.5, 11, 2.5],
    
    # Top Rooms (Meeting rooms)
    [4, 7, 4, 12], [8, 7, 8, 12],
    [2, 9.5, 4, 9.5], [5, 9.5, 8, 9.5], [9, 9.5, 12, 9.5],
    
    # Center Island (Reception/Desks)
    [4, 5.5, 8, 5.5], [4, 6.5, 8, 6.5], [4, 5.5, 4, 6.5], [8, 5.5, 8, 6.5]
]