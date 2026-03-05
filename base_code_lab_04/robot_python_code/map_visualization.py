import matplotlib.pyplot as plt

# Import your local parameters file
import parameters

def visualize_map():
    fig, ax = plt.subplots(figsize=(8, 8))

    print(f"Loaded {len(parameters.wall_corner_list)} walls from parameters.py")

    # Iterate through the wall corners and plot them
    for i, wall in enumerate(parameters.wall_corner_list):
        x1, y1, x2, y2 = wall
        
        # Plot the wall segment as a thick black line
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3)
        
        # Optional: Plot the corner points as blue dots for clarity
        ax.plot([x1, x2], [y1, y2], 'bo', markersize=4)

    # Formatting the plot
    ax.set_title("2D Map Visualization", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Distance (meters)", fontsize=12)
    ax.set_ylabel("Y Distance (meters)", fontsize=12)

    # CRITICAL: Ensure the aspect ratio is equal so 1m in X visually equals 1m in Y
    ax.set_aspect('equal', adjustable='box')

    # Add a grid for easier spatial reasoning
    ax.grid(True, linestyle='--', alpha=0.6)

    # Pad the axes slightly so walls don't hug the absolute edge of the window
    plt.margins(0.1)
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    visualize_map()