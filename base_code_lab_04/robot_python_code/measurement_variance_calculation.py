import csv
import numpy as np

# Change this to match the exact name of the file you generated
CSV_FILENAME = "lidar_characterization_filtered.csv" 

def main():
    ground_truths = []
    measured_dists = []
    biases = []

    print(f"Reading data from {CSV_FILENAME}...\n")

    try:
        with open(CSV_FILENAME, mode='r') as file:
            # DictReader automatically uses the first row as column names
            reader = csv.DictReader(file)
            for row in reader:
                # Handle varying header names from previous script iterations
                gt = float(row.get("Ground_Truth_m", 0))
                
                if "Filtered_Measured_Distance_m" in row:
                    meas = float(row["Filtered_Measured_Distance_m"])
                else:
                    meas = float(row["Measured_Distance_m"])
                    
                bias = float(row.get("Bias_m", meas - gt))

                ground_truths.append(gt)
                measured_dists.append(meas)
                biases.append(bias)
                
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILENAME}'. Please ensure the file is in the same directory.")
        return

    if not biases:
        print("No data found in the CSV.")
        return

    # 1. Calculate Overall Statistics
    # We calculate the variance of the errors (biases) using ddof=1 for sample variance
    biases_array = np.array(biases)
    
    overall_mean_bias = np.mean(biases_array)
    overall_variance = np.var(biases_array, ddof=1) if len(biases_array) > 1 else 0.0
    overall_std_dev = np.std(biases_array, ddof=1) if len(biases_array) > 1 else 0.0

    print("==========================================")
    print(" Lidar Measurement Model Statistics")
    print("==========================================")
    print(f"Total Samples Processed: {len(biases)}")
    print(f"Overall Mean Bias:       {overall_mean_bias:.6f} m")
    print(f"Overall Variance:        {overall_variance:.8f} m^2")
    print(f"Overall Std Deviation:   {overall_std_dev:.6f} m\n")

    # 2. Calculate Variance Grouped by Distance
    print("--- Breakdown by Distance ---")
    unique_dists = sorted(list(set(ground_truths)))
    
    for dist in unique_dists:
        # Extract all biases for this specific ground truth distance
        dist_biases = [b for g, b in zip(ground_truths, biases) if g == dist]
        
        if len(dist_biases) > 1:
            dist_var = np.var(dist_biases, ddof=1)
            dist_mean = np.mean(dist_biases)
            print(f"Distance: {dist:.2f}m | Samples: {len(dist_biases)} | Mean Bias: {dist_mean:.4f}m | Variance: {dist_var:.8f} m^2")
        else:
            # Variance requires at least 2 data points
            print(f"Distance: {dist:.2f}m | Samples: 1 | Mean Bias: {dist_biases[0]:.4f}m | Variance: N/A (need >1 sample)")

if __name__ == '__main__':
    main()