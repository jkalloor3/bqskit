import glob
import csv
import sys
import os

def read_distances(circ_name, tol):
    pattern = f"small_block_checkpoints_final_paper_4_*/{circ_name}_*_{tol}/*.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    
    max_tol = 10 * 10 ** (-tol)
    if tol == 1.0:
        max_ratio = 2
    elif tol == 2.0:
        max_ratio = 5
    else:
        max_ratio = 20

    num_dist_fails = 0
    num_ratio_fails = 0
    num_trials = 0

    for file in files:
        print(f"Reading: {file}")
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Assuming distances are in the first column, adjust if needed
                eps = float(row["Epsilon"])
                ratio = float(row["Ratio"])
                num_trials += 1
                if eps <= max_tol:
                    print(row["Epsilon"], row["Ratio"])
                    if ratio > max_ratio:
                        num_ratio_fails += 1
                else:
                    num_dist_fails += 1
                    exit(1)
    print(f"Number of Dist fails: {num_dist_fails} out of {num_trials}")
    print(f"Number of Ratio fails: {num_ratio_fails} out of {num_trials}")
    print(f"Total Successes: {num_trials - num_dist_fails - num_ratio_fails} out of {num_trials}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <circ_name> <tol>")
        sys.exit(1)
    circ_name = sys.argv[1]
    tol = float(sys.argv[2])
    read_distances(circ_name, tol)