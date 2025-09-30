from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path


form = "small_block_checkpoints_final_paper_4_clifft_tket/*/block_*/ensemble_final_probs_fw.npy"

if __name__ == '__main__':

    all_probs_files = glob(form)
    print(f"Found {len(all_probs_files)} probability files.")

    # Randomly choose SAMPLE_SIZE files
    SAMPLE_SIZE = 5000
    if len(all_probs_files) > SAMPLE_SIZE:
        all_probs_files = np.random.choice(all_probs_files, size=SAMPLE_SIZE, replace=False)

    all_reductions = []

    for probs_file in all_probs_files:
        # Get the corresponding CSV file 
        parent_dir = str(Path(probs_file).parent)
        # Parent is a dir is of form ../block_num/, and want ../block_num.csv
        csv_file = parent_dir + "_fw.csv"
        reader = csv.DictReader(open(csv_file))
        final_ratio = float("inf")

        # Read CSV and get ratio
        for row in reader:
            if 'Ratio' in row:
                final_ratio = min(final_ratio, float(row['Ratio']))

        tol_val = float(probs_file.split("/")[-3].split("_")[-1])
        min_ratio = min(20, 10 ** (tol_val) / 4)
        if final_ratio > min_ratio:
            continue

        # Get tol val
        tol_val = float(probs_file.split("/")[-3].split("_")[-1])
        zero_thresh = min(10 ** (tol_val * -2 - 1), 1e-5)
        # Load probs file
        probs = np.load(probs_file)
        # Calculate the number of probabilities below the threshold
        num_below_thresh = np.sum(probs <= zero_thresh)
        # print ratio of num_below_thresh to total number of probs
        total_probs = probs.shape[0] * probs.shape[1]
        ratio_below_thresh = num_below_thresh / total_probs

        if ratio_below_thresh > 0.8:
            print(csv_file, tol_val, ratio_below_thresh, probs.shape)
            exit(1)
        # print(tol_val, ratio_below_thresh, probs.shape)
        all_reductions.append(ratio_below_thresh)

    # Plot histogram of all reductions and save fig
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_reductions, bins=20, color='blue', alpha=0.7)
    ax.set_title('Histogram of Reductions in Probabilities Below Threshold - Cliff-T')
    ax.set_xlabel('Number of Probabilities Below Threshold')
    ax.set_ylabel('Frequency')
    fig.savefig('reductions_histogram_only_good.png')
