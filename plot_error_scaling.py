import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# List of circuits
circs = ["shor_12", "mult_16", "add17", "qae13", "qaoa10", "lgt_17", "draper_adder_12"]  # Replace with your list of circuits


MAX_DIVERSITY = False

if MAX_DIVERSITY:
    max_diversity_str = "_max_diversity"
else:
    max_diversity_str = ""

# Directory containing block checkpoints
block_checkpoints_dir = f'block_checkpoints_final_paper_clifft_tket{max_diversity_str}'
block_form = "{circ}_*/data.csv"

def mean_std_without_outliers_zscore(data, threshold=3):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Compute Z-scores
    z_scores = (data - mean) / std
    filtered_data = data[np.abs(z_scores) < threshold]  # Remove outliers

    if len(filtered_data) == 0:
        return mean, std
    
    return np.mean(filtered_data), np.std(filtered_data)


# Function to read data.csv from each folder
def read_data_from_folders(circuits, checkpoints_dir):
    all_data = {}
    for circ in circuits:
        all_data[circ] = {}
        csv_files = glob.glob(os.path.join(checkpoints_dir, block_form.format(circ=circ)))
        # print(csv_files)
        for csv_file in csv_files:
            folder = os.path.dirname(csv_file).split('/')[-1]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2])
            if block_num not in all_data[circ]:
                all_data[circ][block_num] = {}
            # df = pd.read_csv(csv_file)
            # Read CSV with csv
            with open(csv_file, 'r') as file:
                # Read the CSV file and return min
                # value of 'Norm. Ratio' column
                reader = csv.DictReader(file)
                final_ratio = float("inf")
                eps = 10 ** (-tol)
                for row in reader:
                    if "Ratio" in row:  # Check if the column value is not empty
                        # final_ratio = min(final_ratio, float(row["Norm. Ratio"]))
                        if float(row["Ratio"]) < final_ratio:
                            final_ratio = float(row["Ratio"])
                            # eps = float(row["Epsilon"])
                            float_eps = float(row["Epsilon"])
                            eps = -1 * np.log10(float_eps)
                            if eps > 1:
                                eps = int(eps)
                            if eps < 1:
                                if eps > 0.5:
                                    eps = 1
                                else:
                                    eps = 0.8
                if final_ratio < 1:
                    final_ratio = 1
                # if final_ratio > 10 and MAX_DIVERSITY:
                #     # Remove folder
                #     print(f"Removing folder {folder} due to high final ratio")
                #     os.system(f"rm -rf {os.path.join(checkpoints_dir, folder)}")
                #     continue
                # print("Final ratio:", final_ratio)
                # print(block_num, tol, eps, final_ratio)
                all_data[circ][block_num][eps] = final_ratio

    # Add some known points
    # all_data["shor_12"]["02"][4.0] = 900
    # all_data["shor_12"]["11"] = {4.0: 1600}
    # all_data["shor_12"]["13"][4.0] = 1300

    return all_data



if __name__ == '__main__':
    # Collect data from all folders
    all_data = read_data_from_folders(circs, block_checkpoints_dir)

    # print(all_data)

    # Combine all data frames into a single data frame (optional)
    # combined_data = pd.concat(data_frames, ignore_index=True)

    # Plotting
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    # Plot each circ separately
    for circ in circs:
        x = []
        y = []
        for block_num in sorted(all_data[circ].keys()):
            for eps in sorted(all_data[circ][block_num].keys()):
                x.append(eps)
                y.append(all_data[circ][block_num][eps])
        
        # For each eps, plot mean and std
        x = np.array(x)
        y = np.array(y)
        epss = np.unique(x)
        # print(epss)
        y_mean = []
        # y_std = []
        y_min = []
        y_max = []
        # print(x)
        for eps in epss:
            y_eps = y[x == eps]
            # # print(eps, y_eps)
            # if len(y_eps) > 1:
            #     m, s = mean_std_without_outliers_zscore(y_eps, 1.5)
            #     # m = np.mean(y_eps)
            #     # s = np.std(y_eps)
            #     print(circ, eps, m , s, y_eps)
            # else:
            #     m = y_eps[0]
            #     s = 0

            # s = max(s, 0.1)
            m = np.mean(y_eps)
            y_mean.append(m)
            y_max.append(np.max(y_eps))
            y_min.append(np.min(y_eps))
        y_mean = np.array(y_mean)
        # y_std = np.array(y_std)
        y_min = np.array(y_min)
        y_max = np.array(y_max)
        # axs.errorbar(x, y_mean, yerr=y_std, label=circ)
        axs.fill_between(epss, y_min, y_max, alpha=0.1)
        axs.plot(epss, y_mean, label=circ)


    axs.set_xlabel("Average Frobenius Distance", fontdict={"size": 24})
    axs.set_ylabel("Scaling Factor    ", fontdict={"size": 24})
    axs.set_yscale("log")
    axs.set_ybound(0, max(axs.get_ybound()[1], 10))
    axs.set_yticklabels(axs.get_yticks(), fontdict={"size": 16})
    axs.set_xticklabels(axs.get_xticks(), fontdict={"size": 16})
    axs.legend(loc='upper left', fontsize=14)

    # Save the figure
    # plt.tight_layout()
    fig.tight_layout()
    fig.savefig(f"error_scaling_cliff_t{max_diversity_str}.png", dpi=300)
        # axs.plot(x, y, label=circ)