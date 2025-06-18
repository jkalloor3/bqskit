import os
import csv
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from util import load_block, GateCounter, load_avg_ensemble_counts_full, get_block_names
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.ir import Circuit
from bqskit.ir.gates import CircuitGate, CNOTGate, QFTGate

# List of circuits
# circs = ["shor_12", "qft_16", "draper_adder_12", "qae13", "qpe_14", "lgt_17"]  # Replace with your list of circuits
# circs = ["lgt_17", "mult16", "add17", "LiH", "qpe_14"] 
circs = ["add17", "shor_12_no_qft", "lgt_17", "qae13", "qpe_14", "QITE_8_0", "LiH", "mult16", "draper_adder_12", "qae11"]

from matplotlib.patches import Patch


circ_labels = {
    "shor_12_w_qft": "Shor Subcircuit - 12q",
    "shor_12_no_qft": "Shor Subcircuit - 12q",
    "lgt_17": "Lattice Simulation - 17q",
    "qae11": "QAE - 11q",
    "qae13": "QAE - 13q",
    "qpe_14": "QPE - 14q",
    "QITE_8_0": "QITE - 8q",
    "LiH": "LiH Simulation - 8q",
    "mult16": "Multiplier - 16q",
    "add17": "Adder - 17q",
    "draper_adder_12": "Draper Adder - 12",
}
# circs = [f"QITE_8_{i}" for i in range(7)]  # For QITE_8 circuits
# circs = ["shor_12_w_qft", "shor_12_no_qft"]
# circs = ["LiH"]

# Directory containing block checkpoints
block_checkpoints_dir = f'block_checkpoints_final_paper_tket_4*'
small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_more_cx_tket"
small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_tket"
# block_form = "{circ}_*/data.csv"
block_csv_form = "{circ}_*/block_*.csv"
block_qasms_form = "{circ}_*/block_*/ensemble_final.qasms"
cache_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_cache_0.pkl"
jiggle_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_final_jiggle.npy"

cx_counter = GateCounter(est=True)
# Function to read data.csv from each folder
def read_cx_data_from_folders(circuits, orig_cx_counts, 
                              checkpoints_dir, small_block = False,
                              cliff_t: bool = False):
    all_data = {}
    if checkpoints_dir is None:
        return all_data
    for circ_name in circuits:
        all_data[circ_name] = {}
        qasms_file_form = os.path.join(checkpoints_dir, block_qasms_form.format(circ=circ_name))
        qasms_files = glob.glob(qasms_file_form)
        for qasms_file in qasms_files:
            folder = os.path.dirname(qasms_file).split('/')[-2]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2]) 
            folder = os.path.dirname(qasms_file).split('/')[-1]
            small_block_num = folder.split("_")[-1]
            if block_num not in all_data[circ_name]:
                all_data[circ_name][block_num] = {}
            if cliff_t:
                cache_file = os.path.join(checkpoints_dir, 
                                            cache_form.format(circ=circ_name, 
                                                            large_block_num=block_num, 
                                                            tol=tol, 
                                                            small_block_num=small_block_num))
                jiggle_file = os.path.join(checkpoints_dir,
                                            jiggle_form.format(circ=circ_name,
                                                                large_block_num=block_num, 
                                                                tol=tol, 
                                                                small_block_num=small_block_num))
            else:
                jiggle_file = None
                cache_file = None

            avg_count = load_avg_ensemble_counts_full(
                qasms_file, jiggle_file=jiggle_file, cache_file=cache_file,
                target_error=(10 ** (-tol)),count_t =cliff_t,
            )

            if small_block:
                if cliff_t:
                    orig_count = orig_cx_counts[circ_name][block_num][small_block_num][tol]
                else:
                    try:
                        orig_count = orig_cx_counts[circ_name][block_num][small_block_num]
                    except KeyError:
                        print(f"KeyError: {circ_name}, {block_num}, {small_block_num}")
                        print(list(orig_cx_counts[circ_name].keys()))
                        exit(1)
                diff = orig_count - avg_count
                if small_block_num not in all_data[circ_name][block_num]:
                    all_data[circ_name][block_num][small_block_num] = {}
                all_data[circ_name][block_num][small_block_num][tol] = diff
            else:
                if cliff_t:
                    orig_count = orig_cx_counts[circ_name][block_num][tol]
                else:
                    orig_count = orig_cx_counts[circ_name][block_num]
                diff = orig_count - avg_count
                all_data[circ_name][block_num][tol] = diff
    return all_data


# Function to read data.csv from each folder
def read_data_from_folders(circuits, checkpoints_dir, small_block = False):
    all_data = {}
    if checkpoints_dir is None:
        return all_data
    for circ in circuits:
        all_data[circ] = {}
        csv_file_form = os.path.join(checkpoints_dir, block_csv_form.format(circ=circ))
        csv_files = glob.glob(csv_file_form)
        # print(csv_files)
        for csv_file in csv_files:
            folder = os.path.dirname(csv_file).split('/')[-1]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2])
            if block_num not in all_data[circ]:
                all_data[circ][block_num] = {}
            # Read CSV with csv
            with open(csv_file, 'r') as file:
                # Read the CSV file and return min
                # value of 'Norm. Ratio' column
                reader = csv.DictReader(file)
                eps = tol
                final_ratio =  float("inf")
                for row in reader:
                    if "Ratio" in row:  # Check if the column value is not empty
                        # final_ratio = min(final_ratio, float(row["Norm. Ratio"]))
                        if float(row["Ratio"]) < final_ratio:
                            actual_eps = float(row["Norm. Epsilon"])
                            if actual_eps > 100 * (10 ** (-tol)):
                                continue
                            final_ratio = min(final_ratio, float(row["Ratio"]))
                if final_ratio < 1:
                    # For visual fidelity, we set this to 1. We can arbitrarily
                    # increase the final ratio by adding noise to the final
                    # circuits
                    final_ratio = 1

                if final_ratio > 1e5:
                    # Just ignore this data point, will not use this block at all
                    continue
                
                if small_block:
                    small_block_num = os.path.basename(csv_file).split(".")[0].split("_")[-1]
                    if small_block_num not in all_data[circ][block_num]:
                        all_data[circ][block_num][small_block_num] = {}
                    all_data[circ][block_num][small_block_num][eps] = final_ratio
                else:
                    all_data[circ][block_num][eps] = final_ratio

    return all_data

def plot_line_bound(data: dict, axs: plt.Axes, color: str, 
                    use_small_block: bool =False):
    x = []
    y = []
    for block_num in sorted(data.keys()):
        if use_small_block:
            for small_block_num in sorted(data[block_num].keys()):
                for eps in sorted(data[block_num][small_block_num].keys()):
                    x.append(eps)
                    y.append(data[block_num][small_block_num][eps])
        else:
            for eps in sorted(data[block_num].keys()):
                x.append(eps)
                y.append(data[block_num][eps])
    
    # For each eps, plot mean and std
    x = np.array(x)
    y = np.array(y)
    epss = np.unique(x)
    # Get rid of all values greater than 5 in epss
    epss = epss[epss <= 5]
    y_mean = []
    y_maxs= []
    y_mins = []
    for eps in epss:
        y_eps = y[x == eps]
        m = np.median(y_eps)
        y_mean.append(m)
        y_maxs.append(np.max(y_eps))
        y_mins.append(np.min(y_eps))
    y_mean = np.array(y_mean)
    y_maxs = np.array(y_maxs)
    y_mins = np.array(y_mins)
    axs.fill_between(epss, y_mins, y_maxs, alpha=0.1, color=color)
    axs.plot(epss, y_mean, label=circ, color=color, marker='*', markersize=5)

def plot_violin_plot(data: dict, axs: plt.Axes, color: str,
                     use_small_block: bool =False, eps_shift: float = 0,
                     label: str = ""):
    x = []
    y = []
    for block_num in sorted(data.keys()):
        if use_small_block:
            for small_block_num in sorted(data[block_num].keys()):
                for eps in sorted(data[block_num][small_block_num].keys()):
                    x.append(eps)
                    y.append(data[block_num][small_block_num][eps])
        else:
            for eps in sorted(data[block_num].keys()):
                x.append(eps)
                y.append(data[block_num][eps])
    x = np.array(x)
    y = np.array(y)
    epss = np.unique(x)
    epss = epss[epss <= 5]
    for eps in epss:
        y_eps = y[x == eps]
        parts = axs.violinplot(y_eps, [eps + eps_shift], showmeans=True,
                       points=100, widths=0.1, bw_method=0.1)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.4)

        for partname in ['cmins', 'cmaxes', 'cbars', 'cmeans']:
            vp = parts.get(partname)
            if vp:
                vp.set_edgecolor(color)
    
    legend_patch = Patch(facecolor=color, edgecolor=color, alpha=0.4, label=label)
    return legend_patch
    # axs.legend(handles=[legend_patch])

def output_csv(data: dict, file_name: str, cliff_t: bool = False):
    # Create a DataFrame from the data
    # Step 1: Collect all unique eps values and sort them
    all_eps = set()
    for block_data in data.values():
        for small_block_data in block_data.values():
            for eps_data in small_block_data.values():
                all_eps.update(eps_data.keys())

    sorted_eps = sorted(all_eps, key=float)

    # Step 2: Build rows
    rows = []
    for circ, block_data in data.items():
        total_diffs = {eps: 0 for eps in sorted_eps}  # Pre-fill with zeros

        for block_num, small_block_data in block_data.items():
            for small_block_num, eps_data in small_block_data.items():
                for eps, value in eps_data.items():
                    if value > 0:
                        total_diffs[eps] += value

        row = [circ] + [total_diffs[eps] for eps in sorted_eps]
        rows.append(row)

    eps_labels = [f"10e-{int(eps * 2)}" for eps in sorted_eps]
    # Step 3: Write to CSV
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Circuit'] + eps_labels)  # Header
        writer.writerows(rows)
    print(f"Data saved to {file_name}")


def get_orig_counts(circuits: list[str], cliff_t: bool = False, 
                    compiler: Compiler = None) -> dict:
    orig_cx_counts = {}

    cliff_t_text = "_cliff_t" if cliff_t else ""
    save_file = f"orig_counts{cliff_t_text}.pickle"

    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            orig_cx_counts = pickle.load(f)
            print(orig_cx_counts["add17"]["03"]["0"])
        return orig_cx_counts

    workflow = [
        ScanPartitioner(4),
    ]

    for circ_name in circuits:
        orig_cx_counts[circ_name] = {}
        for block_num in get_block_names(circ_name=circ_name, extra="_tket"):
            orig_cx_counts[circ_name][block_num] = {}
            tket_file = load_block(circ_name=circ_name, block_num=block_num,
                        extra="_tket")
            circ = Circuit.from_file(tket_file)
            if circ.num_operations == 0:
                orig_cx_counts[circ_name][block_num]["0"] = 0
                continue
            try:
                out_circ: Circuit = compiler.compile(circ.copy(), workflow)
            except Exception as e:
                print(f"Error compiling {circ_name}, block {block_num}: {e}")
                print(out_circ.gate_counts)
                print(circ.gate_counts)
                exit(1)
            num_digits = len(str(out_circ.num_operations))
            for i, (_, op) in enumerate(out_circ.operations_with_cycles()):
                assert isinstance(op.gate, CircuitGate)
                # Check if checkpoint exists:
                # Need to zero pad block ids for consistency
                small_block_num = str(i).zfill(num_digits)
                if cliff_t:
                    orig_cx_counts[circ_name][block_num][small_block_num] = {}
                    t_counter = GateCounter(est = False)
                    for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                        orig_cx_counts[circ_name][block_num][small_block_num][err] = t_counter.count_t(op.gate._circuit, 10 ** (-err))
                else:
                    # print(f"Counting CNOTs for {circ_name}, block {block_num}, small block {small_block_num}", op.gate._circuit.count(CNOTGate()))
                    orig_cx_counts[circ_name][block_num][small_block_num] = op.gate._circuit.count(CNOTGate())

    with open(save_file, 'wb') as f:
        pickle.dump(orig_cx_counts, f)
    return orig_cx_counts

if __name__ == '__main__':
    # Collect data from all folders
    use_small_block = True
    output_cx = False
    cliff_t = True

    if not cliff_t:
        small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_more_cx_tket"
        small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_tket"
    else:
        small_block_checkpoints_dir_1 = None
        small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_clifft_tket"


    good_ratio_data = read_data_from_folders(circs, small_block_checkpoints_dir_2, small_block=use_small_block)
    ratio_data_more_cx = read_data_from_folders(circs, small_block_checkpoints_dir_1, small_block=use_small_block)

    print("Ratio data loaded", flush=True)
    if output_cx:
        compiler = Compiler(num_workers=len(circs))
        orig_counts = get_orig_counts(circs, cliff_t=cliff_t, compiler=compiler)
        compiler.close()
        print("Original counts loaded", flush=True)
        print(orig_counts["add17"]["03"]["0"], flush=True)
        cx_data_more_cx= read_cx_data_from_folders(circs, orig_counts,
                                                    small_block_checkpoints_dir_1, 
                                                    small_block=use_small_block, 
                                                    cliff_t=cliff_t)
        
        print("CX data loaded", flush=True)
        
        good_cx_data = read_cx_data_from_folders(circs, orig_counts,
                                                  small_block_checkpoints_dir_2, 
                                                  small_block=use_small_block, 
                                                  cliff_t=cliff_t)
    else:
        cx_data_more_cx = {}
        good_cx_data = {}

    # Start with more_cx data, and update with good_ratio_data if its better
    ratio_data = good_ratio_data.copy()
    ratio_data.update(ratio_data_more_cx)
    
    cx_data = good_cx_data.copy()
    cx_data.update(cx_data_more_cx)

    RATIO_LIMIT = 10

    # For all circ, block_num, small_block_num
    if not cliff_t:
        for circ in ratio_data.keys():
            for block_num in ratio_data[circ].keys():
                for small_block_num in ratio_data[circ][block_num].keys():
                    for eps in ratio_data[circ][block_num][small_block_num].keys():
                        final_ratio = ratio_data[circ][block_num][small_block_num][eps]
                        if final_ratio > RATIO_LIMIT:
                            # Check if good_data is better
                            if circ in good_ratio_data and block_num in good_ratio_data[circ] and small_block_num in good_ratio_data[circ][block_num]:
                                good_ratio = good_ratio_data[circ][block_num][small_block_num].get(eps, float("inf"))
                                if good_ratio < RATIO_LIMIT:
                                    ratio_data[circ][block_num][small_block_num][eps] = good_ratio_data[circ][block_num][small_block_num][eps]
                                    if output_cx:
                                        cx_data[circ][block_num][small_block_num][eps] = good_cx_data[circ][block_num][small_block_num][eps]
                                else:
                                    if output_cx:
                                        cx_data[circ][block_num][small_block_num][eps] = 0 # Don't count reduction
                            else:
                                if output_cx:
                                    cx_data[circ][block_num][small_block_num][eps] = 0 # Don't count reduction

    # Combine all data frames into a single data frame (optional)
    # combined_data = pd.concat(data_frames, ignore_index=True)
    # Plotting Ratio Data
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes = [axes]

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    circ_colors = {circ: colors[i % len(colors)] for i, circ in enumerate(circs)}


    # Plot each circ separately
    shifts = [0.1 * i for i in range(len(circs))]
    shifts = [a - 0.1 * len(circs) / 2 for a in shifts]
    handles = []
    for i, circ in enumerate(circs):
        # plot_line_bound(plot_data[circ], axes[0], color=circ_colors[circ], use_small_block=use_small_block)
        # print(circ_labels.get(circ, circ), flush=True)
        handles.append(plot_violin_plot(ratio_data[circ], axes[0], color=circ_colors[circ], 
        use_small_block=use_small_block, eps_shift=shifts[i], label=circ_labels.get(circ, circ)))

    for ax in axes:
        if cliff_t:
            ax.set_title("Error Scaling (FT)", fontdict={"size": 28})
        else:
            ax.set_title("Error Scaling (NISQ)", fontdict={"size": 28})
        ax.set_xlabel("Average Frobenius Distance ($\epsilon$)", fontdict={"size": 24})
        if output_cx:
            ax.set_ylabel("CNOT Reduction", fontdict={"size": 24})
        else:
            ax.set_ylabel("Scaling Factor ($\gamma$)", fontdict={"size": 24})
            ax.set_yscale("log")
        # ax.set_xbound(0.8, 5.2)
        # ax.set_ybound(-3, max(ax.get_ybound()[1], 10))
        ax.set_yticklabels(ax.get_yticks(), fontdict={"size": 16})
        ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
        ax.set_xticklabels([f"10e-{i}" for i in range(1, 6)], fontdict={"size": 16})
        ax.legend(handles=handles, loc='upper left', fontsize=12)

    # Save the figure
    # plt.tight_layout()
    fig.tight_layout()
    if output_cx:
        extra = "_cx"
    else:
        extra = "_ratio"

    if cliff_t:
        extra_2 = "_cliff"
    else:
        extra_2 = "_nisq"
    fig.savefig(f"error_scaling_4_all_violin{extra}{extra_2}.png", dpi=300)


    # Output CX data to a csv file
    if output_cx:
        csv_file_name = f"error_scaling_4_all_cx{extra_2}.csv"
        output_csv(cx_data, csv_file_name)