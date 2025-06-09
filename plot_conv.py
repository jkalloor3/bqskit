import os
import glob
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from util import load_block
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate

import matplotlib.pyplot as plt

ens_sizes = [1, 5, 10, 40, 160, 640, 2560]
colors = ["blue", "orange", "green", "red", "purple", "cyan", "pink", "brown"]

def plot_data(circ_name, tol: float, do_blocks: bool = False) -> None:
    if do_blocks:
        data_path_form = f"block_ensemble_sim_{circ_name}/*/{tol}"
    else:
        data_path_form = f"ensemble_sim_{circ_name}_{tol}"

    # All data folders
    data_paths = glob.glob(data_path_form)
    if len(data_paths) == 0:
        print(f"No data found for {circ_name} with tol {tol}")
        return
    

    fig, axes = plt.subplots(1, 1, figsize=(7, 6))

    # Plot every block separately
    for data_path in data_paths:
        if do_blocks:
            label = circ_name + "Block: " + data_path.split("/")[-2]
        else:
            label = circ_name
        # Check if all.pickle exists
        if os.path.exists(f"{data_path}/all_data.pickle"):
            data = pickle.load(open(f"{data_path}/all_data.pickle", "rb"))
            # Just plot frobenius cost
            frob_data = data["frob"]
            x_data = ens_sizes
        else:
            print("Getting data from individual files")
            frob_data = []
            x_data = []
            for ens_size in ens_sizes:
                # Check if the data file exists
                data_file = f"{data_path}/{ens_size}.pickle"
                if os.path.exists(data_file):
                    data = pickle.load(open(data_file, "rb"))
                    print(data["frob"])
                    frob_data.append(data["frob"])
                    x_data.append(ens_size)
                else:
                    continue
        frob_data = np.array(frob_data)
        print("Frob Data Shape: ", frob_data.shape)
        avg_frob_data = np.mean(frob_data, axis=1)
        print(f"Avg frob data: {avg_frob_data}")
        print(f"X data: {x_data}")
        axes.plot(x_data, avg_frob_data, label=label)
        # Plot the fill as well
        max_frob_data = np.max(frob_data, axis=1)
        min_frob_data = np.min(frob_data, axis=1)
        print(f"Max frob data: {max_frob_data}")
        print(f"Min frob data: {min_frob_data}")
        axes.fill_between(x_data, min_frob_data, max_frob_data, alpha=0.2)

    axes.set_yscale('log')
    axes.legend(fontsize=11)
    axes.tick_params(axis='both', which='both', labelsize=14)
    img_path = f'ensemble_conv_images/conv_{circ_name}_{tol}_sim.png'
    Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'ensemble_conv_images/conv_{circ_name}_{tol}_sim.png', bbox_inches='tight')

if __name__ == '__main__':
    circ_name = sys.argv[1]
    tol = float(sys.argv[2])
    # block_num = sys.argv[2]
    do_blocks = True
    # plot_all_noisy_data(circ_name, block_num)
    plot_data(circ_name, tol, do_blocks)
