import os

from matplotlib import pyplot as plt
from util.plot_lib import plot_conv_data
import pickle
import sys

if __name__ == '__main__':
    circ_name = sys.argv[1]
    tol = float(sys.argv[2])
    # block_num = sys.argv[2]
    do_blocks = True
    # gather data
    sample_sizes = [1, 10, 100, 1000, 10000]
    all_data = {}
    for sample_size in sample_sizes:
        data_file = f"ensemble_td_convergences/{circ_name}_{sample_size}_{tol}.pkl"
        if os.path.exists(data_file):
            all_data[sample_size] = pickle.load(open(data_file, "rb"))


    fig, ax = plt.subplots(figsize=(10, 6))
    plot_conv_data(all_data, circ_name, ax)

    fig.savefig(f"ensemble_td_conv_{circ_name}_{tol}.png")