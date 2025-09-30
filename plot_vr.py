from sys import argv

import matplotlib.pyplot as plt
from util.plot_lib import plot_all_vr

if __name__ == '__main__':
    v_num = int(argv[1]) if len(argv) > 1 else 1
    circ_names = ["heisenberg7", "qaoa10"]


    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    plot_all_vr(circ_names, axs, v_num=v_num)

    fig.savefig(f'v_R_all_circs_{v_num}.png', bbox_inches='tight')