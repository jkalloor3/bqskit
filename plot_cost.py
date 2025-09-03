from sys import argv

import matplotlib.pyplot as plt
from util.plot_lib import plot_td_convergence

if __name__ == '__main__':
    circ_name = argv[1] if len(argv) > 1 else 'qae11'
    bias = int(bool(argv[2])) if len(argv) > 2 else False


    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plot_td_convergence(circ_name, axs, bias=bias)

    if bias:
        fig.savefig(f'bias_convergence_{circ_name}.png', dpi=300)
    else:
        fig.savefig(f'td_convergence_{circ_name}.png', dpi=300)