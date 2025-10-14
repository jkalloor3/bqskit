from sys import argv

import matplotlib.pyplot as plt
from util.plot_lib import plot_td_convergence

if __name__ == '__main__':
    circ_name = argv[1] if len(argv) > 1 else 'heisenberg7'

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plot_td_convergence(circ_name, axs)
    fig.savefig(f'td_convergence_{circ_name}.png', dpi=300)