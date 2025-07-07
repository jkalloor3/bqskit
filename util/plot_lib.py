# Constans, colors, and utility functions for plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import glob
import os
import pickle 

from .hamiltonian import generate_hamiltonian, get_obs
from .common import load_circuit
from .distance import get_density_matrix, tvd_dict, trace_distance

# Constants

benchmark_labels = {
    "lgt_11": "Lattice Gauge Sim - 11q",
    "lgt_17": "Lattice Gauge Sim - 17q",
    "qae13": "QAE - 13q",
    "qpe_14": "QPE - 14q",
    "LiH": "Li-H Sim - 8q",
    "mult16": "Multiplier - 16q",
    "mult8": "Multiplier - 8q",
    "qaoa10": "QAOA - 10q",
    "qpe10": "QPE - 10q",
    "heisenberg7": "Heisenberg Model - 7q",
    "qae33": "QAE - 33q",
    "heisenberg64": "Heisenberg Model - 64q",
    "adder63": "Adder - 63q",
    "mult64": "Multiplier - 64q",
    "tfim100": "TFIM - 100q",
    "qpe_11": "QPE - 11q",
    "mult16": "Multiplier - 16q",
    "add17": "Adder - 17q",
    "qae11": "QAE - 11q",
    "draper_adder_12": "Draper QFT Adder - 12q",
    "LiH_hatt": "Li-H Sim - 12q",
    "QITE_8_0": "QITE - 8q",
    "QITE_8_1": "QITE - 8q",
    "QITE_8_2": "QITE - 8q",
    "QITE_8_3": "QITE - 8q",
    "QITE_8_4": "QITE - 8q",
    "QITE_8_5": "QITE - 8q",
    "QITE_8_6": "QITE - 8q",
    "FermiHubbard2x2_fh": "Fermi Hubbard - 2x2",
}

benchmarks = list(benchmark_labels.keys())

benchmark_colors = {
    "lgt_11": "#5cb9fb",
    "lgt_17": "#5cb9fb",
    "qae13": "#86f386",
    "qpe_14": "#ff7070",
    "LiH": "#c98eff",
    "LiH_hatt": "#c98eff",
    "mult16": "#ab776d",
    "mult8": "#ab776d",
    "qaoa10": "#00c8b7",
    "qpe10": "#7f7f7f",
    "heisenberg7": "#ffff56",
    "qae33": "#86f386",
    "heisenberg64": "#ffff56",
    "adder63": "#ff7f0e",
    "mult64": "#ab776d",
    "tfim100": "#ffadad",
    "qpe_11": "#ff7070",
    "add17": "#ff7f0e",
    "qae11": "#86f386",
    "draper_adder_12": "#ffa95d",
    "LiH_hatt": "#c98eff",
    "QITE_8_0": "#e377c2",
    "QITE_8_1": "#e377c2",
    "QITE_8_2": "#e377c2",
    "QITE_8_3": "#e377c2",
    "QITE_8_4": "#e377c2",
    "QITE_8_5": "#e377c2",
    "QITE_8_6": "#e377c2",
    "FermiHubbard2x2_fh": "#2ca02c",
}

def plot_all_circ_violins(plot_data: dict, 
                          ax: plt.Axes,
                          plot_title: str = "Error Scaling (NISQ)",
                          x_axis_label: str = "Average Frobenius Distance ($\epsilon$)",
                          y_axis_label: str = "Scaling Factor ($\gamma$)",
                          y_tick_labels: dict = None,
                          x_tick_labels: dict = None,
                          log_scale: bool = True):
    """
    Plot violin plots for all circuits.

    Plot Data: dict mapping circuit names to data for plotting.


    """
    # Plot each circ separately - calculate shifts
    shifts = [0.1 * i for i in range(len(plot_data))]
    shifts = [a - 0.1 * len(plot_data) / 2 for a in shifts]
    handles = []
    for i, circ_name in enumerate(plot_data.keys()):
        circ_data = plot_data[circ_name]
        color = benchmark_colors.get(circ_name, "black")
        label = benchmark_labels.get(circ_name, circ_name)
        handles.append(plot_error_violins(circ_data, ax, eps_shift=shifts[i], color=color, label=label))


    ax.set_title(plot_title, fontdict={"size": 28})
    ax.set_xlabel(x_axis_label, fontdict={"size": 24})
    ax.set_ylabel(y_axis_label, fontdict={"size": 24})
    if log_scale:
        ax.set_yscale("log")

    # Set y tick label of 1000 to 1000+
    if y_tick_labels is None:
        y_tick_labels = {
            0.1: "0.1",
            1: "1",
            10: "10",
            100: "100",
            1000: "1000+",
            10000: "10000+"
        }
    if x_tick_labels is None:
        x_tick_labels = {
            1.0: "$10^{-1}$",
            2.0: "$10^{-2}$",
            3.0: "$10^{-3}$",
            4.0: "$10^{-4}$",
            5.0: "$10^{-5}$"
        }

    ax.set_yticks(list(y_tick_labels.keys()))
    ax.set_yticklabels(list(y_tick_labels.values()), fontdict={"size": 16})
    ax.set_yticklabels(ax.get_yticks(), fontdict={"size": 16})
    ax.set_xticks(list(x_tick_labels.keys()))
    ax.set_xticklabels(list(x_tick_labels.values()), fontdict={"size": 16})
    ax.legend(handles=handles, loc='upper left', fontsize=12)


def plot_error_violins(circ_data: dict, axs: plt.Axes, color: str,
                      eps_shift: float = 0, label: str = ""):
    '''
    data - dictionary mapping 
    
    
    '''
    x = []
    y = []
    for block_num in sorted(circ_data.keys()):
        for eps in sorted(circ_data[block_num].keys()):
            x.append(eps)
            y.append(circ_data[block_num][eps])
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


def plot_dm_data(circ_names: list[str],
                 axs: plt.Axes, 
                 folder_form="ensemble_dms_{circ_name}",
                 y_label: str = "Trace Distance of Channel"):
    """
    Plot density matrix data for a list of circuits.

    circ_names: List of circuit names to plot.
    folder_form: The folder form for loading the density matrix data.
    """
    folders = [folder_form.format(circ_name=circ_name) for circ_name in circ_names]
    circ_names = [circ_name for circ_name in circ_names if os.path.exists(folder_form.format(circ_name=circ_name))]
    folders = [folder for folder in folders if os.path.exists(folder)]

    circ_folders = zip(circ_names, folders)


    x_vals = []
    y_vals = []

    for circ_name, folder in circ_folders:
        pickle_files = glob.glob(os.path.join(folder, '*.pkl'))
        x_vals = []
        y_vals = []
        full_circ = load_circuit(circ_name)
        full_circ.remove_all_measurements()
        ham = generate_hamiltonian(circ_name, full_circ.num_qudits)

        if ham is not None:
            state = np.zeros(2 ** full_circ.num_qudits)
            state[0] = 1.0
            sv_out = full_circ.get_statevector(state)
            dm = get_density_matrix(sv_out.numpy)
            true_val = get_obs(dm, ham)
            print(f"True value for {circ_name}: {true_val}")
        for pf in pickle_files:
            with open(pf, 'rb') as f:
                item = pickle.load(f)
                y = item[0]
                # Get tol from filename
                filename = os.path.basename(pf)
                parts = filename.split('.')
                tol = float(parts[0])  # Assuming the second part is the tolerance
                x = (10 ** (-tol))
                x_vals.append(x)
                if ham is None:
                    y_vals.append(y)
                else:
                    # Plot Difference from true value
                    y_vals.append(np.abs(true_val - y))
        axs.scatter(x_vals, y_vals, label=benchmark_labels.get(circ_name, circ_name),
                    color=benchmark_colors.get(circ_name, "black"), s=100)


    axs.set_xlabel('Epsilon')
    axs.set_ylabel(y_label)
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend()
    axs.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_tvd_convergence(circ_name: str,
                         axs: plt.Axes,
                         noisy: bool = False):
    """
    Plot trace distance convergence for a given circuit.

    circ_name: Name of the circuit to plot.
    axs: Matplotlib Axes object to plot on.
    folder_form: The folder form for loading the density matrix data.
    y_label: Label for the y-axis.
    x_label: Label for the x-axis.
    """
    base_dir = "ensemble_tvds"
    ensemble_sizes = [1, 10, 100, 1000, 10000]
    if noisy:
        extra = "_noisy"
        base_file = os.path.join(base_dir, f"{circ_name}_base_noisy.pkl")
        base_val = pickle.load(open(base_file, 'rb')) if os.path.exists(base_file) else None
    else:
        extra = ""
    files = [os.path.join(base_dir, f"{circ_name}_{size}{extra}.pkl") for size in ensemble_sizes]


    all_data = [pickle.load(open(file, 'rb')) for file in files if os.path.exists(file)]
    
    y_vals = []
    min_vals = []
    max_vals = []
    for tvd_data in all_data:
        # Keep track of min, max and avg
        min_tvd = min(tvd_data)
        max_tvd = max(tvd_data)
        avg_tvd = np.mean(tvd_data)
        y_vals.append(avg_tvd)
        min_vals.append(min_tvd)
        max_vals.append(max_tvd)

    axs.plot(ensemble_sizes, y_vals, label=benchmark_labels.get(circ_name, circ_name),
                color=benchmark_colors.get(circ_name, "black"))
    axs.fill_between(ensemble_sizes, min_vals, max_vals, color=benchmark_colors.get(circ_name, "black"), alpha=0.2)

    
    if noisy:
        # Plot base value as black dashed line
        axs.axhline(base_val, color='black', linestyle='--', label='Full Circuit')

    axs.set_xlabel("Number of Samples")
    axs.set_ylabel("Total Variational Distance")
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend()
    axs.grid(True, which='both', linestyle='--', linewidth=0.5)