# Constans, colors, and utility functions for plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import glob
import os
import pickle 

from .hamiltonian import generate_hamiltonian, get_obs, generate_init_state
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
    "heisenberg7": "Heisenberg - 7q",
    "qae33": "QAE - 33q",
    "heisenberg64": "Heisenberg - 64q",
    "adder63": "Adder - 63q",
    "mult64": "Multiplier - 64q",
    "tfim100": "TFIM - 100q",
    "qpe_11": "QPE - 11q",
    "mult16": "Multiplier - 16q",
    "add17": "Adder - 17q",
    "qae11": "QAE - 11q",
    "draper_adder_12": "Draper QFT Adder - 12q",
    "LiH_hatt": "Li-H Sim - 12q",
    "LiH_jw_long": "Li-H Sim - 12q",
    "QITE_8_0": "QITE - 8q",
    "QITE_8_1": "QITE - 8q",
    "QITE_8_2": "QITE - 8q",
    "QITE_8_3": "QITE - 8q",
    "QITE_8_4": "QITE - 8q",
    "QITE_8_5": "QITE - 8q",
    "QITE_8_6": "QITE - 8q",
    "FermiHubbard2x2_fh": "Fermi Hubbard - 8q",
    "FermiHubbard2x2_jw_long": "Fermi Hubbard - 8q",
    "neutrino_NX_3_NF_2_jw_long": "Neutrino Oscillation - 12q"
}

benchmarks = list(benchmark_labels.keys())

benchmark_colors = {
    "lgt_11": "#5cb9fb",
    "lgt_17": "#5cb9fb",
    "qae13": "#86f386",
    "qpe_14": "#ff7070",
    "LiH": "#c98eff",
    "LiH_hatt": "#c98eff",
    "LiH_jw_long": "#c98eff",
    "mult16": "#ab776d",
    "mult8": "#ab776d",
    "qaoa10": "#00c8b7",
    "qpe10": "#7f7f7f",
    "heisenberg7": "#b6b664",
    "qae33": "#86f386",
    "heisenberg64": "#b6b664",
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
    "FermiHubbard2x2_jw_long": "#2ca02c",
    "neutrino_NX_3_NF_2_jw_long": "#d62728"
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
                 y_label: str = "Trace Distance of Channel",
                 calc_obs: bool = False):
    """
    Plot density matrix data for a list of circuits.

    circ_names: List of circuit names to plot.
    folder_form: The folder form for loading the density matrix data.
    """
    folders = [folder_form.format(circ_name=circ_name) for circ_name in circ_names]
    circ_names = [circ_name for circ_name in circ_names if os.path.exists(folder_form.format(circ_name=circ_name))]
    folders = [folder for folder in folders if os.path.exists(folder)]

    print("Num Folders: ", len(folders))

    circ_folders = zip(circ_names, folders)


    x_vals = []
    y_vals = []

    if calc_obs:
        file_name = "*rho_out.pkl"
    else:
        file_name = "*rho_outs.pkl"

    superop_file_name = "*superop*.npy"
    min_num_blocks = {circ_name: float('inf') for circ_name in circ_names}

    for circ_name, folder in circ_folders:
        pickle_files = glob.glob(os.path.join(folder, file_name))
        print(f"File name: {os.path.join(folder, file_name)}")
        print(f"Num pickle files for {circ_name}: {len(pickle_files)}")
        num_blocks = len(glob.glob(os.path.join(folder, superop_file_name)))
        if num_blocks < min_num_blocks[circ_name]:
            min_num_blocks[circ_name] = num_blocks
            print(f"Num Blocks for {circ_name}: {num_blocks}")
        x_vals = []
        y_vals = []
        full_circ = load_circuit(circ_name)
        full_circ.remove_all_measurements()
        ham = None
        if calc_obs:
            ham = generate_hamiltonian(circ_name, full_circ.num_qudits)
            init_sv = generate_init_state(circ_name, full_circ.num_qudits)
            sv_out = full_circ.get_statevector(init_sv)
            dm = get_density_matrix(sv_out.numpy)
            true_val = get_obs(dm, ham)
            print(f"True value for {circ_name}: {true_val}")
        for pf in pickle_files:
            with open(pf, 'rb') as f:
                all_data: list[tuple[np.ndarray, np.ndarray]]= pickle.load(f)
                # Get tol from filename
                filename = os.path.basename(pf)
                parts = filename.split('.')
                tol = float(parts[0])  # Assuming the second part is the tolerance
                x = (10 ** (-tol))
                x_vals.append(x)
                if not calc_obs:
                    # Calculate Trace Distance from full circ
                    max_dist = 0
                    for rho_out, sv in all_data:
                        sv_out = full_circ.get_statevector(sv)
                        dm_out = get_density_matrix(sv_out.numpy)
                        y = trace_distance(dm_out, rho_out)
                        if np.abs(y) > max_dist:
                            max_dist = np.abs(y)
                    y_vals.append(max_dist)
                else:
                    rho_out, _ = all_data[0]
                    y = get_obs(rho_out, ham)
                    # Plot Difference from true value
                    y_vals.append(np.abs(true_val - y))
        axs.scatter(x_vals, y_vals, label=benchmark_labels.get(circ_name, circ_name),
                    color=benchmark_colors.get(circ_name, "black"), s=150)
        
    # Plot num_blocks * eps^2 in benchmark color
    for circ_name, num_blocks in min_num_blocks.items():
        if num_blocks == 0:
            continue
        x_range = [10**-5, 0.1]
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        axs.plot(x_vals, num_blocks * (x_vals**2), 
                 color=benchmark_colors.get(circ_name, "black"), 
                 linestyle='--', linewidth=3)

    axs.set_xlabel('Epsilon ($\epsilon$)', fontdict={"size": 16})
    axs.set_ylabel(y_label, fontdict={"size": 16})
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend(fontsize=14)

    # x_range = np.array(axs.get_xlim())
    # x_vals = np.linspace(x_range[0], x_range[1], 100)
    # axs.plot(x_vals, x_vals**2, color='black', linestyle='--', linewidth=3, label='$\eps^2$')

    for label in axs.get_xticklabels():
        label.set_fontsize(14)
    for label in axs.get_yticklabels():
        label.set_fontsize(14)


    # axs.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_td_convergence(circ_name: str,
                         axs: plt.Axes,
                         bias: bool = False):
    """
    Plot trace distance convergence for a given circuit.

    circ_name: Name of the circuit to plot.
    axs: Matplotlib Axes object to plot on.
    folder_form: The folder form for loading the density matrix data.
    y_label: Label for the y-axis.
    x_label: Label for the x-axis.
    """
    base_dir = "ensemble_td_convergences_new"
    if bias:
        base_dir = "ensemble_bias_convergences_new"
    full_form = os.path.join(base_dir, f"{circ_name}_*_*.pkl")

    all_files = glob.glob(full_form)

    # Now for each tol, keep track of the data
    all_data = {}
    for file in all_files:
        base_file = os.path.basename(file)
        parts = base_file.split('_')
        tol = float(parts[-1].split('.')[0])  # Last part is tol
        num_samples = int(parts[-2])  # Second last part is num_samples
        if tol not in all_data:
            all_data[tol] = {}
        all_data[tol][num_samples] = pickle.load(open(file, 'rb'))

    # Sort the data by tol
    all_data = {k: v for k, v in sorted(all_data.items(), key=lambda item: item[0])}

    # Now plot each tol data in a separate line
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for tol, data in all_data.items():
        sample_sizes = sorted(data.keys())
        y_vals = [np.mean(data[size]) for size in sample_sizes]
        min_vals = [np.min(data[size]) for size in sample_sizes]
        max_vals = [np.max(data[size]) for size in sample_sizes]

        color =  colors[int(tol) % len(colors)]

        exponent = -int(tol)
        label = f"Eps: $10^{{{exponent}}}$"
        axs.plot(sample_sizes, y_vals, label=label,color=color)
        axs.fill_between(sample_sizes, min_vals, max_vals,
                         color=color, alpha=0.2)
        # Plot a horizontal dotted line at 10 ** (-2 * tol)
        axs.axhline(y=10 ** (-2 * tol), color=color, linestyle='--', linewidth=2)

    axs.set_xlabel("Number of Samples")
    axs.set_ylabel("Trace Distance")
    axs.set_yscale('log')
    # axs.set_xscale('log')
    axs.legend()
    axs.grid(True, which='both', linestyle='--', linewidth=0.5)

def plot_tvd_convergence(all_data: dict,
                          total_shots: int,
                          sampling_error: float,
                          circ_name: str,
                          axs: plt.Axes):
    """
    Plot TVD convergence for a given circuit.

    circ_name: Name of the circuit to plot.
    axs: Matplotlib Axes object to plot on.
    folder_form: The folder form for loading the density matrix data.
    y_label: Label for the y-axis.
    x_label: Label for the x-axis.
    """
    # Now plot each tol data in a separate line
    color = benchmark_colors.get(circ_name, "black")
    x_vals = []
    y_means = []
    y_mins = []
    y_maxs = []
    for sample_size, data in all_data.items():
        x_vals.append(sample_size)
        y_means.append(np.mean(data))
        y_mins.append(np.min(data))
        y_maxs.append(np.max(data))

    axs.plot(x_vals, y_means, label=benchmark_labels.get(circ_name, circ_name), color=color)
    axs.fill_between(x_vals, y_mins, y_maxs,
                     color=color, alpha=0.2)
    # Plot a horizontal dotted line at sampling error
    axs.axhline(y=sampling_error, color='red', linestyle='--', linewidth=2,
                label='Sampling Error of Full Circuit')

    axs.set_xlabel("Number of Sampled Circuits")
    axs.set_ylabel(f"Num Shots: {total_shots} \n TVD")
    axs.set_yscale('log')
    axs.legend()
    axs.grid(True, which='both', linestyle='--', linewidth=0.5)
