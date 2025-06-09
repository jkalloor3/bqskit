# This script looks at good blocks and plots the number of CNOTs per block, the 
# number of qubits per block and the number of gates per block. It also looks
# at the number of partial solutions per block for different epsilons and then 
from util import get_block_names, load_block
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, U3Gate, CircuitGate
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
import os
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import glob

nisq_benchmark_folder = "block_checkpoints_final_paper_tket"
psol_analysis_folder = "block_ens_analysis_tket"
small_block_folder = "small_block_checkpoints_final_paper_tket"

circs = ["qft_6", "qft_16", "mult16", "draper_adder_12", "shor_12", "qae13", "qaoa10", "heisenberg7", "lgt_17"]
circ_markers = {
    "qft_6": "o",
    "mult16": "s",
    "draper_adder_12": "^",
    "shor_12": "D",
    "qpe_14": "x",
    "qae13": "P",
    "qaoa10": "v",
    "heisenberg7": "H",
    "lgt_17": "X",
    "qft_16": "h",
}

def get_all_circuit_data(circ: Circuit) -> list[int]:
    num_cnots = circ.count(CNOTGate())
    num_u3s = circ.num_operations - num_cnots
    depth = circ.depth
    two_q_depth = circ.multi_qudit_depth

    num_unfixed_params = 0
    for param in circ.params:
        # If param is a multiple of np.pi/4, it is fixed
        mod = param % (np.pi / 4)
        if not np.allclose(mod, 0):
            num_unfixed_params += 1

    cnots_per_param = num_cnots / num_unfixed_params if num_unfixed_params > 0 else 0
    cnots_per_layer = num_cnots / depth if depth > 0 else 0
    cnots_per_qubit = num_cnots / circ.num_qudits if circ.num_qudits > 0 else 0

    # Get number of CNOT ladders
    # Go through circuit as DAG
    num_cnot_ladders = 0
    prev_cnot = False
    for op in circ.operations():
        if isinstance(op.gate, CNOTGate) and prev_cnot:
            num_cnot_ladders += 1
        elif isinstance(op.gate, CNOTGate):
            prev_cnot = True
        else:
            prev_cnot = False


    return num_cnots, num_u3s, depth, two_q_depth, num_unfixed_params, cnots_per_param, cnots_per_layer, cnots_per_qubit, num_cnot_ladders

def get_block_counts(circ, block):
    # Get the number of CNOTs and U3s in the block
    circ_file = load_block(circ, block, "_tket")
    circ = Circuit.from_file(circ_file)
    return get_all_circuit_data(circ)

def get_small_block_scaling_data(circ, block) -> list[tuple[float, int]]:
    csv_file_name = f"{small_block_folder}/{circ}_{block}_" +"{tol}/block_*.csv"
    eps = [5.0]
    ratios = []
    num_psols = []
    for eps in eps:
        eps_data = []
        csv_files = glob.glob(csv_file_name.format(tol=eps))
        if len(csv_files) == 0:
            continue
        else:
            for csv_file in csv_files:
                with open(csv_file, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    min_ratio = float("inf")
                    num_sols = 1
                    for row in reader:
                        if "Ratio" in row:  # Check if the column value is not empty
                            min_ratio = min(min_ratio, float(row["Ratio"]))
                            if min_ratio < 1:
                                min_ratio += 1
                            num_sols = max(num_sols, int(row["Num Circs"]) // 100)

            ratios.append(min_ratio)
            num_psols.append(num_sols)
        # all_final_data[eps] = eps_data

    return ratios, num_psols


def get_all_block_subparts() -> dict[str, dict[str, list]]:
    # Get all block circuits
    all_block_circs = []
    for circ_name in circs:
        for block in get_block_names(circ_name):
            circ_file = load_block(circ_name, block, "_tket")
            circ = Circuit.from_file(circ_file)
            all_block_circs.append(circ)

    # Now partition to size 3
    compiler = Compiler()
    workflow = [
        ScanPartitioner(3)
    ]
    final_ids = []
    for c in all_block_circs:
        final_ids.append(compiler.submit(c, workflow))
        print("*", end="", flush=True)

    # Now get final circuits
    final_block_circs = []
    for circ_id in final_ids:
        final_block_circs.append(compiler.result(circ_id))
        print("-", end="", flush=True)

    print("Finished Partitioning", flush=True)

    # Now get block data
    circ_block_data = {}
    ind = 0
    for circ_name in circs:
        circ_block_data[circ_name] = {}
        for block in get_block_names(circ_name):
            circ: Circuit = final_block_circs[ind]

            # Now get average CNOT count, average depth, and average num params,
            # and average num_fixed_params
            all_block_data = None
            for op in circ.operations():
                assert isinstance(op.gate, CircuitGate)
                block_circ_data = get_all_circuit_data(op.gate._circuit)
                if all_block_data is None:
                    all_block_data = np.array(block_circ_data)
                else:
                    all_block_data = np.vstack((all_block_data, block_circ_data))

            avg_data = np.mean(all_block_data, axis=0)
            circ_block_data[circ_name][block] = avg_data
            ind += 1
            
    return circ_block_data

def get_partial_solution_data(circ, block, orig_count):
    # epsilons = [3.0, 4.0, 5.0]
    epsilons = [5.0]
    psol_count_data = []
    psol_cnot_count_data = []
    psol_avg_err_data = []
    max_ratios = []
    for eps in epsilons:
        stats_file = f"{psol_analysis_folder}/{circ}_{block}_{eps}/stats.pkl"
        if os.path.exists(stats_file):
            # Get avg number of psols
            start_ens_ind = 0
            max_psol_num = 0
            for ens_ind in range(start_ens_ind, 5):
                ensemble_file = f"{psol_analysis_folder}/{circ}_{block}_{eps}/ensemble_{ens_ind}_.qasms"
                if os.path.exists(ensemble_file):
                    with open(ensemble_file, 'r') as qasms_str:
                        num_psols = qasms_str.read().count("BREAK") + 1
                        max_psol_num = max(max_psol_num, num_psols)
            psol_count_data.append(max_psol_num)
            stats = pickle.load(open(stats_file, 'rb'))
            psol_cnot_count_data.append(stats[0] - orig_count)
            psol_avg_err_data.append(stats[1])
            if stats[3] == 0:
                max_ratios.append(1)
            else:
                max_ratios.append(stats[4] / stats[3])
        else:
            psol_cnot_count_data.append(-1)
            psol_count_data.append(-1)
            psol_avg_err_data.append(-1)
            max_ratios.append(-1)

    return psol_count_data, psol_cnot_count_data, psol_avg_err_data, max_ratios


def check_if_good(circ, block):
    data_file = f"{nisq_benchmark_folder}/{circ}_{block}_5.0/data.csv"
    if os.path.exists(data_file):
        # Read CSV and get minimum value of ratio
        reader = csv.DictReader(open(data_file, 'r'))
        min_ratio = float("inf")
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                min_ratio = min(min_ratio, float(row["Ratio"]))
        
        # Cut off at 1 for visual fidelity
        return max(min_ratio, 1)
    return 2000


def plot_gate_counts(good_data: dict, bad_data: dict):
    labels = ["CNOT Count", "1Q Gate Count", "Depth", "Two Qubit Depth"]
    plot_data(good_data, bad_data, labels, "gate_count_analysis.png")

def plot_data(good_data: dict, bad_data: dict, labels: list[str], filename: str, x_scales: list[str] = None):
    # Create figures for cnot count, depth, two_q_depth, and num_u3s
    
    if len(labels) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [[axes]]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    if x_scales is None:
        x_scales = ["linear"] * len(labels)
    
    # block_good_data = [d for block_data in good_data.values() for d in block_data.values()]
    # block_bad_data = [d for block_data in bad_data.values() for d in block_data.values()]

    for i, label in enumerate(labels):
        ax: plt.Axes = axes[i // 2][i % 2]
        ax.set_title(label, fontsize=20)
        ax.set_xlabel(label, fontdict={"fontsize": 20})
        ax.set_ylabel("Scaling Factor", fontdict={"fontsize": 20})
        for circ, block_data in good_data.items():
            x_data = [data[label] for data in block_data.values()]
            y_data = [data["Scaling_Factor"] for data in block_data.values()]
            # Remove all -1s
            good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, -1)]
            x_data = [x_data[i] for i in good_inds] 
            y_data = [y_data[i] for i in good_inds]
            ax.scatter(x_data, y_data, label=circ, alpha=0.5, color="green", marker=circ_markers[circ], s=100)
        for circ, block_data in bad_data.items():
            x_data = [data[label] for data in block_data.values()]
            y_data = [data["Scaling_Factor"] for data in block_data.values()]
            good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, -1)]
            x_data = [x_data[i] for i in good_inds] 
            y_data = [y_data[i] for i in good_inds]
            ax.scatter(x_data, y_data, alpha=0.5, color="red", marker=circ_markers[circ], s=100)
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale(x_scales[i])

    fig.savefig(filename)

def plot_psol_data(good_data: dict, bad_data: dict):
    # Create figures for cnot count, depth, two_q_depth, and num_u3s
    labels = ["Avg. Num Psols", "Avg. Psol CNOT Reduction", "Avg. Psol Error", "Max Distance Ratio"]
    plot_data(good_data, bad_data, labels, "psol_analysis.png", x_scales=["linear", "linear", "log", "linear"])


def plot_gate_count_data2(good_data: dict, bad_data: dict):
    labels = ["Num Free Params", "CNOTs per Param", "CNOTs per Layer", "CNOTs per Qubit"]
    plot_data(good_data, bad_data, labels, "gate_count_analysis2.png")


def plot_cnot_ladder_data(good_data: dict, bad_data: dict):
    labels = ["Num CNOT Ladders"]
    plot_data(good_data, bad_data, labels, "cnot_ladder_analysis.png")


def plot_subpart_data(good_data: dict, bad_data: dict):
    labels = ["Partition CNOT Count", "Partition Depth", "Partition Num Params", "Partition Num Fixed Params"]
    plot_data(good_data, bad_data, labels, "subpart_analysis.png")

def plot_subpart_data2(good_data: dict, bad_data: dict):
    labels = ["Partition CNOTs Per Param", "Partition CNOTs Per Layer", "Partition CNOTs Per Qubit", "Partition Num CNOT Ladders"]
    plot_data(good_data, bad_data, labels, "subpart_analysis2.png")

def plot_small_block_data(good_data: dict, bad_data: dict):
    labels = ["Small Block Ratios", "Small Block Num Psols"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # block_good_data = [d for block_data in good_data.values() for d in block_data.values()]
    # block_bad_data = [d for block_data in bad_data.values() for d in block_data.values()]
    ax.set_xlabel("Small Block Num Psols", fontdict={"fontsize": 20})
    ax.set_ylabel("Small Block Scaling Factor", fontdict={"fontsize": 20})
    for circ, block_data in good_data.items():

        x_data = [data["Small Block Num Psols"] for data in block_data.values()]
        y_data = [data["Small Block Ratios"] for data in block_data.values()]
        # Remove all -1s
        good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, 0)]
        x_data = [x_data[i] for i in good_inds] 
        y_data = [y_data[i] for i in good_inds]
        ax.scatter(x_data, y_data, label=circ, alpha=0.5, color="green", marker=circ_markers[circ], s=100)
    for circ, block_data in bad_data.items():
        x_data = [data["Small Block Num Psols"] for data in block_data.values()]
        y_data = [data["Small Block Ratios"] for data in block_data.values()]
        good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, -1)]
        x_data = [x_data[i] for i in good_inds] 
        y_data = [y_data[i] for i in good_inds]
        ax.scatter(x_data, y_data, alpha=0.5, color="red", marker=circ_markers[circ], s=100)
    ax.legend()
    ax.set_yscale("log")
    fig.savefig("small_block_analysis.png")

if __name__ == '__main__':
    good_data = {}
    bad_data = {}

    # sub_partition_data = get_all_block_subparts()

    for circ in circs:
        good_data[circ] = {}
        bad_data[circ] = {}

        blocks = get_block_names(circ_name=circ)

        if circ.startswith("shor"):
            # Don't crowd data with shor blocks
            blocks = blocks[::3]

        for block in blocks:
            # Get orig cnot and U3 data
            count_stats = get_block_counts(circ, block)
            # Get partial solution data
            psol_counts, psol_cnots, psol_avg_errors, max_ratio = get_partial_solution_data(circ, block, count_stats[0])

            # Get small block scaling data
            small_block_ratios, small_block_num_psols = get_small_block_scaling_data(circ, block)

            num_cnot_ladders = count_stats[-1]
            
            scaling_factor = check_if_good(circ, block)

            # part_data = sub_partition_data[circ][block]
            part_data = list(range(8))

            if scaling_factor < 10:
                print("Good Block: ", circ, block, scaling_factor)
                good_data[circ][block] = {
                    "CNOT Count": count_stats[0],
                    "1Q Gate Count": count_stats[1],
                    "Depth": count_stats[2],
                    "Two Qubit Depth": count_stats[3],
                    "Num Free Params": count_stats[4],
                    "CNOTs per Param": count_stats[5],
                    "CNOTs per Layer": count_stats[6],
                    "CNOTs per Qubit": count_stats[7],
                    "Avg. Psol CNOT Reduction": psol_cnots,
                    "Avg. Num Psols": psol_counts,
                    "Avg. Psol Error": psol_avg_errors,
                    "Max Distance Ratio": max_ratio,
                    "Num CNOT Ladders": num_cnot_ladders,
                    "Scaling_Factor": scaling_factor,
                    "Partition CNOT Count": part_data[0],
                    "Partition Depth": part_data[2],
                    "Partition Num Params": part_data[1] * 3,
                    "Partition Num Fixed Params": part_data[4],
                    "Partition CNOTs Per Param": part_data[5],
                    "Partition CNOTs Per Layer": part_data[6],
                    "Partition CNOTs Per Qubit": part_data[7],
                    "Partition Num CNOT Ladders": part_data[-1],
                    "Small Block Ratios": small_block_ratios,
                    "Small Block Num Psols": small_block_num_psols
                }
            else:
                bad_data[circ][block] = {
                    "CNOT Count": count_stats[0],
                    "1Q Gate Count": count_stats[1],
                    "Depth": count_stats[2],
                    "Two Qubit Depth": count_stats[3],
                    "Num Free Params": count_stats[4],
                    "CNOTs per Param": count_stats[5],
                    "CNOTs per Layer": count_stats[6],
                    "CNOTs per Qubit": count_stats[7],
                    "Avg. Psol CNOT Reduction": psol_cnots,
                    "Avg. Num Psols": psol_counts,
                    "Avg. Psol Error": psol_avg_errors,
                    "Max Distance Ratio": max_ratio,
                    "Num CNOT Ladders": num_cnot_ladders,
                    "Scaling_Factor": scaling_factor,
                    "Partition CNOT Count": part_data[0],
                    "Partition Depth": part_data[2],
                    "Partition Num Params": part_data[1] * 3,
                    "Partition Num Fixed Params": part_data[4],
                    "Partition CNOTs Per Param": part_data[5],
                    "Partition CNOTs Per Layer": part_data[6],
                    "Partition CNOTs Per Qubit": part_data[7],
                    "Partition Num CNOT Ladders": part_data[-1],
                    "Small Block Ratios": small_block_ratios,
                    "Small Block Num Psols": small_block_num_psols
                }

    # Plot data
    # plot_gate_counts(good_data, bad_data)
    # plot_psol_data(good_data, bad_data)
    # plot_gate_count_data2(good_data, bad_data)
    # plot_cnot_ladder_data(good_data, bad_data)
    # plot_subpart_data(good_data, bad_data)
    # plot_subpart_data2(good_data, bad_data)
    plot_small_block_data(good_data, bad_data)
