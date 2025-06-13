# This script looks at good blocks and plots the number of CNOTs per block, the 
# number of qubits per block and the number of gates per block. It also looks
# at the number of partial solutions per block for different epsilons and then 
from util import get_block_names, load_block, load_ensemble, load_jiggled_ensemble, get_ham_shifts, get_ham_shift
from bqskit.ext import bqskit_to_qiskit
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import CNOTGate, U3Gate, CircuitGate
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.compiler import BasePass, PassData
import os
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import glob
from itertools import chain
from pathlib import Path
import multiprocessing as mp
import seaborn as sns
import pandas as pd
from bqskit.runtime import get_runtime
from math import ceil


nisq_benchmark_folder = "block_checkpoints_final_paper_tket"
psol_analysis_folder = "block_ens_analysis_tket"
small_block_folder = "small_block_checkpoints_final_paper_4_tket"
small_block_pngs_folder = "small_block_pngs"

# circs = ["qft_6", "qft_16", "mult16", "draper_adder_12", "shor_12", "qae13", "qaoa10", "heisenberg7", "lgt_17"]
circs = ["shor_12"]
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

compiler = Compiler(num_workers=100)

class ProcessSmallBlock(BasePass):
    def __init__(self, circ: str, block: str, targets: dict):
        self.circ = circ
        self.block = block
        self.targets = targets
        self.csv_file_name = f"{small_block_folder}/{circ}_{block}_" +"{tol}/block_*.csv"
        self.eps = 5.0
        self.pickle_path = f"finished_small_block_data/{circ}/{block}.pkl"
        Path(self.pickle_path).parent.mkdir(parents=True, exist_ok=True)

    def process_csv(self, csv_file: str) -> tuple:
        reader = csv.DictReader(open(csv_file, 'r'))
        min_ratio = float("inf")
        num_sols = 1
        # print(csv_file, reader.fieldnames)
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                min_ratio = min(min_ratio, float(row["Ratio"]))
                qasms_file = csv_file.replace(".csv", "/ensemble_0_.qasms")
                with open(qasms_file, 'r') as qasms_str:
                    num_sols = qasms_str.read().count("BREAK") + 1
        if min_ratio == float("inf"):
            print("No Ratio!")
            min_ratio = -1
        ham_data = ham_distance(csv_file, self.targets)
        return min_ratio, num_sols, ham_data


    async def run(self, circuit: Circuit, data: PassData) -> PassData:
        if os.path.exists(self.pickle_path):
            return pickle.load(open(self.pickle_path, "rb"))
        ratios, num_psols, ham_dists = [], [], []
        all_csv_files = glob.glob(self.csv_file_name.format(tol=self.eps))

        print("Processing CSV files for small block: ", self.circ, self.block)
        print("Number of CSV files: ", len(all_csv_files))

        all_data = await get_runtime().map(self.process_csv, all_csv_files)

        print("Finished processing CSV files for small block: ", 
              self.circ, self.block)

        for min_ratio, num_sols, ham_data in all_data:
            ratios.append(min_ratio)
            num_psols.append(num_sols)

            print("Len Ham Data: ", len(ham_data))
            ham_dists.append(ham_data)
        print("Num Ham Dists Point: ", len(ham_dists))

        # Save data
        pickle.dump((ratios, num_psols, ham_dists), 
                    open(self.pickle_path, "wb"))



def run_all_small_blocks(circ_blocks: list[tuple[str, str, dict]]) -> None:
    bq_passes = [ProcessSmallBlock(circ, block, target) for circ, block, target in circ_blocks]

    ids = []
    for bq_pass in bq_passes:
        workflow = [bq_pass]
        circ = Circuit(1) # Dummy circuit
        circ_id = compiler.submit(circ, workflow)
        ids.append(circ_id)
    
    for id in ids:
        compiler.result(id)

    
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

def ham_distance(csv_file, targets) -> tuple[np.ndarray, np.ndarray]:
    # Given a csv file, get the corresponding .qasms file
    # Get the distance between the original circuit and the final circuit
    qasms_file = csv_file.replace(".csv", "/ensemble_0_.qasms")
    params_file = csv_file.replace(".csv", "/ensemble_0_jiggles_.npy")
    small_block_num = csv_file.split("block_")[2].split(".csv")[0]
    target = targets[int(small_block_num)]
    assert isinstance(targets, dict)
    circ_params = load_jiggled_ensemble(qasms_file, params_file, "")
    # print(target.shape)
    ham_shifts = [get_ham_shifts(circ_param, target) for circ_param in circ_params]
    ham_shifts = np.array(list(chain.from_iterable(ham_shifts)))
    # Calculate trace distance of each hamiltonian to the other
    ham_dists = []
    ham_angles = []
    # TODO: einsum this
    tr_Us = np.einsum("aij,bij->ab", ham_shifts.conj(), ham_shifts, optimize=True)
    ham_dists = np.abs(tr_Us).flatten()
    ham_angles = np.angle(tr_Us).flatten()
    # print(ham_dists.shape, ham_angles.shape)
    # Look at the distribution of this quantity
    print("Mean Ham Dist: ", np.mean(ham_dists))
    print("Mean Ham Angle: ", np.mean(ham_angles))
    return ham_dists, ham_angles

def ham_distance_no_jiggle(csv_file, targets) -> tuple[np.ndarray, np.ndarray]:
    # Given a csv file, get the corresponding .qasms file
    # Get the distance between the original circuit and the final circuit
    qasms_file = csv_file.replace(".csv", "/ensemble_0_.qasms")
    small_block_num = csv_file.split("block_")[2].split(".csv")[0]
    target = targets[int(small_block_num)]
    assert isinstance(targets, dict)
    circs = load_ensemble(qasms_file)
    ham_shifts = [get_ham_shift(c) for c in circs]
    ham_shifts = np.array(ham_shifts)
    # Calculate trace distance of each hamiltonian to the other
    ham_dists = []
    ham_angles = []
    # TODO: einsum this
    tr_Us = np.einsum("aij,bij->ab", ham_shifts.conj(), ham_shifts, optimize=True)
    ham_dists = np.abs(tr_Us).flatten()
    ham_angles = np.angle(tr_Us).flatten()
    # print(ham_dists.shape, ham_angles.shape)
    # Look at the distribution of this quantity
    print("Mean Ham Dist: ", np.mean(ham_dists))
    print("Mean Ham Angle: ", np.mean(ham_angles))
    return ham_dists, ham_angles

def get_small_block_scaling_data(circ, block) -> list[tuple[float, int]]:
    bq_pass = ProcessSmallBlock(circ, block, None)
    ratios, num_psols, ham_dists = pickle.load(open(bq_pass.pickle_path, "rb"))
    return ratios, num_psols, ham_dists


def visualize_small_block_sols(circ_name, block, csv_file: str) -> None:
    qasms_file = csv_file.replace(".csv", "/ensemble_0_.qasms")
    small_block_num = csv_file.split("block_")[2].split(".csv")[0]
    circs = load_ensemble(qasms_file)
    get_partial_solution_data
    i = 0
    for circ in circs:
        qcirc = bqskit_to_qiskit(circ)
        q_path = f"small_block_{circ_name}_{block}_{small_block_num}/{i}.png"
        Path(q_path).parent.mkdir(parents=True, exist_ok=True)
        qcirc.draw(output="mpl", filename=q_path)
        i += 1
        # print(qcirc)


def visualize_small_block(csv_file: str) -> None:
    # Given a csv file, get the corresponding .qasms file
    # Plot the original circuit and the possible solutions
    qasms_file = csv_file.replace(".csv", "/ensemble_final.qasms")
    circs = load_ensemble(qasms_file)
    # Pick 5 random circs
    if len(circs) > 5:
        rand_circs = np.random.choice(circs, 5, replace=False)
    else:
        rand_circs = circs
    for i, circ in enumerate(rand_circs):
        qcirc = bqskit_to_qiskit(circ)
        qcirc.draw(output="mpl", filename=f"small_block_{i}.png")


def get_all_block_subparts() -> dict[str, dict[str, list]]:
    # Get all block circuits
    all_block_circs = []
    for circ_name in circs:
        for block in get_block_names(circ_name):
            circ_file = load_block(circ_name, block, "_tket")
            circ = Circuit.from_file(circ_file)
            all_block_circs.append(circ)

    targets = {}
    # Now partition to size 3
    workflow = [
        ScanPartitioner(4)
    ]
    final_ids = []
    for c in all_block_circs:
        final_ids.append(compiler.submit(c, workflow))
        # print("*", end="", flush=True)

    # Now get final circuits
    final_block_circs = []
    for circ_id in final_ids:
        final_block_circs.append(compiler.result(circ_id))
        # print("-", end="", flush=True)

    # print("Finished Partitioning", flush=True)

    # Now get block data
    circ_block_data = {}
    ind = 0
    for circ_name in circs:
        circ_block_data[circ_name] = {}
        targets[circ_name] = {}
        for block in get_block_names(circ_name):
            targets[circ_name][block] = {}
            circ: Circuit = final_block_circs[ind]
            # visualize_small_block(circ)

            # Now get average CNOT count, average depth, and average num params,
            # and average num_fixed_params
            all_block_data = None
            img_ind = 0
            for op in circ.operations():
                assert isinstance(op.gate, CircuitGate)

                block_circ_data = get_all_circuit_data(op.gate._circuit)
                # qcirc = bqskit_to_qiskit(op.gate._circuit)
                # qcirc.draw(output="mpl", filename=f"{small_block_pngs_folder}/{circ_name}_{block}_{img_ind}.png")
                targets[circ_name][block][img_ind] = op.get_unitary()
                img_ind += 1
                if all_block_data is None:
                    all_block_data = np.array(block_circ_data)
                else:
                    all_block_data = np.vstack((all_block_data, block_circ_data))

            avg_data = np.mean(all_block_data, axis=0)
            circ_block_data[circ_name][block] = avg_data
            ind += 1
            # print(targets[circ_name][block])
            
    return circ_block_data, targets

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
        # Combine all list of lists
        x_data = [item for sublist in x_data for item in sublist]
        y_data = [item for sublist in y_data for item in sublist]
        # Remove all -1s
        good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, 0)]
        x_data = [x_data[i] for i in good_inds] 
        y_data = [y_data[i] for i in good_inds]

        ax.scatter(x_data, y_data, label=circ, alpha=0.5, color="green", marker=circ_markers[circ], s=100)
    for circ, block_data in bad_data.items():
        x_data = [data["Small Block Num Psols"] for data in block_data.values()]
        y_data = [data["Small Block Ratios"] for data in block_data.values()]
        # Combine all list of lists
        x_data = [item for sublist in x_data for item in sublist]
        y_data = [item for sublist in y_data for item in sublist]
        # Remove all -1s
        good_inds = [i for i, x in enumerate(x_data) if not np.allclose(x, 0)]
        x_data = [x_data[i] for i in good_inds] 
        y_data = [y_data[i] for i in good_inds]
        ax.scatter(x_data, y_data, alpha=0.5, color="red", marker=circ_markers[circ], s=100)
    ax.legend()
    ax.set_yscale("log")
    # ax.set_xscale("log")
    fig.savefig("small_block_analysis_2.png")


def plot_small_block_data_ham(all_data: dict):
    x_label = "Small Block Ratios"
    y_label_1 = "Ham. Overlap (Mag)"
    y_label_2 = "Ham. Distance (Angle)"
    psol_label = "Small Block Num Psols"

    x_pts = []
    y_pts_1 = []
    y_pts_2 = []


    for circ, block_data in all_data.items():
        for block, data in block_data.items():
            psols = data[psol_label]

            # Only add data if psols > 3
            psol_inds = [i for i, x in enumerate(psols) if x > 1]
            x_data = [data[x_label][i] for i in psol_inds]
            y_data_1 = [data["Small Block Ham Dists"][i][0] for i in psol_inds]
            y_data_2 = [data["Small Block Ham Dists"][i][1] for i in psol_inds]
            x_pts.extend(x_data)
            y_pts_1.extend(y_data_1)
            y_pts_2.extend(y_data_2)

    def random_sample(lst, n):
        return np.random.choice(lst, n, replace=False)
    
    y_pts_1 = [random_sample(y, 1000000) for y in y_pts_1]
    y_pts_2 = [random_sample(y, 1000000) for y in y_pts_2]

    # Add 1 to all x_pts that are less than 1
    x_pts = [x + 1 if x < 1 else x for x in x_pts]


    print("Total Points: ", len(x_pts))

    # Now create 10 figures with 20 pts selected
    # Max 20 pts per figure
    num_figs = ceil(len(x_pts) / 20)
    # Randomly order the inds
    all_rand_inds = np.random.permutation(len(x_pts))

    num_pts_per_fig = ceil(len(x_pts) / num_figs)

    for img_ind in range(num_figs):
        # 2 figures for dist and angle
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        ax: plt.Axes = axes[0]
        ax_1: plt.Axes = axes[1]

        # Get next 20 pts from x_pts and y_pts
        rand_inds = all_rand_inds[img_ind * num_pts_per_fig:(img_ind + 1) * num_pts_per_fig]
        sub_x_pts = [x_pts[i] for i in rand_inds]
        sub_y_pts_1 = [y_pts_1[i] for i in rand_inds]
        sub_y_pts_2 = [y_pts_2[i] for i in rand_inds]

        # # Only look at x_pts that are greater than 100
        # big_x_inds = [i for i, x in enumerate(sub_x_pts) if x > 100]
        # # Only consider 5 examples
        # big_x_inds = big_x_inds[:5]
        # big_x_pts = [sub_x_pts[i] for i in big_x_inds]
        # big_y_pts_1 = [sub_y_pts_1[i] for i in big_x_inds]

        # big_data = dict([(x, y) for x, y in zip(big_x_pts, big_y_pts_1)])

        # # Only look at x_pts that are less than 20
        # small_x_inds = [i for i, x in enumerate(sub_x_pts) if x < 20]
        # # Only consider 5 examples
        # small_x_inds = small_x_inds[:5]
        # small_x_pts = [sub_x_pts[i] for i in small_x_inds]
        # small_y_pts_1 = [sub_y_pts_1[i] for i in small_x_inds]

        # small_data = dict([(x, y) for x, y in zip(small_x_pts, small_y_pts_1)])

        # # Plot the Distribution for the first and second one
        # if len(big_x_inds) > 0:
        #     sns.kdeplot(big_data, ax=ax)
        # if len(small_y_pts_1) > 0:
        #     sns.kdeplot(small_data, ax=ax_1)

        ax.boxplot(sub_y_pts_1, positions=sub_x_pts, widths=2, showmeans=True)
        ax_1.boxplot(sub_y_pts_2, positions=sub_x_pts, widths=2, showmeans=True)

        ax.set_xscale("log")
        ax_1.set_xscale("log")
        ax.set_xlabel("Scaling Factor", fontdict={"fontsize": 20})
        ax_1.set_xlabel("Scaling Factor", fontdict={"fontsize": 20})

        ax.set_ylabel(y_label_1, fontdict={"fontsize": 20})
        ax_1.set_ylabel(y_label_2, fontdict={"fontsize": 20})

        # ax.set_xlabel(y_label_1, fontdict={"fontsize": 20})
        # ax_1.set_xlabel(y_label_1, fontdict={"fontsize": 20})

        fig.savefig(f"small_block_analysis_ham_{img_ind}.png")
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    good_data = {}
    bad_data = {}
    all_data = {}

    _, targets = get_all_block_subparts()
    
    circ_blocks = []
    for circ in circs:
        for block in get_block_names(circ):
            circ_blocks.append((circ, block, targets[circ][block]))

    run_all_small_blocks(circ_blocks)

    for circ in circs:
        good_data[circ] = {}
        bad_data[circ] = {}
        all_data[circ] = {}

        blocks = get_block_names(circ_name=circ)

        # if circ.startswith("shor"):
        #     # Don't crowd data with shor blocks
        #     blocks = blocks[::3]

        for block in blocks:
            # Get orig cnot and U3 data
            count_stats = get_block_counts(circ, block)
            # Get partial solution data
            psol_counts, psol_cnots, psol_avg_errors, max_ratio = get_partial_solution_data(circ, block, count_stats[0])

            # Get small block scaling data
            small_block_ratios, small_block_num_psols, ham_dists = get_small_block_scaling_data(circ, block)

            num_cnot_ladders = count_stats[-1]
            
            scaling_factor = check_if_good(circ, block)

            # part_data = sub_partition_data[circ][block]
            part_data = list(range(8))

            block_data = {
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
                "Small Block Num Psols": small_block_num_psols,
                "Small Block Ham Dists": ham_dists
            }

            all_data[circ][block] = block_data

            if scaling_factor < 15:
                good_data[circ][block] = block_data
            else:
                bad_data[circ][block] = block_data

    # Plot data
    # plot_gate_counts(good_data, bad_data)
    # plot_psol_data(good_data, bad_data)
    # plot_gate_count_data2(good_data, bad_data)
    # plot_cnot_ladder_data(good_data, bad_data)
    # plot_subpart_data(good_data, bad_data)
    # plot_subpart_data2(good_data, bad_data)
    # plot_small_block_data(good_data, bad_data)
    plot_small_block_data_ham(all_data)
