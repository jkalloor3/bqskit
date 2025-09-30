# This script looks at good blocks and plots the number of CNOTs per block, the 
# number of qubits per block and the number of gates per block. It also looks
# at the number of partial solutions per block for different epsilons and then 
from util import get_block_names, load_block
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, U3Gate, CircuitGate
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle
import time

nisq_benchmark_folder = "block_checkpoints_final_paper_tket"
psol_analysis_folder = "block_ens_analysis_tket"
small_block_folder = "small_block_checkpoints_final_paper_tket"

# circs = ["qft_6", "qft_16", "mult16", "draper_adder_12", "shor_12", "qae13", "qaoa10", "heisenberg7", "lgt_17"]

# Get all circs
dirs = ["ensemble_benchmarks", "qce23_qfactor_benchmarks"]
# # dirs = ["QITE_8"]
# # dirs = ["ham_sim_qasm"]
trial_circs = []
for dir in dirs:
    files = glob.glob(f"{dir}/*.qasm")
    trial_circs.extend([file.split('/')[-1].split(".")[0] for file in files])
circs = []
circs_to_partition = []
for circ in trial_circs:
    blocks = get_block_names(circ, extra="")
    if len(blocks) == 0:
        circs_to_partition.append(circ)
    else:
        circs.append(circ)

print("Circs to partition: ", circs)
    


def get_all_circuit_data(circ: Circuit) -> tuple[int, int]:
    num_cnots = circ.count(CNOTGate())
    num_unfixed_params = 0
    for param in circ.params:
        # If param is a multiple of np.pi/4, it is fixed
        mod = param % (np.pi / 4)
        if not np.allclose(mod, 0):
            num_unfixed_params += 1

    return num_cnots, num_unfixed_params


def get_all_block_subpartdata(block_size: int, compiler: Compiler) -> tuple[list[int], list[int]]:
    partition_pkl_file = f"partition_data_{block_size}.pickle"
    if os.path.exists(partition_pkl_file):
        return pickle.load(open(partition_pkl_file, "rb"))
    
    workflow = [
        ScanPartitioner(block_size=block_size),
    ]

    final_ids = []
    final_block_circs: list[Circuit] = []

    # Get all block circuits
    # all_block_circs = []
    final_block_circs = []
    for circ_name in circs:
        for block in get_block_names(circ_name):
            circ_file = load_block(circ_name, block, "_tket")
            try:
                circ = Circuit.from_file(circ_file)
                out_circ = compiler.compile(circ, workflow)
                final_block_circs.append(out_circ)
            except:
                continue

    # for circ_id in final_ids:
    #     print("Awaiting: ", circ_id)
    #     final_block_circs.append(compiler.result(circ_id))
    #     time.sleep(0.1)
    #     print("Num In Queue: ", len(final_ids))

    # Now get block data
    cnot_counts = []
    rotation_counts = []
    for circ in final_block_circs:
        for op in circ.operations():
            assert isinstance(op.gate, CircuitGate)
            cnots, num_unfixed_params = get_all_circuit_data(op.gate._circuit)
            cnot_counts.append(cnots)
            rotation_counts.append(num_unfixed_params)

    cnot_counts = np.array(cnot_counts)
    rotation_counts = np.array(rotation_counts)

    pickle.dump((cnot_counts, rotation_counts), open(partition_pkl_file, "wb"))
    print("Saved partition data to ", partition_pkl_file)
            
    return cnot_counts, rotation_counts

if __name__ == '__main__':
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes: list[plt.Axes] = axes

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


    block_sizes = [3, 4, 5, 6]
    bar_width = 1 / (len(block_sizes) + 1)
    shifts = [-1.5*bar_width, -0.5*bar_width, 0.5*bar_width, 1.5*bar_width]

    compiler = Compiler(num_workers=2)

    bins = np.concatenate([np.arange(0, 21, 1) - 0.5, [500]])
    print("Bins: ", bins)

    for i, block_size in enumerate([3, 4, 5, 6]):
        cnot_counts, rotation_counts = get_all_block_subpartdata(block_size,
                                                                 compiler)
        
        print("Large CNOT Counts: ", cnot_counts[cnot_counts > 50])
        
        # Remove all counts above 50
        cnot_counts, bin_edges = np.histogram(cnot_counts, bins=bins, density=True)
        print("Bin Edges: ", bin_edges)
        rotation_counts, _ = np.histogram(rotation_counts, bins=bins, density=True)
        positions = np.arange(21) + shifts[i]
        axes[0].bar(positions, cnot_counts, alpha=0.8, 
                     label=f"Block Size {block_size}", color=colors[i],
                     width=bar_width)
        axes[1].bar(positions, rotation_counts , alpha=0.8, 
                     label=f"Block Size {block_size}", color=colors[i],
                     width=bar_width)
        
    # Set axis labels
    axes[0].set_xlabel("CNOT Count")
    axes[0].set_title("CNOT Count Distribution")
    axes[1].set_xlabel("Number of Continous Rotations")
    axes[1].set_title("Continous Rotation Distribution")
    axes[0].legend(title="Block Size")
    axes[1].legend(title="Block Size")
    axes[0].set_xticks(list(range(21)))
    axes[1].set_xticks(list(range(21)))
    tick_labels = [str(i) for i in range(20)] + ["20+"]
    axes[0].set_xticklabels(tick_labels)
    axes[1].set_xticklabels(tick_labels)
    axes[0].set_xlim(-1.5 * bar_width, 20 + 2 * bar_width)
    axes[1].set_xlim(-1.5 * bar_width, 20 + 2 * bar_width)

    # Save figure
    fig.savefig("block_partition_data.png", bbox_inches='tight')