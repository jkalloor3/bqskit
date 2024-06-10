from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.ir.gates import CNOTGate
from bqskit.passes import QuickPartitioner, ScanPartitioner, ForEachBlockPass
from bqskit.compiler import Compiler

from util import save_circuits, load_circuit, AnalyzeBlockPass

import matplotlib.pyplot as plt
import pandas as pd

def plot_data(all_block_data: dict[int, dict[int, pd.DataFrame]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for i, bb in enumerate(all_block_data.keys()):
        ax = axes[i]
        df = all_block_data[bb][3]
        df.plot(x="Block Size", kind="bar", ax=ax, stacked=True, title=f"Gate coverage, Big block size: {bb}")

    return fig


def get_shortest_circuits(circ_name: str, big_block_size: int, small_block_size: int) -> pd.DataFrame:
    circ = load_circuit(circ_name)
    leap_workflow = [
        QuickPartitioner(block_size=big_block_size),
        ForEachBlockPass(
            [
                AnalyzeBlockPass(CNOTGate()),
                ScanPartitioner(block_size=small_block_size),
                ForEachBlockPass(
                [
                    AnalyzeBlockPass(CNOTGate()),
                ])
            ]

        ),
    ]
    num_workers = 256
    compiler = Compiler(num_workers=num_workers)
    _, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)

    big_labels = list(range(2, big_block_size + 1))
    small_labels = list(range(2, small_block_size + 1))


    coverage_big = [0 for _ in big_labels]
    coverage_small = [0 for _ in small_labels]

    twoq_coverage_big = [0 for _ in big_labels]
    twoq_coverage_small = [0 for _ in small_labels]

    for big_block in data[ForEachBlockPass.key][0]:
        # print(list(big_block.keys()))
        coverage_big[big_block["num_qubits"] - 2] += big_block["num_gates"]
        twoq_coverage_big[big_block["num_qubits"] - 2] += big_block["block_twoq_count"]
        for small_block in big_block[ForEachBlockPass.key][0]:
            coverage_small[small_block["num_qubits"] - 2] += small_block["num_gates"]
            twoq_coverage_small[small_block["num_qubits"] - 2] += small_block["block_twoq_count"]

    # print(big_data)
    # print(small_data)
    big_df = pd.DataFrame({
        "Block Size": big_labels,
        "Gate Count": coverage_big,
        # "CNOT Count": twoq_coverage_big
    })
    small_df = pd.DataFrame({
        "Block Size": small_labels,
        "Gate Count": coverage_small,
        "Circ Name": circ_name,
    })
    # exit(0)

    return coverage_small

if __name__ == '__main__':
    big_block_sizes = [8]
    small_block_sizes = [3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # circ_names = ["tfxy_6", "qc_binary_5q", "qc_optimized_5q"]
    circ_names = ["shor_12", "vqe_12", "vqe_14"]
    for i, ax in enumerate(axes):
        circ_name = circ_names[i]
        circ = load_circuit(circ_name)
        rows = []
        for bb in big_block_sizes:
            for sb in small_block_sizes:
                rows.append([bb, *get_shortest_circuits(circ_name, bb, sb)])
                print(rows[-1])

        df = pd.DataFrame(
            rows,
            columns=["Big Block Size", "2", "3"]
        )

        df.plot(x="Big Block Size", kind="bar", stacked=True, ax=ax)

        ax.set_title(f"Gate coverage for {circ_name}")

    # fig: plt.Figure = plot_data(gate_coverage_data)

    fig.savefig(f"block_coverage_all_gates.png")
