import os
import json
import sys
import pandas as pd
from util import load_block
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate

import matplotlib.pyplot as plt

two_q_err =1e-2
one_q_err =1e-4
csv_folder = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_250/data_try1.csv"
file_name = "{circ_name}_conv_data_noisy_{two_q_err:.1e}_{one_q_err:.1e}/{circ_name}_{block_num}_{tol}_250.json"
colors = ["blue", "orange", "green", "red", "purple", "cyan", "pink", "brown"]

def plot_data(circ_name, block_num, tol) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    
    data, _ = get_json_data(circ_name, block_num, tol)

    x_axis = data["Ensemble Size"]

    headers = ["TVD", "Trace Distance", "Frobenius Distance"]
    for i, header in enumerate(headers):
        if header not in data:
            print(f"Header {header} not found in data")
            continue
        y = data[header]
        axes.plot(x_axis, y, label=headers[i], color=colors[i])
        axes.plot(x_axis, y, '*', color=colors[i])


def get_json_data(circ_name: str, block_num: str | int, tol: float) -> tuple[dict, pd.DataFrame]:
    file_path = file_name.format(tol=tol, circ_name=circ_name, 
                                     block_num=block_num, one_q_err=one_q_err, 
                                     two_q_err=two_q_err)
    print(file_path)
    if not os.path.exists(file_path):
        if tol - int(tol) > 0.0001:
            return None, None
        # Try with integer
        file_path = file_name.format(tol=int(tol), circ_name=circ_name, 
                                    block_num=block_num, one_q_err=one_q_err, 
                                    two_q_err=two_q_err)
        if not os.path.exists(file_path):
            # If this doesn't exist, skip this tol
            return None, None
    data = json.load(open(file_path, 'r'))

    csv_path = csv_folder.format(tol=tol, circ_name=circ_name, block_num=block_num)
    print(csv_path)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        csv_path = csv_folder.format(tol=int(tol), circ_name=circ_name, block_num=block_num)
        if not os.path.exists(csv_path):
            return None, None
        df = pd.read_csv(csv_path)
    return data, df


def plot_noisy_data(data: dict, axes: plt.Axes, label: str= "", color: str = ["blue"]):
    x_axis = data["Ensemble Size"]
    headers = ["TVD"]
    for i, header in enumerate(headers):
        if header not in data:
            print(f"Header {header} not found in data")
            continue
        y = data[header]
        axes.plot(x_axis, y, label=label, color=color)
        axes.plot(x_axis, y, '*', color=color)

def plot_all_noisy_data(circ_name, block_num) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Set y to log scale
    orig_circ = Circuit.from_file(load_block(circ_name, block_num))
    orig_cnots = orig_circ.count(CNOTGate())

    original_tvd = 0

    tols = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    for i, tol in enumerate(tols):
        data, df = get_json_data(circ_name, block_num, tol)
        if data is None or df is None:
            continue

        if "Original Circuit TVD" in data:
            original_tvd = data["Original Circuit TVD"]

        # Covert to scientific notation to 3 sig figs
        e1 = max(df['Norm. Epsilon'])
        cnot_count = round(min(df['Avg. CNOT Count']))
        plot_noisy_data(data, ax, label=f"Norm. Distance={e1:.1e}, {cnot_count} CNOTs", color=colors[i])
    
    xmin, xmax = ax.get_xlim()
    ax.hlines([original_tvd], color="black", label=f"Exact Circuit TVD, {orig_cnots} CNOTs", xmin=xmin, xmax=xmax, linestyles="--")

    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.tick_params(axis='both', which='both', labelsize=14)
    fig.savefig(f'conv_{circ_name}_{block_num}.png', bbox_inches='tight')


if __name__ == '__main__':
    circ_name = sys.argv[1]
    block_num = sys.argv[2]
    plot_all_noisy_data(circ_name, block_num)