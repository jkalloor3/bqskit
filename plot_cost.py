import os
import pickle
import glob
import numpy as np
from bqskit.ir import Circuit
from bqskit.qis import StateVector

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.ext import bqskit_to_qiskit

from util.distance import (normalized_gp_frob_cost, trace_distance, tvd, 
                           get_density_matrix, get_average_density_matrix)

def get_qcirc(circ: Circuit) -> QuantumCircuit:
    circ.unfold_all()
    # Remove all GlobalPhaseGates
    circ.remove_all(GlobalPhaseGate())
    qc = bqskit_to_qiskit(circ)
    qc.measure_all()
    return qc

def get_sv(counts: dict[str, int]) -> np.ndarray:
    '''
    Get the statevector from the counts.
    '''
    total_shots = sum(counts.values())
    if total_shots == 0:
        return np.zeros(2**len(counts), dtype=np.complex128)
    
    sv = np.zeros(2**11, dtype=np.complex128)
    for key, count in counts.items():
        index = int(key, 2)
        sv[index] = count / total_shots
    
    return sv

def get_obs(dm: np.ndarray) -> float:
    '''
    Get the observable for the output density matrix.
    '''
    return np.real(np.trace(ham @ dm))

def generate_hamiltonian(num_qubits: int, x: int) -> SparsePauliOp:
    '''
    H = He (electric) + Hb (magnetic)
    
    He = 3/8 * (3N + 1) - 9/8 * (Z_0 + Z_{N-1}) - 3/4 (sum_{n=1}^{N-2} Z_n)
    - 3/8 * (sum_{n=0}^{N-2} Z_n Z_{n+1})

    Hb = -x/2 (3 + Z_1)(X_0) - x/2 (3 + Z_{N-2})(X_{N-1}) - 
    [x/8 (sum_{n=1}^{N-2} (9 + 3Z_{n-1} + 3Z_{n+1} + Z_{n-1}Z_{n+1}))(X_n))
    '''

    # Generate He
    Z_0_term = ("Z" + "I" * (num_qubits - 1), -9/8)
    Z_N1_term = ("I" * (num_qubits - 1) + "Z", -9/8)
    He = [
        Z_0_term,
        Z_N1_term
    ]

    for i in range(1, num_qubits - 1):
        Z_n_term = ("I" * i + "Z" + "I" * (num_qubits - i - 1), -3/4)
        He.append(Z_n_term)
    
    for i in range(num_qubits - 1):
        Z_nZ_n1_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), -3/8)
        He.append(Z_nZ_n1_term)

    # Generate Hb
    X_0_term = ("X" + "I" * (num_qubits - 1), -x/2 * (3))
    X_0_Z_1_term = ("XZ" + "I" * (num_qubits - 2), -x/2)
    X_N1_term = ("I" * (num_qubits - 1) + "X", -x/2 * (3))
    X_N1_Z_N2_term = ("I" * (num_qubits - 2) + "ZX", -x/2)
    Hb = [
        X_0_term,
        X_0_Z_1_term,
        X_N1_term,
        X_N1_Z_N2_term
    ]

    for i in range(1, num_qubits - 1):
        X_term = ("I" * i + "X" + "I" * (num_qubits - i - 1), -9*x/8)
        # 3Z_{n-1}*X_n
        ZX_term = ("I" * (i - 1) + "ZX" + "I" * (num_qubits - i - 1), -3*x/8)
        # 3Z_{n+1}*X_n
        XZ_term = ("I" * i + "XZ" + "I" * (num_qubits - i - 2), -3*x/8)
        ZXZ_term = ("I" * (i - 1) + "ZXZ" + "I" * (num_qubits - i - 2), -x/8)

        Hb.extend(
            [
                X_term,
                ZX_term,
                XZ_term,
                ZXZ_term
            ]
        )

    op = SparsePauliOp.from_list(He + Hb)
    return op.to_matrix()

ham = generate_hamiltonian(11, 2)

# print(ham.shape)

full_circ = Circuit.from_file("ensemble_benchmarks/lgt_11.qasm")
full_circ.unfold_all()
# target = full_circ.get_unitary()
# full_qc = get_qcirc(full_circ)
full_sv = full_circ.get_statevector(StateVector.zero(11))

# shots = 1048576  
# target_counts = AerSimulator().run(full_qc, shots=shots).result().get_counts()
# target_sv = get_sv(target_counts)
true_val = get_obs(get_density_matrix(full_sv.numpy))
print("True Value: ", true_val)
full_circ_val = -3.986027240753174
noisy_full_circ_val = -3.9144861698150635


if __name__ == '__main__':
    # Directory containing pickle files
    data_dir = 'ensemble_costs'
    circ_name = 'lgt_11'
    full_path = f"{data_dir}_{circ_name}_obs_noisy"

    # Collect all pickle files
    pickle_files = glob.glob(os.path.join(full_path, '*.pkl'))

    # Store data for plotting
    ens_sizes = []
    cost_means = []
    cost_maxes = []
    cost_mins = []

    for fname in sorted(pickle_files):
        filename = os.path.basename(fname)
        tol = filename.split('_')[0]
        ens_size = filename.split('_')[1].split('.')[0]
        ens_size = int(ens_size)
        print(f"Processing file: {filename} with tol={tol} and ens_size={ens_size}")
        ens_sizes.append(ens_size)
        with open(fname, 'rb') as f:
            costs = pickle.load(f)
            cost_means.append(np.mean(costs))
            cost_maxes.append(np.max(costs))
            cost_mins.append(np.min(costs))


    # Sort the data by ensemble sizes
    ens_sizes, cost_means, cost_maxes, cost_mins = zip(*sorted(zip(ens_sizes, cost_means, cost_maxes, cost_mins)))

    # Plotting
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    color = 'blue'
    ax.plot(ens_sizes, cost_means, label='Mean Cost', marker='o', color=color)
    ax.fill_between(ens_sizes, cost_mins, cost_maxes, color=color, alpha=0.2, label='Min-Max Range')
    ax.hlines(y=noisy_full_circ_val, xmin=1, xmax=max(ens_sizes), color='red', linestyle='--', label='Full Circuit')
    ax.hlines(y=true_val, xmin=1, xmax=max(ens_sizes), color='black', linestyle='--', label='True Value')
    ax.legend()
    ax.set_xlabel('Ensemble Size', fontdict={"size": 24})
    ax.set_ylabel('Magnitude w/ Noise', fontdict={"size": 24})

    fig.savefig(f'{circ_name}_mag_noisy.png', dpi=300)