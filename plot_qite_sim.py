import os
import pickle
import glob
import numpy as np
from sys import argv
from bqskit.ir import Circuit
from bqskit.qis import StateVector

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from util import get_density_matrix

def generate_tfim_hamiltonian(num_qubits: int) -> SparsePauliOp:
    '''
    1D Transverse Field Ising Model Hamiltonian:
    H = J * sum_{i=0}^{N-2} Z_i Z_{i+1} + mu_x * sum_{i=0}^{N-1} X_i
    '''

    Jz = 1.0
    mu_x = 1.0

    He = []
    Hb = []

    for i in range(num_qubits - 1):
        Z_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), Jz)
        He.append(Z_term)

    for i in range(num_qubits):
        X_term = ("I" * i + "X" + "I" * (num_qubits - i - 1), mu_x)
        Hb.append(X_term)

    op = SparsePauliOp.from_list(He + Hb)
    return op.to_matrix()


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

# print(ham.shape
# print("True Value: ", true_val)
# full_circ_val = -3.986027240753174
# noisy_full_circ_val = -3.9144861698150635

if __name__ == '__main__':
    # Directory containing pickle files
    data_dir = 'ensemble_costs'
    circ_name_text = "QITE_8_{timestep}"
    max_tol = float(argv[1]) if len(argv) > 1 else 1.0
    use_noise = bool(int(argv[2])) if len(argv) > 2 else True
    use_qp = bool(int(argv[3])) if len(argv) > 3 else False

    ham = generate_tfim_hamiltonian(8)
    limit_val = -9.83795144745944
    noisy_vals = [-5.981689453125, -8.36303, -8.3777, -7.9910888671875, -7.62469482421875, -7.21673583984375, -6.80511474609375]
    circ_vals = [-6.2]

    fig, ax = plt.subplots(figsize=(10, 6))

    true_vals = []

    if use_noise:
        noise_text = "_noisy"
    else:
        noise_text = ""

    if use_qp:
        noise_text += "_qp"

    avg_costs = []
    max_costs = []
    min_costs = []

    for timestep in range(7):
        circ_name = circ_name_text.format(timestep=timestep)
        full_circ = Circuit.from_file(f"ensemble_benchmarks/{circ_name}.qasm")
        full_circ.remove_all_measurements()
        full_sv = full_circ.get_statevector(StateVector.zero(full_circ.num_qudits))

        true_val = get_obs(get_density_matrix(full_sv.numpy))
        print(f"True Value: {true_val}")

        true_vals.append(true_val)
        
        full_path = f"{data_dir}_{circ_name}_obs{noise_text}_mpi"
        print(f"Full Path: {full_path}")

        # Collect all pickle files
        pickle_files = glob.glob(os.path.join(full_path, f"{max_tol}_64_*.pkl"))
        if len(pickle_files) == 0:
            pickle_files = glob.glob(os.path.join(full_path, f"{max_tol}_8_*.pkl"))
        
        if len(pickle_files) == 0:
            print(f"No pickle files found for {full_path} with max_tol {max_tol}.")
            continue

        all_costs = []

        for fname in sorted(pickle_files):
            with open(fname, 'rb') as f:
                costs = pickle.load(f)
                all_costs.extend(costs)

        # Sort by ens_size
        avg_costs.append(np.mean(all_costs))
        max_costs.append(np.max(all_costs))
        min_costs.append(np.min(all_costs))

    # Plotting
    # plt.figure(figsize=(10, 6))

    if use_qp:
        label = " (QP)"
    else:
        label = ""
    
    if use_noise:
        label += " w/ Noise"

    ax.plot(list(range(7)), avg_costs, label=f"Ensemble{label}", marker='o', color="blue")
    ax.fill_between(list(range(7)), min_costs, max_costs, color="blue", alpha=0.2)

    if use_noise:
        ax.plot(list(range(7)), noisy_vals, color='green', linestyle='--', label='Full Circuit')
    else:
        ax.plot(list(range(7)), circ_vals, color='green', linestyle='--', label='Full Circuit')
    ax.plot(list(range(7)), true_vals, color='red', label='True Value')
    ax.hlines(y=limit_val, xmin=0, xmax=6, color='black', label='GS Energy')
    ax.legend()
    ax.set_xlabel('QITE Timestep')
    ax.set_ylabel('Magnitude')
    fig.savefig(f'QITE_mag{noise_text}_all_timesteps.png', dpi=300)