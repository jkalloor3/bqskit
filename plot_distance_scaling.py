import os
import glob
import pickle
import numpy as np

from util import load_circuit, get_density_matrix

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp


def generate_lgt_hamiltonian(num_qubits: int, x: int) -> np.ndarray:
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

def generate_tfim_hamiltonian(num_qubits: int) -> np.ndarray:
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

def get_obs(dm: np.ndarray, ham: np.ndarray) -> float:
    '''
    Get the observable for the output density matrix.
    '''
    return np.real(np.trace(ham @ dm))

TRACE_DISTANCE = False
if __name__ == '__main__':
    if TRACE_DISTANCE:
        circ_names = [
            "qaoa10",
            "qpe_11"
        ]
    else:
        circ_names = [
            "lgt_11",
            "QITE_8_0"
        ]
        

    folder_form = "ensemble_dms_{circ_name}/"

    # Find all folders matching the pattern
    folders = [folder_form.format(circ_name=circ_name) for circ_name in circ_names]
    circ_names = [circ_name for circ_name in circ_names if os.path.exists(folder_form.format(circ_name=circ_name))]
    folders = [folder for folder in folders if os.path.exists(folder)]

    circ_folders = zip(circ_names, folders)


    x_vals = []
    y_vals = []

    fig, axs = plt.subplots(figsize=(10, 6))

    for circ_name, folder in circ_folders:
        pickle_files = glob.glob(os.path.join(folder, '*.pkl'))
        x_vals = []
        y_vals = []
        if circ_name.startswith('lgt'):
            full_circ = load_circuit(circ_name)
            state = np.zeros(2 ** full_circ.num_qudits)
            state[0] = 1.0
            sv_out = full_circ.get_statevector(state)
            dm = get_density_matrix(sv_out.numpy)
            ham = generate_lgt_hamiltonian(full_circ.num_qudits, 2)
            true_val = get_obs(dm, ham)
            print(f"True value for {circ_name}: {true_val}")
        elif circ_name.startswith('QITE'):
            full_circ = load_circuit(circ_name)
            full_circ.remove_all_measurements()
            state = np.zeros(2 ** full_circ.num_qudits)
            state[0] = 1.0
            sv_out = full_circ.get_statevector(state)
            dm = get_density_matrix(sv_out.numpy)
            ham = generate_tfim_hamiltonian(full_circ.num_qudits)
            true_val = get_obs(dm, ham)
            print(f"True value for {circ_name}: {true_val}")
        for pf in pickle_files:
            with open(pf, 'rb') as f:
                item = pickle.load(f)
                y = item[0]
                x = np.mean(item[1])
                if TRACE_DISTANCE:
                    x_vals.append(x)
                    y_vals.append(y)
                else:
                    x_vals.append(np.abs(true_val - x))
                    y_vals.append(np.abs(true_val - y))
        axs.scatter(x_vals, y_vals, label=circ_name)

    if TRACE_DISTANCE:
        axs.set_xlabel('Average Trace Distance')
        axs.set_ylabel('Trace Distance of Ensemble')
    else:
        axs.set_xlabel('Average Hamiltonian Observable Error')
        axs.set_ylabel('Hamiltonian Observable Error of Channel')
    axs.set_yscale('log')
    axs.set_xscale('log')
    # axs.set_title('Trace Distance Scaling for Circuits')
    axs.legend()
    axs.grid(True, which='both', linestyle='--', linewidth=0.5)

    if TRACE_DISTANCE:
        plt.savefig('trace_distance_scaling.png', dpi=300)
    else:
        plt.savefig('hamiltonian_scaling.png', dpi=300)