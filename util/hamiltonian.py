from qiskit.quantum_info import SparsePauliOp
from bqskit.qis import StateVector
import numpy as np
import pickle

def get_obs(dm: np.ndarray, ham: np.ndarray) -> float:
    '''
    Get the observable for the output density matrix.
    '''
    return np.real(np.trace(ham @ dm))


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

def generate_heisenberg_hamiltonian(num_qubits: int) -> np.ndarray:
    '''
    Heisenberg Hamiltonian:
    H = J * sum_{i=0}^{N-2} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    '''

    J = 1.0
    He = []
    
    for i in range(num_qubits - 1):
        XX_term = ("I" * i + "XX" + "I" * (num_qubits - i - 2), J)
        YY_term = ("I" * i + "YY" + "I" * (num_qubits - i - 2), J)
        ZZ_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), J)
        He.extend([XX_term, YY_term, ZZ_term])

    # Add a Z term to all qubits
    for i in range(num_qubits):
        Z_term = ("I" * i + "Z" + "I" * (num_qubits - i - 1), J)
        He.append(Z_term)

    op = SparsePauliOp.from_list(He)
    return op.to_matrix()

def generate_hamiltonian(circ_name: str, num_qubits: int) -> np.ndarray:
    '''
    Generate the Hamiltonian for a given circuit name.
    '''
    if circ_name.startswith('lgt'):
        return generate_lgt_hamiltonian(num_qubits, 2)
    elif circ_name.startswith('QITE'):
        return generate_tfim_hamiltonian(num_qubits)
    elif circ_name.startswith('heisenberg'):
        return generate_heisenberg_hamiltonian(num_qubits)
    elif ("Fermi" in circ_name or "H_" in circ_name):
        circ_name = circ_name.replace("_long", "")
        # return generate_tfim_hamiltonian(num_qubits)
        return pickle.load(open(f"out_hamiltonians/{circ_name}.pkl", "rb"))
    else:
        return None

def get_lgt_init_state(num_qubits: int) -> StateVector:
    '''
    Generate the initial state vector for LGT circuits.
    '''
    # TODO: Fix this once Mohan provides the correct initial state
    return StateVector.zero(num_qubits)

def generate_init_state(circ_name: str, num_qubits: int) -> StateVector:
    '''
    Generate the initial state vector for a given circuit name.
    '''
    if circ_name.startswith('lgt'):
        return get_lgt_init_state(num_qubits)
    elif circ_name.startswith('QITE'):
        return StateVector.zero(num_qubits)
    elif circ_name.startswith('heisenberg'):
        return StateVector.zero(num_qubits)
    elif ("Fermi" in circ_name or "H_" in circ_name):
        # LiH, H_2, H_2O etc.
        circ_name = circ_name.replace("_long", "")
        # return generate_tfim_hamiltonian(num_qubits)
        return StateVector(pickle.load(open(f"initial_states/{circ_name}.pkl", "rb")))
    else:
        return None