import numpy as np
import time
from scipy.linalg import ldl, cholesky, schur


def trace_distance(sv1: np.ndarray, sv2: np.ndarray):
    t1 = 0
    trace = 0
    start = time.time()
    if sv1.shape[0] < 2 ** 10:
        diff_2 = np.outer(sv1, sv1.conj()) - np.outer(sv2, sv2.conj())
        eigvals = np.linalg.eigh(diff_2)[0]
        trace_2 = np.sum(np.abs(eigvals)) / 2
    else:
        trace_2 = 0
    t2 = time.time() - start

    trace_3 = 1 - np.abs(np.inner(sv1, sv2.conj())) ** 2
    trace_3 = np.sqrt(trace_3)

    t3 = time.time() - start - t1 - t2

    # diff_3 = np.outer(sv1, sv1.conj()) - np.outer(sv2, sv2.conj())
    # trace_4 = np.trace(np.abs(schur(diff_3 @ diff_3.conj().T)[0]))
    # trace_4 = trace_4 / 2
    trace_4 = 0

    t4 = time.time() - start - t1 - t2 - t3
    # print(trace_1, trace_2, trace_3)
    print("Traces: ", trace_2, trace_3, flush=True)
    return t2, t3

def generate_random_state_vector(num_qubits):
    # Generate random complex numbers
    real_parts = np.random.rand(2**num_qubits)
    imag_parts = np.random.rand(2**num_qubits)
    complex_nums = real_parts + 1j * imag_parts

    # Normalize the state vector
    state_vector = complex_nums / np.linalg.norm(complex_nums)

    return state_vector

# Example usage
for num_qubits in range(5, 30, 2):
    start = time.time()
    sv1 = np.mean([generate_random_state_vector(num_qubits) for _ in range(100)], axis=0)
    sv2 = np.mean([generate_random_state_vector(num_qubits) for _ in range(100)], axis=0)

    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)

    generation_time = time.time() - start

    ts = trace_distance(sv1, sv2)

    print("Num qubits: ", num_qubits, "Generation Time: ", generation_time, "Times: ", *ts, flush=True)