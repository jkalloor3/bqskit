'''
Library to calculate the trace distance between two density matrices.
'''
import numpy as np
from bqskit.qis import UnitaryMatrix
from bqskit.utils.math import canonical_unitary

'''
Calculate the trace distance between two density matrices
'''

def trace_distance(rho: np.ndarray[np.complex128], sigma: np.ndarray[np.complex128]) -> np.float64:
    '''
    Calculate the trace distance between two density matrices. These
    matrices are Hermitian.
    '''
    diff = rho - sigma
    eigvals, _ = np.linalg.eigh(diff)
    return 0.5 * np.sum(np.abs(eigvals))

def get_density_matrix(vector: np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    return np.array(np.outer(vector, vector.conj()), dtype=np.complex128)

def get_average_density_matrix(vectors: list[np.ndarray[np.complex128]]) -> np.ndarray[np.complex128]:
    mat = 0
    for vector in vectors:
        mat += get_density_matrix(vector) / len(vectors)
    return mat

'''
Calculate the TVD between two probability distributions.
'''

def tvd(p: np.ndarray[np.complex128], q: np.ndarray[np.complex128]) -> np.float64:
    return 0.5 * np.sum(np.abs(p - q))

def cross_entropy_fidelity(pure_probs, sampled_probs, DIM: int) -> np.float64:
    e_u = np.sum(pure_probs**2)
    u_u = np.sum(pure_probs) / DIM
    m_u = np.sum(pure_probs * sampled_probs)
    y = m_u - u_u
    x = e_u - u_u

    numerator = x * y
    denominator = x ** 2

    return numerator / denominator

def cross_entropy_fidelity_dict(P_counts: dict[str, int], Q_counts: dict[str, int]) -> np.float64:
    # Convert counts to probability distributions
    total_P = sum(P_counts.values())
    total_Q = sum(Q_counts.values())

    num_qubits = max(len(k) for k in P_counts.keys())
    dim = 2 ** num_qubits
    
    P_dist = {k: v / total_P for k, v in P_counts.items()}
    Q_dist = {k: v / total_Q for k, v in Q_counts.items()}
    
    # Ensure all keys in P are also in Q, adding missing keys with a small epsilon
    all_keys = set(P_dist.keys()).union(Q_dist.keys())
    
    P = np.array([P_dist.get(k, 0) for k in all_keys])
    Q = np.array([Q_dist.get(k, 0) for k in all_keys])  # Avoid log(0)

    
    # Compute cross-entropy fidelity with pure and sampled probs
    return cross_entropy_fidelity(P, Q, dim)

def tvd_dict(p: dict[str, int], q: dict[str, int]) -> np.float64:
    shots = sum(p.values())
    shots_2 = sum(q.values())
    if shots != shots_2:
        print("Shots do not match")
        print("Shots: ", shots)
        print("Shots 2: ", shots_2)
    assert shots == shots_2
    p = {k: v/shots for k, v in p.items()}
    q = {k: v/shots for k, v in q.items()}
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

def frobenius_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the Frobenius distance between two unitaries
    '''
    diff = utry - target
    # This is Frob(u - v)
    inner = np.real(np.einsum("ij,ij->", diff, diff.conj()))
    cost = np.sqrt(inner)

    return cost

def normalized_frob_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    # This is Frob(u - v)
    cost = frobenius_cost(utry, target)

    N = utry.shape[0]
    cost = cost / np.sqrt(2 * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost

def normalized_gp_frob_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    gp_correction = target.get_target_correction_factor(utry)
    utry = utry * gp_correction

    return normalized_frob_cost(utry, target)

def gp_frobenius_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    gp_correction = target.get_target_correction_factor(utry)
    utry = utry * gp_correction

    return frobenius_cost(utry, target)

def normalized_frob_dist_func(target: UnitaryMatrix) -> callable:
    def calc_frob_dist(mat: np.ndarray) -> np.float64:
        return normalized_frob_cost(target, mat)
    return calc_frob_dist