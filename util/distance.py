'''
Library to calculate the trace distance between two density matrices.
'''
import numpy as np
from bqskit.qis import UnitaryMatrix

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
    mat = get_density_matrix(vectors[0]) / len(vectors)
    for vector in vectors[1:]:
        mat += get_density_matrix(vector) / len(vectors)
    return mat

'''
Calculate the TVD between two probability distributions.
'''

def tvd(p: np.ndarray[np.complex128], q: np.ndarray[np.complex128]) -> np.float64:
    return 0.5 * np.sum(np.abs(p - q))

''' Calculate the Frobenius distance between two matrices. '''

def normalized_frob_dist(target: UnitaryMatrix, mat: np.ndarray) -> np.float64:
    diff = target - mat
    cost = np.real(np.trace(diff @ diff.conj().T))
    # Factor Frob distance by 4N^2
    N = target.shape[0] 
    cost = cost / (N * N)
    return np.sqrt(cost)

def normalized_frob_dist_func(target: UnitaryMatrix) -> callable:
    def calc_frob_dist(mat: np.ndarray) -> np.float64:
        return normalized_frob_dist(target, mat)
    return calc_frob_dist