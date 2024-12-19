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
    mat = get_density_matrix(vectors[0]) / len(vectors)
    for vector in vectors[1:]:
        mat += get_density_matrix(vector) / len(vectors)
    return mat

'''
Calculate the TVD between two probability distributions.
'''

def tvd(p: np.ndarray[np.complex128], q: np.ndarray[np.complex128]) -> np.float64:
    return 0.5 * np.sum(np.abs(p - q))

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


def normalized_frob_dist_func(target: UnitaryMatrix) -> callable:
    def calc_frob_dist(mat: np.ndarray) -> np.float64:
        return normalized_frob_cost(target, mat)
    return calc_frob_dist