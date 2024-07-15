from bqskit.qis.unitary import UnitaryMatrix
import numpy as np
import multiprocessing as mp

ensemble = None

def get_ensemble_mean(ensemble: list[UnitaryMatrix]):
    ensemble_mean = np.mean(ensemble, axis=0)
    return ensemble_mean

def get_upperbound_error_mean(unitaries: list[UnitaryMatrix], target):
    if isinstance(unitaries[0], UnitaryMatrix):
        np_uns = [u.numpy for u in unitaries]
    else:
        np_uns = unitaries
    mean = np.mean(np_uns, axis=0)
    errors = [u.get_frobenius_distance(target) for u in unitaries]

    max_error = np.mean(errors)
    max_variance = np.mean([u.get_frobenius_distance(mean) for u in unitaries])

    return max_error, max_variance, mean


def get_upperbound_error_mean_vec(ensemble_vec: np.ndarray[np.float128], target: np.ndarray[np.float128]):
    mean = np.mean(ensemble_vec, axis=0)
    errors = [euclidean_distance(u, target) for u in ensemble_vec]

    max_error = np.mean(errors)
    max_variance = np.mean([euclidean_distance(u, mean) for u in ensemble_vec])
    return max_error, max_variance, mean

def get_average_distance(ensemble_in: list[np.ndarray]):
    avg_cost = 0
    count = 0
    for i in range(len(ensemble_in) - 1):
        for j in range(i + 1, len(ensemble_in)):
            diff = ensemble_in[i] - ensemble_in[j]
            avg_cost +=  np.abs(np.sum(np.einsum("ij,ij->", diff.conj(), diff)))
            count += 1
    return avg_cost / count


def euclidean_distance(a, b):
    return np.sum((a - b) ** 2)


def get_chi_1_chi_2(ensemble: np.ndarray, mean: np.ndarray, mean_epsi: float):
    chi_1 = 0
    chi_2 = 0
    count_2 = 0
    for i in range(len(ensemble)):
        diff_1 = ensemble[i] - mean
        chi_1 += np.abs(np.sum(np.einsum("ij,ij->", diff_1.conj(), diff_1)))
        for j in range(len(ensemble)):
            diff_2 = ensemble[i] - ensemble[j]
            chi_2 += np.abs(np.sum(np.einsum("ij,ij->", diff_2.conj(), diff_2)))
            count_2 += 1
    mean_chi_1 = chi_1 / len(ensemble)
    mean_chi_2 = chi_2 / count_2
    return mean_chi_1 / mean_epsi, mean_chi_2 / mean_epsi


def get_average_distance_vec(ensemble_vec: np.ndarray[np.float128]):
    avg_cost = 0
    count = 0
    for i in range(len(ensemble_vec) - 1):
        for j in range(i + 1, len(ensemble_vec)):
            avg_cost += euclidean_distance(ensemble_vec[i], ensemble_vec[j])
            count += 1
    return avg_cost / count

def get_tvd_magnetization(ensemble: list[UnitaryMatrix], target):
    ensemble_mean = np.mean(ensemble, axis=0)
    final_output = ensemble_mean[:, 0]
    target_output = target[:, 0]
    diff = np.sum(np.abs(final_output - target_output))
    return diff / 2

def get_bias_var_covar(ensemble: list[UnitaryMatrix], target):
    # ensemble is of size M
    M = len(ensemble)
    ensemble_mean = np.mean(ensemble, axis=0)
    bias = UnitaryMatrix(target).get_frobenius_distance(ensemble_mean)

    var = np.mean([u.get_frobenius_distance(ensemble_mean) for u in ensemble])
    covar = 0
    for i in range(M):
        A = UnitaryMatrix(ensemble[i] - ensemble_mean, check_arguments=False)
        for j in range(M):
            if j < i:
                B = ensemble[j] - ensemble_mean
                covar += 2*np.real(np.trace(A.conj().T @ B))

    covar *= (1 /M)*(1/(M-1))

    return bias, var, covar