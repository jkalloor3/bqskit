from bqskit.qis.unitary import UnitaryMatrix
import numpy as np

def get_ensemble_mean(ensemble: list[UnitaryMatrix]):
    ensemble_mean = np.mean(ensemble, axis=0)
    return ensemble_mean

def get_upperbound_error_mean(unitaries: list[UnitaryMatrix], target):
    np_uns = [u.numpy for u in unitaries]
    mean = np.mean(np_uns, axis=0)
    errors = [u.get_frobenius_distance(target) for u in unitaries]

    max_error = np.mean(errors)
    max_variance = np.mean([u.get_frobenius_distance(mean) for u in unitaries])

    return np.sqrt(max_error), np.sqrt(max_variance), mean

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