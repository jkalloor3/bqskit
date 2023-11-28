from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from pathlib import Path

import glob

import matplotlib.pyplot as plt
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator

def get_upperbound_error(unitaries: list[UnitaryMatrix], target):
    np_uns = [u.numpy for u in unitaries]
    mean = np.mean(np_uns, axis=0)
    errors = [u.get_distance_from(target, 1) for u in unitaries]
    mean_error = np.mean(errors)
    variance = np.mean([u.get_distance_from(mean, 1) for u in unitaries])
    return np.sqrt(mean_error), np.sqrt(variance)

def get_bias_var_covar(sub_ensemble: list[UnitaryMatrix], all_unitaries: list[UnitaryMatrix], target):
    sub_mean = np.mean(sub_ensemble, axis=0)


# Circ 
if __name__ == '__main__':
    circ_type = argv[1]

    if circ_type == "TFIM":
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    else:
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)

    method = argv[2]
    # Store approximate solutions
    dir = f"ensemble_approx_circuits/{method}_tighter/{circ_type}/*.pickle"

    circ_files = glob.glob(dir)

    all_circs = []
    all_utries = []

    for circ_file in circ_files:
        circ: Circuit = pickle.load(open(circ_file, "rb"))
        all_circs.append(circ)
        all_utries.append(circ.get_unitary())


    print(get_stats(all_utries, target))


    






    # for seed in range(1, 500):
        # out_circ = compile(target, optimization_level=3, error_threshold=err_thresh, seed=seed)

        # if out_circ not in synth_circs:
        #     synth_circs.append(out_circ)