from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import pickle
import random 

from util import load_circuit, load_compiled_circuits, get_unitary, save_unitaries
from util import load_unitaries, save_send_unitaries, save_target
from util import load_sent_unitaries, load_compiled_circuits_varied, save_compiled_unitaries_varied

import multiprocessing as mp

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator

def get_numpy_circ(circ: Circuit):
    return circ.get_unitary().numpy

def get_numpy(unitary: UnitaryMatrix):
    return unitary.numpy

def get_distance(circ: Circuit):
    global target
    cost_1 = HilbertSchmidtCostGenerator().calc_cost(circ, target)
    # assert(np.allclose(cost_1, cost_2))
    return cost_1

def get_cnot_count(circ: Circuit):
    return circ.count(CNOTGate())

# Circ 
if __name__ == '__main__':
    global basic_circ
    global target

    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])

    orig_circ = load_circuit(circ_name)

    orig_count = get_cnot_count(orig_circ)

    # calc_bias = "_calc_bias"
    calc_bias = ""
    circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min_post_opt{calc_bias}") 

    with mp.Pool() as pool:
        # tols = pool.map(get_distance, circs)
        cnot_counts = pool.map(get_cnot_count, circs)
        # unitaries = pool.map(get_numpy_circ, circs)

    
    print("Orig CNOT Count: ", orig_count)

    # print("Tols:", np.mean(tols), np.std(tols))
    print("CNOT Counts:", np.mean(cnot_counts), np.std(cnot_counts))

    # save_compiled_unitaries_varied(unitaries, circ_name, tol, variance)