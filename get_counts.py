from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.ir.gates import CNOTGate

from util import load_circuit, load_compiled_circuits

import multiprocessing as mp

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator

def get_distance(circ: Circuit):
    global target
    cost_1 = HilbertSchmidtCostGenerator().calc_cost(circ, target)
    cost_2 = circ.get_unitary().get_frobenius_distance(target)
    print(cost_1, cost_2)
    assert(np.allclose(cost_1, cost_2))
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

    orig_circ = load_circuit(circ_name)
    target = orig_circ.get_unitary()
    circs = load_compiled_circuits(circ_name, tol, timestep)[:10]

    with mp.Pool() as pool:
        tols = pool.map(get_distance, circs)
        cnot_counts = pool.map(get_cnot_count, circs)

    # get shortest circuits

    sorted_counts = sorted(zip(tols, cnot_counts), key=lambda x: x[1])


    print("Final Counts: ", sorted_counts[:10])