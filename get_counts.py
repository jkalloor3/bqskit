from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix

from util import load_circuit, load_compiled_circuits, get_unitary, save_unitaries, load_unitaries, save_send_unitaries, save_target, load_sent_unitaries

import multiprocessing as mp

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator

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

    # orig_circ = load_circuit(circ_name)
    # target = orig_circ.get_unitary()
    # circs = load_compiled_circuits(circ_name, tol, timestep)

    # if circ_name.endswith("q"):
    #     circ_name = circ_name[:-1]

    unitaries = load_sent_unitaries(circ_name, tol)

    
    # unitaries = load_unitaries(circ_name, tol, timestep)

    with mp.Pool() as pool:
        # tols = pool.map(get_distance, circs)
        # cnot_counts = pool.map(get_cnot_count, circs)
        unitaries = pool.map(get_numpy, unitaries)

    # # get shortest circuits

    # sorted_counts = sorted(zip(tols, cnot_counts), key=lambda x: x[1])

    save_send_unitaries(unitaries, circ_name, tol)
    # save_target(target, circ_name)

    # print("Final Counts: ", sorted_counts[:10])
    # print("Avg Dist:", np.mean(tols)) 