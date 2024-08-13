from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, PermutationGate

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator

from bqskit.passes import ToVariablePass

from bqskit.ir.gates import VariableUnitaryGate, CNOTGate

from bqskit.compiler.machine import MachineModel

from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler

from util import load_circuit, save_circuits

from bqskit import enable_logging

enable_logging(True)

# Circ 
if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    compiler = Compiler(num_workers=256)

    
    circ: Circuit = load_circuit(circ_name)

    # model = MachineModel(circ.num_qudits, gate_set={CNOTGate(), VariableUnitaryGate(1)})

    initial_pass = [
        ToVariablePass(convert_all_single_qudit_gates=True)
    ]

    circ_2 = compiler.compile(circ, initial_pass)

    print(circ_2.gate_counts, flush=True)

    err_thresh = 10 ** (-1 * tol)

    extra_err_thresh = np.sqrt(1e-3 * err_thresh)

    out_circ, pi, pf = compile(circ_2, optimization_level=3, max_synthesis_size=3, synthesis_epsilon=extra_err_thresh, error_threshold=np.sqrt(err_thresh), error_sim_size=8, with_mapping=True, compiler=compiler)

    loc = tuple(range(circ.num_qudits))

    # Try all permutations
    circ_0 = out_circ.copy()

    circ_0.insert_gate(0, PermutationGate(out_circ.num_qudits, pi), loc)
    circ_0.append_gate(PermutationGate(out_circ.num_qudits, pf), loc)

    # circ_1 = out_circ.copy()

    pi_inds = [pi.index(i) for i in range(out_circ.num_qudits)]
    pf_inds = [pf.index(i) for i in range(out_circ.num_qudits)]

    # circ_1.insert_gate(0, PermutationGate(out_circ.num_qudits, pi_inds), loc)
    # circ_1.append_gate(PermutationGate(out_circ.num_qudits, pf_inds), loc)

    circ_2 = out_circ.copy()

    circ_2.insert_gate(0, PermutationGate(out_circ.num_qudits, pi_inds), loc)
    circ_2.append_gate(PermutationGate(out_circ.num_qudits, pf), loc)

    # circ_3 = out_circ.copy()

    # circ_3.insert_gate(0, PermutationGate(out_circ.num_qudits, pi), loc)
    # circ_3.append_gate(PermutationGate(out_circ.num_qudits, pf_inds), loc)

    target = circ.get_unitary()

    dist0 = HilbertSchmidtCostGenerator().calc_cost(circ_0, target)
    dist2 = HilbertSchmidtCostGenerator().calc_cost(circ_2, target)

    # dist0 = circ_0.get_unitary().get_distance_from(circ.get_unitary())
    # dist1 = circ_1.get_unitary().get_distance_from(circ.get_unitary())
    # dist2 = circ_2.get_unitary().get_distance_from(circ.get_unitary())
    # dist3 = circ_3.get_unitary().get_distance_from(circ.get_unitary())

    print(dist0, dist2)

    print("Orig Count", circ.count(CNOTGate()))
    print("New Count", out_circ.count(CNOTGate())) 

    save_circuits([(out_circ, pi, pf), circ_0, circ_2], circ_name, tol, ignore_timestep=True, extra_str="_opt3")
