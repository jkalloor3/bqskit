import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import *
from bqskit.qis import UnitaryMatrix
from bqskit.compiler import Compiler
import itertools

from util import normalized_gp_frob_cost, PauliTwirlPass, load_block, load_compiled_block_circuits

pauli_pass = PauliTwirlPass(gates_to_twirl=[RZGate(), RXGate(), RYGate()], 
                            num_twirls=10)

coherent_gates = [RZGate().name, RXGate().name, RYGate().name]

def apply_overrotation(circ: Circuit, parameter: float) -> Circuit:
    circuit = circ.copy()
    for op in circuit.operations():
        if op.gate.name in coherent_gates:
            mod = np.random.rand(op.num_qudits) * parameter
            new_params = op.params + mod
            op.params = new_params
    
    return circuit

def get_average_dist(target: UnitaryMatrix, 
                     circs: list[Circuit], 
                     param: float) -> float:
    total_dist = 0.0
    for circ in circs:
        # Apply overrotation
        new_circ = apply_overrotation(circ, 0.1)
        dist = normalized_gp_frob_cost(target, new_circ.get_unitary())
        total_dist += dist
    return total_dist / len(circs)

def randomized_compilation(circ: Circuit) -> list[Circuit]:
    # Get an ensemble of randomly compiled circuits
    compiler = Compiler()
    workflow = [
        pauli_pass
    ]
    _, data = compiler.compile(circ, workflow, request_data=True)
    return data["twirled_circuits"]

if __name__ == '__main__':
    # Create a circuit
    circ_name = "adder9"
    block_num = "0"
    tol = 3.0
    rotation_param = 10 ** (-tol + 1)
    circ_file = load_block(circ_name, block_num)
    circ = Circuit.from_file(circ_file)

    # Get the target unitary
    target = circ.get_unitary()

    orig_avg_dist = get_average_dist(target, [circ], rotation_param)
    print("Original Average Distance: ", orig_avg_dist)

    # Get the randomized compilation ensemble
    rc_circs = randomized_compilation(circ, num_circs=500)
    rc_dist = get_average_dist(target, rc_circs, rotation_param)
    print("Randomized Compilation Average Distance: ", rc_dist)

    # Load the ensemble
    ens = load_compiled_block_circuits(circ_name, block_num, tol)

    # Randomly select 500 circuits from the ensemble
    rand_inds = np.random.choice(len(ens), 500, replace=False)
    ens = [ens[i] for i in rand_inds]

    ens_dist = get_average_dist(target, ens, rotation_param)
    print("Ensemble Average Distance: ", ens_dist)


    circs_2 = [randomized_compilation(c, 1) for c in ens]
    circs_2 = itertools.chain.from_iterable(circs_2)

    ens_rc_dist = get_average_dist(target, circs_2, rotation_param)
    print("Ensemble + RC Average Distance: ", ens_rc_dist)