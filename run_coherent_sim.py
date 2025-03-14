import numpy as np
import multiprocessing as mp
from bqskit.ir import Circuit, Operation, CircuitPoint
from bqskit.ir.gates import *
from bqskit.passes import ToU3Pass
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.qis import UnitaryMatrix, StateVector
from bqskit.compiler import Compiler
from bqskit.ext import bqskit_to_qiskit
from qiskit import transpile
import itertools

from util import normalized_gp_frob_cost, PauliTwirlPass, load_block, load_compiled_block_circuits

rotation_gates = [RXXGate(), RYGate(), RZGate(), CRXGate()]
rotation_gates_names = [g.name for g in rotation_gates]

def lower_to_rx_ry_rz(circ: Circuit) -> Circuit:
    """
    Convert all single qubit gates to RX, RY, and RZ gates
    """
    qcirc = bqskit_to_qiskit(circ)
    qcirc = transpile(qcirc, basis_gates=['rx', 'ry', 'rz', 'cx'], 
                      optimization_level=2)
    
    qasm_str = qcirc.qasm()
    lang = OPENQASM2Language()
    circ.become(lang.decode(qasm_str))

def apply_overrotation(circ: Circuit, parameter: float) -> Circuit:
    circuit = circ.copy()
    # Replace all CZ gates with CRZ(pi) gates
    for cycle, op in circ.operations_with_cycles():
        if isinstance(op.gate, CXGate):
            pt = CircuitPoint(cycle, op.location[0])
            circuit.replace_gate(pt, CRXGate(), op.location, [np.pi])
    for op in circuit.operations():
        if op.gate.name in rotation_gates_names:
            # print("err", end="")
            # Overrations are much more erroneous on multi-qubit gates
            op.params[0] += parameter * (op.num_qudits ** 10)
    
    return circuit

def trace_distance(rho: np.ndarray[np.complex128], sigma: np.ndarray[np.complex128]) -> np.float64:
    '''
    Calculate the trace distance between two density matrices. These
    matrices are Hermitian.
    '''
    diff = rho - sigma
    eigvals, _ = np.linalg.eigh(diff)
    return 0.5 * np.sum(np.abs(eigvals))

def get_circ_rho(circ: Circuit, random_states: list[StateVector]) -> list[np.ndarray[np.complex128]]:
    # out_rho = circ.get_statevector(StateVector.random(circ.num_qudits)).numpy
    out_svs = [circ.get_statevector(sv).numpy for sv in random_states]
    # rho = np.outer(out_rho, out_rho.conj())
    out_rhos = [np.outer(sv, sv.conj()) for sv in out_svs]
    return out_rhos

# def get_probs(target: UnitaryMatrix, 
#                      circs: list[Circuit], 
#                      param: float) -> float:
#     # total_dist = 0.0
#     all_probs = []
#     for circ in circs:
#         # Apply overrotation
#         new_circ = apply_overrotation(circ, param)
#         print(new_circ.gate_counts)
#         probs = get_circ_prob(new_circ)
#         all_probs.append(probs)
#     avg_probs = np.mean(all_probs, axis=0)
#     print("Avg Probs: ", avg_probs.shape)
#     return avg_probs

def get_noisy_rhos(circ: Circuit, param: float, random_states: list[StateVector]) -> list[np.ndarray[np.complex128]]:
    new_circ = apply_overrotation(circ, param)
    rho = get_circ_rho(new_circ, random_states=random_states)
    return rho


def get_avg_rho(circs: list[Circuit], param: float, random_states: list[StateVector]) -> np.ndarray[np.complex128]:
    with mp.Pool(4) as pool:
        all_rhos = pool.starmap(get_noisy_rhos, [(circ, param, random_states) for circ in circs])
    avg_rhos = np.mean(all_rhos, axis=0)
    return avg_rhos

def randomized_compilation(circ: Circuit, num_circs: int = 1, 
                           compiler: Compiler = None) -> list[Circuit]:
    # Get an ensemble of randomly compiled circuits
    pauli_pass =  PauliTwirlPass(gates_to_twirl=[CXGate()],
                            num_twirls=num_circs)
    workflow = [
        pauli_pass
    ]
    _, data = compiler.compile(circ, workflow, request_data=True)
    return data["twirled_circuits"]

def flood_circ(circ: Circuit) -> Circuit:
    flooded_cnot = Circuit(2)
    flooded_cnot.append_gate(CZGate(), (0, 1))
    flooded_cnot.append_gate(RZGate(), (0,), [0])
    flooded_cnot.append_gate(RZGate(), (1,), [0])
    init_u3_circ = Circuit(circ.num_qudits)
    for i in range(circ.num_qudits):
        init_u3_circ.append_gate(U3Gate(), (i,), [0, 0, 0])
    circ.insert_circuit(0, init_u3_circ, tuple(range(circ.num_qudits)))
    for cycle, op in circ.operations_with_cycles():
        if isinstance(op.gate, CZGate):
            op_pt = CircuitPoint(cycle, op.location[0])
            circ.replace_with_circuit(op_pt, flooded_cnot.copy(), as_circuit_gate=True)
    circ.unfold_all()
    # ToU3Pass.run_group_circ(circ)
    return circ

if __name__ == '__main__':

    # Start compiler
    compiler = Compiler(num_workers=4)

    random_states = [StateVector.random(8) for _ in range(10)]


    # print("Pauli Pass: ", pauli_pass.twirl_set)

    # Create a circuit
    circ_name = "mult_16"
    block_num = "05"
    tol = 4.0
    single_rotation_param = 10 ** (-tol * 2)
    circ_file = load_block(circ_name, block_num, extra="_tket")
    circ = Circuit.from_file(circ_file)
    # circ = lower_to_rx_ry_rz(circ)

    # Get the target unitary
    target = circ.get_unitary()
    # true_probs = get_circ_prob(circ)
    rhos = get_circ_rho(circ, random_states=random_states)

    lower_to_rx_ry_rz(circ)


    print("New Dist: ", normalized_gp_frob_cost(target, circ.get_unitary()))

    print(circ.gate_counts)

    # orig_probs = get_probs(target, [circ], single_rotation_param)
    noisy_rhos = get_avg_rho([circ], single_rotation_param, 
                             random_states=random_states)
    orig_tds = [trace_distance(rho, noisy_rho) for rho, noisy_rho in zip(rhos, noisy_rhos)]
    avg_orig_td = np.max(orig_tds)
    print("Original Trace Distance: ", avg_orig_td)

    # Get the randomized compilation ensemble
    rc_circs = randomized_compilation(circ, num_circs=5000, compiler=compiler)

    # Sample at difference num_samples
    sample_sizes = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]

    for num_samples in sample_sizes:
        print("Num Samples: ", num_samples)
        rand_inds = np.random.choice(len(rc_circs), num_samples, replace=False)
        sampled_circs = [rc_circs[i] for i in rand_inds]
        # rc_probs = get_probs(target, rc_circs, single_rotation_param)
        rc_rhos = get_avg_rho(sampled_circs, single_rotation_param, 
                              random_states=random_states)
        rc_tds = [trace_distance(rho, rc_rho) for rho, rc_rho in zip(rhos, rc_rhos)]
        avg_rc_td = np.max(rc_tds)
        print("Randomized Compilation Trace Dist: ", avg_rc_td)

    compiler.close()


    # Unitary dists
    # dists = [normalized_gp_frob_cost(target, c.get_unitary()) for c in rc_circs[:1]]
    # print("RC Dist: ", np.mean(dists))
    # print("RC Circs: ", len(rc_circs))
    # print(rc_circs[0].gate_counts)
    # # rc_probs = get_probs(target, [rc_circs[0]], single_rotation_param)
    # rc_rhos = get_avg_rho(rc_circs, single_rotation_param)
    # rc_tds = [trace_distance(rho, rc_rho) for rho, rc_rho in zip(rhos, rc_rhos)]
    # avg_rc_td = np.mean(rc_tds)
    # print("Randomized Compilation Trace Dist: ", avg_rc_td)

    # # Load the ensemble
    # ens = load_compiled_block_circuits(circ_name, block_num, tol)

    # # Randomly select 500 circuits from the ensemble
    # rand_inds = np.random.choice(len(ens), 500, replace=False)
    # ens = [ens[i] for i in rand_inds]

    # ens_dist = get_average_dist(target, ens, rotation_param)
    # print("Ensemble Average Distance: ", ens_dist)


    # circs_2 = [randomized_compilation(c, 1) for c in ens]
    # circs_2 = itertools.chain.from_iterable(circs_2)

    # ens_rc_dist = get_average_dist(target, circs_2, rotation_param)
    # print("Ensemble + RC Average Distance: ", ens_rc_dist)