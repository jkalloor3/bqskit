from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
import numpy as np
import json
from itertools import chain

from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from util import load_block, load_compiled_block_circuits, load_compiled_block_circuits_qp
from util.distance import frobenius_cost, tvd, trace_distance, get_density_matrix, get_average_density_matrix

from bqskit.ext import bqskit_to_qiskit

shots = 100

def get_ensemble_mags(ens_size, random_states: list[np.ndarray] = None) -> tuple[list[np.ndarray[np.float64]], 
                                                                                 list[np.ndarray[np.float64]], 
                                                                                 np.ndarray[np.complex128] | None]:
    global all_qcircs
    global uns

    ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    ensemble_uns = np.array([uns[i] for i in ensemble_inds])
    mean_un = np.mean(ensemble_uns, axis=0)

    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    noisy_rhos = []
    noisy_probs = []
    mean_uns = []
    for random_state in random_states:
        ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
        ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
        ensemble_uns = np.array([uns[i] for i in ensemble_inds])
        mean_un = np.mean(ensemble_uns, axis=0)
        circs = get_random_init_state_circuits(ensemble, [random_state])[0]
        noisy_svs = np.array([Statevector.from_instruction(circ).data for circ in circs])
        probs = np.array([np.abs(sv)**2 for sv in noisy_svs], dtype=np.float64)
        avg_probs = np.mean(probs, axis=0)
        noisy_rho = get_average_density_matrix(noisy_svs)
        noisy_rhos.append(noisy_rho)
        noisy_probs.append(avg_probs)
        mean_uns.append(mean_un)
    mean_un = np.mean(mean_uns, axis=0)
    return noisy_rhos, noisy_probs, mean_un

def get_qcirc(circ: Circuit):
    for cycle, op in circ.operations_with_cycles():
        if op.num_qudits == 1 and not isinstance(op.gate, U3Gate):
            params = U3Gate().calc_params(op.get_unitary())
            point = CircuitPoint(cycle, op.location[0])
            circ.replace_gate(point, U3Gate(), op.location, params)

    q_circ = bqskit_to_qiskit(circ)
    return q_circ

def get_random_states(num_qubits: int, num_random_states: int = 4):
    states = []
    for i in range(4):
        state = np.random.randint(0, 2, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcircs: list[QuantumCircuit], random_states: list[np.ndarray]) -> list[list[QuantumCircuit]]:
    all_qcircs = []
    for state in random_states:
        circs = []
        for qcirc in qcircs:
            init_circ: QuantumCircuit = qcirc.copy()
            for i in range(init_circ.num_qubits):
                if state[i] == 1:
                    init_circ.h(i)
            init_circ.compose(qcirc, inplace=True)
            circs.append(transpile(init_circ, optimization_level=0))
        all_qcircs.append(circs)
    return all_qcircs

def aggregate_results(results: list):
    total_dict = {}
    for r in results:
        # print("Result: ", r)
        x = total_dict
        y = r.data.meas.get_counts()
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return total_dict

# Circ 
if __name__ == '__main__':
    global all_qcircs
    global uns

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ_name = argv[1]
    block_num = argv[2]
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    cliff = False

    circ_path = load_block(circ_name, block_num, good=True)
    initial_circ = Circuit.from_file(circ_path)
    target = initial_circ.get_unitary()
    print("Got initial circ", flush=True)
    circs = load_compiled_block_circuits_qp(circ_name, block_num, tol, num_unique_circs)
    print("Num Circs: ", len(circs), flush=True)

    if len(circs) == 0:
        print("No circuits found")
        exit(0)

    # dists = [target.get_frobenius_distance(c.get_unitary()) for c in circs[:20]]
    dists = [c[1] for c in circs]
    bqskit_circs = [c[0] for c in circs]
    uns = [c.get_unitary() for c in bqskit_circs]
    frob_dists = [frobenius_cost(target, un) for un in uns]
    print("Avg Norm. Dist: ", np.mean(dists))
    print("Avg Dist: ", np.mean(frob_dists))
    print("Original CX Count: ", initial_circ.count(CNOTGate()))

    opt_str = ""
    print("LOADED CIRCUITS", flush=True)


    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 100, 1000, 5000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    # sampler = Sampler(mode=sim)

    random_states = get_random_states(initial_circ.num_qudits, num_random_states=16)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circs = get_random_init_state_circuits([qiskit_circ], random_states)
    qiskit_circs = list(chain(*qiskit_circs))
    # qiskit_circs = [qiskit_circ]
    print("Got all circuits", flush=True)
    svs = [Statevector.from_instruction(circ).data for circ in qiskit_circs]
    rhos = [get_density_matrix(sv) for sv in svs]
    print("Len of rhos: ", len(rhos))
    true_probs = [np.abs(sv)**2 for sv in svs]
    noisy_svs = [Statevector.from_instruction(circ).data for circ in qiskit_circs]
    noisy_rhos = [get_density_matrix(sv) for sv in noisy_svs]
    noisy_dists = [trace_distance(rhos[i], noisy_rho) for i,noisy_rho in enumerate(noisy_rhos)]
    print("Noisy Distances: ", noisy_dists)
    
    
    # print("Noisy Counts: ", noisy_result_dict)
    print("Finished Running", flush=True)
    # noisy_result_dict = noisy_result.get_counts(qiskit_circ)
    base_excitations.append(0)
    noisy_excitations.append(np.mean(noisy_dists))

    all_qcircs = [get_qcirc(c) for c in bqskit_circs]
    print("Got MAP", flush=True)

    print("Runing PERFECT ENSEMBLES: ")
    print(f"Base TVD: {base_excitations[0]}, Noisy TVD {noisy_excitations[0]}")
    final_tds = []
    final_frobs = []
    final_tvds = []
    for j, ens_size in enumerate(ensemble_sizes):
        final_rhos, final_probs, mean_un = get_ensemble_mags(ens_size, random_states=random_states)
        # print("Len of final rhos: ", len(final_rhos))
        tds = [trace_distance(final_rho, rhos[i]) for i,final_rho in enumerate(final_rhos)]
        tvds = [tvd(prob, true_probs[i]) for i,prob in enumerate(final_probs)]
        td = np.mean(tds)
        mean_tvd = np.mean(tvds)
        final_tds.append(td)
        final_tvds.append(mean_tvd)
        frob_cost = frobenius_cost(target, mean_un)
        final_frobs.append(frob_cost)
        print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}")
        # print(f"Mean Trace Distance: {td}, Mean TVD: {mean_tvd}, Frobenius Distance: {frob_cost}")

    headers = ["Ensemble Size", "Trace Distance", "TVD", "Frobenius Distance"]
    out_data = {}
    out_data["Ensemble Size"] = ensemble_sizes
    out_data["Trace Distance"] = final_tds
    out_data["TVD"] = final_tvds
    out_data["Frobenius Distance"] = final_frobs
    json.dump(out_data, open(f"qp_conv_data_2/{circ_name}_{block_num}_{tol}_{num_unique_circs}.json", "w"))
