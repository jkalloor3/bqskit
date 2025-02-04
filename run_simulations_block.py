from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
import numpy as np
import multiprocessing as mp
import json
from itertools import chain

from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from bqskit.qis import UnitaryMatrix
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from util import load_block, load_compiled_block_circuits, load_compiled_block_circuits_qp_inds, get_unitary
from util.distance import frobenius_cost, tvd, trace_distance, get_density_matrix, get_average_density_matrix, normalized_gp_frob_cost

from bqskit.ext import bqskit_to_qiskit

shots = 100

def get_ensemble_mags_qp(ens_size, 
                         random_states: list[np.ndarray] = None) -> tuple[list[np.ndarray[np.float64]], 
                                                                                 list[np.ndarray[np.float64]], 
                                                                                 np.ndarray[np.complex128] | None]:
    global all_qcircs
    global uns
    global qp_inds
    global circ_probs
    ensemble_inds: list[int] = np.random.choice(qp_inds, ens_size, p=circ_probs)
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
    global qp_inds
    global circ_probs

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ_name = argv[1]
    block_num = argv[2]
    tol = float(argv[3])
    num_unique_circs = int(argv[4])
    cliff = False

    circ_path = load_block(circ_name, block_num, good=True)
    initial_circ = Circuit.from_file(circ_path)
    target = UnitaryMatrix(initial_circ.get_unitary()) 
    print("Got initial circ", flush=True)
    # circs = load_compiled_block_circuits_qp(circ_name, block_num, tol, num_unique_circs)
    circs: list[tuple[Circuit, UnitaryMatrix, float]] = load_compiled_block_circuits(circ_name, block_num, tol, num_unique_circs, target=target)
    qp_inds, circ_probs = load_compiled_block_circuits_qp_inds(circ_name, 
                                                               block_num, 
                                                               tol, 
                                                               num_unique_circs)
    print("Num Circs: ", len(circs), flush=True)

    if len(circs) == 0:
        print("No circuits found")
        exit(0)

    # dists = [target.get_frobenius_distance(c.get_unitary()) for c in circs[:20]]
    # dists = [c[1] for c in circs]
    bqskit_circs = [c[0] for c in circs]
    uns = [c[1] for c in circs]
    dists = [c[2] for c in circs]
    print("Avg Dist: ", np.mean(dists))
    dists_qp = [circs[i][2] for i in qp_inds]
    mean_dists_qp = np.sum([j * circ_probs[i] for i,j in enumerate(dists_qp)])
    print("Avg Dist QP: ", mean_dists_qp)
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
    final_tds_qp = []
    final_frobs_qp = []
    final_tvds_qp = []
    num_trials = 10
    for j, ens_size in enumerate(ensemble_sizes):
        tds = []
        frobs = []
        tvds = []
        tds_qp = []
        frobs_qp = []
        tvds_qp = []
        for _ in range(num_trials):
            final_rhos, final_probs, mean_un = get_ensemble_mags(ens_size, random_states=random_states)
            final_rhos_qp, final_probs_qp, mean_un_qp = get_ensemble_mags_qp(ens_size, random_states=random_states)
            # print("Len of final rhos: ", len(final_rhos))
            tds.extend([trace_distance(final_rho, rhos[i]) for i,final_rho in enumerate(final_rhos)])
            tds_qp.extend([trace_distance(final_rho, rhos[i]) for i,final_rho in enumerate(final_rhos_qp)])
            tvds.extend([tvd(prob, true_probs[i]) for i,prob in enumerate(final_probs)])
            tvds_qp.extend([tvd(prob, true_probs[i]) for i,prob in enumerate(final_probs_qp)])
            frobs.append(normalized_gp_frob_cost(mean_un, target))
            frobs_qp.append(normalized_gp_frob_cost(mean_un_qp, target))
        
        final_tds.append(tds)
        final_tds_qp.append(tds_qp)
        final_tvds.append(tvds)
        final_tvds_qp.append(tvds_qp)
        final_frobs.append(frobs)
        final_frobs_qp.append(frobs_qp)
        print(f"Ran Ensemble")
        # print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}")
        # print(f"Mean Trace Distance: {td}, Mean TVD: {mean_tvd}, Frobenius Distance: {frob_cost}")

    headers = ["Ensemble Size", "Trace Distance", "TVD", "Frobenius Distance"]
    out_data = {}
    out_data["Ensemble Size"] = ensemble_sizes
    out_data["Trace Distance"] = final_tds
    out_data["TVD"] = final_tvds
    out_data["Frobenius Distance"] = final_frobs
    out_data["Trace Distance w/ QP"] = final_tds_qp
    out_data["TVD w/ QP"] = final_tvds_qp
    out_data["Frobenius Distance w/ QP"] = final_frobs_qp
    json.dump(out_data, open(f"no_qp_conv_data/{circ_name}_{block_num}_{tol}_{num_unique_circs}.json", "w"))
