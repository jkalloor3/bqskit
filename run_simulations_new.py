from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
import numpy as np
from itertools import chain

from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from util import load_circuit, load_compiled_circuits
from util.distance import normalized_frob_cost, tvd, trace_distance, get_density_matrix, get_average_density_matrix

from bqskit.ext import bqskit_to_qiskit

shots = 100

def get_ensemble_mags(ens_size, random_states: list[np.ndarray] = None) -> tuple[list[np.ndarray[np.float64]], 
                                                                                 list[np.ndarray[np.float64]], 
                                                                                 np.ndarray[np.complex128] | None]:
    global all_qcircs
    global bqskit_circs
    global target

    print(len(bqskit_circs), len(all_qcircs))

    ensemble_inds: list[int] = np.random.choice(len(bqskit_circs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    
    if ensemble[0].num_qubits <= 10:
        mean_un = np.mean(np.array([bqskit_circs[i].get_unitary().numpy for i in ensemble_inds]), axis=0)
    else:
        mean_un = None

    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    random_qcircs = get_random_init_state_circuits(ensemble, random_states)
    noisy_rhos = []
    noisy_probs = []
    for circs in random_qcircs:
        noisy_svs = np.array([Statevector.from_instruction(circ).data for circ in circs])
        probs = np.array([np.abs(sv)**2 for sv in noisy_svs], dtype=np.float64)
        avg_probs = np.mean(probs, axis=0)
        noisy_rho = get_average_density_matrix(noisy_svs)
        noisy_rhos.append(noisy_rho)
        noisy_probs.append(avg_probs)
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
    global bqskit_circs
    global target

    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)


    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    cliff = bool(int(argv[5])) if len(argv) > 5 else False

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    if initial_circ.num_qudits <= 10:
        target = initial_circ.get_unitary()

    opt_str = ""

    if cliff:
        bqskit_circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_cliff_t_final")
    else:
        bqskit_circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min_post{opt_str}_calc_bias")

    print("LOADED CIRCUITS", flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 100, 1000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    # sampler = Sampler(mode=sim)

    random_states = get_random_states(initial_circ.num_qudits, num_random_states=5)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circs = get_random_init_state_circuits([qiskit_circ], random_states)
    qiskit_circs = list(chain(*qiskit_circs))
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
    for j, ens_size in enumerate(ensemble_sizes):
        final_rhos, final_probs, mean_un = get_ensemble_mags(ens_size, random_states)

        print("Len of final rhos: ", len(final_rhos))

        tds = [trace_distance(final_rho, rhos[i]) for i,final_rho in enumerate(final_rhos)]
        tvds = [tvd(prob, true_probs[i]) for i,prob in enumerate(final_probs)]
        td = np.mean(tds)
        mean_tvd = np.mean(tvds)
        if mean_un is not None:
            frob_cost = normalized_frob_cost(target, mean_un)
            print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}")
            print(f"Mean Trace Distance: {td}, Mean TVD: {mean_tvd}, Frobenius Distance: {frob_cost}")
        else:
            print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}")
            print(f"Mean Trace Distance: {td}, Mean TVD: {mean_tvd}")


