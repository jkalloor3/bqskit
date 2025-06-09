from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np
from pathlib import Path
import json

from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
from bqskit.ir.gates import TGate, TdgGate

from itertools import chain, product
# from qiskit import QuantumCircuit, transpilefrom bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from bqskit.qis import UnitaryMatrix

from bqskit.ir.point import CircuitPoint
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ext import bqskit_to_qiskit

from qiskit.quantum_info import random_statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, depolarizing_error)

from util import (get_block_names, check_param_shape, 
                  load_block, load_compiled_circs_params_separate, 
                  trace_distance, normalized_gp_frob_cost)

import os

def create_noise_model(logical_err: float, t_err: float) -> NoiseModel:
    '''
    Noise Model with overrotated CNOT and stochastic depolarizing noise.

    '''
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    log_error = depolarizing_error(logical_err, 1)
    log_err_cx = depolarizing_error(logical_err, 2)
    t_error = depolarizing_error(t_err, 1)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(log_error, ['h', 's', 'sdg', 'x', 'y', 'z'])
    noise_model.add_all_qubit_quantum_error(log_err_cx, ['cx'])
    noise_model.add_all_qubit_quantum_error(t_err, ['t', 'tdg'])

    return noise_model

ensemble_sizes = [1, 5, 10, 20, 40, 80, 160, 1280, 2560]
TOTAL_SHOTS = 100 * max(ensemble_sizes)
NUM_RANDOM_STATES = 2

# circ_name = "qaoa10"

lang = OPENQASM2Language()

def get_random_circs(num_circs: int,
                    block_circs: list[Circuit], 
                    shm_name: str,
                    param_shape: np.ndarray = None) -> list[str]:
    '''
    Generate a random circuit from the block_circs.
    '''
    if len(block_circs) == 1:
        return [block_circs[0].to("qasm")] * num_circs
    
    shm = SharedMemory(name=shm_name, create=False)
    # Now pick a random param from the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)

    # Else, we are using an ensemble
    rand_circ_inds = np.random.randint(0, len(block_circs), size=num_circs)
    rand_param_inds = np.random.randint(0, shm_array.shape[1], size=num_circs)
    circ_strs = []
    for c_ind, p_ind in zip(rand_circ_inds, rand_param_inds):
        # Now we need to get the random circuits
        rand_circ: Circuit = block_circs[c_ind]
        circ_param = shm_array[c_ind][p_ind]

        # Now we need to set the params of the circuit
        rand_circ.set_params(circ_param)
        # fix_phase(rand_circ, target)
        circ_strs.append(rand_circ.to("qasm"))
    return circ_strs

def sim_full_circuits(circ_name: str,
                      block_name: str,
                      noisy_backend: AerSimulator,
                      random_states: list[np.ndarray],
                      block_circs: dict[str, list[Circuit]], 
                      param_shapes: dict[str, tuple[int]],
                      max_tol: float, 
                      target: UnitaryMatrix,
                      ens_size: int) -> dict[str, int]:
    all_circs = []

    shm_name = circ_name + "_" + str(max_tol) + "_" + block_name
    param_shape = param_shapes.get(block_name, None)
    circ_strs = get_random_circs(ens_size,
                                    block_circs[block_name], 
                                    shm_name=shm_name, 
                                    param_shape=param_shape)
    

    dists = []
    for circ_str in circ_strs:
        circ = lang.decode(circ_str)
        # dists.append(normalized_gp_frob_cost(circ.get_unitary(), target))
        qc = bqskit_to_qiskit(circ)
        qc.save_density_matrix()
        all_circs.append(qc)
    
    # print("Avg Dist: ", np.mean(dists))

    # Now we need to sim the circuits
    shots = TOTAL_SHOTS / ens_size
    avg_dms = run_noisy_sim(all_circs, 
                            random_states=random_states, 
                            shots=shots, 
                            backend=noisy_backend)
    # agg_results = aggregate_results(all_dicts)
    return avg_dms


def run_noisy_sim(circs: list[QuantumCircuit], 
                 random_states: list[np.ndarray],
                 shots: int, backend: AerSimulator) -> np.ndarray:
    '''
    Run a noisy simulation of the circuits. Returns an averaged density matrix
    over the circuits for each random state.
    '''
    # Now we need to run the circuits over the random states
    all_dms = []
    for qc in circs:
        random_qcs = []
        for state in random_states:
            new_qc: QuantumCircuit = qc.copy()
            new_qc.initialize(state, range(qc.num_qubits))
            # print(new_qc.count_ops())
            random_qcs.append(new_qc)
        # Now we need to run the circuits
        noisy_results = backend.run(random_qcs, shots=shots).result()
        # Now we need to get the density matrix
        noisy_dms = [noisy_results.data(i)['density_matrix'] for i in range(len(random_qcs))]
        all_dms.append(noisy_dms)
    # Now we have len(circs) x len(random_states) density matrices
    # Now we need to average the density matrices over the circuits
    avg_dms = np.mean(all_dms, axis=0)
    return avg_dms

def run_block(circ_name: str,
              block_name: str,
              tol: float,
              two_q_err: float) -> Circuit:
    # Create backends
    perfect_backend = AerSimulator(method="density_matrix")
    noise_model = create_noise_model(1e-7, two_q_err=two_q_err)
    noisy_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

    # Get the circuit
    # initial_circ: Circuit = load_circuit(circ_name, opt=False)
    initial_circ_file = load_block(circ_name, block_name)
    initial_circ = Circuit.from_file(initial_circ_file)
    num_q = initial_circ.num_qudits
    initial_cnot_count = initial_circ.count(CNOTGate())
    print("Original CX Count: ", initial_cnot_count)
    initial_qcirc = bqskit_to_qiskit(initial_circ)
    initial_qcirc.save_density_matrix()

    # Run perfect simulation
    random_states = [random_statevector(2 ** num_q) for _ in range(NUM_RANDOM_STATES)]
    # random_states_2 = [random_statevector(2 ** num_q) for _ in range(NUM_RANDOM_STATES)]
    perfect_dms = run_noisy_sim([initial_qcirc], random_states=random_states, shots=TOTAL_SHOTS, backend=perfect_backend)

    # Run initial noisy simulation
    noisy_dms = run_noisy_sim([initial_qcirc, initial_qcirc, initial_qcirc], random_states=random_states, shots=TOTAL_SHOTS, backend=noisy_backend)
    initial_dists = [trace_distance(perfect_dm, noisy_dm) for perfect_dm, noisy_dm in zip(perfect_dms, noisy_dms)]

    print("Initial Trace Distance: ", initial_dists)

    # Figure out how many ensembles we need, and calculate shared memory space
    try:
        param_shapes = {block_name: check_param_shape(circ_name,
                                                    block_name,
                                                    tol,
                                                    extra="_tket")}
    except:
        print("No Checkpoint found for block: ", circ_name, block_name)
        return
    shm_names = {block_name: circ_name + "_" + str(tol) + "_" + block_name} 
    
    # Now for the shared memory
    # We will use a different shared memory space for each block
    float_size = np.dtype(np.float64).itemsize
    # Handles to shared memory objects
    shms: dict[str, SharedMemory] = {}
    param_shape = param_shapes[block_name]
    shm_name = shm_names[block_name]
    try:
        shm = SharedMemory(name=shm_name, 
                        create=True, 
                        size=np.prod(param_shape)*float_size)
    except:
        shm = SharedMemory(name=shm_name, 
                        create=False)
        shm.close()
        shm.unlink()
        shm = SharedMemory(name=shm_name, create=True, 
                            size=np.prod(param_shape)*float_size)

    print("Shared Memory: ", shm_name, shm.size)
    # Now we need to create a numpy array in the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)
    # Now we need to fill the array with the parameters
    shm_array.fill(0)
    shms[block_name] = shm

    print("Created Shared Memories: ", shms.keys())

    # Get all the circuits
    # inds, probs = load_compiled_block_circuits_qp_inds(*block_data)
    circuits, params = load_compiled_circs_params_separate(circ_name,
                                                           block_name,
                                                           tol=tol,
                                                           extra="_tket")
    print("Num Circuits: ", len(circuits), flush=True)
    avg_cnot_count = np.mean([circ.count(CNOTGate()) for circ in circuits])
    print("Avg CX Count: ", avg_cnot_count)
    if avg_cnot_count > 0.92 * initial_cnot_count:
        print("Not enough CX reduction, skipping block: ", circ_name, block_name)
        return
    # Now we need to load params into shared memory
    shm = shms[block_name]
    param_shape = params.shape


    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)
    # Now we need to fill the array with the parameters
    shm_array[:] = params
    block_circs[block_name] = circuits

    print("LOADED CIRCUITS", flush=True)
    # ensemble_sizes = [2, 10, 20]

    num_trials = 6
    ensemble_costs = []

    # Get avg cost
    target = initial_circ.get_unitary()
    for ens_size in ensemble_sizes:
        mean_costs = []
        print("Ensemble Size: ", ens_size, flush=True)
        for i in range(num_trials):
            # print("Trial: ", i, flush=True)
            avg_dms = sim_full_circuits(circ_name,
                                        block_name,
                                       noisy_backend,
                                       random_states,
                                       block_circs, 
                                       param_shapes,
                                       max_tol=tol,
                                       target=target,
                                       ens_size=ens_size)
            # mean_tvd = trace_distance(perfect_dm, avg_dm)
            tds = [trace_distance(perfect_dm, avg_dm) for perfect_dm, avg_dm in zip(perfect_dms, avg_dms)]
            mean_tvd = np.mean(tds)
            mean_costs.append(mean_tvd)
        ensemble_costs.append(mean_costs)
        print("Avg Trace Distance: ", np.mean(mean_costs))
    

    for block_name, shm in shms.items():
        print("Closing Shared Memory: ", shm_name)
        shm.close()
        shm.unlink()

    cost_path = f"ensemble_trace_dists_block/ensemble_trace_dist_qc_{circ_name}/{block_name}/{tol}.pickle"
    Path(cost_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(ensemble_costs, open(cost_path, 'wb'))


# Circ 
if __name__ == '__main__':
    block_circs = {}
    block_names = []

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    circ_name = str(argv[1])
    two_q_err = float(argv[3]) if len(argv) > 3 else 5e-3
    tol = float(argv[2]) if len(argv) > 2 else 3.0

    block_names = get_block_names(circ_name, extra="_tket")
    for block_name in block_names:
        run_block(circ_name, block_name, tol, two_q_err=two_q_err)
# from qiskit_aer.noise import NoiseModel, depolarizing_error
import multiprocessing as mp
import pickle

# from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

# from util import load_cliff_circ, load_block, load_compiled_block_circuits, load_compiled_block_circuits_qp_inds
from util import convert_to_clifft, tvd_dict, convert_to_clifft_tbudget

qcircs = []
pickle_file = "qpe_14_3_3.0_250_debug.pkl"
ensemble_circs = pickle.load(open(pickle_file, "rb"))
t_budgets = [3000, 5000, 10000]
all_t_circs = {}
for t_budget in t_budgets:
    t_circs: list[Circuit] = pickle.load(open(t_budget_file.format(t_budget=t_budget), "rb"))
    # store_ensemble(t_circs, t_budget_qasms_file.format(t_budget=t_budget))
    # t_circs = load_ensemble(t_budget_qasms_file.format(t_budget=t_budget))
    print(f"Loaded T Budget={t_budget} Circuits with len {len(t_circs)}", flush=True)
    # print([circ.gate_counts for circ in t_circs])
    all_t_circs[t_budget] = t_circs

num_random_states = 16

# def run_ensembles(ens_size: int,shots: int, backend: AerSimulator, precisions: list[int]) -> list[dict[str, int]]:
#     '''
#     Run a set of noisy circuits on a given backend over a set of precisions.
#     Returns a final count for each precision.
#     '''
#     global ensemble_circs

#     ensemble_inds: list[int] = np.random.choice(len(ensemble_circs), ens_size)
#     # Conver to cliff_t for each precision
#     # ensemble = [[convert_to_clifft(ensemble_circs[i], prec) for i in ensemble_inds] for prec in precisions]
#     ensemble = []
#     for prec in precisions:
#         with mp.Pool(4) as pool:
#             prec_ens = pool.starmap(convert_to_clifft, [(ensemble_circs[i], prec) for i in ensemble_inds])
#         ensemble.append(prec_ens)

#     print("Converted Ensemble Circuits", len(ensemble), flush=True)
#     # Get the qiskit circuits
#     q_ensemble = [[get_qcirc(circ) for circ in ens] for ens in ensemble]

#     all_results = []
#     for i, qcircs in enumerate(q_ensemble):
#         prec = precisions[i]
#         print(f"Running Ensemble with Precision {prec}", flush=True)
#         results = backend.run(qcircs, shots=shots).result()
#         prec_results = []
#         for i in range(len(qcircs)):
#             prec_results.append(results.get_counts(i))
#         # Aggregate the results
#         agg_results = aggregate_results(prec_results)
#         all_results.append(agg_results)
#     return all_results

# def get_qcirc(circ: Circuit):
#     q_circ = bqskit_to_qiskit(circ)
#     q_circ.measure_all()
#     return q_circ

def aggregate_results(results: list[dict[str, int]]) -> dict[str, int]:
    total_dict = {}
    for y in results:
        x = total_dict
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return total_dict

# def create_noise_model(t_gate_err: float, logical_gate_err: float, one_q_gates: list[str]):

#     # Create an empty noise model
#     noise_model = NoiseModel()

#     # Add depolarizing error to all single qubit u1, u2, u3 gates
#     one_q_error = depolarizing_error(logical_gate_err, 1)
#     two_q_error = one_q_error.tensor(one_q_error)
#     t_error = depolarizing_error(t_gate_err, 1)
#     noise_model.add_all_qubit_quantum_error(t_error, ['t', 'tdg'])
#     # noise_model.add_all_qubit_quantum_error(one_q_error, one_q_gates)
#     # noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

#     return noise_model

# def get_sim_backend(circ: QuantumCircuit, t_err: float, logical_err: float):
#     one_q_gates = list(circ.count_ops().keys())
#     gates_to_remove = ['cx', 'measure', 'barrier']
#     for gate in gates_to_remove:
#         if gate in one_q_gates:
#             one_q_gates.remove(gate)
#     print("One Q Gates: ", one_q_gates)
#     return AerSimulator(noise_model=create_noise_model(t_err, logical_err, one_q_gates))

def get_t_count(circ: Circuit):
    return circ.count(TGate()) + circ.count(TdgGate())

def get_all_clifft_circs(t_budget: int, start_ind: int) -> list[Circuit]:
    print("Len Ensemble Circs: ", len(ensemble_circs), flush=True)
    return t_budget, [convert_to_clifft_tbudget(ensemble_circs[i], t_budget) for i in range(start_ind, start_ind + 20)]

# def run_circuits(shots: int, backend: AerSimulator) -> list[dict[str, int]]:
#     global qcircs
#     all_counts = []
#     circs_to_run = list(chain.from_iterable(qcircs))
#     print(f"Running {len(circs_to_run)} Circuits", flush=True)
#     results = backend.run(circs_to_run, shots=shots).result()
#     all_counts = []
#     for i in range(len(circs_to_run)):
#         all_counts.append(results.get_counts(i))
#     return all_counts

if __name__ == '__main__':
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    # circ_name = argv[1]
    # block_num = int(argv[2])
    circ_name = "qpe_14"
    # circ_name = "LiH"
    block_num = 3
    # tol = int(argv[3])
    # num_unique_circs = int(argv[4])
    t_budgets = [3000, 5000, 10000, 15000]
    t_errs = [10 ** (-i) for i in range(1, 6)]
    logical_err = 10 ** (-8)
    cliff = False

    ens_sizes = [1, 16, 64, 128, 256]
    shot_ratio = max(ens_sizes)
    shots = 1024

    # # Get backend and target
    backend = AerSimulator()

    # Get orig circ and unitary
    orig_circ = load_block(circ_name, block_num)
    orig_circ = Circuit.from_file(orig_circ)
    target = orig_circ.get_unitary()

    print("Converting Original Circuits", flush=True)

    # Run Simulator of circuit with continuos angles
    orig_counts = backend.run(get_qcirc(orig_circ), shots=shots * shot_ratio).result().get_counts(0)
    orig_circs = [convert_to_clifft_tbudget(orig_circ, budget, True) for budget in t_budgets]
    orig_t_counts = [get_t_count(circ) for circ in orig_circs]
    print("Original T Counts: ", orig_t_counts)

    # qcircs: list[QuantumCircuit] = [[get_qcirc(circ) for circ in orig_circs]]

    # orig_statevectors = run_circuits(shots * shot_ratio, backend)
    orig_unitaries = [c.get_unitary() for c in orig_circs]
    # start_tvds = [tvd_dict(orig_counts, statevector) for statevector in orig_statevectors]
    start_tvds = [normalized_gp_frob_cost(unitary, target) for unitary in orig_unitaries]
    print("Got Original Statevectors", flush=True)
    print("Got Original TVDs", flush=True)
    print(start_tvds)

    # Get Ensemble of Circuits
    # ensemble_circs: list[Circuit] = load_compiled_block_circuits(circ_name, block_num, 3.0, 250, target=target, add_unitaries=False)
    # qp_inds, qp_probs = load_compiled_block_circuits_qp_inds(circ_name, block_num, 3.0, 250)
    # # Randomly pick 500 circuits
    # ensemble_circs = [ensemble_circs[i] for i in qp_inds[:500]]
    # # # Save these circuits for debugging
    # pickle_file = "debug_circs_qpe_14.pkl"
    # # # pickle.dump(ensemble_circs, open(pickle_file, "wb"))
    # ensemble_circs = pickle.load(open(pickle_file, "rb"))
    # print("Loaded Ensemble Circuits", flush=True)

    with mp.Pool(100) as pool:
        start_inds = list(range(0, 500, 20))
        params = product(t_budgets, start_inds)
        circ_budgets: list[tuple[int, list[Circuit]]] = pool.starmap(get_all_clifft_circs, params)

    t_budget_circs = {}

    for t_budget, circs in circ_budgets:
        if t_budget not in t_budget_circs:
            t_budget_circs[t_budget] = circs
        else:
            t_budget_circs[t_budget].extend(circs)
    
    for t_budget, circs in t_budget_circs.items():
        print(f"Budget: {t_budget}, Circs: {len(circs)}")
        cliff_pickle_file = f"debug_clifft_circs_qpe_14_t_bud_{t_budget}.pkl"
        pickle.dump(circs, open(cliff_pickle_file, "wb"))
    
    print("Saved Ensemble Circuits", flush=True)
    exit(0)

    # # exit(0)

    # print("Loaded Ensemble Circuits", flush=True)

    # all_tvds = []
    # for ens_size in ens_sizes:
    #     shot_per_circ = (shot_ratio // ens_size) * shots
    #     counts = run_ensembles(ens_size, shot_per_circ, backend, precisions)
    #     tvds = [tvd_dict(orig_counts, count) for count in counts]
    #     all_tvds.append(tvds)


    # pickle.dump(all_tvds, open("frob_dists_analysis_LiH.pkl", "wb"))
    # all_tvds = pickle.load(open("ens_err_analysis.pkl", "rb"))

    headers = [f"T Budget: {t_budg}" for t_budg in t_budgets]

    # Now it is in the form [pre 

    # Create a plot with lines for each precision
    fig, axes = plt.subplots(1, len(t_budgets), figsize=(5 * len(t_budgets), 5))
    if len(t_budgets) == 1:
        axes = [axes]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i = 0
    for t_budget, tvd_data in all_tvds.items():
        start_tvd = tvd_data[0]
        ax: plt.Axes = axes[i]
        ax.plot(ens_sizes, tvd_data[1:], label=headers[i], color=colors[i])
        ax.hlines(start_tvds[i], min(ens_sizes), max(ens_sizes), label="Original", linestyles='dashed', colors=[colors[i]])
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylabel('Normalized Frobenius Distance')
        ax.set_title(f'T Budget: {t_budget}')
        i += 1
    # ax.hlines(start_tvds, min(ens_sizes), max(ens_sizes), label="Original", linestyles='dashed')
    
    fig.savefig(f"frob_dist_analysis_qpe_14.png")

