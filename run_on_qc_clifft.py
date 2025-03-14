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
# from qiskit import QuantumCircuit, transpile
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

