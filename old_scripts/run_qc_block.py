from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import numpy as np
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import concurrent.futures
from multiprocessing.shared_memory import SharedMemory

from bqskit.ir.gates import GlobalPhaseGate
from bqskit.ir.point import CircuitPoint

from bqskit.ir.lang.qasm2 import OPENQASM2Language

from util import (get_block_names, check_param_shape, 
                  load_block, load_compiled_circs_params_separate, 
                  load_compiled_block_circuits_qp_inds)
from util.distance import (normalized_gp_frob_cost, trace_distance, tvd, 
                           get_density_matrix, get_average_density_matrix)
from util.fix_global_phase import fix_phase

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from bqskit.qis import UnitaryMatrix
from bqskit.ext import bqskit_to_qiskit

import time

shots = 100
partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

USE_QP = False

lang = OPENQASM2Language()

def get_qcirc(circ: Circuit) -> QuantumCircuit:
    circ.unfold_all()
    # Remove all GlobalPhaseGates
    circ.remove_all(GlobalPhaseGate())
    return bqskit_to_qiskit(circ)


def get_circs(circs: list[Circuit], params: list[np.ndarray], 
              target: UnitaryMatrix) -> list[Circuit]:
    # Set the params of the circuit
    new_circs = []
    for i, circ in enumerate(circs):
        param = params[i]
        new_circ = circ.copy()
        new_circ.set_params(param)
        fix_phase(new_circ, target)
        new_circs.append(new_circ)
    return new_circs

def get_random_circs(num_circs: int,
                    block_circs: list[Circuit],
                    inds: np.ndarray,
                    probs: np.ndarray, 
                    shm_name: str,
                    param_shape: np.ndarray = None,
                    target: UnitaryMatrix = None) -> list[Circuit]:
    '''
    Generate a random circuit from the block_circs.
    '''
    if len(block_circs) == 1:
        return [block_circs[0].copy()] * num_circs
    
    shm = SharedMemory(name=shm_name, create=False)
    # Now pick a random param from the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)

    # Else, we are using an ensemble
    if USE_QP:
        rand_inds = np.random.choice(inds, size=num_circs, p=probs)
        num_params_per_circ = shm_array.shape[1]
        rand_circ_inds = [i // num_params_per_circ for i in rand_inds]
        rand_param_inds = [i % num_params_per_circ for i in rand_inds]
    else:
        rand_circ_inds = np.random.randint(0, len(block_circs), size=num_circs)
        rand_param_inds = np.random.randint(0, shm_array.shape[1], size=num_circs)
    all_block_circs = []
    # for c_ind, p_ind in zip(rand_circ_inds, rand_param_inds):
    rand_circs = [block_circs[c_ind] for c_ind in rand_circ_inds]
    circ_params = [shm_array[c_ind][p_ind] for c_ind, p_ind in zip(rand_circ_inds, rand_param_inds)]

    # Group into groups of 10
    group_size = 40
    rand_circ_groups = [rand_circs[i:i + group_size] for i in range(0, len(rand_circs), group_size)]
    circ_param_groups = [circ_params[i:i + group_size] for i in range(0, len(circ_params), group_size)]
    num_workers = min(256, len(rand_circ_groups))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for rand_circs, circ_params in zip(rand_circ_groups, circ_param_groups):
            future = executor.submit(get_circs, rand_circs, circ_params, target)
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            all_block_circs.extend(future.result())
    return all_block_circs

def get_data(ens: list[Circuit],
              target: UnitaryMatrix,
              target_dm: np.ndarray,
              target_sv_prob: np.ndarray):
    '''
    Get the data for the ensemble of circuits.
    '''
    mean_un = np.mean(np.array([c.get_unitary() for c in ens]), axis=0)
    # print(ens[0].gate_counts)
    qiskit_circs = [get_qcirc(c) for c in ens]
    svs = [Statevector.from_instruction(circ).data for circ in qiskit_circs]
    sv_probs = [np.abs(sv)**2 for sv in svs]
    mean_sv_prob = np.mean(np.array(sv_probs), axis=0)
    mean_dm = get_average_density_matrix(svs)
    mean_frob_cost = normalized_gp_frob_cost(mean_un, target)
    mean_trace_dist = trace_distance(target_dm, mean_dm)
    mean_tvd_cost = tvd(target_sv_prob, mean_sv_prob)
    return mean_frob_cost, mean_trace_dist, mean_tvd_cost


def run_block(circ_name: str, block_num: int, tol: float) -> None:
    block_circs = {}
    block_names = []
    block_probs = {}
    block_inds = {}
    final_path = f"block_ensemble_sim_{circ_name}/{block_num}/{tol}/all_data.pickle"
    if os.path.exists(final_path):
        print("All data already exists, skipping")
        exit(0)

    # initial_circ = load_circuit(circ_name, opt=False)
    initial_circ_file = load_block(circ_name, block_num, extra="_tket")
    initial_circ = Circuit.from_file(initial_circ_file)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    # if initial_circ.num_qudits <= 10:
    target = initial_circ.get_unitary()
    initial_qcirc = bqskit_to_qiskit(initial_circ)
    target_sv = Statevector.from_instruction(initial_qcirc).data
    target_sv_prob = np.abs(target_sv)**2
    target_dm = get_density_matrix(target_sv)
    try:
        param_shape = check_param_shape(circ_name, block_num, tol, extra="_tket")
    except:
        print("Checkpoint Data does not exist, exiting", circ_name, block_num, tol)
        return
    shm_name = circ_name + "_" + str(tol) + "_" + block_num

    # Now for the shared memory
    # We will use a different shared memory space for each block
    float_size = np.dtype(np.float64).itemsize

    try:
        shm = SharedMemory(name=shm_name, 
                        create=True, 
                        size=np.prod(param_shape)*float_size)
    except:
        print("Shared Memory already exists: ", shm_name)
        shm = SharedMemory(name=shm_name, create=False)
        shm.close()
        shm.unlink()
        shm = SharedMemory(name=shm_name, 
                            create=True, 
                            size=np.prod(param_shape)*float_size)

    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)

    circuits, params = load_compiled_circs_params_separate(circ_name,
                                                           block_num,
                                                           tol,
                                                           extra="_tket")
    shm_array[:] = params

    if USE_QP:
        block_inds, block_probs = load_compiled_block_circuits_qp_inds(circ_name,
                                                            block_num,
                                                            tol,
                                                            extra="_tket")
        if block_probs is  None:
            # Use uniform distribution
            block_inds= np.arange(len(circuits) * params.shape[1])
            block_probs = np.ones(len(circuits) * params.shape[1]) / len(circuits) / params.shape[1]

    ensemble_sizes = [1, 5, 10, 40, 160, 640, 2560]

    ensemble_circuits = []
    num_trials = 15
    ensemble_frob_costs = []
    ensemble_tvd_costs = []
    ensemble_trace_dist_costs = []

    # Get avg cost
    target = initial_circ.get_unitary()
    for ens_size in ensemble_sizes:
        ens_size_path = f"block_ensemble_sim_{circ_name}/{block_num}/{tol}/{ens_size}.pickle"
        if os.path.exists(ens_size_path):
            print("Ensemble Size: ", ens_size, 
                  " already exists, skipping", flush=True)
            size_cost = pickle.load(open(ens_size_path, 'rb'))
            ensemble_frob_costs.append(size_cost["frob"])
            ensemble_tvd_costs.append(size_cost["tvd"])
            ensemble_trace_dist_costs.append(size_cost["trace_dist"])
            continue
        mean_frob_costs = []
        mean_tvd_costs = []
        mean_trace_dist_costs = []
        print("Ensemble Size: ", ens_size, flush=True)
        start_time = time.time()
        all_ens = get_random_circs(ens_size * num_trials,
                                   circuits,
                                    block_inds,
                                    block_probs,
                                    shm_name,
                                    param_shape,
                                    target)
        
        print("Got random circuits: ", time.time() - start_time, flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_trials) as executor:
            futures = []
            for i in range(num_trials):
                ens = all_ens[i * ens_size:(i + 1) * ens_size]
                future = executor.submit(get_data, ens, 
                                         UnitaryMatrix(target), 
                                         target_dm.copy(), 
                                         target_sv_prob.copy())
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                mean_frob_cost, mean_trace_dist_cost, mean_tvd_cost = future.result()
                mean_frob_costs.append(mean_frob_cost)
                mean_trace_dist_costs.append(mean_trace_dist_cost)
                mean_tvd_costs.append(mean_tvd_cost)

        print("Time: ", time.time() - start_time, flush=True)
        ensemble_frob_costs.append(mean_frob_costs)
        ensemble_tvd_costs.append(mean_tvd_costs)
        ensemble_trace_dist_costs.append(mean_trace_dist_costs)
        print("Mean Costs: ", mean_frob_costs)
        size_cost = {
            "frob": mean_frob_costs,
            "tvd": mean_tvd_costs,
            "trace_dist": mean_trace_dist_costs
        }
        Path(ens_size_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(size_cost, open(ens_size_path, 'wb'))
    

    shm.close()
    shm.unlink()

    all_costs = {
        "frob": ensemble_frob_costs,
        "tvd": ensemble_tvd_costs,
        "trace_dist": ensemble_trace_dist_costs
    }

    # Save all costs
    final_path = f"block_ensemble_sim_{circ_name}/{block_num}/{tol}/all_data.pickle"
    Path(final_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(all_costs, open(final_path, 'wb'))

    # Delete all ens_size_path files
    for ens_size in ensemble_sizes:
        ens_size_path = f"block_ensemble_sim_{circ_name}_{block_num}_{tol}/{ens_size}.pickle"
        os.remove(ens_size_path)


# Circ 
if __name__ == '__main__':
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    circ_name = argv[1]
    tol = float(argv[2])

    block_names = get_block_names(circ_name, extra="_tket")

    for block_num in block_names:
        run_block(circ_name, block_num, tol)

