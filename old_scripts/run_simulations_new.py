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

from util import (load_circuit, get_circ_data, check_param_shape, 
                  load_block, load_compiled_circs_params_separate, 
                  load_compiled_block_circuits_qp_inds)
from util.distance import normalized_gp_frob_cost, trace_distance, tvd, get_density_matrix, get_average_density_matrix
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


def generate_full_circuits(circ_name: str,
                           block_circs: dict[str, list[Circuit]], 
                           block_names: list,
                           block_inds: dict[str, np.ndarray],
                           block_probs: dict[str, np.ndarray],
                           param_shapes: dict[str, tuple[int]],
                           block_targets: dict[str, UnitaryMatrix],
                           pcirc: Circuit,
                           max_tol: float, 
                           ens_size: int) -> list[Circuit]:
    all_circs = []

    circ_blocks = {}

    start = time.time()
    for block_name in block_names:
        shm_name = circ_name + "_" + str(max_tol) + "_" + block_name
        param_shape = param_shapes.get(block_name, None)
        target = block_targets.get(block_name, None)
        inds = block_inds.get(block_name, None)
        probs = block_probs.get(block_name, None)
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                  block_circs[block_name],
                                                  inds=inds,
                                                  probs=probs, 
                                                  shm_name=shm_name, 
                                                  param_shape=param_shape,
                                                  target=target)
        
    print("Finish Getting Circs: ", time.time() - start, flush=True)
    start = time.time()
    
    for i in range(ens_size):
        circ = pcirc.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            new_block_circ = circ_blocks[block_names[ind]][i]
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            assert isinstance(new_block_circ, Circuit)
            assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
            circ.replace_with_circuit(pt, new_block_circ, as_circuit_gate=True)
        all_circs.append(circ)

    print("Finish Generating Circs: ", time.time() - start, flush=True)
    return all_circs


def get_data(ens: list[Circuit],
              ens_size: int,
              target: UnitaryMatrix,
              target_dm: np.ndarray,
              target_sv_prob: np.ndarray):
    '''
    Get the data for the ensemble of circuits.
    '''
    ens = all_ens[i * ens_size:(i + 1) * ens_size]
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


# Circ 
if __name__ == '__main__':
    block_circs = {}
    block_names = []
    block_probs = {}
    block_inds = {}

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    circ_name = argv[1]
    max_tol = float(argv[2]) if len(argv) > 2 else 0.01

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    # if initial_circ.num_qudits <= 10:
    target = initial_circ.get_unitary()
    initial_qcirc = bqskit_to_qiskit(initial_circ)
    target_sv = Statevector.from_instruction(initial_qcirc).data
    target_sv_prob = np.abs(target_sv)**2
    target_dm = get_density_matrix(target_sv)

    block_targets = {}
    block_dirs, count = get_circ_data(circ_name, max_tol, use_base=False)
    block_names = sorted([x[0] for x in block_dirs])
    print("Avg Count: ", count, "TKET Count: ", 110)
    print("Block dirs: ", block_dirs)
    print("Block names: ", block_names)

    # Figure out how many ensembles we need, and calculate shared memory space
    param_shapes = {}
    shm_names = {}
    for block_name, block_data in block_dirs:
        if block_data[0] == 'orig' or block_data[0] == "_tket":
            continue
        else:
            param_shape = check_param_shape(*block_data)
            shm_names[block_name] = shm_name = circ_name + "_" + str(max_tol) + "_" + block_name
            param_shapes[block_name] = param_shape
    
    print("Param Shapes: ", param_shapes)
    # Now for the shared memory
    # We will use a different shared memory space for each block
    float_size = np.dtype(np.float64).itemsize
    # Handles to shared memory objects
    shms: dict[str, SharedMemory] = {}
    for block_name in block_names:
        if block_name not in shm_names:
            continue
        param_shape = param_shapes[block_name]
        shm_name = shm_names[block_name]
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
        # print("Shared Memory: ", shm_name, shm.size)
        # Now we need to create a numpy array in the shared memory
        shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)
        # Now we need to fill the array with the parameters
        shm_array.fill(0)
        shms[block_name] = shm

    # print("Created Shared Memories: ", shms.keys())

    # Get all the circuits
    for block_name, block_data in block_dirs:
        bc_file = load_block(block_data[0], block_data[1], extra=block_data[-1])
        block_target = Circuit.from_file(bc_file) 
        block_un = block_target.get_unitary()
        block_targets[block_name] = block_un
        if block_data[0] == 'orig':
            block_circs[block_name] = [block_target]
        elif block_data[0] == "_tket":
            circ_file = load_block(circ_name, block_name, extra="_tket")
            block_circs[block_name] = [Circuit.from_file(circ_file)]
        else:
            # inds, probs = load_compiled_block_circuits_qp_inds(*block_data)
            circuits, params = load_compiled_circs_params_separate(*block_data)
            if USE_QP:
                inds, probs = load_compiled_block_circuits_qp_inds(*block_data)
                if probs is not None:
                    block_inds[block_name] = inds
                    block_probs[block_name] = probs
                    print("Sum of probs: ", np.sum(probs), block_name, flush=True)
                else:
                    # Use uniform distribution
                    block_inds[block_name] = np.arange(len(circuits) * params.shape[1])
                    block_probs[block_name] = np.ones(len(circuits) * params.shape[1]) / len(circuits) / params.shape[1]
            # Now we need to load params into shared memory
            shm = shms[block_name]
            param_shape = params.shape
            shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)
            # Now we need to fill the array with the parameters
            shm_array[:] = params
            block_circs[block_name] = circuits

    # print("Num Circuits: ", [len(ens) for ens in block_ensembles.values()], flush=True)

    print("LOADED CIRCUITS", flush=True)

    pcirc_file = partitioned_circ_save_file.format(circ_name=circ_name)
    partitioned_circ: Circuit = pickle.load(open(pcirc_file, 'rb'))

    ensemble_sizes = [1, 5, 10, 40, 160, 640, 2560]

    ensemble_circuits = []
    num_trials = 15
    ensemble_frob_costs = []
    ensemble_tvd_costs = []
    ensemble_trace_dist_costs = []

    # Get avg cost
    target = initial_circ.get_unitary()
    for ens_size in ensemble_sizes:
        ens_size_path = f"ensemble_sim_{circ_name}_{max_tol}/{ens_size}.pickle"
        if os.path.exists(ens_size_path):
            print("Ensemble Size: ", ens_size, 
                  " already exists, skipping", flush=True)
            continue
        mean_frob_costs = []
        mean_tvd_costs = []
        mean_trace_dist_costs = []
        print("Ensemble Size: ", ens_size, flush=True)
        start_time = time.time()
        # for i in range(num_trials):
            # print("Trial: ", i, flush=True)
        all_ens = generate_full_circuits(circ_name,
                                        block_circs, 
                                        block_names,
                                        block_inds,
                                        block_probs,
                                        param_shapes,
                                        block_targets,
                                        partitioned_circ,
                                        max_tol=max_tol, 
                                        ens_size=ens_size * num_trials)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_trials) as executor:
            futures = []
            for i in range(num_trials):
                ens = all_ens[i * ens_size:(i + 1) * ens_size]
                future = executor.submit(get_data, ens, ens_size, 
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
        print("Avg Cost: ", np.mean(mean_frob_costs))
        size_cost = {
            "frob": mean_frob_costs,
            "tvd": mean_tvd_costs,
            "trace_dist": mean_trace_dist_costs
        }
        Path(ens_size_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(size_cost, open(ens_size_path, 'wb'))
    

    for block_name, shm in shms.items():
        print("Closing Shared Memory: ", shm_name)
        shm.close()
        shm.unlink()

    all_costs = {
        "frob": ensemble_frob_costs,
        "tvd": ensemble_tvd_costs,
        "trace_dist": ensemble_trace_dist_costs
    }

    # Delete all ens_size_path files
    for ens_size in ensemble_sizes:
        ens_size_path = f"ensemble_sim_{circ_name}_{max_tol}/{ens_size}.pickle"
        os.remove(ens_size_path)
    # Save all costs
    final_path = f"ensemble_sim_{circ_name}_{max_tol}/all_data.pickle"
    pickle.dump(all_costs, open(final_path, 'wb'))