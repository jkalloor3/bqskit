from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing.shared_memory import SharedMemory

from bqskit.ir.gates import GlobalPhaseGate
from bqskit.ir.point import CircuitPoint

from bqskit.ir.lang.qasm2 import OPENQASM2Language

from util import (load_circuit, get_circ_data, check_param_shape, 
                  load_block, load_compiled_circs_params_separate)
from util.distance import normalized_gp_frob_cost, trace_distance, tvd, get_density_matrix, get_average_density_matrix
from util.fix_global_phase import fix_phase

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from bqskit.qis import UnitaryMatrix
from bqskit.ext import bqskit_to_qiskit

import time

shots = 100
partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

circ_name = "qae11"

lang = OPENQASM2Language()

def get_qcirc(circ: Circuit) -> QuantumCircuit:
    circ.unfold_all()
    # Remove all GlobalPhaseGates
    circ.remove_all(GlobalPhaseGate())
    return bqskit_to_qiskit(circ)


def get_random_circs(num_circs: int,
                    block_circs: list[Circuit], 
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
    rand_circ_inds = np.random.randint(0, len(block_circs), size=num_circs)
    rand_param_inds = np.random.randint(0, shm_array.shape[1], size=num_circs)
    circ_strs = []
    for c_ind, p_ind in zip(rand_circ_inds, rand_param_inds):
        # Now we need to get the random circuits
        rand_circ: Circuit = block_circs[c_ind].copy()
        circ_param = shm_array[c_ind][p_ind]
        # Now we need to set the params of the circuit
        rand_circ.set_params(circ_param)
        fix_phase(rand_circ, target)
        circ_strs.append(rand_circ)
    return circ_strs


def generate_full_circuits(block_circs: dict[str, list[Circuit]], 
                           block_names: list,
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
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                  block_circs[block_name], 
                                                  shm_name=shm_name, 
                                                  param_shape=param_shape,
                                                  target=target)
        
    # print("Finish Getting Circs: ", time.time() - start, flush=True)
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

    # print("Finish Generating Circs: ", time.time() - start, flush=True)
    return all_circs

# Circ 
if __name__ == '__main__':
    block_circs = {}
    block_names = []

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    max_tol = float(argv[1]) if len(argv) > 1 else 0.01

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    # if initial_circ.num_qudits <= 10:
    target = initial_circ.get_unitary()
    initial_qcirc = bqskit_to_qiskit(initial_circ)
    target_sv = Statevector.from_instruction(initial_qcirc).data
    target_sv_prob = np.abs(target_sv)**2
    target_dm = get_density_matrix(target_sv)

    block_targets = {}
    block_dirs, count = get_circ_data(circ_name, max_tol, False, no_base=True)
    block_names = sorted([x[0] for x in block_dirs])
    print("Avg Count: ", count, "TKET Count: ", 110)
    print("Block dirs: ", block_dirs)
    print("Block names: ", block_names)
    exit(0)

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
        bc_file = load_block(circ_name, block_name)
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
    # print("Partitioned Circuit: ", partitioned_circ.gate_counts)
    # print(partitioned_circ_save_file)

    ensemble_sizes = [1, 5, 10, 20, 40, 80, 160, 320]
    # ensemble_sizes = [2, 10, 20]

    ensemble_circuits = []
    num_trials = 6
    ensemble_frob_costs = []
    ensemble_tvd_costs = []
    ensemble_trace_dist_costs = []

    # Get avg cost
    target = initial_circ.get_unitary()
    for ens_size in ensemble_sizes:
        mean_frob_costs = []
        mean_tvd_costs = []
        mean_trace_dist_costs = []
        print("Ensemble Size: ", ens_size, flush=True)
        for i in range(num_trials):
            # print("Trial: ", i, flush=True)
            ens = generate_full_circuits(block_circs, 
                                         block_names,
                                         param_shapes,
                                         block_targets,
                                         partitioned_circ,
                                         max_tol=max_tol, 
                                         ens_size=ens_size)
            mean_un = np.mean(np.array([c.get_unitary() for c in ens]), axis=0)
            # print(ens[0].gate_counts)
            qiskit_circs = [get_qcirc(c) for c in ens]
            svs = [Statevector.from_instruction(circ).data for circ in qiskit_circs]
            sv_probs = [np.abs(sv)**2 for sv in svs]
            mean_sv_prob = np.mean(np.array(sv_probs), axis=0)
            mean_dm = get_average_density_matrix(svs)
            mean_frob_costs.append(normalized_gp_frob_cost(mean_un, target))
            mean_trace_dist_costs.append(trace_distance(target_dm, mean_dm))
            mean_tvd_costs.append(tvd(target_sv_prob, mean_sv_prob))
        ensemble_frob_costs.append(mean_frob_costs)
        ensemble_tvd_costs.append(mean_tvd_costs)
        ensemble_trace_dist_costs.append(mean_trace_dist_costs)
        print("Avg Cost: ", np.mean(mean_frob_costs))
    

    for block_name, shm in shms.items():
        print("Closing Shared Memory: ", shm_name)
        shm.close()
        shm.unlink()

    all_costs = {
        "frob": ensemble_frob_costs,
        "tvd": ensemble_tvd_costs,
        "trace_dist": ensemble_trace_dist_costs
    }

    pickle.dump(all_costs, open(f"ensemble_costs_{circ_name}_{max_tol}.pickle", 'wb'))