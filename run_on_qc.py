from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
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

from util import (load_circuit, get_circ_data, check_param_shape, 
                  load_block, load_compiled_circs_params_separate, trace_distance)

import os

def create_noise_model(one_q_err: float, 
                             two_q_err: float) -> NoiseModel:
    '''
    Noise Model with overrotated CNOT and stochastic depolarizing noise.

    '''


    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(one_q_err, 1)
    two_q_error = depolarizing_error(two_q_err, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    # cnot_utry = over_rotated_cnot(1e-6).numpy
    # cnot_error = coherent_unitary_error(cnot_utry)
    # noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'])

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
                      noisy_backend: AerSimulator,
                      random_states: list[np.ndarray],
                      block_circs: dict[str, list[Circuit]], 
                      block_names: list,
                      param_shapes: dict[str, tuple[int]],
                      pcirc: Circuit,
                      max_tol: float, 
                      ens_size: int) -> dict[str, int]:
    all_circs = []

    circ_blocks = {}

    for block_name in block_names:
        shm_name = circ_name + "_" + str(max_tol) + "_" + block_name
        param_shape = param_shapes.get(block_name, None)
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                  block_circs[block_name], 
                                                  shm_name=shm_name, 
                                                  param_shape=param_shape)
    for i in range(ens_size):
        circ = pcirc.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            circ_str = circ_blocks[block_names[ind]][i]
            new_block_circ = lang.decode(circ_str)
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            assert isinstance(new_block_circ, Circuit)
            if op.gate._circuit.num_qudits != new_block_circ.num_qudits:
                print("Replacing circuit with different number of qudits")
                print("Old Circuit: ", op.gate._circuit)
                print("New Circuit: ", new_block_circ)
                print("Block Name: ", block_names[ind - 1])
            assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
            circ.replace_with_circuit(pt, new_block_circ, as_circuit_gate=True)
        circ.unfold_all()
        qc = bqskit_to_qiskit(circ)
        # qc.measure_all()
        qc.save_density_matrix()
        all_circs.append(qc)

    # Now we need to sim the circuits
    shots = TOTAL_SHOTS / ens_size
    avg_dms = run_noisy_sim(all_circs, random_states=random_states, shots=shots, backend=noisy_backend)
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


# Circ 
if __name__ == '__main__':
    block_circs = {}
    block_names = []

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    circ_name = str(argv[1])
    two_q_err = float(argv[2]) if len(argv) > 2 else 5e-3
    max_tol = float(argv[3]) if len(argv) > 2 else 0.01

    partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

    if not os.path.exists(partitioned_circ_save_file.format(circ_name=circ_name)):
        print("Partitioned circuit not found")
        exit(0)


    # Create backends
    perfect_backend = AerSimulator(method="density_matrix")
    noise_model = create_noise_model(1e-5, two_q_err=two_q_err)
    noisy_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

    # Get the circuit
    initial_circ: Circuit = load_circuit(circ_name, opt=False)
    num_q = initial_circ.num_qudits
    target = initial_circ.get_unitary()
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    initial_qcirc = bqskit_to_qiskit(initial_circ)
    initial_qcirc.save_density_matrix()

    # Run perfect simulation
    random_states = [random_statevector(2 ** num_q) for _ in range(NUM_RANDOM_STATES)]
    # random_states_2 = [random_statevector(2 ** num_q) for _ in range(NUM_RANDOM_STATES)]
    perfect_dms = run_noisy_sim([initial_qcirc], random_states=random_states, shots=TOTAL_SHOTS, backend=perfect_backend)

    # Run initial noisy simulation
    noisy_dms = run_noisy_sim([initial_qcirc], random_states=random_states, shots=TOTAL_SHOTS, backend=noisy_backend)
    initial_dists = [trace_distance(perfect_dm, noisy_dm) for perfect_dm, noisy_dm in zip(perfect_dms, noisy_dms)]


    print("Initial Trace Distance: ", initial_dists)

    block_dirs, count, tket_count = get_circ_data(circ_name, max_tol)
    block_names = sorted([x[0] for x in block_dirs])
    print("Avg Count: ", count, "TKET Count: ", tket_count)
    print(block_dirs)

    # Figure out how many ensembles we need, and calculate shared memory space
    param_shapes = {}
    shm_names = {}
    for block_name, block_data in block_dirs:
        if block_data[0] == 'orig' or block_data[0] == "tket":
            continue
        else:
            print(block_data)
            param_shape = check_param_shape(*block_data)
            shm_names[block_name] = shm_name = circ_name + "_" + str(max_tol) + "_" + block_name
            param_shapes[block_name] = param_shape
    
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
    for block_name, block_data in block_dirs:
        if block_data[0] == 'orig':
            bc_file = load_block(circ_name, block_name)
            block_target = Circuit.from_file(bc_file) 
            block_circs[block_name] = [block_target]
        elif block_data[0] == "tket":
            circ_file = load_block(circ_name, block_name, extra="_tket")
            block_circs[block_name] = [Circuit.from_file(circ_file)]
        else:
            print(block_name, block_data)
            # inds, probs = load_compiled_block_circuits_qp_inds(*block_data)
            circuits, params = load_compiled_circs_params_separate(*block_data)
            print("Num Circuits: ", len(circuits), flush=True)
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

    # ensemble_sizes = [2, 10, 20]

    ensemble_circuits = []
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
                                       noisy_backend,
                                       random_states,
                                       block_circs, 
                                       block_names,
                                       param_shapes,
                                       partitioned_circ,
                                       max_tol=max_tol,
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

    pickle.dump(ensemble_costs, open(f"ensemble_trace_dist_qc_{circ_name}_{max_tol}.pickle", 'wb'))
