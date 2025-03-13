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

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, depolarizing_error, coherent_unitary_error)

from util import (load_circuit, get_circ_data, check_param_shape, 
                  load_block, load_compiled_circs_params_separate,
                  get_average_density_matrix, trace_distance,
                  get_density_matrix)
from util.distance import normalized_gp_frob_cost

def over_rotated_cnot(two_q_error: float) -> UnitaryMatrix:
    '''
    Overrotated CNOT gate with depolarizing error.

    '''

    utry = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1j * np.sin(two_q_error), np.cos(two_q_error)],
        [0, 0, np.cos(two_q_error), -1j * np.sin(two_q_error)]
    ]
    return UnitaryMatrix(utry)


assert np.allclose(over_rotated_cnot(0), CNOTGate().get_unitary())
# print("Unitary Diff: ",  normalized_gp_frob_cost(over_rotated_cnot(1e-4), CNOTGate().get_unitary()))

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

noise_model = create_noise_model(1e-4, 6e-3)
backend = AerSimulator(method="density_matrix", noise_model=noise_model)
ensemble_sizes = [1, 5, 10, 20, 40, 80, 160]
TOTAL_SHOTS = 100 * max(ensemble_sizes)
partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

circ_name = "qaoa10"

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

def sim_full_circuits(block_circs: dict[str, list[Circuit]], 
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
    results = backend.run(all_circs, shots=shots).result()
    # Aggregate the results
    all_dms = [np.array(results.data(i)['density_matrix']) for i in range(len(all_circs))]
    average_dm = np.mean(all_dms, axis=0)
    # agg_results = aggregate_results(all_dicts)
    return average_dm

# Circ 
if __name__ == '__main__':
    block_circs = {}
    block_names = []

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    max_tol = float(argv[1]) if len(argv) > 1 else 0.01

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    target = initial_circ.get_unitary()
    initial_qcirc = bqskit_to_qiskit(initial_circ)

    initial_qcirc = bqskit_to_qiskit(initial_circ)
    initial_qcirc.save_density_matrix()
    perfect_result = AerSimulator(method="density_matrix", shots=TOTAL_SHOTS).run(initial_qcirc).result()
    # print(perfect_result.data(0)['density_matrix'])
    perfect_dm = perfect_result.data(0)['density_matrix']
    noisy_result = backend.run(initial_qcirc, shots=TOTAL_SHOTS).result()
    noisy_dm = noisy_result.data(0)['density_matrix']

    initial_dist = trace_distance(perfect_dm, noisy_dm)

    print("Initial Trace Distance: ", initial_dist)

    block_dirs, count, tket_count = get_circ_data(circ_name, max_tol, False)
    block_names = sorted([x[0] for x in block_dirs])
    print("Avg Count: ", count, "TKET Count: ", tket_count)

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
        shm = SharedMemory(name=shm_name, 
                           create=True, 
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
            avg_dm = sim_full_circuits(block_circs, 
                                         block_names,
                                         param_shapes,
                                         partitioned_circ,
                                         max_tol=max_tol, 
                                         ens_size=ens_size)
            mean_tvd = trace_distance(perfect_dm, avg_dm)
            mean_costs.append(mean_tvd)
        ensemble_costs.append(mean_costs)
        print("Avg Trace Distance: ", np.mean(mean_costs))
    

    for block_name, shm in shms.items():
        print("Closing Shared Memory: ", shm_name)
        shm.close()
        shm.unlink()

    pickle.dump(ensemble_costs, open(f"ensemble_trace_dist_{circ_name}_{max_tol}.pickle", 'wb'))
