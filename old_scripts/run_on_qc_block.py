from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, XGate, YGate, ZGate, IdentityGate, RXGate
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
from qiskit_aer.noise import (NoiseModel, depolarizing_error, 
                              mixed_unitary_error)

from util import (get_block_names, check_param_shape, 
                  load_block, load_compiled_circs_params_separate, 
                  trace_distance, normalized_gp_frob_cost)

import os

def over_rotated_cnot(eps: float) -> Circuit:
    '''
    Over-rotated CNOT gate.

    Return a list of unitary matrices and probabilities.

    We are assuming that we always over-rotate the CNOT gate by eps.
    Then, with probability eps, we also have Pauli X, Y, and Z errors
    '''

    print("Eps: ", eps)
    mat = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.cos(np.pi/2 + eps), -np.sin(np.pi/2 + eps)],
                    [0, 0, np.sin(np.pi/2 + eps), np.cos(np.pi/2 + eps)]])
    
    # Actual effect is on a CNOT gate
    # cnot_un = CNOTGate().get_unitary()

    # Over-rotate X
    effect = np.kron(IdentityGate().get_unitary(), RXGate().get_unitary([eps]))

    # effect = np.array([[1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]])

    # effect = cnot_un.dagger @ mat @ cnot_un.dagger

    print("Effect: ", effect)
    print("RX Gate: ", RXGate().get_unitary([eps]))

    probs = [1 - eps, eps/3, eps/3, eps/3]

    errors = []
    # pauli errors
    for i, q1_err in enumerate([IdentityGate(), XGate(), YGate(), ZGate()]):
        for j, q2_err in enumerate([IdentityGate(), XGate(), YGate(), ZGate()]):
            err = np.kron(q1_err.get_unitary(), q2_err.get_unitary())
            p1 = probs[i]
            p2 = probs[j]
            effect = effect @ err
            errors.append((effect, p1 * p2))

    # Assert probabilities sum to 1
    assert np.isclose(sum([p for _, p in errors]), 1.0)

    return errors

def create_noise_model(one_q_err: float, 
                             two_q_err: float) -> NoiseModel:
    '''
    Noise Model with overrotated CNOT and stochastic depolarizing noise.

    '''
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(one_q_err, 1)
    # two_q_error = depolarizing_error(two_q_err, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])
    # noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    coherent_utries = over_rotated_cnot(two_q_err)
    cnot_error = mixed_unitary_error(coherent_utries)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'])

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
    
    cost_path = f"ensemble_trace_dists_block/ensemble_trace_dist_qc_{circ_name}/{block_name}/{two_q_err}/{tol}_all.pickle"
    if os.path.exists(cost_path):
        print("Already ran block: ", circ_name, block_name)
        return

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
    if avg_cnot_count >= initial_cnot_count:
        print("No CX reduction, skipping block: ", circ_name, block_name)
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
        ens_cost_path = f"ensemble_trace_dists_block/ensemble_trace_dist_qc_{circ_name}/{block_name}/{two_q_err}/{tol}_{ens_size}.pickle"
        if os.path.exists(ens_cost_path):
            continue
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
        Path(ens_cost_path).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(mean_costs, open(ens_cost_path, 'wb'))
        ensemble_costs.append(mean_costs)
        print("Avg Trace Distance: ", np.mean(mean_costs))
    

    for block_name, shm in shms.items():
        print("Closing Shared Memory: ", shm_name)
        shm.close()
        shm.unlink()

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