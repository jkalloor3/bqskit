# We are plotting QITE sim for 8 spins with initial spins set to 0. We want to see how quickly it converges to the ground state energy
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# Def calculation of Hamiltonian
import scipy.linalg as la
from qiskit_aer.noise import NoiseModel, pauli_error
from util import GridSynthGate
from bqskit.ir.circuit import Operation, Circuit, CircuitGate, CircuitPoint
from bqskit.ir.gates import ZGate
from util import load_ensemble
from util.counter import fix_angle_workflow
import pickle
from math import ceil
import multiprocessing as mp
import concurrent.futures

# Now, we want to plot the values we get from ensembles
from multiprocessing.shared_memory import SharedMemory
from bqskit.ext import bqskit_to_qiskit
from bqskit.ir.lang.qasm2 import OPENQASM2Language

# class SharedDict:
#     def __init__(self):
#         self.manager = mp.Manager()
#         self.data = self.manager.dict()
#         self.queue = mp.Queue()
#         self.process = mp.Process(target=self._process_queue, daemon=True)
#         self.process.start()

#     def _process_queue(self):
#         while True:
#             key, value = self.queue.get()
#             if key is None:  # Shutdown signal
#                 break
#             self.data[key] = value

#     def put(self, key, value):
#         """Delayed put (queued for processing)."""
#         self.queue.put((key, value))

#     def get(self, key, default=None):
#         """Instant get (no conflict)."""
#         return self.data.get(key, default)

#     def shutdown(self):
#         """Gracefully stop the background process."""
#         self.queue.put((None, None))
#         self.process.join()

# t_cache: dict[tuple, Circuit] = pickle.load(open("/home/jkalloor/bqskit/block_checkpoints_final_paper_clifft_tket/QITE_8_4_0_3.0/angles_cache.pkl", "rb"))
# # t_cache = {}
# print("Num Items in T Cache: ", len(t_cache))
t_cache_file = "/home/jkalloor/bqskit/block_checkpoints_final_paper_clifft_tket/QITE_8_4_0_3.0/angles_cache.pkl"

default_precision = 6

#Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
#setup inital hamiltonian

#number of samples
T = 100
nmetts = T+10
#number of qubits
N = 8
#Hamiltonian parameters
Jz = 1.0 #ising interaction strength
mu_x = 1.0 #transverse magnetic field strength
    
def gen_hamiltonian(num_qubits: int = 8, Jz = 1.0, mu_x = 1.0):
    """Generate the Hamiltonian for the system."""
    #initial Hamiltonian
    #size of matrix
    msize = 2 ** num_qubits
    ham = np.zeros((msize, msize), dtype=np.complex128)
    # Ising interaction in 1D line
    for i in range(num_qubits-1):
        ising_term = np.kron(sz, sz)
        # left term
        if i > 0:
            ising_term = np.kron(np.eye(2**i), ising_term)
        if i < num_qubits - 2:
            qubits_right = num_qubits - i - 2
            ising_term = np.kron(ising_term, np.eye(2**qubits_right))
        # add to Hamiltonian
        ham += Jz * ising_term

        transverse_term = np.kron(sx, np.eye(2 ** (num_qubits -i - 1)))
        if i > 0:
            transverse_term = np.kron(np.eye(2**i), transverse_term)
        
        ham += mu_x * transverse_term
    # add the last term
    ham += mu_x * np.kron(np.eye(2**(num_qubits-1)), sx)
    return ham

ham = gen_hamiltonian(num_qubits=N, Jz=Jz, mu_x=mu_x)


def generate_circ(circuit: Circuit, param: np.ndarray, shared_dict = {}) -> str:
    circ = circuit.copy()
    circ.set_params(param)
    # target = circuit.get_unitary()
    # Replace each GridSynthGate with a CircuitGate
    pts = []
    t_circs = []
    print("Num things in shared dict: ", len(shared_dict), flush=True)
    for cycle, op in circ.operations_with_cycles():
        if isinstance(op.gate, GridSynthGate):
            loc = op.location
            pts.append(CircuitPoint(cycle, op.location[0]))
            ind = (op.params[0], op.params[1])
            gg_circ: Circuit = shared_dict.get(ind, None)
            if gg_circ is None:
                # Count the number of T gates
                gg_params = [op.params[0], op.params[1], 0]
                gg_circ = GridSynthGate().get_circuit(gg_params)
                shared_dict[ind] = gg_circ
            new_circ = gg_circ.copy()
            if op.params[2] == 1:
                # Add Z gates to front and back
                new_circ.append_gate(ZGate(), [0])
                new_circ.insert_gate(0, ZGate(), [0])
            new_op = Operation(CircuitGate(new_circ), loc)
            t_circs.append(new_op)
    
    circ.batch_replace(pts, t_circs)
    circ.unfold_all()
    # utry = circ.get_unitary()
    # print(utry.get_distance_from(target))
    return circ.to("qasm")

def get_qcirc(circ: Circuit) -> QuantumCircuit:
    circ.unfold_all()
    # Remove all GlobalPhaseGates
    return bqskit_to_qiskit(circ)

lang = OPENQASM2Language()

def get_random_circs(num_circs: int,
                    block_circs: list[Circuit], 
                    shm_name: str,
                    param_shape: np.ndarray = None,
                    shared_dict = {}) -> list[str]:
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
    rand_circs = [block_circs[i] for i in rand_circ_inds]
    rand_params = [shm_array[i][j] for i, j in zip(rand_circ_inds, rand_param_inds)]

    circ_strs = [generate_circ(circ, param, shared_dict=shared_dict) for circ, param in zip(rand_circs, rand_params)]
    return circ_strs

def sim_full_circuits(block_circs:list[Circuit], 
                      shm_name: str,
                      param_shape: np.ndarray,
                      ens_size: int,
                      t_errs: list[float],
                      shared_dict: dict) -> dict[float, tuple[float, float]]:
    all_circs = []
    circs = get_random_circs(ens_size, block_circs, shm_name=shm_name, param_shape=param_shape, shared_dict=shared_dict)
    num_rzs = [(c.count("rz( ") + c.count("rz(")) for c in circs]
    precisions = []
    default_precision = 6
    gss = {}
    for t_err in t_errs:
        for rz in num_rzs:
            if rz == 0:
                precisions.append(default_precision)
            else:
                precisions.append(default_precision + ceil(np.log10(rz)))
        all_circs = [QuantumCircuit.from_qasm_str(circ) for circ in circs]
        [circ.save_density_matrix() for circ in all_circs]
    
        # Now we need to sim the circuits
        gs = calculate_gs_with_noise(all_circs, precisions, t_err)
        gss[t_err] = gs
    return gss

def get_noise_model(precision = 8, t_err = 1e-5):
    noise_model = NoiseModel()
    # Add depolarizing error
    # An RZ decomposes in 10 * precision T gates
    t_fid = 1 - t_err
    num_ts_per_rz = 10 * precision
    rz_fid = (t_fid ** num_ts_per_rz)
    rz_err = 1 - rz_fid
    err = pauli_error([('X', rz_err / 3), ('Y', rz_err / 3), ('Z', rz_err / 3), ('I', rz_fid)])
    err_t = pauli_error([('X', t_err / 3), ('Y', t_err / 3), ('Z', t_err / 3), ('I', t_fid)])
    u3_fid = 1 - 3 * rz_err
    err_u3 = pauli_error([('X', rz_err), ('Y', rz_err), ('Z', rz_err), ('I', u3_fid)])

    noise_model.add_all_qubit_quantum_error(err, ['rz'])
    noise_model.add_all_qubit_quantum_error(err_t, ['t'])

    return noise_model

def calculate_gs_with_noise(circs: list[QuantumCircuit], 
                            precisions: list[int],
                            t_err = 1e-5):
    svs = []
    for i, circ in enumerate(circs):
        prec = precisions[i]
        # circ.metadata = {"precision": precisions[i]}
        noise_model = get_noise_model(prec, t_err)
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(circ).result()
        svs.append(result.data(0)['density_matrix'].data)
    # Calculate the energy
    engs = [eng_func(sv) for sv in svs]
    return np.mean(engs), np.std(engs)

# Lambda function to calculate energy
eng_func = lambda sv: np.real(np.trace(ham @ sv))

def calculate_gs(circs: list[QuantumCircuit]):
    sim = AerSimulator(method="density_matrix")
    result = sim.run(circs).result()
    # counts = result.get_counts(circ)
    # print(result.data(0))
    svs = [result.data(i)['density_matrix'].data for i in range(len(circs))]
    # Calculate the energy
    engs = [eng_func(sv) for sv in svs]
    return np.mean(engs)

if __name__ == '__main__':

    t_cache = pickle.load(open(t_cache_file, "rb"))
    # shared_dict = SharedDict()
    manager = mp.Manager()
    shared_dict = manager.dict()
    # t_cache = {}

    for key, value in t_cache.items():
        shared_dict[key] = value
        print("-", end="")
    print()
    print("Num Items in T Cache: ", len(t_cache))


    t_errs = [1e-6, 5e-6, 5e-5]
    print("T Error Rates: ", t_errs)
    qite_file = "/home/jkalloor/bqskit/good_blocks_tket/QITE_8_{timestep}_0.qasm"
    timesteps = range(5)
    energies = []
    circs = []
    precisions = []
    for t_step in timesteps:
        b_circ = Circuit.from_file(qite_file.format(timestep=t_step))
        fix_angle_workflow(b_circ, default_precision)
        precisions.append(default_precision + ceil(np.log10(b_circ.num_params)))
        print(b_circ.gate_counts)
        circ = get_qcirc(b_circ)

        # circ = QuantumCircuit.from_qasm_file(qite_file.format(timestep=t_step))
        # circ.measure_all()
        circ.save_density_matrix()
        circs.append(circ)

    print("Precisions: ", precisions)

    # # #initial Hamiltonian
    # # ham = gen_hamiltonian(num_qubits=N, Jz=Jz, mu_x=mu_x)
    # # #ground state energy
    # eigvals_i, eigvecs_i = la.eig(ham)
    # gs_eng = np.real(min(eigvals_i))
    # print("Ground state energy: ", gs_eng)

    # energies = []
    # for circ in circs:
    #     eng = calculate_gs([circ])
    #     energies.append(eng)
    #     # print(f"Energy for timestep {len(energies)-1}: {eng}")
    # print("Original energies: ", energies)

    # # For each circ, simulate the output and calculate the energy
    # # noise_model = get_noise_model(0.001)
    # noisy_energies = []
    # for circ in circs:
    #     eng = calculate_gs_with_noise([circ], precisions)
    #     noisy_energies.append(eng)
    
    # print("Noisy energies: ", noisy_energies)

    circ_name_ft = "QITE_8_{timestep}"
    shm_names = []
    shms = []
    block_circs: list[list[Circuit]] = []
    param_shapes = []

    for timestep in timesteps:
        circ_name = circ_name_ft.format(timestep=timestep)
        params = np.load(f"/home/jkalloor/bqskit/block_checkpoints_final_paper_clifft_tket/{circ_name}_0_3.0/ensemble_0_jiggles_.npy")

        shm_name = f"{circ_name}_shm"
        try:
            shm = SharedMemory(create=True, size=params.nbytes, name=shm_name)
        except:
            shm = SharedMemory(name=shm_name, create=False)
        shm_array = np.ndarray(params.shape, dtype=np.float64, buffer=shm.buf)
        shm_array[:] = params[:]


        print("Shared memory created", flush=True)

        param_shapes.append(params.shape)
        shm_names.append(shm_name)
        shms.append(shm)



    def load_ens(timestep: int):
        circ_name = circ_name_ft.format(timestep=timestep)
        bq_circs = load_ensemble(f"/home/jkalloor/bqskit/block_checkpoints_final_paper_clifft_tket/{circ_name}_0_3.0/ensemble_0_.qasms")
        return bq_circs

    with mp.Pool(len(timesteps)) as pool:
        block_circs = pool.map(load_ens, timesteps)

    ens_energies = {}
    ensemble_sizes = [32, 64, 128, 256, 512]
    for ens_size in ensemble_sizes:
        ens_energies[ens_size] = [0] * len(timesteps)
    # Now sim_full_circuits
    all_processes = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        for i, shm_name in enumerate(shm_names):
            bq_circs = block_circs[i]
            param_shape = param_shapes[i]
            for ens_size in ensemble_sizes:
                # p = mp.Process(target=sim_full_circuits, args=(bq_circs, shm_name, param_shape, ens_size, t_errs))
                fut = executor.submit(sim_full_circuits, bq_circs, shm_name, param_shape, ens_size, t_errs, shared_dict=shared_dict)
                all_processes.append((i, ens_size, fut))

        print("All processes started")

    for i, ens_size, future in all_processes:
        ens_energies[ens_size][i] = future.result()

    # Ens Energies is a dict[ens_size, list (of timestep data)[dict[t_err, gs]]]

    print("Ensemble energies: ", ens_energies)

    # print("Final Num Items in T Cache: ", len(ultra))
    # pickle.dump(ultra.data, open("qite_sim_t_cache.pkl", "wb"))

    for shm in shms:
        shm.close()
        shm.unlink()

    sizes = "_".join([str(s) for s in ensemble_sizes])
    t_err_str = "_".join([str(t) for t in t_errs])
    pickle.dump(ens_energies, open(f"ens_energies_3.0_{sizes}_{t_err_str}_0_4.pkl", "wb"))


    # Plot Energies and Noisy Energies
    # fig, ax = plt.subplots()
    # ax.plot(timesteps, energies, label="Noise-Free Simulation", marker='o')
    # noisy_mean_energies = np.array([d[0] for d in noisy_energies])
    # ax.plot(timesteps, noisy_mean_energies, label="Noisy Simulation", marker='x')
    # for ens_size, ens_data in ens_energies.items():
    #     # ens_data is list of tuples
    #     mean_energies = np.array([d[0] for d in ens_data])
    #     std_energies = np.array([d[1] for d in ens_data])
    #     ax.plot(timesteps, mean_energies, label=f"Ensemble Size={ens_size}", marker='x')
    #     ax.fill_between(timesteps, mean_energies - std_energies, mean_energies + std_energies, alpha=0.2)
    # ax.axhline(y=gs_eng, color='b', linestyle='--')
    # ax.set_xlabel("Timestep")
    # ax.set_ylabel("Energy")
    # ax.set_title("QITE Simulation")
    # ax.legend()
    # # fig.show()
    # fig.savefig(f"qite_sim_ensemble_3.0_{t_err}.png", dpi=300)