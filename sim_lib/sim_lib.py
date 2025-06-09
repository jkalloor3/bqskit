import cudaq
from cudaq import spin
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, U3Gate

from qiskit import QuantumCircuit
import numpy as np

from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
from qiskit import transpile

from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error

from qiskit.quantum_info import SparsePauliOp

#### NOISE MODELS #####

def create_nisq_noise_model(rb_err_1q, rb_err_2q):
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(rb_err_1q, 1)
    two_q_error = depolarizing_error(rb_err_2q, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model

def create_nisq_noise_model_coherent(rb_err_1q, rb_err_2q):

    # Define a small unitary error on 2 qubits (e.g., ZZ rotation error)
    theta = rb_err_2q / 10
    # Construct a coherent error: small ZZ interaction
    zz_error = np.diag([np.exp(-1j*theta), np.exp(1j*theta), 
                        np.exp(1j*theta), np.exp(-1j*theta)])
    coherent_error = coherent_unitary_error(zz_error)

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(rb_err_1q, 1)
    two_q_error = depolarizing_error(rb_err_2q, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])
    # Add coherent error to CX gates
    noise_model.add_all_qubit_quantum_error(coherent_error, ['cx'])

    return noise_model


def create_cliff_noise_model(t_err, logical_err):
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    t_err = depolarizing_error(t_err, 1)
    one_q_error = depolarizing_error(logical_err, 1)
    two_q_error = depolarizing_error(logical_err, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['h', 's', 'z', 'x', 'y', 'i', 'u', 'u3'])
    noise_model.add_all_qubit_quantum_error(t_err, ['t', 'tdg'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model


def create_rz_err_model(t_err: float, prec: int, logical_err: float):
    noise_model = NoiseModel()
    # Add depolarizing error
    # An RZ decomposes in 10 * precision T gates
    t_fid = 1 - t_err
    num_ts_per_rz = 10 * prec
    rz_fid = (t_fid ** num_ts_per_rz)
    rz_err_rate = 1 - rz_fid
    rz_err = depolarizing_error(rz_err_rate, 1)
    t_err = depolarizing_error(t_err, 1)
    u3_fid = (t_fid ** (num_ts_per_rz * 3))  # u3 decomposes in 3 RZ gates
    u3_err = depolarizing_error(1 - u3_fid, 1)

    one_q_error = depolarizing_error(logical_err, 1)
    two_q_error = depolarizing_error(logical_err, 2)

    noise_model.add_all_qubit_quantum_error(one_q_error, ['h', 's', 'z', 'x', 'y', 'i', 'u', 'u3'])
    noise_model.add_all_qubit_quantum_error(t_err, ['t', 'tdg'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model


# CUDA-Q NOISE MODELS
def create_cudaq_nisq_noise_model(rb_err_1q, rb_err_2q, num_qubits: int = 2): 
    # We model the 2-qubit error as a 1-qubit depolarization err
    # Create an empty noise model
    noise = cudaq.NoiseModel()
    dep_err = cudaq.Depolarization1(rb_err_1q)
    dep_err_2 = cudaq.Depolarization1(rb_err_2q / 2)
    for i in range(num_qubits):
        noise.add_channel("u3", [i], dep_err)
        # Add to each qubit after CNOT gate
        noise.add_channel("rx", [i], dep_err_2)
    # noise.add_channel("cx", [0, 1], dep_err_2)
    return noise


# QISKIT SIMULATION
def run_single_nisq_circ(job, 
                   noisy: bool = True, 
                   num_shots: int = 1024,
                   num_threads: int = 1) -> list[float]:
    '''
    Run a single circuit and get the observable for the output density matrix.
    '''
    if noisy:
        noise_model = create_nisq_noise_model_coherent(1e-4, 1e-3)
        backend = AerSimulator(noise_model=noise_model, 
                               max_parallel_threads=num_threads)
    else:
        backend = AerSimulator(max_parallel_threads=num_threads)

    estimator = Estimator(backend, options={"default_shots": num_shots})
    job = estimator.run([job])
    ev = job.result()[0].data.evs
    return ev

def run_nisq_circs(job_args, 
                   noisy: bool = True, 
                   num_shots: int = 1024,
                   **kwargs) -> list[float]:
    '''
    Run a single circuit and get the observable for the output density matrix.
    '''
    if noisy:
        noise_model = create_nisq_noise_model_coherent(1e-4, 1e-3)
        backend = AerSimulator(noise_model=noise_model, **kwargs)
    else:
        backend = AerSimulator(**kwargs)

    estimator = Estimator(backend, options={"default_shots": num_shots})
    job = estimator.run(job_args)
    evs = [job.result()[i].data.evs for i in range(len(job_args))]
    return evs


def create_ham_args(qcircs: list[QuantumCircuit], ham: SparsePauliOp):
    """
    Return Job args for Estimator to run
    """
    return [(transpile(c, basis_gates=["cx", "u"]), ham) for c in qcircs]


# CUDAQ SIMULATION
def cuda_kernel(circ: Circuit, 
                measure: bool = False,
                add_coherent_error: float = 0) -> cudaq.Kernel:
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(circ.num_qudits)
    for op in circ.operations():
        q0 = op.location[0]
        if isinstance(op.gate, CNOTGate):
            q1 = op.location[1]
            kernel.cx(qubits[q0], qubits[q1])
            # Add dummy RX gates to inject noise
            kernel.rx(1e-14 + add_coherent_error,  qubits[q0])
            kernel.rx(1e-14 + add_coherent_error, qubits[q1])
        elif isinstance(op.gate, U3Gate):
            kernel.u3(op.params[0], op.params[1], op.params[2], 
                      qubits[q0])
    
    if measure:
        kernel.mz(qubits)
    return kernel


def create_ham_args_cudaq(circs: list[Circuit], ham: cudaq.SpinOperator, 
                          add_coherent_error: float = 0) -> list[
                              tuple[cudaq.Kernel, cudaq.SpinOperator]]:
    
    ks = []
    for c in circs:
        k = cuda_kernel(c, False, add_coherent_error)
        ks.append((k, ham))

    return ks


def run_cudaq_nisq_circs(circs: list[Circuit],
                         ham: cudaq.SpinOperator,
                         add_coherent_error: float = 0,
                         num_shots: int = 1024,
                         use_noise: bool = True,
                         average: bool = False) -> list[float]:
    '''
    Run a single circuit and get the observable for the output density matrix.
    '''
    if use_noise:
        noise_model =create_cudaq_nisq_noise_model(1e-4, 
                                                   1e-3, 
                                                   num_qubits=circs[0].num_qudits)
    else:
        noise_model = cudaq.NoiseModel()
    results = []
    for circ in circs:
        result = cudaq.observe(cuda_kernel(circ, False, add_coherent_error), ham, 
                            shots_count=num_shots, noise_model=noise_model)
        results.append(result.expectation())
    if average:
        return np.mean(results)
    return results


# Creating Hamiltonians

def generate_lgt_hamiltonian(num_qubits: int, x: int) -> SparsePauliOp:
    '''
    H = He (electric) + Hb (magnetic)
    
    He = 3/8 * (3N + 1) - 9/8 * (Z_0 + Z_{N-1}) - 3/4 (sum_{n=1}^{N-2} Z_n)
    - 3/8 * (sum_{n=0}^{N-2} Z_n Z_{n+1})

    Hb = -x/2 (3 + Z_1)(X_0) - x/2 (3 + Z_{N-2})(X_{N-1}) - 
    [x/8 (sum_{n=1}^{N-2} (9 + 3Z_{n-1} + 3Z_{n+1} + Z_{n-1}Z_{n+1}))(X_n))
    '''

    # Generate He
    Z_0_term = ("Z" + "I" * (num_qubits - 1), -9/8)
    Z_N1_term = ("I" * (num_qubits - 1) + "Z", -9/8)
    He = [
        Z_0_term,
        Z_N1_term
    ]

    for i in range(1, num_qubits - 1):
        Z_n_term = ("I" * i + "Z" + "I" * (num_qubits - i - 1), -3/4)
        He.append(Z_n_term)
    
    for i in range(num_qubits - 1):
        Z_nZ_n1_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), -3/8)
        He.append(Z_nZ_n1_term)

    # Generate Hb
    X_0_term = ("X" + "I" * (num_qubits - 1), -x/2 * (3))
    X_0_Z_1_term = ("XZ" + "I" * (num_qubits - 2), -x/2)
    X_N1_term = ("I" * (num_qubits - 1) + "X", -x/2 * (3))
    X_N1_Z_N2_term = ("I" * (num_qubits - 2) + "ZX", -x/2)
    Hb = [
        X_0_term,
        X_0_Z_1_term,
        X_N1_term,
        X_N1_Z_N2_term
    ]

    for i in range(1, num_qubits - 1):
        X_term = ("I" * i + "X" + "I" * (num_qubits - i - 1), -9*x/8)
        # 3Z_{n-1}*X_n
        ZX_term = ("I" * (i - 1) + "ZX" + "I" * (num_qubits - i - 1), -3*x/8)
        # 3Z_{n+1}*X_n
        XZ_term = ("I" * i + "XZ" + "I" * (num_qubits - i - 2), -3*x/8)
        ZXZ_term = ("I" * (i - 1) + "ZXZ" + "I" * (num_qubits - i - 2), -x/8)

        Hb.extend(
            [
                X_term,
                ZX_term,
                XZ_term,
                ZXZ_term
            ]
        )

    op = SparsePauliOp.from_list(He + Hb)
    return op

def generate_lgt_hamiltonian_cudaq(num_qubits: int, x: int) -> cudaq.SpinOperator:
    '''
    H = He (electric) + Hb (magnetic)
    
    He = 3/8 * (3N + 1) - 9/8 * (Z_0 + Z_{N-1}) - 3/4 (sum_{n=1}^{N-2} Z_n)
    - 3/8 * (sum_{n=0}^{N-2} Z_n Z_{n+1})

    Hb = -x/2 (3 + Z_1)(X_0) - x/2 (3 + Z_{N-2})(X_{N-1}) - 
    [x/8 (sum_{n=1}^{N-2} (9 + 3Z_{n-1} + 3Z_{n+1} + Z_{n-1}Z_{n+1}))(X_n))
    '''

    H = cudaq.SpinOperator()

    # Electric part (He)
    H += -9/8 * spin.z(0)
    H += -9/8 * spin.z(num_qubits - 1)
    for i in range(1, num_qubits - 1):
        H += -3/4 * spin.z(i)
    for i in range(num_qubits - 1):
        H += -3/8 * spin.z(i) * spin.z(i + 1)

    # Magnetic part (Hb)
    H += -x/2 * 3 * spin.x(0)
    H += -x/2 * spin.x(0) * spin.z(1)
    H += -x/2 * 3 * spin.x(num_qubits - 1)
    H += -x/2 * spin.z(num_qubits - 2) * spin.x(num_qubits - 1)

    for i in range(1, num_qubits - 1):
        # -9x/8 * X_n
        H += -9 * x / 8 * spin.x(i)

        # -3x/8 * Z_{n-1} X_n
        H += -3 * x / 8 * spin.z( i - 1) * spin.x(i)

        # -3x/8 * Z_{n+1} X_n
        H += -3 * x / 8 * spin.x(i) * spin.z(i + 1)

        # -x/8 * Z_{n-1} X_n Z_{n+1}
        H += -x / 8 * spin.z(i - 1) * spin.x(i) * spin.z(i + 1)

    return H

def generate_tfim_hamiltonian_cudaq(num_qubits: int) -> cudaq.SpinOperator:
    '''
    1D Transverse Field Ising Model Hamiltonian:
    ZiZj + 
    '''

    Jz = 1.0
    mu_x = 1.0

    H = cudaq.SpinOperator()

    for i in range(num_qubits - 1):
        H += Jz * spin.z(i) * spin.z(i + 1)

    for i in range(num_qubits):
        H += mu_x * spin.x(i)

    return H