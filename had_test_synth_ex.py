import numpy as np
from bqskit.ir.gates import VariableUnitaryGate, HGate
from bqskit.passes import FullBlockZXZPass
from bqskit.qis.pauli import PauliMatrices
from scipy.linalg import expm
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler.compile import compile
import time

if __name__ == '__main__':
    time_evol_qubits = 4

    # time_evolution Hamiltonian
    # Create a random hermitian matrix on 4 qubits
    # Do a random sum of Paulis
    paulis = PauliMatrices(time_evol_qubits)
    random_coeffs = np.random.rand(len(paulis))
    H = paulis.dot_product(random_coeffs)


    # Ok, now calculate the unitary for the hadamard test
    U = expm(-1j * H)

    # Now create a controlled version of this unitary on N + 1 qubits
    # Has structure: 
    # |I 0|
    # |0 U|
    full_un = np.eye(2**(time_evol_qubits + 1), dtype=np.complex128)
    full_un[2**(time_evol_qubits):, 2**(time_evol_qubits):] = U

    # Now run the decomp on this unitary
    # To do this with BQSKit, need to create a circuit with a single variable
    # unitary gate and call BlockZXZ Decomp on it

    start = time.time()

    # Create a circuit with N + 1 qubits
    circ = Circuit(time_evol_qubits + 1)

    # Now create a unitary gate for the controlled unitary
    var_gate = VariableUnitaryGate(time_evol_qubits + 1)
    var_params = var_gate.calc_params(full_un)
    var_loc = list(range(time_evol_qubits + 1))

    circ.append_gate(var_gate, var_loc, var_params)

    workflow = [
        FullBlockZXZPass(min_qudit_size=2)
    ]

    # Compile this workflow
    print("Starting compilation...", flush=True)
    compiler = Compiler(num_workers=4)
    controlled_u_circ = compiler.compile(circ, workflow)
    # with Compiler(num_workers=2) as compiler:
    #     controlled_u_circ = compiler.compile(circ, workflow)
    print(controlled_u_circ.gate_counts)

    zxz_time = time.time() - start


    # Now, once this is compiled, we can compile the full hadamard test
    had_test_circ = Circuit(time_evol_qubits + 1)

    had_test_circ.append_gate(HGate(), [0])  # Hadamard on the first qubit
    had_test_circ.append_circuit(controlled_u_circ, list(range(time_evol_qubits + 1)))  # Controlled unitary on the full circuit
    had_test_circ.append_gate(HGate(), [0])  # Hadamard on the first qubit again

    # Pass to full bqskit compiler
    print("Starting full hadamard test compilation...", flush=True)
    had_test_compiled = compile(had_test_circ, max_synthesis_size=3, optimization_level=3, compiler=compiler)

    print(had_test_compiled.gate_counts)

    full_time = time.time() - start

    print(f"ZXZ Decomp Time: {zxz_time:.2f} seconds")
    print(f"Full Hadamard Test Compilation Time: {full_time:.2f} seconds")

    compiler.close()

    