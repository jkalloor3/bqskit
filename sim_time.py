import time
from bqskit import Circuit
from bqskit.ir.gates import CXGate, U3Gate
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import QasmSimulator, StatevectorSimulator, UnitarySimulator, Aer
from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit.
import numpy as np

def tvd(p, q, shots):
    p = {k: v/shots for k, v in p.items()}
    q = {k: v/shots for k, v in q.items()}
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

# Function to generate random quantum circuits
def generate_random_circuits(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    bcirc = Circuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            if np.random.random() > 0.5:
                bcirc.append_gate(CXGate(), [qubit, (qubit + 1) % num_qubits])
                circuit.cx(qubit, (qubit + 1) % num_qubits)
            
            if np.random.random() > 0.3:
                thetas = np.random.random(3)
                bcirc.append_gate(U3Gate(), [qubit], thetas)
                circuit.u(*thetas, qubit)
    return circuit, bcirc



# Define the number of qubits and depths for the circuits
num_qubits = [11, 12, 13, 14, 15, 16] 
# depths = [1, 2, 3, 4, 5]

# Create an empty list to store the execution times
statevec_times = []
shot_times = []
unitary_times = []
bc_un_times = []

# Iterate over the depths and generate circuits
for num_qubit in num_qubits:
    # Generate a random circuit
    circuit, bcirc = generate_random_circuits(num_qubit, 10)

    # Start the timer
    start_time = time.time()

    # Calculate the output state vector
    # backend =Aer.get_backend("statevector_simulator")
    # sampler = Sampler(mode=backend)
    # # qobj = assemble(transpile(circuit, backend=backend))
    # job = sampler.run([circuit])
    # result = job.result()
    # state_vector = result.get_statevector()

    statevector = Statevector.from_instruction(circuit)

    sv_counts = statevector.sample_counts(1000 * num_qubit)

    vec_time = time.time() - start_time

    statevec_times.append(vec_time)

    start_time = time.time()

    # Calculate the unitary
    start_time = time.time()
    unitary = Operator(circuit).data

    # Stop the timer and calculate the execution time
    end_time = time.time()
    unitary_time = end_time - start_time

    # BQSkit unitary
    start_time = time.time()
    un = bcirc.get_unitary()
    end_time = time.time()
    bq_un_time = end_time - start_time

    bc_un_times.append(bq_un_time)


    # Append the execution time to the list
    unitary_times.append(unitary_time)

    circuit.measure_all()


    # Run a simulation
    backend =Aer.get_backend("statevector_simulator")
    sampler = Sampler(mode=backend)
    start_time = time.time()
    job = sampler.run([circuit], shots=1000 * num_qubit)
    result = job.result()
    sim_counts = result[0].data.meas.get_counts()
    # state_vector = result.get_statevector()

    shot_time = time.time() - start_time

    shot_times.append(shot_time)

    # Print the results
    print(f"Circuit depth: {10}, num_qubits: {num_qubit}")
    print(f"Sim time: {shot_time} seconds")
    print(f"Statevector time: {vec_time} seconds")
    print(f"Unitary time: {unitary_time} seconds")
    print(f"TVD: {tvd(sv_counts, sim_counts, 1000 * num_qubit)}")
    print(f"BQSkit Unitary time: {bq_un_time} seconds")
    print()

# Print the overall execution times
print("Overall execution times:")
for i, (num_qubit, shot_time) in enumerate(zip(num_qubits, shot_times)):
    print(f"Num Qubits: {num_qubit} Shot Time: {shot_time} Statevector Time: {statevec_times[i]} Unitary Time: {unitary_times[i]} BQSkit Unitary Time: {bc_un_times[i]}")