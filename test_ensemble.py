import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import U3Gate, CNOTGate


from bqskit.passes import LEAPSynthesisPass, UpdateDataPass
from bqskit.compiler import Compiler

def calc_circ_obs(circ: Circuit) -> float:
    # Dummy observable calculation function
    # Replace with actual observable calculation logic
    return np.random.rand()

def run_circ(circ: Circuit) -> dict[str, int]:
    # Dummy circuit execution function
    # Replace with actual circuit execution logic
    return {"0": np.random.randint(0, 100), "1": np.random.randint(0, 100)}

def random_circuit(num_qubits: int, num_cnots: int) -> Circuit:
    circ = Circuit(num_qubits)

    for i in range(num_qubits):
        circ.append_gate(U3Gate(), [i], np.random.rand(3) * 2 * np.pi)

    for i in range(num_cnots):
        q0 = np.random.randint(0, num_qubits)
        q1 = q0
        while q1 == q0:
            q1 = np.random.randint(0, num_qubits)
        circ.append_gate(CNOTGate(), [q0, q1])
        circ.append_gate(U3Gate(), [q0], np.random.rand(3) * 2 * np.pi)
        circ.append_gate(U3Gate(), [q1], np.random.rand(3) * 2 * np.pi)
    return circ


if __name__ == '__main__':
    # Build a random circuit
    circuit = random_circuit(3, 2)
    print(circuit)

    compiler = Compiler(num_workers=2)

    leap_pass = LEAPSynthesisPass(
        success_threshold=1e-4,
        max_solutions=10
    )

    set_data_pass = UpdateDataPass("run_ensemble", True)

    target = circuit.get_unitary()

    _, data = compiler.compile(circuit, [set_data_pass, leap_pass], request_data=True)

    from bqskit.compiler import Compiler
    # Set up circuit and compiler ...

    # Create circuit sampler
    circuit_sampler = compiler.compile_ensemble(circuit)

    # Calculate probability distribution over
    # randomly sampled channel
    all_counts = {}
    NUM_SAMPLES = 1000
    for _ in range(NUM_SAMPLES):
        next_circ = circuit_sampler.random_sample()
        # Run circuit on HW and get counts
        circ_counts = run_circ(next_circ)
        all_counts.update(circ_counts)



    # # print(len(data["ensemble_circuits"]))

    # for circ in data["ensemble_circuits"]:
    #     # print(circ.gate_counts)
    #     new_un = circ.get_unitary()
    #     dist = target.get_distance_from(new_un)
    #     print(f'Distance: {dist}')