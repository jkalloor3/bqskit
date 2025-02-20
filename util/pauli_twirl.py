from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit, CircuitPoint, Operation
from bqskit.ir.gates import *
from bqskit.ir.gate import Gate
from bqskit.qis import PauliMatrices, UnitaryMatrix
import numpy as np

from .distance import normalized_gp_frob_cost

class PauliTwirlPass(BasePass):
    def __init__(self, gates_to_twirl: list[Gate] = None, num_twirls: int = 1):
        if gates_to_twirl is None:
            gates_to_twirl = [CXGate(), ECRGate()]
        self.gates_to_twirl = gates_to_twirl
        self.twirl_set = {}
        self.build_twirl_set()
        self.num_twirls = num_twirls

    def build_twirl_set(self):
        """
        Build a set of Paulis to twirl for each gate
        """
        # iterate through gates to be twirled
        for twirl_gate in self.gates_to_twirl:
            twirl_list = []
            num_params = twirl_gate.num_params
 
            # iterate through Paulis on left of gate to twirl
            for pauli_left in PauliMatrices(twirl_gate.num_qudits):
                # iterate through Paulis on right of gate to twirl
                for pauli_right in PauliMatrices(twirl_gate.num_qudits):
                    #Save pairs that produce same operation as gate to twirl
                    # Try 5 rand params to check
                    close = True
                    for _ in range(5):
                        rand_params = np.random.rand(num_params) * 2 * np.pi
                        left = pauli_left @ twirl_gate.get_unitary(rand_params)
                        right = twirl_gate.get_unitary(rand_params) @ pauli_right
                        if not np.allclose(left, right):
                            close = False
                            break
                    if close:
                        # print("Pauli Pair: ", pauli_left, pauli_right)
                        # print("Gate: ", twirl_gate.name)
                        loc = tuple(range(twirl_gate.num_qudits))
                        left_op = Operation(U3Gate(), loc, 
                                            U3Gate().calc_params(UnitaryMatrix(pauli_left)))
                        right_op = Operation(U3Gate(), loc, 
                                             U3Gate().calc_params(UnitaryMatrix(pauli_right)))
                        twirl_list.append((left_op, right_op))

            self.twirl_set[twirl_gate.name] = twirl_list

    async def run(self, circ: Circuit, data: PassData) -> None:
        circs = []
        for _ in range(self.num_twirls):
            # Create a copy of the circuit
            circuit = circ.copy()
            for cycle, op in circuit.operations_with_cycles():
                if op.gate.name in self.gates_to_twirl:
                    # Get the twirl set for the gate
                    twirl_list = self.twirl_set[op.gate.name]
                    # Get a random twirl pair
                    twirl_pair = np.random.choice(twirl_list)
                    # Replace gate with left, twirl_gate, right
                    repl_circ = Circuit(op.num_qudits)
                    repl_circ.append(twirl_pair[0])
                    repl_circ.append(op, tuple(range(op.num_qudits)))
                    repl_circ.append(twirl_pair[1])
                    circuit.replace_with_circuit((cycle, 
                                                  op.location[0]), 
                                                 repl_circ, 
                                                 as_circuit_gate=True)
            # Unfold the circuit
            circuit.unfold_all()
            # Add the twirled circuit to the list
            circs.append(circuit)
        data['twirled_circuits'] = circs