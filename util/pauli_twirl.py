from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit, CircuitPoint
from bqskit.ir.gates import *
from bqskit.ir.gate import Gate
from bqskit.runtime import get_runtime
from bqskit.qis import PauliMatrices, UnitaryMatrix
import numpy as np

from .distance import normalized_gp_frob_cost

def get_pauli_circ(pauli_str: str = "I"):
    circ = Circuit(len(pauli_str))
    for i, p in enumerate(pauli_str):
        if p == 'X':
            circ.append_gate(RXGate(), [i], [np.pi])
        elif p == 'Y':
            circ.append_gate(RYGate(), [i], [np.pi])
        elif p == 'Z':
            circ.append_gate(RZGate(), [i], [np.pi])
    return circ

def combine_rotations(circ: Circuit):
    """
    Combine all consecutive RX, RY and RZ gates into a single RX, RY, or RZ gate
    """
    pts_to_remove = []
    for cycle, op in circ.operations_with_cycles():
        if op.gate.name in [RXGate().name, RYGate().name, RZGate().name]:
            # Form a region of consecutive RX, RY and RZ gates
            pt = CircuitPoint(cycle, op.location[0])
            while True:
                # Change all consecutive RX, RY, and RZ params to 0 and add
                # angle to current gate
                next_pts = circ.next(pt)
                if len(next_pts) == 0:
                    break
                assert len(next_pts) == 1
                next_pt = next_pts.pop()
                next_op = circ.get_operation(next_pt)
                if next_op.gate.name == op.gate.name:
                    op.params[0] += next_op.params[0]
                    next_op.params[0] = 0
                    pts_to_remove.append(next_pt)
                    pt = next_pt
                else:
                    break
    if len(pts_to_remove) == 0:
        print("Warning: No consecutive RX, RY or RZ gates found to combine")
        return
    circ.batch_pop(pts_to_remove)

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
            num_q = twirl_gate.num_qudits
 
            # iterate through Paulis on left of gate to twirl
            for left_str in PauliMatrices.get_pauli_strings(num_q, num_q):
                # iterate through Paulis on right of gate to twirl
                for right_str in PauliMatrices.get_pauli_strings(num_q, num_q):
                    #Save pairs that produce same operation as gate to twirl
                    # Try 5 rand params to check
                    close = True
                    pauli_left = PauliMatrices.from_string(left_str)
                    pauli_right = PauliMatrices.from_string(right_str)
                    for _ in range(5):
                        rand_params = np.random.rand(num_params) * 2 * np.pi
                        twirled = pauli_left @ twirl_gate.get_unitary(rand_params) @ pauli_right
                        if not np.allclose(twirled, twirl_gate.get_unitary(rand_params)):
                            close = False
                            break
                    if close:
                        print("Pauli Pair: ", left_str, right_str)

                        # print("Gate: ", twirl_gate.name)
                        left_circ = get_pauli_circ(left_str)
                        right_circ = get_pauli_circ(right_str)
                        twirl_list.append((left_circ, right_circ))

            self.twirl_set[twirl_gate.name] = twirl_list


    def create_rc_circ(circ: Circuit, twirl_set: dict[str, list[tuple[Circuit, Circuit]]]) -> Circuit:
        circuit = circ.copy()
        for cycle, op in circuit.operations_with_cycles():
            if op.gate.name in twirl_set:
                # Get the twirl set for the gate
                twirl_list = twirl_set[op.gate.name]
                # Get a random twirl pair
                rand_pair_ind = np.random.randint(0, len(twirl_list))
                twirl_pair = twirl_list[rand_pair_ind]
                # Replace gate with left, twirl_gate, right
                left_circ: Circuit = twirl_pair[0]
                right_circ: Circuit = twirl_pair[1]
                mini_loc = tuple(range(op.num_qudits))
                repl_circ = left_circ.copy()
                repl_circ.append_gate(op.gate, mini_loc, op.params)
                repl_circ.append_circuit(right_circ, mini_loc)
                # Replace the operation in the circuit
                circuit.replace_with_circuit(CircuitPoint(cycle, op.location[0]),
                                    repl_circ, as_circuit_gate=True)
                        # Unfold the circuit
        circuit.unfold_all()
        # Combine all RX, RY and RZ gates
        combine_rotations(circuit)
        return circuit

    async def run(self, circ: Circuit, data: PassData) -> None:
        circs = [circ] * self.num_twirls
        rc_circs = await get_runtime().map(PauliTwirlPass.create_rc_circ, circs, twirl_set=self.twirl_set)
        data['twirled_circuits'] = rc_circs