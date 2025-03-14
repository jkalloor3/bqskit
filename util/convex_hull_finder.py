import numpy as np
import qiskit.qasm2
import qiskit.quantum_info
import qiskit.synthesis
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.qis import UnitaryMatrix
from bqskit.passes import FullBlockZXZPass, QSDPass
from bqskit.ir.circuit import Circuit
from bqskit.qis.pauli import PauliMatrices
from bqskit.runtime import get_runtime
from bqskit.utils.math import dot_product
import scipy as sp
from util.common import store_jiggled_ensemble
import os
import qiskit
from bqskit.ext import qiskit_to_bqskit
from util.distance import normalized_gp_frob_cost

lang = OPENQASM2Language()

# class KAKDecompositionPass(BasePass):

#     async def run(self, circuit: Circuit, data: dict) -> None:
#         '''
#         Perform a KAK decomposition on all 2-qubit variable unitary gates
#         '''
#         # Get all 2-qubit variable unitary gates
#         unitaries, pts, locations = QSDPass.get_variable_unitary_pts(
#             circuit, 1,
#         )
#         print(circuit.gate_counts)
#         print("Num 2Q Unitaries: ", len(unitaries))
#         if len(unitaries) == 0:
#             return
#         # Perform the decomposition
#         i = 0
#         for unitary, pt in zip(unitaries, pts):
#             if unitary.num_qudits == 2:
#                 qcirc = qiskit.synthesis.two_qubit_cnot_decompose(unitary.numpy)
#                 qcirc = qiskit.transpile(qcirc, basis_gates=['u3', 'cx'], optimization_level=1)
#                 if i == 0:
#                     print(qcirc.count_ops())
#                     q_un = qiskit.quantum_info.Operator(qcirc).data
#                     print("Qiskit Unitary Distance: ", normalized_gp_frob_cost(q_un, unitary))
#                     qiskit.qasm2.dump(qcirc, "qiskit.qasm")
#                 # bcirc = qiskit_to_bqskit(qcirc)
#                 q_str = qiskit.qasm2.dumps(qcirc)
#                 bcirc = lang.decode(q_str)
#                 if i == 0:
#                     print(bcirc.gate_counts)
#                     print(normalized_gp_frob_cost(bcirc.get_unitary(), unitary))
#                 circuit.replace_with_circuit(
#                     pt, bcirc, as_circuit_gate=True
#                 )
#                 i += 1
#         circuit.unfold_all()
        
class ConvexHullFinderPass(BasePass):
    '''
    
    '''
    def __init__(self, num_unitaries: int, epsilon: float):
        self.num_unitaries = num_unitaries
        self.bzxz = FullBlockZXZPass(2, False, perform_extract=False)
        self.epsilon = epsilon


    def generate_convex_unitaries(self, unitary: UnitaryMatrix) -> list[Circuit]:
        perturbations = []
        num_qubits = unitary.num_qudits
        ens_size = self.num_unitaries
        pauli_strings = PauliMatrices.get_pauli_strings(num_qubits, 4)
        paulis = [PauliMatrices.from_string(pauli) for 
                  pauli in pauli_strings]
        epsilon = self.epsilon
        
        all_coeffs = []
        for _ in range(ens_size // 2):
            all_coeffs.append(np.random.rand(len(paulis)))
            
        for coeff in all_coeffs:
            coeff /= np.linalg.norm(coeff)
            coeff *= epsilon
            H_1 = dot_product(coeff, paulis)
            H_2 = dot_product(-1 * coeff, paulis)
            assert np.allclose(H_1, H_1.conj().T)
            assert np.allclose(H_2, H_2.conj().T)
            eiH_1 = sp.linalg.expm(1j * H_1)
            eiH_2 = sp.linalg.expm(1j * H_2)
            perturbations.append(UnitaryMatrix(eiH_1))
            perturbations.append(UnitaryMatrix(eiH_2))

        circs = [Circuit.from_unitary(unitary @ p) for p in perturbations] 
        return circs

    async def run(self, circuit: Circuit, data: dict) -> None:
        uns: list[Circuit] = self.generate_convex_unitaries(circuit.get_unitary())

        # targets = [c.get_unitary() for c in uns]

        # For each unitary, run Block ZXZ
        for c in uns:
            await self.bzxz.run(c, PassData(c))
        # await self.bzxz.run(uns[0], run_data)
        # await self.bzxz.run(uns[1], run_data)
        # dists = [normalized_gp_frob_cost(c.get_unitary(), targets[i]) for i, c in enumerate(uns)]
        # print("Dists: ", dists)
        # await KAKDecompositionPass().run(uns[0], run_data)

        # print(uns[0].gate_counts)

        # Get finals dists
        # dists = [normalized_gp_frob_cost(c.get_unitary(), targets[i]) for i, c in enumerate(uns)]
        # print("Dists: ", dists)

        # print gate counts
        
        # Store the circuits
        params = [np.array([c.params]) for c in uns]
        # print(params[0].shape)
        ensemble = [list(zip(uns, params))]

        # checkpoint_dir = data["checkpoint_dir"]
        # checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper_clifft/QITE_8_1_0_5.0"
        # ens_file = os.path.join(checkpoint_dir, "ensemble_0_.qasms")
        # jiggle_file = os.path.join(checkpoint_dir, "ensemble_0_jiggles_.npy")
        # store_jiggled_ensemble(ensemble, ens_file, jiggle_file)
        data["ensemble"] = ensemble