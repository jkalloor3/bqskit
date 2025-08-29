"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ToU3Pass
from bqskit.ir.circuit import Circuit, CircuitPoint
from bqskit.runtime import get_runtime
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import *
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

import os

_logger = logging.getLogger(__name__)

class DiversifyEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(
            self, 
            success_threshold = 1e-4, 
            cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
            ignore_oneq_cost: bool = False
        ) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.cost = cost
        self.ignore_oneq_cost = ignore_oneq_cost

    def get_perturbations(
            self, 
            epsilon: float, 
            ens_size: int
        ) -> list[UnitaryMatrix]:
        perturbations = []
        pauli_strings = ["X", "Y", "Z"]
        paulis = [PauliMatrices.from_string(pauli) for 
                  pauli in pauli_strings]
        
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
        
        return perturbations

    def get_ham_perturbations(
            self, 
            u3_utry: UnitaryMatrix, 
            dist: float, 
            num_options: int
        ) -> list[list[float]]:
        perturbations = self.get_perturbations(dist, num_options)
        final_matrices = [u3_utry @ p for p in perturbations]
        final_params = [U3Gate().calc_params(mat) for mat in final_matrices]
        return final_params
    
    def single_jiggle_ham(
            self,
            circ: Circuit,
            initial_dist: float,
            num: int,
            success_threshold: float
    ) -> np.ndarray[float]:
        '''
        Apply Hamiltonian perturbations to the U3 gates in the circuit.

        Input:
        circ_str

        circ_str is the string representation of the circuit

        We require the circ_num to create a local cache for Gridsynth Strings

        target:
        target is the target unitary matrix

        num: number of circuits to generate

        success_threshold: success threshold for the perturbation
        

        Returns:
        A numpy array of size (num, num_params) where num_params is the number 
        of parameters in the circuit
        '''
        # For each U3 gate, calculate do a Hamiltonian perturbation
        num_u3s = circ.count(U3Gate())
        if num_u3s == 0:
            empty_params = [[]]
            return empty_params

        # For each u3, come up with 16 param perturbations (jiggle, anti-jiggle
        # pairs)
        num_options = 16
        # Map U3 params to perturbed params
        u3_param_options: dict[tuple[float, float, float], 
                               list[list[float]]] = {}

        # Perturb the solution with the remaining error budget
        perturb_dist = success_threshold - initial_dist
        # Perturbation distance per U3
        perturb_dist = perturb_dist / (num_u3s + 1)

        for op in circ.operations():
            if isinstance(op.gate, U3Gate):
                key = tuple(op.params)
                if key not in u3_param_options:
                    u3_param_options[key] = self.get_ham_perturbations(
                        op.get_unitary(), perturb_dist, num_options)

        final_params = []
        new_circ = circ.copy()
        for _ in range(num):
            new_circ.set_params(circ.params)
            u3_ind = 0
            rand_inds = np.random.choice(num_options, num_u3s, replace=True)
            # Use jiggle
            for op in new_circ.operations():
                if isinstance(op.gate, U3Gate):
                    key = tuple(op.params)
                    op.params = u3_param_options[key][rand_inds[u3_ind] * 2]
                    u3_ind += 1

            full_params_1 = new_circ.params
            u3_ind = 0
            new_circ.set_params(circ.params)
            # Use corresponding anti-jiggle
            for op in new_circ.operations():
                if isinstance(op.gate, U3Gate):
                    key = tuple(op.params)
                    op.params = u3_param_options[key][rand_inds[u3_ind] * 2 + 1]
                    u3_ind += 1
            full_params_2 = new_circ.params
            final_params.append(full_params_1)
            final_params.append(full_params_2)

        return np.vstack(final_params)

    # def add_u3s(
    #         self, 
    #         circ: Circuit
    #     ) -> Circuit:
    #     '''
    #     Add U3 gates to the circuit.
    #     '''
    #     flooded_cnot = Circuit(2)
    #     flooded_cnot.append_gate(CNOTGate(), (0, 1))
    #     flooded_cnot.append_gate(U3Gate(), (0,), [0, 0, 0])
    #     flooded_cnot.append_gate(U3Gate(), (1,), [0, 0, 0])
    #     init_u3_circ = Circuit(circ.num_qudits)
    #     for i in range(circ.num_qudits):
    #         init_u3_circ.append_gate(U3Gate(), (i,), [0, 0, 0])
    #     circ.insert_circuit(0, init_u3_circ, tuple(range(circ.num_qudits)))
    #     for cycle, op in circ.operations_with_cycles():
    #         if isinstance(op.gate, CNOTGate):
    #             op_pt = CircuitPoint(cycle, op.location[0])
    #             circ.replace_with_circuit(op_pt, flooded_cnot.copy(), 
    #                                       as_circuit_gate=True)
    #     circ.unfold_all()
    #     ToU3Pass.run_group_circ(circ)
    #     return circ

    async def run(
            self, 
            circuit: Circuit, 
            data: PassData
        ) -> None:
        # This pass should only be called if we are in ensemble mode
        assert "run_ensemble" in data and data["run_ensemble"] == True

        # Get ensemble circuits
        circuits: list[Circuit] = data.get("ensemble_circuits", [])

        # # Add extra U3s if we aren't optimizing the number of 1Q gates
        # if self.ignore_oneq_cost:
        #     circuits = [self.add_u3s(c) for c in circuits]
        #     data["ensemble_circuits"] = circuits

        # For each circuit, generate a list of parameters
        all_params = await get_runtime().map(self.single_jiggle_ham, 
                                circuits, 
                                target=data.target,
                                num = ceil(6000 / len(circuits)),
                                success_threshold=self.success_threshold)
        
        # Store the parameters
        data["ensemble_params"] = all_params

        # Assume uniform probability (TODO: will update for FT compilation)
        all_probs = [np.ones((p.shape[0], )) / p.shape[0] for p in all_params]
        data["ensemble_probs"] = all_probs