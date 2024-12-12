"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import U3Gate, GlobalPhaseGate
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain

from util.distance import normalized_gp_frob_cost

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)

class  JiggleScansPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(self, success_threshold = 1e-4, 
                 cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
                 use_calculated_error: bool = True,
                 use_ham_perturbation: bool = True) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.cost = cost
        self.use_calculated_error = use_calculated_error
        self.use_ham_perturbation = use_ham_perturbation

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy

    def get_perturbations(num_qudits: int, epsilon: float, ens_size: int) -> list[UnitaryMatrix]:
        perturbations = []
        pauli_strings = ["I", "X", "Y", "Z"]
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


    def get_ham_perturbations(self, u3_utry: UnitaryMatrix, dist: float, num_options: int) -> list[list[float]]:
        perturbations = JiggleScansPass.get_perturbations(1, dist, num_options)
        final_matrices = [u3_utry @ perturbation for perturbation in perturbations]
        final_params = [U3Gate().calc_params(mat) for mat in final_matrices]
        return final_params


    async def single_jiggle_ham(self, circ: Circuit, dist: float, target: UnitaryMatrix, num: int, success_threshold: float) -> list[tuple[Circuit, float]]:
        # circ is guaranteed to only have u3s and cnots
        # For each U3 gate, calculate do a Hamiltonian perturbation
        print("Circ Gate Counts: ", circ.gate_counts, flush=True)
        num_u3s = circ.count(U3Gate())
        # For each u3, come up with 32 param perturbations
        num_options = 16
        u3_param_options: list[list[list[float]]] = []
        perturb_dist = (success_threshold - dist) / (num_u3s * 2)
        for op in circ.operations():
            if isinstance(op.gate, U3Gate):
                cur_u3_utry = op.get_unitary()
                # Note that it alternates between positive and negative perturbations
                u3_param_options.append(self.get_ham_perturbations(cur_u3_utry, perturb_dist, num_options * 2))
        
        # Now randomly pick num combinations of these options
        final_circs = []
        for _ in range(num // 2):
            rand_inds = np.random.choice(num_options, num_u3s, replace=True)
            # Positive perturbation
            full_params_1: list[list[float]] = [u3_param_options[i][ind * 2] for i, ind in enumerate(rand_inds)]
            # Negative perturbation
            full_params_2: list[list[float]] = [u3_param_options[i][ind * 2 + 1] for i, ind in enumerate(rand_inds)]
            full_params_1 = list(chain.from_iterable(full_params_1))
            full_params_2 = list(chain.from_iterable(full_params_2))
            new_circ_1 = circ.copy()
            new_circ_1.set_params(full_params_1)
            new_circ_2 = circ.copy()
            new_circ_2.set_params(full_params_2)
            dist_1 = normalized_gp_frob_cost(new_circ_1.get_unitary(), target)
            dist_2 = normalized_gp_frob_cost(new_circ_2.get_unitary(), target)
            print("Original Dist: ", dist, "New Dist 1: ", dist_1, "New Dist 2: ", dist_2, "Success Threshold: ", success_threshold, flush=True)
            final_circs.append((new_circ_1, dist_1))
            final_circs.append((new_circ_2, dist_2))

        return final_circs


    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int, success_threshold: float) -> list[tuple[Circuit, float]]:
        circs = []
        for _ in range(num):
            cost_fn = self.cost.gen_cost(circ.copy(), target)
            trials = 0
            best_params = np.array(params.copy(), dtype=np.float64)
            extra_diff = max(success_threshold - dist, success_threshold / len(params))
            while trials < 7:
                trial_costs = []
                if len(params) < 10:
                    num_params_to_jiggle = ceil(len(params) / 2)
                else:
                    num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + ceil(len(params) / 10)
                    num_params_to_jiggle = min(num_params_to_jiggle, len(params))
                # num_params_to_jiggle = len(params)
                params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
                jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
                # print("jiggle_amounts", jiggle_amounts, flush=True)
                next_params = best_params.copy()
                next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
                circ_cost = cost_fn.get_cost(next_params)
                # circ_cost = normalized_frob_cost(circ.get_unitary(), target)
                trial_costs.append(circ_cost)
                if (circ_cost < success_threshold):
                    extra_diff = extra_diff * 1.5
                    best_params = next_params
                    # Randomly choose to finish early
                    if np.random.uniform() < 0.2 and trials > 4:
                        break
                else:
                    extra_diff = extra_diff / 10
                trials += 1
            
            circ_copy = circ.copy()
            circ_copy.set_params(best_params)
            circ_cost = cost_fn.get_cost(best_params)
            circs.append((circ_copy, circ_cost))
        return circs

    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int, success_threshold: float) -> list[tuple[Circuit, float]]:
        circ, dist = circ_dist
        params = circ.params
        if self.use_ham_perturbation:
            all_circs: list[tuple[Circuit, float]] = await self.single_jiggle_ham(circ, dist, target, num_circs, success_threshold=success_threshold)
        else:
            all_circs: list[tuple[Circuit, float]] = await self.single_jiggle(params, circ, dist, target, num_circs, success_threshold=success_threshold)
        return all_circs

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        self.finished_pass_str = f"finished_jiggle_scans"

        if self.finished_pass_str in data:
            return

        # Collected one solution from synthesis
        print("Starting JIGGLE Scan Sols", flush=True)

        ensemble = data["scan_sols"]

        if self.use_calculated_error:
            success_threshold = self.success_threshold * data.get("error_percentage_aloocated", 1)
        else:
            success_threshold = self.success_threshold

        print("Number of Circs pre Jiggle", len(ensemble), flush=True)

        if len(ensemble) > 40:
            data[self.finished_pass_str] = True
            return
    
        scan_sols_list: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.jiggle_circ, 
                                                ensemble, 
                                                target=data.target, 
                                                num_circs=10, 
                                                success_threshold=success_threshold)
        
        if len(ensemble) <= 4:
            new_scan_sols = list(chain.from_iterable(scan_sols_list))
        else:
            new_scan_sols = []
            for circs in scan_sols_list:
                # Randomly pick 5 from each set
                num_inds = min(5, len(circs))
                new_scan_sols_inds = np.random.choice(len(circs), num_inds, replace=False)
                new_scan_sols.extend([circs[i] for i in new_scan_sols_inds])
        print("Number of Circs post Jiggle", len(new_scan_sols), flush=True)
        # Add default circuit at end
        new_scan_sols.append((circuit.copy(), 0))
        data["scan_sols"] = new_scan_sols

        data[self.finished_pass_str] = True

        
