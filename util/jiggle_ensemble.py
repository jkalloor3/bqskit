"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ToU3Pass
from bqskit.ir.circuit import Circuit, CircuitPoint
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import U3Gate, CNOTGate, GlobalPhaseGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain
import pickle 

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)

class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = FrobeniusCostGenerator(),
                 use_ensemble: bool = True,
                 use_calculated_error: bool = True,
                 count_t: bool = False,
                 checkpoint_extra_str: str = "",
                 jiggle_skew: int = 0,
                 do_u3_perturbation: bool = True,
                 flood_circ: bool = False) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.cost = cost
        self.use_ensemble = use_ensemble
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        self.use_calculated_error = use_calculated_error
        self.count_t = count_t
        self.checkpoint_extra_str = checkpoint_extra_str
        self.jiggle_skew = jiggle_skew
        self.do_u3_perturbation = do_u3_perturbation
        self.flood_circ = flood_circ

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy

    def get_perturbations(num_qudits: int, epsilon: float, ens_size: int) -> list[UnitaryMatrix]:
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


    def get_ham_perturbations(self, u3_utry: UnitaryMatrix, dist: float, num_options: int) -> list[list[float]]:
        perturbations = JiggleEnsemblePass.get_perturbations(1, dist, num_options)
        final_matrices = [u3_utry @ perturbation for perturbation in perturbations]
        final_params = [U3Gate().calc_params(mat) for mat in final_matrices]
        return final_params

    async def single_jiggle_ham(self, circ: Circuit, dist: float, num: int, target: UnitaryMatrix) -> list[tuple[Circuit, float]]:
        # For each U3 gate, calculate do a Hamiltonian perturbation
        num_u3s = circ.count(U3Gate())
        # For each u3, come up with 16 param perturbations
        num_options = 16
        u3_param_options: list[list[list[float]]] = []
        perturb_dist = (self.success_threshold - dist) / (num_u3s)
        for op in circ.operations():
            if isinstance(op.gate, U3Gate):
                cur_u3_utry = op.get_unitary()
                u3_param_options.append(self.get_ham_perturbations(cur_u3_utry, perturb_dist, num_options * 2))
        
        # Now randomly pick num combinations of these options
        final_circs = []
        for _ in range(num):
            rand_inds = np.random.choice(num_options, num_u3s, replace=True)
            # Positive perturbation
            full_params_1: list[list[float]] = [u3_param_options[i][ind * 2] for i, ind in enumerate(rand_inds)]
            # Negative perturbation
            full_params_2: list[list[float]] = [u3_param_options[i][ind * 2 + 1] for i, ind in enumerate(rand_inds)]
            # full_params_1 = list(chain.from_iterable(full_params_1))
            new_circ_1 = circ.copy()
            ind = 0
            for op in new_circ_1.operations():
                if isinstance(op.gate, U3Gate):
                    op.params = full_params_1[ind]
                    ind += 1
            
            new_circ_2 = circ.copy()
            ind = 0
            for op in new_circ_2.operations():
                if isinstance(op.gate, U3Gate):
                    op.params = full_params_2[ind]
                    ind += 1

            global_phase_correction = target.get_target_correction_factor(new_circ_1.get_unitary())
            new_circ_1.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            global_phase_correction = target.get_target_correction_factor(new_circ_2.get_unitary())
            new_circ_2.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            
            dist_1 = self.cost.calc_cost(new_circ_1, target)
            dist_2 = self.cost.calc_cost(new_circ_2, target)

            # print("Orig Dist: ", dist, " New Cost: ", dist_1, " Threshold: ", self.success_threshold, flush=True)

            full_params_1 = new_circ_1.params
            full_params_2 = new_circ_2.params
            new_c_1 = self.jiggle_params(full_params_1, new_circ_1, dist_1, target)
            new_c_2 = self.jiggle_params(full_params_2, new_circ_2, dist_2, target)
            final_circs.append(new_c_1)
            final_circs.append(new_c_2)
        return final_circs


    def jiggle_params(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix) -> tuple[Circuit, float]:
            cost_fn = self.cost.gen_cost(circ.copy(), target)
            trials = 0
            best_params = np.array(params.copy(), dtype=np.float64)
            extra_diff = max(self.success_threshold - dist, self.success_threshold / len(params))
            while trials < 10:
                trial_costs = []
                if len(params) < 10:
                    num_params_to_jiggle = ceil(len(params) / 2)
                else:
                    num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + ceil(len(params) / 10)
                    # num_params_to_jiggle = min(num_params_to_jiggle, len(params))
                # num_params_to_jiggle = len(params)
                # Vary probability proportional to param location
                p = (np.arange(len(params)) + 1) ** self.jiggle_skew
                p = p / np.sum(p)
                params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False, p=p)
                jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
                # print("jiggle_amounts", jiggle_amounts, flush=True)
                next_params = best_params.copy()
                next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
                circ_cost = cost_fn.get_cost(next_params)
                # circ_cost = normalized_frob_cost(circ.get_unitary(), target)
                trial_costs.append(circ_cost)
                if (circ_cost < self.success_threshold):
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

            return circ_copy, circ_cost

    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int) -> list[tuple[Circuit, float]]:
        # print("Starting Jiggle", flush=True)
        # start = time.time()
        circs = []
        for _ in range(num):
            c = self.jiggle_params(params, circ, dist, target)
            if c:
                circs.append(c)
        JiggleEnsemblePass.num_jiggles += 1
        return circs

    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int) -> list[tuple[Circuit, float]]:
        circ, dist = circ_dist
        
        params = circ.params

        orig_gate_counts =  circ.gate_counts

        # assert than each circ has U3 gates after their CNOTs
        if self.flood_circ:
            flooded_cnot = Circuit(2)
            flooded_cnot.append_gate(CNOTGate(), (0, 1))
            flooded_cnot.append_gate(U3Gate(), (0,), [0, 0, 0])
            flooded_cnot.append_gate(U3Gate(), (1,), [0, 0, 0])
            init_u3_circ = Circuit(circ.num_qudits)
            for i in range(circ.num_qudits):
                init_u3_circ.append_gate(U3Gate(), (i,), [0, 0, 0])
            circ.insert_circuit(0, init_u3_circ, tuple(range(circ.num_qudits)))
            for cycle, op in circ.operations_with_cycles():
                if isinstance(op.gate, CNOTGate):
                    op_pt = CircuitPoint(cycle, op.location[0])
                    circ.replace_with_circuit(op_pt, flooded_cnot.copy(), as_circuit_gate=True)
            circ.unfold_all()
            ToU3Pass.run_group_circ(circ)
        
        all_circs = []

        # print(f"Awaiting All {num_circs // 50} Jiggles")
        # print(f"Launching {ceil(num_circs / 20)} Tasks", flush=True)
        num_tasks = ceil(num_circs / 40)
        circs_per_task = ceil(num_circs / num_tasks)
        if self.do_u3_perturbation:
            in_circs = [circ.copy() for _ in range(num_tasks)]
            all_circs: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.single_jiggle_ham, in_circs, dist=dist, num=circs_per_task, target=target)
        else:
            all_circs: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.single_jiggle, [params] * num_tasks, circ=circ, dist=dist, target=target, num=circs_per_task)
            # print(f"Finished {ceil(num_circs / 20)} Tasks", flush=True)
        
        all_circs: list[tuple[Circuit, float]] = list(chain.from_iterable(all_circs))
        
        return all_circs

    async def jiggle_ensemble(self, scan_sols: list[tuple[Circuit, float]], target: UnitaryMatrix) -> list[tuple[Circuit, float]]:
            print("Number of SCAN SOLS", len(scan_sols))
            # For each params come up with nth root of num_circs number of extra params
            circuits = [psol[0] for psol in scan_sols]
            dists = [psol[1] for psol in scan_sols]

            circ_dists = list(zip(circuits, dists))

            # print("Initial Distances before jiggle", dists, flush=True)
            circ_dists = await get_runtime().map(self.jiggle_circ,
                                                circ_dists,
                                                target=target,
                                                num_circs = ceil(self.num_circs / len(circuits)))
            
            return list(chain.from_iterable(circ_dists))


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Collected one solution from synthesis
        print("Starting JIGGLE ENSEMBLE", flush=True)

        checkpoint_str = JiggleEnsemblePass.finished_pass_str + self.checkpoint_extra_str
        
        # if checkpoint_str in data and data[checkpoint_str]:
        #     print("Already Jiggled", flush=True)
        #     return

        print("Number of ensembles", len(data["ensemble"]), flush=True)

        if self.use_calculated_error:
            # print("OLD", self.success_threshold)
            self.success_threshold = self.success_threshold * data.get("error_percentage_allocated", 1)
            # print("NEW", self.success_threshold)

        data[checkpoint_str] = False

        ensemble = []

        for scan_sols in data["ensemble"]:
            print("Number of SCAN SOLS", len(scan_sols), flush=True)
            # For each params come up with nth root of num_circs number of extra params
            if self.use_ensemble:
                circuits = [psol[0] for psol in scan_sols]
                dists = [psol[1] for psol in scan_sols]
            else:
                circuits = [circuit]
                dists = [self.cost.calc_cost(circuit, data.target)]


            if len(circuits) == 0:
                continue
            circ_dists = list(zip(circuits, dists))

            jiggled_circ_dists = await get_runtime().map(self.jiggle_circ, 
                                                         circ_dists, 
                                                         target=data.target, 
                                                         num_circs = ceil(self.num_circs / len(circuits))
                                                        )
            ensemble.append(list(chain.from_iterable(jiggled_circ_dists)))

        print("Number of Circs post Jiggle", [len(ens) for ens in ensemble], flush=True)

        data["ensemble"] = ensemble

        if "checkpoint_dir" in data:
            data[checkpoint_str] = True
            checkpoint_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(checkpoint_data_file, "wb"))
        return

        
