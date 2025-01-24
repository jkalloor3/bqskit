"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ToU3Pass
from bqskit.ir.circuit import Circuit, CircuitPoint
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.lang import get_language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import U3Gate, CNOTGate, GlobalPhaseGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
import itertools

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

import os
from .common import store_jiggled_ensemble, load_jiggled_ensemble, create_single_jiggled_ensemble, stack_padding

_logger = logging.getLogger(__name__)

frob_cost = GPNormalizedFrobeniusCostGenerator()

lang = get_language("qasm")

class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = GPNormalizedFrobeniusCostGenerator(),
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

    async def single_jiggle_ham(self, circ: Circuit, dist: float, num: int, target: UnitaryMatrix) -> np.ndarray[float]:
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
        final_params = []
        # dists = []
        # frob_dists = []
        for _ in range(num):
            rand_inds = np.random.choice(num_options, num_u3s, replace=True)
            # Positive perturbation
            full_params_1: list[list[float]] = [u3_param_options[i][ind * 2] for i, ind in enumerate(rand_inds)]
            # Negative perturbation
            full_params_2: list[list[float]] = [u3_param_options[i][ind * 2 + 1] for i, ind in enumerate(rand_inds)]
            # full_params_1 = list(itertools.chain.from_iterable(full_params_1))
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
            
            
            dist_1 = self.cost.calc_cost(new_circ_1, target)
            dist_2 = self.cost.calc_cost(new_circ_2, target)


            dist_1 = self.cost.calc_cost(new_circ_1, target)
            dist_2 = self.cost.calc_cost(new_circ_2, target)

            # print("Orig Dist: ", dist, " New Cost: ", dist_1, " Threshold: ", self.success_threshold, flush=True)

            full_params_1 = new_circ_1.params
            full_params_2 = new_circ_2.params

            dist_1 = self.cost.calc_cost(new_circ_1, target)
            dist_2 = self.cost.calc_cost(new_circ_2, target)
            full_params_1 = self.jiggle_params(full_params_1, new_circ_1, dist_1, target)
            full_params_2= self.jiggle_params(full_params_2, new_circ_2, dist_2, target)
            final_params.append(full_params_1)
            final_params.append(full_params_2)
            new_circ_1.set_params(full_params_1)
            new_circ_2.set_params(full_params_2)
            dist_1 = self.cost.calc_cost(new_circ_1, target)
            dist_2 = self.cost.calc_cost(new_circ_2, target)
            # frob_dists.append(frob_cost.calc_cost(new_circ_1, target))
            # frob_dists.append(frob_cost.calc_cost(new_circ_2, target))
            # dists.append(dist_1)
            # dists.append(dist_2)

        # print("Avg. Dist Post Jiggle: ", np.mean(dists), flush=True)
        # print("Avg. Frob Dist Post Jiggle: ", np.mean(frob_dists), flush=True)

        return np.vstack(final_params)


    def jiggle_params(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix) -> np.ndarray[float]:
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

            return best_params

    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int) -> np.ndarray[float]:
        # print("Starting Jiggle", flush=True)
        # start = time.time()
        params = []
        for _ in range(num):
            p = self.jiggle_params(params, circ, dist, target)
            if p:
                params.append(p)
        return np.vstack(params)

    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int) -> tuple[Circuit, np.ndarray[float]]:
        circ, dist = circ_dist

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

        # This ensures that qasm will be decoded the same way later
        circ =  lang.decode(lang.encode(circ))
        num_tasks = ceil(num_circs / 40)
        circs_per_task = ceil(num_circs / num_tasks)
        if self.do_u3_perturbation:
            in_circs = [circ.copy() for _ in range(num_tasks)]
            params: list[np.ndarray[float]] = await get_runtime().map(self.single_jiggle_ham, in_circs, dist=dist, num=circs_per_task, target=target)
        else:
            orig_params = circ.params
            params: list[np.ndarray[float]] = await get_runtime().map(self.single_jiggle, [orig_params] * num_tasks, circ=circ, dist=dist, target=target, num=circs_per_task)
            # print(f"Finished {ceil(num_circs / 20)} Tasks", flush=True)
        
        all_params = np.vstack(params)
        
        return circ, all_params

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
            
            return list(itertools.chain.from_iterable(circ_dists))


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Collected one solution from synthesis
        print("Starting JIGGLE ENSEMBLE", flush=True)

        checkpoint_dir = data["checkpoint_dir"]
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")

        if self.use_calculated_error:
            # print("OLD", self.success_threshold)
            self.success_threshold = self.success_threshold * data.get("error_percentage_allocated", 1)
            # print("NEW", self.success_threshold)
        
        print("Success Threshold", self.success_threshold, flush=True)
        
        ens_ind = 0
        ens_file = ensemble_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
        if os.path.exists(jiggle_file):
            # Load the ensemble from the checkpoint
            ensembles = []
            while os.path.exists(jiggle_file):
                jiggled_circs = load_jiggled_ensemble(ens_file, jiggle_file)
                # dists = [self.cost.calc_cost(circ, data.target) for circ in jiggled_circs]
                jiggled_circs = [jiggled_circs[i] for i, dist in enumerate(dists) if dist < self.success_threshold]
                ensembles.append(jiggled_circs)
                ens_ind += 1
                ens_file = ensemble_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
                jiggle_file = jiggle_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
                # print("Avg Dist Post Jiggle Load: ", np.mean(dists), flush=True)
                # print("Num Circs: ", len(jiggled_circs), flush=True)
            print("Ensemble Size", [len(ens) for ens in ensembles], flush=True)
            print("Finished Jiggle Ensemble Pass", flush=True)
            data["ensemble"] = ensembles
            return

        print("Number of ensembles", len(data["ensemble"]), flush=True)

        ensemble = []
        # all_circ_params = []

        for ens_ind, scan_sols in enumerate(data["ensemble"]):
            print("Number of SCAN SOLS", len(scan_sols), flush=True)
            # For each params come up with nth root of num_circs number of extra params
            if self.use_ensemble:
                circuits = scan_sols
                dists = [self.cost.calc_cost(circ, data.target) for circ in circuits]
            else:
                circuits = [circuit]
                dists = [self.cost.calc_cost(circuit, data.target)]

            print("Avg Dist", np.mean(dists), flush=True)

            if len(circuits) == 0:
                continue
            circ_dists = list(zip(circuits, dists))

            circ_params: list[tuple[Circuit, np.ndarray]] = await get_runtime().map(self.jiggle_circ, 
                                                         circ_dists, 
                                                         target=data.target, 
                                                         num_circs = ceil(self.num_circs / len(circuits))
                                                        )
            # all_circ_params.append(circ_params)
            jiggled_circ_dists: list[list[Circuit]] = await get_runtime().map(create_single_jiggled_ensemble, circ_params)
            jiggled_circs: list[Circuit] = list(itertools.chain.from_iterable(jiggled_circ_dists))
            print("Num Circs: ", len(jiggled_circs), flush=True)
            dists = [self.cost.calc_cost(circ, data.target) for circ in jiggled_circs[:len(jiggled_circs):80]]
            print("Dists Post Jiggle Combo: ", dists, flush=True)
            ens_file = ensemble_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            jiggle_file = jiggle_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            store_jiggled_ensemble(circ_params, ens_file, jiggle_file)
            ensemble.append(jiggled_circs)

        print("Number of Circs post Jiggle", [len(ens) for ens in ensemble], flush=True)
        data["ensemble"] = ensemble
        return

        
