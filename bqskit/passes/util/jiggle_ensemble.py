"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import U3Gate, CNOTGate, TGate, TdgGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain
import pickle 
import csv

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
                 checkpoint_extra_str: str = "") -> None:
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

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy

    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int) -> list[tuple[Circuit, float]]:
        # print("Starting Jiggle", flush=True)
        # start = time.time()
        circs = []
        for _ in range(num):
            cost_fn = self.cost.gen_cost(circ.copy(), target)
            trials = 0
            best_params = np.array(params.copy(), dtype=np.float64)
            extra_diff = max(self.success_threshold - dist, self.success_threshold / len(params))
            while trials < 20:
                trial_costs = []
                if len(params) < 10:
                    num_params_to_jiggle = ceil(len(params) / 2)
                else:
                    num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + ceil(len(params) / 10)
                # num_params_to_jiggle = len(params)
                params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
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
            # circ_cost = frob_cost.calc_cost(circ_copy, target)
            # circ_cost = normalized_frob_cost(circ_copy.get_unitary(), target)
            circs.append((circ_copy, circ_cost))
        JiggleEnsemblePass.num_jiggles += 1
        return circs

    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int) -> list[tuple[Circuit, float]]:
        circ, dist = circ_dist
        
        params = circ.params

        while len(params) < 30:
            # Insert random U3 Identities
            random_cycle = np.random.randint(0, circ.num_cycles)
            random_loc = np.random.randint(0, circ.num_qudits)
            circ.insert_gate(random_cycle, U3Gate(), random_loc, [0, 0, 0])
            params = circ.params
        all_circs = []

        if num_circs > 60:
            # print(f"Awaiting All {num_circs // 50} Jiggles")
            # print(f"Launching {ceil(num_circs / 20)} Tasks", flush=True)
            all_circs: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.single_jiggle, [params] * ceil(num_circs / 20), circ=circ, dist=dist, target=target, num=20)
            # print(f"Finished {ceil(num_circs / 20)} Tasks", flush=True)
            all_circs: list[tuple[Circuit, float]] = list(chain.from_iterable(all_circs))
        else:
            all_circs: list[tuple[Circuit, float]] = await self.single_jiggle(params, circ, dist, target, num_circs)
        
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
        
        if checkpoint_str in data and data[checkpoint_str]:
            print("Already Jiggled", flush=True)
            return

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

        
