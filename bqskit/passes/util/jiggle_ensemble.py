"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HSCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import U3Gate, CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain
import time
import pickle 

_logger = logging.getLogger(__name__)


class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HSCostGenerator(),
                 use_ensemble: bool = True,
                 use_calculated_error: bool = True) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.cost = cost
        # self.instantiate_options: dict[str, Any] = {
        #     'dist_tol': self.success_threshold,
        #     'min_iters': 100,
        #     'cost_fn_gen': self.cost,
        # }
        self.use_ensemble = use_ensemble
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        self.use_calculated_error = use_calculated_error

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy

    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int):
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

                try:
                    jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
                except Exception as e:
                    print("Extra Diff", extra_diff, flush=True)
                    print("Num Params to Jiggle", num_params_to_jiggle, flush=True)
                    print("Len Params", len(params), flush=True)
                    print(self.success_threshold, flush=True)
                    print("Dist", dist, flush=True)
                    _logger.error(e)
                # print("jiggle_amounts", jiggle_amounts, flush=True)
                next_params = best_params.copy()
                next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
                circ_cost = cost_fn.get_cost(next_params)
                trial_costs.append(circ_cost)
                try:
                    a = circ_cost < self.success_threshold
                except Exception as e:
                    print("Circ Cost", circ_cost, flush=True)
                    print("Success Threshold", self.success_threshold, flush=True)
                    _logger.error(e)
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
            circs.append(circ_copy)
        JiggleEnsemblePass.num_jiggles += 1
        # print(f"Single Jiggle ({JiggleEnsemblePass.num_jiggles}) time {time.time() - start}", flush=True)
        return circs


    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int):
        circ, dist = circ_dist
        
        params = circ.params

        if len(params) < 10:
            # print(circ.gate_counts)
            # print("WTF NO PARAMS")
            for i in range(circ.num_qudits):
                # Add a layer of U3 gates
                circ.append_gate(U3Gate(), (i,), [0, 0, 0])
            params = circ.params
        all_circs = []

        # print(f"Awaiting All {num_circs // 50} Jiggles")
        all_circs: list[Circuit] = await get_runtime().map(self.single_jiggle, [params] * ceil(num_circs / 20), circ=circ, dist=dist, target=target, num=20)
        
        # print("Unitaries: ", [circ.get_unitary() for circ in all_circs[0]])
        
        all_circs = list(chain.from_iterable(all_circs))
        # print("Done All Jiggles")
        return all_circs


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Collected one solution from synthesis
        print("Starting JIGGLE ENSEMBLE", flush=True)
        
        if "finished_jiggle" in data:
            print("Already Jiggled", flush=True)
            return


        params = circuit.params

        print("Number of ensembles", len(data["ensemble"]), flush=True)

        if self.use_calculated_error:
            print("OLD", self.success_threshold)
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]
            print("NEW", self.success_threshold)

        data["finished_jiggle"] = False

        ensemble = []
        futures = []

        for scan_sols in data["ensemble"]:
            all_circs = []
            print("Number of SCAN SOLS", len(scan_sols))

            # For each params come up with nth root of num_circs number of extra params
            if self.use_ensemble:
                circuits = [psol[0] for psol in scan_sols]
                dists = [psol[1] for psol in scan_sols]
            else:
                circuits = [circuit]
                dists = [self.cost.calc_cost(circuit, data.target)]

            circ_dists = zip(circuits, dists)
            
            # for i, dist in enumerate(dists):
            #     print(dist, end=",")
            #     if dist == np.NaN:
            #         print("DIST IS NAN")
            #         print("Circuit", circuits[i])
            #         print("Target", data.target)
            #         _logger.error("DIST IS NAN")
            #         # dist = self.cost.calc_cost(circuit, data.target)
            #         # dists[i] = dist

            all_circs_future = get_runtime().map(self.jiggle_circ,
                                                circ_dists,
                                                target=data.target,
                                                num_circs = ceil(self.num_circs / len(circuits)))
            futures.append(all_circs_future)


        all_circs = [await future for future in futures]

        ensemble = [list(chain.from_iterable(all_circ)) for all_circ in all_circs]  
        print("Number of Circs post Jiggle", [len(ens) for ens in ensemble])
        # ensemble.append(all_circs)

        data["ensemble"] = ensemble
        
        if "checkpoint_dir" in data:
            data["finished_jiggle"] = True
            checkpoint_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(checkpoint_data_file, "wb"))


        return

        
