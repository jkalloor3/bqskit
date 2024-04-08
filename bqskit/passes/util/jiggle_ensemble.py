"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np

_logger = logging.getLogger(__name__)


class JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HilbertSchmidtCostGenerator(),) -> None:
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

        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Collected one solution from synthesis
        print("Starting JIGGLE ENSEMBLE")

        params = circuit.params

        # For each params come up with nth root of num_circs number of extra params


        print(f"Num Params: {len(params)}")

        extra_diff = self.success_threshold

        all_combos = []
        for i in range(self.num_circs):
            next_params = np.array(params.copy())
            num_params_to_jiggle = int(np.random.uniform() * min(len(params) / 2, 50))
            params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
            jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
            next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
            all_combos.append(next_params)



        # for i,param in enumerate(params):
        #     if total_combos < self.num_circs:
        #         num_extra = num_extra_params
        #     else:
        #         num_extra = 0
        #     extra_params = np.random.uniform(param - extra_diff, param + extra_diff, num_extra)
        #     all_params[i].extend(extra_params)
        #     total_combos *= (len(extra_params) + 1)
        #     print(total_combos)

        # Now create all circuits
        # all_combos = [np.zeros(len(params)) for _ in range(total_combos)]
        # i = 0
        # # np.set_printoptions(linewidth=np.inf, threshold=np.inf, precision=6)

        # for i in range(total_combos):
        #     total = i
        #     for j, params_list in enumerate(all_params):
        #         ind = total % len(params_list)
        #         total = total // len(params_list)
        #         all_combos[i][j] = params_list[ind]

        #     # print(all_combos[i])
        # print(f"Generated all {len(all_combos)} param lists")
        
        # all_circs = await get_runtime().map(
        #     JiggleEnsemblePass.get_circ,
        #     all_combos,
        #     circuit=circuit
        # )
        # all_circs = [None for _ in all_combos]
        # for i, l_params in enumerate(all_combos):
        #     unit = circuit.get_unitary(l_params)
        #     # circ_copy.set_params(l_params)
        #     if i % 1000 == 0:
        #         print("1000 done")
        #     all_circs[i] = unit

        data["ensemble_params"] = all_combos

        print("FINISHED!")
        return

        
