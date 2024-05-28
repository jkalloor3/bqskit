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
from math import ceil

_logger = logging.getLogger(__name__)


class JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
                 use_ensemble: bool = True) -> None:
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

        all_circs = []

        # For each params come up with nth root of num_circs number of extra params
        if self.use_ensemble:
            circuits = data["ensemble"]
        else:
            circuits = [circuit]
        for circ in circuits:
            params = circ.params
            print(f"Num Params: {len(params)}")

            extra_diff = self.success_threshold

            for _ in range(ceil(self.num_circs / len(circuits))):
                next_params = np.array(params.copy())
                num_params_to_jiggle = int(np.random.uniform() * min(len(params) / 2, 50))
                params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
                jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
                next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
                circ_copy = circ.copy()
                circ_copy.set_params(next_params)
                all_circs.append(circ_copy)
                # all_combos.append(next_params)

        data["ensemble"] = all_circs

        print("FINISHED!")
        return

        
