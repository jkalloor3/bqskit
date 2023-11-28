"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator

_logger = logging.getLogger(__name__)


class CreateEnsemblePass(BasePass):
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

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        init_approx_circuits: list[tuple[Circuit, float]] = data["scan_sols"]

        init_circs = [x[0] for x in init_approx_circuits]

        all_circs = [x for x in init_circs]
        all_dists = set([x[1] for x in init_approx_circuits])

        print(all_dists)

        target = self.get_target(circuit, data)

        # From our circuits, we are going to try to instantiate a bunch of circuits for our ensemble
        seed = 0
        fails = 0

        print("Init Circs!", len(init_circs))
        while len(all_circs) < self.num_circs:
            # Pick all circuits

            # working_copy.instantiate(target, **instantiate_options)
            instantiated_circuits: list[Circuit] = await get_runtime().map(
                        Circuit.instantiate,
                        init_circs,
                        target=target,
                        seed=seed,
                        **self.instantiate_options
            )

            added_circ = False
            for circ in instantiated_circuits:
                cost = self.cost(circ, target)
                if cost < self.success_threshold:
                    # if cost not in all_dists:
                    #     # Cheating way to check if unitary is new
                    all_circs.append(circ)
                    all_dists.add(cost)
                    added_circ = True
                    print(len(all_circs))
                
            if not added_circ:
                fails += 1

            if fails > 50:
                print("Failed to generate all circs!")
                data["ensemble"] = all_circs
                return circuit

            seed += 1

        data["ensemble"] = all_circs
        # data["ensemble_dists"] = all_dists
        return circuit

        
