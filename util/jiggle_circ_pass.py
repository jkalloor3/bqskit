"""This module implements the ToU3Pass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator, HilbertSchmidtResidualsGenerator, FrobeniusCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.qis import UnitaryMatrix
import numpy as np


cost_1 = FrobeniusCostGenerator()

cost = HilbertSchmidtCostGenerator()


class  GetErrorsPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        unfolded_circ = circuit.copy()
        unfolded_circ.unfold_all()
        full_dist = cost.calc_cost(unfolded_circ, data.target)

        full_dist_2 = cost_1.calc_cost(unfolded_circ, data.target)

        print(full_dist, full_dist_2, flush=True)

        assert np.allclose(full_dist, np.sqrt(full_dist_2))
        # full_dist = dist_cost(unfolded_circ.get_unitary(), data.target)

        block_data = data[ForEachBlockPass.key][0]
        targets: list[UnitaryMatrix] = []

        dists = []

        for i, (cycle, op) in enumerate(circuit.operations_with_cycles()):
            block = block_data[i]
            targets.append(block["target"])
            subcircuit = op.gate._circuit.copy()
            subcircuit.set_params(op.params) 
            dists.append(cost.calc_cost(subcircuit, block["target"]))
            # dists.append(dist_cost(subcircuit.get_unitary(), block["target"]))

        print("Distances", dists, flush=True)
        print("Upperbound", sum(dists), flush=True)
        print("Full Distance", full_dist, flush=True)


class  JiggleCircPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        params = circuit.params
        if len(params) > 0:
            extra_diff = 0.01
            num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + len(params) // 10
            # num_params_to_jiggle = len(params)
            params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
            jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
            # print("jiggle_amounts", jiggle_amounts, flush=True)
            params[params_to_jiggle] = params[params_to_jiggle] + jiggle_amounts
            circuit.set_params(params)
            final_cost = cost.calc_cost(circuit, data.target)
            # if final_cost > 0:
            #     print(cost.calc_cost(circuit, data.target), flush=True)
        return

        
