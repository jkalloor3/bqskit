"""This module implements the ToU3Pass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator, HilbertSchmidtResidualsGenerator, FrobeniusCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.qis import UnitaryMatrix
import numpy as np


cost1 = FrobeniusCostGenerator()

cost = HilbertSchmidtCostGenerator()

cost_3 = FrobeniusNoPhaseCostGenerator()

def cost_2(circuit: Circuit, target: UnitaryMatrix) -> float:
    a = circuit.get_unitary().numpy
    b = target.numpy
    prod = np.einsum("ij,ij->", a, b.conj())
    norm = np.abs(prod) ** 2
    try:
        return np.sqrt(1 - (norm / a.shape[0] / a.shape[0]))
    except:
        print("Error: ", 1 - (norm / a.shape[0] / a.shape[0]), flush=True)
        exit(1)

def cost_1(circuit: Circuit, target: UnitaryMatrix) -> float:
    unitary = circuit.get_unitary()
    fix = unitary.get_target_correction_factor(target) * np.eye(unitary.shape[0])
    return cost1.calc_cost(circuit, target @ fix)


class  GetErrorsPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        unfolded_circ = circuit.copy()
        unfolded_circ.unfold_all()
        full_dist = cost_1(unfolded_circ, data.target)

        full_dist_2 = cost_2(unfolded_circ, data.target)

        # full_dist_3 = cost_3.calc_cost(unfolded_circ, data.target)

        # print(full_dist, full_dist_2, full_dist_3, flush=True)

        # assert np.allclose(full_dist, np.sqrt(full_dist_2))
        # full_dist = dist_cost(unfolded_circ.get_unitary(), data.target)

        block_data = data[ForEachBlockPass.key][0]
        targets: list[UnitaryMatrix] = []

        dists_1 = []
        dists_2 = []

        for i, (cycle, op) in enumerate(circuit.operations_with_cycles()):
            block = block_data[i]
            targets.append(block["target"])
            subcircuit = op.gate._circuit.copy()
            subcircuit.set_params(op.params) 
            dists_1.append(cost_1(subcircuit, block["target"]))
            dists_2.append(cost_2(subcircuit, block["target"]))
            # residuals.append(cost_4.get_residuals(subcircuit, block["target"]))
            # dists.append(dist_cost(subcircuit.get_unitary(), block["target"]))
        print("Initial Error: ", data.error, flush=True)
        # print("Distances", dists_1, flush=True)
        # print("Distances 2", dists_2, flush=True)
        print("Upperbound 1:", sum(dists_1), "Upperbound 2:", sum(dists_2), flush=True)
        print("Full Distance", full_dist, flush=True)
        print("Full Distance 2", full_dist_2, flush=True)


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

        