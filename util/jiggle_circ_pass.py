"""This module implements the ToU3Pass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.passes import ForEachBlockPass
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator, HilbertSchmidtResidualsGenerator, FrobeniusCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.qis import UnitaryMatrix
import numpy as np

def cost_1(circuit: Circuit, unitary_2: UnitaryMatrix) -> float:
    unitary_1 = circuit.get_unitary()
    inside = np.trace(np.dot(unitary_1.conj().T, unitary_2))
    N = unitary_1.shape[0]
    cost_inside = min((np.abs(inside) / N) ** 2, 1)
    return np.sqrt(1 - cost_inside)

cost_2 = FrobeniusCostGenerator().calc_cost

cost_3 = HilbertSchmidtResidualsGenerator().calc_cost

def cost_4(circuit: Circuit, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    random_phase = np.exp(-1j * np.angle(np.random.uniform(0, 2 * np.pi)))
    circuit.append_gate(GlobalPhaseGate(global_phase=random_phase), (0,))
    utry = circuit.get_unitary()
    diff = utry- target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    N = utry.shape[0]
    cost = cost / np.sqrt(2 * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost


class  GetErrorsPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        unfolded_circ = circuit.copy()
        unfolded_circ.unfold_all()
        # full_dist = cost_1(unfolded_circ, data.target)
        # full_dist_2 = cost_2(unfolded_circ, data.target)
        # full_dist_3 = cost_3(unfolded_circ, data.target)
        full_dist_4 = cost_4(unfolded_circ, data.target)

        # print(full_dist, full_dist_2, full_dist_3, flush=True)

        # assert np.allclose(full_dist, np.sqrt(full_dist_2))
        # full_dist = dist_cost(unfolded_circ.get_unitary(), data.target)

        block_data = data[ForEachBlockPass.key][0]
        targets: list[UnitaryMatrix] = []

        dists_1 = []
        dists_2 = []
        dists_3 = []
        dists_4 = []

        for i, (cycle, op) in enumerate(circuit.operations_with_cycles()):
            block = block_data[i]
            targets.append(block["target"])
            subcircuit = op.gate._circuit.copy()
            subcircuit.set_params(op.params) 
            dists_1.append(cost_1(subcircuit, block["target"]))
            dists_2.append(cost_2(subcircuit, block["target"]))
            dists_3.append(cost_3(subcircuit, block["target"]))
            dists_4.append(cost_4(subcircuit, block["target"]))

        if sum(dists_4) < full_dist_4:
            print("FAIL")
            print("Initial Error: ", data.error, flush=True)
            print("Upperbound:", sum(dists_4), flush=True)
            print("Full Distance 4", full_dist_4, flush=True)

        if "ind" in data:
            data["ind"] += 1
        else:
            data["ind"] = 1


class  JiggleCircPass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        params = circuit.params
        if len(params) > 0:
            extra_diff = 0.001
            num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + len(params) // 10
            # num_params_to_jiggle = len(params)
            params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)
            jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
            # print("jiggle_amounts", jiggle_amounts, flush=True)
            params[params_to_jiggle] = params[params_to_jiggle] + jiggle_amounts
            circuit.set_params(params)

            # Add Global Phase as well
            random_angle = np.random.uniform(0, 2 * np.pi)
            random_phase = np.exp(-1j * np.angle(random_angle))
            circuit.append_gate(GlobalPhaseGate(global_phase=random_phase), (0,))
        return

        
