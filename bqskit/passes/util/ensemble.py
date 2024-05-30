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
        self.instantiate_options={
            'min_iters': 100,
            'ftol': self.success_threshold,
            'gtol': 1e-10,
            'cost_fn_gen': self.cost,
            'dist_tol': self.success_threshold,
            'method': 'qfactor'
        }

        # self.instantiate_options: dict[str, Any] = {
        #     'dist_tol': self.success_threshold,
        #     'min_iters': 100,
        #     'cost_fn_gen': self.cost,
        #     'method': 'minimization',
        #     'minimizer': LBFGSMinimizer(),
        # }

    async def unfold_circ(config: list[CircuitGate], circuit: Circuit, pts: list[CircuitPoint], locations):
            operations = [
                Operation(cg, loc, cg._circuit.params)
                for cg, loc
                in zip(config, locations)
            ]
            copied_circuit = circuit.copy()
            copied_circuit.batch_replace(pts, operations)
            copied_circuit.unfold_all()
            return copied_circuit

    async def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint]
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""


        # print("ASSEMBLING CIRCUIT")
        all_combos = []
        i = 0

        used_inds = set()
        num_circs = 0
        trials = 0

        inds = [[] for _ in range(len(psols))]
        for i in range(len(psols)):
            numbers = np.arange(0, len(psols[i]))
            weights = np.array([10 - i for i in numbers])
            weights = weights / np.sum(weights)
            inds[i] = np.random.choice(numbers, p=weights, size=(self.num_circs * 2))

        inds = np.array(inds)

        # Randomly sample a bunch of psols
        while (num_circs < self.num_circs and trials < self.num_circs * 2):
            random_inds = inds[:, trials]
            trials += 1
            if (str(random_inds) in used_inds):
                continue
            else:
                used_inds.add(str(random_inds))
                circ_list = [psols[i][ind] for i, ind in enumerate(random_inds)]
                new_config = [CircuitGate(circ) for circ in circ_list]
                all_combos.append(new_config)
                num_circs += 1

        locations = [circuit[pt].location for pt in pts]
        all_circs = await get_runtime().map(
            CreateEnsemblePass.unfold_circ,
            all_combos,
            circuit=circuit,
            pts=pts,
            locations=locations
        )
        return all_circs


    def parse_data(self,
        blocked_circuit: Circuit,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[Circuit]] = []
        pts: list[CircuitPoint] = []
        targets: list[UnitaryMatrix] = []

        num_sols = 1
        print("PARSING DATA")

        # print(num_sols_per_block)

        # TODO: Add some randomness
        for block in block_data:
            pts.append(block['point'])
            targets.append(block["target"])
            exact_block: Circuit = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            dist_2 = self.cost(exact_block, block["target"])
            # print(dist_2)
            psols.append([])

            if 'scan_sols' not in block:
                print("NO SCAN SOLS")
                continue

            i = 0

            # print("Num Sols for layer", len(block['scan_sols']))
            while i < min(len(block['scan_sols']), 4):
                psol: Circuit = block["scan_sols"][i]
                psols[-1].append(psol)
                # if len(block["scan_sols"][i]) > 1:
                # print(psol.num_operations, psol.num_params)
                i += 1

            # print("I", i)

            num_sols *= i


        print("Total Solutions", num_sols)

        self.num_circs = min(self.num_circs, num_sols)

        return psols, pts

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("Running Ensemble Pass", flush=True)

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]
        approx_circs, pts = self.parse_data(circuit, block_data)
        
        all_circs: list[Circuit] = await self.assemble_circuits(circuit, approx_circs, pts)

        all_circs = sorted(all_circs, key=lambda x: x.count(CNOTGate()))

        data["scan_sols"] = all_circs
        # data["ensemble_dists"] = all_dists
        return circuit

        
