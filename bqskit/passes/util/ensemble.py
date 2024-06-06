"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation, CircuitLocationLike
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
                 cost: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
                 use_calculated_error: bool = False,
                 solve_exact_dists: bool = False) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.cost = cost
        self.solve_exact_dists = solve_exact_dists
        self.use_calculated_error = use_calculated_error
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

    async def unfold_circ(
            config_dist: tuple[list[CircuitGate], float], 
            circuit: Circuit, 
            pts: list[CircuitPoint], 
            locations: list[CircuitLocationLike], 
            solve_exact_dists: bool = False, 
            cost: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
            target: UnitaryMatrix = None) -> tuple[Circuit, float]:
        
        config, dist = config_dist
        operations = [
            Operation(cg, loc, cg._circuit.params)
            for cg, loc
            in zip(config, locations)
        ]
        copied_circuit = circuit.copy()
        copied_circuit.batch_replace(pts, operations)
        copied_circuit.unfold_all()

        if solve_exact_dists:
            dist = cost.calc_cost(copied_circuit, target)

        return copied_circuit, dist

    async def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint],
        dists: list[list[float]],
        target: UnitaryMatrix = None
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
            assert(len(psols[i]) > 0)
            numbers = np.arange(0, len(psols[i]))
            # weight by number of CNOTs multiplied by log of distance

            weight_eqn = lambda x: int(psols[i][x].count(CNOTGate())) + int(-1 * np.log10(dists[i][x] + (self.success_threshold / 1000)))
            weights = np.array([weight_eqn(j) for j in range(len(psols[i]))])
            # weights = np.exp(weights - np.max(weights))
            weights = weights / np.sum(weights)
            inds[i] = np.random.choice(numbers, p=weights, size=(self.num_circs * 5))

        inds = np.array(inds)
        # Randomly sample a bunch of psols
        while (num_circs < self.num_circs and trials < self.num_circs * 5):
            random_inds = inds[:, trials]
            total_dist = sum(dists[i][ind] for i, ind in enumerate(random_inds))
            # print(total_dist, flush=True)
            trials += 1
            if (str(random_inds) in used_inds):
                continue
            else:
                used_inds.add(str(random_inds))
                circ_list = [psols[i][ind] for i, ind in enumerate(random_inds)]
                new_config = [CircuitGate(circ) for circ in circ_list]
                all_combos.append((new_config, total_dist))
                num_circs += 1

        locations = [circuit[pt].location for pt in pts]
        all_circs_dists = await get_runtime().map(
            CreateEnsemblePass.unfold_circ,
            all_combos,
            circuit=circuit,
            pts=pts,
            locations=locations,
            solve_exact_dists=self.solve_exact_dists,
            cost=self.cost,
            target=target
        )

        # all_circs_dists = np.array(all_circs_dists)
        all_circs_dists = [(circ, dist) for circ, dist in all_circs_dists if dist < self.success_threshold]

        print("Final Number of Ensemble Circs: ", len(all_circs_dists))

        return all_circs_dists


    def parse_data(self,
        blocked_circuit: Circuit,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint], list[list[float]]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[Circuit]] = [[] for _ in block_data]
        dists: list[list[float]] = [[] for _ in block_data]
        pts: list[CircuitPoint] = []
        targets: list[UnitaryMatrix] = []

        num_sols = 1
        print("PARSING DATA", flush=True)

        # print(num_sols_per_block)

        # TODO: Add some randomness
        for i, block in enumerate(block_data):
            pts.append(block['point'])
            targets.append(block["target"])
            exact_block: Circuit = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            # dist_2 = self.cost(exact_block, block["target"])
            # print(dist_2)
            # psols.append([])

            if 'scan_sols' not in block:
                print("NO SCAN SOLS")
                continue

            psols[i] = [block['scan_sols'][j][0] for j in range(len(block['scan_sols']))]
            dists[i] = [block['scan_sols'][j][1] for j in range(len(block['scan_sols']))]   
            num_sols *= len(psols[i])


        print("Total Potential Solutions", num_sols)

        self.num_circs = min(self.num_circs, num_sols)

        return psols, pts, dists

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("Running Ensemble Pass", flush=True)

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]

        if self.use_calculated_error:
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]

        approx_circs, pts, dists = self.parse_data(circuit, block_data)
        
        all_circs: list[tuple[Circuit, float]] = await self.assemble_circuits(circuit, approx_circs, pts, dists=dists, target=data.target)

        all_circs = sorted(all_circs, key=lambda x: x[0].count(CNOTGate()))

        data["scan_sols"] = all_circs
        # data["ensemble_dists"] = all_dists
        return

        
