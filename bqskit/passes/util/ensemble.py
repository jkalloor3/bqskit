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

    def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint],
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""


        print("ASSEMBLING CIRCUIT")
        all_combos = [[]]
        i = 0
        while i < len(psols):
            new_combos = []
            circ_list = psols[i]
            print("Num Circs in layer", len(circ_list))
            for circ in circ_list:
                cg = CircuitGate(circ)
                new_configs = [config + [cg] for config in all_combos]
                new_combos.extend(new_configs)
            all_combos = new_combos
            print(len(all_combos))
            i += 1

        locations = [circuit[pt].location for pt in pts]
        all_circs = []
        for config in all_combos:
            operations = [
                Operation(cg, loc, cg._circuit.params)
                for cg, loc
                in zip(config, locations)
            ]
            copied_circuit = circuit.copy()
            copied_circuit.batch_replace(pts, operations)
            copied_circuit.unfold_all()
            all_circs.append(copied_circuit)
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

        num_sols_per_block = max(int(self.num_circs ** (1/len(block_data))), 2)

        num_sols = 1

        print(num_sols_per_block)

        # TODO: Add some randomness
        for block in block_data:
            pts.append(block['point'])
            targets.append(block["target"])
            exact_block: Circuit = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            dist_2 = self.cost(exact_block, block["target"])
            print(dist_2)
            psols.append([])

            if 'scan_sols' not in block:
                print("NO SCAN SOLS")
                continue

            i = 0

            print("Num Sols for layer", len(block['scan_sols']))

            while i < min(num_sols_per_block, len(block['scan_sols']), max(self.num_circs - num_sols, 1)):
                psols[-1].append(block["scan_sols"][i][0])
                i += 1

            print("I", i)

            num_sols *= i


        print("Total Solutions", num_sols)

        return psols, pts, targets, num_sols


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]
        approx_circs, pts, targets, num_sols = self.parse_data(circuit, block_data)

        print("Num Blocks", len(approx_circs))
        
        seed = 0

        print([len(x) for x in approx_circs])
        while num_sols < self.num_circs:
            for i, circ_list in enumerate(approx_circs):
                print("Num Sols", num_sols)
                print("Current list has this many psols: ", len(circ_list))
                num_sols /= len(circ_list)
                target = targets[i]
                # num_workers = 512
                # num_multistarts = min(num_workers // len(circ_list), 4)
                # if num_multistarts > 0: # Utilize workers better
                #     self.instantiate_options["multistarts"] = num_multistarts
                #     print("Num Multistarts:", num_multistarts)
                print("Instantiating")
                instantiated_circuits: list[Circuit] = await get_runtime().map(
                    Circuit.instantiate,
                    circ_list,
                    target=target,
                    seed=seed,
                    **self.instantiate_options
                )
                for circ in instantiated_circuits:
                    cost = self.cost(circ, target)
                    if cost < self.success_threshold:
                        # if cost not in all_dists:
                        #     # Cheating way to check if unitary is new
                        approx_circs[i].append(circ)
                        # all_dists.add(cost)
                        added_circ = True


                num_sols *= len(approx_circs[i])
                seed += 1
                if num_sols > self.num_circs:
                    break


        print([len(x) for x in approx_circs])
        all_circs = self.assemble_circuits(circuit, approx_circs, pts)
        # init_approx_circuits: list[tuple[Circuit, float]] = CreateEnsemblePass.assemble_circuits(approx_circs, pts)

        # Only use the shortest distance / 10
        # print("All Circuits: ", len(init_approx_circuits))
        # num_init_circs = min(max(self.num_circs // 10, 4), len(init_approx_circuits))
        # all_init_circs = [x[0] for x in init_approx_circuits]
        # counts = [x.count(CNOTGate()) for x in all_init_circs]
        # sort_inds = np.argsort(counts)[:num_init_circs]

        # init_circs = [all_init_circs[i] for i in sort_inds]

        # all_circs = [x for x in init_circs]
        # all_dists = set([x[1] for x in init_approx_circuits])

        # print(all_dists)

        # target = self.get_target(circuit, data)

        # # From our circuits, we are going to try to instantiate a bunch of circuits for our ensemble
        # seed = 0
        # fails = 0

        # print("Init Circs!", len(init_circs))

        # while len(all_circs) < self.num_circs:
        #     # Pick all circuits

        #     # working_copy.instantiate(target, **instantiate_options)
        #     instantiated_circuits: list[Circuit] = await get_runtime().map(
        #                 Circuit.instantiate,
        #                 init_circs,
        #                 target=target,
        #                 seed=seed,
        #                 **self.instantiate_options
        #     )
        #     _logger.debug(f"Instantiated {len(instantiated_circuits)} circuits.")

        #     added_circ = False
        #     for circ in instantiated_circuits:
        #         cost = self.cost(circ, target)
        #         if cost < self.success_threshold:
        #             # if cost not in all_dists:
        #             #     # Cheating way to check if unitary is new
        #             all_circs.append(circ)
        #             # all_dists.add(cost)
        #             added_circ = True
                
        #     if not added_circ:
        #         fails += 1

        #     if fails > 50:
        #         print("Failed to generate all circs!")
        #         data["ensemble"] = all_circs
        #         return circuit

        #     seed += 1

        data["ensemble"] = all_circs
        # data["ensemble_dists"] = all_dists
        return circuit

        
