"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation, CircuitLocationLike
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
import pickle

_logger = logging.getLogger(__name__)

frob_cost = FrobeniusCostGenerator()
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
            success_threshold: float = 1e-4,
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
            orig_dist = dist
            dist = cost.calc_cost(copied_circuit, target)
            frob_dist = frob_cost.calc_cost(copied_circuit, target)
            # dist_2 = CreateEnsemblePass.no_phase_cost.calc_cost(copied_circuit, target)

            # print("Exact Dist: ", dist, "Upperbound: ", orig_dist, "Frob Dist:", frob_dist, "Success Threshold: ", success_threshold, flush=True)

        return copied_circuit, dist

    async def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint],
        dists: list[list[float]],
        targets: list[UnitaryMatrix],
        thresholds: list[float],
        target: UnitaryMatrix = None
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""
        # print("ASSEMBLING CIRCUIT")
        all_combos = []
        i = 0

        used_inds = set()
        num_circs = 0
        trials = 0

        inds_1 = [[] for _ in range(len(psols))]
        inds_2 = [[] for _ in range(len(psols))]
        inds_3 = [[] for _ in range(len(psols))]
        inds_4 = [[] for _ in range(len(psols))]
        default_inds = [[-1] for _ in range(len(psols))]
        random_inds = [[] for _ in range(len(psols))]
        possible_solutions = 1

        print("COUNTS FOR EACH PSOLS", flush=True)
        for i in range(len(psols)):
            print("New Counts: ", [psol.count(CNOTGate()) for psol in psols[i]], "Orig Counts: ", psols[i][-1].count(CNOTGate()), flush=True)


        for i in range(len(psols)):
            assert(len(psols[i]) > 0)
            numbers = np.arange(0, len(psols[i]))
            # weight by number of CNOTs multiplied by log of distance
            weight_eqn_1 = lambda x: psols[i][x].count(CNOTGate())
            weight_eqn_2 = lambda x: np.sqrt(psols[i][x].count(CNOTGate()))
            weight_eqn_3 = lambda x: np.sqrt(psols[i][x].count(CNOTGate())) * -1 * np.log10(dists[i][x] + self.success_threshold / 1000) # Higher weight to 
            weight_eqn_4 = lambda x: np.sqrt(-1 * np.log10(dists[i][x] + self.success_threshold / 1000)) # Only care about distance
            weights_1 = np.array([weight_eqn_1(j) for j in range(len(psols[i]))])
            weights_2 = np.array([weight_eqn_2(j) for j in range(len(psols[i]))])
            weights_3 = np.array([weight_eqn_3(j) for j in range(len(psols[i]))])
            weights_4 = np.array([weight_eqn_4(j) for j in range(len(psols[i]))])
            weights_1 = weights_1 / np.sum(weights_1)
            weights_2 = weights_2 / np.sum(weights_2)
            weights_3 = weights_3 / np.sum(weights_3)
            weights_4 = weights_4 / np.sum(weights_4)
            inds_1[i] = np.random.choice(numbers, p=weights_1, size=(self.num_circs * 10))
            inds_2[i] = np.random.choice(numbers, p=weights_2, size=(self.num_circs * 10))
            inds_3[i] = np.random.choice(numbers, p=weights_3, size=(self.num_circs * 10))
            inds_4[i] = np.random.choice(numbers, p=weights_4, size=(self.num_circs * 10))
            random_inds[i] = np.random.choice(numbers, size=(self.num_circs * 10))
            possible_solutions *= len(psols[i])

        inds_1 = np.array(inds_1)
        inds_2 = np.array(inds_2)
        inds_3 = np.array(inds_3)
        inds_4 = np.array(inds_4)
        inds_5 = np.array(random_inds)

        # Metric based on distance is worst
        all_inds = [inds_1, inds_2, inds_3, inds_5, inds_4]

        all_ensembles = []

        for k, inds in enumerate(all_inds):
            # if k == 0:
            #     print("CNOT based sampling")
            # elif k == 1:
            #     print("CNOT sqrt based sampling")
            # else:
            #     print("Other sampling")

            num_collisions = 0

            # Randomly sample a bunch of psols
            while (num_circs < self.num_circs and trials < self.num_circs * 10):
                random_inds = inds[:, trials]
                total_dist = sum(dists[i][ind] for i, ind in enumerate(random_inds))
                # print(total_dist, flush=True)
                trials += 1
                if (str(random_inds) in used_inds):
                    num_collisions += 1
                    continue
                else:
                    used_inds.add(str(random_inds))
                    circ_list = [psols[i][ind] for i, ind in enumerate(random_inds)]
                    new_config = [CircuitGate(circ) for circ in circ_list]
                    all_combos.append((new_config, total_dist))
                    if total_dist > self.success_threshold:
                        print("Dists: ", [dists[i][ind] for i, ind in enumerate(random_inds)], "Thresholds: ", thresholds, flush=True)
                        print("Total Dist: ", total_dist, "Threshold: ", self.success_threshold, flush=True)
                    # actual_dists = [self.cost.calc_cost(psols[i][ind], targets[i]) for i, ind in enumerate(random_inds)]
                    # print("Actual Dists: ", actual_dists, flush=True)
                    # frob_dists = [frob_cost.calc_cost(psols[i][ind], targets[i]) for i, ind in enumerate(random_inds)]
                    # print("Frob Dists: ", frob_dists, flush=True)
                    num_circs += 1

            # Add original circuit
            circ_list = [psols[i][-1] for i in range(len(psols))]
            default_config = [CircuitGate(circ) for circ in circ_list]
            all_combos.append((default_config, 0))

            locations = [circuit[pt].location for pt in pts]
            all_circs_dists = await get_runtime().map(
                CreateEnsemblePass.unfold_circ,
                all_combos,
                circuit=circuit,
                pts=pts,
                locations=locations,
                solve_exact_dists=self.solve_exact_dists,
                cost=self.cost,
                success_threshold=self.success_threshold,
                target=target
            )

            dists = [dist for circ, dist in all_circs_dists]

            # print("Final Dists: ", dists, flush=True)
            initial_circs = len(dists)
            
            all_circs_dists = [(circ, dist) for circ, dist in all_circs_dists if dist < self.success_threshold]

            if len(all_circs_dists) == 0:
                print(all_circs_dists, flush=True)
                print("No Circuits Found", flush=True)
                return None


            assert len(all_circs_dists) > 0

            if len(all_circs_dists) > 1:
                # Get rid of the default case
                all_circs_dists.pop()

            # print("Final Number of Circs: ", len(all_circs_dists), flush=True)

            # print("Possible Solutions: ", possible_solutions, "Initial Circs: ", initial_circs, "Final Circs: ", len(all_circs_dists), "Num Collisions: ", num_collisions, flush=True)

            all_ensembles.append(all_circs_dists)


        return all_ensembles


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
        thresholds: list[float] = []

        num_sols = 1
        # print("PARSING DATA", flush=True)

        for i, block in enumerate(block_data):
            pts.append(block['point'])
            targets.append(block["target"])
            exact_block: Circuit = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            thresholds.append(block["error_percentage_allocated"] * self.success_threshold)

            if 'scan_sols' not in block:
                print("NO SCAN SOLS")
                continue

            psols[i] = [block['scan_sols'][j][0] for j in range(len(block['scan_sols']))]
            dists[i] = [block['scan_sols'][j][1] for j in range(len(block['scan_sols']))]   
            num_sols *= len(psols[i])


        # print([len(psols[i]) for i in range(len(psols))])
        # print("Total Potential Solutions", num_sols)

        self.num_circs = min(self.num_circs, num_sols)

        return psols, pts, dists, targets, thresholds

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("Running Ensemble Pass", flush=True)

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]


        if self.use_calculated_error:
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]

        approx_circs, pts, dists, targets, thresholds = self.parse_data(circuit, block_data)

        if "finished_create_ensemble" in data:
            print("Ensemble Already Found", flush=True)
            return
        
        all_ensembles: list[tuple[Circuit, float]] = await self.assemble_circuits(circuit, approx_circs, pts, dists=dists, targets=targets, thresholds=thresholds, target=data.target)

        if all_ensembles is None:
            print("No Circuits Found", flush=True)
            print(circuit, flush=True)
            exit(1)

        data["scan_sols"] = []
        data["ensemble"] = []

        for all_circs in all_ensembles:
            all_circs = sorted(all_circs, key=lambda x: x[0].count(CNOTGate()))

            data["scan_sols"].append(all_circs)
            data["ensemble"].append([circ for circ, dist in all_circs])

        if "checkpoint_dir" in data:
            checkpoint_data_file = data["checkpoint_data_file"]
            data["finished_create_ensemble"] = True
            pickle.dump(data, open(checkpoint_data_file, "wb"))
        
        return

        
