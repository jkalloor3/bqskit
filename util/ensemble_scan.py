"""This module implements the ScanningGateRemovalPass."""
from __future__ import annotations

import logging
from os.path import exists, join
from os import mkdir
import numpy as np
import pickle
from typing import Any
from typing import Callable
from itertools import chain

from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


class EnsembleScanningGateRemovalPass(BasePass):
    """
    The ScanningGateRemovalPass class.

    Starting from one side of the circuit, attempt to remove gates one-by-one.
    """

    def __init__(
        self,
        start_from_left: bool = True,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        collection_filter: Callable[[Operation], bool] | None = None,
        use_calculated_error: bool = False,
    ) -> None:
        """
        Construct a ScanningGateRemovalPass.

        Args:
            start_from_left (bool): Determines where the scan starts
                attempting to remove gates from. If True, scan goes left
                to right, otherwise right to left. (Default: True)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should be
                attempted to be removed. Called with each operation
                in the circuit. If this returns true, this pass will
                attempt to remove that operation. Defaults to all
                operations.
        """

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.collection_filter = collection_filter or default_collection_filter

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        self.start_from_left = start_from_left
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.use_calculated_error = use_calculated_error

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation on the given circuit."""
        ens = data["scan_sols"]
        # for ens in ensembles:
        print("Launching Scanning Gate Removal on Ensemble", flush=True)
        all_sols: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.run_circ, ens, data=data)
        # Pick 3 solutions from each list into final list
        final_sols = []
        for i in range(len(all_sols)):
            num_sols = min(5, len(all_sols[i]))
            random_inds = np.random.choice(len(all_sols[i]), num_sols, replace=False)
            final_sols.extend([all_sols[i][j] for j in random_inds])

        print(f"After Scanning, we have {len(final_sols)} solutions", flush=True)
        data["scan_sols"] = final_sols

        if "checkpoint_dir" in data:
            checkpoint_data_file = data["checkpoint_data_file"]
            data[ "finished_scanning_gate_removal"] = True
            pickle.dump(data, open(checkpoint_data_file, "wb"))
        return


    async def run_circ(self, circ_float: tuple[Circuit, float], data: PassData) -> list[tuple[Circuit, float]]:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circuit, dist = circ_float
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        start = 'left' if self.start_from_left else 'right'
        _logger.debug(f'Starting scanning gate removal on the {start}.')

        target = self.get_target(circuit, data)
        all_solutions: list[tuple[Circuit, float]] = [(circuit.copy(), dist)]

        circuit_copy = circuit.copy()
        reverse_iter = not self.start_from_left

        iterator = circuit.operations_with_cycles(reverse=reverse_iter)
        all_ops = [x for x in iterator]

        if self.use_calculated_error:
            success_threshold = self.success_threshold * data["error_percentage_allocated"]
        else:
            success_threshold = self.success_threshold

        # print(f"Initial Gate Counts: {circuit.gate_counts}")
        # print(f"Initial Width: {circuit.num_qudits}", flush=True)
        ops_removed = []

        for i, (cycle, op) in enumerate(all_ops):

            if not self.collection_filter(op):
                _logger.debug(f'Skipping operation {op} at cycle {cycle}.')
                continue

            # print(f'Attempting removal of operation at cycle {cycle}.')
            # print(f'Operation: {op}')

            working_copy = circuit_copy.copy()

            # If removing gates from the left, we need to track index changes.
            if self.start_from_left:
                idx_shift = circuit.num_cycles
                idx_shift -= working_copy.num_cycles
                cycle -= idx_shift

            pot_op = working_copy.pop((cycle, op.location[0]))
            working_copy.instantiate(target, **instantiate_options)

            working_cost = self.cost(working_copy, target)
            # print(f"Cost after removing {op} is {working_cost}", flush=True)
            if working_cost < success_threshold:
                ops_removed.append(op.gate)
                all_solutions.append((working_copy.copy(), working_cost))
                _logger.debug('Successfully removed operation.')
                circuit_copy = working_copy        

        print("Ops Removed: ", ops_removed, flush=True)
        circuit.become(circuit_copy)
        all_solutions = sorted(all_solutions, key=lambda x: x[0].count(CNOTGate()))
        return all_solutions


def default_collection_filter(op: Operation) -> bool:
    return True
