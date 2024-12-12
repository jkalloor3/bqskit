"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import linregress
from os.path import join, exists
import pickle
from pathlib import Path

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic, DijkstraHeuristic
from bqskit.compiler.basepass import BasePass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from itertools import chain, zip_longest
from .distance import normalized_frob_cost

_logger = logging.getLogger(__name__)


class SecondLEAPSynthesisPass(BasePass):
    """
    A pass implementing the LEAP search synthesis algorithm.

    References:
        Ethan Smith, Marc G. Davis, Jeffrey M. Larson, Ed Younis,
        Lindsay Bassman, Wim Lavrijsen, and Costin Iancu. 2022. LEAP:
        Scaling Numerical Optimization Based Synthesis Using an
        Incremental Approach. ACM Transactions on Quantum Computing
        (June 2022). https://doi.org/10.1145/3548693
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = DijkstraHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int = 40,
        min_prefix_size: int = 3,
        instantiate_options: dict[str, Any] = {},
        partial_success_threshold: float = 1e-3,
        use_calculated_error: bool = False,
        max_psols: int = 5,
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

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

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.partial_success_threshold = partial_success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.min_prefix_size = min_prefix_size
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': HilbertSchmidtResidualsGenerator(),
        }
        self.use_calculated_error = use_calculated_error
        self.instantiate_options.update(instantiate_options)
        self.max_psols = max_psols

    async def synthesize(self, data: PassData, target: UnitaryMatrix, default_circuit: Circuit) -> Circuit:
        # Synthesize every circuit in the ensemble
        circs: list[Circuit] = [d[0] for d in data['scan_sols'][:-1]]
        block_id = f"Block {data.get('super_block_num', -1)}_{data.get('block_num', -1)}:"
        factor = data["error_percentage_allocated"]
        partial_success_threshold = self.partial_success_threshold * factor
        print(f"{block_id} Partial Success Threshold: ", partial_success_threshold, flush=True)
        print(f"{block_id} After Leap 1 distances: ", [d[1] for d in data['scan_sols']], flush=True)
        if len(circs) > 0:
            for c in circs:
                assert(isinstance(c, Circuit))

            utries = [c.get_unitary() for i, c in enumerate(circs)]

            new_circs: list[list[Circuit]] = await get_runtime().map(
                self.synthesize_circ,
                utries,
                data=data.copy(),
                default_count=default_circuit.count(CNOTGate())
            )

            for i, circs in enumerate(new_circs):
                if not isinstance(circs, list):
                    print(len(utries), len(new_circs), len(circs), type(circs), flush=True)
                assert(isinstance(circs, list))
                for c in circs:
                    assert(isinstance(c, Circuit))

            for i, circs in enumerate(new_circs):
                new_circs[i] = sorted(circs, key=lambda x: x.count(CNOTGate()))
            
            # Now interleave so you get a decent mix
            new_circs = list(chain(*zip_longest(*new_circs, fillvalue=None)))
            new_circs: list[tuple[Circuit, float]] = [(c, self.cost(c, target)) for c in new_circs if c is not None]
            new_circs = sorted(new_circs, key=lambda x: x[0].count(CNOTGate()))
        else:
            new_circs = []

        # Randomly choose up to 12 circuits
        if len(new_circs) > 8:
            # Weight by count, less gates is more likely
            print(f"{block_id} Subselecting 8 circuits", flush=True)
            print(f"{block_id} Original Distances: ", [c[1] for c in new_circs], flush=True)
            print(f"{block_id} Original Counts: ", [c[0].count(CNOTGate()) for c in new_circs], flush=True)
            counts = np.array([1 / (c[0].count(CNOTGate()) ** 2 + 1) for c in new_circs])
            counts = counts / np.sum(counts)
            print("Probabilities: ", counts, flush=True)
            inds = np.random.choice(len(new_circs), 8, replace=False, p=counts)
            new_circs = [new_circs[i] for i in inds]
            print(f"{block_id} Success Threshold: ", partial_success_threshold, flush=True)
            print(f"{block_id} Final Distances: ", [c[1] for c in new_circs], flush=True)
            print(f"{block_id} Final Counts: ", [c[0].count(CNOTGate()) for c in new_circs], flush=True)    

        new_circs.append((default_circuit, 0))

        assert(isinstance(new_circs[0][0], Circuit))

        assert(len(new_circs) > 0)
        # print("Return Scan Sols")
        data['scan_sols'] = new_circs

        return new_circs[0][0]

    async def synthesize_circ(
        self,
        utry: UnitaryMatrix,
        data: PassData,
        default_count: int
    ) -> list[Circuit]:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        if self.use_calculated_error:
            # use sqrt
            factor = data["error_percentage_allocated"]
            partial_success_threshold = self.partial_success_threshold * factor
        else:
            partial_success_threshold = self.partial_success_threshold

        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()
        instantiate_options['ftol'] = partial_success_threshold

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        max_layer = min(self.max_layer, default_count + 2)

        # Begin the search with an initial layer
        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)

        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_layer = 0

        # Track partial solutions
        scan_sols: list[tuple[Circuit, float]] = []

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < partial_success_threshold:
            _logger.debug('Successful synthesis.')
            scan_sols.append(initial_layer.copy())

        # Main loop
        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                continue

            # Instantiate successors
            circuits: list[Circuit] = await get_runtime().map(
                Circuit.instantiate,
                successors,
                target=utry,
                **instantiate_options,
            )

            # Evaluate successors
            for circuit in circuits:
                dist = normalized_frob_cost(circuit.get_unitary(), utry)
                if dist < best_dist:
                    best_dist = dist

                if dist < partial_success_threshold:
                    scan_sols.append(circuit.copy())
                    if len(scan_sols) >= self.max_psols:
                        return scan_sols

                if max_layer is None or layer + 1 < max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        return scan_sols

    def check_new_best(
        self,
        layer: int,
        dist: float,
        best_layer: int,
        best_dist: float,
        success_threshold: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
            dist < best_dist
            and (
                best_dist >= success_threshold
                or layer <= best_layer
            )
        )
        better_dist_and_layer = (
            dist < success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
        self,
        new_layer: int,
        best_dist: float,
        best_layers: list[int],
        best_dists: list[float],
        last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)

        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.min_prefix_size

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """
        Set the layer generator.

        If a layer generator has been passed into the constructor, then that
        layer generator will be used. Otherwise, a default layer generator will
        be selected by the gateset.

        If seeds are passed into the data dict, then a SeedLayerGenerator will
        wrap the previously selected layer generator.
        """
        # TODO: Deduplicate this code with qsearch synthesis
        # layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        layer_gen = SimpleLayerGenerator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print(f"Starting Second LEAP for block {data.get('super_block_num', -1)} : {data.get('block_num', -1)}", flush=True)
        save_file: str = data.get("checkpoint_data_file", None)
        if "finished_second_leap" in data:
            print("Already finished second leap!", flush=True)
            return

        await self.synthesize(data, target=data.target, default_circuit=data['scan_sols'][-1][0])

        data["finished_second_leap"] = True
        if save_file:
            pickle.dump(data, open(save_file, "wb"))