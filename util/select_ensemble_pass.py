"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation, CircuitLocationLike
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from itertools import zip_longest, chain


_logger = logging.getLogger(__name__)


class SelectFinalEnsemblePass(BasePass):
    """ 
    Selects the final ensemble of circuits 
    to use by sampling using the marginal distributions
    of each block. 
    """
    def __init__(self, size):
        self.num_circs = size


    async def unfold_circ(
            configs: list[list[CircuitGate]], 
            circuit: Circuit, 
            pts: list[CircuitPoint], 
            locations: list[CircuitLocationLike]):
        """Unfold a circuit from a list of CircuitGates."""

        circs = []
        for config in configs:
            assert len(config) == len(pts)
            operations = [
                Operation(cg, loc, cg._circuit.params)
                for cg, loc
                in zip(config, locations)
            ]
            copied_circuit = circuit.copy()
            copied_circuit.batch_replace(pts, operations)
            copied_circuit.unfold_all()
            circs.append(copied_circuit)

        return circs

    async def create_configs(self, inds_list: list[list[int]]) -> list[list[CircuitGate]]:
        configs = []
        for random_inds in inds_list:
            assert len(random_inds) == len(self.ensemble)
            circ_list = [self.ensemble[i][ind] for i, ind in enumerate(random_inds)]
            # print("Circ List: ", [type(c) for c in circ_list])
            new_config = [CircuitGate(circ) for circ in circ_list]
            configs.append(new_config)
        return configs

    async def assemble_circuits(
        self,
        circuit: Circuit,
        pts: list[CircuitPoint],
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""
        # print("ASSEMBLING CIRCUIT")
        
        all_combos = []
        i = 0

        inds: list[list[int]] = [[] for _ in range(len(self.ensemble))]

        for i in range(len(self.ensemble)):
            probs = self.probs[i]
            assert(len(probs) > 0)
            # Sample from the marginal distribution
            inds[i] = np.random.choice(np.arange(len(probs)), p=probs, size=(self.num_circs))

        print("Sampled Indices", flush=True)
        print("Number of Ensembles: ", len(inds), flush=True)

        inds: np.ndarray = np.array(inds).transpose()
        print(inds.shape)
        arr_split = np.array_split(inds, 100)
        print("Split into ", len(arr_split), flush=True)
        print("First Split Shape: ", arr_split[0].shape, flush=True)

        # Combine blocks
        grouped_combos: list[list[list[CircuitGate]]] = await get_runtime().map(self.create_configs, arr_split)

        # print(f"Created {len(all_combos)} Configs", flush=True)

        # grouped_combos = zip_longest(*(iter(all_combos),) * 50)

        print(f"Grouped into {len(grouped_combos)} Configs", flush=True)

        locations = [circuit[pt].location for pt in pts]
        all_circs= await get_runtime().map(
            SelectFinalEnsemblePass.unfold_circ,
            grouped_combos,
            circuit=circuit,
            pts=pts,
            locations=locations
        )

        all_circs = list(chain.from_iterable(all_circs))
        print("Semi-Final Number of Ensemble Circs: ", len(all_circs), flush=True)

        return all_circs


    def parse_data(self,
        data: dict[Any, Any],
    ) -> tuple[list[list[Circuit]], list[CircuitPoint], list[list[float]]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[Circuit]] = [[] for _ in block_data]
        probs: list[list[float]] = [[] for _ in block_data]
        pts: list[CircuitPoint] = []

        total_solutions = 1
        for i, block in enumerate(block_data):
            pts.append(block['point'])
            psols[i] = block["final_ensemble"]
            probs[i] = block["final_ensemble_probs"]
            total_solutions *= len(psols[i])

        print("Total FINAL Solutions", total_solutions, flush=True)

        return psols, pts, probs


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Selects the final ensemble of circuits
        """
        print("RUNNING SELECT ENSEMBLE PASS", flush=True)
        block_data = data[ForEachBlockPass.key]
        ensembles, pts, probs = self.parse_data(block_data)

        # self.ensemble_probs = list(zip(ensembles, probs))
        self.ensemble = ensembles
        self.probs = probs

        # print(ensemble_probs[0])

        all_circs: list[Circuit] = await self.assemble_circuits(circuit=circuit, pts=pts)
        all_circs = sorted(all_circs, key=lambda x: x.count(CNOTGate()))

        print("FINAL ENSEMBLE SIZE", len(all_circs))

        data["final_ensemble"] = all_circs

        return

        
