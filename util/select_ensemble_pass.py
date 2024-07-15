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
            config: list[CircuitGate], 
            circuit: Circuit, 
            pts: list[CircuitPoint], 
            locations: list[CircuitLocationLike]):
        """Unfold a circuit from a list of CircuitGates."""
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
        ensemble_probs: list[tuple[Circuit, list[float]]],
        circuit: Circuit,
        pts: list[CircuitPoint],
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""
        # print("ASSEMBLING CIRCUIT")
        
        all_combos = []
        i = 0

        num_circs = 0
        trials = 0

        inds = [[] for _ in range(len(ensemble_probs))]
        for i in range(len(ensemble_probs)):
            _, probs = ensemble_probs[i]
            assert(len(probs) > 0)
            # Sample from the marginal distribution
            inds[i] = np.random.choice(np.arange(len(probs)), p=probs, size=(self.num_circs))

        inds = np.array(inds)
        # Combine blocks
        for i in range(self.num_circs):
            random_inds = inds[:, i]
            circ_list = [ensemble_probs[i][0][ind] for i, ind in enumerate(random_inds)]
            new_config = [CircuitGate(circ) for circ in circ_list]
            all_combos.append(new_config)
            num_circs += 1

        locations = [circuit[pt].location for pt in pts]
        all_circs= await get_runtime().map(
            SelectFinalEnsemblePass.unfold_circ,
            all_combos,
            circuit=circuit,
            pts=pts,
            locations=locations
        )
        print("Semi-Final Number of Ensemble Circs: ", len(all_circs))

        return all_circs


    def parse_data(self,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint], list[list[float]]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[Circuit]] = [[] for _ in block_data]
        probs: list[list[float]] = [[] for _ in block_data]
        pts: list[CircuitPoint] = []

        print("PARSING DATA", flush=True)

        for i, block in enumerate(block_data):
            pts.append(block['point'])
            psols[i] = block["final_ensemble"]
            probs[i] = block["final_ensemble_probs"]

        return psols, pts, probs


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Selects the final ensemble of circuits
        """
        print("RUNNING SELECT ENSEMBLE PASS", flush=True)
        block_data = data[ForEachBlockPass.key]
        ensembles, pts, probs = self.parse_data(block_data)

        ensemble_probs = list(zip(ensembles, probs))

        all_circs: list[Circuit] = await self.assemble_circuits(ensemble_probs, circuit=circuit, pts=pts)
        all_circs = sorted(all_circs, key=lambda x: x.count(CNOTGate()))

        print("FINAL ENSEMBLE SIZE", len(all_circs))

        data["final_ensemble"] = all_circs

        return

        
