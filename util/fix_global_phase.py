"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

from bqskit.qis import UnitaryMatrix
from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.runtime import get_runtime

from bqskit.ir.opt.cost.functions import NormalizedFrobeniusCostGenerator
# from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator

# hs_cost = HilbertSchmidtResidualsGenerator()
frob_cost = NormalizedFrobeniusCostGenerator()


def fix_phase(circuit: Circuit, target: UnitaryMatrix) -> float:
    unitary = circuit.get_unitary()
    global_phase_correction = target.get_target_correction_factor(unitary)
    # old_cost = frob_cost.calc_cost(circuit, target)
    circuit.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
    new_cost = frob_cost.calc_cost(circuit, target)
    return new_cost

class FixGlobalPhasePass(BasePass):
    
    def __init__(self):
        super().__init__()
        self.target = None


    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        target = data.target
        new_scan_sols = []
        distances = []
        for psol in data["scan_sols"]:
            new = fix_phase(psol[0], target)
            new_scan_sols.append((psol[0], new))
            distances.append(new)
        # print("After GP Distances: ", distances, flush=True)
        data["scan_sols"] = new_scan_sols
