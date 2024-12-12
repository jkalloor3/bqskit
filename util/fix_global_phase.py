"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import GlobalPhaseGate

from bqskit.ir.opt.cost.functions import  FrobeniusCostGenerator, HilbertSchmidtResidualsGenerator

hs_cost = HilbertSchmidtResidualsGenerator()
frob_cost = FrobeniusCostGenerator()

class FixGlobalPhasePass(BasePass):
     
    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        target = data.target
        new_scan_sols = []
        for psol in data["scan_sols"]:
            unitary = psol[0].get_unitary()
            old = frob_cost.calc_cost(psol[0], target)
            global_phase_correction = target.get_target_correction_factor(unitary)
            psol[0].append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            new = frob_cost.calc_cost(psol[0], target)
            hs = hs_cost.calc_cost(psol[0], target)
            new_scan_sols.append((psol[0], new))
            print("Old cost: ", old, "New cost: ", new, "HS Cost: ", hs, flush=True)
        data["scan_sols"] = new_scan_sols