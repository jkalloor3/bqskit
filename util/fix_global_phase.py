"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import GlobalPhaseGate



class FixGlobalPhasePass(BasePass):
     
    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        for psol in data["scan_sols"]:
            target = data.target
            unitary = psol.get_unitary()
            global_phase_correction = target.get_target_correction_factor(unitary)
            psol.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))