"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import pickle
import os

from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class SynthesisPass(BasePass):
    """
    SynthesisPass abstract class.

    The SynthesisPass is a base class that exposes an abstract synthesize
    function. Inherit from this class and implement the synthesize function to
    create a synthesis tool.

    A SynthesisPass will synthesize a new circuit targeting the input circuit's
    unitary.
    """

    @abstractmethod
    async def synthesize(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """
        Synthesis abstract method to synthesize a UnitaryMatrix into a Circuit.

        Args:
            utry (UnitaryMatrix): The unitary to synthesize.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous passes. This function should never error based
                on what is in this dictionary.

        Note:
            This function should be self-contained and have no side effects.
        """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if data.get("run_ensemble", False):
            data["max_layer"] = circuit.count(CNOTGate())
            # If we are running in ensemble mode, we need to synthesize
            # the circuit for each input circuit.
            new_ensemble_circs = []
            cur_ensemble_circs: list[Circuit] = data.get("ensemble_circuits", [circuit])
            p_file = str(data["ensemble_name"]) + "_leap_" + str(data["index"]) + ".pkl"
            if os.path.exists(p_file):
                # print("Reloading pickle file")
                new_ensemble_circs = pickle.load(open(p_file, "rb"))
            else:
                for circ in cur_ensemble_circs:
                    new_ensemble_circs.extend(await self.synthesize(circ.get_unitary(), data))
                pickle.dump(new_ensemble_circs, open(p_file, "wb"))
            # print distance
            data["ensemble_circuits"] = new_ensemble_circs
        else:
            circuit.become(await self.synthesize(data.target, data))
