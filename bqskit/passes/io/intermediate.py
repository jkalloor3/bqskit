"""This module implements the various intermediate results classes."""
from __future__ import annotations

import logging
import pickle
from os import listdir
from os import mkdir
from os.path import exists
from re import findall
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.passes.util.converttou3 import ToU3Pass

_logger = logging.getLogger(__name__)


class SaveIntermediatePass(BasePass):
    """
    The SaveIntermediate class.

    The SaveIntermediatePass stores individual CircuitGates in pickle or qasm
    format.
    """

    def __init__(
        self,
        path_to_save_dir: str,
        project_name: str | None = None,
        save_as_qasm: bool = True,
        is_block: bool = False,
        collection_filter: Callable[[Operation], bool] | None = None,
        overwrite: bool = True
    ) -> None:
        """
        Constructor for the SaveIntermediatePass.

        Args:
            path_to_save_dir (str): Path to the directory in which inter-
                qasm for circuit blocks should be saved.

            project_name (str): Name of the project files.

        Raises:
            ValueError: If `path_to_save_dir` is not an existing directory.
        """
        if exists(path_to_save_dir):
            self.pathdir = path_to_save_dir
            if self.pathdir[-1] != '/':
                self.pathdir += '/'
        else:
            raise ValueError(
                f'Path {path_to_save_dir} does not exist',
            )
        self.projname = project_name if project_name is not None \
            else 'unnamed_project'

        self.as_qasm = save_as_qasm
        self.is_block = is_block
        self.collection_filter = collection_filter or default_collection_filter
        enum = 1
        if exists(self.pathdir + self.projname) and not self.is_block and not overwrite:
            while exists(self.pathdir + self.projname + f'_{enum}'):
                enum += 1
            self.projname += f'_{enum}'
            _logger.warning(
                f'Path {path_to_save_dir} already exists, '
                f'saving to {self.pathdir + self.projname} '
                'instead.',
            )

        if not exists(self.pathdir + self.projname):
            mkdir(self.pathdir + self.projname)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        
        if (self.is_block):
            block_id = data["block_num"]
            block_skeleton = self.pathdir + self.projname + '/block_' + str(block_id)
            with open(f'{block_skeleton}.pickle', 'wb') as f:
                pickle.dump(circuit, f)
            with open(f'{block_skeleton}.data', 'wb') as f:
                pickle.dump(data, f)
            return
        
        # Gather and enumerate CircuitGates to save
        # Collect blocks
        blocks_to_save: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                blocks_to_save.append((cycle, op))

        # Set up path and file names
        structure_file = self.pathdir + self.projname + '/structure.pickle'
        block_skeleton = self.pathdir + self.projname + '/block_'
        # num_digits = len(str(len(blocks_to_save)))

        with open(self.pathdir + self.projname + '/data.pickle', 'wb') as data_f:
            pickle.dump(data, data_f)

        structure_list: list[list[int]] = []
        # NOTE: Block numbers are gotten by iterating through the circuit so
        # there is no guarantee that the blocks were partitioned in that order.
        for enum, (cycle, block) in enumerate(blocks_to_save):
            # enum = str(enum).zfill(num_digits)  # type: ignore
            structure_list.append(block.location)  # type: ignore
            subcircuit = Circuit(block.num_qudits)
            subcircuit.append_gate(
                block.gate,
                list(
                    range(
                        block.num_qudits,
                    ),
                ),
                block.params,
            )
            subcircuit.unfold((0, 0))
            await ToU3Pass().run(subcircuit, PassData(subcircuit))
            if self.as_qasm:
                with open(block_skeleton + f'{enum}.qasm', 'w') as f:
                    f.write(OPENQASM2Language().encode(subcircuit))
            else:
                with open(
                    f'{block_skeleton}{enum}.pickle', 'wb',
                ) as f:
                    pickle.dump(subcircuit, f)

        with open(structure_file, 'wb') as f:
            pickle.dump(structure_list, f)


class RestoreIntermediatePass(BasePass):
    def __init__(self, project_directory: str, load_blocks: bool = True, is_block: bool = False, read_as_qasm: bool = False):
        """
        Constructor for the RestoreIntermediatePass.

        Args:
            project_directory (str): Path to the checkpoint block files. This
                directory must also contain a valid "structure.pickle" file.

            load_blocks (bool): If True, blocks in the project directory will
                be loaded to the block_list during the constructor. Otherwise
                the user must explicitly call load_blocks() themselves. Defaults
                to True.

        Raises:
            ValueError: If `project_directory` does not exist or if
                `structure.pickle` is invalid.
        """
        self.proj_dir = project_directory
        self.do_something = True
        self.is_block = is_block
        self.read_as_qasm = read_as_qasm

        if self.is_block:
            return

        if not exists(self.proj_dir) or not exists(self.proj_dir + '/structure.pickle'):
            return

        self.block_list: list[str] = []
        if load_blocks:
            self.reload_blocks()

    def reload_blocks(self) -> None:
        """
        Updates the `block_list` variable with the current contents of the
        `proj_dir`.

        Raises:
            ValueError: if there are more block files than indices in the
            `structure.pickle`.
        """
        files = listdir(self.proj_dir)
        all_block_list = [f for f in files if 'block_' in f]
        if self.read_as_qasm:
            end = ".qasm"
        else:
            end = ".pickle"

        self.block_list = [f for f in all_block_list if end in f]
        # if len(self.block_list) > len(self.structure):
        #     raise ValueError(
        #         'More block files than indicies in `structure.pickle`',
        #     )

    async def run(self, orig_circuit: Circuit, data: PassData) -> None:
        """
        Perform the pass's operation, see BasePass for more info.

        Raises:
            ValueError: if a block file and the corresponding index in
                `structure.pickle` are differnt lengths.
        """
        if not exists(self.proj_dir + '/structure.pickle'):
            if (self.is_block):
                raise("Project has not been made yet!")
            # Do nothing
            return

        with open(self.proj_dir + '/structure.pickle', 'rb') as f:
            self.structure = pickle.load(f)

        if not isinstance(self.structure, list):
            raise TypeError('The provided `structure.pickle` is not a list.')

        print("Trying next")

        if self.is_block:
            block_id = data["block_num"]
            print(block_id)
            block_skeleton = self.proj_dir + '/block_' + str(block_id)
            print(block_skeleton)
            with open(f'{block_skeleton}.pickle', 'rb') as f:
                circ = pickle.load(f)
                orig_circuit.become(circ)
            print("Opened circuit")
            with open(f'{block_skeleton}.data', 'rb') as f:
                new_data = pickle.load(f)
                data.update(new_data)
            print("Opened daata")
            return
        

        # If the circuit is empty, just append blocks in order
        with open(self.proj_dir + '/data.pickle', 'rb') as data_f:
            new_data = pickle.load(data_f)
            data.update(new_data)

        circuit = Circuit(orig_circuit.num_qudits)
        # if circuit.depth == 0:
        #     print("circuit is empty")

        # print(self.block_list)

        for block in self.block_list:
            # Get block
            block_num = int(findall(r'\d+', block)[0])
            with open(self.proj_dir + '/' + block, 'rb') as f:
                #block_circ = OPENQASM2Language().decode(f.read())
                block_circ = pickle.load(f)
            # Get location
            block_location = self.structure[block_num]
            if block_circ.num_qudits != len(block_location):
                print("This is the error")
                raise ValueError(
                    f'{block} and `structure.pickle` locations are '
                    'different sizes.',
                )
            # Append to circuit
            circuit.append_circuit(block_circ, block_location, as_circuit_gate=True)
        # Check if the circuit has been partitioned, if so, try to replace
        # blocks

        orig_circuit.become(circuit)

def default_collection_filter(op: Operation) -> bool:
    return isinstance(
        op.gate, (
            CircuitGate,
            ConstantUnitaryGate,
            VariableUnitaryGate,
            PauliGate,
        ),
    )