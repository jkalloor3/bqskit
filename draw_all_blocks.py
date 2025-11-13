import os
import glob
from qiskit import QuantumCircuit
from bqskit.compiler import Compiler
from qiskit.visualization import circuit_drawer
from bqskit.passes import ScanPartitioner, ForEachBlockPass
from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
from pathlib import Path
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.lang.qasm2 import OPENQASM2Language

qasm_lang = OPENQASM2Language()

class DrawPartitionPass(BasePass):
     
     def __init__(self, output_path: str) -> None:
        """
        Construct a Draw Partition pass that saves the partitioned circuit image to output_path.
        """
        self.output_path = output_path
     
     async def run(self, circuit: Circuit, data: PassData) -> None:
        block_num = data["block_num"]
        file_name = os.path.join(self.output_path, f"partitioned_circuit_block_{block_num}.tex")
        # qcirc = bqskit_to_qiskit(partitioned_circuit)
        qasm = qasm_lang.encode(circuit)
        qcirc = QuantumCircuit.from_qasm_str(qasm)
        if circuit.num_operations < 12:
            file_name = os.path.join(self.output_path, f"partitioned_circuit_block_{block_num}_small.tex")
        circuit_drawer(qcirc, output='latex_source', filename=file_name)
        return




# Go through all folders in fixed_block_checkpoint_min. For every qasm, read it in
# and draw the circuit. Save the circuit as a png in the same folder.
def draw_all_blocks(circ_name: str, output_dir):
    # Get all good/bad blocks for this circuit
    good_block_files = glob.glob(os.path.join("good_blocks_tket", f"{circ_name}*.qasm"))
    bad_block_files = glob.glob(os.path.join("bad_blocks_tket", f"{circ_name}*.qasm"))

    good_output_dir = os.path.join(output_dir, "good_blocks")
    bad_output_dir = os.path.join(output_dir, "bad_blocks")

    compiler = Compiler()

    for file in good_block_files:
        circ = Circuit.from_file(file)
        basename = os.path.basename(file).split(".")[0]
        block_output_dir = os.path.join(good_output_dir, basename)
        Path(block_output_dir).mkdir(parents=True, exist_ok=True)
        print("Drawing good block circuits to ", block_output_dir, flush=True)
        workflow = [
            ScanPartitioner(4),
            ForEachBlockPass(
                DrawPartitionPass(block_output_dir)
            )
        ]
        compiler.compile(circ, workflow)    

    for file in bad_block_files:
        circ = Circuit.from_file(file)
        basename = os.path.basename(file).split(".")[0]
        block_output_dir = os.path.join(bad_output_dir, basename)
        Path(block_output_dir).mkdir(parents=True, exist_ok=True)
        print("Drawing bad block circuits to ", block_output_dir, flush=True)
        workflow = [
            ScanPartitioner(),
            ForEachBlockPass(
                DrawPartitionPass(block_output_dir)
            )
        ]
        compiler.compile(circ, workflow)

if __name__ == "__main__":
    circ_name = argv[1]
    # checkpoint_dir = "good_blocks_tket"
    # output_dir = "good_blocks_tket_pngs"
    # checkpoint_dir = "block_checkpoints_final_paper_tket"
    # checkpoint_dir = "block_checkpoints_final_paper_tket/adder9_0_0.8"
    print(f"Drawing circuits for {circ_name}")
    output_dir = circ_name + "_pngs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    draw_all_blocks(circ_name, output_dir)

