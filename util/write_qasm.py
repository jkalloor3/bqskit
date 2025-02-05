"""This module implements the WriteQASM pass"""
from __future__ import annotations
import os
import glob
from pathlib import Path

from bqskit.ir import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ScanPartitioner

class WriteQasmPass(BasePass):
    def __init__(self, checkpoint_dir: str = None, write: bool = False):
        self.default_dir = checkpoint_dir
        self.write = write

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        if "checkpoint_dir" in data:
            # checkpoint_dir = data["checkpoint_dir"]
            if "checkpoint_data_file" in data:
                file_name = str(data["checkpoint_data_file"]).replace(".data", ".qasm")
            else:
                checkpoint_dir = data["checkpoint_dir"]
                file_name = f"{checkpoint_dir}/circuit.qasm"
        else:
            block_num = data.get("block_num", "-1")
            file_name = f"{self.default_dir}/block_{block_num}.qasm"
            
        cc = circuit.copy()
        cc.unfold_all()
        qasm_str = cc.to("qasm")
        if self.write:
            print("Writing Block to : ", file_name, flush=True)
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            f.write(qasm_str)

class ReplaceWithQasmPass(BasePass):

    def __init__(self, partition_size: int = 3):
        super().__init__()
        self.partition_size = 3

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        if "checkpoint_dir" in data:
            # checkpoint_dir = data["checkpoint_dir"]
            if "checkpoint_data_file" in data:
                file_name = str(data["checkpoint_data_file"]).replace(".data", "_decomposed.qasm")
            else:
                return

            if os.path.exists(file_name):
                new_circ = Circuit.from_file(file_name)
                print("Replacing circuit. New Gate Counts: ", new_circ.gate_counts, flush=True)
                print("Partitioning!", flush=True)
                await ScanPartitioner(self.partition_size).run(new_circ, data)
                print("After partitioning New Gate Counts: ", new_circ.gate_counts, flush=True)
                data["min_cnot_count"] = new_circ.count(CNOTGate()) // 2
                circuit.become(new_circ)

class CleanupBlockFiles(BasePass):
    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        if "checkpoint_dir" in data:
            # checkpoint_dir = data["checkpoint_dir"]
            if "checkpoint_data_file" in data:
                file_name = os.path.join(data["checkpoint_dir"], "block_*")
                file_names = glob.glob(file_name)
                for file_name in file_names:
                    if os.path.isdir(file_name):
                        # Remove everything in directory
                        for f in os.listdir(file_name):
                            os.remove(os.path.join(file_name, f))
                        # Remove directory
                        os.rmdir(file_name)
                    else:
                        # Remove file
                        os.remove(file_name)