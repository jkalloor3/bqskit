"""This module implements the WriteQASM pass"""
from __future__ import annotations
import os

from bqskit.ir import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ScanPartitioner

class WriteQasmPass(BasePass):
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
            
            cc = circuit.copy()
            cc.unfold_all()
            qasm_str = cc.to("qasm")
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