"""This module implements the WriteQASM pass"""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

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