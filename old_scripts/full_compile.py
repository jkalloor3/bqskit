from bqskit.ir.circuit import Circuit
from sys import argv

from pathlib import Path
import os
import glob

from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler

from util import load_block, normalized_gp_frob_cost

from bqskit import enable_logging

base_dir = "/pscratch/sd/j/jkalloor/bqskit"
input_dir = f"{base_dir}/bad_blocks"
output_dir = f"{input_dir}_opt3"
enable_logging(True)

def full_compile(circ_name: str, block_num: str, tol: float = 10e-10):
    compiler = Compiler(num_workers=128)
    circ_file = load_block(circ_name, block_num)
    circ = Circuit.from_file(circ_file)
    out_circ: Circuit = compile(circ, optimization_level=3, max_synthesis_size=3, synthesis_epsilon=tol, error_sim_size=8, compiler=compiler)
    cost = normalized_gp_frob_cost(circ.get_unitary(), out_circ.get_unitary())
    print("Cost: ", cost, flush=True)
    if cost < 2e-8:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("Writing optimized circuit", flush=True)
        file_name = Path(circ_file).name
        new_circ_file = os.path.join(output_dir, file_name)
        out_circ.save(new_circ_file)
        print(f"Optimized {circ_file}", flush=True)

# Circ 
if __name__ == '__main__':
    circ_name = argv[1]
    
    files = glob.glob(os.path.join(input_dir, f"*{circ_name}*.qasm"))


    for file in files:
        file_name = Path(file).name.split(".")[0]
        parts = file_name.split("_")
        block_num = parts[-1]
        circ_name = "_".join(parts[:-1])
        new_circ_file = os.path.join(output_dir, file_name)
        if os.path.exists(new_circ_file):
            continue
        print(file, flush=True)
        full_compile(circ_name, block_num)