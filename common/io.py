import os
import glob
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir import Circuit
# from util.gg import gg_gate_def, GridSynthGate

base_bqskit_dir = "/pscratch/sd/j/jkalloor/ensemble_paper/bqskit"
good_block_dir = f"{base_bqskit_dir}/good_blocks"
bad_block_dir = f"{base_bqskit_dir}/bad_blocks"
base_checkpoint_dir = f"{base_bqskit_dir}/block_checkpoints_final_paper"

qlang = OPENQASM2Language()

def load_block(circ_name, block_num, extra="") -> str:
    circ_name = f"{circ_name}_{block_num}"
    circ_file = f"good_blocks{extra}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        circ_file = f"bad_blocks{extra}/{circ_name}.qasm"
    return circ_file

def load_ensemble(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    # print("SPlit String", flush=True)
    circs = [qlang.decode(qasm) for qasm in qasms]
    # print("Decoded", flush=True)
    return circs

def get_block_names(circ_name: str, extra: str= "") -> list[str]:
    good_circ_files = glob.glob(f"{good_block_dir}{extra}/{circ_name}_*.qasm")
    bad_circ_files = glob.glob(f"{bad_block_dir}{extra}/{circ_name}_*.qasm")
    all_circ_files = good_circ_files + bad_circ_files

    block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
    return block_nums