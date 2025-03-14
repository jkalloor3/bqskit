from bqskit.ir.circuit import Circuit
from sys import argv
import os
import numpy as np
import glob
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass
from util import CollectAnglesPass

extra = "_tket"
base_checkpoint_dir = f"block_checkpoints_final_paper_clifft{extra}/"
good_block_folder = f"good_blocks{extra}/"
bad_block_folder = f"bad_blocks{extra}/"
NUM_UNIQUE_CIRCS = 250

def get_final_workflow(circ_name: str, tol: float,) -> WorkflowLike | None:
    # Check if already finished
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    print(f"Checkpoint Dir: {checkpoint_dir}", flush=True)
    workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[]),
        CollectAnglesPass()
    ]
    return workflow

def check_if_finished(circ_name: str, tol: float) -> tuple[bool, bool, str]:
    '''
    Given a circ_name and a tolerance, check if the ensemble is complete
    ret_1 -> completely finished
    ret_2 -> jiggle pass finished
    ret_3 -> extra_str
    
    '''
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    final_file = os.path.join(checkpoint_dir, "ensemble_final_rand_inds.npy")
    if os.path.exists(final_file):
        return True, True, ""
    # Check if there is a jiggle .npy file in the checkpoint dir for at least 5
    jiggle_file = os.path.join(checkpoint_dir, "*4_jiggles*.npy")
    jiggle_files = glob.glob(jiggle_file)
    if len(jiggle_files) == 0:
        return False, False, ""
    # Try to get extra_str from jiggle files
    extra_str = jiggle_files[0].split("4_jiggles_")[1]
    # Remove everything after .
    extra_str = extra_str.split(".npy")[0]
    return False, True, extra_str

def get_shortest_circuits(circ_data: list[tuple[str, str, float]]) -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
    workflows = [
        get_final_workflow(circ_name, tol)
        for circ_name, _, tol in circ_data
    ]

    num_workers = 4
    compiler = Compiler(num_workers=num_workers)
    
    workflow_ind = 0
    ids: list[int] = []

    for circ_name, circ_file, tol in circ_data:
        workflow = workflows[workflow_ind]
        workflow_ind += 1
        if workflow:
            circ = Circuit.from_file(circ_file)
            print("Original CNOT Count: ", circ.count(CNOTGate()), flush=True)
            ids.append(compiler.submit(circ, workflow))

    ind = 0
    for id in ids:
        compiler.result(id)
        print("Finished: ", circ_data[ind][0], flush=True)
        ind += 1
    compiler.close()
    return

def find_file(circ_name: str, block_num: str) -> tuple[str, str]:
    circ_name = f"{circ_name}_{block_num}"

    circ_file = f"{good_block_folder}/{circ_name}.qasm"
    if os.path.exists(circ_file):
        return circ_name, circ_file
    circ_file = f"{bad_block_folder}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        raise Exception(f"File not found for {circ_name}: {block_num}")

    return circ_name, circ_file

def get_circ_data() -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them
    tols = [3.0, 5.0]
    # Get all the circ names, block_nums and tols which have a .npy file
    # but no ensemble_final.qasms file
    circ_data = []
    print("Finding all circ data", flush=True)
    for dir in os.listdir(base_checkpoint_dir):
        parts = dir.split("_")
        block_num = parts[-2]
        tol = float(parts[-1])
        circ_name = "_".join(parts[:-2])
        circ_name, circ_file = find_file(circ_name, block_num)
        # print(f"Checking {circ_name} {block_num} {tol} {finished} {jiggle_finished}", flush=True)
        for tol in tols:
            finished, jiggle_finished, _ = check_if_finished(circ_name, tol)
            if not finished and jiggle_finished:
                circ_data.append((circ_name, circ_file, tol))
    if len(circ_data) > 20:
        circ_data = circ_data[:20]
        print("Limiting to 20 circs", flush=True)
    return circ_data


if __name__ == '__main__':
    circ_data = get_circ_data()
    get_shortest_circuits(circ_data)