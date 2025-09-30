from bqskit.ir.circuit import Circuit
from sys import argv
import os
import glob
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, NOOPPass
from util import JiggleEnsemblePass
from util import GenerateProbabilityPass, FixAnglesPass
from util import CheckEnsembleQualityPass
from util import get_circ_names, get_block_names

base_checkpoint_dir = "block_checkpoints_final_paper_zs_twirl/"

def get_final_workflow(circ_name: str, tol: float) -> WorkflowLike | None:
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol)
    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=10000, 
                                  use_ensemble=False,
                                  use_calculated_error=False,
                                  checkpoint_extra_str="",
                                  count_t=True,
                                  flood_circ=False,
                                  do_u3_perturbation=True)
    workflow = [
        FixAnglesPass(10),
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[]),
        jiggle_pass, # To reload jiggled unitaries
        CheckEnsembleQualityPass(False),
        GenerateProbabilityPass()
    ]
    return workflow

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

    num_workers = 128
    compiler = Compiler(num_workers=num_workers)
    
    workflow_ind = 0
    ids = []
    for _, circ_file, _ in circ_data:
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
    return

def find_file(circ_name: str, block_num: str) -> tuple[str, str]:
    circ_name = f"{circ_name}_{block_num}"

    circ_file = f"good_blocks/{circ_name}.qasm"
    if os.path.exists(circ_file):
        return circ_name, circ_file
    circ_file = f"bad_blocks/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        raise Exception("File not found")

    return circ_name, circ_file

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
    jiggle_file = os.path.join(checkpoint_dir, "*5_jiggles*.npy")
    jiggle_files = glob.glob(jiggle_file)
    if len(jiggle_files) == 0:
        return False, False, ""
    # Try to get extra_str from jiggle files
    extra_str = jiggle_files[0].split("5_jiggles_")[1]
    # Remove everything after .
    extra_str = extra_str.split(".npy")[0]
    return False, True, extra_str


def get_circ_data(circ_name: str, block_num: str | int, tol: float) -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them

    if circ_name == "all_circs":
        # Get all the circ names, block_nums and tols which have a .npy file
        # but no ensemble_final.qasms file
        circ_names = get_circ_names()
        circ_data = []
        for name in circ_names:
            blocks = get_block_names(name)
            for block in blocks:
                circ_name, circ_file = find_file(name, block)
                finished, _ , _ = check_if_finished(circ_name, tol)
                if finished:
                    # Do not add this
                    continue
                else:
                    circ_data.append((circ_name, circ_file, tol))
    else:
        if block_num == "all_blocks":
            # Get all blocks
            good_circ_files = glob.glob(f"good_blocks/{circ_name}_*.qasm")
            bad_circ_files = glob.glob(f"bad_blocks/{circ_name}_*.qasm")
            all_circ_files = good_circ_files + bad_circ_files
            block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
            circ_data = []
            for i, block_num in enumerate(block_nums):
                circ_file = all_circ_files[i]
                circ_data.append((f"{circ_name}_{block_num}", circ_file, tol))
            return circ_data
        else:
            circ_name, circ_file = find_file(circ_name, block_num)
            return [(circ_name, circ_file, tol)]

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else ""
    tol = float(argv[3]) if len(argv) > 3 else 0.0
    circ_data = get_circ_data(circ_name, block_num, tol)
    print(circ_data)
    get_shortest_circuits(circ_data)