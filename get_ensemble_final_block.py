from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner
from util import JiggleEnsemblePass, CleanupBlockFiles
from util import  LEAPSynthesisPass2, SecondLEAPSynthesisPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass
from util import GenerateProbabilityPass
from util import CreateEnsemblePass

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

base_checkpoint_dir = "block_checkpoints_final_paper/"
NUM_UNIQUE_CIRCS = 250

def get_ensemble_workflow(circ_name: str, tol: float) -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol)

    extra_err_thresh = err_thresh * 0.01
    small_block_size = 3
    print("Checkpoint Dir: ", checkpoint_dir, flush=True)
    print("Error Threshold: ", err_thresh, flush=True)

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
    ]
    partitioner_passes = slow_partitioner_passes
    instantiation_options = good_instantiation_options

    create_ensemble_pass = CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=False, 
            num_circs=NUM_UNIQUE_CIRCS,
            num_random_ensembles=3,
            solve_exact_dists=True,
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=10
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=5
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=10000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  jiggle_skew=0,
                                  do_u3_perturbation=True,
                                  flood_circ=True)

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                synthesis_pass,
                second_synthesis_pass,
                FixGlobalPhasePass(),
            ],
            allocate_error=True,
        ),
        create_ensemble_pass,
        CleanupBlockFiles(),
        jiggle_pass
    ]
    return leap_workflow


def get_final_workflow(circ_name: str, tol: float) -> WorkflowLike | None:
    # Check if already finished
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    final_file = os.path.join(checkpoint_dir, "ensemble_final_rand_inds.npy")
    if os.path.exists(final_file):
        print(f"Already finished {circ_name}:{tol}!", flush=True)
        return None
    # Check if there is a .npy file in the checkpoint dir
    jiggle_file = f"{checkpoint_dir}/*.npy"
    jiggle_files = glob.glob(jiggle_file)
    if len(jiggle_files) == 0:
        print(f"Jiggle Pass is not completed yet for {circ_name}:{tol}!", 
              flush=True)
        return get_ensemble_workflow(circ_name, tol)
    workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[]),
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


def get_circ_data(circ_name: str, block_num: str | int, tol: float) -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them

    if circ_name == "all_probs":
        # Get all the circ names, block_nums and tols which have a .npy file
        # but no ensemble_final.qasms file
        circ_data = []
        print("Finding all circ data", flush=True)
        for dir in os.listdir(base_checkpoint_dir):
            final_file = os.path.join(base_checkpoint_dir, dir, 
                                      "ensemble_final_rand_inds.npy")
            if os.path.exists(final_file):
                continue
            jiggle_file = os.path.join(base_checkpoint_dir, dir, "*.npy")
            jiggle_files = glob.glob(jiggle_file)
            if len(jiggle_files) == 0:
                continue
            parts = dir.split("_")
            circ_name = parts[0]
            block_num = parts[1]
            circ_name, circ_file = find_file(circ_name, block_num)
            tol = float(parts[2])
            circ_data.append((circ_name, circ_file, tol))
        return circ_data
    
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