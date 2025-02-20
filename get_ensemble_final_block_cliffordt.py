from bqskit.ir.circuit import Circuit
from sys import argv
import os
import glob
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, ToU3Pass
from bqskit.passes import ForEachBlockPass, ScanPartitioner
from util import JiggleEnsemblePass, CreateEnsemblePass, WriteQasmPass, CleanupBlockFiles
from ntro import NumericalTReductionPass
from util import LEAPSynthesisPass2, GenerateProbabilityPass, FixAnglesPass, UnFixTPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass, ConvertToZXZXZSimple

# enable_logging(True)
good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

extra = "_tket"
base_checkpoint_dir = f"block_checkpoints_final_paper_clifft{extra}/"
good_block_folder = f"good_blocks{extra}/"
bad_block_folder = f"bad_blocks{extra}/"
NUM_UNIQUE_CIRCS = 250

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

def get_ensemble_workflow(circ_name: str, tol: float, num_processes: int = 1) -> list:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = err_thresh * 0.01
    # small_block_size = 3
    block_size = 3

    slow_partitioner_passes = [
        ScanPartitioner(block_size=block_size),
    ]
    create_ensemble_pass = CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=False, 
            num_circs=NUM_UNIQUE_CIRCS,
            num_random_ensembles=2,
            solve_exact_dists=True,
            sort_by_t=True,
            checkpoint_extra_str=""
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 2,
        instantiate_options=good_instantiation_options,
        max_layer=14,
        max_psols=4
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=10000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  checkpoint_extra_str="",
                                  count_t=True,
                                  flood_circ=False,
                                  do_u3_perturbation=True)

    leap_workflow = [
        FixAnglesPass(15),
        UnFixTPass(),
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=slow_partitioner_passes),
        ForEachBlockPass(
            [
                synthesis_pass,
                FixAnglesPass(tol * 2 + 2, run_scan_sols=True),
                ConvertToZXZXZSimple(group=False),
                WriteQasmPass(write=False),
                NumericalTReductionPass(
                    full_loops=3,
                    success_threshold=err_thresh,
                    use_calculated_error=True),
                FixAnglesPass(tol * 2 + 2, run_scan_sols=True),
                ToU3Pass(ensemble=True, group=True),
                FixGlobalPhasePass(),
            ],
            allocate_error=True,
            skip_file="ensemble_0_.qasms"
        ),
        create_ensemble_pass,
        jiggle_pass,
        # CleanupBlockFiles(),
        CheckEnsembleQualityPass(True, shm_percentage=(1.0 / num_processes)),
        GenerateProbabilityPass(shm_percentage=(1.0 / num_processes)),
    ]
    return leap_workflow


def get_final_workflow(circ_name: str, tol: float, num_processes: int = 1) -> WorkflowLike | None:
    # Check if already finished
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    print(f"Checkpoint Dir: {checkpoint_dir}", flush=True)
    finished, jiggle_finished, _ = check_if_finished(circ_name, tol)
    if finished:
        print(f"Already finished {circ_name} {tol}", flush=True)
        return None
    if not jiggle_finished:
        print(f"Jiggle not finished {circ_name} {tol}", flush=True)
        return get_ensemble_workflow(circ_name, tol, num_processes)
    jiggle_pass = JiggleEnsemblePass()
    workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[]),
        jiggle_pass, # To reload jiggled unitaries
        CheckEnsembleQualityPass(True, shm_percentage=(1.0 / num_processes)),
        GenerateProbabilityPass(shm_percentage=(1.0 / num_processes))
    ]
    return workflow

def get_shortest_circuits(circ_data: list[tuple[str, str, float]]) -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''

    num_processes = len(circ_data)
    print(f"Num Processes: {num_processes}", flush=True)
    workflows = [
        get_final_workflow(circ_name, tol, num_processes)
        for circ_name, _, tol in circ_data
    ]

    num_workers = min(os.cpu_count(), 250)
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

    circ_file = f"{good_block_folder}/{circ_name}.qasm"
    if os.path.exists(circ_file):
        return circ_name, circ_file
    circ_file = f"{bad_block_folder}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        raise Exception(f"File not found for {circ_name}: {block_num}")

    return circ_name, circ_file

def get_circ_data(circ_name: str, block_num: str | int, tol: float) -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them
    if tol == -1.0:
        tols = [3.0, 5.0]
    else:
        tols = [tol]
    if circ_name == "all_probs":
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
            finished, jiggle_finished, _ = check_if_finished(circ_name, tol)
            # print(f"Checking {circ_name} {block_num} {tol} {finished} {jiggle_finished}", flush=True)
            if not finished and jiggle_finished:
                for tol in tols:
                    circ_data.append((circ_name, circ_file, tol))
        return circ_data
    
    else:
        if block_num == "all_blocks":
            # Get all blocks
            good_circ_files = glob.glob(f"{good_block_folder}/{circ_name}_*.qasm")
            bad_circ_files = glob.glob(f"{bad_block_folder}/{circ_name}_*.qasm")
            all_circ_files = good_circ_files + bad_circ_files
            block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
            circ_data = []
            for i, block_num in enumerate(block_nums):
                name, circ_file = find_file(circ_name, block_num)
                circ_file = all_circ_files[i]
                for tol in tols:
                    circ_data.append((name, circ_file, tol))
            return circ_data
        else:
            circ_name, circ_file = find_file(circ_name, block_num)
            circ_data = []
            for tol in tols:
                circ_data.append((circ_name, circ_file, tol))
            return circ_data

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else ""
    tol = float(argv[3]) if len(argv) > 3 else -1.0
    circ_data = get_circ_data(circ_name, block_num, tol)
    print(circ_data)
    get_shortest_circuits(circ_data)