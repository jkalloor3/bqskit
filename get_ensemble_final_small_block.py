from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
import numpy as np
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, NOOPPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner, IfThenElsePass, PassPredicate
from util import JiggleEnsemblePass, CleanupBlockFiles
from util import  LEAPSynthesisPass2, SecondLEAPSynthesisPass, EnsScanningGateRemovalPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass
from util import GenerateProbabilityPass
from util import CreateEnsemblePass
from util import get_block_names, load_block


class CountPredicate(PassPredicate):
    def __init__(self, count_scan_sols: bool = False) -> None:
        super().__init__()
        self.count_scan_sols = count_scan_sols


    def get_truth_value(self, circuit, data):
        if self.count_scan_sols:
            # Check if any of the ensemble circuits have a count < circuit
            base_count = circuit.count(CNOTGate())
            for c, _ in data["scan_sols"]:
                if c.count(CNOTGate()) < base_count:
                    return True
            return False

        return circuit.count(CNOTGate()) < 30

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

SMALL_BLOCK_SIZE = 4
base_checkpoint_dir_form = "small_block_checkpoints_final_paper_{block_size}_more_cx{extra}"
NUM_UNIQUE_CIRCS = 250

def get_ensemble_workflow(circ_name: str, tol: float, extra: str = "") -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol) / 10

    extra_err_thresh = err_thresh * 0.01
    small_block_size = SMALL_BLOCK_SIZE
    print("Checkpoint Dir: ", checkpoint_dir, flush=True)
    print("Error Threshold: ", err_thresh, flush=True)

    has_qft = False
    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size, ignore_qft=has_qft),
    ]
    partitioner_passes = slow_partitioner_passes
    instantiation_options = good_instantiation_options
    
    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh / 5,
        partial_success_threshold=err_thresh / 5,
        max_layer_factor=1.01,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=10
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh / 5,
        partial_success_threshold=err_thresh / 5,
        max_layer_factor=1.01,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=20
    )

    deletion_pass = EnsScanningGateRemovalPass(
        success_threshold=err_thresh / 5,
        tree_depth=3,
        max_psols=20
    )

    extra_str = "_fw"

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=2000, 
                                  use_scan_sols=True,
                                  use_ensemble=False,
                                  use_calculated_error=False,
                                  jiggle_skew=0,
                                  count_t=False,
                                  do_u3_perturbation=True,
                                  flood_circ=True,
                                  checkpoint_extra_str=extra_str)

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                IfThenElsePass(
                    CountPredicate(),
                    synthesis_pass,
                    deletion_pass
                ),
                # Apply a second round of synthesis if count is bad still
                IfThenElsePass(
                    CountPredicate(count_scan_sols=True),
                    NOOPPass(),
                    second_synthesis_pass,
                ),
                # If counts are OK, then create ensemble and jiggle
                IfThenElsePass(
                    CountPredicate(count_scan_sols=True),
                    [
                        jiggle_pass,
                        GenerateProbabilityPass(eps=err_thresh * 10,
                            run_on_ensemble_0=True,
                            checkpoint_extra_str=extra_str),
                        CheckEnsembleQualityPass(False,
                            checkpoint_extra_str=extra_str,
                            zero_threshold=(err_thresh ** 2) / 10),
                    ]
                )
            ]
        ),
    ]
    return leap_workflow


def check_if_finished(circ_name: str, tol: float, extra: str = "") -> tuple[bool, bool, str]:
    '''
    Given a circ_name and a tolerance, check if the ensemble is complete
    
    ret_1 -> completely finished
    ret_2 -> jiggle pass finished
    ret_3 -> extra_str
    
    '''
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    final_file = os.path.join(checkpoint_dir, "data.csv")
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

def get_shortest_circuits(circ_data: list[tuple[str, str, float]], extra: str = "") -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
    workflows = [
        get_ensemble_workflow(circ_name, tol, extra)
        for circ_name, _, tol in circ_data
    ]

    num_workers = min(os.cpu_count(), 250)
    compiler = Compiler(num_workers=num_workers)
    
    workflow_ind = 0
    ids: list[int] = []

    for _, circ_file, _ in circ_data:
        workflow = workflows[workflow_ind]
        workflow_ind += 1
        if workflow:
            circ = Circuit.from_file(circ_file)
            ccount = circ.count(CNOTGate())
            if ccount > 3 and circ.num_qudits >= 3:
                print("Original CNOT Count: ", circ.count(CNOTGate()), 
                      flush=True)
                ids.append(compiler.submit(circ, workflow))
            else:
                print(f"Skipping {circ_file}, not enough CNOTs", flush=True)

    ind = 0
    for id in ids:
        compiler.result(id)
        print("Finished: ", circ_data[ind][0], flush=True)
        ind += 1
    compiler.close()
    print("Compiler Closed", flush=True)
    return

def find_file(circ_name: str, block_num: str, extra="") -> tuple[str, str]:
    circ_name = f"{circ_name}_{block_num}"

    if extra != "_tket":
        extra = ""

    circ_file = f"good_blocks{extra}/{circ_name}.qasm"
    if os.path.exists(circ_file):
        return circ_name, circ_file
    circ_file = f"bad_blocks{extra}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        raise Exception(f"File not found for {circ_name}: {block_num}")

    return circ_name, circ_file


def get_circ_data(circ_name: str, block_num: str | int, 
                  tol: float, extra: str = "") -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them
    if tol == -1.0:
        tols = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif tol == -2.0:
        tols = [1.0, 2.0, 3.0]
    elif tol == -3.0:
        tols = [4.0, 5.0]
    else:
        tols = [tol]
    ckpt_extra = extra
    
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
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
            if circ_name.startswith("shor"):
                continue
            circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
            for tol in tols:
                finished, jiggle_finished, _ = check_if_finished(circ_name, tol, extra=extra)
                if not finished and jiggle_finished:
                    circ_data.append((circ_name, circ_file, tol))

        if len(circ_data) > 40:
            circ_data = circ_data[:40]
            print("Limiting to 10 circs", flush=True)
        return circ_data
    
    else:
        if "blocks" in block_num:
            # all_blocks, first_half_blocks, second_half_blocks
            block_nums = get_block_names(circ_name, extra=extra)
            print("Number of blocks: ", len(block_nums), flush=True)
            if block_num.startswith("first_half"):
                block_nums = block_nums[:len(block_nums) // 2]
            elif block_num.startswith("second_half"):
                block_nums = block_nums[len(block_nums) // 2:]

            print("Running on blocks: ", len(block_nums), flush=True)
            circ_data = []
            for i, block_num in enumerate(block_nums):
                name, circ_file = find_file(circ_name, block_num, extra=extra)
                for tol in tols:
                    circ_data.append((name, circ_file, tol))

            # Only do 250 blocks
            if len(circ_data) > 250:
                rand_inds = np.random.choice(len(circ_data), 250, replace=False)
                circ_data = [circ_data[i] for i in rand_inds]
                print("Limiting to 250 blocks", flush=True)
            return circ_data
        else:
            circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
            circ_data = []
            for tol in tols:
                circ_data.append((circ_name, circ_file, tol))
            return circ_data

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else "all_blocks"
    tol = float(argv[3]) if len(argv) > 3 else -1.0
    extra = argv[5] if len(argv) > 5 else "_tket"
    np.random.seed(42)
    circ_data = get_circ_data(circ_name, block_num, tol, extra=extra)
    print(circ_data)
    get_shortest_circuits(circ_data, extra=extra)