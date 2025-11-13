import glob
import os
import csv
import pickle

from bqskit.ir.circuit import Circuit

from .common import DEFAULT_RATIO_LIMIT, get_block_names, good_block_dir, bad_block_dir, get_circ_dir
from .counter import GateCounter



def check_if_finished(circ_name: str, 
                      tol: float, 
                      cliff_t: bool = False) -> tuple[bool, bool]:
    '''
    Returns if all blocks have been processed for a circ_name, tol.

    return_1 - True if all blocks have been processed and QP has been run
    return_2  - True if all blocks have been processed minus QP and Check Ensemble
    Quality

    Note: If blocks do not exist, return_1 and return_2 will both be True
    '''
    # Get all block nums for a circ_name
    good_circ_files = glob.glob(f"{good_block_dir}/{circ_name}_*.qasm")
    bad_circ_files = glob.glob(f"{bad_block_dir}/{circ_name}_*.qasm")
    all_circ_files = good_circ_files + bad_circ_files

    if len(all_circ_files) == 0:
        print("No blocks found for circ", circ_name, flush=True)
        return True, True
    
    block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
    # Check if all blocks have been processed
    ret_1 = True
    ret_2 = True
    for block_num in block_nums:
        circ_dir = get_circ_dir(circ_name, block_num, tol, cliff_t)
        full_path = f"{circ_dir}/ensemble_final_rand_ind*.npy"
        jiggle_path = f"{circ_dir}/ensemble_0_jiggles*.npy"
        rand_ind_files = glob.glob(full_path)
        jiggle_files = glob.glob(jiggle_path)
        ret_1 = ret_1 and (len(rand_ind_files) > 0)
        ret_2 = ret_2 and (len(jiggle_files) > 0)
    return ret_1, ret_2


def check_good(csv_file: str, max_ratio: float, bias: bool = False) -> float:
    """
    Check if the csv file has a good ratio for the given tolerance.
    """
    final_ratio = float("inf")

    if not os.path.exists(csv_file):
        return final_ratio, float("inf")
    
    best_count = float("inf")

    bias_methods = ["Uniform", "NTRO Default", "Default Jiggle"]

    smallest_ratio = float("inf")

    with open(csv_file, 'r') as csv_file_obj:
        reader = csv.DictReader(csv_file_obj)
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                # print(list(row.keys()), csv_file)
                if bias and row["Probability Method"] not in bias_methods:
                    # If we are plotting bias, only look at GG distribution
                    continue
                new_ratio = float(row["Ratio"])
                count = float(row.get("Count", float("inf")))
                if new_ratio < max_ratio and count < best_count:
                    best_count = count
                    final_ratio = new_ratio
                smallest_ratio = min(smallest_ratio, new_ratio)

    if final_ratio == float("inf"):
        assert best_count == float("inf")
        final_ratio = smallest_ratio

    return final_ratio, best_count


def get_file_names(large_checkpoint_dir, 
                   small_block_num: str,
                   no_qp: bool = False,
                   ratio_text: str = "") -> tuple[str, str, str, str, str]:
    small_checkpoint_dir = os.path.join(large_checkpoint_dir, f"block_{small_block_num}")

    if no_qp:
        extra_str = "_no_qp"
    else:
        extra_str = "_FINAL"
    # Try outputs of newest passes
    final_probs_file = os.path.join(small_checkpoint_dir, f"ensemble_final_probs_fw{extra_str}.npy")
    if os.path.exists(final_probs_file):
        ensemble_file = os.path.join(small_checkpoint_dir, f"ensemble_final_fw{extra_str}.qasms")
        jiggle_file = os.path.join(small_checkpoint_dir, f"ensemble_final_jiggle_fw{extra_str}.npy")
        cache_file = os.path.join(small_checkpoint_dir, f"ensemble_final_cache_fw{extra_str}.pkl")
        csv_file = os.path.join(large_checkpoint_dir, f"block_{small_block_num}_fw{ratio_text}.csv")
        return ensemble_file, jiggle_file, cache_file, final_probs_file, csv_file

    # Otherwise, we do not have the newest set of files, so return the old ones
    csv_file = os.path.join(large_checkpoint_dir,
                             f"block_{small_block_num}_fw.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(large_checkpoint_dir, 
                                 f"block_{small_block_num}.csv")
    ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_0__fw.qasms")
    jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_0_jiggles__fw.npy")
    probs_file = os.path.join(small_checkpoint_dir, "ensemble_0_probs__fw.npy")
    cache_file = os.path.join(small_checkpoint_dir, "ensemble_0_cache__fw.pkl")
    return ensemble_file, jiggle_file, cache_file, probs_file, csv_file

def get_good_blocks(circ_name: str, 
                    tol: float, 
                    cliff_t : bool,
                    checkpoint_folder_form: str,
                    partitioned_data: dict) -> dict[str, set[str]]:
    '''
    Get the good blocks for each circuit from the checkpoint folder.
    
    checkpoint_folder_form: The form of the checkpoint folder path.
    '''
    good_blocks = {}
    counter = GateCounter(est=False, cache_file=None)

    if tol == 1.0:
        ratio_limit = DEFAULT_RATIO_LIMIT / 10
        ratio_str = "_2"
    else:
        ratio_limit = DEFAULT_RATIO_LIMIT
        ratio_str = "_20"
        
    num_good_blocks = 0
    for large_block_num in get_block_names(circ_name, extra="_tket"):
        good_blocks[large_block_num] = set()
        small_block_circs, _ = partitioned_data[large_block_num]
        large_block_dir = checkpoint_folder_form.format(large_block_num=large_block_num)
        # print(f"Large Block Dir: {large_block_dir}", flush=True)
        for small_block_num, small_circ in small_block_circs.items():
            # print(f"Small Block: {small_block_num} in {large_block_num}", flush=True)
            csv_file = get_file_names(large_block_dir, small_block_num, 
                                      ratio_text=ratio_str)[-1]
            ratio, count = check_good(csv_file, ratio_limit)
            good = ratio < ratio_limit
            if cliff_t:
                original_count = counter.count_t(small_circ, target_error=(10 ** (- 2 * tol)))
            else:
                # Count CNOTs in the circuit
                original_count = counter.count_cx(small_circ)
            if not good or count > original_count:
                continue
            else:
                # Use block
                num_good_blocks += 1
                good_blocks[large_block_num].add(small_block_num)

        if len(good_blocks[large_block_num]) == 0:
            good_blocks.pop(large_block_num)
    print(f"Found {num_good_blocks} good blocks for circuit {circ_name}", flush=True)
    return good_blocks, num_good_blocks