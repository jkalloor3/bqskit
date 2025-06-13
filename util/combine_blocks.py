import os
import glob
import csv
import numpy as np
from bqskit.ir import Circuit
from .common import load_block, get_block_names, get_circ_names
from .counter import (load_avg_ensemble_counts_est, load_avg_ensemble_counts_full, get_circ_counts)

base_dir = "/home/jkalloor/bqskit"
nisq_checkpoint_dir = f"{base_dir}/block_checkpoints_final_paper"
clifft_checkpoint_dir = f"{base_dir}/block_checkpoints_final_paper_clifft_tket_prev"


frob_factor = lambda dim: np.sqrt(dim * 2)

def get_completed_blocks(cliff_t: bool = False) -> list[str]:
    # Check if there is at least one completed ensemble for each
    # block in a circ name
    if cliff_t:
        checkpoint_dir = clifft_checkpoint_dir
    else:
        checkpoint_dir = nisq_checkpoint_dir
    
    finished_circs = []
    circ_names = get_circ_names()
    for circ_name in circ_names:
        block_names = get_block_names(circ_name)
        all_blocks_done = True
        for block_name in block_names:
            folders = glob.glob(os.path.join(checkpoint_dir, f"{circ_name}_{block_name}_*"))
            if len(folders) == 0:
                all_blocks_done = False
                break
            block_done = True
            for folder in folders:
                # csv file
                csv_files = glob.glob(os.path.join(folder, f".qasms"))
                block_done = block_done or len(csv_files) > 0
            all_blocks_done = all_blocks_done and block_done
        if all_blocks_done:
            finished_circs.append(circ_name)
    return finished_circs

def get_circ_data(circ_name: str, err_threshold: float, use_base: bool = True, 
                    count_rz: bool = False, 
                    count_t: bool = False) -> list[tuple[str, tuple[str, 
                                                                    str, 
                                                                    float]]]:
    '''
    Takes in a circ name and total error, and calculates the 
    best combination of blocks that minimizes the count
    while being below the error threshold
    '''
    if count_t or count_rz:
        checkpoint_dir_1 = clifft_checkpoint_dir
        # checkpoint_dir_3 = clifft_checkpoint_dir + "_tket"
        checkpoint_dir_2 = f"{base_dir}/block_checkpoints_final_paper_clifft_tket_nou3"
    else:
        checkpoint_dir_1 = nisq_checkpoint_dir
        checkpoint_dir_2 = nisq_checkpoint_dir + "_tket"

    # print("Checkpoint dir: ", checkpoint_dir)
    # print("Circ name: ", circ_name)
    folder_files = glob.glob(os.path.join(checkpoint_dir_1, f"{circ_name}_*"))
    folder_files_2 = glob.glob(os.path.join(checkpoint_dir_2, f"{circ_name}_*"))
    # folder_files_3 = glob.glob(os.path.join(checkpoint_dir_3, f"{circ_name}_*"))
    folder_files_3 = []
    folder_files = folder_files + folder_files_2 + folder_files_3

    block_names = get_block_names(circ_name, extra="_tket")
    block_data = {}
    # print("Precision: ", precision)
    circ_files_orig = [load_block(circ_name, block_name) for block_name in block_names]
    circ_files_tket = [load_block(circ_name, block_name, extra="_tket") for block_name in block_names]
    # print("Circ files tket: ", circ_files_tket)
    if use_base:
        orig_counts = get_circ_counts(circ_files_orig, 
                                            count_t=count_t, 
                                            count_rz=count_rz, 
                                            target_error=err_threshold)
        tket_counts = get_circ_counts(circ_files_tket, 
                                            count_t=count_t, 
                                            count_rz=count_rz, 
                                            target_error=err_threshold)
        
        # print("Orig counts: ", orig_counts)
        # print("TKET counts: ", tket_counts)

    for i, block_name in enumerate(block_names):
        block_counts = []
        if use_base:
            orig_count = orig_counts[i]
            tket_count = tket_counts[i]
            if orig_count < tket_count:
                file_d = (circ_files_orig[i], None)
                block_counts.append((0, orig_count, ("orig", "", 0), file_d))
            else:
                file_d_tket = (circ_files_tket[i], None)
                block_counts.append((0, tket_count, ("tket", "", 0), file_d_tket))
        block_data[block_name] = block_counts
    

    num_qubits = {}
    for block_name in block_names:
        if os.path.exists(circ_files_orig[i]):
            circ = Circuit.from_file(circ_files_orig[i])
        elif os.path.exists(circ_files_tket[i]):
            circ = Circuit.from_file(circ_files_tket[i])
        num_qubits[block_name] = circ.num_qudits

    # print(folder_files)

    for folder_name in folder_files:
        tol = float(folder_name.split('_')[-1])
        block_num = folder_name.split('_')[-2]
        # Read CSV
        csv_files = glob.glob(os.path.join(folder_name, f"*.csv"))

        extra = ""
        if "_tket" in folder_name:
            extra = "_tket"

        if len(csv_files) == 0:
            # Just pick ensemble 0
            ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_0_.qasms"))
            if len(ensemble_file) == 0:
                continue
            ensemble_file = ensemble_file[0]
            params_file = None
            if count_t:
                # Get params file as well
                params_file = glob.glob(os.path.join(folder_name, f"ensemble_0_jiggles_.npy"))
                if len(params_file) == 0:
                    continue
                params_file = params_file[0]
            dim = 2 ** num_qubits[block_num]
            if tol > 4.0:
                add_factor = 100 # Bad scaling for smaller errors
            else:
                add_factor = 1
            final_threshold = (10 ** (-2 * tol)) * frob_factor(dim) * add_factor
        else:
            csv_file = csv_files[0]
            ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_final.qasms"))[0]
            if count_t:
                # Get params file as well
                params_file = glob.glob(os.path.join(folder_name, f"ensemble_final_jiggle.npy"))[0]
            reader = csv.DictReader(open(csv_file, 'r'))
            final_threshold = (10 ** -tol)
            for row in reader:
                if "Norm. Bias" in row:  # Check if the column value is not empty
                    final_threshold = min(final_threshold, float(row["Norm. Bias"]))
        avg_count = load_avg_ensemble_counts_est(ensemble_file, params_file, 
                                                 err_threshold, 
                                                 count_t=count_t, 
                                                 count_rz=count_rz)
        # print("Block num: ", block_num, " Avg count: ", avg_count, "Final threshold: ", final_threshold)
        file_d = (ensemble_file, params_file)
        block_data[block_num].append((final_threshold, avg_count, 
                                      (circ_name, block_num, tol, extra), 
                                      file_d))

    # Now for every block, try to select one folder name per block_num 
    # which minimizes the count and is below the threshold
    # print("Block data: ", block_data)
    # Get all possible combinations of block_dirs
    block_nums = sorted(list(block_data.keys()))
    # print(block_data)
    if len(block_nums) == 0:
        return [], 0, 0
    all_combos = block_data[block_nums[0]]
    all_combos = [(t, c, [d], [f]) for t, c, d, f in all_combos]
    block_ind = 1
    while block_ind < len(block_nums):
        new_combos = []
        for orig_threshold, orig_count, datas, files, in all_combos:
            for threshold, avg_count, data, file in block_data[block_nums[block_ind]]:
                new_item = (orig_threshold + threshold, orig_count + avg_count, datas + [data], files + [file])
                new_combos.append(new_item)
        all_combos = new_combos
        block_ind += 1
    
    # Now filter out the ones that are above the error threshold
    all_combos = [x for x in all_combos if x[0] <= err_threshold]

    # Now get the one with the min count
    if len(all_combos) == 0:
        return [], 0
    min_count = min([x[1] for x in all_combos])
    min_combos = [x for x in all_combos if x[1] == min_count]
    # print(min_combos)
    # Now get the actual count
    final_data = min_combos[0][2]
    final_files = min_combos[0][3]

    # Get actual count
    # print("Doing Final Count")
    actual_counts = [load_avg_ensemble_counts_full(e_file, p_file, err_threshold,  
                            count_t=count_t, count_rz=count_rz) for e_file, p_file in final_files]
    min_count = np.sum(actual_counts)

    if use_base:
        tket_count = min(np.sum(orig_counts), np.sum(tket_counts))
        return [(block_nums[i], d) for i, d in enumerate(final_data)], min_count, tket_count
    else:
        return [(block_nums[i], d) for i, d in enumerate(final_data)], min_count


def get_circ_data_basic(circ_name: str, 
                        err_threshold: float, 
                        count_rz: bool = False, 
                        count_t: bool = False,
                        use_cache: bool = False) -> tuple[int, int]:
    '''
    Takes in a circ name and total error, and calculates the 
    best combination of blocks that minimizes the count
    while being below the error threshold
    '''
    if count_t or count_rz:
        checkpoint_dir_1 = clifft_checkpoint_dir
        checkpoint_dir_2 = clifft_checkpoint_dir +"_tket"
    else:
        checkpoint_dir_1 = nisq_checkpoint_dir
        checkpoint_dir_2 = nisq_checkpoint_dir + "_tket"

    # print("Checkpoint dir: ", checkpoint_dir)
    # print("Circ name: ", circ_name)
    folder_files = glob.glob(os.path.join(checkpoint_dir_1, f"{circ_name}_*"))
    folder_files_2 = glob.glob(os.path.join(checkpoint_dir_2, f"{circ_name}_*"))
    # folder_files_3 = glob.glob(os.path.join(checkpoint_dir_3, f"{circ_name}_*"))
    folder_files_3 = []
    folder_files = folder_files + folder_files_2 + folder_files_3

    block_names = get_block_names(circ_name)
    block_data = {}
    # print("Precision: ", precision)
    circ_files_orig = [load_block(circ_name, block_name) for block_name in block_names]
    circ_files_tket = [load_block(circ_name, block_name, extra="_tket") for block_name in block_names]
    # print("Circ files tket: ", circ_files_tket)
    orig_counts = get_circ_counts(circ_files_orig, 
                                        count_t=count_t, 
                                        count_rz=count_rz, 
                                        target_error=err_threshold)
    tket_counts = get_circ_counts(circ_files_tket, 
                                        count_t=count_t, 
                                        count_rz=count_rz, 
                                        target_error=err_threshold) 

    for i, block_name in enumerate(block_names):
        orig_count = orig_counts[i]
        tket_count = tket_counts[i]
        if orig_count < tket_count:
            block_data[block_name] = orig_count
        else:
            block_data[block_name] = tket_count

    orig_count = sum(block_data.values())

    num_qubits = {}
    for block_name in block_names:
        if os.path.exists(circ_files_orig[i]):
            circ = Circuit.from_file(circ_files_orig[i])
            num_qubits[block_name] = circ.num_qudits
        elif os.path.exists(circ_files_tket[i]):
            circ = Circuit.from_file(circ_files_tket[i])
            num_qubits[block_name] = circ.num_qudits
        else:
            num_qubits[block_name] = 8

    # print(folder_files)

    for folder_name in folder_files:
        tol = float(folder_name.split('_')[-1])
        block_num = folder_name.split('_')[-2]
        # Read CSV
        csv_files = glob.glob(os.path.join(folder_name, f"*.csv"))
        params_file = None
        cache_file = None
        if len(csv_files) == 0:
            # Just pick ensemble 0
            ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_0_.qasms"))
            if len(ensemble_file) == 0:
                continue
            ensemble_file = ensemble_file[0]
            if count_t:
                # Get params file as well
                params_file = glob.glob(os.path.join(folder_name, f"ensemble_0_jiggles_.npy"))
                if len(params_file) == 0:
                    continue
                params_file = params_file[0]
                if use_cache:
                    cache_file = glob.glob(os.path.join(folder_name, f"ensemble_0_cache.pkl"))
                    if len(cache_file) == 0:
                        continue
                    cache_file = cache_file[0]
            dim = 2 ** num_qubits[block_num]
            if tol > 4.0:
                add_factor = 100 # Bad scaling for smaller errors
            else:
                add_factor = 1
            final_threshold = (10 ** (-2 * tol)) * frob_factor(dim) * add_factor
        else:
            csv_file = csv_files[0]
            ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_final.qasms"))[0]
            if count_t:
                # Get params file as well
                params_file = glob.glob(os.path.join(folder_name, f"ensemble_final_jiggle.npy"))[0]
                if use_cache:
                    cache_file = glob.glob(os.path.join(folder_name, f"ensemble_0_cache.pkl"))[0]
            reader = csv.DictReader(open(csv_file, 'r'))
            final_threshold = (10 ** -tol)
            for row in reader:
                if "Norm. Bias" in row:  # Check if the column value is not empty
                    final_threshold = min(final_threshold, float(row["Norm. Bias"]))

        if not use_cache:
            assert cache_file is None

        if final_threshold < err_threshold:
            avg_count = load_avg_ensemble_counts_est(ensemble_file, params_file, 
                                                        err_threshold, 
                                                        count_t=count_t, 
                                                        count_rz=count_rz)
            # print("Avg count: ", avg_count)
            if avg_count < block_data[block_num]:
                avg_count_actual = load_avg_ensemble_counts_full(ensemble_file, 
                                                                 params_file,
                                                                 cache_file,
                                                                err_threshold, 
                                                                count_t=count_t, 
                                                                count_rz=count_rz)
                block_data[block_num] = avg_count_actual

    # Sum block_data values
    total_count = sum(block_data.values())
    return orig_count, total_count



def get_counts(circ_name: str, err_threshold: float, cliff_t: bool = False) -> float:
    return get_circ_data(circ_name, err_threshold, cliff_t)[1]

def get_circ_block_dirs(circ_name: str, err_threshold: float, cliff_t: bool = False):
    return get_circ_data(circ_name, err_threshold, cliff_t)[0]