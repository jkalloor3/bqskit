import os
import glob
import csv
from .common import load_block, get_block_names, get_circ_names, load_ensemble_cx_counts
from .fix_angles import FixAnglesPass
from .distance import normalized_gp_frob_cost
from bqskit.ir import Circuit
from bqskit.ir.gates import *

nisq_checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper"
clifft_checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper_clifft"

def count(c: Circuit, cliff_t: bool = False) -> int:
    if cliff_t:
        return c.num_params
    else:
        return c.count(CNOTGate())


def get_base_circ_count(circ_file: str, cliff_t: bool = False) -> int:
    circ = Circuit.from_file(circ_file)

    if cliff_t:
        FixAnglesPass.run_circ(circ, 10)

    return count(circ, cliff_t=cliff_t)


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
                csv_files = glob.glob(os.path.join(folder, f"*.csv"))
                block_done = block_done or len(csv_files) > 0
            all_blocks_done = all_blocks_done and block_done
        if all_blocks_done:
            finished_circs.append(circ_name)
    return finished_circs

def get_circ_block_dirs(circ_name: str, err_threshold: float, 
                        cliff_t: bool = False) -> list[tuple[str, tuple[str, str, float]]]:
    '''
    Takes in a circ name and total error, and calculates the 
    best combination of blocks that minimizes the count
    while being below the error threshold
    '''
    if cliff_t:
        checkpoint_dir = clifft_checkpoint_dir
    else:
        checkpoint_dir = nisq_checkpoint_dir

    folder_files = glob.glob(os.path.join(checkpoint_dir, f"{circ_name}_*"))

    block_names = get_block_names(circ_name)
    block_data = {}
    targets = {}
    for block_name in block_names:
        circ_file = load_block(circ_name, block_name)
        # Load Circuit
        circ = Circuit.from_file(circ_file)
        targets[block_name] = circ.get_unitary()
        init_count = get_base_circ_count(circ_file, cliff_t=cliff_t)
        print("Base count: ", init_count, flush=True)
        block_data[block_name] = [(0, init_count, ("orig", "", 0))]

    # Now read through opt3 blocks
    for block_name in ["2"]:
        circ_file = load_block(circ_name, block_name, extra="_opt3")
        circ = Circuit.from_file(circ_file)
        init_count = get_base_circ_count(circ_file, cliff_t=cliff_t)
        print("Opt 3 count: ", init_count, flush=True)
        initial_dist = normalized_gp_frob_cost(circ.get_unitary(), targets[block_name])
        block_data[block_name].append((initial_dist, init_count, ("_opt3", "", 0)))

    for folder_name in folder_files:
        tol = float(folder_name.split('_')[-1])
        block_num = folder_name.split('_')[-2]
        # Read CSV
        csv_files = glob.glob(os.path.join(folder_name, f"*.csv"))
        if len(csv_files) == 0:
            continue
        csv_file = csv_files[0]
        ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_final.qasms"))[0]
        reader = csv.DictReader(open(csv_file, 'r'))
        min_value = float("inf")  # Initialize to a large number

        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                min_value = min(min_value, float(row["Ratio"]))
        ratio = min_value
        final_threshold = (10 ** (-2 * tol)) * ratio
        # Get avg count of ensemble
        avg_count = load_ensemble_cx_counts(ensemble_file, cliff_t=cliff_t)
        block_data[block_num].append((final_threshold, avg_count, (circ_name, block_num, tol)))

    # Now for every block, try to select one folder name per block_num 
    # which minimizes the count and is below the threshold

    # Get all possible combinations of block_dirs
    block_nums = sorted(list(block_data.keys()))
    all_combos = block_data[block_nums[0]]
    all_combos = [(t, c, [d]) for t, c, d in all_combos]
    block_ind = 1
    while block_ind < len(block_nums):
        new_combos = []
        for orig_threshold, orig_count, datas in all_combos:
            for threshold, avg_count, data in block_data[block_nums[block_ind]]:
                new_item = (orig_threshold + threshold, orig_count + avg_count, datas + [data])
                new_combos.append(new_item)
        all_combos = new_combos
        block_ind += 1
    
    # Now filter out the ones that are above the error threshold
    all_combos = [x for x in all_combos if x[0] <= err_threshold]

    # Now get the one with the min count
    if len(all_combos) == 0:
        return []
    min_count = min([x[1] for x in all_combos])
    min_combos = [x for x in all_combos if x[1] == min_count]
    # print(min_combos)
    # Now get the folder names
    block_data = min_combos[0][2]

    print("Avg count: ", min_count)

    return [(block_nums[i], d) for i, d in enumerate(block_data)]


# if __name__ == "__main__":
    # print(get_circ_block_dirs("adder9", 0.3, False))
    # print(get_completed_blocks(False))