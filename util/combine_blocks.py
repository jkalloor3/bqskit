import os
import glob
import csv
import numpy as np
from .common import load_block, get_block_names, get_circ_names, load_ensemble_cx_counts
from .fix_angles import FixAnglesPass
from .distance import normalized_gp_frob_cost
from bqskit.ir import Circuit
from bqskit.compiler import Compiler
from bqskit.ir.gates import *
from .gg import GridSynthGate
from .convert_to_cliff import ConvertToZXZXZSimple

base_dir = "/pscratch/sd/j/jkalloor/bqskit"
nisq_checkpoint_dir = f"{base_dir}/block_checkpoints_final_paper"
clifft_checkpoint_dir = f"{base_dir}/block_checkpoints_final_paper_clifft_tket"

def count(c: Circuit, cliff_t: bool = False) -> int:
    if cliff_t:
        # GGs have 3 params but only 1 angle
        num_ggs = c.count(GridSynthGate())
        return c.num_params - 2 * num_ggs
    else:
        return c.count(CNOTGate())

def fix_angle_workflow(precision: int) -> list:
    return [
        FixAnglesPass(15),
        ConvertToZXZXZSimple(),
        FixAnglesPass(precision + 1),
    ]

def get_base_circ_counts(circ_files: list[tuple[str, int]] | list[str], cliff_t: bool = False) -> list[int]:
    circs = []
    if cliff_t:
        compiler = Compiler(num_workers=100)
        for circ_file, precision in circ_files:
            circ = Circuit.from_file(circ_file)
            workflow = fix_angle_workflow(precision)
            out_circ = compiler.compile(circ, workflow)
            circs.append(out_circ)
        compiler.close()
    else:
        for circ_file in circ_files:
            circs.append(Circuit.from_file(circ_file))

    return [count(c, cliff_t=cliff_t) for c in circs]

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

def get_circ_data(circ_name: str, err_threshold: float, 
                        cliff_t: bool = False, add_tket_count: bool = True) -> list[tuple[str, tuple[str, str, float]]]:
    '''
    Takes in a circ name and total error, and calculates the 
    best combination of blocks that minimizes the count
    while being below the error threshold
    '''
    if cliff_t:
        checkpoint_dir = clifft_checkpoint_dir
    else:
        checkpoint_dir = nisq_checkpoint_dir

    # print("Checkpoint dir: ", checkpoint_dir)
    # print("Circ name: ", circ_name)
    folder_files = glob.glob(os.path.join(checkpoint_dir, f"{circ_name}_*"))

    block_names = get_block_names(circ_name, extra="_tket")
    block_data = {}
    precision = np.log10(err_threshold) * -1
    precision = int(np.ceil(precision))
    # print("Precision: ", precision)
    circ_files_orig = [load_block(circ_name, block_name) for block_name in block_names]
    circ_files_tket = [load_block(circ_name, block_name, extra="_tket") for block_name in block_names]
    if cliff_t:
        circ_files_orig = [(circ_file, precision) for circ_file in circ_files_orig]
        circ_files_tket = [(circ_file, precision) for circ_file in circ_files_tket]
    orig_counts = get_base_circ_counts(circ_files_orig, cliff_t=cliff_t)
    tket_counts = get_base_circ_counts(circ_files_tket, cliff_t=cliff_t)

    # print("Orig counts: ", orig_counts)
    # print("Tket counts: ", tket_counts)

    for i, block_name in enumerate(block_names):
        orig_count = orig_counts[i]
        tket_count = tket_counts[i]
        block_counts = []
        block_counts.append((0, orig_count, ("orig", "", 0)))
        block_counts.append((0, tket_count, ("tket", "", 0)))
        block_data[block_name] = block_counts

    # print(folder_files)

    for folder_name in folder_files:
        tol = float(folder_name.split('_')[-1])
        block_num = folder_name.split('_')[-2]
        # Read CSV
        csv_files = glob.glob(os.path.join(folder_name, f"*.csv"))
        if len(csv_files) == 0:
            # Just pick ensemble 0
            # print(folder_name)
            ensemble_file = glob.glob(os.path.join(folder_name, f"ensemble_0_.qasms"))
            if len(ensemble_file) == 0:
                # print("No ensemble file found")
                continue
            ensemble_file = ensemble_file[0]
            final_threshold = (10 ** (-2 * tol))
        else:
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
        # print(ensemble_file)
        avg_count = load_ensemble_cx_counts(ensemble_file, cliff_t=cliff_t)
        block_data[block_num].append((final_threshold, avg_count, (circ_name, block_num, tol)))

    # Now for every block, try to select one folder name per block_num 
    # which minimizes the count and is below the threshold
    # print("Block data: ", block_data)
    # Get all possible combinations of block_dirs
    block_nums = sorted(list(block_data.keys()))
    # print(block_data)
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
        return [], 0
    min_count = min([x[1] for x in all_combos])
    min_combos = [x for x in all_combos if x[1] == min_count]
    # print(min_combos)
    # Now get the folder names
    final_data = min_combos[0][2]
    block_counts = []

    # print("Avg count: ", min_count)
    if add_tket_count:
        return [(block_nums[i], d) for i, d in enumerate(final_data)], min_count, np.sum(tket_counts)
    else:
        return [(block_nums[i], d) for i, d in enumerate(final_data)], min_count

def get_counts(circ_name: str, err_threshold: float, cliff_t: bool = False) -> float:
    return get_circ_data(circ_name, err_threshold, cliff_t)[1]

def get_circ_block_dirs(circ_name: str, err_threshold: float, cliff_t: bool = False):
    return get_circ_data(circ_name, err_threshold, cliff_t)[0]

# if __name__ == "__main__":
#     # print(get_circ_block_dirs("adder9", 0.3, False))
#     print(get_completed_blocks(True))