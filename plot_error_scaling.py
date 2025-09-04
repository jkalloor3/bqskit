import os
import csv
import pickle
import matplotlib.pyplot as plt
import glob
from util import load_block, GateCounter, load_avg_ensemble_counts_full, get_block_names
from util.unitary_dm_pass import get_file_names
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner, ExtendBlockSizePass
from bqskit.ir import Circuit
from bqskit.ir.gates import CircuitGate, CNOTGate
from util.plot_lib import plot_all_circ_violins, benchmark_labels

# List of circuits
circs = ["draper_adder_12", "qae13", "qpe_14", "lgt_17"]  # Replace with your list of circuits
plot_circs = ["lgt_17", "mult16", "add17", "qpe_14", "qae11", "LiH_jw_long", "FermiHubbard2x2_jw_long", "heisenberg7"] 
# circs = ["add17", "lgt_17", "qae13", "qpe_14", "QITE_8_0", "mult16", "draper_adder_12", "qae11"]
large_circs = ["qae33"] #, "heisenberg64", "adder63"]
# plot_circs = []
# circs = ["LiH_jw_long", "FermiHubbard2x2_jw_long"]
# large_circs = []

all_circs = plot_circs + large_circs + circs
all_circs = set(all_circs)

# block_form = "{circ}_*/data.csv"
block_csv_form = "{circ}_*/block_*.csv"
block_qasms_form = "{circ}_*/block_*/ensemble_final_fw.qasms"
cx_counter = GateCounter(est=True)

def find_tket_qasm(circ_name: str) -> str:
    """
    Find the tket qasm file for the given circuit name.
    """
    # Check if the file exists in the current directory
    dir_1 = "ensemble_benchmarks_tket"
    dir_2 = "qce23_qfactor_benchmarks_tket"
    dir_5 = "ensemble_benchmarks_new"
    dir_3 = "ensemble_benchmarks"
    dir_4 = "qce23_qfactor_benchmarks"

    if os.path.exists(os.path.join(dir_1, f"{circ_name}.qasm")):
        return os.path.join(dir_1, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_2, f"{circ_name}.qasm")):
        return os.path.join(dir_2, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_5, f"{circ_name}.qasm")):
        return os.path.join(dir_5, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_3, f"{circ_name}.qasm")):
        return os.path.join(dir_3, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_4, f"{circ_name}.qasm")):
        return os.path.join(dir_4, f"{circ_name}.qasm") 

    return None

def check_good(csv_file: str, tol: float) -> tuple[bool, float]:
    """
    Check if the csv file has a good ratio for the given tolerance.
    """
    final_ratio = float("inf")
    if tol == 1.0:
        ratio_limit = 2.5
    elif tol < 3.0:
        ratio_limit = 20.0
    else:
        ratio_limit = 50.0

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                if float(row["Ratio"]) < final_ratio:
                    max_dist = ratio_limit * (10 ** (-2 * tol))
                    bias = float(row["Bias"])
                    if bias > max_dist:
                        new_ratio = bias / (10 ** (-2 * tol))
                    else:
                        new_ratio = float(row["Ratio"])
                    final_ratio = min(final_ratio, new_ratio)
    return final_ratio < ratio_limit, final_ratio

def get_avg_count(checkpoints_dir: str,
                  circ_name: str, 
                  block_num: str,
                  tol: float,
                  small_block_num: str, 
                  cliff_t: bool = False):

    large_checkpoint_dir = os.path.join(checkpoints_dir, f"{circ_name}_{block_num}_{tol}")
    qasms_file, jiggle_file, _, cache_file, csv_file = get_file_names(large_checkpoint_dir, small_block_num)
    if not check_good(csv_file, tol):
        return float("inf")

    return load_avg_ensemble_counts_full(
        qasms_file, jiggle_file=jiggle_file, cache_file=cache_file,
        target_error=(10 ** (-tol * 2)), count_t =cliff_t,
    )

# Function to read data.csv from each folder
def update_cx_data_from_folders(orig_cx_counts, 
                              checkpoints_dir, cliff_t: bool = False):
    for circ_name, block_data in orig_cx_counts.items():
        for (large_block_num, small_block_num) in block_data.keys():
            for tol in block_data[(large_block_num, small_block_num)].keys():
                # print(circ_name, large_block_num, small_block_num, tol, flush=True)
                try:
                    avg_count = get_avg_count(checkpoints_dir, 
                                            circ_name, 
                                            large_block_num, 
                                            tol, 
                                            small_block_num, 
                                            cliff_t=cliff_t)
                except:
                    avg_count = float("inf")

                block_ind = (large_block_num, small_block_num)
                orig_count = orig_cx_counts[circ_name][block_ind][tol]
                # If orig count is a tuple, take the 0th element
                if isinstance(orig_count, tuple):
                    orig_count, prev_count = orig_count
                else:
                    prev_count = orig_count

                new_count = min(prev_count, avg_count)
                orig_counts[circ_name][block_ind][tol] = (orig_count, new_count)
                # if avg_count < 100000:
                #     print(f"New Val: ", (orig_count, new_count), flush=True)
    return orig_counts

# Function to read data.csv from each folder
def update_data_from_folders(all_data: dict, checkpoints_dir):
    if checkpoints_dir is None:
        return all_data
    for circ, block_data in all_data.items():
        csv_file_form = os.path.join(checkpoints_dir, block_csv_form.format(circ=circ))
        csv_files = glob.glob(csv_file_form)
        for csv_file in csv_files:
            folder = os.path.dirname(csv_file).split('/')[-1]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2])
            small_block_num = os.path.basename(csv_file).split(".")[0].split("_")[1]
            block_ind = (block_num, small_block_num)
            _, final_ratio = check_good(csv_file, tol)
            if final_ratio > 10000:
                # Just set it to 10000 and we will plot it as 10000+
                final_ratio = 10000
            if block_ind not in block_data:
                block_data[block_ind] = {}
            if tol not in block_data[block_ind]:
                block_data[block_ind][tol] = final_ratio
            else:
                # If it already exists, take the minimum ratio
                block_data[block_ind][tol] = min(block_data[block_ind][tol], final_ratio)
            all_data[circ] = block_data

def output_csv(data: dict, file_name: str, cliff_t: bool = False):
    # Create a DataFrame from the data
    # Step 1: Collect all unique eps values and sort them
    all_eps = set()
    for block_data in data.values():
        for eps_data in block_data.values():
            all_eps.update(eps_data.keys())

    sorted_eps = sorted(all_eps, key=float)

    # Step 2: Get the original counts from ensemble_benchmarks_tket and qce_qfactor_benchmarks_tket
    full_orig_counts = {}
    if not cliff_t:
        for circ in data.keys():
            orig_qasm_file = find_tket_qasm(circ)
            if orig_qasm_file is None:
                print(f"Warning: No original QASM file found for {circ}. Skipping.")
                continue
            orig_circ = Circuit.from_file(orig_qasm_file)
            orig_counts = orig_circ.count(CNOTGate())
            full_orig_counts[circ] = orig_counts

    
    # Step 2: Build rows
    rows = []
    for circ, block_data in data.items():
        total_counts = {eps: 0 for eps in sorted_eps}  # Pre-fill with zeros
        total_orig_counts = {eps: 0 for eps in sorted_eps}  # Pre-fill with zeros

        for eps_data in block_data.values():
            for eps, value in eps_data.items():
                # print(f"Processing {circ} {eps}: {value}", flush=True)
                # if isinstance(value, int):
                #     print(f"{circ} {eps}: {value}", flush=True)
                #     continue
                orig_count, value = value
                total_counts[eps] += value
                total_orig_counts[eps] += orig_count

        if not cliff_t:
            # See if full_orig_counts is smaller than total_orig_counts
            if circ in full_orig_counts:
                orig_count = full_orig_counts[circ]
                for eps in sorted_eps:
                    if total_orig_counts[eps] > orig_count:
                        total_orig_counts[eps] = orig_count

        # Interleave the original counts with the total counts
        row = [benchmark_labels.get(circ, circ)]
        for eps in sorted_eps:
            row += [total_orig_counts[eps], total_counts[eps]]
        rows.append(row)

    # Step 3: Create labels for the columns
    labels = ['Circuit']
    for eps in sorted_eps:
        labels += [f"Tket: 10e-{int(eps * 2)}", f"Ens: 10e-{int(eps * 2)}"]
    # Step 3: Write to CSV
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(labels)  # Header
        writer.writerows(rows)
    print(f"Data saved to {file_name}")

def get_orig_counts(circuits: list[str], cliff_t: bool = False, 
                    compiler: Compiler = None) -> dict:
    orig_cx_counts = {}

    cliff_t_text = "_cliff_t" if cliff_t else ""
    save_file = f"orig_counts{cliff_t_text}.pickle"

    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            orig_cx_counts = pickle.load(f)
        # If any circuits are missing, then compile only those
        missing_circuits = [circ for circ in circuits if circ not in orig_cx_counts]
        if len(missing_circuits) == 0:
            print(f"Loaded original counts from {save_file}")
            return orig_cx_counts
        print(f"Missing circuits: {missing_circuits}, compiling those only")
        circuits = missing_circuits

    workflow = [
        ScanPartitioner(4),
        ExtendBlockSizePass(4)
    ]

    id_data = {}
    for circ_name in circuits:
        orig_cx_counts[circ_name] = {}
        id_data[circ_name] = {}
        if circ_name in large_circs:
            # Just read the qasms in bad and good blocks
            print(f"Already Partitioned, reading direclty from files for {circ_name}", flush=True)
            block_names = get_block_names(circ_name=circ_name, extra="_tket")
            for block_num in block_names:
                tket_file = load_block(circ_name=circ_name, block_num=block_num,
                        extra="_tket")
                small_circ = Circuit.from_file(tket_file)
                if small_circ.num_qudits <= 4:
                    if cliff_t:
                        orig_cx_counts[circ_name][(block_num, "0")] = {}
                        t_counter = GateCounter(est = False)
                        for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                            # print(f"{circ_name}:{block_num}:{err}:", flush=True)
                            orig_cx_counts[circ_name][(block_num, "0")][err] = t_counter.count_t(small_circ, 10 ** (-err * 2))
                    else:
                        cx_count = open(tket_file, 'r').read().count("cx")
                        orig_cx_counts[circ_name][(block_num, "0")] = cx_count
                else:
                    print("Partitioning small block for", circ_name, block_num, flush=True)
                    # We have to partition this small block FML
                    if small_circ.num_operations == 0:
                        orig_cx_counts[circ_name][(block_num, "0")] = 0
                        continue
                    id = compiler.submit(small_circ.copy(), workflow)
                    id_data[circ_name][block_num] = id
        else:
            # Have to partition the circuit into blocks to get block counts
            for block_num in get_block_names(circ_name=circ_name, extra="_tket"):
                if block_num in orig_cx_counts[circ_name]:
                    # Already compiled this block
                    continue
                tket_file = load_block(circ_name=circ_name, block_num=block_num,
                            extra="_tket")
                circ = Circuit.from_file(tket_file)
                if circ.num_operations == 0:
                    orig_cx_counts[circ_name][(block_num, "0")] = 0
                    continue
                id = compiler.submit(circ.copy(), workflow)
                id_data[circ_name][block_num] = id

    for circ_name, block_data in id_data.items():
        for block_num, id in block_data.items():
            out_circ: Circuit = compiler.result(id)
            num_digits = len(str(out_circ.num_operations))
            for i, (_, op) in enumerate(out_circ.operations_with_cycles()):
                assert isinstance(op.gate, CircuitGate)
                # Check if checkpoint exists:
                # Need to zero pad block ids for consistency
                small_block_num = str(i).zfill(num_digits)
                if cliff_t:
                    orig_cx_counts[circ_name][(block_num, small_block_num)] = {}
                    t_counter = GateCounter(est = False)
                    for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                        # print(f"{circ_name}:{block_num}:{small_block_num}:{err}:", flush=True)
                        orig_cx_counts[circ_name][(block_num, small_block_num)][err] = t_counter.count_t(op.gate._circuit, 10 ** (-err * 2))
                else:
                    orig_cx_counts[circ_name][(block_num, small_block_num)] = op.gate._circuit.count(CNOTGate())

    with open(save_file, 'wb') as f:
        pickle.dump(orig_cx_counts, f)
    return orig_cx_counts

if __name__ == '__main__':
    # Collect data from all folders
    use_small_block = True
    plot = True
    if plot:
        circs = plot_circs
        output_cx = False
    else:
        circs = all_circs
        output_cx = True
    cliff_t = True

    if not cliff_t:
        small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_more_cx_tket"
        small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_tket"
    else:
        small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_clifft_tket"
        small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_clifft_less_t_tket"

    ratio_data = {c: {} for c in circs}
    update_data_from_folders(ratio_data, small_block_checkpoints_dir_1)

    print("Ratio data loaded", flush=True)
    if output_cx:
        compiler = Compiler('localhost')
        # compiler = Compiler(num_workers=128)
        orig_counts = get_orig_counts(circs, cliff_t=cliff_t, compiler=compiler)
        # Only use orig_counts for the circs we want
        orig_counts = {circ: orig_counts[circ] for circ in circs}
        compiler.close()
        print("Original counts loaded", flush=True)
        update_cx_data_from_folders(orig_counts, small_block_checkpoints_dir_1,
                                      cliff_t=cliff_t)

        # update_cx_data_from_folders(orig_counts, small_block_checkpoints_dir_2,
        #                                           cliff_t=cliff_t)
        print("CX data loaded", flush=True)
        print("CX data more cx:", list(orig_counts.keys()), flush=True)

    if cliff_t:
        extra = "_cliff"
    else:
        extra = "_nisq"

    if plot:
        # Plot ratio data
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))

        if cliff_t:
            title = "Error Scaling (FT)"
        else:
            title = "Error Scaling (NISQ)"

        plot_all_circ_violins(ratio_data, axes, plot_title=title)

        # Save the figure
        fig.tight_layout()
        fig.savefig(f"error_scaling_final{extra}.png", dpi=300)


    # Output CX data to a csv file
    if output_cx:
        print("Outputting CX data to CSV", flush=True)
        csv_file_name = f"count_data_final{extra}_final.csv"
        output_csv(orig_counts, csv_file_name, cliff_t=cliff_t)