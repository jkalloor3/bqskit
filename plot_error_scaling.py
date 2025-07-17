import os
import csv
import pickle
import time
import matplotlib.pyplot as plt
import glob
from util import load_block, GateCounter, load_avg_ensemble_counts_full, get_block_names
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.ir import Circuit
from bqskit.ir.gates import CircuitGate, CNOTGate
from util.plot_lib import plot_all_circ_violins, benchmark_labels

# List of circuits
# circs = ["shor_12", "qft_16", "draper_adder_12", "qae13", "qpe_14", "lgt_17"]  # Replace with your list of circuits
plot_circs = ["lgt_17", "mult16", "add17", "qpe_14", "shor_12_no_qft", "qae11"] 
circs = ["add17", "shor_12_no_qft", "lgt_17", "qae13", "qpe_14", "QITE_8_0", "mult16", "draper_adder_12", "qae11"]
large_circs = ["qae33", "heisenberg64", "adder63"]

all_circs = plot_circs + large_circs + circs
all_circs = set(all_circs)

# Directory containing block checkpoints
block_checkpoints_dir = f'block_checkpoints_final_paper_tket_4*'
small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_more_cx_tket"
small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_tket"
# block_form = "{circ}_*/data.csv"
block_csv_form = "{circ}_*/block_*.csv"
block_qasms_form = "{circ}_*/block_*/ensemble_final.qasms"
cache_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/*cache*.pkl"
jiggle_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_final_jiggle.npy"

cx_counter = GateCounter(est=True)

def find_tket_qasm(circ_name: str) -> str:
    """
    Find the tket qasm file for the given circuit name.
    """
    # Check if the file exists in the current directory
    dir_1 = "ensemble_benchmarks_tket"
    dir_2 = "qce23_qfactor_benchmarks_tket"
    dir_3 = "ensemble_benchmarks"
    dir_4 = "qce23_qfactor_benchmarks"

    if os.path.exists(os.path.join(dir_1, f"{circ_name}.qasm")):
        return os.path.join(dir_1, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_2, f"{circ_name}.qasm")):
        return os.path.join(dir_2, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_3, f"{circ_name}.qasm")):
        return os.path.join(dir_3, f"{circ_name}.qasm")
    if os.path.exists(os.path.join(dir_4, f"{circ_name}.qasm")):
        return os.path.join(dir_4, f"{circ_name}.qasm") 

    return None


def get_avg_count(checkpoints_dir: str,
                  circ_name: str, 
                  block_num: str,
                  tol: float,
                  small_block_num: str, 
                  cliff_t: bool = False):
    qasms_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_0_*.qasms"

    qasms_file = os.path.join(checkpoints_dir, 
                                qasms_form.format(circ=circ_name, 
                                                 large_block_num=block_num, 
                                                 tol=tol, 
                                                 small_block_num=small_block_num))
    
    qasms_files = glob.glob(qasms_file)
    use_fw = False
    final_qasm_file = None
    for qasms_file in qasms_files:
        if "_fw" in qasms_file:
            use_fw = True
            final_qasm_file = qasms_file
            break

    if final_qasm_file is None and len(qasms_files) > 0:
        final_qasm_file = qasms_files[0]
    elif final_qasm_file is None:
        return float('inf')

    if use_fw:
        extra_str = "_fw"
    else:
        extra_str = ""

    if cliff_t:
        cache_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_0_cache_{extra}.pkl"
        jiggle_form = "{circ}_{large_block_num}_{tol}/block_{small_block_num}/ensemble_0_jiggles_{extra}.npy"

        cache_file = os.path.join(checkpoints_dir, 
                                    cache_form.format(circ=circ_name,
                                                    large_block_num=block_num,
                                                    tol=tol,
                                                    small_block_num=small_block_num,
                                                    extra=extra_str))
        if not os.path.exists(cache_file):
            cache_file = None

        jiggle_file = os.path.join(checkpoints_dir,
                                    jiggle_form.format(circ=circ_name,
                                                    large_block_num=block_num,
                                                    tol=tol,
                                                    small_block_num=small_block_num,
                                                    extra=extra_str))
    else:
        cache_file = None
        jiggle_file = None

    return load_avg_ensemble_counts_full(
        qasms_file, jiggle_file=jiggle_file, cache_file=cache_file,
        target_error=(10 ** (-tol * 2)), count_t =cliff_t,
    )


# Function to read data.csv from each folder
def read_cx_data_from_folders(circuits, orig_cx_counts, 
                              checkpoints_dir, small_block = False,
                              cliff_t: bool = False):
    all_data = {}
    if checkpoints_dir is None:
        return all_data
    for circ_name in circuits:
        all_data[circ_name] = {}
        qasms_file_form = os.path.join(checkpoints_dir, block_qasms_form.format(circ=circ_name))
        qasms_files = glob.glob(qasms_file_form)
        for qasms_file in qasms_files:
            folder = os.path.dirname(qasms_file).split('/')[-2]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2]) 
            folder = os.path.dirname(qasms_file).split('/')[-1]
            small_block_num = folder.split("_")[-1]
            if block_num not in all_data[circ_name]:
                all_data[circ_name][block_num] = {}

            avg_count = get_avg_count(checkpoints_dir, 
                                      circ_name, 
                                      block_num, 
                                      tol, 
                                      small_block_num, 
                                      cliff_t=cliff_t)
            if small_block:
                block_ind = (block_num, small_block_num)
            else:
                block_ind = (block_num, 0)
            if cliff_t:
                orig_count = orig_cx_counts[circ_name][block_ind][tol]
            else:
                try:
                    orig_count = orig_cx_counts[circ_name][block_ind]
                except KeyError:
                    print(f"KeyError: {circ_name}, {block_ind}, {tol}")
                    orig_count = 0
            if block_ind not in all_data[circ_name]:
                all_data[circ_name][block_ind] = {}
            all_data[circ_name][block_ind][tol] = (orig_count, avg_count)
    return all_data


# Function to read data.csv from each folder
def read_data_from_folders(circuits, checkpoints_dir, small_block = False):
    all_data = {}
    if checkpoints_dir is None:
        return all_data
    for circ in circuits:
        all_data[circ] = {}
        csv_file_form = os.path.join(checkpoints_dir, block_csv_form.format(circ=circ))
        csv_files = glob.glob(csv_file_form)
        for csv_file in csv_files:
            folder = os.path.dirname(csv_file).split('/')[-1]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2])
            # Read CSV with csv
            with open(csv_file, 'r') as file:
                # Read the CSV file and return min
                # value of 'Norm. Ratio' column
                reader = csv.DictReader(file)
                eps = tol
                if small_block:
                    small_block_num = os.path.basename(csv_file).split(".")[0].split("_")[1]
                    block_ind = (block_num, small_block_num)
                else:
                    block_ind = (block_num, 0)
                if block_ind not in all_data[circ]:
                    all_data[circ][block_ind] = {}
                final_ratio = all_data[circ][block_ind].get(eps, float("inf"))
                for row in reader:
                    if "Ratio" in row:  # Check if the column value is not empty
                        # final_ratio = min(final_ratio, float(row["Norm. Ratio"]))
                        if float(row["Ratio"]) < final_ratio:
                            actual_eps = float(row["Norm. Epsilon"])
                            if actual_eps > 100 * (10 ** (-tol)):
                                continue
                            final_ratio = min(final_ratio, float(row["Ratio"]))
                if final_ratio > 10000:
                    # Just set it to 10000 and we will plot it as 10000+
                    final_ratio = 10000
                all_data[circ][block_ind][eps] = final_ratio

    return all_data

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
    for circ in data.keys():
        orig_qasm_file = find_tket_qasm(circ)
        if orig_qasm_file is None:
            print(f"Warning: No original QASM file found for {circ}. Skipping.")
            continue
        if not cliff_t:
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
                if isinstance(value, int):
                    print(f"{circ} {eps}: {value}", flush=True)
                    continue
                orig_count, avg_count = value
                total_counts[eps] += min(orig_count, avg_count)
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
                        print(f"{circ_name}:{block_num}:{small_block_num}:{err}:", flush=True)
                        orig_cx_counts[circ_name][(block_num, small_block_num)][err] = t_counter.count_t(op.gate._circuit, 10 ** (-err * 2))
                else:
                    orig_cx_counts[circ_name][(block_num, small_block_num)] = op.gate._circuit.count(CNOTGate())

    with open(save_file, 'wb') as f:
        pickle.dump(orig_cx_counts, f)
    return orig_cx_counts

if __name__ == '__main__':
    # Collect data from all folders
    use_small_block = True
    plot = False
    if plot:
        circs = plot_circs
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


    good_ratio_data = read_data_from_folders(circs, small_block_checkpoints_dir_2, small_block=use_small_block)
    ratio_data_more_cx = read_data_from_folders(circs, small_block_checkpoints_dir_1, small_block=use_small_block)

    print("Ratio data loaded", flush=True)
    if output_cx:
        compiler = Compiler(num_workers=len(circs))
        orig_counts = get_orig_counts(circs, cliff_t=cliff_t, compiler=compiler)
        compiler.close()
        print("Original counts loaded", flush=True)
        cx_data_more_cx= read_cx_data_from_folders(circs, orig_counts,
                                                    small_block_checkpoints_dir_1, 
                                                    small_block=use_small_block, 
                                                    cliff_t=cliff_t)
        
        print("CX data loaded", flush=True)
        
        good_cx_data = read_cx_data_from_folders(circs, orig_counts,
                                                  small_block_checkpoints_dir_2, 
                                                  small_block=use_small_block, 
                                                  cliff_t=cliff_t)
    else:
        cx_data_more_cx = {}
        good_cx_data = {}

    # Start with more_cx data, and update with good_ratio_data if its better
    ratio_data = good_ratio_data.copy()
    ratio_data.update(ratio_data_more_cx)
    
    cx_data = good_cx_data.copy()
    cx_data.update(cx_data_more_cx)

    RATIO_LIMIT = 10

    # For all circ, block_num, small_block_num
    if not cliff_t:
        for circ in ratio_data.keys():
            for block_ind in ratio_data[circ].keys():
                for eps in ratio_data[circ][block_ind].keys():
                    final_ratio = ratio_data[circ][block_ind][eps]
                    min_ratio = RATIO_LIMIT
                    if eps == 1.0:
                        min_ratio = 2.5
                    if final_ratio > min_ratio:
                        # Check if good_data is better
                        if circ in good_ratio_data and block_ind in good_ratio_data[circ]:
                            good_ratio = good_ratio_data[circ][block_ind].get(eps, float("inf"))
                            if good_ratio < min_ratio:
                                ratio_data[circ][block_ind][eps] = good_ratio_data[circ][block_ind][eps]
                                if output_cx:
                                    cx_data[circ][block_ind][eps] = good_cx_data[circ][block_ind][eps]
                            else:
                                if output_cx:
                                    new_val = (cx_data[circ][block_ind][eps][0], float("inf"))
                                    cx_data[circ][block_ind][eps] = new_val  # Don't count reduction
                        else:
                            if output_cx:
                                try:
                                    new_val = (cx_data[circ][block_ind][eps][0], float("inf"))
                                    cx_data[circ][block_ind][eps] = new_val  # Don't count reduction
                                except KeyError:
                                    print(f"KeyError: {circ}, {block_ind}, {eps}")
                                    continue

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
        csv_file_name = f"count_data_final{extra}.csv"
        output_csv(cx_data, csv_file_name)