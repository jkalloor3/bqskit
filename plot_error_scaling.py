import json
from sys import argv
import os
import csv
import pickle
import matplotlib.pyplot as plt
import glob
from util import load_block, GateCounter, check_good, get_block_names
from util.experiment_util import get_file_names
from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner, ExtendBlockSizePass
from bqskit.ir import Circuit
from bqskit.ir.gates import CircuitGate, CNOTGate
from util.plot_lib import plot_all_circ_violins, benchmark_labels

# List of circuits
base_circs = ["qae13", "qpe_14", "lgt_17"]  # Replace with your list of circuits
plot_circs = ["heisenberg7", "qaoa10", "LiH_jw_long", "qae33", "lgt_380"] 
base_circs += ["add17", "lgt_17", "qae13", "qpe_14", "mult16", "draper_adder_12", "qae11", "qaoa10"]
large_circs = ["qae33", "qaoa_148", "lgt_380"]

# all_circs = ["draper_adder_12"]
all_circs = ["heisenberg7", "FermiHubbard2x2_jw_long", "LiH_jw_long", "draper_adder_12", "qaoa10", "qae13", "qpe_14", "mult16", "lgt_17"]
all_circs += large_circs
# all_circs = ["qaoa10", "heisenberg7", "FermiHubbard2x2_jw_long", "LiH_jw_long", "qaoa10", "qae13", "qpe_14"]
# all_circs = ["heisenberg7", "FermiHubbard2x2_jw_long", "qaoa10"]


block_csv_form = "{circ}_*/block_*fw_2*.csv"
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

def get_avg_count(checkpoints_dir: str,
                  circ_name: str, 
                  block_num: str,
                  tol: float,
                  small_block_num: str, 
                  cliff_t: bool,
                  default_ratio_limit: float):
    
    if tol == 1.0:
        ratio_limit = default_ratio_limit / 10
        ratio_text = "_2"
    else:
        ratio_limit = default_ratio_limit
        ratio_text = "_20"

    large_checkpoint_dir = os.path.join(checkpoints_dir, f"{circ_name}_{block_num}_{tol}")
    csv_file = get_file_names(large_checkpoint_dir,small_block_num, 
                              no_qp=False, ratio_text=ratio_text)[-1]

    ratio_1, count_1 = check_good(csv_file, ratio_limit)

    # csv_file = get_file_names(large_checkpoint_dir, small_block_num, no_qp=True,
    #                           ratio_text=ratio_text)[-1]
    # ratio_2, count_2 = check_good(csv_file, default_ratio_limit)

    count_2 = float("inf")


    # if count_1 < 40:
    #     print(csv_file)
    #     print(ratio_1, count_1, flush=True)
    if not os.path.exists(csv_file):
        print(csv_file)
    return min(count_1, count_2)

# Function to read data.csv from each folder
def update_count_data(orig_cx_counts, checkpoints_dir, 
                      cliff_t: bool, default_ratio_limit: float) -> dict:
    print("Updating count data from", checkpoints_dir, flush=True)
    for circ_name, block_data in orig_cx_counts.items():
        for (large_block_num, small_block_num) in block_data.keys():
            for tol in block_data[(large_block_num, small_block_num)].keys():
                # print(circ_name, large_block_num, small_block_num, tol, flush=True)
                avg_count = get_avg_count(checkpoints_dir, 
                                        circ_name, 
                                        large_block_num, 
                                        tol, 
                                        small_block_num, 
                                        cliff_t=cliff_t,
                                        default_ratio_limit=default_ratio_limit)

                block_ind = (large_block_num, small_block_num)
                orig_count, prev_count = orig_cx_counts[circ_name][block_ind][tol]
                # print(prev_count, orig_count, avg_count, flush=True)
                new_count = min(prev_count, avg_count)
                orig_counts[circ_name][block_ind][tol] = (orig_count, new_count)
    return orig_counts

# Function to read data.csv from each folder
def update_ratio_data(all_data: dict, checkpoints_dir,
                      default_ratio_limit: float, bias: bool) -> dict:
    if checkpoints_dir is None:
        return all_data
    for circ, block_data in all_data.items():
        csv_file = block_csv_form.format(circ=circ)
        csv_files = glob.glob(os.path.join(checkpoints_dir, csv_file))
        for csv_file in csv_files:
            folder = os.path.dirname(csv_file).split('/')[-1]
            tol = float(folder.split("_")[-1])
            block_num = str(folder.split("_")[-2])
            small_block_num = os.path.basename(csv_file).split(".")[0].split("_")[1]
            block_ind = (block_num, small_block_num)

            if tol < 1.5:
                ratio_limit = default_ratio_limit / 10
            else:
                ratio_limit = default_ratio_limit

            final_ratio, _ = check_good(csv_file, max_ratio=ratio_limit, bias=bias)
            if final_ratio > 20:
                print(csv_file, final_ratio, flush=True)
            if final_ratio > 1000:
                # Just set it to 1000 and we will plot it as 1000+
                final_ratio = 1000
            if block_ind not in block_data:
                block_data[block_ind] = {}
            if tol not in block_data[block_ind]:
                block_data[block_ind][tol] = final_ratio
            else:
                # If it already exists, take the minimum ratio
                block_data[block_ind][tol] = min(block_data[block_ind][tol], 
                                                 final_ratio)
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
        if cliff_t:
            for eps in sorted_eps:
                row += [total_orig_counts[eps], total_counts[eps]]
        else:
            # Just have Baseline at the beginning
            row += [total_orig_counts[sorted_eps[0]]]
            row += [total_counts[eps] for eps in sorted_eps]
        rows.append(row)

    # Step 3: Create labels for the columns
    labels = ['Circuit']
    if cliff_t:
        for eps in sorted_eps:
            labels += [f"Baseline", f"Ens: 10e-{int(eps * 2)}"]
    else:
        labels += ['Baseline']
        for eps in sorted_eps:
            labels += [f"10e-{int(eps * 2)}"]
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
    save_file = f"orig_counts{cliff_t_text}_full.pickle"

    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            orig_cx_counts = pickle.load(f)
        # If any circuits are missing, then compile only those
        missing_circuits = [circ for circ in circuits if circ not in orig_cx_counts]
        if len(missing_circuits) == 0:
            print(f"Loaded original counts from {save_file}")
            # Dump to JSON
            # json.dump(orig_cx_counts, open(f"orig_counts{cliff_t_text}.json", 'w'), indent=4)
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
                    orig_cx_counts[circ_name][(block_num, "0")] = {}
                    if cliff_t:
                        t_counter = GateCounter(est = False)
                        for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                            t_count = t_counter.count_t(small_circ, 10 ** (-err * 2))
                            orig_cx_counts[circ_name][(block_num, "0")][err] = (t_count, t_count)
                    else:
                        cx_count = open(tket_file, 'r').read().count("cx")
                        for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                            orig_cx_counts[circ_name][(block_num, "0")][err] = (cx_count,
                                                                       cx_count)
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
                    orig_cx_counts[circ_name][(block_num, "0")] = (0, 0)
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
                orig_cx_counts[circ_name][(block_num, small_block_num)] = {}
                if cliff_t:
                    t_counter = GateCounter(est = False)
                    for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                        t_count = t_counter.count_t(op.gate._circuit, 10 ** (-err * 2))
                        orig_cx_counts[circ_name][(block_num, small_block_num)][err] = (t_count, t_count)
                else:
                    for err in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                        ccount = op.gate._circuit.count(CNOTGate())
                        orig_cx_counts[circ_name][(block_num, small_block_num)][err] = (ccount, ccount)

    with open(save_file, 'wb') as f:
        pickle.dump(orig_cx_counts, f)
    return orig_cx_counts


if __name__ == '__main__':
    # Collect data from all folders
    plot = True
    cliff_t = True
    bias = True

    default_ratio_limit = float(argv[1])

    print("Ratio limit:", default_ratio_limit, flush=True)

    if plot:
        circs = plot_circs
        output_cx = False
    else:
        circs = all_circs
        output_cx = True

    print("Circuits to process:", list(circs), flush=True)

    if not cliff_t:
        small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_more_cx_tket"
        # small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_tket"
    else:
        small_block_checkpoints_dir_1 = f"small_block_checkpoints_final_paper_4_clifft_tket"
        small_block_checkpoints_dir_2 = f"small_block_checkpoints_final_paper_4_clifft_tket_final"

    ratio_data = {c: {} for c in circs}

    # print("Ratio data loaded", flush=True)
    if output_cx:
        # compiler = Compiler('localhost')
        # compiler = Compiler()
        compiler = None
        orig_counts = get_orig_counts(circs, cliff_t=cliff_t, compiler=compiler)
        # compiler.close()
        # Only use orig_counts for the circs we want
        orig_counts = {circ: orig_counts[circ] for circ in circs}
        print("Original counts loaded", flush=True)
        update_count_data(orig_counts, small_block_checkpoints_dir_1,
                                      cliff_t=cliff_t,
                                      default_ratio_limit=default_ratio_limit)

        # update_count_data(orig_counts, small_block_checkpoints_dir_2,
        #                                           cliff_t=cliff_t,
        #                                           default_ratio_limit=default_ratio_limit)
        print("CX data loaded", flush=True)
        print("CX data more cx:", list(orig_counts.keys()), flush=True)


    if cliff_t:
        extra = "_cliff"
    else:
        extra = "_nisq"

    if plot:
        # Plot ratio data
        update_ratio_data(ratio_data, small_block_checkpoints_dir_1, 
                          default_ratio_limit=default_ratio_limit, bias=bias)
        # update_ratio_data(ratio_data, small_block_checkpoints_dir_2, 
        #                   default_ratio_limit=default_ratio_limit, bias=bias)
        print("Ratio data loaded", flush=True)


        fig, axes = plt.subplots(1, 1, figsize=(12, 6))

        if bias:
            scale_text = "Bias"
        else:
            scale_text = "Error"

        if cliff_t:
            title = f"{scale_text} Scaling (FT)"
        else:
            title = f"{scale_text} Scaling (NISQ)"

        plot_all_circ_violins(ratio_data, axes, plot_title=title)

        # Save the figure
        fig.tight_layout()
        fig.savefig(f"{scale_text}_scaling_final{extra}_{int(default_ratio_limit)}.png", dpi=300)


    # Output CX data to a csv file
    if output_cx:
        print("Outputting CX data to CSV", flush=True)
        csv_file_name = f"count_data_final{extra}_{int(default_ratio_limit)}_full.csv"
        output_csv(orig_counts, csv_file_name, cliff_t=cliff_t)