from util import get_completed_blocks, get_circ_data_basic
from util import get_block_names, load_block
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate

def get_cx_diff(circ_name: str):
    all_data = {}
    for tol in [5.0, 4.0, 3.0, 2.0, 1.0]:
        all_data[tol] = []
        count_data = get_circ_data_basic(circ_name, tol, 
                                                    return_dict=True)
        for block, ens_count in count_data.items():
            orig_block_file = load_block(circ_name, block, extra="_tket")
            orig_circ = Circuit.from_file(orig_block_file)
            orig_count = orig_circ.count(CNOTGate())
            print(f"Block: {block}")
            diff = orig_count - ens_count
            all_data[tol].append((diff, circ_name, block))

    return all_data

if __name__ == '__main__':
    # circ_names = get_completed_blocks(cliff_t)

    # dirs = ["good_blocks"]
    # trial_circs = set()
    # for dir in dirs:
    #     files = glob.glob(f"{dir}/*.qasm")
    #     file_names = [f.split("/")[-1].split(".")[0] for f in files]
    #     file_arrs = [f.split("_") for f in file_names]
    #     circ_names = set(["_".join(file_arr[:-1]) for file_arr in file_arrs])
    #     trial_circs.update(circ_names)
    
    # circs = set()
    # for circ in trial_circs:
    #     blocks = get_block_names(circ, extra="_tket")
    #     if len(blocks) > 0:
    #         circs.add(circ)

    circs = ["adder9"]


    cx_diffs: dict[float, list] = {}

    for circ in circs:
        cx_diff_block = get_cx_diff(circ)
        for float, cx_data in cx_diff_block.items():
            if float not in cx_diffs:
                cx_diffs[float] = []
            cx_diffs[float].append(cx_data)



    # Print the results
    for key, values in cx_diffs.items():
        print(f"Error Threshold: {key}")
        # Sort values
        values.sort(key=lambda x: x[0], reverse=True)
        for value in values:
            print(f"  {value[0]} | {value[1]} : {value[2]}", end=", ")
        print()