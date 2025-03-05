from util import get_completed_blocks, get_circ_data
from util import get_block_names
import glob

cliff_t = True
if __name__ == '__main__':
    # circ_names = get_completed_blocks(cliff_t)

    dirs = ["good_blocks"]
    trial_circs = set()
    for dir in dirs:
        files = glob.glob(f"{dir}/*.qasm")
        file_names = [f.split("/")[-1].split(".")[0] for f in files]
        file_arrs = [f.split("_") for f in file_names]
        circ_names = set(["_".join(file_arr[:-1]) for file_arr in file_arrs])
        trial_circs.update(circ_names)
    
    circs = set()
    for circ in trial_circs:
        blocks = get_block_names(circ, extra="_tket")
        if len(blocks) > 0:
            circs.add(circ)

    # print(circs)
    # circ_names = ["qae11"]
    err_thresholds = [1e-4, 1e-6, 1e-8]
    # print("Circ names: ", circs)
    print("Circ Name, TKET Count, " + ",".join([f"Error Threshold {err_threshold}" for err_threshold in err_thresholds]))
    for circ_name in circs:
        count = get_circ_data(circ_name, err_thresholds[0], cliff_t=cliff_t, add_tket_count=False)
        print(f"{circ_name}, 0, {count[1]}", end="")
        for err_threshold in err_thresholds[1:]:
            count = get_circ_data(circ_name, err_threshold, cliff_t=cliff_t)
            print(f", {count[1]}", end="")
        print()