from util import get_completed_blocks, get_circ_data
from util import get_block_names
import glob

cliff_t = False
if __name__ == '__main__':
    # circ_names = get_completed_blocks(cliff_t)

    dirs = ["ensemble_benchmarks", "qce23_qfactor_benchmarks"]
    trial_circs = []
    for dir in dirs:
        files = glob.glob(f"{dir}/*.qasm")
        trial_circs.extend([file.split('/')[-1].split(".")[0] for file in files])

    circs = set()
    for circ in trial_circs:
        blocks = get_block_names(circ, extra="_tket")
        if len(blocks) > 0:
            circs.add(circ)

    # circ_names = ["qae11"]
    err_thresholds = [0.5, 0.1, 1e-3]
    # print("Circ names: ", circs)
    print("Circ Name, TKET Count, " + ",".join([f"Error Threshold {err_threshold}" for err_threshold in err_thresholds]))
    for circ_name in circs:
        count = get_circ_data(circ_name, err_thresholds[0], cliff_t=cliff_t)
        print(f"{circ_name}, {count[2]}, {count[1]}", end="")
        for err_threshold in err_thresholds[1:]:
            count = get_circ_data(circ_name, err_threshold, cliff_t=cliff_t)
            print(f", {count[1]}", end="")
        print()