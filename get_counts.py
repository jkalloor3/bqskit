from util import get_completed_blocks, get_circ_data
from util import get_block_names
import glob
import concurrent.futures

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
    # circs = ["add17"]
    # circs = ["qae11"]
    # circs = ["QITE_8_1"]
    # circs.remove("qae11")
    # circs.remove("qae13")
    circs.remove("hhl8")
    # circs.remove("add17")
    # circs = ["add17"]
    for i in range(7):
        circs.remove(f"QITE_8_{i}")
    # circs.remove(f"QITE_8_{i}")

    for c in ["qae13", "qaoa10", "shor_12"]:
        circs.remove(c)

    print(circs)
    # exit(0)
    # circs = ["mult16"]
    err_thresholds = [1e-3, 1e-5, 1e-7]
    # print("Circ names: ", circs)
    print("Circ Name, TKET Count, " + ",".join([f"Error Threshold {err_threshold}" for err_threshold in err_thresholds]))
    all_counts = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=120) as executor:
        for circ_name in circs:
            all_counts[circ_name] = []
            for err_threshold in err_thresholds:
                fut = executor.submit(get_circ_data, circ_name, err_threshold, use_base=True, count_t=cliff_t)
                all_counts[circ_name].append(fut)

    for i, circ_name in enumerate(circs):
        count = all_counts[circ_name][0].result()
        print(f"{circ_name}, - , {count[2]} | {count[1]}", end="")
        for j in range(1, len(err_thresholds)):
            count = all_counts[circ_name][j].result()
            print(f", {count[2]} | {count[1]}", end="")
        print("")