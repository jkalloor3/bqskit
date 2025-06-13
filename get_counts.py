from util import get_completed_blocks, get_circ_data_basic
from util import get_block_names
import glob
import concurrent.futures

cliff_t = True

if not cliff_t:
    err_thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
else:
    err_thresholds = [1e-3, 1e-5, 1e-7]

def print_header():
    print("Circ Name, Exact Opt. Count, " + 
          ",".join([f"Error Threshold {err_threshold}" for 
                    err_threshold in err_thresholds]))

def print_row(circ_name:str, counts: list[tuple[int, int]]):
    if cliff_t:
        count = counts[0]
        print(f"{circ_name}, - , {count[0]} | {count[1]}", end="")
        for j in range(1, len(counts)):
            count = counts[j]
            print(f", {count[0]} | {count[1]}", end="")
        print("")
    else:
        base_count = counts[0][0]
        print(f"{circ_name}, {base_count}", end="")
        for j in range(len(counts)):
            count = counts[j]
            print(f", {count[1]}", end="")
        print("")

if __name__ == '__main__':

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

    circs.remove("hhl8")
    for i in range(7):
        try:
            circs.remove(f"QITE_8_{i}")
        except KeyError:
            continue
    # circs = ["add17", "qae13", "qaoa10", "shor_12", "lgt_17"]

    for c in ["add17", "qae13", "qaoa10", "shor_12", "lgt_17"]:
        try:
            circs.remove(c)
        except KeyError:
            continue

    # print(circs)
    # exit(0)
    # circs = ["lgt_17"]
    all_counts: dict[str, list[concurrent.futures.Future]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for circ_name in circs:
            all_counts[circ_name] = []
            for err_threshold in err_thresholds:
                fut = executor.submit(get_circ_data_basic, circ_name, err_threshold, count_t=cliff_t,)
                all_counts[circ_name].append(fut)
    
    print_header()
    for i, circ_name in enumerate(circs):
        counts = [all_counts[circ_name][j].result() for j in range(len(err_thresholds))]
        print_row(circ_name, counts)