import glob
from os.path import join
from dateutil import parser
import pytz
from sys import argv
partition_size = 7
circuit = "adder63"
compiler = "QFACTOR-JAX"
log_dir = "logs_tol_a_0"
checkpoint_dir = "checkpoints"
GMT_tz = pytz.timezone("Etc/GMT-1:00")

failure_dict = {"Preempted": [], "Nvidia failure": set(), "other": set()}

def get_num_blocks():
    block_files = join(checkpoint_dir, f"{circuit}_{partition_size}", "*.pickle")
    num_blocks = len(glob.glob(block_files))
    return num_blocks

def calc_time_taken_preempt(log_str: str):
    # return 0
    start_time_str = log_str.split("\n", 1)[0].strip(" ")
    start_time = parser.parse(start_time_str)
    total_time = start_time - start_time # start at 0
    premption_strs = log_str.split("CANCELLED AT ")[1:]
    for prempt_str in premption_strs:
        finishing_time = parser.parse(prempt_str.split("DUE", 2)[0].strip(" ")).replace(tzinfo=GMT_tz)
        next_str = prempt_str.split("\n", 2)[1].strip(" ")
        total_time += (finishing_time - start_time)
        try:
            start_time = parser.parse(next_str)
        except:
            break

    return total_time


def parse_log(log_str: str):
    block_str = log_str.split("blocks_to_run", 2)[1]
    block_str = block_str.split("\n", 1)[0].strip(" ")
    blocks = [int(x) for x in block_str.split(" ")]
    completed = log_str.find("Partitioning + Synthesis took") > 0

    if not completed:
        # Find out failure
        preempted = log_str.find("PREEMPTION") > 0
        nvidia = log_str.find("Custom ptxas location") > 0
        if (preempted):
            time_taken = calc_time_taken_preempt(log_str)
            failure_dict["Preempted"].append({"Blocks": blocks, "Time Taken": str(time_taken)})
            # failure_dict["Preempted"].update(blocks)
        if (nvidia):
            failure_dict["Nvidia failure"].update(blocks)
        if (not nvidia and not preempted):
            failure_dict["other"].update(blocks)

    return completed, blocks


def get_completed_blocks(all_logs: list[str]):
    all_blocks = set()
    for log_file in all_logs:
        with open(log_file, "r") as log_f:
            log_str = log_f.read()
            completed, blocks = parse_log(log_str)
            if completed:
                all_blocks.update(blocks)

    return all_blocks

if __name__ == '__main__':
    
    partition_size = argv[2]
    circuit = argv[1]
    file_name_start = f"{circuit}_{partition_size}p_{compiler}_attached_runtime*"
    all_logs = glob.glob(join(log_dir, file_name_start))

    num_blocks = get_num_blocks()
    blocks_finished = get_completed_blocks(all_logs)
    blocks_left = set(range(num_blocks)).difference(set(blocks_finished))

    print(sorted(blocks_left))
    # print(len(blocks_finished))
    # print(num_blocks)
    # for i in failure_dict:
    #     failure_dict[i] = sorted(failure_dict[i].intersection(blocks_left))
    print(failure_dict)

    # print(len(blocks_left))
    # print(len(failure_dict["Preempted"]))
    # print(len(failure_dict["Nvidia failure"]))