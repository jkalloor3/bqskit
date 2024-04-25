import time
import subprocess
import pathlib
from os.path import exists
import os

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=03:55:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/slurm_run.log

module load python
conda activate /pscratch/sd/j/jkalloor/profiler_env
echo "{file}.py {method} {num_qudit} {min_qudit} {tree_depth} {num_worker}"
python {file}.py {method} {num_qudit} {min_qudit} {tree_depth} {num_worker} > updated_runtime/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/log.txt
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "qsd_test"
    # tols = range(1, 7)
    methods = ["qft", "random"]
    num_qudits = [4]
    min_qudits = [2]
    tree_depths = [4, 8, 12]
    # tree_depths = [1]
    for method in methods:
        for num_qudit in num_qudits:
            for min_qudit in min_qudits:
                for tree_depth in tree_depths:
                    # for exp in range(tree_depth - 4, tree_depth + 1):
                    for num_worker in [4, 8, 64]:
                        # num_worker = 2 ** exp
                        if exists(f"updated_runtime/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/log.txt"):
                            with open(f"updated_runtime/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/log.txt") as log_file:
                                log = log_file.readlines()
                                if len(log) > 0 and (not log[-1].startswith("Worker")):
                                    print(log[-1])
                                    continue
                        pathlib.Path(f"updated_runtime/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}").mkdir(parents=True, exist_ok=True)
                        
                        # print(f"python {file}.py {method} {num_qudit} {min_qudit} {tree_depth} {num_worker}")
                        # os.system(f"python {file}.py {method} {num_qudit} {min_qudit} {tree_depth} {num_worker} > updated_runtime/{method}/{num_qudit}/{min_qudit}/{tree_depth}/{num_worker}/log.txt")
                        # time.sleep(2*sleep_time)
                        
                        to_write = open(file_name, 'w')
                        to_write.write(header.format(file=file, method=method, num_qudit=num_qudit, min_qudit=min_qudit,
                                                      tree_depth=tree_depth, num_worker=num_worker))
                        to_write.close()
                        time.sleep(2*sleep_time)
                        print(method, num_qudit, min_qudit, tree_depth, num_worker)
                        output = subprocess.check_output(['sbatch' , file_name])
                        # print(output)
                        time.sleep(sleep_time)