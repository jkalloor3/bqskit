import time
import subprocess

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=06:25:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/expectations/{method}/{circ}/{tol}_tol/block_size_{m}

module load python
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
echo "{file}.py {circ} {timestep} {method} {tol} {m}"
TF_CPP_MIN_LOG_LEVEL=0 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 python {file}.py {circ} {method} {tol} {m} 7 6
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "get_ensemble_expectations"
    # file = "full_compile"
    circs = ["Heisenberg_7", "TFXY_8"]
    # tols = range(1, 7)
    tols = [1,3,5]
    methods = ["jiggle_post_gpu"] #, "treescan"]
    part_sizes = [3]
    for circ in circs:
        for timestep in range(10, 11):
            for method in methods:
                for tol in tols:
                    for m in [6]:
                        to_write = open(file_name, 'w')
                        to_write.write(header.format(file=file, circ=circ, method=method, tol=tol, m=m, timestep=timestep))
                        to_write.close()
                        time.sleep(2*sleep_time)
                        print(circ, method, tol)
                        output = subprocess.check_output(['sbatch' , file_name])
                        print(output)
                        time.sleep(sleep_time)
