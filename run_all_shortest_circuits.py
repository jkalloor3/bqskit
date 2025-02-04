import time
import subprocess
import os
import glob
from util import check_if_finished


sleep_time = 0.05
file_name = 'job.sh'



header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH --time=5:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}/{circ}/{tol}_tol_block_size

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "python {file}.py {circ} {timestep} {tol}"
python {file}.py {circ} {timestep} {tol}
"""

one_q_err = 1e-4
two_q_err = 4e-3

if __name__ == '__main__':
    file = "get_ensemble_final_block"
    
    # Get all circs
    # dirs = ["ensemble_benchmarks", "qce23_qfactor_benchmarks"]
    dirs = ["QITE_8"]
    circs = []
    for dir in dirs:
        files = glob.glob(f"{dir}/*.qasm")
        circs.extend([file.split('/')[-1].split(".")[0] for file in files])
    # circs = ["adder9"]

    # circs = ["shor_12"]

    tols = [1.0, 3.0]
    skips = ["vqe", "heisenberg_3", "tfim_3", "qft", "tf"]
    for circ in circs:
        timesteps = ["all_blocks"]
        for skip in skips:
            if circ.startswith(skip):
                timesteps = []
        for timestep in timesteps:
            for tol in tols:
                # print(check_if_finished(circ, tol))
                if check_if_finished(circ, tol)[1]:
                    print(f"Skipping {circ} {tol}, already finished")
                    continue
                else:
                    to_write = open(file_name, 'w')
                    to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep))
                    to_write.close()
                    print(f"python {file}.py {circ} {timestep} {tol}")
                    # os.system(f"python {file}.py {circ} {timestep} {tol}")
                    time.sleep(2*sleep_time)
                    output = subprocess.check_output(['sbatch' , file_name])
                    print(output)
                    time.sleep(sleep_time)
