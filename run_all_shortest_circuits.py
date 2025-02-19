import time
import subprocess
import os
import glob
from util import get_block_names


sleep_time = 0.05
file_name = 'job.sh'



header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=09:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}_tket/{circ}/{tol}_tol_block_size

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "python {file}.py {circ} {timestep} {tol} _tket"
python {file}.py {circ} {timestep} {tol} _tket
"""

cliff_t = True
cliff_t = False

if __name__ == '__main__':
    file = "get_ensemble_final_block"
    if cliff_t:
        file = "get_ensemble_final_block_cliffordt"

    # file = "initial_optimize"
    
    # Get all circs
    dirs = ["ensemble_benchmarks", "qce23_qfactor_benchmarks"]
    # dirs = ["QITE_8"]
    # dirs = ["ham_sim_qasm"]
    trial_circs = []
    for dir in dirs:
        files = glob.glob(f"{dir}/*.qasm")
        trial_circs.extend([file.split('/')[-1].split(".")[0] for file in files])

    circs = []
    circs_to_partition = []
    for circ in trial_circs:
        blocks = get_block_names(circ, extra="_tket")
        if len(blocks) == 0:
            circs_to_partition.append(circ)
        else:
            circs.append(circ)
    
    print("Circs to partition: ", circs_to_partition)

    tols = [-1.0]
    skips = ["vqe", "heisenberg_3", "tf"]
    for circ in circs:
        timesteps = ["all_blocks"]
        # timesteps = ["1"]
        # for skip in skips:
        #     if circ.startswith(skip):
        #         timesteps = []
        # timesteps = ["1"]
        for timestep in timesteps:
            for tol in tols:
                # print(check_if_finished(circ, tol))
                # if check_if_finished(circ, tol, cliff_t=cliff_t)[1]:
                #     print(f"Skipping {circ} {tol}, already finished")
                #     continue
                # else:
                to_write = open(file_name, 'w')
                to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep))
                to_write.close()
                print(f"python {file}.py {circ} {timestep} {tol}")
                # os.system(f"python {file}.py {circ} {timestep} {tol}")
                time.sleep(2*sleep_time)
                output = subprocess.check_output(['sbatch' , file_name])
                print(output)
                time.sleep(sleep_time)
