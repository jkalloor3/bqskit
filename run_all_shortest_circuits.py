import time
import os
import subprocess


sleep_time = 0.05
file_name = 'job.sh'

# /pscratch/sd/j/jkalloor/my_mpi4py_env
# /global/common/software/m4141/ensemble_env_2

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=11:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}_{diversity}/{circ}/{timestep}/{tol}_tol_block_size

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "OMP_NUM_THREAD=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python {file}.py {circ} {timestep} {tol} {diversity}"
OMP_NUM_THREAD=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python {file}.py {circ} {timestep} {tol} {diversity}
"""

cliff_t = True
# cliff_t = False

if __name__ == '__main__':
    file = "get_ensemble_final_small_block"

    circs = ["LiH_jw_long"]

    for circ in circs:
        timesteps = ["all_blocks"]
        for timestep in timesteps:
            tols = [-1.0]
            for tol in tols:
                diversities = [1]
                for diversity in diversities:
                    to_write = open(file_name, 'w')
                    to_write.write(header.format(file=file, circ=circ, tol=tol, 
                                                 timestep=timestep, 
                                                 diversity=diversity))
                    to_write.close()
                    time.sleep(2*sleep_time)
                    print(f"python {file}.py {circ} {timestep} {tol} {diversity}")
                    output = subprocess.check_output(['sbatch' , file_name])
                    print(output)
