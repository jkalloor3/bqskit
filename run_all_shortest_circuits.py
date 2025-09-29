import time
import subprocess
import os
import glob
from util import get_block_names, check_if_finished


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
conda activate /pscratch/sd/j/jkalloor/my_mpi4py_env
echo "OMP_NUM_THREAD=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 python {file}.py {circ} {timestep} {tol} {diversity}"
OMP_NUM_THREAD=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python {file}.py {circ} {timestep} {tol} {diversity}
"""

cliff_t = True
# cliff_t = False

if __name__ == '__main__':
    file = "get_ensemble_final_small_block_clifft"
    # if cliff_t:
    #     file += "_clifft"
    # file = "run_full_circ_sim"
    # file = "run_full_circ_td"
    # file = "run_full_circ_sv"
    # circs = []
    # circs += ["LiH_hatt"]
    # circs = [f"QITE_8_{i}" for i in range(7)]
    # circs.extend(["qae11", "qaoa10", "qpe_11"])
    # circs += ["qaoa10", "qpe10", "LiH", "mult16"]
    # circs = ["heisenberg7"]
    # circs.extend(["qae11"])
    # circs = ["lgt_17", "mult16", "qpe_14", "qae11", "LiH_jw_long", "FermiHubbard2x2_jw_long", "heisenberg7"] 
    circs = ["mult8"]

    circs = set(circs)

    for circ in circs:
        timesteps = ["all_blocks"]
        # timesteps = ["first_half_blocks", "second_half_blocks"]
        # if cliff_t:
        # timesteps = get_block_names(circ)
        # timesteps = [1.0, 2.0, 3.0, 4.0, 5.0]
        for timestep in timesteps:
            # tols  = [0, 1]
            # tols = [0]
            # tols = [-2.0, -3.0]
            # tols = [1.0, 2.0, 3.0, 4.0, 5.0]
            tols = [5.0]
            # tols = [0]
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
                    has_err = os.system(f"python {file}.py {circ} {timestep} {tol} {diversity}")
                    time.sleep(2*sleep_time)
                    # output = subprocess.check_output(['sbatch' , file_name])
                    # print(output)
                    # time.sleep(sleep_time)
