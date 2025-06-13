import time
import subprocess
import os
import glob
from util import get_block_names, check_if_finished


sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=02:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}_{diversity}/{circ}/{timestep}/{tol}_tol_block_size

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "python {file}.py {circ} {timestep} {tol} {diversity}"
while true; do free -h >> {file}_{circ}_ram_usage.log; sleep 240; done &
python {file}.py {circ} {timestep} {tol} {diversity}
"""

# sleep 10
# echo "Waiting for 10 seconds"
# sleep 20
# echo "Waiting for 20 seconds"
# python {file}.py {circ} {timestep} {tol}
# sleep 10
# echo "Waiting for 10 seconds"
# sleep 10
# echo "Waiting for 10 seconds"
# python {file}.py {circ} {timestep} {tol}


cliff_t = True
cliff_t = False

if __name__ == '__main__':
    # file = "get_ensemble_final_block"
    # if cliff_t:
    #     file = "get_ensemble_final_block_cliffordt"
    # file = "get_ensemble_final_small_block"
    # file = "minimal_EOF_example_real"
    # file = "run_simulations_block"
    # file = "run_on_qc_block"
    file = "run_small_block_lgt"
    # file = "get_bl    ock_psol_analysis"

    # file = "get_ensemble_final_mapped"

    # file = "initial_optimize"
    
    # Get all circs
    # dirs = ["ensemble_benchmarks", "qce23_qfactor_benchmarks"]
    # # # dirs = ["QITE_8"]
    # # # dirs = ["ham_sim_qasm"]
    # trial_circs = []
    # for dir in dirs:
    #     files = glob.glob(f"{dir}/*.qasm")
    #     trial_circs.extend([file.split('/')[-1].split(".")[0] for file in files])

    # trial_circs = ["qpe10", "qft_8", "qml_19", "heisenberg7", "qft_30", "tfim16"]

    # circs = []
    # circs_to_partition = []
    # for circ in trial_circs:
    #     blocks = get_block_names(circ, extra="")
    #     if len(blocks) == 0:
    #         circs_to_partition.append(circ)
    #     else:
    #         circs.append(circ)
    
    # print("Circs to partition: ", circs_to_partition)
    # exit(0)

    # circs = ["qft_16"]
    # circs = ["draper_adder_12", "LiH"]
    # circs = ["add17", "qpe_14", "mult16", "draper_adder_12", "qae11"]
    # circs = ["qae11"]
    circs = [1.0]

    # tols = [-1.0]
    # tols = [5.0, 4.0, 3.0, 2.0, 1.0]
    # diversities = [1, 0]
    # tols = [0.8, 1.5, 2.0, 2.5]
    # skips = ["vqe", "heisenberg_3"]
    for circ in circs:
        # timesteps = ["all_blocks"]
        # timesteps = [0.5, 0.1, 0.01, 0.001, 0.0001]
        # timesteps = get_block_names(circ, extra="_tket")
        # timesteps = [5.0, 3.0, 1.0]
        timesteps = [1, 0]
        for timestep in timesteps:
            # tols = [""]
            # tols = [0.005, 0.001, 0.0005, 0.0001]
            tols  = [1, 0]
            for tol in tols:
                diversity = ""
                to_write = open(file_name, 'w')
                to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep, diversity=diversity))
                to_write.close()
                print(f"python {file}.py {circ} {timestep} {tol} {diversity}")
                # has_err = os.system(f"python {file}.py {circ} {timestep} {tol} {diversity}")
                time.sleep(2*sleep_time)
                output = subprocess.check_output(['sbatch' , file_name])
                print(output)
                time.sleep(sleep_time)
