import time
import subprocess
import os

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH --time=05:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}{extra}_{unique_circs}/{circ}/{tol}_tol_block_size_6

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "python {file}.py {circ} {timestep} {tol} {unique_circs} 0"
python {file}.py {circ} {timestep} {tol} {unique_circs} 0
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats_new"
    # file = "test_clustering"
    # file = "get_counts"
    # file = "get_ensemble_expectations"
    # file = "get_shortest_circuits_new"
    file = "get_ensemble_final"
    # file = "get_ensemble_final_cliffordt"
    # file = "run_simulations_new"
    # file = "get_shortest_circuits_qsearch"
    # file = "plot_ensemble_data"
    # file = "run_simulations_new"
    # file = "full_compile"
    # circs = ["Heisenberg_7"] #, 
    # circs = ["Heisenberg_7", "TFXY_8"]
    # circs = ["qft_8", "heisenberg7"]
    # circs = ["tfxy_6", "qc_binary_5q"] #, "qc_gray_5q", "qc_optimized_5q"]
    circs = ["vqe_12", "shor_12", "qml_19", "qml_25"]
    circs.extend(["qae13"])
    circs.extend([f"qft_{i}" for i in range(12, 25, 4)])
    # circs.extend([f"JWCirc_{i}" for i in range(1, 4, 2)])
    # circs = ["qae13"]
    # circs = ["hubbard_4"]
    # circs =  ["shor_12", "qft_10", "vqe_12"]
    # tols = range(1, 7)
    tols = [1, 2, 3]
    # tols = [6]
    unique_circss = [100] #, 5, 20, 100, 1000, 10000]
    # extra = "cliffordt"
    # extra = "_clifft"
    extra = "_calc_bias"
    for circ in circs:
        for timestep in [0]:
            for tol in tols:
                for unique_circs in unique_circss:
                    utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_{unique_circs}_circ_final_min_post_calc_bias/{circ}/{tol}/{circ}.pkl"
                    # log_file = f"/pscratch/sd/j/jkalloor/bqskit/slurm_logs/run_simulations_new_post_opt_{unique_circs}/{circ}/{tol}_tol_block_size_8"
                    # utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_{unique_circs}_circ_cliff_t_final/{circ}/{tol}/{circ}.pkl"

                    if os.path.exists(utries_file):
                        continue

                    # if os.path.exists(log_file):
                    #     continue

                    to_write = open(file_name, 'w')
                    to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep, extra=extra, unique_circs=unique_circs))
                    to_write.close()
                    print(f"python {file}.py {circ} {timestep} {tol} {unique_circs}")
                    # os.system(f"python {file}.py {circ} {timestep} {tol} {unique_circs}")
                    time.sleep(2*sleep_time)
                    output = subprocess.check_output(['sbatch' , file_name])
                    print(output)
                    time.sleep(sleep_time)
