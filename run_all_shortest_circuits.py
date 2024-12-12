import time
import subprocess
import os
import glob

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH --time=4:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}{extra}_{unique_circs}/{circ}/{tol}_tol_block_size_6_{jiggle_skew}

module load conda
conda activate /global/common/software/m4141/ensemble_env_2
echo "python {file}.py {circ} {timestep} {tol} {unique_circs} {jiggle_skew}"
python {file}.py {circ} {timestep} {tol} {unique_circs} {jiggle_skew}
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats_new"
    # file = "test_clustering"
    # file = "get_counts"
    # file = "get_ensemble_expectations"
    # file = "get_shortest_circuits_new"
    # file = "get_ensemble_final_block"
    file = "create_block_data_hist"
    
    # Get all circs
    dir_1 = "ensemble_benchmarks"
    dir_2 = "qce23_qfactor_benchmarks"
    files = glob.glob(f"{dir_1}/*.qasm")
    circs = [file.split('/')[-1].split(".")[0] for file in files]
    files = glob.glob(f"{dir_2}/*.qasm")
    circs.extend([file.split('/')[-1].split(".")[0] for file in files])

    # file = "test_hamiltonian_perturbation"
    # file = "get_ensemble_final_cliffordt"
    # file = "run_simulations_new"
    # file = "get_shortest_circuits_qsearch"
    # file = "plot_ensemble_data"
    # file = "run_simulations_new"
    # file = "full_compile"
    # circs = ["Heisenberg_7"] #, 
    # circs = ["Heisenberg_7", "TFXY_8"]
    # circs = ["adder9"]
    # circs = [5, 6]
    # circs = ["heisenberg_3"]
    # circs = ["tfim_3"] #, "qc_gray_5q", "qc_optimized_5q"]
    # circs = ["vqe_12", "shor_12", "qml_19", "qml_25"]
    # circs = ["qae11", "qpe12"]
    # circs.extend(["qae13"])
    # circs.extend(["mult8", "qpe8"])
    # circs.extend([f"qft_{i}" for i in range(12, 25, 4)])
    # circs.extend([f"JWCirc_{i}" for i in range(1, 4, 2)])
    # circs = ["qae13"]
    # circs = ["hubbard_4"]
    # circs =  ["shor_12", "qft_10", "vqe_12"]
    # tols = range(1, 7)
    tols = [5]
    # tols = [6]
    unique_circss = [500] #, 5, 20, 100, 1000, 10000]
    jiggle_skews = [0]
    # extra = "cliffordt"
    # extra = "_clifft"
    extra = "_block"
    for circ in circs:
        # Get all files of form good_blocks/{circ}_{block_num}.qasm
        # circ_files = glob.glob(f"good_blocks/{circ}_*.qasm")
        # block_nums = [file.split('_')[-1].split('.')[0] for file in circ_files]
        # print(circ_files)
        # print(block_nums)
        block_nums = [0]
        for timestep in block_nums:
            for tol in tols:
                for unique_circs in unique_circss:
                    for jiggle_skew in jiggle_skews:
                        # utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_{unique_circs}_circ_final_min_post_calc_bias/{circ}/{tol}/{circ}.pkl"
                        # # log_file = f"/pscratch/sd/j/jkalloor/bqskit/slurm_logs/run_simulations_new_post_opt_{unique_circs}/{circ}/{tol}_tol_block_size_8"
                        # # utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_{unique_circs}_circ_cliff_t_final/{circ}/{tol}/{circ}.pkl"

                        # if os.path.exists(utries_file):
                        #     continue

                        # if os.path.exists(log_file):
                        #     continue

                        # to_write = open(file_name, 'w')
                        # to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep, extra=extra, unique_circs=unique_circs, jiggle_skew=jiggle_skew))
                        # to_write.close()
                        print(f"python {file}.py {circ} {timestep} {tol} {unique_circs} {jiggle_skew}")
                        os.system(f"python {file}.py {circ} {timestep} {tol} {unique_circs} {jiggle_skew}")
                        time.sleep(2*sleep_time)
                        # output = subprocess.check_output(['sbatch' , file_name])
                        # print(output)
                        # time.sleep(sleep_time)
