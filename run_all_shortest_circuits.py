import time
import subprocess
import os

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=03:55:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}{extra}_{unique_circs}/{circ}/{tol}_tol_block_size_8

module load conda
conda activate /pscratch/sd/j/jkalloor/ensemble_env
echo "python {file}.py {circ} {timestep} {tol} {unique_circs}"
python {file}.py {circ} {timestep} {tol} {unique_circs}
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats_new"
    # file = "test_clustering"
    # file = "get_counts"
    # file = "get_ensemble_expectations"
    # file = "get_shortest_circuits_new"
    file = "get_ensemble_final"
    # file = "run_simulations_new"
    # file = "get_shortest_circuits_qsearch"
    # file = "plot_ensemble_data"
    # file = "run_simulations"
    # file = "full_compile"
    # circs = ["Heisenberg_7"] #, 
    # circs = ["Heisenberg_7", "TFXY_8"]
    # circs = ["tfxy_6", "qc_binary_5q"] #, "qc_gray_5q", "qc_optimized_5q"]
    circs = ["heisenberg7", "vqe_12", "shor_12", "qml_19", "qml_25"]
    # circs = [f"qft_{i}" for i in range(8, 24, 2)]
    # circs = ["hubbard_4"]
    # circs =  ["shor_12", "qft_10", "vqe_12"]
    # tols = range(1, 7)
    tols = [1,3]
    unique_circss = [100] #, 5, 20, 100, 1000, 10000]
    extra = ""
    for circ in circs:
        for timestep in [0]:
            for tol in tols:
                for unique_circs in unique_circss:
                    # if circ == "TFXY_8":
                    #     m = 7
                    # param_file = f"ensemble_approx_circuits_qfactor/{method}/{circ}/{tol}/{m}/{timestep}/jiggled_circ.pickle"
                    # param_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ}/{tol}/{timestep}/{circ}.pkl"
                    utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_{unique_circs}_circ_final/{circ}/{tol}/{circ}.pkl"
                    # utries_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits_100_circ_final/qc_binary_5q/1/qc_binary_5q.pkl"
                    # graph_file = f"/pscratch/sd/j/jkalloor/bqskit/{circ}_{tol}_errors_comp.png"

                    # if os.path.exists(graph_file):
                    #     print(f"Skipping {graph_file}")
                    #     continue

                    if os.path.exists(utries_file):
                        continue


                    to_write = open(file_name, 'w')
                    to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep, extra=extra, unique_circs=unique_circs))
                    to_write.close()
                    print(f"python {file}.py {circ} {timestep} {tol} {unique_circs}")
                    # os.system(f"python {file}.py {circ} {timestep} {tol} {unique_circs}")
                    # time.sleep(2*sleep_time)
                    output = subprocess.check_output(['sbatch' , file_name])
                    print(output)
                    time.sleep(sleep_time)
