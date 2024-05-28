import time
import subprocess
import os

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=03:50:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/{file}/{circ}/{tol}_tol

module load conda
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
echo "python {file}.py {circ} {timestep} {tol}"
python {file}.py {circ} {timestep} {tol} 1000
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats_new"
    # file = "get_ensemble_expectations"
    file = "get_shortest_circuits"
    # file = "plot_ensemble_data"
    # file = "run_simulations"
    # file = "full_compile"
    # circs = ["Heisenberg_7"] #, 
    # circs = ["Heisenberg_7", "TFXY_8"]
    circs = ["tfxy_6", "qc_binary_5q", "qc_gray_5q"]
    # tols = range(1, 7)
    tols = [1,2,3,4,5]#,3,5]
    for circ in circs:
        for timestep in [0]:
            for tol in tols:
                # if circ == "TFXY_8":
                #     m = 7
                # param_file = f"ensemble_approx_circuits_qfactor/{method}/{circ}/{tol}/{m}/{timestep}/jiggled_circ.pickle"
                # param_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits/{circ}/{tol}/{timestep}/{circ}.pkl"

                # if (os.path.exists(param_file)):
                #     continue

                to_write = open(file_name, 'w')
                to_write.write(header.format(file=file, circ=circ, tol=tol, timestep=timestep))
                to_write.close()
                print(f"python {file}.py {circ} {timestep} {tol}")
                # os.system(f"python {file}.py {circ} {timestep} {tol}")
                # time.sleep(2*sleep_time)
                output = subprocess.check_output(['sbatch' , file_name])
                print(output)
                time.sleep(sleep_time)
