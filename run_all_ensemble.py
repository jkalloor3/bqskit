import time
import subprocess
import os

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=06:50:00
#SBATCH -N 1
#SBATCH --signal=B:USR1@1
#SBATCH --output=./slurm_logs/pams/{method}/{circ}/{tol}_tol/block_size_{m}

module load conda
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
echo "python {file}.py {circ} {timestep} {method} {tol} {m}"
python {file}.py {circ} {timestep} {method} {tol} {m} 8 {m}
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    # file = "get_ensemble_expectations"
    file = "get_superensemble"
    # file = "run_simulations"
    # file = "full_compile"
    # circs = ["Heisenberg_7"] #, 
    # circs = ["Heisenberg_7", "TFXY_8"]
    circs = ["hubbard_4", "heisenberg_3", "shor_12", "vqe_12", "vqe_14", "qft_10"]
    # tols = range(1, 7)
    tols = [8]#,3,5]
    # methods = ["pam"] #, "treescan"]
    methods = ["jiggle"]
    part_sizes = [3]
    for circ in circs:
        for timestep in range(1,2):
            for method in methods:
                for tol in tols:
                    for m in part_sizes:
                        # if circ == "TFXY_8":
                        #     m = 7
                        # param_file = f"ensemble_approx_circuits_qfactor/{method}/{circ}/{tol}/{m}/{timestep}/jiggled_circ.pickle"
                        param_file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_approx_circuits_pam/pam/{circ}/{tol}/{m}/{timestep}/circ.pickle"

                        if (os.path.exists(param_file)):
                            continue

                        to_write = open(file_name, 'w')
                        to_write.write(header.format(file=file, circ=circ, method=method, tol=tol, m=m, timestep=timestep))
                        to_write.close()
                        print(f"python {file}.py {circ} {timestep} {method} {tol} {m} 8 {m}")
                        os.system(f"python {file}.py {circ} {timestep} {method} {tol} {m} 8 {m}")
                        time.sleep(2*sleep_time)
                        # output = subprocess.check_output(['sbatch' , file_name])
                        # print(output)
                        # time.sleep(sleep_time)
