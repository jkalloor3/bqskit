import time
import subprocess

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=3:55:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --output=./slurm_logs/{method}/{circ}/{tol}

module load python
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
echo "{file}.py {circ} {timestep} {method} {tol} {m}"
python {file}.py {circ} {timestep} {method} {tol} {m}
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "get_superensemble"
    circs = ["TFIM"]
    tols = range(1, 5)
    methods = ["leap"]
    # device_gates = ["cx-b"]
    part_sizes = [3]
    for circ in circs:
        for timestep in range(10, 42):
            for method in methods:
                for tol in tols:
                    # for m in [10, 20, 40, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]:
                    for m in [1]:
                        to_write = open(file_name, 'w')
                        to_write.write(header.format(file=file, circ=circ, method=method, tol=tol, m=m, timestep=timestep))
                        to_write.close()
                        time.sleep(2*sleep_time)
                        print(circ, method, tol)
                        output = subprocess.check_output(['sbatch' , file_name])
                        print(output)
                        time.sleep(sleep_time)
