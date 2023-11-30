import time
import subprocess

sleep_time = 0.05
file_name = 'job.sh'

header = """#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH --time=11:55:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --output=./slurm_logs/{method}/{circ}/{tol}

module load python
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
echo "get_superensemble.py {circ} {method} {tol}"
python get_superensemble.py {circ} {method} {tol}
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    circs = ["TFIM", "Heisenberg"]
    tols = range(3, 10)
    methods = ["treescan"]
    # device_gates = ["cx-b"]
    part_sizes = [3]
    for circ in circs:
        for method in methods:
            for tol in tols:
                to_write = open(file_name, 'w')
                to_write.write(header.format(circ=circ, method=method, tol=tol))
                to_write.close()
                time.sleep(2*sleep_time)
                print(circ, method, tol)
                output = subprocess.check_output(['sbatch' , file_name])
                print(output)
                time.sleep(sleep_time)
