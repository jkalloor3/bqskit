import time
import subprocess
import os

sleep_time = 0.05
file_name = 'job_detached.sh'

header = """#!/bin/bash
#SBATCH --job-name={circ}_{m}
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 11:50:00
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --gpus=4
#SBATCH --output=./slurm_logs/detached/{method}/{circ}/{tol}_tol/block_size_{m}
date
uname -a
module load python
module load cudnn/8.9.3_cuda12
module load nccl/2.18.3-cu12
conda activate /pscratch/sd/j/jkalloor/justin_env_clone
#module load nvidia
echo "starting BQSKit managers on all nodes"
pwd
ls run_workers_and_managers.sh
source /pscratch/sd/j/jkalloor/bqskit/run_workers_and_managers.sh 4 {num_workers} &
#srun run_workers_and_managers.sh 4 4 &
managers_pid=$!
filename=$SCRATCH/managers_${{SLURM_JOB_ID}}_started
n=1
while [[ ! -f "$filename" ]]
do
        sleep 0.5
done
while [ "$(cat "$filename" | wc -l)" -lt "$n" ]; do
    sleep 1
done
echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ') &> $SCRATCH/bqskit_logs/server_${{SLURM_JOB_ID}}.log &
server_pid=$!
uname -a >> $SCRATCH/server_${{SLURM_JOB_ID}}_started
echo $server_pid
echo "python {file}.py {circ} {timestep} {method} {tol} {m}"
TF_CPP_MIN_LOG_LEVEL=0 XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1,2,3 python {file}.py {circ} {timestep} {method} {tol} {m}
date
echo "Killing the server"
kill -2 $server_pid
sleep 2
"""

if __name__ == '__main__':
    # mesh_gates = ["cx", "ecr"]
    # file = "get_ensemble_stats"
    file = "get_superensemble_detached"
    # file = "full_compile"
    circs = ["TFXY_t"]
    circs = ["Heisenberg_7", "QITE_8", "TFXY_8"]
    # tols = range(1, 7)
    tols = [7]
    methods = ["gpu"] #, "treescan"]
    # part_sizes = [3]
    for circ in circs:
        for timestep in range(10, 20):
            for method in methods:
                for tol in tols:
                    if "7" in circ:
                        ms = [6]
                    else:
                        ms = [6,7]
                    for m in ms:
                        num_workers = 64 if m == 6 else 32
                        dir = f"ensemble_approx_circuits_qfactor/{method}_real/{circ}"
                        if os.path.exists(f"{dir}/jiggled_circs/{tol}/{m}/{timestep}/jiggled_circ.pickle"):
                            continue
                        to_write = open(file_name, 'w')
                        to_write.write(header.format(file=file, circ=circ, method=method, tol=tol, m=m, timestep=timestep, num_workers=num_workers))
                        to_write.close()
                        time.sleep(2*sleep_time)
                        print(circ, method, tol)
                        output = subprocess.check_output(['sbatch' , file_name])
                        print(output)
                        time.sleep(sleep_time)