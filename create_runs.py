import os
import sys
import subprocess 
import time

file_name = 'job.sh'
sleep_time = 0.05
global_i = 0

detached_runtime_header="""#!/bin/bash
#SBATCH --job-name=syn{i}_{cir_name}_{instantiator}_detached_runtime
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 11:50:00
#SBATCH -n {nodes}
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={amount_of_gpus_per_node}
#SBATCH --output=./logs_tol_a_0/{cir_name}_{partitions_size}p_{instantiator}_detached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt

date
uname -a
module load python
conda activate my_env
module load nvidia



echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh {amount_of_gpus_per_node} {amount_of_workers_per_gpu} &
managers_pid=$!

filename=$SCRATCH/managers_${{SLURM_JOB_ID}}_started
n={nodes}

while [[ ! -f "$filename" ]]
do
        sleep 0.5
done

while [ "$(cat "$filename" | wc -l)" -lt "$n" ]; do
    sleep 1
done

echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\\n' ' ') &> $SCRATCH/bqskit_logs/server_${{SLURM_JOB_ID}}.log &
server_pid=$!

uname -a >> $SCRATCH/server_${{SLURM_JOB_ID}}_started

echo "will run {env_vars} python  ./{python_file} --diff_tol_a 0  --use_detached {command_args} "

{env_vars} python  ./{python_file} --diff_tol_a 0 --use_detached {command_args}


echo "Killing the server"
kill -2 $server_pid

sleep 2

"""


attached_runtime_header="""#!/bin/bash
#SBATCH --job-name=syn{i}_{cir_name}_{instantiator}_attached_runtime
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -n {nodes}
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={amount_of_gpus_per_node}
#SBATCH --output=./logs_tol_a_0/{cir_name}_{partitions_size}p_{instantiator}_attached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt

date
uname -a
module load python
conda activate my_env
module load nvidia

echo "starting MPS server on node"
nvidia-cuda-mps-control -d

echo "will run {env_vars} python  ./{python_file}  --diff_tol_a 0 {command_args} "

{env_vars} python  ./{python_file}  --diff_tol_a 0 {command_args}

sleep 1
echo "Trying to stop MPS on all nodes"
echo quit | nvidia-cuda-mps-control

# {amount_of_workers_per_gpu}

"""


def create_and_run_a_job(cir_path, partitions_size, python_file,
        instantiator, num_multistarts, seed, nodes, workers_per_node,
        amount_of_gpus_per_node, use_detached):
    
    global global_i

    to_write = open(file_name, 'w')
    global_i += 1
    command_args = f"--input_qasm {cir_path} --multistarts {num_multistarts} --partitions_size {partitions_size} --print_amount_of_nodes {nodes} --instantiator {instantiator}"
    env_vars = ""
    
    if seed is not None:
        command_args += f" --seed {seed}"

    if instantiator == 'QFACTOR-JAX':

        command_args += f' --amount_of_workers {workers_per_node}'
        command_args += f' --amount_gpus_per_node {amount_of_gpus_per_node}'
        if not use_detached:
            env_vars += " XLA_PYTHON_CLIENT_PREALLOCATE=false"
            env_vars += " CUDA_VISIBLE_DEVICES=" + str(list(range(amount_of_gpus_per_node)))[1:-1].replace(" ","")
    else:
        if not use_detached:
            workers_per_node = -1 
        command_args += f' --amount_of_workers {workers_per_node}'
        command_args += f' --amount_gpus_per_node 0'

    if amount_of_gpus_per_node == 0:

        amount_of_workers_per_gpu = workers_per_node
    else:
        amount_of_workers_per_gpu = workers_per_node // amount_of_gpus_per_node


    if use_detached:
        runtime_type='detached'
        header_to_use = detached_runtime_header
    else:
        assert(nodes==1)
        runtime_type='attached'
        header_to_use = attached_runtime_header

    cir_name = cir_path.split("/")[-1].split(".qasm")[0]
    to_write.write(header_to_use.format(i=global_i,
                        cir_name=cir_name, instantiator=instantiator,
                         partitions_size=partitions_size, 
                        num_multistarts=num_multistarts, 
                        env_vars=env_vars, command_args=command_args,
                        nodes = nodes, amount_of_gpus_per_node=amount_of_gpus_per_node,
                        workers_per_node=workers_per_node, python_file=python_file,
                        runtime_type=runtime_type,
                                        amount_of_workers_per_gpu=amount_of_workers_per_gpu))

    to_write.close()
    time.sleep(2*sleep_time)

    subprocess.check_output(['sbatch' , file_name])
    time.sleep(sleep_time)



if __name__ == '__main__':

    # circuits =  [f'{f}.qasm' for f in ['qaoa10_u3', 'qaoa12_u3', 'mul10_u3']]

    # circuits =  [('qaoa5.qasm', 5), ('grover5_u3.qasm', 5), ('adder9_u3.qasm', 9), ('hub4.qasm', 4)]
    # circuits =  [ ('qaoa12_u3.qasm', 12)]
    # circuits =  [ ('adder63_u3.qasm', 63), ('shor26.qasm', 26), ('hub18.qasm', 18)]
    # circuits =  [ ('adder63_u3.qasm', 63)]
    # circuits =  [ ('hub4.qasm', 4)]
    circuits =  [ ('hub18.qasm', 18)]
    # circuits =  [ ('shor26.qasm', 26)]
    # circuits =  [ ('grover5_u3.qasm', 5)]
    # circuits =  [ ('qaoa5.qasm', 5)]
    # circuits =  [ ('adder9_u3.qasm', 9)]

    # instantiators = ['CERES', 'QFACTOR-RUST', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX']
    # instantiators = ['CERES', 'QFACTOR-RUST',  'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST']
    # instantiators = ['CERES', 'QFACTOR-JAX']
    # instantiators = ['CERES']
    # instantiators = ['QFACTOR-JAX']
    # instantiators = ['QFACTOR-RUST']
    # instantiators = ['QFACTOR-RUST', 'LBFGS']
    instantiators = ['LBFGS']

    partisions_size_l = [4,5, 6]
    #partisions_size_l = [3,4,5,6,7,8]
    # partisions_size_l = [3,4,5,6,7,8, 9,11,12]
    # partisions_size_l = [5, 6, 7,8, 9]
    # partisions_size_l = [13, 14, 15]

    num_multistarts_l = [32]

    seed_l = [None]


    # n_nodes = [4, 2, 1]
    # n_nodes = [1]
    n_nodes = [8]

    n_workers_per_node = [-1]
    n_amount_of_gpus_in_node=[0]

    # use_detached = False
    use_detached = True

    python_file = 'gate_deletion_perf_measurement.py'
    # python_file = 'gate_deletion_perf_measurement_test.py'

    for amount_of_gpus_per_node in n_amount_of_gpus_in_node:
        for workers_per_node in n_workers_per_node:
            for nodes in n_nodes:
                for cir_path, qubit_count in circuits:
                    for partitions_size in partisions_size_l:
                        if partitions_size > qubit_count:
                             continue
                        for seed in seed_l:               
                            for num_multistarts in num_multistarts_l:
                                for instantiator in instantiators:
                                            create_and_run_a_job(cir_path,
                                                    partitions_size, python_file,
                                                    instantiator=instantiator,
                                                    num_multistarts=num_multistarts,
                                                    seed=seed,
                                                    nodes=nodes,
                                                    workers_per_node=workers_per_node,
                                                    amount_of_gpus_per_node=amount_of_gpus_per_node,
                                                    use_detached=use_detached)
