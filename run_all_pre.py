import os
import sys
import subprocess 
import time
from os.path import join

file_name = 'job.sh'
circuits_dir = "/pscratch/sd/j/jkalloor/bqskit/qce23_qfactor_benchmarks"
sleep_time = 0.05
global_i = 0

attached_runtime_header="""#!/bin/bash
#SBATCH --job-name=syn{i}_{cir_name}_{instantiator}_attached_pre
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH -n {nodes}
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=./logs_tol_a_0/{cir_name}_{partitions_size}p_{instantiator}_attached_runtime_pre.txt



#alon --output=./logs_qfactor_jax_new_termination_cond/{cir_name}_{partitions_size}p_{instantiator}_attached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt

date
uname -a
module load python
conda activate /global/common/software/m4141/justin_env_2

echo "will run python ./{python_file}  --diff_tol_a 0 {command_args} "

python  ./{python_file}  --diff_tol_a 0 {command_args}

sleep 1
# {amount_of_workers_per_gpu}

"""

def create_and_run_a_job(cir_path, partitions_size, python_file,
        instantiator, num_multistarts, seed, nodes, workers_per_node,
        amount_of_gpus_per_node):
    
    global global_i

    to_write = open(file_name, 'w')
    global_i += 1
    full_cir_path = join(circuits_dir, cir_path)
    command_args = f"--input_qasm {full_cir_path} --multistarts {num_multistarts} --partitions_size {partitions_size} --print_amount_of_nodes {nodes} --instantiator {instantiator}"
    if seed is not None:
        command_args += f" --seed {seed}"

    if amount_of_gpus_per_node == 0:
        amount_of_workers_per_gpu = workers_per_node
    else:
        amount_of_workers_per_gpu = workers_per_node // amount_of_gpus_per_node


    assert(nodes==1)
    runtime_type='attached'
    header_to_use = attached_runtime_header

    cir_name = cir_path.split("/")[-1].split(".qasm")[0]
    to_write.write(header_to_use.format(i=global_i,
                        cir_name=cir_name, instantiator=instantiator,
                         partitions_size=partitions_size, 
                        num_multistarts=num_multistarts, command_args=command_args,
                        nodes = nodes, amount_of_gpus_per_node=amount_of_gpus_per_node,
                        workers_per_node=workers_per_node, python_file=python_file,
                        runtime_type=runtime_type,
                                        amount_of_workers_per_gpu=amount_of_workers_per_gpu))

    to_write.close()
    time.sleep(2*sleep_time)

    subprocess.run(['sbatch' , file_name])
    time.sleep(sleep_time)



if __name__ == '__main__':  

    # circuits =  [f'{f}.qasm' for f in ['qaoa10_u3', 'qaoa12_u3', 'mul10_u3']]

    # circuits =  [('qaoa5.qasm', 5), ('grover5_u3.qasm', 5), ('adder9_u3.qasm', 9), ('hub4.qasm', 4)]
    # circuits =  [ ('qaoa12_u3.qasm', 12)]
    # circuits =  [ ('adder63_u3.qasm', 63), ('shor26.qasm', 26), ('hub18.qasm', 18)]
    # circuits =  [ ('adder63_u3.qasm', 63)]
    # circuits = [ ('tfim8.qasm', 8), ('tfim16.qasm', 16), ('qpe8.qasm', 8), ('qpe10.qasm', 10), ('qpe12.qasm', 12), ('hhl8.qasm', 8),  ('heisenberg8.qasm', 8),   ('heisenberg7.qasm', 7), ('qae11.qasm', 11), ('qae13.qasm', 13)]
    # circuits = [('qae11.qasm', 11), ('qae13.qasm', 13)]
    # circuits =  [ ('shor64.qasm', 64), ('qae33.qasm', 33), ('qae81.qasm', 81), ('mult64.qasm', 64), ('mult16.qasm', 16), ('heisenberg64.qasm', 64),  ('add17.qasm', 17), ('tfim400.qasm', 400)]
    # circuits =  [ ('vqe12.qasm', 12), ('vqe14.qasm', 14), ('qae33.qasm', 33), ('qae81.qasm', 81), ('mult64.qasm', 64), ('mult16.qasm', 16), ('heisenberg64.qasm', 64),  ('add17.qasm', 17)]
    # circuits =  [ ('qae33.qasm', 33), ('qae81.qasm', 81)]
    # circuits =  [ ('hub4.qasm', 4)]
    # circuits =  [ ('tfim16.qasm', 16)]
    # circuits =  [ ('hub18.qasm', 18)]
    circuits =  [('adder63.qasm', 63)]
    # circuits =  [ ('grover5_u3.qasm', 5)]
    # circuits =  [ ('qaoa5.qasm', 5)]
    # circuits =  [ ('adder9_u3.qasm', 9)]
    # circuits = [('hhl8.qasm', 8)]
    # circuits = [('heisenberg7.qasm', 7)]
    # circuits = [('heisenberg64.qasm', 64)]
    
    # instantiators = ['CERES', 'QFACTOR-RUST', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX']
    # instantiators = ['CERES', 'QFACTOR-RUST',  'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST']
    # instantiators = ['CERES', 'QFACTOR-JAX']
    # instantiators = ['CERES']
    # instantiators = ['QFACTOR-JAX']
    instantiators = ['QFACTOR-RUST']
    # instantiators = ['QFACTOR-RUST', 'LBFGS']
    # instantiators = ['CERES', 'LBFGS']
    # instantiators = ['LBFGS']

    partisions_size_l = [9,10]
    #partisions_size_l = [3,4,5,6,7,8]
    # partisions_size_l = [3,4,5,6,7,8, 9,11,12]
    # partisions_size_l = [5, 6, 7,8, 9]
    # partisions_size_l = [13, 14, 15]

    num_multistarts_l = [32]

    seed_l = [None]


    # n_nodes = [4, 2, 1]
    n_nodes = [1]
    # n_nodes = [16]

    # n_workers_per_node = [4*3, 4*4, 4*5, 4*6, 4*7]
    # n_workers_per_node = [4*8, 4*10, 4*12, 4*14, 4*16]
    n_workers_per_node = [6]
    n_amount_of_gpus_in_node=[1]

    use_detached = False
    # use_detached = True

    python_file = 'gate_deletion_preprocessing.py'
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
                                                    amount_of_gpus_per_node=amount_of_gpus_per_node)
