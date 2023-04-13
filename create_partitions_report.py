#%%
from bqskit import Circuit
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.passes import *





#%%




# enable_logging(True)
import params

run_params = params.get_params()

file_path = run_params.input_qasm
file_name = file_path.split("/")[-1]




partition_size = run_params.partitions_size

print(f"Will partition {file_path}")

in_circuit = Circuit.from_file(file_path)


passes =         [
        # Convert U3's to VU
        # FromU3ToVariablePass(),

        QuickPartitioner(partition_size),
        RecordPartitionsStatsPass(),
        # SaveIntermediatePass('/pscratch/sd/a/alonkukl/part_stats/partitions_qasms/', f'{file_name}.{partition_size}')
        SaveIntermediatePass('/global/homes/a/alonkukl/part_stats/partitions_qasms/', f'{file_name}.{partition_size}')
        # SaveIntermediatePass('/pscratch/sd/a/alonkukl/part_stats/partitions_qasms/', f'{file_name}.{partition_size}', False)
        
        ]


task = CompilationTask(in_circuit, passes)
print(f"Circuit has gates: {in_circuit.gate_counts}.")

with Compiler(num_workers=2) as compiler:          
            task_id = compiler.submit(task)
            cout = compiler.result(task_id) # type: ignore


# for p in passes:
#     in_circuit.perform(p)
