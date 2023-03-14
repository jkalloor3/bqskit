#%%
from timeit import default_timer as timer
from bqskit import Circuit
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.opt.instantiaters import QFactor_jax_batched_jit
from bqskit.ir.opt.instantiaters import QFactor
from bqskit.passes import *



import jax
from bqskit.passes.search.generators.twounitarylayergen import TwoUnitaryLayerGen, TwoUnitaryLayerGenCeres

jax.config.update('jax_enable_x64', True)
#%%
from bqskit import enable_logging, enable_dashboard


enable_logging(True)
import params

run_params = params.get_params()

use_qfactor = run_params.use_qfactor
file_path = run_params.input_qasm
file_name = file_path.split("/")[-1]
partition_size = run_params.partitions_size
num_multistarts = run_params.multistarts
iterations_with_out_dask = run_params.iterations_no_dask
should_use_dask = run_params.use_dask
seed = run_params.seed
use_rust = run_params.use_rust
output_in_u3s_cnots = run_params.output_in_u3s_cnots
use_variable_untry = run_params.use_variable_untry
scheduler_file = run_params.scheduler_file
nodes = run_params.nodes
print_amount_of_workers_per_gpu = run_params.print_amount_of_workers_per_gpu
print_amount_gpus_in_run = run_params.print_amount_gpus_in_run
gate_size = run_params.gate_size
calculate_error_bound = run_params.calculate_error_bound



if should_use_dask:
    dask_type = "DASK"
else:
    dask_type = "No DASK"

print(f"Will compile {file_path}")

batched_instantiation = QFactor_jax_batched_jit(diff_tol_r=1e-5, diff_tol_a=1e-10, min_iters=10, max_iters=100000, dist_tol=1e-10)

def replace_filer(new_circuit, old_op):
    old_ops = old_op.gate._circuit.num_operations
    new_ops = new_circuit.num_operations

    return new_ops < old_ops

layer_generator = SimpleLayerGenerator()
if use_qfactor:
    if use_rust:
        instantiation_type = "RUST Qfactor "
        instantiate_options={
                    'method': 'QFactor',
                    'diff_tol_r':1e-5,
                    'diff_tol_a':1e-10,
                    'min_iters':10,
                    'dist_tol':1e-10,     # Stopping criteria for distance
                    'max_iters': 100000,
                    'multistarts': num_multistarts,
                    'seed': seed,
                    'parallel': False,
                }
    else:
        instantiation_type = "JAX Qfactor "
    
        instantiate_options={
                    'method': batched_instantiation,
                    'multistarts': num_multistarts,
                    'seed': seed,
                }

    if use_variable_untry:
        layer_generator=TwoUnitaryLayerGen()
else:
    instantiation_type = "CERES "
    instantiate_options={
                    'multistarts': num_multistarts,
                    'seed': seed,
                }
    if use_variable_untry:
        layer_generator=TwoUnitaryLayerGenCeres()




in_circuit = Circuit.from_file(file_path)

passes =         [
        # Convert U3's to VU
        # BlockConversionPass('variable', convert_constant=False),
        FromU3ToVariablePass(),
        QuickPartitioner(partition_size),
        # Delete gates using qfactor
        ForEachBlockPass([
            WhileLoopPass(GateCountPredicate(CXGate()),
                [
                    ScanningGateRemovalPass(instantiate_options=instantiate_options),  
                ]),
                # ], calculate_error_bound=calculate_error_bound, replace_filter=replace_filer),
                                    # ], calculate_error_bound=calculate_error_bound),
        ]),
        UnfoldPass(),
        # Convert back to u3 the VU
        ToU3Pass()
        ]
if not use_qfactor:
    passes = passes[1:-1] # no need to convert to variable and then back to U3s
task = CompilationTask(in_circuit.copy(), passes)

if should_use_dask:
    print(f"using s_file {scheduler_file}")
    compiler =  Compiler() if scheduler_file is None else Compiler(scheduler_file=scheduler_file)
else:
    out_circuit = in_circuit.copy()

print(f"Starting {instantiation_type}")
start = timer()
if should_use_dask:
    out_circuit = compiler.compile(task)
else:
    data = {}
    for pas in passes:
        pas.run(out_circuit, data)
end = timer()
run_time = end - start
print(
    f"Partitioning + Synthesis took {run_time}"
    f"seconds using the { instantiation_type } instantiation method."
)
print(f"Circuit finished with gates: {out_circuit.gate_counts}.")
final_gates_count_by_qudit_number = {g.num_qudits:v for g,v in out_circuit.gate_counts.items()}
print(f"{use_variable_untry},{instantiation_type},{dask_type},{file_name},{num_multistarts},{partition_size},{in_circuit.num_qudits},{run_time},{final_gates_count_by_qudit_number[1]},{final_gates_count_by_qudit_number[2]},{output_in_u3s_cnots},{nodes},{print_amount_of_workers_per_gpu},{print_amount_gpus_in_run}")

compiler.close()