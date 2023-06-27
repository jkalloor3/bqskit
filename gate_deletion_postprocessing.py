#%%
from timeit import default_timer as timer
from bqskit import Circuit
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.opt.cost.functions.cost.hilbertschmidt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.instantiaters import QFactor_jax_batched_jit
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes import *
from os.path import join



import jax
from bqskit.passes.search.generators.twounitarylayergen import TwoUnitaryLayerGen, TwoUnitaryLayerGenCeres

jax.config.update('jax_enable_x64', True)
#%%
from bqskit import enable_logging, enable_dashboard


# enable_logging(True)
import params

CHECKPOINT_DIR = "checkpoints"

run_params = params.get_params()
partition_size = run_params.partitions_size
use_detached = run_params.use_detached
detached_server_ip = run_params.detached_server_ip
detached_server_port = run_params.detached_server_port
print(run_params)

file_path = run_params.input_qasm
file_name = file_path.split("/")[-1]
file_dir_name = file_name.split(".")[0] + f"_{partition_size}"


orig_circuit = Circuit.from_file(file_path)
in_circuit = Circuit(orig_circuit.num_qudits)

checkpoint_proj_dir = join(CHECKPOINT_DIR, file_dir_name)

passes =         [
        RestoreIntermediatePass(checkpoint_proj_dir, as_circuit_gate=True),
        UnfoldPass(),
        ToU3Pass()
]


task = CompilationTask(in_circuit, passes)

if not use_detached:
    compiler =  Compiler()
else:
    compiler = Compiler(detached_server_ip, detached_server_port)

out_circuit = compiler.compile(task)

print(f"Circuit finished with gates: {out_circuit.gate_counts}.")
final_gates_count_by_qudit_number = {g.num_qudits:v for g,v in out_circuit.gate_counts.items()}
print(f"True,{in_circuit.num_qudits},{final_gates_count_by_qudit_number[1]},{final_gates_count_by_qudit_number[2]}")

compiler.close()
