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
file_dir_name = file_name.split(".")[0]



in_circuit = Circuit.from_file(file_path)

passes =         [
        QuickPartitioner(partition_size),
        SaveIntermediatePass(CHECKPOINT_DIR, f"{file_dir_name}_{partition_size}", save_as_qasm=False)
        ]


task = CompilationTask(in_circuit.copy(), passes)

if not use_detached:
    compiler =  Compiler()
else:
    compiler = Compiler(detached_server_ip, detached_server_port)

compiler.compile(task)
compiler.close()
