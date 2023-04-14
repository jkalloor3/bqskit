
import json
import time

import jax
from bqskit.compiler.basepass import BasePass
from inst_pass import InstPass
from bqskit.ir.circuit import Circuit
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.opt.cost.functions.cost.hilbertschmidt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.passes.util.convertu3tovar import FromU3ToVariablePass
from bqskit.runtime import get_runtime

import params

jax.config.update('jax_enable_x64', True)

run_params = params.get_params()
print(run_params)

file_path = run_params.input_qasm
file_name = file_path.split("/")[-1]

seed = run_params.seed

num_multistarts = run_params.multistarts
instantiator = run_params.instantiator
max_iters = run_params.max_iters
min_iters = run_params.min_iters
diff_tol_r = run_params.diff_tol_r
diff_tol_a = run_params.diff_tol_a
dist_tol = run_params.dist_tol

diff_tol_step_r = run_params.diff_tol_step_r
diff_tol_step = run_params.diff_tol_step
beta = run_params.beta

print(f"Will compile {file_path}")



instantiators_opts =  {
                    'QFACTOR-RUST':{
                        
                            'method': 'qfactor',
                            'diff_tol_r':diff_tol_r,
                            'diff_tol_a':diff_tol_a,
                            'min_iters':min_iters,
                            'dist_tol':dist_tol,  
                            'max_iters': max_iters,
                            'multistarts': num_multistarts,
                            'seed':seed,
                        },
                    'QFACTOR-JAX': {
                            'method': 'qfactor_jax_batched_jit',
                            'diff_tol_r':diff_tol_r,
                            'diff_tol_a':diff_tol_a,
                            'min_iters':min_iters,
                            'dist_tol':dist_tol,  
                            'max_iters': max_iters,
                            'diff_tol_step_r':diff_tol_step_r,
                            'diff_tol_step': diff_tol_step,
                            'multistarts': num_multistarts,
                            'beta': beta,
                            'seed':seed,
                    },

                    'LBFGS':{
                            'method':'minimization',
                            'minimizer':LBFGSMinimizer(),
                            'multistarts': num_multistarts,
                            'cost_fn_gen': HilbertSchmidtCostGenerator(),     
                            'seed':seed,
                    },

                    'CERES':{
                            'multistarts': num_multistarts,
                            'seed':seed,
                    }
            }


instantiate_options = instantiators_opts[instantiator.replace("_P", "")]



circ =  Circuit.from_file(file_path)

if 'QFACTOR' in instantiator:
    with Compiler(num_workers=1) as compiler:
        task = CompilationTask(circ, [FromU3ToVariablePass()])
        task_id = compiler.submit(task)
        circ = compiler.result(task_id) # type: ignore


# circ2 = circ.copy()
# if 'JAX' in instantiator:
#         circ2.pop_qudit(0)
# target = circ2.get_unitary()


# tic = time.perf_counter()

# circ2.instantiate(
# target,
# **instantiate_options
# )
# inst_time = time.perf_counter() - tic


# inst_dist_from_target = circ2.get_unitary().get_distance_from(target, 1)
# print('***' + json.dumps([inst_time, inst_dist_from_target]))

target = circ.get_unitary()
if '_P'  in instantiator:
    #Create run time with many workers and send them the job
    #The do it by creating a compiler
    with Compiler(num_workers=num_multistarts+1) as compiler:
        task = CompilationTask(circ, [InstPass(instantiate_options, target)])
        tic = time.perf_counter()
        
        task_id = compiler.submit(task)
        circ = compiler.result(task_id) # type: ignore
        # circ = asyncio.run(multi_start_instantiate_async(instantiator,circ, target, num_multistarts))
        inst_time = time.perf_counter() - tic
        inst_dist_from_target = circ.get_unitary().get_distance_from(target, 1)

else:

    tic = time.perf_counter()
    circ.instantiate(
        target,
        **instantiate_options
        )
    inst_time = time.perf_counter() - tic
    inst_dist_from_target = circ.get_unitary().get_distance_from(target, 1)



print('***' + json.dumps([inst_time, inst_dist_from_target]))
      
