#%%
from timeit import default_timer as timer
from bqskit import Circuit
from bqskit.compiler import Compiler, CompilationTask
# from bqskit.ir.gates.constant.cx import CXGate
# from bqskit.ir.opt.cost.functions.cost.hilbertschmidt import HilbertSchmidtCostGenerator
# from bqskit.ir.opt.instantiaters import QFactor_jax_batched_jit
# from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes import *



# import jax
# from bqskit.passes.search.generators.twounitarylayergen import TwoUnitaryLayerGen, TwoUnitaryLayerGenCeres

# jax.config.update('jax_enable_x64', True)
#%%
from bqskit import enable_logging, enable_dashboard

file_path = "adder9.qasm"
in_circuit = Circuit.from_file(file_path)

while_scanning_removal = [WhileLoopPass(ChangePredicate(), [ #RestoreIntermediatePass("int_blocks/scan"), 
                                                            OneGateRemovalPass(), 
                                                            #SaveIntermediatePass("int_blocks", "scan", save_as_qasm=False)
                                                            ])]

operations_to_perfrom_on_block = [
                    ScanningGateRemovalPass(),  
                ]

operations_to_perfrom_on_block_while = while_scanning_removal  

passes =         [
        # Convert U3's to VU
        FromU3ToVariablePass(),

        QuickPartitioner(3),
        ForEachBlockPass(operations_to_perfrom_on_block_while),
        UnfoldPass(),
        
        # Convert back to u3 the VU
        ToU3Pass()
        ]

task = CompilationTask(in_circuit.copy(), passes)

compiler =  Compiler(num_workers=1)


start = timer()

out_circuit = compiler.compile(task)

end = timer()
run_time = end - start
print(
    f"Partitioning + Synthesis took {run_time}"
)
print(f"Circuit finished with gates: {out_circuit.gate_counts}.")
final_gates_count_by_qudit_number = {g.num_qudits:v for g,v in out_circuit.gate_counts.items()}
compiler.close()
