from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, VariableUnitaryGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import  HilbertSchmidtCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.passes import ScanningGateRemovalPass, IfThenElsePass, PassPredicate


from util import SecondLEAPSynthesisPass, GenerateProbabilityPass, SelectFinalEnsemblePass, LEAPSynthesisPass2, SubselectEnsemblePass
from util import WriteQasmPass, ReplaceWithQasmPass, CheckEnsembleQualityPass, HamiltonianNoisePass, EnsembleLeap, EnsembleZXZXZ

from bqskit import enable_logging

from util import save_circuits, load_circuit, FixGlobalPhasePass, CalculateErrorBoundPass


class BadEnsemblePredicate(PassPredicate):

    def __init__(self, remove_ensemble: bool = True) -> None:
        self.remove_ensemble = remove_ensemble

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        has_good_ensemble = data.get("good_ensemble", False)
        if not has_good_ensemble:
            print("NO GOOD ENSEMBLES, TRYING AGAIN!", flush=True)
            data.pop("good_ensemble", None)
            block_data = data[ForEachBlockPass.key]
            previous_data_key = ForEachBlockPass.key + "_previous"
            if previous_data_key in data:
                data[previous_data_key].append(block_data[0])
            else:
                data[previous_data_key] = [block_data[0]]
            data[ForEachBlockPass.key] = []
            if self.remove_ensemble:
                data.pop("ensemble", None)
                data.pop("scan_sols", None)
        else:
            print("GOOD ENSEMBLES FOUND!", flush=True)
        return not has_good_ensemble

def get_distance(circ1: Circuit) -> float:
    global target
    return circ1.get_unitary().get_frobenius_distance(target)


def get_shortest_circuits(circ_name: str, timestep: int) -> list[Circuit]:
    circ = load_circuit(circ_name, opt=False)

    print("Original CNOT Count: ", circ.count(CNOTGate()))
    
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"checkpoint_qasms/{circ_name}_{big_block_size}_{small_block_size}/"

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        ScanPartitioner(block_size=big_block_size),
        # ExtendBlockSizePass(),
    ]

    fast_partitioner_passes = [
        QuickPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        QuickPartitioner(block_size=big_block_size),
        # ExtendBlockSizePass(),
    ]

    if circ.num_qudits > 20:
        partitioner_passes = fast_partitioner_passes
    else:
        partitioner_passes = slow_partitioner_passes

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                WriteQasmPass(),
            ]
        )
    ]
    num_workers = 1
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    compiler.compile(circ, workflow=leap_workflow)
    return

if __name__ == '__main__':
    global target
    circ_name = argv[1]
    timestep = int(argv[2])
    get_shortest_circuits(circ_name, timestep)