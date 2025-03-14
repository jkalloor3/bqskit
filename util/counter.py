from cachetools import LRUCache
import numpy as np
from math import ceil
from bqskit.ir import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.gates import *
import numpy as np
from .fix_angles import FixAnglesPass
from .convert_to_cliff import ConvertToZXZXZSimple
from .gg import GridSynthGate, gg_gate_def
import os

qlang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

def fix_angle_workflow(circ: Circuit, precision: int, skip_first: bool = False) -> None:
    if not skip_first:
        FixAnglesPass.run_circ(circ, 15)
    ConvertToZXZXZSimple.run_circuit(circ)
    FixAnglesPass.run_circ(circ, precision)



t_cache = {}


class GateCounter:
    def __init__(self, est: bool = True):
        self.est = est

    @staticmethod
    def has_non_rz(circ: Circuit) -> bool:
        rz_params = circ.count(RZGate()) + circ.count(GridSynthGate()) * 3
        return rz_params != circ.num_params

    def count_qasm_file(self, circ_file: str, target_error: float = None, 
                        count_t: bool = False, count_rz: bool = False) -> int:
        
        if not os.path.exists(circ_file):
            return (10 ** 8)
        qasm_str = open(circ_file).read()
        return self.count_qasm(qasm_str, target_error, count_t, count_rz)
    
    def count_qasm(self, qasm: str, target_error: float = None, 
                    count_t: bool = False, count_rz: bool = False) -> int:
        if not count_t and not count_rz:
            # Count CX
            return qasm.count('cx ')
        # Load the circuit
        assert not (count_t and count_rz), "Cannot count T and RZ gates"

        if self.est and count_rz:
            # Count RZ gates
            # Just count the number of RZ gates from str
            gg_count = qasm.count('gg(') + qasm.count('gg (')
            rz_count = qasm.count('rz(') + qasm.count('rz (')
            u3_count = qasm.count('u3(') + qasm.count('u3 (')
            # print(gg_count, rz_count, u3_count)
                # Some U3s are just identity
            id_string = "(0.0, 0.0, 0.0)"
            id_count = qasm.count(id_string)
            u3_count -= id_count
            num_rotations = rz_count + gg_count + u3_count * 3
            if count_rz:
                return num_rotations
        circ = qlang.decode(qasm)
        if count_rz:
            return self.count_rz(circ, target_error)
        elif count_t:
            return self.count_t(circ, target_error)
            
    def count_cx(self, circ: Circuit)  -> int:
        return circ.count(CNOTGate())

    def count_rz(self, circ: Circuit, target_error: float = None, 
                    skip_fix: bool = False)  -> int:
        if target_error is None:
            precision = 18
        else:
            error_per_param = target_error / circ.num_params
            precision = ceil(-np.log10(error_per_param))
        
        # Count RZ gates
        if self.has_non_rz(circ) and not skip_fix:
            out_circ = circ.copy()
            fix_angle_workflow(out_circ, precision=precision)
        else:
            out_circ = circ

        # U3 params
        u3_params = 0
        for op in out_circ.operations():
            if isinstance(op.gate, U3Gate):
                u3_params += len(np.nonzero(op.params))

        final_count = out_circ.count(RZGate()) + out_circ.count(GridSynthGate()) + u3_params
        return final_count

    def count_t(self, circ: Circuit, target_error: float = None, skip_fix: bool = False) -> int:
        if target_error is None:
            precision = 18
        else:
            num_params = circ.count(U3Gate()) * 3 + circ.count(RZGate())
            if num_params == 0:
                return circ.count(TGate()) + circ.count(TdgGate())
            error_per_param = target_error / num_params
            precision = ceil(-np.log10(error_per_param))
        
        if GateCounter.has_non_rz(circ) and not skip_fix:
            out_circ = circ.copy()
            fix_angle_workflow(out_circ, precision=precision)
        else:
            out_circ = circ

        # print("OutCirc Gate Counts: ", out_circ.gate_counts, flush=True)

        # Count the number of T gates
        num_t = out_circ.count(TGate()) + out_circ.count(TdgGate())
        for op in out_circ.operations():
            if isinstance(op.gate, RZGate):
                # On average, num_ts is about 10 * precisions
                num_t += 10 * precision
            elif isinstance(op.gate, GridSynthGate):
                gg_prec = op.params[1]
                num_t += 10 * gg_prec
            elif isinstance(op.gate, U3Gate):
                # Assume 3 RZ gates per U3 -> This is explicitly converted in
                # the fix_angle_workflow
                num_t += 30 * precision
        return num_t

gate_counter_est = GateCounter(est=True)
gate_counter_full = GateCounter(est=False)

def get_circ_counts(circ_files: list[str], 
                         count_t: bool = False,
                         count_rz: bool = False,
                         target_error: float = 1e-8,
                         ) -> list[int]:
            
    return [gate_counter_full.count_qasm_file(circ_file, target_error, 
            count_t, count_rz) for circ_file in circ_files]


def load_ensemble_counts_est(ensemble_file: str, jiggle_file: str, target_error: float, 
                                    count_t: bool = False, count_rz: bool = False) -> float:

    with open(ensemble_file, "r") as f:
        qasms = f.read().split("\nBREAK\n")

    if jiggle_file is None:
        counts = [gate_counter_est.count_qasm(q, target_error, count_rz=count_rz, count_t=count_t) for q in qasms]
    
    else:
        params = np.load(jiggle_file)
        # Sample 10 of the qasms
        rand_inds = np.random.randint(0, len(qasms), size=min(10, len(qasms)))
        qasms = [qasms[i] for i in rand_inds]

        # For each qasm, sample 3 random params
        all_qasms = []
        for i, qasm in enumerate(qasms):
            rand_params = np.random.randint(0, len(params[i]), size=3)
            for j in rand_params:
                new_circ = qlang.decode(qasm)
                new_circ.set_params(params[i][j])
                all_qasms.append(new_circ)

        # Ensemble should already be fixed
        counts = [gate_counter_est.count_t(circ, target_error, skip_fix=True) for circ in all_qasms]
    return counts

def load_avg_ensemble_counts_est(ensemble_file: str, jiggle_file: str, target_error: float, 
                                    count_t: bool = False, count_rz: bool = False) -> float:

    counts = load_ensemble_counts_est(ensemble_file, jiggle_file, target_error, count_t, count_rz)
    return np.mean(counts)
    
def load_ensemble_counts_full(ensemble_file: str, jiggle_file: str, target_error: float, 
                                  count_t: bool = False, count_rz: bool = False) -> float:
    with open(ensemble_file, "r") as f:
        qasms = f.read().split("\nBREAK\n")

    if jiggle_file is None:
        counts = [gate_counter_full.count_qasm(q, target_error, count_rz=count_rz, count_t=count_t) for q in qasms]
    else:
        params = np.load(jiggle_file)

        # Sample 50 of the qasms
        rand_inds = np.random.randint(0, len(qasms), size=min(50, len(qasms)))
        qasms = [qasms[i] for i in rand_inds]

        # For each qasm, sample 4 random params
        all_qasms = []
        for i, qasm in enumerate(qasms):
            rand_params = np.random.randint(0, len(params[i]), size=4)
            for j in rand_params:
                new_circ = qlang.decode(qasm)
                new_circ.set_params(params[i][j])
                all_qasms.append(new_circ)

        # Ensemble should already be fixed
        counts = [gate_counter_full.count_t(circ, target_error, skip_fix=True) for circ in all_qasms]
    return np.mean(counts)

def load_avg_ensemble_counts_full(ensemble_file: str, jiggle_file: str, target_error: float, 
                                    count_t: bool = False, count_rz: bool = False) -> float:

    counts = load_ensemble_counts_full(ensemble_file, jiggle_file, target_error, count_t, count_rz)
    return np.mean(counts)


def count_params(circ: Circuit) -> int:
    return gate_counter_full.count_rz(circ, skip_fix=True)

def count_params_str(qasm_str: str) -> int:
    return gate_counter_est.count_qasm(qasm_str, target_error=1e-12, 
                                       count_rz=True)
def count_curr_t(circ: Circuit) -> int:
    return gate_counter_full.count_t(circ, skip_fix=True)

def count_all_t(circ: Circuit) -> int:
    return gate_counter_full.count_t(circ, skip_fix=False)

def count_all_rz(circ: Circuit) -> int:
    return gate_counter_full.count_rz(circ, skip_fix=False)