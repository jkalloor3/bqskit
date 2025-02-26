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

qlang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

def fix_angle_workflow(circ: Circuit, precision: int) -> None:
    FixAnglesPass.run_circ(circ, 15)
    ConvertToZXZXZSimple.run_circ(circ)
    FixAnglesPass.run_circ(circ, precision)


class GateCounter:
    def __init__(self, est: bool = True):
        self.est = est
        self.cache = LRUCache(maxsize=10000)

    @staticmethod
    def has_non_rz(circ: Circuit) -> bool:
        rz_params = circ.count(RZGate()) + circ.count(GridSynthGate()) * 3
        return rz_params != circ.num_params

    def count_qasm_file(self, circ_file: str, target_error: float = None, 
                        count_t: bool = False, count_rz: bool = False) -> int:
        qasm_str = open(circ_file).read()
        return self.count_qasm(qasm_str, target_error, count_t, count_rz)
    
    def count_qasm(self, qasm: str, target_error: float = None, 
                    count_t: bool = False, count_rz: bool = False) -> int:
        if not count_t and not count_rz:
            # Count CX
            return qasm.count('cx ')
        # Load the circuit
        assert not (count_t and count_rz), "Cannot count T and RZ gates"

        if self.est:
            # Count RZ gates
            # Just count the number of RZ gates from str
            gg_count = qasm.count('gg ')
            rz_count = qasm.count('rz ')
            u3_count = qasm.count('u3 ')
                # Some U3s are just identity
            id_string = "(0.0, 0.0, 0.0)"
            id_count = qasm.count(id_string)
            u3_count -= id_count
            num_rotations = rz_count + gg_count + u3_count * 3
            if count_rz:
                return num_rotations
            if count_t:
                # Assume that GG's are half precision and RZs are full precision
                error_per_param = target_error / num_rotations
                precision = ceil(-np.log10(error_per_param))
                gg_precision = ceil(-np.log10(error_per_param) / 2)
                
                ts_per_rz = 10 * precision
                ts_per_gg = 10 * gg_precision
                ts_per_u3 = 30 * precision

                # Count the number of T gates
                return (rz_count * ts_per_rz + gg_count * ts_per_gg + 
                        u3_count * ts_per_u3)
        else:
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

        return circ.count(RZGate()) + circ.count(GridSynthGate()) + u3_params

    def count_t(self, circ: Circuit, target_error: float = None, skip_fix: bool = False) -> int:
        if target_error is None:
            precision = 18
        else:
            error_per_param = target_error / circ.num_params
            precision = ceil(-np.log10(error_per_param))
        
        if GateCounter.has_non_rz(circ) and not skip_fix:
            out_circ = circ.copy()
            workflow = fix_angle_workflow(out_circ, precision=precision)
        else:
            out_circ = circ

        print(out_circ.gate_counts)

        # Count the number of T gates
        num_t = out_circ.count(TGate()) + out_circ.count(TdgGate())
        for op in out_circ.operations():
            if isinstance(op.gate, RZGate):
                angle = op.params[0]
                ind = (angle, precision)
                num_ts = self.cache.get(ind, None)
                if num_ts is None:
                    # Count the number of T gates
                    gg_params = [angle, precision, 0]
                    gg_circ = GridSynthGate().get_circuit(gg_params)
                    gg_ts = gg_circ.count(TGate()) + gg_circ.count(TdgGate())

                    self.cache[ind] = gg_ts
                    num_ts = gg_ts
                num_t += num_ts
            elif isinstance(op.gate, GridSynthGate):
                ind = (op.params[0], op.params[1])
                num_ts = self.cache.get(ind, None)
                if num_ts is None:
                    # Count the number of T gates
                    gg_params = [op.params[0], op.params[1], 0]
                    gg_circ = GridSynthGate().get_circuit(gg_params)
                    gg_ts = gg_circ.count(TGate()) + gg_circ.count(TdgGate())

                    self.cache[ind] = gg_ts
                    num_ts = gg_ts
                num_t += num_ts
        
        return num_t

gate_counter_est = GateCounter(est=False)
gate_counter_full = GateCounter(est=True)

def get_circ_counts(circ_files: list[str], 
                         count_t: bool = False,
                         count_rz: bool = False,
                         target_error: float = 1e-8,
                         ) -> list[int]:
            
    return [gate_counter_full.count_qasm_file(circ_file, target_error, 
            count_t, count_rz) for circ_file in circ_files]


def load_ensemble_counts_est(ensemble_file: str, target_error: float, 
                                    count_t: bool = False, count_rz: bool = False) -> float:

    with open(ensemble_file, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    
    counts = [gate_counter_est.count_qasm(qasm, target_error, count_t=count_t, count_rz=count_rz) for qasm in qasms]
    return counts

def load_avg_ensemble_counts_est(ensemble_file: str, target_error: float, 
                                    count_t: bool = False, count_rz: bool = False) -> float:

    counts = load_ensemble_counts_est(ensemble_file, target_error, count_t, count_rz)
    return np.mean(counts)
    
    
def load_ensemble_counts_full(ensemble_file: str, target_error: float, 
                                  count_t: bool = False, count_rz: bool = False) -> float:
    with open(ensemble_file, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    
    counts = [gate_counter_full.count_qasm(qasm, target_error, 
                count_t=count_t, count_rz=count_rz) for qasm in qasms]
    return counts

def load_avg_ensemble_counts_full(ensemble_file: str, target_error: float, 
                                  count_t: bool = False, count_rz: bool = False) -> float:
    
    counts = load_ensemble_counts_full(ensemble_file, target_error, count_t, count_rz)
    return np.mean(counts)

def count_params(circ: Circuit) -> int:
    gate_counter_full.count_rz(circ, skip_fix=True)

def count_curr_t(circ: Circuit) -> int:
    gate_counter_full.count_t(circ, skip_fix=True)

def count_all_t(circ: Circuit) -> int:
    gate_counter_full.count_t(circ, skip_fix=False)

def count_all_rz(circ: Circuit) -> int:
    gate_counter_full.count_rz(circ, skip_fix=False)