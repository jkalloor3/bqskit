
import numpy as np
from bqskit.ir.circuit import Circuit, CircuitPoint 
from bqskit.qis import UnitaryMatrix
from .common import create_jiggled_unitaries
from .gg import gridsynth_gates_to_cir, GridSynthGate

class EnsembleSampler:
    def __init__(self, all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 cliff_t: bool = False):
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.circ_probs = np.array([np.sum(p) for p in self.all_probs])
        self.circ_probs /= np.sum(self.circ_probs)
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t

    def __call__(self) -> np.ndarray[np.complex128] | list[tuple[UnitaryMatrix, float]]:
        return create_jiggled_unitaries(self.circ_params, self.target, self.add_cost)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Create a random circuit based on the probabilities
        rand_circ_idx = np.random.choice(len(self.circ_probs), p=self.circ_probs)
        circ = self.circs[rand_circ_idx]
        param_options = self.params[rand_circ_idx]
        param_probs = self.all_probs[rand_circ_idx]
        rand_param_idx = np.random.choice(len(param_options), p=param_probs)
        rand_param = param_options[rand_param_idx]
        cache = self.caches[rand_circ_idx]
        out_circ = circ.copy()
        out_circ.set_params(rand_param)
        if self.cliff_t:
            # Lower all GridSynthGates to corresponding Cliff T circs
            for cycle, op in out_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    pt = CircuitPoint(cycle, op.location)
                    cache_ind = (op.params[0], int(op.params[1]))
                    t_str = cache[cache_ind]
                    if op.params[2] == 1:
                        t_str = "Z" + t_str + "Z"
                    clifft_circ = gridsynth_gates_to_cir(t_str)
                    out_circ.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)
            out_circ.unfold_all()
        return out_circ
