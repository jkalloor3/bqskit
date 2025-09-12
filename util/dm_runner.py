import numpy as np
import pickle

from bqskit.ir.circuit import Circuit, CircuitPoint, CircuitLocationLike
from bqskit.runtime import get_runtime
from .samplers import get_file_names, get_single_rho, apply_superoperator, get_superop
from .common import get_block_names, load_jiggled_ensemble
from .distance import trace_distance

def create_large_block_runner(circ_name: str,
                              large_block_num: str,
                              max_tol: float,
                              partitioned_data: tuple[dict[str, Circuit], Circuit],
                              checkpoint_form: str = "",
                              cliff_t: bool = False) -> "DensityMatrixRunner":
        """Create a DensityMatrixRunner for a large block."""
    
        large_checkpoint = checkpoint_form.format(circ_name=circ_name,
                                                  large_block_num=large_block_num,
                                                  max_tol=max_tol)
    
        sub_circs, large_block_circ = partitioned_data

        small_block_runners = {}
        num_digits = len(str(large_block_circ.num_operations))
        for i, (cycle, op) in enumerate(large_block_circ.operations_with_cycles()):
            block_num = str(i).zfill(num_digits)
            if block_num in sub_circs:
                pt = CircuitPoint(cycle, op.location[0])
                # Get file names
                file_names = get_file_names(large_checkpoint_dir=large_checkpoint,
                                            small_block_num=block_num)[:-1]
                small_block_runners[pt] = DensityMatrixRunner(ensemble_file_names=file_names,
                                                              cliff_t=cliff_t)
    
        # Create the DensityMatrixRunner for this large block
        runner = DensityMatrixRunner(partitioned_circ=large_block_circ,
                                     block_runners=small_block_runners,
                                     cliff_t=cliff_t)
        
        print(f"Created runner for large block {large_block_num}", flush=True)
        return runner


def generate_full_runner(circ_name: str, 
                         max_tol: float, 
                         partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                         checkpoint_form: str = "",
                         partitioned_circ_file: str = "",
                         cliff_t: bool = False):
    
    """Generate a DensityMatrixRunner for the full circuit."""

    # First create all of the large block runners
    large_block_nums = get_block_names(circ_name=circ_name, extra="_tket")

    # Now get the corresponding CircuitPoints
    full_circ: Circuit = pickle.load(open(partitioned_circ_file, "rb"))
    num_digits = len(str(full_circ.num_operations))
    block_runners = {}

    for i, (cycle, op) in enumerate(full_circ.operations_with_cycles()):
        block_num = str(i).zfill(num_digits)
        if block_num in large_block_nums:
            pt = CircuitPoint(cycle, op.location[0])
            runner = create_large_block_runner(
                    circ_name=circ_name,
                    large_block_num=block_num,
                    max_tol=max_tol,
                    partitioned_data=partitioned_data[block_num],
                    checkpoint_form=checkpoint_form,
                    cliff_t=cliff_t
                )
            block_runners[pt] = runner

    return DensityMatrixRunner(partitioned_circ=full_circ,
                               block_runners=block_runners,
                               cliff_t=cliff_t)


class EnsembleDMRunner:
    def __init__(self, all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 cliff_t: bool = False):
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t
        self.superoperator = None
    
    async def initialize(self) -> None:
        ''' Generate a superoperator for the entire ensemble. '''

        n = self.circs[0].num_qudits

        super_op = np.zeros((2**(2*n), 2**(2*n)), dtype=np.complex128)

        for i, circ in enumerate(self.circs):
            superops = await get_runtime().map(get_superop, 
                              self.params[i],
                              circ=circ,
                              cache=self.caches[i],
                              cliff_t=self.cliff_t)
            probs = self.all_probs[i]
            circ_superop = np.tensordot(probs, list(superops), axes=([0], [0]))
            super_op += circ_superop
        self.superoperator = super_op


    def run(self, rho_in: np.ndarray, qubits: int) -> np.ndarray:
        ''' Run the entire ensemble superoperator on the input density matrix.'''
        # rho_in should be a 2^n x 2^n matrix
        num_qubits = np.log2(rho_in.shape[0]).astype(int)
        return apply_superoperator(rho_in, self.superoperator, 
                                   num_qubits, qubits)

class DensityMatrixRunner:
    """A class for running density matrix evaluations on ensembles of
    circuits."""

    def __init__(self, 
                 partitioned_circ: Circuit = None,
                 block_runners: dict[CircuitPoint, "DensityMatrixRunner"] = None,
                 ensemble_file_names: list[str] | None = None,
                 cliff_t: bool = True
                 ) -> None:
        """Initialize the DensityMatrixRunner."""
        # If we are given ensemble file names, then we should not be given
        # partitioned circuits or block runners.
        if ensemble_file_names is not None:
            assert partitioned_circ is None
            assert block_runners is None
            self.sampler = EnsembleDMRunner(load_jiggled_ensemble(*ensemble_file_names), 
                                                  cliff_t=cliff_t)
        else:
            assert partitioned_circ is not None
            assert block_runners is not None
            self.sampler = None
        
        self.partitioned_circ = partitioned_circ
        self.block_runners = block_runners
        self.ensemble_file_names = ensemble_file_names
        self.superoperators = {}

    async def initialize(self) -> None:
        """Initialize all the samplers with their superoperators."""
        
        futs = []
        if self.sampler is not None:
            futs.append(self.sampler.initialize())
        else:
            for cycle, op in self.partitioned_circ.operations_with_cycles():
                pt = CircuitPoint(cycle, op.location[0])
                if pt in self.block_runners:
                    futs.append(self.block_runners[pt].initialize())
        
        # Await all initializations
        for fut in futs:
            await fut

    def run(self, init_rho: np.ndarray, qubits: np.ndarray[int]) -> np.ndarray:
        """Apply the channel onto the density matrix at the qubits specified"""
        # print("Running on qubits: ", self.qubits, flush=True)
        if self.sampler is not None:
            return self.sampler.run(init_rho, qubits)
        else:
            rho = init_rho.copy()
            for cycle, op in self.partitioned_circ.operations_with_cycles():
                pt = CircuitPoint(cycle, op.location[0])

                if pt in self.block_runners:
                    rho_2 = get_single_rho(op.get_unitary(), rho, qubits[op.location])
                    rho = self.block_runners[pt].run(rho, qubits[op.location])
                    print(trace_distance(rho, rho_2), flush=True)
                else:
                    # Apply U rho U^\dagger to the correct qubits
                    # Index qubits by location
                    actual_qubits = qubits[op.location]
                    rho = get_single_rho(op.get_unitary(), rho, actual_qubits)
            return rho
                    
