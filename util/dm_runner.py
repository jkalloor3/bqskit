import numpy as np
import pickle

from bqskit.ir.circuit import Circuit, CircuitPoint, CircuitLocationLike
from .samplers import OrderedEnsembleSampler, get_file_names, get_single_rho
from .common import get_block_names, load_jiggled_ensemble


def create_large_block_runner(qubits: np.ndarray[int], circ_name: str,
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
                actual_qubits = qubits[op.location]
                pt = CircuitPoint(cycle, op.location[0])
                # Get file names
                file_names = get_file_names(large_checkpoint_dir=large_checkpoint,
                                            small_block_num=block_num)[:-1]
                small_block_runners[pt] = DensityMatrixRunner(qubits=actual_qubits,
                                                              ensemble_file_names=file_names,
                                                              cliff_t=cliff_t)
    
        # Create the DensityMatrixRunner for this large block
        runner = DensityMatrixRunner(qubits=qubits,
                                     partitioned_circ=large_block_circ,
                                     block_runners=small_block_runners,
                                     cliff_t=cliff_t)
        
        print(f"Created runner for large block {large_block_num} on qubits {qubits}", flush=True)
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
                        qubits=np.array(op.location),
                        circ_name=circ_name,
                        large_block_num=block_num,
                        max_tol=max_tol,
                        partitioned_data=partitioned_data[block_num],
                        checkpoint_form=checkpoint_form,
                        cliff_t=cliff_t
                )
            block_runners[pt] = runner

    return DensityMatrixRunner(qubits=np.arange(full_circ.num_qudits),
                               partitioned_circ=full_circ,
                               block_runners=block_runners,
                               cliff_t=cliff_t)


class DensityMatrixRunner:
    """A class for running density matrix evaluations on ensembles of
    circuits."""

    def __init__(self, 
                 qubits: np.ndarray[int],
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
            assert qubits is not None
            self.sampler = OrderedEnsembleSampler(load_jiggled_ensemble(*ensemble_file_names), 
                                                  cliff_t=cliff_t)
        else:
            assert partitioned_circ is not None
            assert block_runners is not None
            self.sampler = None
        
        self.qubits = qubits
        self.partitioned_circ = partitioned_circ
        self.block_runners = block_runners
        self.ensemble_file_names = ensemble_file_names


    async def run(self, init_rho: np.ndarray) -> np.ndarray:
        """Run the density matrix evaluation."""
        # print("Running on qubits: ", self.qubits, flush=True)
        if self.sampler is not None:
            return await self.sampler.output_rho(init_rho, qubits=self.qubits)
        else:
            rho = init_rho.copy()
            for cycle, op in self.partitioned_circ.operations_with_cycles():
                pt = CircuitPoint(cycle, op.location[0])

                if pt in self.block_runners:
                    rho = await self.block_runners[pt].run(rho)
                else:
                    # Apply U rho U^\dagger to the correct qubits
                    # Index qubits by location
                    actual_qubits = self.qubits[op.location]
                    rho = get_single_rho(op.get_unitary(), rho, actual_qubits)
            return rho
                    
