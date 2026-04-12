import numpy as np
import pickle
import time
import os
from pathlib import Path

from bqskit.ir.circuit import Circuit, CircuitPoint, CircuitGate
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import (TGate, TdgGate, ConstantGate, IdentityGate,
                             XGate, YGate, ZGate)
from bqskit.runtime import get_runtime
from .samplers import get_single_rho, apply_superoperator, get_superop
from .common import get_block_names, load_jiggled_ensemble
from .experiment_util import get_file_names
from .distance import frobenius_cost
from .gg import GridSynthGate, get_t_gates_from_string

def create_large_block_runner(circ_name: str,
                              large_block_num: str,
                              max_tol: float,
                              partitioned_data: tuple[dict[str, Circuit], Circuit],
                              checkpoint_form: str = "",
                              cliff_t: bool = False) -> "DensityMatrixRunner":
        """Create a DensityMatrixRunner for a large block."""
    
    
        _, large_block_circ = partitioned_data

        small_block_runners = generate_small_block_runners(
            circ_name=circ_name,
            large_block_num=large_block_num,
            max_tol=max_tol,
            partitioned_data=partitioned_data,
            checkpoint_form=checkpoint_form,
            cliff_t=cliff_t
        )
    
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


def generate_all_block_runners(circ_name: str,
                               max_tol: float,
                               partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                               checkpoint_form: str = "",
                               cliff_t: bool = False) -> dict[tuple[str, str], 
                                                               "DensityMatrixRunner"]:
    """Generate DensityMatrixRunners for all large blocks."""
    large_block_nums = get_block_names(circ_name=circ_name, extra="_tket")
    block_runners = {}
    for large_block_num in large_block_nums:
        small_block_runners = generate_small_block_runners(
            circ_name=circ_name,
            large_block_num=large_block_num,
            max_tol=max_tol,
            partitioned_data=partitioned_data[large_block_num],
            checkpoint_form=checkpoint_form,
            cliff_t=cliff_t,
            use_pt=False
        )
        for small_block_num, runner in small_block_runners.items():
            block_runners[(large_block_num, small_block_num)] = runner
    return block_runners


def generate_small_block_runners(circ_name: str,
                                 large_block_num: str,
                                 max_tol: float,
                                 partitioned_data: tuple[dict[str, Circuit], Circuit],
                                 checkpoint_form: str = "",
                                 cliff_t: bool = False,
                                 use_pt: bool = True) -> dict[str, 
                                                                 "DensityMatrixRunner"]:  
    
    """Generate DensityMatrixRunners for all small blocks in a large block."""

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
            if use_pt:
                key = pt
            else:
                key = block_num
            # Get file names
            file_names = get_file_names(large_checkpoint_dir=large_checkpoint,
                                        small_block_num=block_num)[:-1]
            small_block_runners[key] = DensityMatrixRunner(
                ensemble_file_names=file_names,
                target=op.get_unitary(),
                cliff_t=cliff_t,
                label = f"Large {large_block_num} Small {block_num}"
            )

    return small_block_runners

def apply_depolarizing_channel(rho: np.ndarray,
                               qubits: list[int],
                               noise_level: float) -> np.ndarray:
    ''' Apply a depolarizing channel to the given qubits in the density matrix.'''
    d = 2 ** len(qubits)
    probs = [1 - noise_level, noise_level / 3, noise_level / 3, noise_level / 3]
    uns = [IdentityGate().get_unitary(), 
           XGate().get_unitary(), 
           YGate().get_unitary(), 
           ZGate().get_unitary()]
    
    rho_out = np.zeros_like(rho, dtype=np.complex128)
    for p, un in zip(probs, uns):
        single_rho = get_single_rho(un, rho, qubits)
        rho_out += p * single_rho
    return rho_out

def get_noisy_rho(circ: Circuit,
                  params: np.ndarray,
                  cache: dict,
                  rho_in: np.ndarray,
                  noise_level: float = 0) -> np.ndarray:
    ''' Get the output density matrix after applying noisy circuit.'''
    # un = circ.get_unitary(params=params, cache=cache)
    circ_copy = circ.copy()
    circ_copy.set_params(params)

    # Calculate the output density matrix from each operation
    for op in circ_copy.operations():
        qubits = op.location
        if isinstance(op.gate, GridSynthGate):
            t_str = cache[(op.params[0], op.params[1])]
            if op.params[2] == 1:
                t_str = "Z" + t_str + "Z"
            t_gates: list[ConstantGate] = get_t_gates_from_string(t_str)
            for tg in t_gates:
                U = tg.get_unitary()
                rho_in = get_single_rho(U, rho_in, qubits)
                if isinstance(tg, TGate) or isinstance(tg, TdgGate):
                    # Apply noise channel
                    rho_in = apply_depolarizing_channel(rho_in, 
                                                        qubits, noise_level)
        else:      
            # Get the unitary matrix for this operation
            U = op.get_unitary()
            # Apply U rho U^\dagger
            rho_in = get_single_rho(U, rho_in, qubits)

    return rho_in

class EnsembleDMRunner:
    def __init__(self, 
                 all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 target: UnitaryMatrix,
                 target_circ: Circuit = None,
                 cliff_t: bool = False) -> None:
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t
        self.target = target
        self.target_circ = target_circ
        self.superoperator = None

    async def initialize(self, file_name: str = "") -> None:
        ''' Generate a superoperator for the entire ensemble. '''

        if os.path.exists(file_name) and file_name != "":
            try:
                self.superoperator = np.load(file_name)
            except:
                # If there is an error loading, then it is because superoperator
                # is None
                self.superoperator = None
            return

        n = self.circs[0].num_qudits

        super_op = np.zeros((2**(2*n), 2**(2*n)), dtype=np.complex128)
        un = np.zeros((2**n, 2**n), dtype=np.complex128)

        for i, circ in enumerate(self.circs):
            superop_uns = await get_runtime().map(get_superop, 
                              self.params[i],
                              circ=circ,
                              cache=self.caches[i],
                              cliff_t=self.cliff_t,
                              target=self.target)
            superops = [so for so, _ in superop_uns]
            uns = [u for _, u in superop_uns]
            probs = self.all_probs[i]
            circ_superop = np.tensordot(probs, list(superops), axes=([0], [0]))
            circ_un = np.tensordot(probs, list(uns), axes=([0], [0]))
            un += circ_un
            super_op += circ_superop

        frob_dist = frobenius_cost(un, self.target)
        if frob_dist < 1e-1: # Extra check on bias -> some files may be corrupted
            # We can use this block, otherwise there is some weird error
            self.superoperator = super_op
        else:
            self.superoperator = None

    def apply_noisy_circuits(self, rho_in: np.ndarray, noise_level: float) -> np.ndarray:
        ''' Apply all noisy circuits in the ensemble to the input density matrix.'''
        avg_rho = np.zeros_like(rho_in, dtype=np.complex128)
        for i, circ in enumerate(self.circs):
            cache = self.caches[i]
            for j, param in enumerate(self.params[i]):
                prob = self.all_probs[i][j]

                rho_out = get_noisy_rho(circ, param, cache, rho_in, noise_level)

                avg_rho += prob * rho_out
        return avg_rho
        
    def run_noisy(self, rho_in: np.ndarray, noise_level: float) -> np.ndarray:
        ''' Run the entire ensemble superoperator on the input density matrix with noise.'''
        # rho_in should be a 2^n x 2^n matrix
        if self.superoperator is None:
            # Just apply target unitary onto rho_in
            return None
        else:
            noisy_rho = self.apply_noisy_circuits(rho_in, noise_level)
            return noisy_rho

class DensityMatrixRunner:
    """A class for running density matrix evaluations on ensembles of
    circuits."""

    def __init__(self, 
                 partitioned_circ: Circuit = None,
                 block_runners: dict[CircuitPoint, "DensityMatrixRunner"] = None,
                 ensemble_file_names: list[str] | None = None,
                 target: UnitaryMatrix = None,
                 cliff_t: bool = True,
                 label: str = ""
                ) -> None:
        """Initialize the DensityMatrixRunner."""
        # If we are given ensemble file names, then we should not be given
        # partitioned circuits or block runners.
        self.correct = True
        if ensemble_file_names is not None:
            assert partitioned_circ is None
            assert block_runners is None
            assert target is not None
            self.runner = EnsembleDMRunner(
                load_jiggled_ensemble(*ensemble_file_names),
                target=target, 
                cliff_t=cliff_t
            )
            self.correct = (self.runner.circs[0].num_qudits == target.num_qudits)
        else:
            assert partitioned_circ is not None
            assert block_runners is not None
            self.runner = None
        
        self.partitioned_circ = partitioned_circ
        self.block_runners = block_runners
        self.ensemble_file_names = ensemble_file_names
        self.superoperators = {}
        self.label = label

    async def initialize(self, file_name: str = "") -> None:
        """Initialize all the samplers with their superoperators."""
        start = time.time()
        futs = []
        if self.runner is not None:
            futs.append(self.runner.initialize(file_name + ".npy"))
        else:
            for cycle, op in self.partitioned_circ.operations_with_cycles():
                pt = CircuitPoint(cycle, op.location[0])
                if pt in self.block_runners:
                    futs.append(self.block_runners[pt].initialize(file_name + f"_{pt.cycle}_{pt.qudit}"))

        # Await all initializations
        for fut in futs:
            await fut
        end = time.time()
        if self.runner is not None:
            print(f"Initialized ensemble runner in {end-start} seconds", 
                  flush=True)
        else:
            print(f"Initialized block runners in {end-start} seconds", 
                  flush=True)

    
    def save(self, file_name: str) -> None:
        ''' Save the DensityMatrixRunner to a file. '''

        if self.runner is not None:
            # Save the runner superoperator
            Path(file_name + ".npy").parent.mkdir(parents=True, exist_ok=True)
            np.save(file_name + ".npy", self.runner.superoperator)
        else:
            # Save the block runners recursively
            for pt, runner in self.block_runners.items():
                runner.save(file_name + f"_{pt.cycle}_{pt.qudit}")


    def run(self, init_rho: np.ndarray, qubits: np.ndarray[int]) -> np.ndarray:
        """Apply the channel onto the density matrix at the qubits specified"""
        # print("Running on qubits: ", self.qubits, flush=True)
        if self.runner is not None:
            print(self.label, flush=True)
            return self.runner.run(init_rho, qubits)
        else:
            rho = init_rho.copy()
            for cycle, op in self.partitioned_circ.operations_with_cycles():
                pt = CircuitPoint(cycle, op.location[0])
                assert isinstance(op.gate, CircuitGate)

                if pt in self.block_runners:
                    rho = self.block_runners[pt].run(rho, qubits[op.location])
                else:
                    # Apply U rho U^\dagger to the correct qubits
                    # Index qubits by location
                    actual_qubits = qubits[op.location]
                    rho = get_single_rho(op.get_unitary(), rho, actual_qubits)
            return rho
        
    def run_noisy(self, init_rho: np.ndarray, noise_level: float) -> np.ndarray:
        """Apply the noisy channel onto the density matrix at the qubits specified"""
        assert self.runner is not None, "No ensemble runner available for noisy run."
        return self.runner.run_noisy(init_rho, noise_level)
