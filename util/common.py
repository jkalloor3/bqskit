from bqskit.ir.circuit import Circuit
from .fix_global_phase import fix_phase
from bqskit.qis import UnitaryMatrix
from pathlib import Path
from multiprocessing import shared_memory
from itertools import chain
import numpy as np
import os
import glob
import numpy as np
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from .distance import frobenius_cost, normalized_frob_cost
import multiprocessing as mp

from .gg import gg_gate_def, GridSynthGate

base_bqskit_dir = "/pscratch/sd/j/jkalloor/bqskit"
good_block_dir = f"{base_bqskit_dir}/good_blocks"
bad_block_dir = f"{base_bqskit_dir}/bad_blocks"
base_checkpoint_dir = f"{base_bqskit_dir}/block_checkpoints_final_paper"

qlang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

NUM_UNIQUE_CIRCS = 250

def stack_padding(it: list[np.ndarray], vertical: bool = True) -> np.ndarray:
    max_width = max(a.shape[1] for a in it)
    # Pad each 2D array with zeros to match the maximum width
    padded_arrays = [np.pad(a, ((0, 0), (0, max_width - a.shape[1])), mode='constant') for a in it]
    # Vertically stack the padded arrays
    if vertical:
        result = np.vstack(padded_arrays)
    else:
        result = np.stack(padded_arrays)
        print("Final Param Arr Shape: ", result.shape, flush=True)
    return result

def store_params(all_params: list[np.ndarray], jiggle_file_name: str):
    # Get max param size
    params = stack_padding(all_params, vertical=False)
    np.save(jiggle_file_name, params)

def store_jiggled_ensemble(ensemble: list[tuple[Circuit, np.ndarray]], file_name: str, jiggle_file_name: str):
    # Get circuits
    circs = [circ for circ, _ in ensemble ]
    store_ensemble(circs, file_name)
    store_params([params for _, params in ensemble], jiggle_file_name)

def create_single_jiggled_ensemble(circ_params: tuple[Circuit, np.ndarray], 
                                   target: UnitaryMatrix = None, phase_fix: bool = False,
                                   add_cost: bool = False) -> list[Circuit] | list[tuple[Circuit, UnitaryMatrix, float]]:
    circ, params = circ_params
    ens = []
    # print("Target: ", type(target), flush=True)
    # print("Params Shape: ", params.shape, "Circuit Params: ", circ.num_params, flush=True)
    for param in params.tolist():
        new_circ = circ.copy()
        new_circ.set_params(param)
        if phase_fix:
            fix_phase(new_circ, target)
        if add_cost:
            un = new_circ.get_unitary()
            cost_1 = normalized_frob_cost(un, target)
            ens.append((new_circ, un, cost_1))
        else:
            ens.append(new_circ)
    return ens

def create_avg_utry(circ_params: tuple[Circuit, np.ndarray], 
                    target: UnitaryMatrix,  
                    add_cost: bool = False) -> UnitaryMatrix | tuple[UnitaryMatrix, float]:
    circ, params = circ_params
    avg_utry = np.zeros_like(circ.get_unitary())
    avg_dist = 0
    for param in params.tolist():
        new_circ = circ.copy()
        new_circ.set_params(param)
        fix_phase(new_circ, target)
        un = new_circ.get_unitary()
        avg_utry += un
        if add_cost:
            avg_dist += normalized_frob_cost(un, target)
    avg_utry = avg_utry / len(params)
    avg_dist = avg_dist / len(params)
    if add_cost:
        return (avg_utry, avg_dist)
    else:
        return avg_utry


def creat_single_unitary(circ: Circuit, param: np.ndarray, target: UnitaryMatrix) -> tuple[UnitaryMatrix, float]:
    utry = circ.get_unitary(param)
    gp_correction = target.get_target_correction_factor(utry)
    utry = utry * gp_correction
    cost_1 = normalized_frob_cost(utry, target)
    return (utry, cost_1)

def get_unitary(circ: Circuit, target: UnitaryMatrix) -> tuple[UnitaryMatrix, float]:
    utry = circ.get_unitary()
    gp_correction = target.get_target_correction_factor(utry)
    utry = utry * gp_correction
    cost_1 = normalized_frob_cost(utry, target)
    return (utry, cost_1)


def create_jiggled_unitaries(circ_params: tuple[Circuit, np.ndarray], 
                                   target: UnitaryMatrix = None,
                                   add_cost: bool = True) -> np.ndarray[np.complex128] | list[tuple[UnitaryMatrix, float]]:
    circ, params = circ_params
    ens = []
    correct = target is not None
    # print("Params Shape: ", params.shape, "Circuit Params: ", circ.num_params, flush=True)
    for param in params.tolist():
        utry = circ.get_unitary(param)
        if correct:
            gp_correction = target.get_target_correction_factor(utry)
            utry = utry * gp_correction
        if add_cost:
            cost_1 = normalized_frob_cost(utry, target)
            ens.append((utry, cost_1))
        else:
            ens.append(utry)
    if not add_cost:
        ens = np.array(ens, dtype=np.complex128)
        return ens
    return ens

# def calc_avg_unitary_shm(circ_ind: tuple[Circuit, int],
#                                  shm_name: str, 
#                                  shm_shape: tuple[int, int, int], 
#                                  target: UnitaryMatrix = None) -> tuple[UnitaryMatrix, float]:
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     shared_array = np.ndarray(shm_shape, dtype=np.float64, buffer=existing_shm.buf)
#     circ, param_ind = circ_ind
#     params: np.ndarray = shared_array[param_ind]
#     orig_unitary = circ.get_unitary()
#     avg_unitary = np.zeros_like(orig_unitary)
#     avg_dist = 0.0
#     for i, param in enumerate(params.tolist()):
#         utry = circ.get_unitary(param)
#         gp_correction = target.get_target_correction_factor(utry)
#         utry: UnitaryMatrix = utry * gp_correction
#         cost_1 = normalized_frob_cost(utry, target)
#         avg_unitary += utry.numpy
#         avg_dist += cost_1

#     existing_shm.close()
#     avg_unitary = avg_unitary / len(params)
#     avg_dist = avg_dist / len(params)
#     return avg_unitary, avg_dist

# def create_jiggled_unitaries_shm(circ_ind: tuple[Circuit, int],
#                                  shm_name: str, 
#                                  shm_ret_name: str,
#                                  shm_shape: tuple[int, int, int], 
#                                  shm_ret_shape: tuple[int, int, int],
#                                  target: UnitaryMatrix) -> np.ndarray:
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     existing_shm_ret = shared_memory.SharedMemory(name=shm_ret_name)
#     shared_array = np.ndarray(shm_shape, dtype=np.float64, buffer=existing_shm.buf)
#     shared_array_ret = np.ndarray(shm_ret_shape, dtype=np.complex128, buffer=existing_shm_ret.buf)
#     circ, param_ind = circ_ind
#     params: np.ndarray = shared_array[param_ind]

#     for i, param in enumerate(params.tolist()):
#         utry = circ.get_unitary(param)
#         gp_correction = target.get_target_correction_factor(utry)
#         utry: UnitaryMatrix = utry * gp_correction
#         shared_array_ret[i + param_ind * 4] = utry.numpy
        

#     existing_shm.close()
#     existing_shm_ret.close()
#     return


def create_jiggled_ensemble(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    return list(chain.from_iterable(ensemble))

def create_jiggled_ensemble_mp(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    # ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    with mp.Pool(processes=5) as pool:
        ensemble = pool.map(create_single_jiggled_ensemble, circ_params)
    return list(chain.from_iterable(ensemble))

def load_jiggled_ensemble_separate(file_name: str, jiggle_file_name: str) -> tuple[list[Circuit], np.ndarray]:
    circs = load_ensemble(file_name)
    print("Num Circs: ", len(circs), flush=True)
    params: np.ndarray = np.load(jiggle_file_name)
    print("Params Shape: ", params.shape, flush=True)
    return circs, params

def load_jiggled_ensemble(file_name: str, jiggle_file_name: str) -> list[tuple[Circuit, np.ndarray]]:
    circs = load_ensemble(file_name)
    print("Num Circs: ", len(circs), flush=True)
    params: np.ndarray = np.load(jiggle_file_name)
    print("Params Shape: ", params.shape, flush=True)
    circ_params = list(zip(circs, params))
    return circ_params
    # if not use_mp:
    #     return create_jiggled_ensemble(circ_params)
    # else:
    #     return create_jiggled_ensemble_mp(circ_params)

def store_ensemble(ensemble: list[Circuit], file_name: str):
    # Store as list of qasm strings
    qasms = [qlang.encode(circ) for circ in ensemble]
    with open(file_name, "w") as f:
        f.write("\nBREAK\n".join(qasms))

def store_ensemble_strs(qasms: list[str], file_name: str):
    # Store as list of qasm strings
    with open(file_name, "w") as f:
        f.write("\nBREAK\n".join(qasms))

def load_ensemble(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    print("SPlit String", flush=True)
    circs = [qlang.decode(qasm, gate_defs = [("gg", gg_gate_def)]) for qasm in qasms]
    print("Decoded", flush=True)
    return circs

def load_ensemble_strs(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    return qasms

def load_ensemble_mp(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    print("SPlit String", flush=True)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        circs = pool.map(qlang.decode, qasms)
    print("Decoded", flush=True)
    return circs

def load_block(circ_name, block_num, extra="") -> str:
    circ_name = f"{circ_name}_{block_num}"
    circ_file = f"good_blocks{extra}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        circ_file = f"bad_blocks{extra}/{circ_name}.qasm"
    return circ_file

def load_cliff_circ(circ_name, precision: int = 5) -> str:
    folder = f"clifft_benchmarks_{precision}"
    circ_file = f"{folder}/{circ_name}.qasm"
    return circ_file

def load_circuit(circ_name: str, timestep: int = 0, opt: bool = False) -> Circuit:
    if "JW" in circ_name:
        circ_name = f"JWCircs/{circ_name}.qasm"

    if opt:
        extra = "_tket"
    else:
        extra = ""
    
    file_name = f"{base_bqskit_dir}/ensemble_benchmarks{extra}/{circ_name}.qasm"
    if not os.path.exists(file_name):
        file_name = f"{base_bqskit_dir}/qce23_qfactor_benchmarks{extra}/{circ_name}.qasm"

    if not os.path.exists(file_name):
        file_name = f"{base_bqskit_dir}/ensemble_benchmarks_new{extra}/{circ_name}.qasm"

    return Circuit.from_file(filename=file_name)

def get_circ_dir(circ_name: int, block_num: int, tol: float, 
                 cliff_t: bool = False, extra: str="") -> str:
    if cliff_t:
        circ_dir = f"{base_checkpoint_dir}_clifft{extra}/{circ_name}_{block_num}_{tol}"
    else:
        circ_dir = f"{base_checkpoint_dir}{extra}/{circ_name}_{block_num}_{tol}"
    return circ_dir

def check_param_shape(circ_name: str, block_num: int, tol: float, extra: str = "") -> list[int]:
    circ_dir = get_circ_dir(circ_name, block_num, tol, extra=extra)
    print("Circ Dir: ", circ_dir, flush=True)
    full_path = f"{circ_dir}/ensemble_final_jiggle.npy"
    if not os.path.exists(full_path):
        # Default to ensemble 0 for now
        full_path = f"{circ_dir}/ensemble_0_jiggles_.npy"
        print("Using default ensemble 0", flush=True)
    params: np.ndarray = np.load(full_path)
    return params.shape

def load_compiled_circs_params_separate(circ_name: str, block_num: int, tol: float, extra: str = "") -> tuple[list[Circuit], 
                                                                                             np.ndarray]:
    circ_dir = get_circ_dir(circ_name, block_num, tol, extra=extra)
    full_path = f"{circ_dir}/ensemble_final_jiggle.npy"
    full_ens_path = f"{circ_dir}/ensemble_final.qasms"
    if not os.path.exists(full_path):
        # Default to ensemble 0 for now
        full_path = f"{circ_dir}/ensemble_0_jiggles_.npy"
        full_ens_path = f"{circ_dir}/ensemble_0_.qasms"
        print("Using default ensemble 0", flush=True)
    print("Full Path: ", full_path, flush=True)
    params: np.ndarray = np.load(full_path)
    circs = load_ensemble(full_ens_path)
    return circs, params

def load_compiled_block_circuits(circ_name: int, 
                                 block_num: int,  
                                 tol: int,
                                 target: UnitaryMatrix = None) -> list[Circuit] | list[tuple[Circuit, 
                                                                             UnitaryMatrix, 
                                                                             float]]:
    
    circ_dir = get_circ_dir(circ_name, block_num, tol)
    full_path = f"{circ_dir}/ensemble_final_jiggle.npy"
    full_ens_path = f"{circ_dir}/ensemble_final.qasms"
    circ_params = load_jiggled_ensemble(full_ens_path, full_path)

    if target is None:
        print("Returning just Circuits", flush=True)
        with mp.Pool(processes=os.cpu_count()) as pool:
            ens: list[list[Circuit]] = pool.map(create_single_jiggled_ensemble, circ_params)
    else:
        with mp.Pool(processes=os.cpu_count()) as pool:
            params = list(zip(circ_params, [target] * len(circ_params), [True] * len(circ_params), [True] * len(circ_params)))
            ens: list[list[tuple[Circuit, UnitaryMatrix, float]]] = pool.starmap(create_single_jiggled_ensemble, 
                                                                                params)
    ens = list(chain.from_iterable(ens))
    return ens

def load_compiled_block_circuits_qp_inds(circ_name: int, 
                                         block_num: int,  
                                         tol: float,
                                         extra: str="") -> tuple[np.ndarray, np.ndarray]:
    circ_dir = get_circ_dir(circ_name, block_num, tol, extra=extra)
    inds_file = f"{circ_dir}/ensemble_final_rand_inds.npy"
    if os.path.exists(inds_file):
        circ_inds = np.load(inds_file)
        probs_file = f"{circ_dir}/ensemble_final_probs.npy"
        circ_probs = np.load(probs_file)
        return circ_inds, circ_probs
    else:
        print("No Indices found for circ", circ_name, block_num, tol, extra, flush=True)
        return None, None

def get_unitary(circ: Circuit):
    return circ.get_unitary()

def get_unitary_vec(circ: Circuit) -> np.ndarray[np.float128]:
    return circ.get_unitary().get_flat_vector()


def get_block_names(circ_name: str, extra: str= "") -> list[str]:
    good_circ_files = glob.glob(f"{good_block_dir}{extra}/{circ_name}_*.qasm")
    bad_circ_files = glob.glob(f"{bad_block_dir}{extra}/{circ_name}_*.qasm")
    all_circ_files = good_circ_files + bad_circ_files

    block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
    return block_nums

def get_circ_names(extra: str = "_tket") -> list[str]:
    good_circ_files = glob.glob(f"{good_block_dir}{extra}/*.qasm")
    bad_circ_files = glob.glob(f"{bad_block_dir}{extra}/*.qasm")
    all_circ_files = good_circ_files + bad_circ_files

    def extract_circ_name(circ_file: str):
        parts = circ_file.split('/')[-1].split('_')
        return '_'.join(parts[:-1])

    circ_names = [extract_circ_name(file) for file in all_circ_files]
    return list(set(circ_names))


def check_if_finished(circ_name: str, 
                      tol: float, 
                      cliff_t: bool = False) -> tuple[bool, bool]:
    '''
    Returns if all blocks have been processed for a circ_name, tol.

    return_1 - True if all blocks have been processed and QP has been run
    return_2  - True if all blocks have been processed minus QP and Check Ensemble
    Quality

    Note: If blocks do not exist, return_1 and return_2 will both be True
    '''
    # Get all block nums for a circ_name
    good_circ_files = glob.glob(f"{good_block_dir}/{circ_name}_*.qasm")
    bad_circ_files = glob.glob(f"{bad_block_dir}/{circ_name}_*.qasm")
    all_circ_files = good_circ_files + bad_circ_files

    if len(all_circ_files) == 0:
        print("No blocks found for circ", circ_name, flush=True)
        return True, True
    
    block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
    # Check if all blocks have been processed
    ret_1 = True
    ret_2 = True
    for block_num in block_nums:
        circ_dir = get_circ_dir(circ_name, block_num, tol, cliff_t)
        full_path = f"{circ_dir}/ensemble_final_rand_ind*.npy"
        jiggle_path = f"{circ_dir}/ensemble_0_jiggles*.npy"
        rand_ind_files = glob.glob(full_path)
        jiggle_files = glob.glob(jiggle_path)
        ret_1 = ret_1 and (len(rand_ind_files) > 0)
        ret_2 = ret_2 and (len(jiggle_files) > 0)
    return ret_1, ret_2