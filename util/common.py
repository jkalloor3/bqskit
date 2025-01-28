from bqskit.ir.circuit import Circuit
from .fix_global_phase import FixGlobalPhasePass
from bqskit.qis import UnitaryMatrix
from pathlib import Path
import pickle
from itertools import chain
import numpy as np
import os
import pandas as pd
import numpy as np
from bqskit.ir.lang import get_language
from .distance import frobenius_cost, normalized_frob_cost
import multiprocessing as mp
from bqskit.runtime import get_runtime

extra = "_qsearch"

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

def store_jiggled_ensemble(ensemble: list[tuple[Circuit, np.ndarray]], file_name: str, jiggle_file_name: str):
    # Get circuits
    circs = [circ for circ, _ in ensemble ]
    store_ensemble(circs, file_name)
    params = stack_padding([params for _, params in ensemble], vertical=False)
    # params = np.stack([params for _, params in ensemble])
    np.save(jiggle_file_name, params)

def create_single_jiggled_ensemble(circ_params: tuple[Circuit, np.ndarray], 
                                   target: UnitaryMatrix = None, fix_phase: bool = False,
                                   add_cost: bool = False) -> list[Circuit] | list[tuple[Circuit, UnitaryMatrix, float]]:
    circ, params = circ_params
    ens = []
    # print("Params Shape: ", params.shape, "Circuit Params: ", circ.num_params, flush=True)
    for param in params.tolist():
        new_circ = circ.copy()
        new_circ.set_params(param)
        if fix_phase:
            FixGlobalPhasePass.fix_phase(new_circ, target)
        if add_cost:
            un = new_circ.get_unitary()
            cost_1 = normalized_frob_cost(un, target)
            ens.append((new_circ, un, cost_1))
        else:
            ens.append(new_circ)
    return ens

def create_jiggled_unitaries(circ_params: tuple[Circuit, np.ndarray], 
                                   target: UnitaryMatrix = None, fix_phase: bool = False,
                                   add_cost: bool = False) -> list[tuple[UnitaryMatrix]] | list[tuple[UnitaryMatrix, float]]:
    circ, params = circ_params
    ens = []
    # print("Params Shape: ", params.shape, "Circuit Params: ", circ.num_params, flush=True)
    for param in params.tolist():
        utry = circ.get_unitary(param)
        if fix_phase:
            gp_correction = target.get_target_correction_factor(utry)
            utry = utry * gp_correction
        if add_cost:
            cost_1 = normalized_frob_cost(utry, target)
            ens.append((utry, cost_1))
        else:
            ens.append(utry)
    return ens



def create_jiggled_ensemble(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    return list(chain.from_iterable(ensemble))

def create_jiggled_ensemble_mp(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    # ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    with mp.Pool(processes=128) as pool:
        ensemble = pool.map(create_single_jiggled_ensemble, circ_params)
    return list(chain.from_iterable(ensemble))

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
    lang = get_language("qasm")
    qasms = [lang.encode(circ) for circ in ensemble]
    with open(file_name, "w") as f:
        f.write("\nBREAK\n".join(qasms))

def load_ensemble(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    lang = get_language("qasm")
    print("SPlit String", flush=True)
    # with mp.Pool(processes=128) as pool:
    #     circs = pool.map(lang.decode, qasms)
    circs = [lang.decode(qasm) for qasm in qasms]
    # circs = [lang.decode(qasm) for qasm in qasms]
    print("Decoded", flush=True)
    return circs

def load_ensemble_mp(file_name: str) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    lang = get_language("qasm")
    print("SPlit String", flush=True)
    with mp.Pool(processes=128) as pool:
        circs = pool.map(lang.decode, qasms)
    print("Decoded", flush=True)
    return circs

def load_block(circ_name, block_num, good=True) -> str:
    circ_name = f"{circ_name}_{block_num}"
    if good:
        circ_file = f"good_blocks/{circ_name}.qasm"
    else:
        circ_file = f"bad_blocks/{circ_name}.qasm"
    return circ_file


def load_circuit(circ_name: str, timestep: int = 0, opt: bool = False) -> Circuit:
    opt_str = "_opt" if opt else ""
    
    if "JW" in circ_name:
        circ_name = f"JWCircs/{circ_name}"
    
    ext = ".qasm"
    
    if timestep > 0:
        return Circuit.from_file(f"/home/jkalloor/bqskit/ensemble_benchmarks/{circ_name}_{timestep}.qasm")
    else:
        return Circuit.from_file(f"/home/jkalloor/bqskit/ensemble_benchmarks/{circ_name}.qasm")


def save_circuits(circs: list[Circuit], circ_name: str, tol: int, timestep: int, ignore_timestep: bool = False, extra_str=extra) -> None:
    if ignore_timestep:
        full_path = Path(f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{circ_name}.pkl")
    else:
        full_path = Path(f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    print(full_path)
    pickle.dump(circs, open(full_path, "wb"))

def save_unitaries(utries: list[UnitaryMatrix], circ_name: str, tol: int, timestep: int) -> None:
    full_path = Path(f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(utries, open(full_path, "wb"))

def load_compiled_circuits(circ_name: int, tol: int, timestep: int, extra_str=extra, ignore_timestep: bool = False) -> list[Circuit]:
    full_path = f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl"
    if ignore_timestep:
        full_path = f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def get_circ_dir(circ_name: int, block_num: int, tol: int, num_unique_circs: int) -> str:
    circ_dir = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}"
    full_path = f"{circ_dir}/data.data"
    if not os.path.exists(full_path):
        print("File not found, trying with integer tol", flush=True)
        tol_2 = int(tol)
        circ_dir = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol_2}_{num_unique_circs}"
    return circ_dir

def load_compiled_block_circuits(circ_name: int, 
                                 block_num: int,  
                                 tol: int, 
                                 num_unique_circs: int,
                                 target: UnitaryMatrix = None) -> list[tuple[Circuit, 
                                                                             UnitaryMatrix, 
                                                                             float]]:
    circ_dir = get_circ_dir(circ_name, block_num, tol, num_unique_circs)
    full_path = f"{circ_dir}/ensemble_final_jiggle.npy"
    full_ens_path = f"{circ_dir}/ensemble_final.qasms"
    circ_params = load_jiggled_ensemble(full_ens_path, full_path)
    with mp.Pool(processes=128) as pool:
        params = list(zip(circ_params, [target] * len(circ_params), [True] * len(circ_params), [True] * len(circ_params)))
        ens: list[list[tuple[Circuit, UnitaryMatrix, float]]] = pool.starmap(create_single_jiggled_ensemble, 
                                                                              params)
    ens = list(chain.from_iterable(ens))
    return ens

def load_compiled_block_circuits_qp_inds(circ_name: int, 
                                         block_num: int,  
                                         tol: int, 
                                         num_unique_circs: int) -> tuple[np.ndarray, np.ndarray]:
    circ_dir = get_circ_dir(circ_name, block_num, tol, num_unique_circs)
    inds_file = f"{circ_dir}/ensemble_final_rand_inds.npy"
    circ_inds = np.load(inds_file)
    probs_file = f"{circ_dir}/ensemble_final_probs.npy"
    circ_probs = np.load(probs_file)
    return circ_inds, circ_probs

def load_compiled_block_circuits_qp(circ_name: int, block_num: int,  tol: int, num_unique_circs: int) -> list[tuple[Circuit, float]]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}/data.data"
    data = pickle.load(open(full_path, "rb"))
    if "final_ensemble_probs" not in data:
        return []
    orig_ensemble = data["final_ensemble"]
    probs = data["final_ensemble_probs"]
    # Sample 10000 circuits according to probs
    ens_inds = np.random.choice(len(orig_ensemble), size=10000, p=probs)
    ens = [orig_ensemble[i] for i in ens_inds]
    return ens


def load_compiled_circuits_varied(circ_name: int, tol: int, vary: int) -> list[Circuit]:
    full_path = f"/home/jkalloor/bqskit/ensemble_circ_varied/ensemble_shortest_circuits_{vary}_circ/{circ_name}/{tol}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_compiled_unitaries_varied(unitaries, circ_name: int, tol: int, vary: int) -> list[Circuit]:
    full_path = Path(f"/home/jkalloor/bqskit/ensemble_unitaries_varied/{vary}_circ/{circ_name}/{tol}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    print(full_path)
    return pickle.dump(unitaries, open(full_path, "wb"))

def load_unitaries(circ_name: int, tol: int, timestep: int) -> list[UnitaryMatrix]:
    full_path = f"/home/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_send_unitaries(unitaries: list[np.ndarray], circ_name: int, tol: int) -> None:
    full_path = f"/home/jkalloor/bqskit/unitaries_to_send_fix/{tol}/{circ_name}/utries.pkl"
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    return pickle.dump(unitaries, open(full_path, "wb"))

def load_sent_unitaries(circ_name: int, tol: int) -> list[np.ndarray]:
    full_path = f"/home/jkalloor/bqskit/unitaries_to_send/{tol}/{circ_name}/{circ_name}_utries.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_target(target: UnitaryMatrix, circ_name: int) -> None:
    full_path = f"/home/jkalloor/bqskit/unitaries/{circ_name}.pkl"
    return pickle.dump(target.numpy, open(full_path, "wb"))

def get_unitary(circ: Circuit):
    return circ.get_unitary()

def get_unitary_vec(circ: Circuit) -> np.ndarray[np.float128]:
    return circ.get_unitary().get_flat_vector()