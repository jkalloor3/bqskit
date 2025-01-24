from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from pathlib import Path
import pickle
from itertools import chain
import numpy as np
import os
import pandas as pd
import numpy as np
from bqskit.ir.lang import get_language
from .distance import frobenius_cost
import multiprocessing as mp

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
    # params = stack_padding([params for _, params in ensemble], vertical=False)
    params = np.stack([params for _, params in ensemble])
    np.save(jiggle_file_name, params)

def create_single_jiggled_ensemble(circ_params: tuple[Circuit, np.ndarray]) -> list[Circuit]:
    circ, params = circ_params
    ens = []
    # print("Params Shape: ", params.shape, "Circuit Params: ", circ.num_params, flush=True)
    for param in params.tolist():
        new_circ = circ.copy()
        new_circ.set_params(param)
        ens.append(new_circ)
    return ens

def create_jiggled_ensemble(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    return list(chain.from_iterable(ensemble))

def create_jiggled_ensemble_mp(circ_params: list[tuple[Circuit, np.ndarray]]) -> list[Circuit]:
    # ensemble = [create_single_jiggled_ensemble(c) for c in circ_params]
    with mp.Pool(processes=128) as pool:
        ensemble = pool.map(create_single_jiggled_ensemble, circ_params)
    return list(chain.from_iterable(ensemble))

def load_jiggled_ensemble(file_name: str, jiggle_file_name: str, 
                          use_mp: bool = False) -> list[Circuit]:
    circs = load_ensemble(file_name)
    print("Num Circs: ", len(circs), flush=True)
    params: np.ndarray = np.load(jiggle_file_name)
    print("Params Shape: ", params.shape, flush=True)
    circ_params = list(zip(circs, params))
    if not use_mp:
        return create_jiggled_ensemble(circ_params)
    else:
        return create_jiggled_ensemble_mp(circ_params)

def store_ensemble(ensemble: list[Circuit], file_name: str):
    # Store as list of qasm strings
    qasms = [circ.to("qasm") for circ in ensemble]
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
        file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks{opt_str}/{circ_name}_{timestep}{ext}"
    else:
        file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks{opt_str}/{circ_name}{ext}"
    
    if not os.path.exists(file):
        file = f"/pscratch/sd/j/jkalloor/bqskit/qce23_qfactor_benchmarks/{circ_name}{ext}"
    
    if ext == ".qasm":
        return Circuit.from_file(file)
    else:
        return pickle.load(open(file, "rb"))


def save_circuits(circs: list[Circuit], circ_name: str, tol: int, timestep: int, ignore_timestep: bool = False, extra_str=extra) -> None:
    if ignore_timestep:
        full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{circ_name}.pkl")
    else:
        full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    print(full_path)
    pickle.dump(circs, open(full_path, "wb"))

def save_unitaries(utries: list[UnitaryMatrix], circ_name: str, tol: int, timestep: int) -> None:
    full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(utries, open(full_path, "wb"))

def load_compiled_circuits(circ_name: int, tol: int, timestep: int, extra_str=extra, ignore_timestep: bool = False) -> list[Circuit]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl"
    if ignore_timestep:
        full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra_str}/{circ_name}/{tol}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def load_compiled_block_circuits(circ_name: int, block_num: int,  tol: int, num_unique_circs: int) -> list[Circuit]:
    circ_dir = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}"
    full_path = f"{circ_dir}/data.data"
    if not os.path.exists(full_path):
        print("File not found, trying with integer tol", flush=True)
        tol_2 = int(tol)
        circ_dir = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol_2}_{num_unique_circs}"

    csv_path = f"{circ_dir}/data_try1.csv"
    df = pd.read_csv(csv_path, header=0)
    ind = np.argmin(df["Ratio"])
    full_path = f"{circ_dir}/ensemble_final_jiggle.npy"
    full_ens_path = f"{circ_dir}/ensemble_final.qasms"
    if os.path.exists(full_path):
        # ens = load_ensemble_mp(full_path)
        ens = load_jiggled_ensemble(full_ens_path, full_path)
    else:
        data_file = f"{circ_dir}/data.data"
        data = pickle.load(open(data_file, "rb"))
        ens = [x[0] for x in data["ensemble"][ind]]
    print("Selecting ensemble number ", ind, "with a ratio of ", df["Ratio"][ind], "and circ count of ", len(ens), flush=True)
    return ens

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
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_circ_varied/ensemble_shortest_circuits_{vary}_circ/{circ_name}/{tol}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_compiled_unitaries_varied(unitaries, circ_name: int, tol: int, vary: int) -> list[Circuit]:
    full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_unitaries_varied/{vary}_circ/{circ_name}/{tol}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    print(full_path)
    return pickle.dump(unitaries, open(full_path, "wb"))

def load_unitaries(circ_name: int, tol: int, timestep: int) -> list[UnitaryMatrix]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_send_unitaries(unitaries: list[np.ndarray], circ_name: int, tol: int) -> None:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/unitaries_to_send_fix/{tol}/{circ_name}/utries.pkl"
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    return pickle.dump(unitaries, open(full_path, "wb"))

def load_sent_unitaries(circ_name: int, tol: int) -> list[np.ndarray]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/unitaries_to_send/{tol}/{circ_name}/{circ_name}_utries.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_target(target: UnitaryMatrix, circ_name: int) -> None:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/unitaries/{circ_name}.pkl"
    return pickle.dump(target.numpy, open(full_path, "wb"))

def get_unitary(circ: Circuit):
    return circ.get_unitary()

def get_unitary_vec(circ: Circuit) -> np.ndarray[np.float128]:
    return circ.get_unitary().get_flat_vector()