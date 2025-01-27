from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from pathlib import Path
import pickle
import numpy as np
import os
import pandas as pd
import numpy as np
from bqskit.ir.lang import get_language
from .distance import frobenius_cost

extra = "_qsearch"

def store_ensemble(ensemble: list[Circuit, float], file_name: str, has_float: bool = True):
    # Store as list of qasm strings
    if has_float:
        qasms = [circ.to("qasm") for circ, _ in ensemble]
    else:
        qasms = [circ.to("qasm") for circ in ensemble]
    with open(file_name, "w") as f:
        f.write("\nBREAK\n".join(qasms))

def load_ensemble(file_name: str, target: UnitaryMatrix, add_floats: bool = True) -> list[Circuit]:
    with open(file_name, "r") as f:
        qasms = f.read().split("\nBREAK\n")
    lang = get_language("qasm")
    circs = [lang.decode(qasm) for qasm in qasms]
    if add_floats:
        return [(circ, frobenius_cost(circ.get_unitary(), target)) for circ in circs]
    else:
        return circs

def load_block(circ_name, block_num, good=True) -> str:
    circ_name = f"{circ_name}_{block_num}"
    if good:
        circ_file = f"good_blocks/{circ_name}.qasm"
    else:
        circ_file = f"bad_blocks/{circ_name}.qasm"
    return circ_file

def load_cliff_circ(circ_name, precision: int = 5) -> str:
    folder = f"clifft_benchmarks_{precision}"
    circ_file = f"{folder}/{circ_name}.qasm"
    return circ_file

def load_circuit(circ_name: str, timestep: int = 0, opt: bool = False) -> Circuit:
    opt_str = "_opt" if opt else ""
    
    if "JW" in circ_name:
        circ_name = f"JWCircs/{circ_name}"
    
    ext = ".qasm"
    
    if timestep > 0:
        file = f"/Users/jkalloor3/BQSKit/bqskit/ensemble_benchmarks{opt_str}/{circ_name}_{timestep}{ext}"
    else:
        file = f"/Users/jkalloor3/BQSKit/bqskit/ensemble_benchmarks{opt_str}/{circ_name}{ext}"
    
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
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}/data.data"
    csv_path = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}/data_try1.csv"
    ens_path = f"/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_nisq_0/adder9_0_1_250/ensemble_final.qasms"
    if os.path.exists(ens_path):
        return load_ensemble(ens_path, None, add_floats=False)
    data = pickle.load(open(full_path, "rb"))["ensemble"]
    df = pd.read_csv(csv_path, header=0)
    ind = np.argmin(df["Ratio"])
    print("Selecting ensemble number ", ind, "with a ratio of ", df["Ratio"][ind], "and circ count of ", len(data[ind]), flush=True)
    return [x for x, _ in data[ind]]

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