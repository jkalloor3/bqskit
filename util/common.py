from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from pathlib import Path
import pickle
import numpy as np
import os

extra = "_qsearch"

def load_circuit(circ_name: str, timestep: int = 0) -> Circuit:
    if timestep > 0:
        file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_name}_{timestep}.qasm"
    else:
        file = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_name}.qasm"
    
    if not os.path.exists(file):
        file = f"/pscratch/sd/j/jkalloor/bqskit/qce23_qfactor_benchmarks/{circ_name}.qasm"

    return Circuit.from_file(file)


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