from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from pathlib import Path
import pickle
import numpy as np

extra = "_bounded_2"

def load_circuit(circ_name: str, timestep: int = 0) -> Circuit:
    if timestep > 0:
        return Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_name}_{timestep}.qasm")
    else:
        return Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_name}.qasm")


def save_circuits(circs: list[Circuit], circ_name: str, tol: int, timestep: int) -> None:
    full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(circs, open(full_path, "wb"))

def save_unitaries(utries: list[UnitaryMatrix], circ_name: str, tol: int, timestep: int) -> None:
    full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(utries, open(full_path, "wb"))

def load_compiled_circuits(circ_name: int, tol: int, timestep: int) -> list[Circuit]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def load_unitaries(circ_name: int, tol: int, timestep: int) -> list[UnitaryMatrix]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits{extra}/{circ_name}/{tol}/{timestep}/{circ_name}_utries.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def save_send_unitaries(unitaries: list[np.ndarray], circ_name: int, tol: int) -> None:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/unitaries_to_send_2/{tol}/{circ_name}/utries.pkl"
    print(full_path)
    return pickle.dump(unitaries, open(full_path, "wb"))

def get_unitary(circ: Circuit):
    return circ.get_unitary()