from bqskit.ir.circuit import Circuit
from pathlib import Path
import pickle

def load_circuit(circ_name: str) -> Circuit:
    return Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_name}.qasm")


def save_circuits(circs: list[Circuit], circ_name: str, tol: int, timestep: int) -> None:
    full_path = Path(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits/{circ_name}/{tol}/{timestep}/{circ_name}.pkl")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(circs, open(full_path, "wb"))

def load_compiled_circuits(circ_name: int, tol: int, timestep: int) -> list[Circuit]:
    full_path = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_shortest_circuits/{circ_name}/{tol}/{timestep}/{circ_name}.pkl"
    print(full_path)
    return pickle.load(open(full_path, "rb"))

def get_unitary(circ: Circuit):
    return circ.get_unitary()