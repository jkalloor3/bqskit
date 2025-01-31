import pickle
from bqskit.ir import Circuit
from util import load_compiled_block_circuits, load_compiled_block_circuits_qp_inds
from sys import argv

if __name__ == "__main__":
    circ_name = argv[1]
    block_num = int(argv[2])
    tol = float(argv[3])
    unique_circs = int(argv[4])

    full_ens: list[Circuit] = load_compiled_block_circuits(circ_name, block_num, tol, unique_circs)

    print("Read circuits", flush=True)
    qp_inds, _ = load_compiled_block_circuits_qp_inds(circ_name, block_num, tol, unique_circs)

    print("Read Inds")

    debug_inds = qp_inds[:500]

    debug_ens = [full_ens[i] for i in debug_inds]
    print("Created Debug Ensemble", flush=True)
    pickle.dump(debug_ens, open(f"{circ_name}_{block_num}_{tol}_{unique_circs}_debug.pkl", 'wb'))