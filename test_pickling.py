from bqskit.compiler.passdata import PassData
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import pickle


un = UnitaryMatrix.random(4)

circ = Circuit.from_unitary(un)
data = PassData(circ)

data["baloney"] = "sandwich"

print("Starting to pickle", flush=True)

pickle.dump(circ, open("circ.pkl", "wb"))

pickle.dump(data, open("data.pkl", "wb"))



# Now open the pickle

new_circ = pickle.load(open("circ.pkl", "rb"))

new_data = pickle.load(open("data.pkl", "rb"))

print(list(new_data.keys()))

print("Done pickling", flush=True)

data.update(new_data)

print("Done updating", flush=True)

