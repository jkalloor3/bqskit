import numpy as np
import csv 
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.runtime import get_runtime

from .distance import normalized_frob_cost, frobenius_cost

class CheckEnsembleQualityPass(BasePass):
    def __init__(self, 
                 count_t: bool = False,
                 csv_name: str = "",
                 ) -> None:
        self.count_t = count_t
        self.csv_name = csv_name
        self.ensemble_names = ["Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        self.gate_title = "T Count" if count_t else "CNOT Count"
        self.gate_func = lambda x: x.count(TGate()) + x.count(TdgGate()) + x.num_params * 60 if count_t else x.count(CNOTGate())
    
    async def get_ensemble_data(self, ens: list[tuple[Circuit, float]], target: UnitaryMatrix, orig_count: int) -> dict[str, Any]:
        ensemble_data = {}
        unitaries: list[UnitaryMatrix] = [x[0].get_unitary() for x in ens]
        norm_e1s = [normalized_frob_cost(un, target) for un in unitaries]
        frob_e1s = [frobenius_cost(un, target) for un in unitaries]
        norm_e1 = np.mean(norm_e1s)
        frob_e1 = np.mean(frob_e1s)
        mean_un = np.mean(unitaries, axis=0)
        norm_bias = normalized_frob_cost(mean_un, target)
        frob_bias = frobenius_cost(mean_un, target)
        
        final_counts = [self.gate_func(circ) for circ, _ in ens]
        ensemble_data["Ensemble Generation Method"] = ""
        ensemble_data["Num Circs"] = len(ens)
        ensemble_data[f"Orig. {self.gate_title}"] = orig_count
        ensemble_data[f"Avg. {self.gate_title}"] = np.mean(final_counts)
        ensemble_data["Norm. Epsilon"] = norm_e1
        ensemble_data["Epsilon"] = frob_e1
        ensemble_data["Norm. Bias"] = norm_bias
        ensemble_data["Bias"] = frob_bias
        norm_ratio = norm_bias / (norm_e1 * norm_e1)
        ensemble_data["Norm. Ratio"] = norm_ratio
        ratio = frob_bias / (frob_e1 * frob_e1)
        ensemble_data["Ratio"] = ratio

        return ensemble_data


    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Check Ensemble Quality and output it to a CSV
        if "ensemble" not in data:
            data["good_ensemble"] = False
            return

        ensemble: list[tuple[Circuit, float]] = data["ensemble"]

        for i in range(3, len(ensemble)):
            self.ensemble_names.append(f"Random Circuits #{i-2}")
        
        target = data.target
        csv_dict: list[dict[str, Any]] = await get_runtime().map(self.get_ensemble_data, ensemble, target=target, orig_count = self.gate_func(circuit))
        final_ratios = []
        for i in range(len(ensemble)):
            csv_dict[i]["Ensemble Generation Method"] = self.ensemble_names[i]
            final_ratios.append(csv_dict[i]["Norm. Ratio"])

        # Ensemble is good if any of the final ratios is less than 10
        data["good_ensemble"] = any([x < 10 for x in final_ratios])

        if data["good_ensemble"]:
            print("FOUND GOOD ENSEMBLE", flush=True)
        
        if "checkpoint_dir" in data:
            checkpoint_data_file: str = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}.csv")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict:
                writer.writerow(row)