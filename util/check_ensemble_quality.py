import numpy as np
import csv 
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from math import ceil
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.ir import Circuit
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.qis import UnitaryMatrix
from bqskit.runtime import get_runtime
import os
import time
import shutil
from .common import load_jiggled_ensemble, create_avg_utry
from .counter import count_params

from .distance import frobenius_cost, normalized_frob_cost

norm_cost = GPNormalizedFrobeniusCostGenerator()
frob_cost = GPNormalizedFrobeniusCostGenerator()

class CheckEnsembleQualityPass(BasePass):
    def __init__(self, 
                 count_t: bool = False,
                 csv_name: str = "",
                 checkpoint_extra_str: str = "",
                 shm_percentage: float = 1.0,
                 ) -> None:
        self.count_t = count_t
        self.csv_name = csv_name
        self.ensemble_names = ["Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        for i in range(10):
            # Default Names
            self.ensemble_names.append(f"Random Circuits #{i}")
        self.gate_title = "Num Params" if count_t else "CNOT Count"
        self.gate_func = lambda x: x.count(TGate()) + x.count(TdgGate()) + x.num_params * 60 if count_t else x.count(CNOTGate())
        self.checkpoint_extra_str = checkpoint_extra_str
        self.shm_percentage = shm_percentage

    def get_ensemble_data(self, avg_utry: np.ndarray, 
                          avg_dist: float, 
                          target: UnitaryMatrix, 
                          orig_count: int) -> dict[str, Any]:
        ensemble_data = {}
        dim = avg_utry.shape[0]
        print("Average Norm Epsilon: ", avg_dist, flush=True)
        frob_factor = np.sqrt(dim * 2)
        norm_e1 = avg_dist
        frob_e1 = avg_dist * frob_factor
        mean_un = avg_utry
        norm_bias = normalized_frob_cost(mean_un, target)
        frob_bias = frobenius_cost(mean_un, target)
        
        # final_counts = [self.gate_func(circ) for circ in ens]
        ensemble_data["Ensemble Generation Method"] = ""
        ensemble_data["Num Circs"] = 20000
        ensemble_data[f"Orig. {self.gate_title}"] = orig_count
        # ensemble_data[f"Avg. {self.gate_title}"] = np.mean(final_counts)
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
        print("Check Ensemble Quality Pass", flush=True)
        checkpoint_dir: str = data["checkpoint_dir"]
        final_ens_file = f"{checkpoint_dir}/ensemble_final.qasms"
        final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle.npy"

        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting Check Ensemble Quality Pass", flush=True)
        
        if os.path.exists(final_ens_file):
            # Load the ensemble from the checkpoint
            print("Already Checked!", flush=True)
            return
        
        # Otherwise, reload from saved files - Would have done in 
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        final_ens_file = os.path.join(checkpoint_dir, "ensemble_final.qasms")
        start_ens_ind = 0
        ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        ensemble_counts = {}

        target = data.target
        csv_dict = {}
        
        # Create new shared memory
        best_ind = 0
        best_ratio = float("inf")
        best_count = float("inf")

        if os.path.exists(jiggle_file):
            # Load the ensemble from the checkpoint
            while os.path.exists(jiggle_file):
                circ_params = load_jiggled_ensemble(ens_file, jiggle_file)
                circuits = [circ for circ, _ in circ_params]
                avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                           circ_params, 
                                                           target=data.target, 
                                                           add_cost=True)
                
                utries = [avg_utry for avg_utry, _ in avg_utries_dists]
                dists = [dist for _, dist in avg_utries_dists]
                avg_utry = np.mean(utries, axis=0)
                avg_dist = np.mean(dists)
                print("Avg Dist: ", avg_dist, flush=True)
                csv_dict[start_ens_ind] = self.get_ensemble_data(avg_utry, avg_dist, target, None)
                csv_dict[start_ens_ind]["Ensemble Generation Method"] = self.ensemble_names[start_ens_ind]
                ratio = csv_dict[start_ens_ind]["Ratio"]
                print("Ratio: ", ratio, flush=True)
                count = np.mean([count_params(c) for c in circuits])
                csv_dict[start_ens_ind]["Avg. Count"] = count
                ensemble_counts[start_ens_ind] = count
                print("Avg Count Post Jiggle Load: ", count, flush=True)
                if ratio < 10:
                    best_ind = start_ens_ind
                    best_ratio = ratio
                    best_count = count
                    print("FOUND GOOD ENSEMBLE", flush=True)
                    break
                else:
                    if ratio < best_ratio and count < best_count:
                        best_ind = start_ens_ind
                        best_ratio = ratio
                        best_count = count
                    # Keep Looking
                    start_ens_ind += 1
                    ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
                    jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        
        if "checkpoint_dir" in data:
            checkpoint_data_file: str = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}.csv")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                    fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict.values():
                writer.writerow(row)
            # Copy best jiggled ensemble file to new file name
            best_ensemble_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_{self.checkpoint_extra_str}.qasms"
            best_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_jiggles_{self.checkpoint_extra_str}.npy"
            shutil.copyfile(best_ensemble_file_name, final_ens_file)
            shutil.copyfile(best_file_name, final_ens_jiggle_file)

