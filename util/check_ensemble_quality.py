import numpy as np
import csv 
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from math import ceil
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.ir import Circuit
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation, CircuitLocationLike
from bqskit.passes import ForEachBlockPass
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.qis import UnitaryMatrix
from bqskit.runtime import get_runtime
import os
import time
import shutil
from .common import load_jiggled_ensemble, create_avg_utry
from .counter import count_params

from .distance import frobenius_cost, normalized_frob_cost, hs_cost

norm_cost = GPNormalizedFrobeniusCostGenerator()
frob_cost = GPNormalizedFrobeniusCostGenerator()

class CheckEnsembleQualityPass(BasePass):
    def __init__(self, 
                 count_t: bool = False,
                 csv_name: str = "",
                 checkpoint_extra_str: str = "",
                 calculate_hs: bool = False,
                 sample_blocks: bool = False,
                 zero_threshold: float = 1e-10
                 ) -> None:
        self.count_t = count_t
        self.csv_name = csv_name
        self.ensemble_names = ["Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        self.sample_blocks = sample_blocks
        for i in range(10):
            # Default Names
            self.ensemble_names.append(f"Random Circuits #{i}")
        self.gate_title = "Num Params" if count_t else "CNOT Count"
        self.gate_func = lambda x: x.count(TGate()) + x.count(TdgGate()) + x.num_params * 60 if count_t else x.count(CNOTGate())
        self.checkpoint_extra_str = checkpoint_extra_str
        self.calculate_hs = calculate_hs
        self.zero_threshold = zero_threshold

    def get_ensemble_data(self, avg_utry: np.ndarray, 
                          avg_dist: float, 
                          target: UnitaryMatrix, 
                          orig_count: int,
                          avg_hs: float = None) -> dict[str, Any]:
        ensemble_data = {}
        dim = avg_utry.shape[0]
        frob_factor = np.sqrt(dim * 2)
        norm_e1 = avg_dist
        frob_e1 = avg_dist * frob_factor
        mean_un = avg_utry
        norm_bias = normalized_frob_cost(mean_un, target)
        frob_bias = frobenius_cost(mean_un, target)

        if self.calculate_hs:
            mean_hs = hs_cost(mean_un, target)
            ensemble_data["HS of Mean"] = mean_hs
            ensemble_data["Avg. HS"] = avg_hs
        
        # final_counts = [self.gate_func(circ) for circ in ens]
        ensemble_data["Ensemble Generation Method"] = ""
        ensemble_data["Num Circs"] = 20000
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
        checkpoint_data_file: str = data["checkpoint_data_file"]
        csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}{self.checkpoint_extra_str}.csv")
        final_qasms_file = f"{checkpoint_dir}/ensemble_final{self.checkpoint_extra_str}.qasms"
        final_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle{self.checkpoint_extra_str}.npy"
        final_probs_file = f"{checkpoint_dir}/ensemble_final_probs{self.checkpoint_extra_str}.npy"
        final_cache_file = f"{checkpoint_dir}/ensemble_final_cache{self.checkpoint_extra_str}.pkl"

        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting Check Ensemble Quality Pass", flush=True)
        
        if os.path.exists(final_probs_file):
            print("Final Probs File already exists, skipping pass", flush=True)
            return

        
        # Otherwise, reload from saved files - Would have done in 
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        jiggle_file_name_sub = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}_sub.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
        ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file_sub = jiggle_file_name_sub.format(ind=0, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=0, extra=self.checkpoint_extra_str)

        # Compare 3 different probability files
        ensemble = []
        for i in range (1, 4):
            probs_file = f"{checkpoint_dir}/ensemble_all_probs_{i}_{self.checkpoint_extra_str}.npy"
            if i == 3:
                # Last ensemble may be shorter, so find the smaller jiggle file
                if os.path.exists(jiggle_file_sub):
                    print("Using Smaller Jiggle File: ", jiggle_file_sub, flush=True)
                    jiggle_file = jiggle_file_sub
            if os.path.exists(probs_file):
                ensemble.append((ens_file, jiggle_file, cache_file, probs_file))

        if len(ensemble) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return

        target = data.target
        csv_data = []
        
        # Create new shared memory
        best_ind = 0
        best_ratio = float("inf")

        probs_titles = ["Uniform", "FW Outer", "FW Seeded"]


        for ens_ind, ens in enumerate(ensemble):
            ens_file, jiggle_file, cache_file, probs_file = ens
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file, cache_file, probs_file)
            if len(circ_params[0][1]) == 0:
                print("No params found in ensemble, skipping ensemble", 
                      flush=True)
                continue
            if len(circ_params) < 4:
                # Make 10 copies of each circuit and split the params amongst them
                new_circ_params = []
                for circ, params, probs, cache in circ_params:
                    param_chunks = np.array_split(params, 10)
                    probs_chunks = np.array_split(probs, 10)
                    for i in range(10):
                        if len(param_chunks[i]) == 0:
                            break
                        else:
                            new_circ_params.append((circ, 
                                                    param_chunks[i], 
                                                    probs_chunks[i], 
                                                    cache))

                circ_params = new_circ_params
            all_probs = np.concatenate([probs for _, _, probs, _ in circ_params])
            # Calculate number of non-zero (below a threshold) probs in all_probs
            non_zero_probs = np.sum(all_probs > self.zero_threshold)
            print("All Probs Shape: ", all_probs.shape, np.sum(all_probs), flush=True)
            avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                        circ_params,
                                                        target=data.target, 
                                                        add_cost=True)
            
            utries = [avg_utry for avg_utry, _, _ in avg_utries_dists]
            dists = [dist for _, dist, _ in avg_utries_dists]
            hs_dists = [hs for _, _, hs in avg_utries_dists]
            avg_utry = np.sum(utries, axis=0)
            avg_dist = np.sum(dists)
            avg_hs = np.sum(hs_dists)
            print("Avg Dist: ", avg_dist, flush=True)
            ensemble_data = self.get_ensemble_data(avg_utry, avg_dist, target, None, avg_hs=avg_hs)
            ensemble_data["Num Non-Zero Probs"] = non_zero_probs
            params = [params for _, params, _, _ in circ_params]
            ensemble_data["Num Circs"] = len(circ_params) * params[0].shape[0]
            ensemble_data["Ensemble Generation Method"] = self.ensemble_names[ens_ind]
            ratio = ensemble_data["Ratio"]
            print("New Ratio: ", ratio, flush=True)
            ensemble_data["Probability Method"] = probs_titles[ens_ind]
            csv_data.append(ensemble_data)
            if ratio <= 2.5:
                best_ind = ens_ind
                best_ratio = ratio
                print("FOUND GOOD ENSEMBLE", flush=True)
                break
            else:
                if ratio < best_ratio:
                    best_ind = ens_ind
                    best_ratio = ratio
                    print("FOUND BETTER ENSEMBLE", flush=True)
        
        if len(csv_data) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return

        if "checkpoint_dir" in data:
            writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                    fieldnames=csv_data[0].keys())
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
            # Copy best jiggled ensemble file to new file name
            best_ensemble_file_name = ensemble[best_ind][0]
            best_jiggle_file_name = ensemble[best_ind][1]
            best_cache_file_name = ensemble[best_ind][2]
            best_probs_file = ensemble[best_ind][3]
            if os.path.exists(best_ensemble_file_name):
                shutil.copyfile(best_ensemble_file_name, final_qasms_file)
                shutil.copyfile(best_jiggle_file_name, final_jiggle_file)
                shutil.copyfile(best_probs_file, final_probs_file)
                if os.path.exists(best_cache_file_name):
                    shutil.copyfile(best_cache_file_name, final_cache_file)
                print("Copied best ensemble files to final files", flush=True)