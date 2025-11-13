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
from .counter import GateCounter

from .distance import frobenius_cost, normalized_frob_cost, hs_cost

norm_cost = GPNormalizedFrobeniusCostGenerator()
frob_cost = GPNormalizedFrobeniusCostGenerator()

def get_avg_count(circ_params: tuple[Circuit, np.ndarray, np.ndarray, Any],
                  count_t: bool = False,
                  success_threshold: float = 0.0) -> float:
    circ, params, probs, cache = circ_params
    gate_counter_full = GateCounter(est=False, 
                                    cache=cache)
    
    total_count = 0
    if np.sum(probs) < 1e-4:
        # No need to add this to the count, not significant
        return 0.0
    
    for j, param in enumerate(params):
        circ.set_params(param)
        if count_t:
            new_count = gate_counter_full.count_t(circ, 
                                                success_threshold,
                                                skip_fix=True)
        else:
            new_count = circ.count(CNOTGate())
        total_count += new_count * probs[j]
    return total_count

class CheckEnsembleQualityPass(BasePass):
    def __init__(self,
                 success_threshold: float = 1e-4,
                 count_t: bool = False,
                 csv_name: str = "",
                 max_ratio: float = 20.0,
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
        self.max_ratio = max_ratio
        self.success_threshold = success_threshold

    def get_ensemble_data(self, avg_utry: np.ndarray, 
                          avg_dist: float, 
                          target: UnitaryMatrix, 
                          avg_count: float,
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
        ensemble_data["Count"] = avg_count

        return ensemble_data

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Check Ensemble Quality and output it to a CSV
        print("Check Ensemble Quality Pass", flush=True)
        checkpoint_dir: str = data["checkpoint_dir"]
        checkpoint_data_file: str = data["checkpoint_data_file"]
        csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}{self.checkpoint_extra_str}_{int(self.max_ratio)}.csv")
        final_qasms_file = f"{checkpoint_dir}/ensemble_final{self.checkpoint_extra_str}_FINAL.qasms"
        final_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle{self.checkpoint_extra_str}_FINAL.npy"
        final_probs_file = f"{checkpoint_dir}/ensemble_final_probs{self.checkpoint_extra_str}_FINAL.npy"
        final_cache_file = f"{checkpoint_dir}/ensemble_final_cache{self.checkpoint_extra_str}_FINAL.pkl"

        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting Check Ensemble Quality Pass", flush=True)

        # Otherwise, reload from saved files
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        jiggle_file_name_sub = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}_sub.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
        ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file_sub = jiggle_file_name_sub.format(ind=0, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=0, extra=self.checkpoint_extra_str)

        # Compare different probability files
        ensemble = []
        probs_titles = []
        for ind in range (3):
            prob_titles = ["Uniform", "FW Outer", "FW Seeded"]
            for prob in range(1, 4):
                ens_file = ensemble_file_name.format(ind=ind, extra=self.checkpoint_extra_str)
                jiggle_file = jiggle_file_name.format(ind=ind, extra=self.checkpoint_extra_str)
                jiggle_file_sub = jiggle_file_name_sub.format(ind=ind, extra=self.checkpoint_extra_str)
                cache_file = cache_file_name.format(ind=ind, extra=self.checkpoint_extra_str)
                probs_file = f"{checkpoint_dir}/ensemble_{ind}_probs_{prob}_{self.checkpoint_extra_str}.npy"
                if ind == 0 and prob == 3:
                    # Last ensemble may be shorter, so find the smaller jiggle file
                    if os.path.exists(jiggle_file_sub):
                        print("Using Smaller Jiggle File: ", jiggle_file_sub, flush=True)
                        jiggle_file = jiggle_file_sub
                if os.path.exists(probs_file):
                    ensemble.append((ens_file, jiggle_file, cache_file, probs_file))
                    probs_titles.append(prob_titles[prob - 1])


        # Add default ensemble
        default_ens_file = ensemble_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        default_jiggle_file = jiggle_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        default_cache_file = cache_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        default_probs_file = f"{checkpoint_dir}/ensemble_def_probs_{self.checkpoint_extra_str}.npy"
        probs_titles.append("Default Ensemble")

        # Add default ensemble if it exists
        if os.path.exists(default_probs_file):
            ensemble.append((default_ens_file, 
                             default_jiggle_file, 
                             default_cache_file, 
                             default_probs_file))
            # If file already exists, only append new data
            if os.path.exists(csv_file):
                ensemble = [ensemble[-1]]
                probs_titles = [probs_titles[-1]]

        if len(ensemble) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return

        target = data.target

        csv_data = []
        
        best_ind = -1
        min_count = float('inf')

        for ens_ind, ens in enumerate(ensemble):
            ens_file, jiggle_file, cache_file, probs_file = ens
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file, cache_file, probs_file)
            if len(circ_params[0][1]) == 0:
                print("No params found in ensemble, skipping ensemble", 
                      flush=True)
                continue
            # if len(circ_params) < 4:
            #     # Make 10 copies of each circuit and split the params amongst them
            #     new_circ_params = []
            #     for circ, params, probs, cache in circ_params:
            #         param_chunks = np.array_split(params, 10)
            #         probs_chunks = np.array_split(probs, 10)
            #         for i in range(10):
            #             if len(param_chunks[i]) == 0:
            #                 break
            #             else:
            #                 new_circ_params.append((circ, 
            #                                         param_chunks[i], 
            #                                         probs_chunks[i], 
            #                                         cache))

                # circ_params = new_circ_params
            all_probs = np.concatenate([probs for _, _, probs, _ in circ_params])
            # Calculate number of non-zero (below a threshold) probs in all_probs
            non_zero_probs = np.sum(all_probs > self.zero_threshold)
            avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                        circ_params,
                                                        target=data.target, 
                                                        add_cost=True)
            avg_counts = await get_runtime().map(get_avg_count,
                                                circ_params,
                                                count_t = self.count_t,
                                                success_threshold=self.success_threshold
            )
            avg_count = np.sum(avg_counts)
            utries = [avg_utry for avg_utry, _, in avg_utries_dists]
            dists = [dist for _, dist in avg_utries_dists]
            avg_utry = np.sum(utries, axis=0)
            avg_norm_dist = np.sum(dists)
            avg_dist = avg_norm_dist * np.sqrt(avg_utry.shape[0] * 2)
            total_error = frobenius_cost(avg_utry, target)
            ratio = total_error / (avg_dist * avg_dist)
            # print("Ratio: ", ratio, "Avg Count: ", avg_count, flush=True)
            ensemble_data = {}
            ensemble_data["Probability Method"] = probs_titles[ens_ind]
            ensemble_data["Avg. Dist"] = avg_dist
            ensemble_data["Total Error"] = total_error
            ensemble_data["Non-Zero Probs"] = int(non_zero_probs)
            ensemble_data["Num Circs"] = len(circ_params)
            ensemble_data["Ratio"]  = ratio
            ensemble_data["Count"] = avg_count
            ensemble_data["Orig Count"] = circuit.count(CNOTGate())
            csv_data.append(ensemble_data)

            if ratio <= self.max_ratio and avg_count < min_count:
                best_ind = ens_ind
                min_count = avg_count
        
        if len(csv_data) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return
        
        if best_ind >= 0 and not self.count_t:
            best_ens = ensemble[best_ind]
            ens_file, jiggle_file, cache_file, probs_file = best_ens
            best_ratio = csv_data[best_ind]["Ratio"]
            old_best_ratio = best_ratio
            best_count = csv_data[best_ind]["Count"]
            old_best_count = best_count
            num_adjustments = 0
            circ_params_best = load_jiggled_ensemble(ens_file, 
                                                     jiggle_file, 
                                                     cache_file, 
                                                     probs_file)
            
            orig_counts = [c.count(CNOTGate()) for c, _, _, _ in circ_params_best]
            diff = [best_count - a for a in orig_counts]

            probs = [p for _, _, p, _ in circ_params]
            out_probs = np.array([np.sum(p) for p in probs])
            # Clip out very small probs
            out_probs.clip(1e-11, 1)
            inner_probs = np.array([p / np.sum(p) for p in probs])
            adjustment_factor = 5.0
            np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
            while best_ratio < self.max_ratio and num_adjustments < 20:
                # Adjust probabilities to minimize count
                new_out_probs = out_probs.copy()
                # Add an adjustment proportional to the counts
                new_out_probs = np.array([p + p * (adjustment_factor * d) for p, d in zip(new_out_probs, diff)])
                new_out_probs.clip(1e-11, 1)
                new_out_probs = new_out_probs / np.sum(new_out_probs)
                # print(out_probs, new_out_probs, flush=True)
                new_probs = [p * n for p, n in zip(inner_probs, new_out_probs)]
                new_circ_params = [(circ, param, new_p, cache) for (circ, param, _, cache), new_p in zip(circ_params, new_probs)]
                avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                            new_circ_params,
                                                            target=data.target, 
                                                            add_cost=True)
                avg_counts = await get_runtime().map(get_avg_count,
                                                    new_circ_params,
                                                    count_t = self.count_t,
                                                    success_threshold=self.success_threshold
                )
                avg_count = np.sum(avg_counts)
                utries = [avg_utry for avg_utry, _, in avg_utries_dists]
                dists = [dist for _, dist in avg_utries_dists]
                avg_utry = np.sum(utries, axis=0)
                avg_norm_dist = np.sum(dists)
                avg_dist = avg_norm_dist * np.sqrt(avg_utry.shape[0] * 2)
                total_error = frobenius_cost(avg_utry, target)
                ratio = total_error / (avg_dist * avg_dist)
                if ratio < self.max_ratio:
                    best_ratio = ratio
                    out_probs = new_out_probs
                    if avg_count < best_count:
                        best_count = avg_count
                        final_probs = np.array(new_probs)
                else:
                    adjustment_factor *= 0.9
                num_adjustments += 1

            if best_count < old_best_count:
                print(f"Adjusted probabilities to reduce count from {old_best_count:.2f} to {best_count:.2f}", flush=True)
                print("Old Ratio: ", old_best_ratio, "New Ratio: ", best_ratio, flush=True)
                # Save new probs file
                best_probs_file = ensemble[best_ind][3]
                np.save(best_probs_file, final_probs)
                # Change csv data
                csv_data[best_ind]["Count"] = best_count
                csv_data[best_ind]["Ratio"] = best_ratio

        if "checkpoint_dir" in data:
            # Write CSV data
            if os.path.exists(csv_file):
                print("CSV file already exists, appending data", flush=True)
                writer = csv.DictWriter(open(csv_file, "a", newline=""), 
                                        fieldnames=csv_data[0].keys())
                for row in csv_data:
                    writer.writerow(row)
            else:
                print("Writing CSV file: ", csv_file, flush=True)
                writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                        fieldnames=csv_data[0].keys())
                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)

            # Copy best jiggled ensemble file to new file name
            if best_ind >= 0:
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