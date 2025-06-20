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
        checkpoint_data_file: str = data["checkpoint_data_file"]
        csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}{self.checkpoint_extra_str}.csv")
        final_ens_file = f"{checkpoint_dir}/ensemble_final{self.checkpoint_extra_str}.qasms"
        final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle{self.checkpoint_extra_str}.npy"

        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting Check Ensemble Quality Pass", flush=True)
        
        if os.path.exists(csv_file):
            # Load the ensemble from the checkpoint
            print("Already Checked!", flush=True)
            return
        
        # Otherwise, reload from saved files - Would have done in 
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
        cache_file_name_2 = os.path.join(checkpoint_dir, "ensemble_cache_{ind}.pkl")
        final_ens_file = os.path.join(checkpoint_dir, "ensemble_final.qasms")
        start_ens_ind = 0
        ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        if not os.path.exists(cache_file):
            cache_file = cache_file_name_2.format(ind=start_ens_ind)
        ensemble_counts = {}
        probs_file = os.path.join(checkpoint_dir,"ensemble_final_probs.npy")

        target = data.target
        csv_dict = {}
        
        # Create new shared memory
        best_ind = 0
        best_ratio = float("inf")
        best_count = float("inf")

        ensemble_files = []
        if os.path.exists(jiggle_file):
            while os.path.exists(jiggle_file):
                ensemble_files.append((ens_file, jiggle_file, cache_file, probs_file))
                start_ens_ind += 1
                ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
                jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
                cache_file = cache_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)

        ensemble = data.get("ensemble", ensemble_files)

        start_ens_ind = 0

        if len(ensemble) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return

        for ens in ensemble:
            if len(ens) > 0 and isinstance(ens[0], str):
                ens_file, jiggle_file, cache_file, probs_file = ens
                circ_params = load_jiggled_ensemble(ens_file, jiggle_file, cache_file, probs_file)
            else:
                circ_params = ens
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
                        if param_chunks[i].shape[0] > 5000:
                            # pick a random subset of 5000
                            rand_inds = np.random.choice(param_chunks[i].shape[0], 5000, replace=False)
                            param_chunks[i] = param_chunks[i][rand_inds]
                            probs_chunks[i] = probs_chunks[i][rand_inds]
                        new_circ_params.append((circ, param_chunks[i], probs_chunks[i], cache))

                circ_params = new_circ_params
            avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                        circ_params,
                                                        target=data.target, 
                                                        add_cost=True)
            
            utries = [avg_utry for avg_utry, _, _ in avg_utries_dists]
            dists = [dist for _, dist, _ in avg_utries_dists]
            hs_dists = [hs for _, _, hs in avg_utries_dists]
            avg_utry_dists = [normalized_frob_cost(avg_utry, target) for avg_utry in utries]
            print("Avg Utry Dists: ", avg_utry_dists, flush=True)
            avg_utry = np.mean(utries, axis=0)
            avg_dist = np.mean(dists)
            avg_hs = np.mean(hs_dists)
            print("Avg Dist: ", avg_dist, flush=True)
            csv_dict[start_ens_ind] = self.get_ensemble_data(avg_utry, avg_dist, target, None, avg_hs=avg_hs)
            params = [params for _, params, _, _ in circ_params]
            csv_dict[start_ens_ind]["Num Circs"] = len(circ_params) * params[0].shape[0]
            csv_dict[start_ens_ind]["Ensemble Generation Method"] = self.ensemble_names[start_ens_ind]
            ratio = csv_dict[start_ens_ind]["Ratio"]
            print("Ratio: ", ratio, flush=True)
            count = np.mean([count_params(c) for c, _, _,_ in circ_params])
            csv_dict[start_ens_ind]["Avg. Count"] = count
            ensemble_counts[start_ens_ind] = count
            print("Avg Count Post Jiggle Load: ", count, flush=True)
            if ratio < 1:
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
                    print("FOUND BETTER ENSEMBLE", flush=True)
            # Keep Looking
            start_ens_ind += 1
            ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
            jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
            cache_file = cache_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
            if not os.path.exists(cache_file):
                cache_file = cache_file_name_2.format(ind=start_ens_ind)
        
        if len(csv_dict) == 0:
            print("No ensembles found, skipping pass", flush=True)
            return

        if "checkpoint_dir" in data:
            writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                    fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict.values():
                writer.writerow(row)
            # Copy best jiggled ensemble file to new file name
            best_ensemble_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_{self.checkpoint_extra_str}.qasms"
            best_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_jiggles_{self.checkpoint_extra_str}.npy"
            if os.path.exists(best_ensemble_file_name):
                shutil.copyfile(best_ensemble_file_name, final_ens_file)
                shutil.copyfile(best_file_name, final_ens_jiggle_file)


class AddHSCostPass(BasePass):

    def __init__(self,
                 csv_name: str = "",
                 checkpoint_extra_str: str = "",
                 ) -> None:
        self.csv_name = csv_name
        self.checkpoint_extra_str = checkpoint_extra_str

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Check Ensemble Quality and output it to a CSV
        print("Check Ensemble Quality Pass", flush=True)
        checkpoint_dir: str = data["checkpoint_dir"]

        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting Add HS Cost Pass", flush=True)
        
        # Otherwise, reload from saved files - Would have done in 
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache.pkl")
        start_ens_ind = 0
        ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)

        target = data.target
        csv_dict = {}

        ensemble_files = []
        if os.path.exists(jiggle_file):
            while os.path.exists(jiggle_file):
                ensemble_files.append((ens_file, jiggle_file, cache_file))
                start_ens_ind += 1
                ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
                jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
                cache_file = cache_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)

        ensemble = data.get("ensemble", ensemble_files)

        print("Ensemble: ", ensemble, flush=True)

        start_ens_ind = 0

        # Read in data.csv
        checkpoint_data_file: str = data["checkpoint_data_file"]
        csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}.csv")
        f = open(csv_file, "r")
        reader = csv.DictReader(f)
        csv_dict = [row for row in reader]
        print("CSV Dict: ", csv_dict, flush=True)
        out_csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}_hs.csv")
        g = open(out_csv_file, "w", newline="")
        field_names = csv_dict[0].keys()
        field_names = list(field_names)
        field_names.append("HS of Mean")
        field_names.append("Avg. HS")
        writer = csv.DictWriter(g, fieldnames=field_names)
        writer.writeheader()

        for ens_ind, ens in enumerate(ensemble):
            if len(ens) > 0 and isinstance(ens[0], str):
                ens_file, jiggle_file, cache_file = ens
                circ_params = load_jiggled_ensemble(ens_file, jiggle_file, cache_file)
            else:
                circ_params = ens
            circuits = [circ for circ, _, _ in circ_params]
            if len(circuits) < 4:
                # Make 10 copies of each circuit and split the params amongst them
                new_circ_params = []
                for circ, params, cache in circ_params:
                    param_chunks = np.array_split(params, 10)
                    for i in range(10):
                        if param_chunks[i].shape[0] > 5000:
                            # pick a random subset of 5000
                            rand_inds = np.random.choice(param_chunks[i].shape[0], 5000, replace=False)
                            param_chunks[i] = param_chunks[i][rand_inds]
                        new_circ_params.append((circ, param_chunks[i], cache))

                circ_params = new_circ_params
            avg_utries_dists = await get_runtime().map(create_avg_utry, 
                                                        circ_params, 
                                                        target=data.target, 
                                                        add_cost=True)
            
            utries = [avg_utry for avg_utry, _, _ in avg_utries_dists]
            hs_dists = [hs for _, _, hs in avg_utries_dists]
            avg_utry = np.mean(utries, axis=0)
            avg_hs = np.mean(hs_dists)
            avg_utry = np.mean(utries, axis=0)
            mean_hs = hs_cost(avg_utry, target)
            row = csv_dict[ens_ind]
            row["HS of Mean"] = mean_hs
            row["Avg. HS"] = avg_hs
            writer.writerow(row)

        f.close()
        g.close()
        print("Finished writing HS data to CSV", flush=True)