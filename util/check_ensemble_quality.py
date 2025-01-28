import numpy as np
import csv 
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
import pickle
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.ir import Circuit
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.qis import UnitaryMatrix
from bqskit.runtime import get_runtime
import os
import itertools
import shutil
from util.common import load_jiggled_ensemble, create_jiggled_unitaries

from .distance import frobenius_cost, normalized_frob_cost


norm_cost = GPNormalizedFrobeniusCostGenerator()
frob_cost = GPNormalizedFrobeniusCostGenerator()
class CheckEnsembleQualityPass(BasePass):
    def __init__(self, 
                 count_t: bool = False,
                 csv_name: str = "",
                 checkpoint_extra_str: str = ""
                 ) -> None:
        self.count_t = count_t
        self.csv_name = csv_name
        self.ensemble_names = ["Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        self.gate_title = "Num Params" if count_t else "CNOT Count"
        self.gate_func = lambda x: x.count(TGate()) + x.count(TdgGate()) + x.num_params * 60 if count_t else x.count(CNOTGate())
        self.checkpoint_extra_str = checkpoint_extra_str
    
    def get_ensemble_data(self, ens: list[tuple[UnitaryMatrix, float]], target: UnitaryMatrix, orig_count: int) -> dict[str, Any]:
        ensemble_data = {}
        print("In Get Ensemble Data", flush=True)
        unitaries: list[UnitaryMatrix] = [x[0] for x in ens]
        norm_e1s = [x[1] for x in ens]
        print("Average Norm Epsilon: ", np.mean(norm_e1s), flush=True)
        frob_factor = np.sqrt(unitaries[0].shape[0] * 2)
        print("Frob Factor: ", frob_factor, flush=True)
        frob_e1s = [frob_factor * c for c in norm_e1s]
        # norm_e1_nogps = [normalized_frob_cost(un, target) for un in unitaries[:2000]]
        # norm_e1_2s = [norm_cost.calc_cost(c, target)for c in ens[:2000]]
        # norm_e1_nogp_2s = [frob_cost.calc_cost(c, target)for c in ens[:2000]]
        norm_e1 = np.mean(norm_e1s)
        frob_e1 = np.mean(frob_e1s)
        mean_un = np.mean(unitaries, axis=0)
        norm_bias = normalized_frob_cost(mean_un, target)
        frob_bias = frobenius_cost(mean_un, target)
        
        # final_counts = [self.gate_func(circ) for circ in ens]
        ensemble_data["Ensemble Generation Method"] = ""
        ensemble_data["Num Circs"] = len(ens)
        ensemble_data[f"Orig. {self.gate_title}"] = orig_count
        # ensemble_data[f"Avg. {self.gate_title}"] = np.mean(final_counts)
        ensemble_data["Norm. Epsilon"] = norm_e1
        ensemble_data["Epsilon"] = frob_e1
        ensemble_data["Max Epsilon"] = np.max(frob_e1s)
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
        checkpoint_dir = data["checkpoint_dir"]
        final_ens_file = f"{checkpoint_dir}/ensemble_final.qasms"
        final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle.npy"
        
        if os.path.exists(final_ens_file):
            # Load the ensemble from the checkpoint
            circ_params: list[tuple[Circuit, np.ndarray]] = load_jiggled_ensemble(final_ens_file, 
                                                                                  final_ens_jiggle_file)
            best_ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                                  target=data.target, 
                                                  add_cost = False, 
                                                  fix_phase=True)
            best_ensemble: list[UnitaryMatrix] = list(itertools.chain(*best_ensemble))
            if len(best_ensemble) > 4000:
                rand_inds_file = f"{checkpoint_dir}/ensemble_final_rand_inds.npy"
                if os.path.exists(rand_inds_file):
                    rand_inds = np.load(rand_inds_file)
                else:
                    rand_inds = np.random.choice(len(best_ensemble), 4000, replace=False)
                    np.save(rand_inds_file, rand_inds)
                best_ensemble = [best_ensemble[i] for i in rand_inds]
            data["final_ensemble_unitaries"] = best_ensemble
            print("Check Ensemble Quality Pass", flush=True)
            return
        
        ensemble_unitaries: list[list[tuple[UnitaryMatrix, float]]] = data["ensemble_unitaries"]
        print("Num Ensembles: ", len(ensemble_unitaries), flush=True)
        print("Ensemble Lengths: ", [len(x) for x in ensemble_unitaries], flush=True)

        for i in range(3, len(ensemble_unitaries)):
            self.ensemble_names.append(f"Random Circuits #{i-2}")
        
        target = data.target
        # if len(ensemble_unitaries) == 1:
        #     csv_dict = [await self.get_ensemble_data(ensemble_unitaries[0], target, 
        #                                              self.gate_func(circuit))]
        # else:
        #     csv_dict: list[dict[str, Any]] = await get_runtime().map(
        #         self.get_ensemble_data, ensemble_unitaries, target=target, 
        #         orig_count = self.gate_func(circuit))
        csv_dict = []
        for ens_un in ensemble_unitaries:
            csv_dict.append(self.get_ensemble_data(ens_un, target, None))


        final_ratios = []
        for i in range(len(ensemble_unitaries)):
            csv_dict[i]["Ensemble Generation Method"] = self.ensemble_names[i]
            final_ratios.append(csv_dict[i]["Norm. Ratio"])

        # Ensemble is good if any of the final ratios is less than 10
        data["good_ensemble"] = any([x < 10 for x in final_ratios])

        print("Final Ratios: ", final_ratios, flush=True)

        if data["good_ensemble"]:
            print("FOUND GOOD ENSEMBLE", flush=True)

        # Pick best ensemble
        best_ind = np.argmin(final_ratios)
        # Randomly sample 2000 circuits from the best ensemble
        best_ensemble: list[tuple[UnitaryMatrix, float]] = ensemble_unitaries[best_ind]
        if len(best_ensemble) > 4000:

            rand_inds = np.random.choice(len(best_ensemble), 4000, replace=False)
            best_ensemble = [best_ensemble[i] for i in rand_inds]
            rand_inds_file = f"{checkpoint_dir}/ensemble_final_rand_inds.npy"
            np.save(rand_inds_file, rand_inds)
            # best_ensemble = np.random.choice(best_ensemble, 2000, replace=False)

        best_ensemble_unitaries = [u for u, _ in best_ensemble]
        data["final_ensemble_unitaries"] = best_ensemble_unitaries
        
        if "checkpoint_dir" in data:
            checkpoint_data_file: str = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}.csv")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                    fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict:
                writer.writerow(row)
            # Copy best jiggled ensemble file to new file name
            best_ensemble_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_{self.checkpoint_extra_str}.qasms"
            best_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_jiggles_{self.checkpoint_extra_str}.npy"
            shutil.copyfile(best_ensemble_file_name, final_ens_file)
            shutil.copyfile(best_file_name, final_ens_jiggle_file)

