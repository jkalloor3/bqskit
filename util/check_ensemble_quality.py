import numpy as np
import csv 
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from multiprocessing import shared_memory
from math import ceil
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.ir import Circuit
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.qis import UnitaryMatrix
from bqskit.runtime import get_runtime
import os
import time
import itertools
import shutil
from util.common import load_jiggled_ensemble_separate, create_jiggled_unitaries_shm, count_params

from .distance import frobenius_cost, normalized_frob_cost

norm_cost = GPNormalizedFrobeniusCostGenerator()
frob_cost = GPNormalizedFrobeniusCostGenerator()

BASE_SHM_NAME = "param_arr"
NUM_FINAL_CIRCS = 4000
NUM_UNIQUE_CIRCS = 250
# MAX_SENDABLE_PARAMS = 150
MAX_PARAMS_PER_CIRC = 80
# Set to 32GB
MAX_SHM_SIZE = 32 * 1024 * 1024 * 1024  # 32GB


async def calculate_unitaries(circuits, params: np.ndarray, target: UnitaryMatrix, shm_name: str, shm_percentage: float = 1.0) -> list[tuple[UnitaryMatrix, float]]:
    # Check if circ_params are too large
    assert len(circuits) == params.shape[0]
    # Params is of shape (num_circuits, diff_params_per_circ, params_in_circ)
    param_size = params.nbytes
    print("Param Size", param_size, flush=True)
    max_shm_size = int(MAX_SHM_SIZE * shm_percentage)
    # Split the params to form (num_chunks, num_circuits,
    # diff_params_per_circ /num_chunks , params_in_circ)
    if param_size > max_shm_size:
        # Split the params along axis 1
        num_chunks = ceil(param_size / max_shm_size)
        print("Num Chunks", num_chunks, flush=True)
        # Split the params into chunks
        chunk_params = np.array_split(params, num_chunks, axis=1)
    else:
        chunk_params = [params]

    # print("Chunk Shapes", [x.shape for x in chunk_params], flush=True)

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    print(f"Sending {len(chunk_params)} times", flush=True)
    jiggled_unitaries: list[tuple[UnitaryMatrix, float]] = []
    for circ_param_chunk in chunk_params:
        # print("Chunk Size", shm_size, circ_param_chunk.shape, flush=True)
        shared_array = np.ndarray(circ_param_chunk.shape, dtype=np.float64, buffer=existing_shm.buf)
        shared_array[:] = circ_param_chunk[:]
        print("Wrote to Shared Memory", flush=True)
        param_inds = np.arange(len(circuits))
        circ_inds = list(zip(circuits, param_inds))
        jiggled_unitaries_chunk: list[list[tuple[UnitaryMatrix, float]]] = await get_runtime().map(
            create_jiggled_unitaries_shm , circ_inds, shm_name=shm_name, 
            shm_shape=circ_param_chunk.shape, target=target)
        print("Jiggled Unitaries Chunk", len(jiggled_unitaries_chunk), flush=True)
        jiggled_unitaries_chunk = list(itertools.chain.from_iterable(jiggled_unitaries_chunk))
        jiggled_unitaries.extend(jiggled_unitaries_chunk)

    existing_shm.close()
    return jiggled_unitaries


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

    def get_ensemble_data(self, ens: list[tuple[UnitaryMatrix, float]], target: UnitaryMatrix, orig_count: int) -> dict[str, Any]:
        ensemble_data = {}
        unitaries: list[UnitaryMatrix] = [x[0] for x in ens]
        norm_e1s = [x[1] for x in ens]
        print("Average Norm Epsilon: ", np.mean(norm_e1s), flush=True)
        frob_factor = np.sqrt(unitaries[0].shape[0] * 2)
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
        start_ens_ind = 1
        ens_file = ensemble_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)
        ensemble_unitaries = {}
        ensemble_counts = {}

        target = data.target
        csv_dict = {}
        # Create Shared Memory

        shm_name = checkpoint_dir.split("/")[-1] + "_" + BASE_SHM_NAME
        print("Shared Memory Name: ", shm_name, flush=True)
        shm = shared_memory.SharedMemory(create=True, size=MAX_SHM_SIZE * self.shm_percentage, name=shm_name)

        best_ind = 0
        best_ratio = float("inf")
        best_count = float("inf")

        if os.path.exists(jiggle_file):
            # Load the ensemble from the checkpoint
            while os.path.exists(jiggle_file):
                circuits, params = load_jiggled_ensemble_separate(ens_file, jiggle_file)
                start = time.time()
                jiggled_unitaries = await calculate_unitaries(circuits, 
                                                              params, 
                                                              target=data.target,
                                                              shm_name=shm_name,
                                                              shm_percentage=self.shm_percentage)
                end = time.time()
                print("Time to calculate unitaries: ", end - start, flush=True)
                csv_dict[start_ens_ind] = self.get_ensemble_data(jiggled_unitaries, target, None)
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

        shm.close()
        shm.unlink()

        # Randomly sample 4000 circuits from the best ensemble
        num_unitaries = len(ensemble_unitaries[best_ind])
        if num_unitaries > NUM_FINAL_CIRCS:
            rand_inds = np.random.choice(num_unitaries, 
                                         NUM_FINAL_CIRCS, 
                                         replace=False)
            rand_inds_file = f"{checkpoint_dir}/ensemble_final_rand_inds.npy"
            np.save(rand_inds_file, rand_inds)
        
        if "checkpoint_dir" in data:
            checkpoint_data_file: str = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", f"{self.csv_name}.csv")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), 
                                    fieldnames=csv_dict[1].keys())
            writer.writeheader()
            for row in csv_dict.values():
                writer.writerow(row)
            # Copy best jiggled ensemble file to new file name
            best_ensemble_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_{self.checkpoint_extra_str}.qasms"
            best_file_name = f"{checkpoint_dir}/ensemble_{best_ind}_jiggles_{self.checkpoint_extra_str}.npy"
            shutil.copyfile(best_ensemble_file_name, final_ens_file)
            shutil.copyfile(best_file_name, final_ens_jiggle_file)

