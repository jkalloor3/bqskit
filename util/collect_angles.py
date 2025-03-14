import numpy as np
import pickle
from typing import Any

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from math import ceil
from bqskit.ir.gates import RZGate
from util import GridSynthGate
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime
import os
import shutil
from .common import load_jiggled_ensemble

async def collect_angles(circ_params: tuple[Circuit, np.ndarray]):
    circ, params = circ_params
    default_thresholds = [1e-1 / circ.num_params, 1e-3 / circ.num_params, 1e-5 / circ.num_params]
    int_min_thresh = [ceil(-1 * np.log10(t)) for t in default_thresholds] 
    all_angles = []
    for param in params:
        circ.set_params(param)
        for op in circ.operations():
            if isinstance(op.gate, RZGate):
                # Add a bunch of precisions 
                all_angles.extend([[op.params[0], t] for t in int_min_thresh])
            elif isinstance(op.gate, GridSynthGate):
                all_angles.append([op.params[0], op.params[1]])
    if len(all_angles) == 0:
        return None
    return np.array(all_angles)



class CollectAnglesPass(BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Check Ensemble Quality and output it to a CSV
        print("Collect Angles Pass", flush=True)
        checkpoint_dir: str = data["checkpoint_dir"]
        
        # Otherwise, reload from saved files - Would have done in 
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_.npy")
        start_ens_ind = 0
        ens_file = ensemble_file_name.format(ind=start_ens_ind)
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind)

        output_file = os.path.join(checkpoint_dir, "angles.npy")
        total_angles = []

        if os.path.exists(output_file):
            print(f"Angles file already exists: {output_file}", flush=True)
            # Load the angles from the file
            angles = np.load(output_file)
            print(f"Loaded angles: {angles.shape}", flush=True)
            # Now generate a cache for the angles -> circs
            cache_file = os.path.join(checkpoint_dir, "angles_cache.pkl")
            if os.path.exists(cache_file):
                print(f"Angles cache already exists: {cache_file}", flush=True)
                return
            cache = {}
            angles = [(a[0], a[1]) for a in angles]
            angles = set(angles)
            print("Num Unique Angles: ", len(angles), flush=True)
            for ind in angles:
                cache[tuple(ind)] = GridSynthGate().get_circuit([ind[0], ind[1], 0])
            # Save the cache to a file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            print(f"Saved angles cache: {cache_file}", flush=True)
            del cache
            return

        if os.path.exists(jiggle_file):
            # Load the ensemble from the checkpoint
            while os.path.exists(jiggle_file):
                circ_params = load_jiggled_ensemble(ens_file, jiggle_file)
                all_angles = await get_runtime().map(collect_angles, circ_params)
                all_angles = [p for p in all_angles if p is not None]
                if len(all_angles) > 0:
                    all_angles = np.vstack(all_angles)
                    total_angles.append(all_angles)
                # print([p.shape for p in all_angles], flush=True)
                start_ens_ind += 1
                ens_file = ensemble_file_name.format(ind=start_ens_ind)
                jiggle_file = jiggle_file_name.format(ind=start_ens_ind)


        total_angles = np.vstack(total_angles)
        print(f"Total angles: {total_angles.shape}", flush=True)
        # Save the angles to a file
        np.save(output_file, total_angles)
