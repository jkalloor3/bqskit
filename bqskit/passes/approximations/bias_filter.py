import numpy as np
import pickle
import os

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator

class BiasFilterPass(BasePass):
    def __init__(self, 
                cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
                 max_scaling_factor: float = 10,
                 ) -> None:
        """
        Filter ensemble circuits based on their bias.

        Args:
            cost (CostFunctionGenerator): The cost function used to evaluate
            the bias (should be a Frobenius norm).

            max_scaling_factor (float): The maximum scaling factor for the bias
            to be considered reduced.
        """
        self.cost = cost
        self.max_scaling_factor = max_scaling_factor

    async def create_avg_utry(
            self,
            circ_data: tuple[Circuit, np.ndarray, np.ndarray],
            target: UnitaryMatrix
    ) -> tuple[np.ndarray, float]:
        ''' Return average unitary and distance'''
        circuit, params, probabilities = circ_data
        avg_utry = np.zeros_like(circuit.get_unitary())
        avg_dist = 0
        for i, param in enumerate(params.tolist()):
            circuit.set_params(param)
            un = circuit.get_unitary()
            dist = self.cost.calc_cost(circuit, target=target)
            p = probabilities[i]
            avg_utry += p * un
            avg_dist += p * dist
        return avg_utry, avg_dist

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # This pass should only be called if we are in ensemble mode
        assert "run_ensemble" in data and data["run_ensemble"] == True
        leap_file = str(data["ensemble_name"]) + "_leap_" + str(data["index"]) + ".pkl"
        probs_file = str(data["ensemble_name"]) + "_probs_" + str(data["index"]) + ".pkl"
        param_file = str(data["ensemble_name"]) + "_params_" + str(data["index"]) + ".pkl"
        bias_file = str(data["ensemble_name"]) + "_bias_" + str(data["index"]) + ".pkl"

        if os.path.exists(bias_file):
            scaling_factors = pickle.load(open(bias_file, "rb"))
            # print("Scaling Factors: ", scaling_factors, flush=True)

            scaling_factor = min(scaling_factors)

            if scaling_factor > self.max_scaling_factor:
                # Default to original circuit, delete ensemble data
                data["use_ensemble"] = False
            else:
                data["use_ensemble"] = True
            return


        # Get ensemble circuits, params, and probabilities
        circuits = pickle.load(open(leap_file, "rb"))
        params = pickle.load(open(param_file, "rb"))
        all_probabilities = pickle.load(open(probs_file, "rb"))

        # Zip together each circuit with corresponding parameters 
        # and probabilities
        ensembles = [list(zip(circuits, params, p)) for p in all_probabilities]

        scaling_factors = []
        for ensemble in ensembles:
            # Now calculate the average unitaries and distances 
            avg_utries_dists = await get_runtime().map(self.create_avg_utry, 
                                                    ensemble,
                                                    target=data.target)

            # Calculate average unitaries and distances
            avg_utry = np.sum([utry for utry, _ in avg_utries_dists], axis=0)
            avg_dist = np.sum([dist for _, dist in avg_utries_dists])

            # Calculate bias
            bias = data.target.get_frobenius_distance_from(avg_utry)
            # Scaling Factor is bias / (epsilon ^ 2)
            scaling_factor = bias / (avg_dist * avg_dist)
            # If the Scaling Factor is less than max_scaling factor, then we
            # have reduced the bias, otherwise default to original circuit
            # with single probability

            scaling_factors.append(scaling_factor)

        scaling_factor = min(scaling_factors)
        if scaling_factor > self.max_scaling_factor:
            # Default to original circuit, delete ensemble data
            data["use_ensemble"] = False
        else:
            data["use_ensemble"] = True

        pickle.dump(scaling_factors, open(bias_file, "wb"))