import numpy as np
import pickle
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime
from util import normalized_frob_cost, normalized_gp_frob_cost


class HamiltonianNoisePass(BasePass):
    def __init__(self, ensemble_size: int, 
                 epsilon: float,
                 kmeans: bool = False,
                 use_calculated_error: bool = False) -> None:
        self.ensemble_size = ensemble_size
        self.epsilon = epsilon
        self.use_calculated_error = use_calculated_error
        self.kmeans = kmeans


    def get_perturbations(num_qudits: int, epsilon: float, ens_size: int) -> list[UnitaryMatrix]:
        perturbations = []
        # Limit to length 2 paulis
        pauli_strings = PauliMatrices.get_pauli_strings(num_qudits, 2)
        # Get rid of all I string
        pauli_strings.remove("I" * num_qudits)
        print("Num Pauli Strings: ", len(pauli_strings), flush=True)
        # TODO: Try subselecting better points
        # paulis = PauliMatrices(num_qudits)
        paulis = [PauliMatrices.from_string(pauli) for 
                  pauli in pauli_strings]
        
        all_coeffs = []
        for _ in range(ens_size // 2):
            all_coeffs.append(np.random.rand(len(paulis)))
            
        for coeff in all_coeffs:
            coeff /= np.linalg.norm(coeff)
            coeff *= epsilon
            H_1 = dot_product(coeff, paulis)
            H_2 = dot_product(-1 * coeff, paulis)
            assert np.allclose(H_1, H_1.conj().T)
            assert np.allclose(H_2, H_2.conj().T)
            eiH_1 = sp.linalg.expm(1j * H_1)
            eiH_2 = sp.linalg.expm(1j * H_2)
            perturbations.append(UnitaryMatrix(eiH_1))
            perturbations.append(UnitaryMatrix(eiH_2))
        
        return perturbations

    async def run(self, circuit: Circuit, data: dict) -> None:
        if "finished_perturbation" in data:
            print("Reloading Perturbations!", flush=True)
            return


        if self.use_calculated_error:
            # use sqrt
            factor = np.sqrt(data["error_percentage_allocated"])
            epsilon = self.epsilon* factor
        else:
            epsilon = self.epsilon


        num_qudits = circuit.num_qudits
        base_un = circuit.get_unitary()
        perturbations = HamiltonianNoisePass.get_perturbations(num_qudits, epsilon=epsilon, 
                                               ens_size=self.ensemble_size)
        bias = np.mean(perturbations, axis=0)
        # get norm of bias
        bias_norm = normalized_frob_cost(bias, np.eye(2 ** num_qudits))
        # pert_dists = [normalized_frob_cost(pert, np.eye(2 ** num_qudits)) for pert in perturbations]
        # pert_dists_gp = [normalized_gp_frob_cost(pert, np.eye(2 ** num_qudits)) for pert in perturbations]
        # print("Perturbation Dists: ", pert_dists)
        # print("Perturbation Dists GP: ", pert_dists_gp)
        print("Bias Norm: ", bias_norm)
        # assert bias_norm < (epsilon ** 2)
        targets = [base_un @ pert for pert in perturbations]
        data["ensemble_targets"] = targets
        data["ensemble_perturbations"] = perturbations

        data["finished_perturbation"] = True
        if "checkpoint_data_file" in data:
            save_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(save_data_file, "wb"))
