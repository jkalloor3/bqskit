import abc
import numpy as np
import pickle
from bqskit.ir.circuit import Circuit, CircuitPoint 
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes.control.foreach import ForEachBlockPass


class CircuitSampler(abc.ABC):
    @abc.abstractmethod
    def random_sample(self) -> Circuit:
        '''
        Randomly Sample a Circuit according to the underlying probability 
        distribution
        '''
        pass

    @abc.abstractmethod
    def next_circuit(self) -> tuple[Circuit, float]:
        '''
        Get the next circuit and its probability from the sampler

        TODO: Order circuits by probability
        '''
        pass

    def current_circuit(self) -> tuple[Circuit, float]:
        '''
        Get the current circuit and its probability from the sampler
        '''
        pass

class BlockCircuitSampler(CircuitSampler):

    def __init__(
            self,
            circuits: list[Circuit],
            params: list[np.ndarray],
            probs: list[np.ndarray],
    ) -> None:
        self.circuits = circuits
        self.params = params
        self.all_probs = [probs.flatten() for probs in probs]
        self.circ_probs = np.array([np.sum(p) for p in self.all_probs])
        self.circ_probs /= np.sum(self.circ_probs)
        self.circ_ind = 0
        self.param_inds = [0] * len(self.circuits)

    def random_sample(self) -> Circuit:
        rand_circ_idx = np.random.choice(len(self.circ_probs),
                                        p=self.circ_probs)
        circ = self.circuits[rand_circ_idx]
        param_options = self.params[rand_circ_idx]
        param_probs = self.all_probs[rand_circ_idx]
        # print(param_options.shape, param_probs.shape, np.sum(param_probs))
        norm_probs = param_probs / (np.sum(param_probs) + 1e-10)
        rand_param_idx = np.random.choice(len(param_options), p=norm_probs)
        rand_param = param_options[rand_param_idx]
        out_circ = circ.copy()
        out_circ.set_params(rand_param)
        return out_circ
    
    def next_circuit(self) -> tuple[Circuit, float]:
        circ = self.circuits[self.circ_ind]
        param_ind = self.param_inds[self.circ_ind]
        params = self.params[self.circ_ind][param_ind]
        out_circ = circ.copy()
        out_circ.set_params(params)
        prob = self.all_probs[self.circ_ind][param_ind]

        # Increment Param Index and Circuit Index
        self.param_inds[self.circ_ind] += 1
        self.circ_ind += 1
        # Skip circuits with no params left to explore
        param_options = self.params[self.circ_ind]
        while self.param_inds[self.circ_ind] >= len(param_options):
            self.circ_ind += 1
            param_options = self.params[self.circ_ind]
        return out_circ, prob
    
    def current_circuit(self) -> tuple[Circuit, float]:
        circ = self.circuits[self.circ_ind]
        param_ind = self.param_inds[self.circ_ind]
        params = self.params[self.circ_ind][param_ind]
        out_circ = circ.copy()
        out_circ.set_params(params)
        prob = self.all_probs[self.circ_ind][param_ind]
        return out_circ, prob

class FullCircuitSampler:
    def __init__(
        self,
        partitioned_circuit: Circuit,
        block_samplers: dict[CircuitPoint, CircuitSampler],
    ) -> None:
        self.partitioned_circuit = partitioned_circuit
        self.block_samplers = block_samplers
        self.sampler_ind = 0

    def random_sample(self) -> Circuit:
        out_circ = self.partitioned_circuit.copy()
        for cycle, op in out_circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            block_circ = self.block_samplers[pt].random_sample()
            out_circ.replace_with_circuit(pt, block_circ, as_circuit_gate=True)
        out_circ.unfold_all()
        return out_circ
    
    def next_circuit(self) -> tuple[Circuit, float]:
        out_circ = self.partitioned_circuit.copy()
        total_prob = 1
        block_ind = 0
        for cycle, op in out_circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            if block_ind == self.sampler_ind:
                # Get next circuit for this sampler
                block_circ, prob = self.block_samplers[pt].next_circuit()
            else:
                block_circ, prob = self.block_samplers[pt].current_circuit()
            out_circ.replace_with_circuit(pt, block_circ, as_circuit_gate=True)
            total_prob *= prob
        out_circ.unfold_all()
        # Update sampler index
        self.sampler_ind += 1
        self.sampler_ind %= len(self.block_samplers)
        return out_circ, total_prob

    def current_circuit(self) -> tuple[Circuit, float]:
        out_circ = self.partitioned_circuit.copy()
        for cycle, op in out_circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            if pt in self.block_samplers:
                block_circ, prob = self.block_samplers[pt].current_circuit()
            out_circ.replace_with_circuit(pt, block_circ, as_circuit_gate=True)
        out_circ.unfold_all()
        return out_circ, prob

class GenerateCircuitSamplerPass(BasePass):

    def __init__(
            self,
            combine_sub_blocks: bool = True,
    ) -> None:
        self.combine_sub_blocks = combine_sub_blocks


    async def run(
            self,
            circuit: Circuit,
            data: PassData
    ) -> None:
        # This pass should only be called if we are in ensemble mode
        assert "run_ensemble" in data and data["run_ensemble"] == True

        # If we are inside a single block, create a BlockCircuitSampler
        if not self.combine_sub_blocks:
            if data["use_ensemble"]:
                circuits = data["ensemble_circuits"]
                params = data["ensemble_params"]
                probs_ind = data["probs_ind"]
                # probs = data["ensemble_probabilities"][probs_ind]
                probs_file = str(data["ensemble_name"]) + "_probs_" + str(data["index"]) + ".pkl"
                probs = pickle.load(open(probs_file, "rb"))
                sampler = BlockCircuitSampler(
                    circuits=circuits,
                    params=params,
                    probs=probs[probs_ind]
                )
            else:
                return
        else:
            # Get the circuit sampler for each block
            all_samplers = {}
            block_data = data[ForEachBlockPass.key][-1]
            i = 0
            for cycle, op in circuit.operations_with_cycles():
                sampler = block_data[i].get("circuit_sampler", None)
                if sampler:
                    all_samplers[CircuitPoint(cycle, op.location[0])] = sampler
                i += 1

            sampler = FullCircuitSampler(
                partitioned_circuit=circuit,
                block_samplers=all_samplers
            )
        # Store Circuit Sampler
        data["circuit_sampler"] = sampler