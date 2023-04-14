from bqskit.compiler.basepass import BasePass
from bqskit.ir.opt.cost.functions.cost.hilbertschmidt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.runtime import get_runtime


class InstPass(BasePass):
    
    def __init__(
        self,
        instantiator_args,
        target
    ):
         
        self.target = target
        instantiator_args = instantiator_args.copy()
        self.instantiator_args = instantiator_args

    

    async def run(self, circuit, data) -> None:
        
        # _logger.debug('Converting single-qubit general gates to U3Gates.')

        cost_fn = HilbertSchmidtCostGenerator().gen_cost(circuit, self.target)
        instantiator_args = self.instantiator_args.copy()
        num_starts = instantiator_args.pop('multistarts', 1)
        method = instantiator_args.pop('method', 'minimization')



        circ_list = await get_runtime().map(
            circuit.instantiate,
            [self.target] * num_starts,
            [method] * num_starts,
            # kwargs= instantiator_args
            **instantiator_args
        )

        circuit = sorted(circ_list, key=lambda x: cost_fn(x.params))[0]
        return circuit

