"""This module implements the HilbertSchmidtCost and
HilbertSchmidtCostGenerator."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskitrs import HilbertSchmidtCostFunction

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction, DifferentiableResidualsFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitary import RealVector
import numpy.typing as npt
import numpy as np

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.opt.cost.function import CostFunction


class FrobeniusCost(
    DifferentiableCostFunction,
):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.
    """
    def __init__(self, circuit: Circuit, target: npt.NDArray[np.complex128]) -> None:
        self.circuit = circuit
        self.target = target
        super().__init__()


    def get_cost(self, params:RealVector) -> np.float64:
        # Get the cost
        utry = self.circuit.get_unitary(params)
        diff = self.target - utry
        cost = np.real(np.trace(diff @ diff.conj().T))
        # print("Params:", params)
        # print("Cost:", cost)
        # print(cost)
        return cost

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the cost gradient given the input parameters."""
        _, grad = self.circuit.get_unitary_and_grad(params=params)

        updates = []
        traces = []
        # print("Target", self.target)
        for grad_i in grad:
            # grad_i is dU/di
            # print("JI:", grad_i)
            # print("MM:", self.target @ grad_i.conj().T)
            ji = np.trace(self.target @ grad_i.conj().T)
            # print("Diag:", np.diag(self.target @ grad_i.conj().T))
            # print("Trace:", ji)
            traces.append(ji)
            updates.append(-2 * np.real(ji))

        # print("Traces:", traces)
        # print("Updates:", updates)
        return updates

    def get_cost_and_grad(
        self,
        params: RealVector,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Return the cost and gradient given the input parameters."""
        return self.get_cost(params), self.get_grad(params)

class FrobeniusCostGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtCostGenerator class.

    This generator produces configured HilbertSchmidtCost functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return FrobeniusCost(circuit, target)


class HilbertSchmidtCost(
    HilbertSchmidtCostFunction,
    DifferentiableCostFunction,
):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.
    """


class HilbertSchmidtCostGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtCostGenerator class.

    This generator produces configured HilbertSchmidtCost functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return HilbertSchmidtCost(circuit, target)

class FrobeniusNoPhaseCost(
    DifferentiableCostFunction,
):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.
    """
    def __init__(self, circuit: Circuit, target: npt.NDArray[np.complex128]) -> None:
        self.circuit = circuit
        self.target = target
        super().__init__()


    def get_cost(self, params:RealVector) -> np.float64:
        # Get the cost
        utry = self.circuit.get_unitary(params)
        diff = self.target - utry
        cost = np.abs(np.trace(diff @ diff.conj().T))
        return cost

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the cost gradient given the input parameters."""
        _, grad = self.circuit.get_unitary_and_grad(params=params)

        updates = []
        traces = []
        # print("Target", self.target)
        for grad_i in grad:
            # grad_i is dU/di
            # print("JI:", grad_i)
            # print("MM:", self.target @ grad_i.conj().T)
            ji = np.trace(self.target @ grad_i.conj().T)
            # print("Diag:", np.diag(self.target @ grad_i.conj().T))
            # print("Trace:", ji)
            traces.append(ji)
            updates.append(-2 * np.real(ji))

        # print("Traces:", traces)
        # print("Updates:", updates)
        return updates

    def get_cost_and_grad(
        self,
        params: RealVector,
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Return the cost and gradient given the input parameters."""
        return self.get_cost(params), self.get_grad(params)

class FrobeniusNoPhaseCostGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtCostGenerator class.

    This generator produces configured HilbertSchmidtCost functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return FrobeniusNoPhaseCost(circuit, target)


class HSCost(
    DifferentiableCostFunction,
    DifferentiableResidualsFunction,
):
    """
    The HilbertSchmidtCost CostFunction implementation.

    The Hilbert-Schmidt CostFuction is a differentiable map from circuit
    parameters to a cost value that is based on the Hilbert-Schmidt inner
    product. This function is global-phase-aware, meaning that the cost is zero
    if the target and circuit unitary differ only by a global phase.
    """
    def __init__(self, circuit: Circuit, target: npt.NDArray[np.complex128]) -> None:
        self.circuit = circuit
        self.target = target
        super().__init__()


    def get_cost(self, params:RealVector) -> np.float64:
        # Get the cost
        a = self.circuit.get_unitary(params)
        b = self.target
        prod = np.einsum("ij,ij->", a, b.conj())
        norm = np.linalg.norm(prod)
        return 1 - (norm / a.shape[0])

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the cost gradient given the input parameters."""
        _, grad = self.circuit.get_unitary_and_grad(params=params)

        updates = []
        traces = []
        # print("Target", self.target)
        for grad_i in grad:
            # grad_i is dU/di
            # print("JI:", grad_i)
            # print("MM:", self.target @ grad_i.conj().T)
            ji = np.trace(self.target @ grad_i.conj().T)
            # print("Diag:", np.diag(self.target @ grad_i.conj().T))
            # print("Trace:", ji)
            traces.append(ji)
            updates.append(-2 * np.real(ji))

        return updates
    
    def get_residuals(self, params: RealVector) -> npt.NDArray[np.float64]:
        """Return the vector of residuals given the input parameters."""
        prod = self.circuit.get_unitary(params) @ self.target.conj().T

        re = np.real(prod) - np.eye(prod.shape[0])
        img = np.imag(prod)

        return np.concatenate((re.flatten(), img.flatten()))


class HSCostGenerator(CostFunctionGenerator):
    """
    The HilbertSchmidtCostGenerator class.

    This generator produces configured HilbertSchmidtCost functions.
    """

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return HSCost(circuit, target)