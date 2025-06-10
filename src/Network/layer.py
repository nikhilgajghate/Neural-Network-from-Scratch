from typing import Protocol, TypeVar, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.Network.linear import Linear
from src.Network.relu import ReLU


# Define a protocol for layers that have params
@runtime_checkable
class Layer(Protocol):
    name: str
    params: list[NDArray[np.float64]]

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def backward(
        self, nextgrad: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        pass


# Define a protocol for layers that have W and b
@runtime_checkable
class LayerWithWeights(Layer, Protocol):
    W: NDArray[np.float64]
    b: NDArray[np.float64]
    name: str


# Define a type variable for layers
T = TypeVar("T", bound=Union[Linear, ReLU])

# Define the concrete layer types
ConcreteLayer = Union[Linear, ReLU]
