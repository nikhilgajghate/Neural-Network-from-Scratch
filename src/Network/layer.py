from typing import Protocol, TypeVar, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.Network.linear import Linear
from src.Network.relu import ReLU


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


@runtime_checkable
class LayerWithWeights(Layer, Protocol):
    W: NDArray[np.float64]
    b: NDArray[np.float64]
    name: str


T = TypeVar("T", bound=Union[Linear, ReLU])
ConcreteLayer = Union[Linear, ReLU]
