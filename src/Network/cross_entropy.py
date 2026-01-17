import numpy as np
from numpy.typing import NDArray

from src.Network.softmax import softmax


class CrossEntropy:
    """
    This class is used to calculate the cross entropy loss and its gradient.
    It takes the output of the network and the true labels as input.
    It returns the loss and the gradient.
    """

    def __init__(self) -> None:
        self.name: str = "CrossEntropy"
        self.params: list[NDArray[np.float64]] = []

    def forward(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        self.m: int = y.shape[0]
        self.p: NDArray[np.float64] = softmax(X)
        cross_entropy: NDArray[np.float64] = -np.log(self.p[range(self.m), y])
        loss: float = float(cross_entropy[0] / self.m)
        return loss

    def backward(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        grad: NDArray[np.float64] = softmax(X)
        grad[range(self.m), y] -= 1.0
        grad /= self.m
        return grad
