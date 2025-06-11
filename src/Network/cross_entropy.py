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
        # Layer name for identification
        self.name: str = "CrossEntropy"
        # CrossEntropy has no learnable parameters
        self.params: list[NDArray[np.float64]] = []

    def forward(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        # Store the number of samples for later use
        self.m: int = y.shape[0]

        # Apply softmax to get probability distribution
        self.p: NDArray[np.float64] = softmax(X)

        # Calculate cross entropy loss: -log(p[true_class])
        # range(self.m) creates indices for each sample
        # y contains the true class indices
        cross_entropy: NDArray[np.float64] = -np.log(self.p[range(self.m), y])

        # Average the loss across all samples
        loss: float = float(cross_entropy[0] / self.m)
        return loss

    def backward(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        # Initialize gradient with softmax probabilities
        grad: NDArray[np.float64] = softmax(X)

        # Subtract 1 from the true class probabilities
        # This is the gradient of cross entropy with respect to softmax input
        grad[range(self.m), y] -= 1.0

        # Normalize gradient by number of samples
        grad /= self.m
        return grad
