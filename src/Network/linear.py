import numpy as np
from numpy.typing import NDArray


class Linear:
    def __init__(self, input_size: int, output_size: int) -> None:
        # Layer name for identification
        self.name = "Linear"

        # Initialize weights with small random values
        # Using 0.01 as scaling factor to prevent large initial values
        self.W: NDArray[np.float64] = (
            np.random.randn(input_size, output_size) * 0.01
        ).astype(np.float64)

        # Initialize biases to zero
        self.b: NDArray[np.float64] = np.zeros((1, output_size), dtype=np.float64)

        # Store learnable parameters
        self.params = [self.W, self.b]

        # Initialize gradient storage
        self.gradW: NDArray[np.float64] | None = None  # Gradient for weights
        self.gradB: NDArray[np.float64] | None = None  # Gradient for biases
        self.gradInput: NDArray[np.float64] | None = None  # Gradient for input

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Store input for backward pass
        self.X: NDArray[np.float64] = X.astype(np.float64)

        # Linear transformation: y = Wx + b
        # X shape: (batch_size, input_size)
        # W shape: (input_size, output_size)
        # b shape: (1, output_size)
        dot_product: NDArray[np.float64] = np.dot(self.X, self.W).astype(np.float64)
        self.output: NDArray[np.float64] = (dot_product + self.b).astype(np.float64)
        return self.output

    def backward(
        self, nextgrad: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        # Compute gradient with respect to weights
        # dL/dW = X^T * dL/dy
        self.gradW = np.dot(self.X.T, nextgrad).astype(np.float64)

        # Compute gradient with respect to biases
        # dL/db = sum(dL/dy)
        self.gradB = np.sum(nextgrad, axis=0).astype(np.float64)

        # Compute gradient with respect to input
        # dL/dX = dL/dy * W^T
        self.gradInput = np.dot(nextgrad, self.W.T).astype(np.float64)

        return (self.gradInput, [self.gradW, self.gradB])
