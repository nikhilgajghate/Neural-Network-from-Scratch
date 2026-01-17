import numpy as np
from numpy.typing import NDArray


class ReLU:
    def __init__(self) -> None:
        self.name: str = "ReLU"
        self.params: list[NDArray[np.float64]] = []
        self.gradInput: NDArray[np.float64] | None = None
        self.output: NDArray[np.float64] | None = None

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Forward pass of the ReLU activation function.
        """
        # ReLU activation: max(0, x)
        # Element-wise maximum between input and 0
        self.output = np.maximum(X, 0)
        return self.output

    def backward(
        self, nextgrad: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        self.gradInput = nextgrad.copy()

        # Zero out gradients where input was <= 0
        # This implements the ReLU gradient: 1 if x > 0, 0 if x <= 0
        if self.output is not None:
            self.gradInput[self.output <= 0] = 0
        return self.gradInput, []
