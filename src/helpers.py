import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Softmax function.
    Args:
        x: NDArray[np.float64]
            The input array.
    Returns:
        NDArray[np.float64]
            The softmax of the input array.
    """
    exp_x: NDArray[np.float64] = np.exp(x - np.max(x, axis=1, keepdims=True))
    out: NDArray[np.float64] = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return out.astype(np.float64)
