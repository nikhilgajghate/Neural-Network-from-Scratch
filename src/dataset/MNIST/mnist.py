from typing import Any

import numpy as np
from keras.datasets import mnist
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class MNIST(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "MNIST"
    X_train: NDArray[np.float64] | None = None
    y_train: NDArray[np.int64] | None = None
    X_test: NDArray[np.float64] | None = None
    y_test: NDArray[np.int64] | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        print(f"Loading {self.name} dataset")

    def load_data(
        self,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]
    ]:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def flatten_data(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        return X_train, X_test

    def normalize_data(
        self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        return X_train, X_test
