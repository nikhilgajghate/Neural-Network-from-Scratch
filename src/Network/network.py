from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from src.Network.cross_entropy import CrossEntropy
from src.Network.layer import ConcreteLayer, LayerWithWeights
from src.Network.linear import Linear
from src.Network.relu import ReLU


class Network:
    def __init__(self, lossfunc: Optional[CrossEntropy] = None) -> None:
        self.params: list[list[NDArray[np.float64]]] = []
        self.layers: list[ConcreteLayer] = []
        self.loss_func: CrossEntropy = (
            lossfunc if lossfunc is not None else CrossEntropy()
        )
        self.grads: list[list[NDArray[np.float64]]] = []

    def add_layer(self, layer: ConcreteLayer) -> None:
        self.layers.append(layer)
        if isinstance(layer, (Linear, ReLU)):
            self.params.append(layer.params)

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(
        self, nextgrad: NDArray[np.float64]
    ) -> list[list[NDArray[np.float64]]]:
        self.clear_grad_param()
        for layer in reversed(self.layers):
            nextgrad, grad = layer.backward(nextgrad)
            self.grads.append(grad)
        return self.grads

    def train_step(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> tuple[float, list[list[NDArray[np.float64]]]]:
        out = self.forward(X)
        loss = self.loss_func.forward(out, y)
        nextgrad = self.loss_func.backward(out, y)
        l2 = self.backward(nextgrad)
        return loss, l2

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        X = self.forward(X)
        predictions = np.argmax(X, axis=1)
        return cast(NDArray[np.int64], predictions.astype(np.int64))

    def predict_scores(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = self.forward(X)
        return X

    def clear_grad_param(self) -> None:
        self.grads = []

    def summary(self) -> None:
        print("=============================")
        print("Network Summary:")
        for layer in self.layers:
            print("------")
            if isinstance(layer, LayerWithWeights):
                print(f"{layer.name}")
                print(f"W shape: {layer.W.shape}")
                print(f"b shape: {layer.b.shape}")
            else:
                print(f"{layer.name}")
        print("=============================")
