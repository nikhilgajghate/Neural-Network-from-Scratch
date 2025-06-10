import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.dataset.MNIST.mnist import MNIST
from src.Network.linear import Linear
from src.Network.network import Network
from src.Network.relu import ReLU

np.random.seed(3)
SIXTEEN = 16
THIRTY_TWO = 32
SIXTY_FOUR = 64
ONE_HUNDRED_TWENTY_EIGHT = 128
TWO_HUNDRED_FIFTY_SIX = 256
NUM_UNIQUE_CLASSES = 10


def update_params(
    velocity: list[list[NDArray[np.float64]]],
    params: list[list[NDArray[np.float64]]],
    grads: list[list[NDArray[np.float64]]],
    learning_rate: float = 0.01,
    mu: float = 0.9,
) -> None:
    for v, p, g in zip(velocity, params, reversed(grads), strict=False):
        for i in range(len(g)):
            v[i] = mu * v[i] + learning_rate * g[i]
            p[i] -= v[i]


def minibatch(
    X: NDArray[np.float64], y: NDArray[np.int64], batch_size: int
) -> list[tuple[NDArray[np.float64], NDArray[np.int64]]]:
    n = X.shape[0]
    minibatches = []
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    for i in range(0, n, batch_size):
        X_batch = X[i : i + batch_size, :]
        y_batch = y[i : i + batch_size,]
        minibatches.append((X_batch, y_batch))
    return minibatches


def get_accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    return float(np.mean(y_pred == y_true))


def train(
    net: Network,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    minibatch_size: int,
    epoch: int,
    learning_rate: float,
    mu: float = 0.9,
    X_val: NDArray[np.float64] | None = None,
    y_val: NDArray[np.int64] | None = None,
) -> tuple[Network, list[float]]:
    val_loss_epoch: list[float] = []

    minibatches = minibatch(X_train, y_train, minibatch_size)
    print("Number of Minibatches:", len(minibatches))

    if X_val is not None and y_val is not None:
        minibatches_val = minibatch(X_val, y_val, minibatch_size)
        print("Number of Minibatches for Validation:", len(minibatches_val))
    else:
        minibatches_val = []

    for i in range(epoch):
        loss_batch: list[float] = []
        val_loss_batch: list[float] = []
        velocity: list[list[NDArray[np.float64]]] = []

        for param_layer in net.params:
            p = [np.zeros_like(param) for param in param_layer]
            velocity.append(p)

        # iterate over mini batches
        for X_mini, y_mini in minibatches:
            loss, grads = net.train_step(X_mini, y_mini)
            loss_batch.append(loss)
            update_params(
                velocity, net.params, grads, learning_rate=learning_rate, mu=mu
            )

        if X_val is not None and y_val is not None:
            for X_mini_val, y_mini_val in minibatches_val:
                val_loss, _ = net.train_step(X_mini_val, y_mini_val)
                val_loss_batch.append(val_loss)

        # accuracy of model at end of epoch after all mini batch updates
        m_train = X_train.shape[0]
        m_val = X_val.shape[0] if X_val is not None else 0
        y_train_pred = np.array([], dtype=np.int64)
        y_val_pred = np.array([], dtype=np.int64)
        y_train1 = np.array([], dtype=np.int64)
        y_vall = np.array([], dtype=np.int64)

        for i in range(0, m_train, minibatch_size):
            X_tr = X_train[i : i + minibatch_size, :]
            y_tr = y_train[i : i + minibatch_size]
            y_train1 = np.append(y_train1, y_tr)
            y_train_pred = np.append(y_train_pred, net.predict(X_tr))

        if X_val is not None and y_val is not None:
            for i in range(0, m_val, minibatch_size):
                X_va = X_val[i : i + minibatch_size, :]
                y_va = y_val[i : i + minibatch_size]
                y_vall = np.append(y_vall, y_va)
                y_val_pred = np.append(y_val_pred, net.predict(X_va))

        train_acc = get_accuracy(y_train1, y_train_pred)
        val_acc = get_accuracy(y_vall, y_val_pred) if X_val is not None else 0.0

        mean_train_loss = sum(loss_batch) / float(len(loss_batch))
        mean_val_loss = (
            sum(val_loss_batch) / float(len(val_loss_batch)) if val_loss_batch else 0.0
        )

        val_loss_epoch.append(mean_val_loss)
        print(
            "Loss = {0} | Training Accuracy = {1} | "
            "Val Loss = {2} | Val Accuracy = {3}".format(
                mean_train_loss, train_acc, mean_val_loss, val_acc
            )
        )
    return net, val_loss_epoch


def load_dataset(
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset and return the features and labels.
    TODO: Add more datasets like CIFAR10, CIFAR100, FashionMNIST, etc.
    Args:
        dataset_name: str
            The name of the dataset to load.
    Returns:
        X_train: np.ndarray
            The training features.
        y_train: np.ndarray
            The training labels.
        X_test: np.ndarray
            The test features.
        y_test: np.ndarray
            The test labels.
    """
    if dataset_name.lower() == "mnist":
        mnist = MNIST()
        X_train, y_train, X_test, y_test = mnist.load_data()

        # Flatten and normalize the features
        X_train, X_test = mnist.flatten_data(X_train, X_test)
        X_train, X_test = mnist.normalize_data(X_train, X_test)

        return X_train, y_train, X_test, y_test
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def visualize_dataset(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Visualize the dataset.
    """
    # Visualize the first 10 images in the dataset
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            # Reshape the image to 28x28 before plotting it.
            # The reason is that the image is a 1D array of 784 elements.
            axs[i, j].imshow(X_train[i * 5 + j].reshape(28, 28))
            axs[i, j].set_title(f"Label: {y_train[i * 5 + j]}")
            axs[i, j].axis("off")
    plt.show()


def create_network(input_size: int, output_size: int) -> Network:
    """
    Create a network with the given input and output sizes.
    """
    net = Network()
    net.add_layer(Linear(input_size, THIRTY_TWO))
    net.add_layer(ReLU())
    net.add_layer(Linear(THIRTY_TWO, output_size))
    return net


def main() -> None:

    BATCH_SIZE = 200
    EPOCH = 10
    LEARNING_RATE = 0.01

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--visualize", type=bool, default=False)

    args = parser.parse_args()

    # Print the arguments
    print("=============================")
    print("Arguments passed in:")
    print(f"Dataset: {args.dataset}")
    print("-----------------------------")
    print(f"Visualize: {args.visualize}")
    print("=============================")

    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Visaulize dataset
    if args.visualize:
        visualize_dataset(X_train, y_train)

    # Create network
    net = create_network(X_train.shape[1], NUM_UNIQUE_CLASSES)
    net.summary()

    # Train model
    train(
        net,
        X_train,
        y_train,
        BATCH_SIZE,
        EPOCH,
        LEARNING_RATE,
        mu=0.9,
        X_val=X_test,
        y_val=y_test,
    )


if __name__ == "__main__":
    main()
