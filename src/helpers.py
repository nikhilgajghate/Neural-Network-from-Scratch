import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (  # type: ignore[import-untyped]
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from src.dataset.MNIST.mnist import MNIST
from src.Network.linear import Linear
from src.Network.network import Network
from src.Network.relu import ReLU
from src.Network.training_params import THIRTY_TWO


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


def get_batches(
    X: NDArray[np.float64], y: NDArray[np.int64], batch_size: int
) -> list[tuple[NDArray[np.float64], NDArray[np.int64]]]:
    n = X.shape[0]
    batches = []
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    for i in range(0, n, batch_size):
        X_batch = X[i : i + batch_size, :]
        y_batch = y[i : i + batch_size,]
        batches.append((X_batch, y_batch))
    return batches


def get_accuracy(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    return float(np.mean(y_pred == y_true))


def plot_loss(loss_epoch: list[float]) -> None:
    plt.plot(loss_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.show()


def plot_accuracy(accuracy_epoch: list[float]) -> None:
    plt.plot(accuracy_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.show()


def train(
    net: Network,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    batch_size: int,
    epoch: int,
    learning_rate: float,
    mu: float = 0.9,
) -> Network:
    mean_train_loss_epoch: list[float] = []
    train_acc_epoch: list[float] = []

    batches = get_batches(X_train, y_train, batch_size)
    print("Number of Minibatches:", len(batches))

    for _ in range(epoch):
        loss_batch: list[float] = []
        velocity: list[list[NDArray[np.float64]]] = []

        for param_layer in net.params:
            p = [np.zeros_like(param) for param in param_layer]
            velocity.append(p)

        # iterate over mini batches
        for X_mini, y_mini in batches:
            loss, grads = net.train_step(X_mini, y_mini)
            loss_batch.append(loss)
            update_params(
                velocity, net.params, grads, learning_rate=learning_rate, mu=mu
            )

        # accuracy of model at end of epoch after all mini batch updates
        m_train = X_train.shape[0]
        y_train_pred = np.array([], dtype=np.int64)
        y_train1 = np.array([], dtype=np.int64)

        for i in range(0, m_train, batch_size):
            X_tr = X_train[i : i + batch_size, :]
            y_tr = y_train[i : i + batch_size]
            y_train1 = np.append(y_train1, y_tr)
            y_train_pred = np.append(y_train_pred, net.predict(X_tr))

        train_acc = get_accuracy(y_train1, y_train_pred)
        train_acc_epoch.append(train_acc)

        mean_train_loss = sum(loss_batch) / float(len(loss_batch))
        mean_train_loss_epoch.append(mean_train_loss)

        print("Loss = {0} | Training Accuracy = {1}".format(mean_train_loss, train_acc))

        print("--------------------------------")
        print("Mean train loss:", mean_train_loss)
        print("Training accuracy:", train_acc)
        print("--------------------------------")

    plot_loss(mean_train_loss_epoch)
    plot_accuracy(train_acc_epoch)

    return net


def validate(
    net: Network,
    X_val: NDArray[np.float64],
    y_val: NDArray[np.int64],
) -> NDArray[np.int64]:
    return net.predict(X_val)


def evaluate(
    y_pred: NDArray[np.int64],
    y_test: NDArray[np.int64],
) -> None:
    accuracy = get_accuracy(y_test, y_pred)
    print(f"Test accuracy: {accuracy}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(include_values=True, cmap="Blues", ax=None, xticks_rotation="horizontal")
    plt.show()


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
