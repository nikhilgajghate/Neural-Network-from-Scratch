import argparse

import numpy as np

from src.helpers import (
    create_network,
    evaluate,
    load_dataset,
    train,
    validate,
    visualize_dataset,
)
from src.Network.training_params import BATCH_SIZE, EPOCH, LEARNING_RATE, MOMENTUM


def main() -> None:

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
    net = create_network(X_train.shape[1], len(np.unique(y_test)))
    net.summary()

    # Train model
    net = train(net, X_train, y_train, BATCH_SIZE, EPOCH, LEARNING_RATE, mu=MOMENTUM)
    print("=============================")

    # Validate model
    y_pred = validate(net, X_test, y_test)
    print("=============================")

    # Evaluate model
    evaluate(y_pred, y_test)
    print("=============================")


if __name__ == "__main__":
    main()
