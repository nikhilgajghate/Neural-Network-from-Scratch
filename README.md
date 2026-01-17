# MNIST Neural Network from Scratch

A complete implementation of a neural network from scratch using NumPy for the MNIST digit classification task. This project demonstrates the fundamentals of deep learning by implementing all core components including forward/backward propagation, optimization algorithms, and loss functions.

## Requirements

- Python 3.11+
- See `requirements.txt` for all dependencies

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MNIST
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Training

Train the neural network on MNIST dataset:

```bash
python src/main.py --dataset mnist
```

### Training with Visualization

Train and visualize the dataset:

```bash
python src/main.py --dataset mnist --visualize True
```

### Command Line Arguments

- `--dataset`: Dataset to use (currently supports "mnist"). Will consider adding CIFAR-10, CIFAR-100, and Fashion MNIST
- `--visualize`: Whether to visualize the dataset (default: False)

## Project Structure

```
MNIST/
├── src/
│   ├── main.py                 # Main entry point
│   ├── helpers.py              # Training utilities and helpers
│   ├── Network/                # Neural network components
│   │   ├── network.py          # Main Network class
│   │   ├── layer.py            # Base layer abstractions
│   │   ├── linear.py           # Linear/fully connected layer
│   │   ├── relu.py             # ReLU activation function
│   │   ├── softmax.py          # Softmax activation
│   │   ├── cross_entropy.py    # Cross-entropy loss function
│   │   └── training_params.py  # Training hyperparameters
│   └── dataset/
│       └── MNIST/
│           └── mnist.py        # MNIST dataset loader
├── scripts/
│   └── check_code.py           # Code quality checks
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Network Architecture

The default network architecture consists of:

1. **Input Layer**: 784 neurons (28×28 flattened MNIST images)
2. **Hidden Layer**: 32 neurons with ReLU activation
3. **Output Layer**: 10 neurons (digits 0-9) with softmax activation
4. **Loss Function**: Cross-entropy loss

### Training Parameters

- **Batch Size**: 200
- **Epochs**: 10
- **Learning Rate**: 0.01
- **Momentum**: 0.9

## Component Documentation

### Network Class

The main `Network` class orchestrates the entire neural network.

#### Methods

##### `add_layer(layer)`
- **Parameters**: `layer` - A concrete layer implementation
- **Description**: Adds a layer to the network

##### `forward(X)`
- **Parameters**: `X` - Input tensor (numpy array)
- **Returns**: Network output (numpy array)
- **Description**: Performs forward propagation through all layers

##### `backward(nextgrad)`
- **Parameters**: `nextgrad` - Gradient from loss function
- **Returns**: List of gradients for each layer
- **Description**: Performs backpropagation through all layers

##### `train_step(X, y)`
- **Parameters**: 
  - `X` - Input batch (numpy array)
  - `y` - Target labels (numpy array)
- **Returns**: Tuple of (loss, gradients)
- **Description**: Performs one training step (forward + backward pass)

##### `predict(X)`
- **Parameters**: `X` - Input tensor (numpy array)
- **Returns**: Predicted class labels (numpy array)
- **Description**: Makes predictions on input data

##### `summary()`
- **Description**: Prints network architecture summary

### Cross Entropy Loss

The `CrossEntropy` class implements the cross-entropy loss function for classification tasks.

#### Methods

##### `forward(X, y)`
- **Parameters:**
  - `X`: Input predictions from the network (numpy array)
  - `y`: True labels (numpy array)
- **Returns:**
  - `loss`: The computed cross-entropy loss (scalar)
- **Description:** Computes the cross-entropy loss between the network predictions and true labels.

##### `backward(X, y)`
- **Parameters:**
  - `X`: Input predictions from the network (numpy array)
  - `y`: True labels (numpy array)
- **Returns:**
  - `grad`: Gradient of the loss with respect to the input (numpy array)
- **Description:** Computes the gradient of the cross-entropy loss for backpropagation.

### ReLU Activation

The `ReLU` class implements the Rectified Linear Unit (ReLU) activation function.

#### Methods

##### `forward(X)`
- **Parameters:**
  - `X`: Input tensor (numpy array)
- **Returns:**
  - `output`: ReLU-activated output (numpy array)
- **Description:** Applies the ReLU activation function: max(0, x)

##### `backward(nextgrad)`
- **Parameters:**
  - `nextgrad`: Gradient from the next layer (numpy array)
- **Returns:**
  - `gradInput`: Gradient with respect to the input (numpy array)
  - `[]`: Empty list (no parameters to update)
- **Description:** Computes the gradient of the ReLU function for backpropagation.

### Linear Layer

The `Linear` class implements a fully connected (linear) layer in the neural network.

#### Methods

##### `forward(X)`
- **Parameters:**
  - `X`: Input tensor (numpy array)
- **Returns:**
  - `output`: Linear transformation output (numpy array)
- **Description:** Performs the linear transformation: WX + b

##### `backward(nextgrad)`
- **Parameters:**
  - `nextgrad`: Gradient from the next layer (numpy array)
- **Returns:**
  - `gradInput`: Gradient with respect to the input (numpy array)
  - `[gradW, gradB]`: Gradients with respect to weights and biases
- **Description:** Computes gradients for backpropagation, including gradients for weights and biases.

#### Attributes

- `W`: Weight matrix (initialized with random values scaled by 0.01)
- `b`: Bias vector (initialized with zeros)
- `params`: List containing the learnable parameters [W, b]
- `gradW`: Gradient of the loss with respect to weights
- `gradB`: Gradient of the loss with respect to biases
- `gradInput`: Gradient of the loss with respect to the input

### Softmax Activation

The `Softmax` class implements the softmax activation function for multi-class classification.

#### Methods

##### `forward(X)`
- **Parameters:**
  - `X`: Input tensor (numpy array)
- **Returns:**
  - `output`: Softmax-activated output (numpy array)
- **Description:** Applies the softmax function to convert logits to probabilities

##### `backward(nextgrad)`
- **Parameters:**
  - `nextgrad`: Gradient from the next layer (numpy array)
- **Returns:**
  - `gradInput`: Gradient with respect to the input (numpy array)
  - `[]`: Empty list (no parameters to update)
- **Description:** Computes the gradient of the softmax function for backpropagation.

## Training Process

### 1. Data Loading
- Loads MNIST dataset using Keras
- Flattens 28×28 images to 784-dimensional vectors
- Normalizes pixel values to [0, 1] range

### 2. Network Initialization
- Creates network with specified architecture
- Initializes weights with small random values
- Sets up momentum buffers for optimization

### 3. Training Loop
- **Epoch Loop**: Iterates through all training data
- **Batch Processing**: Processes data in mini-batches
- **Forward Pass**: Computes predictions through network
- **Loss Computation**: Calculates cross-entropy loss
- **Backward Pass**: Computes gradients via backpropagation
- **Parameter Update**: Updates weights using momentum SGD
- **Progress Tracking**: Monitors loss and accuracy

### 4. Evaluation
- **Validation**: Makes predictions on test set
- **Metrics**: Computes accuracy and confusion matrix
- **Visualization**: Plots training curves and confusion matrix

## 🛠️ Development

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Static type checking
- **Hatchling**: Build system

### Running Quality Checks

```bash
# Format code
black src/ scripts/

# Lint code
ruff check src/ scripts/

# Type checking
mypy src/ scripts/

# Run all checks
python scripts/check_code.py
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
