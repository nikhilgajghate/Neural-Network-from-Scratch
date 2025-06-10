## Neural Network Components Documentation

This document provides detailed documentation for the core components of the neural network implementation.

### Cross Entropy Loss

The `CrossEntropy` class implements the cross-entropy loss function, which is commonly used for classification tasks.

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