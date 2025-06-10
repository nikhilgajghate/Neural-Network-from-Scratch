import numpy as np

class Linear():
    def __init__(self, input_size, output_size):
        # Layer name for identification
        self.name = "Linear"
        
        # Initialize weights with small random values
        # Using 0.01 as scaling factor to prevent large initial values
        self.W = np.random.randn(input_size, output_size) * 0.01
        
        # Initialize biases to zero
        self.b = np.zeros((1, output_size))
        
        # Store learnable parameters
        self.params = [self.W, self.b]
        
        # Initialize gradient storage
        self.gradW = None  # Gradient for weights
        self.gradB = None  # Gradient for biases
        self.gradInput = None  # Gradient for input

    def forward(self, X):
        # Store input for backward pass
        self.X = X
        
        # Linear transformation: y = Wx + b
        # X shape: (batch_size, input_size)
        # W shape: (input_size, output_size)
        # b shape: (1, output_size)
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, nextgrad):
        # Compute gradient with respect to weights
        # dL/dW = X^T * dL/dy
        self.gradW = np.dot(self.X.T, nextgrad)
        
        # Compute gradient with respect to biases
        # dL/db = sum(dL/dy)
        self.gradB = np.sum(nextgrad, axis=0)
        
        # Compute gradient with respect to input
        # dL/dX = dL/dy * W^T
        self.gradInput = np.dot(nextgrad, self.W.T)
        
        return self.gradInput, [self.gradW, self.gradB]