import numpy as np
from helpers import softmax

class CrossEntropy:
    """
    This class is used to calculate the cross entropy loss and its gradient.
    It takes the output of the network and the true labels as input.
    It returns the loss and the gradient.
    """
    def forward(self, X, y):
        # Store the number of samples for later use
        self.m = y.shape[0]
        
        # Apply softmax to get probability distribution
        self.p = softmax(X)
        
        # Calculate cross entropy loss: -log(p[true_class])
        # range(self.m) creates indices for each sample
        # y contains the true class indices
        cross_entropy = -np.log(self.p[range(self.m), y])
        
        # Average the loss across all samples
        loss = cross_entropy[0] / self.m
        return loss
    
    def backward(self, X, y):
        # Get the index of the true class
        y_idx = y.argmax()        
        
        # Initialize gradient with softmax probabilities
        grad = softmax(X)
        
        # Subtract 1 from the true class probabilities
        # This is the gradient of cross entropy with respect to softmax input
        grad[range(self.m), y] -= 1
        
        # Normalize gradient by number of samples
        grad /= self.m
        return grad