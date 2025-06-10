from Network.cross_entropy import CrossEntropy

class Network():
    def __init__(self, lossfunc=CrossEntropy()):
        self.params = []
        self.layers = []
        self.loss_func = lossfunc
        self.grads = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.append(layer.params)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, nextgrad):
        self.clear_grad_param()
        for layer in reversed(self.layers):
            nextgrad, grad = layer.backward(nextgrad)
            self.grads.append(grad)
        return self.grads
    
    def train_step(self, X, y):
        out = self.forward(X)
        loss = self.loss_func.forward(out, y)
        nextgrad = self.loss_func.backward(out, y)
        l2 = self.backward(nextgrad)
        return loss, l2
    
    def predict(self, X):
        X = self.forward(X)
        return np.argmax(X, axis=1)
    
    def predict_scores(self, X):
        X = self.forward(X)
        return X
    
    def clear_grad_param(self):
        self.grads = []

    def display(self):
        for layer in self.layers:
            if layer.name == "Linear":
                print(f"{layer.name}")
                print(f"W shape: {layer.W.shape}")
                print(f"b shape: {layer.b.shape}")
                print("------")
            else:
                print(f"{layer.name}")
                print("------")