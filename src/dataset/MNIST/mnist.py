from keras.datasets import mnist
from pydantic import BaseModel, ConfigDict
import numpy as np

class MNIST(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = 'MNIST'
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    def __init__(self, **data):
        super().__init__(**data)
        print(f"Loading {self.name} dataset")
   
    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def flatten_data(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        return X_train, X_test
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        return X_train, X_test