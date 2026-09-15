from abc import ABC, abstractmethod

class AbstractNeuralNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def one_hot(self, y, num_classes):
        pass

    @abstractmethod
    def backward(self, x, y, a_1, a_2, z_1):
        pass

    @abstractmethod
    def update(self, dW_1, db_1, dW_2, db_2, alpha):
        pass

    @abstractmethod
    def test_eval(self, X, y):
        pass