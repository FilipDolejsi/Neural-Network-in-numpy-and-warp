import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from abstract_neural_network import AbstractNeuralNetwork


def relu(X):
    return np.maximum(X, 0)

def relu_deriv(X):
    return (X > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)

def get_predictions(a_2):
    return np.argmax(a_2, 0)

def get_accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.size

class NumpyNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.w_1, self.b_1, self.w_2, self.b_2 = self.init_params()

    def init_params(self):
        w_1 = np.random.rand(10, 28 * 28) - 0.5
        b_1 = np.random.rand(10, 1) - 0.5
        w_2 = np.random.rand(10, 10) - 0.5
        b_2 = np.random.rand(10, 1) - 0.5
        return w_1, b_1, w_2, b_2

    def forward(self, x):
        z_1 = self.w_1.dot(x) + self.b_1
        a_1 = relu(z_1)
        z_2 = self.w_2.dot(a_1) + self.b_2
        a_2 = softmax(z_2)
        return z_1, a_1, z_2, a_2

    def one_hot(self, y, num_classes):
        one_hot_encoding = np.zeros((y.size, num_classes))
        one_hot_encoding[np.arange(y.size), y] = 1
        return one_hot_encoding.T

    def backward(self, x, y, a_1, a_2, z_1):
        m = y.size
        one_hot_y = self.one_hot(y, 10)
        dZ_2 = a_2 - one_hot_y
        dW_2 = 1/m * dZ_2.dot(a_1.T)
        db_2 = 1/m * np.sum(dZ_2, axis=1, keepdims=True)
        dZ_1 = self.w_2.T.dot(dZ_2) * relu_deriv(z_1)
        dW_1 = 1/m * dZ_1.dot(x.T)
        db_1 = 1/m * np.sum(dZ_1, axis=1, keepdims=True)
        return dW_1, db_1, dW_2, db_2

    def update(self, dW_1, db_1, dW_2, db_2, alpha):
        self.w_1 = self.w_1 - alpha * dW_1
        self.b_1 = self.b_1 - alpha * db_1
        self.w_2 = self.w_2 - alpha * dW_2
        self.b_2 = self.b_2 - alpha * db_2
        return self.w_1, self.b_1, self.w_2, self.b_2

    def gradient_descent(self, X, y, iterations, alpha):
        print("-"*20 + "NUMPY Neural Network TRAINING" + "-"*20)
        train_loss = []
        pbar = tqdm(range(iterations), desc="training")
        for i in pbar:
            z_1, a_1, z_2, a_2 = self.forward(x=X)
            dW_1, db_1, dW_2, db_2 = self.backward(x=X, y=y, a_1=a_1, a_2=a_2, z_1=z_1)
            w_1, b_1, w_2, b_2 = self.update(dW_1=dW_1, db_1=db_1, dW_2=dW_2, db_2=db_2, alpha=alpha)

            acc = get_accuracy(get_predictions(a_2), y)
            loss = -np.mean(np.log(a_2[y, np.arange(y.size)] + 1e-9))

            train_loss.append((i, loss))

            pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{loss:.3f}")
        iters, total_loss = zip(*train_loss)
        plt.plot(iters, total_loss)
        plt.ylabel("Train Loss")
        plt.xlabel("Iterations")
        plt.savefig("numpy_train_loss.png")
        print("-"*60)
        return w_1, b_1, w_2, b_2

    def test_eval(self, X, y):
        _, _, _, test_forward = self.forward(x=X)
        acc = get_accuracy(get_predictions(test_forward), y)
        loss = -np.mean(np.log(test_forward[y, np.arange(y.size)] + 1e-9))
        print("-"*20 + "NUMPY Neural Network TESTING" + "-"*20)
        print("Test Accuracy:", acc)
        print("Test Loss:", loss)
        print("-"*60)
        return acc, loss