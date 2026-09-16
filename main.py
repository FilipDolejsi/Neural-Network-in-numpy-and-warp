from os.path import join
from mnist_data_loader import MnistDataloader
from numpy_version import NumpyNeuralNetwork, get_accuracy, get_predictions
import argparse

from warp_version import WarpNeuralNetwork


def main():
    parser = argparse.ArgumentParser(description="Picking which model to use.")
    parser.add_argument("-np", "--np_model", action="store_true", help="Use Numpy model")
    parser.add_argument("-wp", "--wp_model", action="store_true", help="Use Warp model")
    args = parser.parse_args()

    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    X_train_flattened = x_train.reshape(x_train.shape[0], -1).T / 255.0
    X_test_flattened = x_test.reshape(x_test.shape[0], -1).T / 255.0

    models = []
    if args.np_model:
        models.append(NumpyNeuralNetwork())
    if args.wp_model:
        models.append(WarpNeuralNetwork())

    for model in models:
        model.gradient_descent(X=X_train_flattened, y=y_train, iterations=500, alpha=0.10)
        model.test_eval(X=X_test_flattened, y=y_test)

if __name__ == "__main__":
    main()
