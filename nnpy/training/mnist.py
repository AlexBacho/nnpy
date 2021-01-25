import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from pathlib import Path

from nnpy.models import NeuralNetwork, load_nn
from nnpy.models import FullyConnectedLayer, ActivationLayer

from nnpy.functions import tanh, prime_tanh, mse, prime_mse, softmax, prime_softmax

NN_PATH = "/home/sani/code/nnpy/test/mnist_nn"
DEBUG = True

def train(retrain=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    """
    Since the pixel values are represented in the range [0; 255], 
    we are going to scale that down to a range of [0.0, 1.0].
    """

    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = to_categorical(y_train, 10)

    if Path(NN_PATH).exists() and not retrain:
        nn = load_nn(NN_PATH)
    else:
        nn = NeuralNetwork(loss_function=mse, prime_loss_function=prime_mse)
        nn = nn.add_layers(
            FullyConnectedLayer(100),
            ActivationLayer(tanh, prime_tanh),
            FullyConnectedLayer(50),
            ActivationLayer(tanh, prime_tanh),
            FullyConnectedLayer(10),
            ActivationLayer(tanh, prime_tanh),
            ActivationLayer(softmax, prime_softmax),
        ).fit(x_train[0:1000], y_train[0:1000], epochs=30, learning_rate=0.1)

    predicted = nn.predict(x_test[:100])

    normal_results = [np.argmax(sample) for sample in predicted]

    if DEBUG:
        print(normal_results[0], y_test[0])

    print(
        "Accuracy | Normal net: {}".format(accuracy_score(y_test[:100], normal_results))
    )

    nn.save(NN_PATH)

    return nn