import numpy as np


def tanh(x):
    """
    Hyberbolic tangent.
    Used for forwards propagation.
    """
    return np.tanh(x)


def prime_tanh(x, output):
    """
    Derivative of tanh.
    User for back propagation.
    """
    return 1 - np.power(tanh(x), 2)


def softmax(x):
    """
    Calculates the probability that the sample matches a certain class
    """
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)


def prime_softmax(x, output):
    result = np.zeros(x.shape)

    for i in range(len(output)):
        for j in range(len(x)):
            if i == j:
                result = output[i] * (1 - x[i])
            else:
                result = -output[i] * x[j]

    return result
