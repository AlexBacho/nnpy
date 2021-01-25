import numpy as np


def get_random_array(*dims, offset=0):
    return np.random.rand(*dims) + offset
