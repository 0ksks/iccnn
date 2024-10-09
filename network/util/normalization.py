import numpy as np

from global_variable import EPSILON


class Normalization:
    @staticmethod
    def z_score(input: np.ndarray):
        return (input - input.mean(axis=-1)) / (input.std(axis=-1) + EPSILON)

    @staticmethod
    def max_min(input: np.ndarray):
        input_min = np.min(input, axis=-1)
        return (input - input_min) / (input.max(axis=-1) - input_min + EPSILON)
