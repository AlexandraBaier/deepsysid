from typing import List, Tuple

import numpy as np


def mean_stddev(array_seq: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(np.vstack(array_seq), axis=0)
    stddev = np.std(np.vstack(array_seq), axis=0)
    return mean, stddev


def denormalize(x, mean, stddev):
    return x * stddev + mean


def normalize(x, mean, stddev):
    return (x - mean) / stddev
