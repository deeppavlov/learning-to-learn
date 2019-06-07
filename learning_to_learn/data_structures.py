import numpy as np


def is_sparse_counter(counter, threshold):
    n = len(counter)
    max_key = max(counter.keys())
    return max_key / n > threshold


def counter_to_array(counter):
    n = max(counter.keys())
    array = np.zeros([n])
    for key, value in counter.items():
        array[key] = value
    return array
