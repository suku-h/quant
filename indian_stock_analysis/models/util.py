import numpy as np


def removeNaN(arr):
    return arr[~np.isnan(arr)]