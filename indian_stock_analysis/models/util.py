from copy import deepcopy
import numpy as np


def removeNaN(arr):
    return arr[~np.isnan(arr)]


def getMaxDataPoints(res: np.ndarray, indicatorVals: np.ndarray, r1, r2, r3, r4):
    if len(r1) == 0:
        r1 = deepcopy(res)
    if len(r2) == 0:
        r2 = deepcopy(res)
    if len(r3) == 0:
        r3 = deepcopy(res)
    if len(r4) == 0:
        r4 = deepcopy(res)

    return min(len(res), len(indicatorVals), len(r1), len(r2), len(r3), len(r4))


def getUsableDataPoints(arr, max_len):
    if len(arr) > 0:
        return arr[len(arr) - max_len:]
    else:
        return arr