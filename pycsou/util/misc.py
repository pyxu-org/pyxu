from typing import Tuple
import numpy as np


def is_range_broadcastable(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> bool:
    if shape1[1] != shape2[1]:
        return False
    elif shape1[0] == shape2[0]:
        return True
    elif shape1[0] == 1 or shape2[0] == 1:
        return True
    else:
        return False


def range_broadcast_shape(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Tuple[int, int]:
    if not is_range_broadcastable(shape1, shape2):
        raise ValueError('Shapes are not (range) broadcastable.')
    shape = tuple(np.fmax(shape1, shape2).tolist())
    return shape
