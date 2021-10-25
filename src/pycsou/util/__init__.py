import typing as typ

import numpy as np


def broadcast_sum_shapes(
    shape1: typ.Tuple[int, ...],
    shape2: typ.Tuple[int, ...],
) -> typ.Tuple[int, ...]:
    return np.broadcast_shapes(shape1, shape2)


def broadcast_matmul_shapes(
    shape1: typ.Tuple[int, ...],
    shape2: typ.Tuple[int, ...],
) -> typ.Tuple[int, ...]:
    sh1_was_1d = False
    if len(shape1) == 1:
        sh1_was_1d = True
        shape1 = (1,) + shape1

    sh2_was_1d = False
    if len(shape2) == 1:
        sh2_was_1d = True
        shape2 = shape2 + (1,)

    if shape1[-1] != shape2[-2]:
        raise ValueError(f"Cannot @-multiply shapes {shape1}, {shape2}.")

    sh = np.broadcast_shapes(shape1[:-2], shape2[:-2])
    if sh1_was_1d and sh2_was_1d:
        pass
    elif (not sh1_was_1d) and sh2_was_1d:
        sh = sh + (shape1[-2],)
    elif sh1_was_1d and (not sh2_was_1d):
        sh = sh + (shape2[-1],)
    else:
        sh = sh + (shape1[-2], shape2[-1])
    return sh
