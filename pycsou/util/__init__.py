import numpy as np
import typing as ty


def broadcasted_sum(shape1: ty.Tuple[int, ...], shape2: ty.Tuple[int, ...]) -> ty.Tuple[
    bool, ty.Optional[ty.Tuple[int, ...]]]:
    valid_shapes = True
    try:
        arr1 = np.zeros(shape1)
        arr2 = np.zeros(shape2)
        arr3 = arr1 + arr2
        output_shape = arr3.shape
    except ValueError:
        valid_shapes = False
        output_shape = None
    return valid_shapes, output_shape


def broadcasted_matmul(shape1: ty.Tuple[int, ...], shape2: ty.Tuple[int, ...]) -> ty.Tuple[
    bool, ty.Optional[ty.Tuple[int, ...]]]:
    valid_shapes = True
    try:
        arr1 = np.zeros(shape1)
        arr2 = np.zeros(shape2)
        arr3 = arr1 @ arr2
        output_shape = arr3.shape
    except ValueError:
        valid_shapes = False
        output_shape = None
    return valid_shapes, output_shape
