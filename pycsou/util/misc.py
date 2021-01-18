# #############################################################################
# misc.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Miscellaneous functions.
"""


from typing import Tuple
import numpy as np


def is_range_broadcastable(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> bool:
    r"""
    Check if two shapes satisfy Numpy's broadcasting rules.

    Parameters
    ----------
    shape1: Tuple[int, int]
    shape2: Tuple[int, int]

    Returns
    -------
    bool
         ``True`` if broadcastable, ``False`` otherwise.

    Examples
    --------

    .. testsetup::

       from pycsou.util.misc import is_range_broadcastable

    .. doctest::

       >>> is_range_broadcastable((3,2), (1,2))
       True
       >>> is_range_broadcastable((3,2), (4,2))
       False
    """
    if shape1[1] != shape2[1]:
        return False
    elif shape1[0] == shape2[0]:
        return True
    elif shape1[0] == 1 or shape2[0] == 1:
        return True
    else:
        return False


def range_broadcast_shape(shape1: Tuple[int, int], shape2: Tuple[int, int]) -> Tuple[int, int]:
    r"""
    Given two shapes, determine broadcasting shape.

    Parameters
    ----------
    shape1: Tuple[int, int]
    shape2: Tuple[int, int]

    Returns
    -------
    Tuple[int, int]
        Broadcasting shape.

    Raises
    ------
    ValueError
        If the two shapes cannot be broadcasted.

    Examples
    --------

    .. testsetup::

       from pycsou.util.misc import range_broadcast_shape

    .. doctest::

       >>> range_broadcast_shape((3,2), (1,2))
       (3, 2)

    """
    if not is_range_broadcastable(shape1, shape2):
        raise ValueError('Shapes are not (range) broadcastable.')
    shape = tuple(np.fmax(shape1, shape2).tolist())
    return shape


def peaks(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Matlab 2D peaks function.

    Peaks is a function of two variables, obtained by translating and scaling Gaussian distributions (see `Matlab's peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`).
    This function is useful for testing purposes.

    Parameters
    ----------
    x: np.ndarray
        X coordinates.
    y: np.ndarray
        Y coordinates.

    Returns
    -------
    np.ndarray
        Values of the 2D function ``peaks`` at the points specified by the entries of ``x`` and ``y``.

    Examples
    --------
    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import peaks

       x = np.linspace(-3,3, 1000)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X,Y)
       plt.figure()
       plt.imshow(Z)

    """
    z = 3 * ((1 - x) ** 2) * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1 / 3) * np.exp(-(x + 1) ** 2 - y ** 2)
    return z
