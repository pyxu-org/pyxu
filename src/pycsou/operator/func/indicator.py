import numpy as np

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = ["NonNegativeOrthant"]


class NonNegativeOrthant(pyca.ProxFunc):
    """
    Indicator function of the non-negative orthant (positivity constraint).

    It is used to enforce non-negative real solutions and defined as:

    .. math::

       \iota(\mathbf{x}):=\begin{cases}
        0 \,\text{if} \,\mathbf{x}\in \mathbb{R}^N_+,\\
         \, +\infty\,\text{ortherwise}.
         \end{cases}
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.where(xp.all(arr >= 0, axis=-1), 0.0, np.infty)

    @pycrt.enforce_precision(i="arr")
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.clip(arr, 0.0, None)
