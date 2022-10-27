import collections.abc as cabc
import types
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "SubSample",
]


class SubSample(pyca.LinOp):
    r"""
    Multi-dimensional sub-sampling operator.

    This operator extracts a subset of the input matching the provided subset specifier.
    Its Lipschitz constant is 1.

    Examples
    --------
    .. code-block:: python3

       ### Extract even samples of a 1D signal.
       import pycsou.operator.linop as pycl
       x = np.arange(10)
       S = pycl.SubSample(
             x.shape,
             slice(0, None, 2),
       )
       y = S(x)  # array([0, 2, 4, 6, 8])


    .. code-block:: python3

       ### Extract columns[1, 3, -1] from a 2D matrix
       import pycsou.operator.linop as pycl
       x = np.arange(3 * 40).reshape(3, 40)  # the input
       S = pycl.SubSample(
             x.shape,
             slice(None),  # take all rows
             [1, 3, -1],   # and these columns
       )
       y = S(x.reshape(-1)).reshape(3, 3)  # array([[  1.,   3.,  39.],
                                           #        [ 41.,  43.,  79.],
                                           #        [ 81.,  83., 119.]])

    .. code-block:: python3

       ### Extract all red rows of an (D,H,W) RGB image matching a boolean mask.
       import pycsou.operator.linop as pycl
       x = np.arange(3 * 5 * 4).reshape(3, 5, 4)
       mask = np.r_[True, False, False, True, False]
       S = pycl.SubSample(
             x.shape,
             0,            # red channel
             mask,         # row selector
             slice(None),  # all columns; this field can be omitted.
       )
       y = S(x.reshape(-1)).reshape(1, mask.sum(), 4)  # array([[[ 0.,  1.,  2.,  3.],
                                                       #         [12., 13., 14., 15.]]])
    """
    IndexSpec = typ.Union[
        pyct.Integer,
        cabc.Sequence[pyct.Integer],
        slice,
        pyct.NDArray,  # ints or boolean mask (per dimension)
    ]

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        *indices: IndexSpec,
    ):
        self._arg_shape = pycu.as_canonical_shape(arg_shape)

        assert 1 <= len(indices) <= len(self._arg_shape)
        self._idx = [slice(None)] * len(self._arg_shape)
        for i, idx in enumerate(indices):
            if isinstance(idx, pyct.Integer):
                idx = slice(idx, idx + 1)
            self._idx[i] = idx
        self._idx = tuple(self._idx)

        output = np.broadcast_to(0, self._arg_shape)[self._idx]
        self._sub_shape = np.atleast_1d(output).shape

        super().__init__(shape=(np.prod(self._sub_shape), np.prod(self._arg_shape)))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""

        Parameters
        ----------
        arr: NDArray
            Array the samples are extracted from.

        Returns
        -------
            A sub-sampled array.
        """
        return arr[..., self.sampling_indices]

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        y = xp.zeros((*arr.shape[:-1], self.input_size), dtype=arr.dtype)
        y[..., self.sampling_indices] = arr
        return y
