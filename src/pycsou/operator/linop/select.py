import collections.abc as cabc
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.operator.interop.source as pycsrc
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "SubSample",
    "Trim",
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

    TrimSpec = typ.Union[
        pyct.Integer,
        cabc.Sequence[pyct.Integer],
        cabc.Sequence[tuple[pyct.Integer, pyct.Integer]],
    ]

    def __init__(
        self,
        arg_shape: pyct.NDArrayShape,
        *indices: IndexSpec,
    ):
        """
        Parameters
        ----------
        arg_shape: pyct.NDArrayShape
            Shape of the data to be sub-sampled.
        indices: IndexSpec
            Sub-sample specifier per dimension. (See examples.)

            Valid specifiers are:

            * integers
            * lists (or arrays) of indices
            * slices
            * 1D boolean masks

            Unspecified trailing dimensions are not sub-sampled.
        """
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
        """
        Sub-sample the data.

        Parameters
        ----------
        arr: pyct.NDArray
            (..., arg_shape.prod()) data

        Returns
        -------
        out: pyct.NDArray
            (..., sub_shape.prod()) sub-sampled data points.
        """
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self._arg_shape)

        selector = (*[slice(None) for dim in sh], *self._idx)
        out = arr[selector].reshape(*sh, -1)

        out = pycu.read_only(out)
        return out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        """
        Up-sample the data.

        Parameters
        ----------
        arr: pyct.NDArray
            (..., sub_shape.prod()) data points.

        Returns
        -------
        out: pyct.NDArray
            (..., arg_shape.prod()) up-sampled data points. (Zero-filled.)
        """
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self._sub_shape)

        xp = pycu.get_array_module(arr)
        out = xp.zeros((*sh, *self._arg_shape), dtype=arr.dtype)
        selector = (*[slice(None) for dim in sh], *self._idx)
        out[selector] = arr

        out = out.reshape(*sh, -1)
        return out

    @pycrt.enforce_precision()
    def lipschitz(self, **kwargs) -> pyct.Real:
        self._lipschitz = 1
        return self._lipschitz

    def svdvals(self, **kwargs) -> pyct.NDArray:
        D = pyca.UnitOp.svdvals(self, **kwargs)
        return D

    def gram(self) -> pyct.OpT:
        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            _op = _._op
            out = _op.adjoint(_op.apply(arr))
            return out

        op = pycsrc.from_source(
            cls=pyca.OrthProjOp,
            shape=(self.dim, self.dim),
            embed=dict(_op=self),
            apply=op_apply,
        )
        return op

    def cogram(self) -> pyct.OpT:
        from pycsou.operator.linop.base import IdentityOp

        CG = IdentityOp(dim=self.codim)
        return CG.squeeze()

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = self.adjoint(arr)
        if not np.isclose(damp := kwargs.get("damp", 0), 0):
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        op = self.T / (1 + kwargs.get("damp", 0))
        return op


def Trim(
    arg_shape: pyct.NDArrayShape,
    trim_width: SubSample.TrimSpec,
) -> pyct.OpT:
    """
    Multi-dimensional trimming operator.

    This operator trims the input array in each dimension according to specified widths.

    Parameters
    ----------
    arg_shape: pyct.NDArrayShape
        Shape of the input array.
    trim_width: TrimSpec
        Number of values trimmed from the edges of each axis.
        Multiple forms are accepted:

        * int: trim each dimension's head/tail by `trim_width`.
        * tuple[int, ...]: trim dimension[k]'s head/tail by `trim_width[k]`.
        * tuple[tuple[int, int], ...]: trim dimension[k]'s head/tail by `trim_width[k][0]` /
          `trim_width[k][1]` respectively.

    Returns
    -------
    op: pyct.OpT
    """
    arg_shape = tuple(arg_shape)
    N_dim = len(arg_shape)

    # transform `trim_width` to canonical form tuple[tuple[int, int]]
    is_seq = lambda _: isinstance(_, cabc.Sequence)
    if not is_seq(trim_width):  # int-form
        trim_width = ((trim_width, trim_width),) * N_dim
    assert len(trim_width) == N_dim, f"arg_shape/trim_width are length-mismatched."
    if not is_seq(trim_width[0]):  # tuple[int, ...] form
        trim_width = tuple((w, w) for w in trim_width)
    else:  # tuple[tuple[int, int], ...] form
        pass

    # translate `trim_width` to `indices` needed for SubSample
    indices = []
    for (w_head, w_tail), dim_size in zip(trim_width, arg_shape):
        s = slice(w_head, dim_size - w_tail)
        indices.append(s)

    op = SubSample(arg_shape, *indices)
    return op
