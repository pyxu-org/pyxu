import collections.abc as cabc
import typing as typ

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "SubSample",
    "Trim",
]


class SubSample(pxa.LinOp):
    r"""
    Multi-dimensional sub-sampling operator.

    This operator extracts a subset of the input matching the provided subset specifier.
    Its Lipschitz constant is 1.

    Examples
    --------
    .. code-block:: python3

       ### Extract even samples of a 1D signal.
       import pyxu.operator as pxo
       x = np.arange(10)
       S = pxo.SubSample(
             x.shape,
             slice(0, None, 2),
       )
       y = S(x)  # array([0, 2, 4, 6, 8])


    .. code-block:: python3

       ### Extract columns[1, 3, -1] from a 2D matrix
       import pyxu.operator as pxo
       x = np.arange(3 * 40).reshape(3, 40)  # the input
       S = pxo.SubSample(
             x.shape,
             slice(None),  # take all rows
             [1, 3, -1],   # and these columns
       )
       y = S(x.reshape(-1)).reshape(3, 3)  # array([[  1.,   3.,  39.],
                                           #        [ 41.,  43.,  79.],
                                           #        [ 81.,  83., 119.]])

    .. code-block:: python3

       ### Extract all red rows of an (D,H,W) RGB image matching a boolean mask.
       import pyxu.operator as pxo
       x = np.arange(3 * 5 * 4).reshape(3, 5, 4)
       mask = np.r_[True, False, False, True, False]
       S = pxo.SubSample(
             x.shape,
             0,            # red channel
             mask,         # row selector
             slice(None),  # all columns; this field can be omitted.
       )
       y = S(x.reshape(-1)).reshape(1, mask.sum(), 4)  # array([[[ 0.,  1.,  2.,  3.],
                                                       #         [12., 13., 14., 15.]]])
    """
    IndexSpec = typ.Union[
        pxt.Integer,
        cabc.Sequence[pxt.Integer],
        slice,
        pxt.NDArray,  # ints or boolean mask (per dimension)
    ]

    TrimSpec = typ.Union[
        pxt.Integer,
        cabc.Sequence[pxt.Integer],
        cabc.Sequence[tuple[pxt.Integer, pxt.Integer]],
    ]

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        *indices: IndexSpec,
    ):
        """
        Parameters
        ----------
        arg_shape: NDArrayShape
            Shape of the data to be sub-sampled.
        indices: ~pyxu.operator.SubSample.IndexSpec
            Sub-sample specifier per dimension. (See examples.)

            Valid specifiers are:

            * integers
            * lists (or arrays) of indices
            * slices
            * 1D boolean masks

            Unspecified trailing dimensions are not sub-sampled.
        """
        self._arg_shape = pxu.as_canonical_shape(arg_shape)

        assert 1 <= len(indices) <= len(self._arg_shape)
        self._idx = [slice(None)] * len(self._arg_shape)
        for i, idx in enumerate(indices):
            if isinstance(idx, pxt.Integer):
                idx = slice(idx, idx + 1)
            self._idx[i] = idx
        self._idx = tuple(self._idx)

        output = np.broadcast_to(0, self._arg_shape)[self._idx]
        self._sub_shape = np.atleast_1d(output).shape

        super().__init__(shape=(np.prod(self._sub_shape), np.prod(self._arg_shape)))
        self.lipschitz = 1

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Sub-sample the data.

        Parameters
        ----------
        arr: NDArray
            (..., arg_shape.prod()) data

        Returns
        -------
        out: NDArray
            (..., sub_shape.prod()) sub-sampled data points.
        """
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self._arg_shape)

        selector = (*[slice(None) for dim in sh], *self._idx)
        out = arr[selector].reshape(*sh, -1)

        out = pxu.read_only(out)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Up-sample the data.

        Parameters
        ----------
        arr: NDArray
            (..., sub_shape.prod()) data points.

        Returns
        -------
        out: NDArray
            (..., arg_shape.prod()) up-sampled data points. (Zero-filled.)
        """
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self._sub_shape)

        xp = pxu.get_array_module(arr)
        out = xp.zeros((*sh, *self._arg_shape), dtype=arr.dtype)
        selector = (*[slice(None) for dim in sh], *self._idx)
        out[selector] = arr

        out = out.reshape(*sh, -1)
        return out

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxa.UnitOp.svdvals(self, **kwargs)
        return D

    def gram(self) -> pxt.OpT:
        @pxrt.enforce_precision(i="arr")
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            _op = _._op
            out = _op.adjoint(_op.apply(arr))
            return out

        op = px_src.from_source(
            cls=pxa.OrthProjOp,
            shape=(self.dim, self.dim),
            embed=dict(_op=self),
            apply=op_apply,
        )
        return op

    def cogram(self) -> pxt.OpT:
        from pyxu.operator.linop.base import IdentityOp

        CG = IdentityOp(dim=self.codim)
        return CG.squeeze()

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / (1 + damp)
        return op


def Trim(
    arg_shape: pxt.NDArrayShape,
    trim_width: SubSample.TrimSpec,
) -> pxt.OpT:
    """
    Multi-dimensional trimming operator.

    This operator trims the input array in each dimension according to specified widths.

    Parameters
    ----------
    arg_shape: NDArrayShape
        Shape of the input array.
    trim_width: ~pyxu.operator.SubSample.TrimSpec
        Number of values trimmed from the edges of each axis.
        Multiple forms are accepted:

        * ``int``: trim each dimension's head/tail by `trim_width`.
        * ``tuple[int, ...]``: trim dimension[k]'s head/tail by `trim_width[k]`.
        * ``tuple[tuple[int, int], ...]``: trim dimension[k]'s head/tail by `trim_width[k][0]` / `trim_width[k][1]`
          respectively.

    Returns
    -------
    op: OpT
    """
    arg_shape = tuple(arg_shape)
    N_dim = len(arg_shape)

    # transform `trim_width` to canonical form tuple[tuple[int, int]]
    is_seq = lambda _: isinstance(_, cabc.Sequence)
    if not is_seq(trim_width):  # int-form
        trim_width = ((trim_width, trim_width),) * N_dim
    assert len(trim_width) == N_dim, "arg_shape/trim_width are length-mismatched."
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
