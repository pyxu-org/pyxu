import collections.abc as cabc
import typing as typ

import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.interop.source as px_src
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
       y = S(x)  # array([[  1.,   3.,  39.],
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
       y = S(x)  # array([[[ 0,  1,  2,  3],
                 #         [12, 13, 14, 15]]])
    """
    IndexSpec = typ.Union[
        pxt.Integer,
        cabc.Sequence[pxt.Integer],
        cabc.Sequence[bool],
        slice,
    ]

    TrimSpec = typ.Union[
        pxt.Integer,
        cabc.Sequence[pxt.Integer],
        cabc.Sequence[tuple[pxt.Integer, pxt.Integer]],
    ]

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        *indices: IndexSpec,
    ):
        """
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) domain dimensions.
        indices: ~pyxu.operator.SubSample.IndexSpec
            Sub-sample specifier per dimension. (See examples.)

            Valid specifiers are:

            * integers
            * 1D sequence of int/bool-s
            * slices

            Unspecified trailing dimensions are not sub-sampled.

        Notes
        -----
        The co-dimension rank **always** matches the dimension rank, i.e. sub-sampling does not drop dimensions.
        Single-element dimensions can be removed by composing :py:class:`~pyxu.operator.SubSample` with
        :py:class:`~pyxu.operator.SqueezeAxes`.
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,  # temporary; just to validate dim_shape
        )
        assert 1 <= len(indices) <= self.dim_rank

        # Explicitize missing trailing indices.
        idx = [slice(None)] * self.dim_rank
        for i, _idx in enumerate(indices):
            idx[i] = _idx

        # Replace integer indices with slices.
        for i, _idx in enumerate(idx):
            if isinstance(_idx, pxt.Integer):
                M = self.dim_shape[i]
                _idx = (_idx + M) % M  # get rid of negative indices
                idx[i] = slice(_idx, _idx + 1)

        # Compute output shape, then re-instantiate `self`.
        self._idx = tuple(idx)
        out = np.broadcast_to(0, self.dim_shape)[self._idx]
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=out.shape,
        )
        self.lipschitz = 1

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Sub-sample the data.

        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) data points.

        Returns
        -------
        out: NDArray
            (..., N1,..,NK) sub-sampled data points.
        """
        sh = arr.shape[: -self.dim_rank]

        selector = ((slice(None),) * len(sh)) + self._idx
        out = arr[selector]
        return pxu.read_only(out)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Up-sample the data.

        Parameters
        ----------
        arr: NDArray
            (..., N1,...,NK) data points.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) up-sampled data points. (Zero-filled.)
        """
        sh = arr.shape[: -self.codim_rank]

        ndi = pxd.NDArrayInfo.from_obj(arr)
        kwargs = dict(
            shape=(*sh, *self.dim_shape),
            dtype=arr.dtype,
        )
        if ndi == pxd.NDArrayInfo.DASK:
            stack_chunks = arr.chunks[: -self.codim_rank]
            core_chunks = ("auto",) * self.dim_rank
            kwargs.update(chunks=stack_chunks + core_chunks)
        out = ndi.module().zeros(**kwargs)

        selector = ((slice(None),) * len(sh)) + self._idx
        out[selector] = arr
        return out

    def svdvals(self, **kwargs) -> pxt.NDArray:
        D = pxa.UnitOp.svdvals(self, **kwargs)
        return D

    def gram(self) -> pxt.OpT:
        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            _op = _._op
            out = _op.adjoint(_op.apply(arr))
            return out

        G = px_src.from_source(
            cls=pxa.OrthProjOp,
            dim_shape=self.dim_shape,
            codim_shape=self.dim_shape,
            embed=dict(_op=self),
            apply=op_apply,
        )
        return G

    def cogram(self) -> pxt.OpT:
        from pyxu.operator import IdentityOp

        CG = IdentityOp(dim_shape=self.codim_shape)
        return CG

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / (1 + damp)
        return op


def Trim(
    dim_shape: pxt.NDArrayShape,
    trim_width: SubSample.TrimSpec,
) -> pxt.OpT:
    """
    Multi-dimensional trimming operator.

    This operator trims the input array in each dimension according to specified widths.

    Parameters
    ----------
    dim_shape: NDArrayShape
        (M1,...,MD) domain dimensions.
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
    dim_shape = pxu.as_canonical_shape(dim_shape)
    N_dim = len(dim_shape)

    # transform `trim_width` to canonical form tuple[tuple[int, int]]
    is_seq = lambda _: isinstance(_, cabc.Sequence)
    if not is_seq(trim_width):  # int-form
        trim_width = ((trim_width, trim_width),) * N_dim
    assert len(trim_width) == N_dim, "dim_shape/trim_width are length-mismatched."
    if not is_seq(trim_width[0]):  # tuple[int, ...] form
        trim_width = tuple((w, w) for w in trim_width)
    else:  # tuple[tuple[int, int], ...] form
        pass

    # translate `trim_width` to `indices` needed for SubSample
    indices = []
    for (w_head, w_tail), dim_size in zip(trim_width, dim_shape):
        s = slice(w_head, dim_size - w_tail)
        indices.append(s)

    op = SubSample(dim_shape, *indices)
    return op
