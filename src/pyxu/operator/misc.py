import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "BroadcastAxes",
    "ReshapeAxes",
    "SqueezeAxes",
    "TransposeAxes",
]


class TransposeAxes(pxa.UnitOp):
    """
    Reverse or permute the axes of an array.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
    ):
        """
        Parameters
        ----------
        axes: NDArrayAxis
            New axis order.

            If specified, must be a tuple or list which contains a permutation of [0,1,...,D-1].
            All axes are reversed if unspecified. (Default)
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,  # preliminary; just to get dim_rank computed correctly.
        )

        if axes is None:
            axes = np.arange(self.dim_rank)[::-1]
        axes = pxu.as_canonical_axes(axes, rank=self.dim_rank)
        assert len(axes) == len(set(axes)) == self.dim_rank  # right number of axes provided & no duplicates

        # update codim to right shape
        self._codim_shape = tuple(self.dim_shape[ax] for ax in axes)
        self._axes_fw = axes
        self._axes_bw = tuple(np.argsort(axes))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]
        N = len(sh)
        axes = tuple(range(N)) + tuple(N + ax for ax in self._axes_fw)
        out = arr.transpose(axes)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.codim_rank]
        N = len(sh)
        axes = tuple(range(N)) + tuple(N + ax for ax in self._axes_bw)
        out = arr.transpose(axes)
        return out


class SqueezeAxes(pxa.UnitOp):
    """
    Remove axes of length one.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
    ):
        """
        Parameters
        ----------
        axes: NDArrayAxis
            Axes to drop.

            If unspecified, all axes of shape 1 will be dropped.
            If an axis is selected with shape greater than 1, an error is raised.

        Notes
        -----
        * 1D arrays cannot be squeezed.
        * Given a D-dimensional input, at most D-1 dimensions may be dropped.
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,  # preliminary; just to get dim_rank computed correctly.
        )

        dim_shape = np.array(self.dim_shape)  # for advanced indexing below.
        if axes is None:
            axes = np.arange(self.dim_rank)[dim_shape == 1]
        axes = pxu.as_canonical_axes(axes, rank=self.dim_rank)
        axes = np.unique(axes)  # drop duplicates
        if len(axes) > 0:
            assert np.all(dim_shape[axes] == 1)  # only squeezing size-1 dimensions
            assert len(axes) < self.dim_rank  # cannot squeeze to 0d array.

        # update codim to right shape
        self._codim_shape = tuple(dim_shape[ax] for ax in range(self.dim_rank) if ax not in axes)
        self._idx_fw = tuple(0 if (ax in axes) else slice(None) for ax in range(self.dim_rank))
        self._idx_bw = tuple(np.newaxis if (ax in axes) else slice(None) for ax in range(self.dim_rank))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr[..., *self._idx_fw]
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        out = arr[..., *self._idx_bw]
        return out


class ReshapeAxes(pxa.UnitOp):
    """
    Reshape an array.

    Notes
    -----
    * If an integer, then the result will be a 1D array of that length. One co-dimension can be -1. In this case, the
      value is inferred from the length of the array and remaining dimensions.
    * Reshaping DASK inputs may be sub-optimal based on the array's chunk structure: use at your own risk.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        dim_shape = pxu.as_canonical_shape(dim_shape)
        codim_shape = pxu.as_canonical_shape(codim_shape)

        if all(ax >= 1 for ax in codim_shape):
            pass  # all good
        elif sum(ax == -1 for ax in codim_shape) == 1:
            # infer missing dimension value
            size = np.prod(dim_shape) // abs(np.prod(codim_shape))

            codim_shape = list(codim_shape)
            codim_shape[codim_shape.index(-1)] = size
        else:
            raise ValueError("Only one -1 entry allowed.")

        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        assert self.dim_size == self.codim_size  # reshaping does not change cell count.

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]
        out = arr.reshape(*sh, *self.codim_shape)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.codim_rank]
        out = arr.reshape(*sh, *self.dim_shape)
        return out


class BroadcastAxes(pxa.LinOp):
    """
    Broadcast an array.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )

        # Fail if not broadcastable.
        assert self.codim_size >= self.dim_size
        np.broadcast_shapes(self.dim_shape, self.codim_shape)

        self.lipschitz = self.estimate_lipschitz()

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # Compute (expand,) assuming no stacking dimensions.
        rank_diff = self.codim_rank - self.dim_rank
        expand = (np.newaxis,) * rank_diff

        # Extend (expand,) to handle stacking dimensions.
        sh = arr.shape[: -self.dim_rank]
        expand = ((slice(None),) * len(sh)) + expand

        xp = pxu.get_array_module(arr)
        y = xp.broadcast_to(
            arr[expand],
            (*sh, *self.codim_shape),
        )
        return pxu.read_only(y)

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        # Compute (axis, select) assuming no stacking dimensions.
        rank_diff = self.codim_rank - self.dim_rank
        dim_shape_bcast = ((1,) * rank_diff) + self.dim_shape
        axis = filter(
            lambda i: self.codim_shape[i] != dim_shape_bcast[i],
            range(self.codim_rank),
        )
        select = (0,) * rank_diff

        # Extend (axis, select) to handle stacking dimensions
        sh = arr.shape[: -self.codim_rank]
        axis = tuple(ax + len(sh) for ax in axis)
        select = ((slice(None),) * len(sh)) + select

        y = arr.sum(axis=axis, keepdims=True)[select]
        return y

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        L = np.sqrt(self.codim_size / self.dim_size)
        return L

    @pxrt.enforce_precision()
    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        width = pxrt.getPrecision()

        D = xp.full(kwargs["k"], self.lipschitz, dtype=width.value)
        return D

    def gram(self) -> pxt.OpT:
        from pyxu.operator import HomothetyOp

        op = HomothetyOp(
            dim_shape=self.dim_shape,
            cst=self.codim_size / self.dim_size,
        )
        return op

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = pxu.copy_if_unsafe(self.adjoint(arr))
        cst = self.codim_size / self.dim_size
        out /= cst + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        cst = self.codim_size / self.dim_size
        op = self.T / (cst + damp)
        return op
