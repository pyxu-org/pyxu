import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "TransposeAxes",
    "SqueezeAxes",
]


class TransposeAxes(pxa.UnitOp):
    """
    Reverse or permute the axes of an array.
    """

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
    ):
        """
        Parameters
        ----------
        arg_shape: NDArrayShape
            (N_1,...,N_D) input dimensions.
        axes: NDArrayAxis
            New axis order.

            If specified, must be a tuple or list which contains a permutation of [0,1,...,D-1].
            All axes are reversed if unspecified. (Default)
        """
        arg_shape = pxu.as_canonical_shape(arg_shape)
        N_dim = len(arg_shape)

        if axes is None:
            axes = np.arange(N_dim)[::-1]
        axes = pxu.as_canonical_shape(axes)
        assert len(axes) == len(set(axes)) == N_dim  # right number of axes provided & no duplicates
        assert all(-N_dim <= ax < N_dim for ax in axes)  # all axes in valid range
        axes = (np.array(axes, dtype=int) + N_dim) % N_dim  # get rid of negative axes

        dim = codim = np.prod(arg_shape)
        super().__init__(shape=(codim, dim))
        self._arg_shape = arg_shape
        self._out_shape = tuple(arg_shape[ax] for ax in axes)
        self._axes_fw = tuple(axes)
        self._axes_bw = tuple(np.argsort(axes))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        N_dim = len(sh)
        axes = tuple(range(N_dim)) + tuple(N_dim + ax for ax in self._axes_fw)

        arr = arr.reshape(*sh, *self._arg_shape)
        out = arr.transpose(axes)
        out = out.reshape(*sh, -1)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        N_dim = len(sh)
        axes = tuple(range(N_dim)) + tuple(N_dim + ax for ax in self._axes_bw)

        arr = arr.reshape(*sh, *self._out_shape)
        out = arr.transpose(axes)
        out = out.reshape(*sh, -1)
        return out

    def cogram(self) -> pxt.OpT:
        from pyxu.operator import IdentityOp

        return IdentityOp(dim=self.codim).squeeze()


class SqueezeAxes(pxa.UnitOp):
    """
    Remove axes of length one.
    """

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        axes: pxt.NDArrayAxis = None,
    ):
        """
        Parameters
        ----------
        arg_shape: NDArrayShape
            (N_1,...,N_D) input dimensions.
        axes: NDArrayAxis
            Axes to drop.

            If unspecified, all axes of shape 1 will be dropped.
            If an axis is selected with shape greater than 1, an error is raised.

        Notes
        -----
        * 1D arrays cannot be squeezed.
        * Given a D-dimensional input, at most D-1 dimensions may be dropped.
        """
        arg_shape = pxu.as_canonical_shape(arg_shape)
        N_dim = len(arg_shape)

        if axes is None:
            axes = np.arange(N_dim)[np.array(arg_shape) == 1]
        axes = pxu.as_canonical_shape(axes)
        assert all(-N_dim <= ax < N_dim for ax in axes)  # all axes in valid range
        axes = (np.unique(axes) + N_dim) % N_dim  # get rid of negative axes
        if len(axes) > 0:
            assert np.all(np.array(arg_shape)[axes] == 1)  # only squeezing size-1 dimensions
            assert len(axes) < N_dim  # cannot squeeze to 0d array.

        dim = codim = np.prod(arg_shape)
        super().__init__(shape=(codim, dim))

        self._arg_shape = arg_shape
        self._out_shape = tuple(arg_shape[ax] for ax in range(N_dim) if ax not in axes)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        out = arr.reshape(*sh, *self._out_shape)
        out = out.reshape(*sh, -1)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        out = arr.reshape(*sh, *self._arg_shape)
        out = out.reshape(*sh, -1)
        return out
