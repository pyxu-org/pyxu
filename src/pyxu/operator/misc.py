import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
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


class ReshapeAxes(pxa.UnitOp):
    """
    Reshape operator's output.
    """

    def __init__(
        self,
        arg_shape: pxt.NDArrayShape,
        out_shape: pxt.NDArrayShape,
    ):
        """
        Parameters
        ----------
        arg_shape: NDArrayShape
            (N_1,...,N_D) input dimensions.
        out_shape: NDArrayShape
            (M_1,...,M_D) output dimensions.

        Notes
        -----
        The new shape should be compatible with the original shape. If an integer, then the result will be a 1D array of
        that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and
        remaining dimensions.
        """
        arg_shape = pxu.as_canonical_shape(arg_shape)
        out_shape = pxu.as_canonical_shape(out_shape)

        if all(ax >= 1 for ax in out_shape):
            pass  # all good
        elif sum(ax == -1 for ax in out_shape) == 1:
            # infer missing dimension value
            size = np.prod(arg_shape) // abs(np.prod(out_shape))

            out_shape = list(out_shape)
            out_shape[out_shape.index(-1)] = size
        else:
            raise ValueError("Only one -1 entry allowed.")

        dim = np.prod(arg_shape)
        codim = np.prod(out_shape)
        super().__init__(shape=(codim, dim))

        self._arg_shape = arg_shape
        self._out_shape = tuple(out_shape)

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
