import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
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

    def cogram(self) -> pxt.OpT:
        from pyxu.operator import IdentityOp

        return IdentityOp(dim=self.codim).squeeze()
