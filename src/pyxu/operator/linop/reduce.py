import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.util as pxu

__all__ = [
    "Sum",
]


class Sum(pxa.LinOp):
    r"""
    Multi-dimensional sum reduction :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1}
    \times\cdots\times N_{D}}`.

    Notes
    -----
    * The co-dimension rank **always** matches the dimension rank, i.e. summed-over dimensions are not dropped.
      Single-element dimensions can be removed by composing :py:class:`~pyxu.operator.Sum` with
      :py:class:`~pyxu.operator.SqueezeAxes`.

    * The matrix operator of a 1D reduction applied to :math:`\mathbf{x} \in \mathbb{R}^{M}` is given by

      .. math::

         \mathbf{A}(x) = \mathbf{1}^{T} \mathbf{x},

      where :math:`\sigma_{\max}(\mathbf{A}) = \sqrt{M}`.  An ND reduction is a chain of 1D reductions in orthogonal
      dimensions.  Hence the Lipschitz constant of an ND reduction is the product of Lipschitz constants of all 1D
      reductions involved, i.e.:

      .. math::

         L = \sqrt{\prod_{i_{k}} M_{i_{k}}},

      where :math:`\{i_{k}\}_{k}` denotes the axes being summed over.
    """

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        axis: pxt.NDArrayAxis = None,
    ):
        r"""
        Multi-dimensional sum reduction.

        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) domain dimensions.
        axis: NDArrayAxis
            Axis or axes along which a sum is performed.  The default, axis=None, will sum all the elements of the input
            array.
        """
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=dim_shape,  # temporary; to canonicalize dim_shape.
        )

        if axis is None:
            axis = np.arange(self.dim_rank)
        axis = pxu.as_canonical_axes(axis, rank=self.dim_rank)
        axis = set(axis)  # drop duplicates

        codim_shape = list(self.dim_shape)  # array shape after reduction
        for i in range(self.dim_rank):
            if i in axis:
                codim_shape[i] = 1
        self._codim_shape = tuple(codim_shape)

        self._axis = tuple(axis)
        self.lipschitz = self.estimate_lipschitz()

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.dim_rank]
        axis = tuple(ax + len(sh) for ax in self._axis)
        out = arr.sum(axis=axis, keepdims=True)
        return out

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[: -self.codim_rank]
        xp = pxu.get_array_module(arr)
        out = xp.broadcast_to(arr, sh + self.dim_shape)
        return out

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        M = np.prod(self.dim_shape) / np.prod(self.codim_shape)
        L = np.sqrt(M)
        return L
