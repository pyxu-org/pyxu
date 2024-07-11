import numpy as np

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu

__all__ = [
    "KLDivergence",
]


class KLDivergence(pxa.ProxFunc):
    r"""
    Generalised Kullback-Leibler divergence
    :math:`D_{KL}(\mathbf{y}||\mathbf{x}) := \sum_{i} y_{i} \log(y_{i} / x_{i}) - y_{i} + x_{i}`.
    """

    def __init__(self, data: pxt.NDArray):
        r"""
        Parameters
        ----------
        data: NDArray
            (M1,...,MD) non-negative input data.

        Examples
        --------
        .. code-block:: python3

           import numpy as np
           from pyxu.operator import KLDivergence

           y = np.arange(5)
           loss = KLDivergence(y)

           loss(2 * y)  # [3.06852819]
           np.round(loss.prox(2 * y, tau=1))  # [0. 2. 4. 6. 8.]

        Notes
        -----
        * When :math:`\mathbf{y}` and :math:`\mathbf{x}` sum to one, and hence can be interpreted as discrete
          probability distributions, the KL-divergence corresponds to the relative entropy of :math:`\mathbf{y}` w.r.t.
          :math:`\mathbf{x}`, i.e. the amount of information lost when using :math:`\mathbf{x}` to approximate
          :math:`\mathbf{y}`. It is particularly useful in the context of count data with Poisson distribution; the
          KL-divergence then corresponds (up to an additive constant) to the likelihood of :math:`\mathbf{y}` where each
          component is independent with Poisson distribution and respective intensities given by :math:`\mathbf{x}`. See
          [FuncSphere]_ Chapter 7, Section 5 for the computation of its proximal operator.
        * :py:class:`~pyxu.operator.KLDivergence` is not backend-agnostic: inputs to arithmetic methods must have the
          same backend as `data`.
        * If `data` is a DASK array, it's entries are assumed non-negative de-facto.  Reason: the operator should be
          quick to build under all circumstances, and this is not guaranteed if we have to check that all entries are positive for out-of-core arrays.
        * If `data` is a DASK array, the core-dimensions of arrays supplied to arithmetic methods **must** have the
          same chunk-size as `data`.
        """
        super().__init__(
            dim_shape=data.shape,
            codim_shape=1,
        )

        ndi = pxd.NDArrayInfo.from_obj(data)
        if ndi != pxd.NDArrayInfo.DASK:
            assert (data >= 0).all(), "KL Divergence only defined for non-negative arguments."
        self._data = data

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        axis = tuple(range(-self.dim_rank, 0))
        out = self._kl_div(arr, self._data)
        out = out.sum(axis=axis)[..., np.newaxis]
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = arr - tau
        out = x + xp.sqrt((x**2) + ((4 * tau) * self._data))
        out *= 0.5
        return out

    @staticmethod
    def _kl_div(x: pxt.NDArray, data: pxt.NDArray) -> pxt.NDArray:
        # Element-wise KL-divergence
        N = pxd.NDArrayInfo  # short-hand
        ndi = N.from_obj(x)

        if ndi == N.NUMPY:
            sp = pxu.import_module("scipy.special")
            out = sp.kl_div(data, x)
        elif ndi == N.CUPY:
            sp = pxu.import_module("cupyx.scipy.special")
            out = sp.kl_div(data, x)
        elif ndi == N.DASK:
            assert x.chunks[-data.ndim :] == data.chunks
            xp = ndi.module()

            out = xp.map_blocks(
                KLDivergence._kl_div,
                x,
                data,
                dtype=data.dtype,
                meta=data._meta,
            )
        else:
            raise NotImplementedError
        return out
