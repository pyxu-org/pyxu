import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu

__all__ = [
    "shift_loss",
    "KLDivergence",
]


def shift_loss(
    op: pxt.OpT,
    data: pxt.NDArray = None,
) -> pxt.OpT:
    r"""
    Shift a functional :math:`f(x)` to a loss functional.

    .. math::

       g(x; c) = f(x - c)

    Parameters
    ----------
    data: NDArray
        (M,) input data.

    Returns
    -------
    op: OpT
        (1, M) Loss functionial.  If `data` is omitted, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)


class KLDivergence(pxa.ProxFunc):
    r"""
    Generalised Kullback-Leibler divergence :math:`D_{KL}(\mathbf{y}||\mathbf{x}):=\sum_{i=1}^N y_i\log(y_i/x_i) -y_i
    +x_i`.
    """

    def __init__(
        self,
        dim: pxt.Integer,
        data: pxt.NDArray,
    ):
        r"""
        Parameters
        ----------
        dim: Integer
        data: NDArray
            (M,) strictly positive input data.

        Examples
        --------
        .. code-block:: python3

           import numpy as np
           from pyxu.operator import KLDivergence

           y = np.arange(10)
           loss = KLDivergence(dim=y.size, data=y)
           x = 2 * np.arange(10)
           loss(x)
           # 13.80837687480246
           np.round(loss.prox(x, tau=1))
           # array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])

        Notes
        -----
        In information theory, and in the case where :math:`\mathbf{y}` and :math:`\mathbf{x}`  sum to one  --and hence
        can be interpreted as discrete probability distributions, the KL-divergence can be interpreted as the relative
        entropy of :math:`\mathbf{y}` w.r.t. :math:`\mathbf{x}`, i.e. the amount of information lost when using
        :math:`\mathbf{x}` to approximate :math:`\mathbf{y}`. It is particularly useful in the context of count data
        with Poisson distribution. Indeed, the KL-divergence corresponds –up to an additive constant– to the likelihood
        of the data :math:`\mathbf{y}` where each component is independent with Poisson distribution and respective
        intensities given by the entries of :math:`\mathbf{x}`. See [FuncSphere]_ Section 5 of Chapter 7 for the
        computation of its proximal operator.
        """
        super().__init__(shape=(1, dim))

        xp = pxu.get_array_module(data)
        assert data.ndim == 1, "`data` must have 1 axis only."
        assert data.size == dim, "`data` must have the same size as the domain dimension."
        assert xp.all(data > 0), "KL Divergence only defined for positive-valued arguments."
        self.data = data.reshape(1, -1)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        assert xp.all(arr > 0), "KL Divergence only defined for positive-valued arguments."
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, self.data.size)
        out = xp.true_divide(self.data, arr)
        out = xp.log(out)
        out *= self.data
        out -= self.data
        out += arr
        return xp.sum(out, axis=-1).reshape(*sh, -1)

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        sh = arr.shape[:-1]
        arr = arr.reshape(-1, self.data.size)
        out = xp.sqrt((arr - tau) ** 2 + 4 * tau * self.data)
        out += arr
        out -= tau
        out /= 2
        return out.reshape(*sh, -1)
