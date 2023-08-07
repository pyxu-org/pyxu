import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "shift_loss",
    "KLDivergence",
]


def shift_loss(
    op: pyct.OpT,
    data: pyct.NDArray = None,
) -> pyct.OpT:
    r"""
    Shift a functional :math:`f(x)` to a loss functional :math:`g(x) = f(x - c)`.

    Parameters
    ----------
    data: pyct.NDArray
        (M,) input data.

    Returns
    -------
    op: pyct.OpT
        (1, M) Loss functionial.
        If `data = None`, then this function is a no-op.
    """
    if data is None:
        return op
    else:
        return op.argshift(-data)


class KLDivergence(pyca.ProxFunc):
    r"""
    Generalised Kullback-Leibler divergence :math:`D_{KL}(\mathbf{y}||\mathbf{x}):=\sum_{i=1}^N y_i\log(y_i/x_i) -y_i +x_i`.

    The generalised Kullback-Leibler divergence is defined as:

    .. math::

       D_{KL}(\mathbf{y}||\mathbf{x}):=\sum_{i=1}^N H(y_i,x_i) -y_i +x_i, \quad \forall \mathbf{y}, \mathbf{x} \in \mathbb{R}^N,

    where

    .. math::

       H(y,x):=\begin{cases}
       y\log(y/x) &\, \text{if} \,x>0, y>0,\\
       0&\, \text{if} \,x=0, y\geq 0,\\
       +\infty &\,\text{otherwise.}
       \end{cases}

    Parameters
    ----------
    dim: int
        Dimension of the domain.
    data: Union[Number, np.ndarray]
        Data vector :math:`\mathbf{y}` to match.

    Returns
    -------
    :py:class:`~pycsou.core.functional.ProximableFunctional`
        The KL-divergence.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.func.loss import KLDivergence

    .. doctest::

       >>> y = np.arange(10)
       >>> loss = KLDivergence(dim=y.size, data=y)
       >>> x = 2 * np.arange(10)
       >>> loss(x)
       13.80837687480246
       >>> np.round(loss.prox(x, tau=1))
       array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.])

    Notes
    -----
    In information theory, and in the case where :math:`\mathbf{y}` and :math:`\mathbf{x}`  sum to one  --and hence can be interpreted as discrete probability distributions,
    the KL-divergence can be interpreted as the relative entropy of :math:`\mathbf{y}` w.r.t. :math:`\mathbf{x}`,
    i.e. the amount of information lost when using :math:`\mathbf{x}` to approximate :math:`\mathbf{y}`.
    It is particularly useful in the context of count data with Poisson distribution. Indeed, the KL-divergence corresponds
    –up to an additive constant– to the likelihood of the data :math:`\mathbf{y}` where each component is independent
    with Poisson distribution and respective intensities given by the entries of :math:`\mathbf{x}`.
    See [FuncSphere]_ Section 5 of Chapter 7 for the computation of its proximal operator.

    See Also
    --------
    :py:class:`~pycsou.func.penalty.ShannonEntropy`, :py:class:`~pycsou.func.penalty.LogBarrier`
    """

    def __init__(
        self,
        dim: pyct.Integer = None,
        data: pyct.NDArray = None,
    ):
        super().__init__(shape=(1, dim))
        self.data = data

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        out = xp.true_divide(self.data, arr, where=(arr > 0) * (self.data > 0), out=xp.zeros_like(arr))
        out = xp.log(out, where=out > 0, out=xp.zeros_like(out))
        out *= self.data
        out -= self.data
        out += arr
        return xp.sum(out, axis=-1, keepdims=True)

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        r"""
        Proximal operator of the KL-divergence functional (see [FuncSphere]_ Section 5 of Chapter 7).

        Parameters
        ----------
        arr: pyct.NDArray
            Input.
        tau: pyct.Real
            Scaling constant.

        Returns
        -------
        pyct.NDArray
            Proximal point of arr.
        """
        xp = pycu.get_array_module(arr)
        return (arr - tau + xp.sqrt((arr - tau) ** 2 + 4 * tau * self.data)) / 2
