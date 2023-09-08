import math
import types
import typing as typ

import pyxu.info.ptype as pxt
import pyxu.util as pxu

__all__ = [
    "OnlineMoment",
    "OnlineCenteredMoment",
    "OnlineVariance",
    "OnlineStd",
    "OnlineSkewness",
    "OnlineKurtosis",
]


class _OnlineStat:
    """
    Abstract base class to compute online statistics based on outputs of a Sampler object.

    An _OnlineStat object should be paired with a single Sampler object.

    Composite _OnlineStat objects can be implemented via the overloaded +, -, *, and ** operators.
    """

    def __init__(self):
        self._num_samples = 0
        self._stat = None

    def update(self, x: pxt.NDArray) -> pxt.NDArray:
        """
        Update the online statistic based on a new sample. Should update `_num_samples` and `_stat` attributes.

        Parameters
        ----------
        x: NDArray
            New sample.

        Returns
        -------
        stat: NDArray
            Updated statistic.
        """
        raise NotImplementedError

    def stat(self) -> pxt.NDArray:
        """Get current online statistic estimate."""
        return self._stat

    def __add__(self, other: "_OnlineStat"):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) + other.update(x)
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __sub__(self, other: "_OnlineStat"):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) - other.update(x)
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __mul__(self, other: typ.Union[pxt.Real, pxt.Integer, "_OnlineStat"]):
        stat = _OnlineStat()
        if isinstance(other, _OnlineStat):
            stat_update = lambda _, x: self.update(x) * other.update(x)
        elif isinstance(other, pxt.Real) or isinstance(other, pxt.Integer):
            stat_update = lambda _, x: self.update(x) * other
        else:
            return NotImplemented
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __rmul__(self, other: typ.Union[pxt.Real, pxt.Integer]):
        return self.__mul__(other)

    def __truediv__(self, other: typ.Union[pxt.Real, pxt.Integer, "_OnlineStat"]):
        stat = _OnlineStat()
        if isinstance(other, _OnlineStat):

            def stat_update(stat, x):
                xp = pxu.get_array_module(x)
                out = xp.divide(self.update(x), other.update(x), where=(other.update(x) != 0))
                out[other.update(x) == 0] = xp.nan
                return out

        elif isinstance(other, pxt.Real) or isinstance(other, pxt.Integer):
            stat_update = lambda _, x: self.update(x) / other
        else:
            return NotImplemented
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __pow__(self, expo: typ.Union[pxt.Real, pxt.Integer]):
        if not (isinstance(expo, pxt.Real) or isinstance(expo, pxt.Integer)):
            return NotImplemented
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) ** expo
        stat.update = types.MethodType(stat_update, stat)
        return stat


class OnlineMoment(_OnlineStat):
    r"""
    Pointwise online moment.

    For :math:`d \geq 1`, the :math:`d`-th order centered moment of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq
    k \leq K}` is given by :math:`\frac{1}{K}\sum_{k=1}^K \mathbf{x}_k^d`.
    """

    def __init__(self, order: pxt.Real = 1):
        super().__init__()
        self._order = order

    def update(self, x: pxt.NDArray) -> pxt.NDArray:
        if self._num_samples == 0:
            self._stat = x**self._order
        else:
            self._stat *= self._num_samples
            self._stat += x**self._order
        self._num_samples += 1
        self._stat /= self._num_samples
        return self._stat


class OnlineCenteredMoment(_OnlineStat):
    r"""
    Pointwise online centered moment.

    For :math:`d \geq 2`, the :math:`d`-th order centered moment of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq
    k \leq K}` is given by :math:`\boldsymbol{\mu}_d=\frac{1}{K} \sum_{k=1}^K (\mathbf{x}_k-\boldsymbol{\mu})^d`, where
    :math:`\boldsymbol{\mu}` is the sample mean.

    Notes
    -----
    This class implements the *Welford algorithm* described in [WelfordAlg]_, which is a numerically stable algorithm
    for computing online centered moments. In particular, it avoids `catastrophic cancellation
    <https://en.wikipedia.org/wiki/Catastrophic_cancellation>`_ issues that may arise when naively computing online
    centered moments, which would lead to a loss of numerical precision.

    Note that this class internally stores the values of all online centered moments of order :math:`d'` for :math:`2
    \leq d' \leq d` in the attribute ``_corrected_sums`` as well as the online mean (``_mean`` attribute). More
    precisely, the array ``_corrected_sums[i, :]`` corresponds to the online sum :math:`\boldsymbol{\mu}_{i+2}=
    \sum_{k=1}^K (\mathbf{x}_k-\boldsymbol{\mu})^{i+2}` for :math:`0 \leq i \leq d-2`.
    """

    def __init__(self, order: pxt.Real = 2):
        super().__init__()
        self._order = order
        self._corrected_sums = None
        self._mean = None

    def update(self, x: pxt.NDArray) -> pxt.NDArray:
        if self._num_samples == 0:
            xp = pxu.get_array_module(x)
            self._corrected_sums = xp.zeros((self._order - 1,) + x.shape)
            self._mean = x.copy()
        else:
            temp = (x - self._mean) / (self._num_samples + 1)
            for r in range(self._order, 1, -1):  # Update in descending order because updates depend on lower orders
                for s in range(2, r):  # s = r term excluded since it corresponds to previous iterate
                    self._corrected_sums[r - 2, :] += (
                        math.comb(r, s) * self._corrected_sums[s - 2, :] * (-temp) ** (r - s)
                    )
                self._corrected_sums[r - 2, :] += self._num_samples * (-temp) ** r  # Contribution of s = 0 term
                self._corrected_sums[r - 2, :] += (self._num_samples * temp) ** r
            self._mean *= self._num_samples / (self._num_samples + 1)
            self._mean += x / (self._num_samples + 1)
        self._num_samples += 1
        self._stat = self._corrected_sums[-1, :] / self._num_samples
        return self._stat


def OnlineVariance():
    r"""
    Pointwise online variance.

    The pointwise online variance of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq k \leq K}` is given by
    :math:`\boldsymbol{\sigma}^2 = \frac{1}{K}\sum_{k=1}^K (\mathbf{x}_k-\boldsymbol{\mu})^2`, where
    :math:`\boldsymbol{\mu}` is the sample mean.
    """
    return OnlineCenteredMoment(order=2)


def OnlineStd():
    r"""
    Pointwise online standard deviation.

    The pointwise online standard deviation of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq k \leq K}` is given
    by :math:`\boldsymbol{\sigma}=\sqrt{\frac{1}{K}\sum_{k=1}^K (\mathbf{x}_k - \boldsymbol{\mu})^2}`, where
    :math:`\boldsymbol{\mu}` is the sample mean.
    """
    return OnlineVariance() ** (1 / 2)


def OnlineSkewness():
    r"""
    Pointwise online skewness.

    The pointwise online skewness of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq k \leq K}` is given by
    :math:`\frac{1}{K}\sum_{k=1}^K \left( \frac{\mathbf{x}_k-\boldsymbol{\mu}}{\boldsymbol{\sigma}}\right)^3`, where
    :math:`\boldsymbol{\mu}` is the sample mean and :math:`\boldsymbol{\sigma}` its standard deviation.

    `Skewness <https://en.wikipedia.org/wiki/Skewness>`_ is a measure of asymmetry of a distribution around its mean.
    Negative skewness indicates that the distribution has a heavier tail on the left side than on the right side,
    positive skewness indicates the opposite, and values close to zero indicate a symmetric distribution.
    """
    return OnlineCenteredMoment(order=3) / OnlineStd() ** 3


def OnlineKurtosis():
    r"""
    Pointwise online kurtosis.

    The pointwise online variance of the :math:`K` samples :math:`(\mathbf{x}_k)_{1 \leq k \leq K}` is given by
    :math:`\frac{1}{K}\sum_{k=1}^K \left( \frac{\mathbf{x}_k-\boldsymbol{\mu}}{\boldsymbol{\sigma}}\right)^4`, where
    :math:`\boldsymbol{\mu}` is the sample mean and :math:`\boldsymbol{\sigma}` its standard deviation.

    `Kurtosis <https://en.wikipedia.org/wiki/Kurtosis>`_ is a measure of the heavy-tailedness of a distribution. In
    particular, the kurtosis of a Gaussian distribution is always 3.
    """
    return OnlineCenteredMoment(order=4) / OnlineStd() ** 4
