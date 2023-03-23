from __future__ import annotations

import types
import typing as typ

import pycsou.util as pycu
import pycsou.util.ptype as pyct


class _OnlineStat:
    """Abstract base class to compute online statistics based on outputs of a Sampler object. An _OnlineStats object
    should be paired with a single Sampler object."""

    def __init__(self):
        self._num_samples = 0
        self._stat = None

    def update(self, x: pyct.NDArray) -> pyct.NDArray:
        """
        Returns the updated statistics based on last sample.

        Must be overriden in sub-classes and update `_num_samples` and `_stat`.
        """
        raise NotImplementedError

    def stat(self) -> pyct.NDArray:
        """Get current online estimate."""
        return self._stat

    def __add__(self, other: _OnlineStat):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) + other.update(x)
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __sub__(self, other: _OnlineStat):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) - other.update(x)
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __mul__(self, other: typ.Union[pyct.Real, pyct.Integer, _OnlineStat]):
        stat = _OnlineStat()
        if isinstance(other, _OnlineStat):
            stat_update = lambda _, x: self.update(x) * other.update(x)
        elif isinstance(other, pyct.Real) or isinstance(other, pyct.Integer):
            stat_update = lambda _, x: self.update(x) * other
        else:
            return NotImplemented
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __rmul__(self, other: typ.Union[pyct.Real, pyct.Integer]):
        return self.__mul__(other)

    def __truediv__(self, other: typ.Union[pyct.Real, pyct.Integer, _OnlineStat]):
        stat = _OnlineStat()
        if isinstance(other, _OnlineStat):

            def stat_update(stat, x):
                xp = pycu.get_array_module(x)
                out = xp.divide(self.update(x), other.update(x), where=(other.update(x) != 0))
                out[other.update(x) == 0] = xp.nan
                return out

        elif isinstance(other, pyct.Real) or isinstance(other, pyct.Integer):
            stat_update = lambda _, x: self.update(x) / other
        else:
            return NotImplemented
        stat.update = types.MethodType(stat_update, stat)
        return stat

    def __pow__(self, expo: typ.Union[pyct.Real, pyct.Integer]):
        if not (isinstance(expo, pyct.Real) or isinstance(expo, pyct.Integer)):
            return NotImplemented
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) ** expo
        stat.update = types.MethodType(stat_update, stat)
        return stat


class OnlineMoment(_OnlineStat):
    """Online moment of any order."""

    def __init__(self, order: pyct.Integer = 1):
        super().__init__()
        self._order = int(order)

    def update(self, x: pyct.NDArray) -> pyct.NDArray:
        if self._num_samples == 0:
            self._stat = x**self._order
        else:
            self._stat *= self._num_samples
            self._stat += x**self._order
        self._num_samples += 1
        self._stat /= self._num_samples
        return self._stat


def OnlineVariance():
    """Online variance."""
    return OnlineMoment(2) - OnlineMoment(1) ** 2


def OnlineStd():
    """Online standard deviation."""
    return OnlineVariance() ** (1 / 2)


def OnlineSkew():
    """Online skew."""
    return (OnlineMoment(3) - 3 * OnlineMoment(2) * OnlineMoment(1) + 2 * OnlineMoment(1)) / OnlineStd() ** 3


def OnlineKurtosis():
    """Online kurtosis."""
    kurtosis = OnlineMoment(4) - 4 * OnlineMoment(3) * OnlineMoment(1)
    kurtosis += 6 * OnlineMoment(2) * OnlineMoment(1) ** 2 - 3 * OnlineMoment(1) ** 4
    return kurtosis / OnlineStd() ** 4
