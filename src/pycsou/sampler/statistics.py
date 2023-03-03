from __future__ import annotations

import types

import pycsou.util.ptype as pyct


class _OnlineStat:
    """Abstract base class to compute online statistics based on outputs of a Sampler object. An _OnlineStats object
    should be paired with a single Sampler object."""

    def __init__(self):
        self._num_samples = 0
        self._stat = None

    def update(self, x: pyct.NDArray):
        """Update the statistics based on last sample."""
        out = self._return_update(x)
        self._stat = out
        self._num_samples += 1
        return out

    def _return_update(self, x: pyct.NDArray):
        """Returns the updated statistics (to be implemented in subclasses)."""
        raise NotImplementedError

    def __add__(self, other: _OnlineStat):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) + other.update(x)
        stat._return_update = types.MethodType(stat_update, stat)
        return stat

    def __sub__(self, other: _OnlineStat):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) - other.update(x)
        stat._return_update = types.MethodType(stat_update, stat)
        return stat

    def __mul__(self, fact: pyct.Real):
        stat = _OnlineStat()
        stat_update = lambda _, x: fact * self.update(x)
        stat._return_update = types.MethodType(stat_update, stat)
        return stat

    def __pow__(self, expo: pyct.Real):
        stat = _OnlineStat()
        stat_update = lambda _, x: self.update(x) ** expo
        stat._return_update = types.MethodType(stat_update, stat)
        return stat


class OnlineMoment(_OnlineStat):
    """Online moment of any order."""

    def __init__(self, order: pyct.Integer = 1):
        super().__init__()
        self._order = order

    def _return_update(self, x: pyct.NDArray) -> pyct.NDArray:
        if self._num_samples == 0:
            stat = x**self._order
        else:
            stat = self._stat
            stat *= self._num_samples
            stat += x**self._order
        stat /= self._num_samples + 1
        return stat


def OnlineVariance():
    """Online variance."""
    return OnlineMoment(2) - OnlineMoment(1) ** 2
