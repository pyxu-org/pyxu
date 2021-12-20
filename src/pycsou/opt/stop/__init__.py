import collections.abc as cabc
import datetime as dt
import typing as typ

import pycsou.abc.solver as pycs


class MaxIter(pycs.StoppingCriterion):
    """
    Stop iterative solver after a fixed number of iterations.
    """

    def __init__(self, n: typ.Optional[int] = None):
        """
        Parameters
        ----------
        n: int | None
            Max number of iterations allowed.
            Defaults to infinity if unspecified, i.e. never halt.
        """
        super().__init__()
        self._n = n
        if n is not None:
            try:
                assert n > 0
                self._n = int(n)
            except:
                raise ValueError(f"n: expected positive integer, got {n}.")
        self._i = 0

    def stop(self, state: cabc.Mapping) -> bool:
        self._i += 1
        if self._n is None:
            return False
        else:
            return self._i > self._n

    def info(self) -> cabc.Mapping[str, float]:
        return dict(N_iter=self._i)

    def clear(self):
        self._i = 0


class MaxDuration(pycs.StoppingCriterion):
    """
    Stop iterative solver after a specified duration has elapsed.
    """

    def __init__(self, t: dt.timedelta):
        """
        Parameters
        ----------
        t: dt.timedelta
            Max runtime allowed.
        """
        super().__init__()
        try:
            assert t > dt.timedelta()
            self._t_max = t
        except:
            raise ValueError(f"t: expected positive duration, got {t}.")
        self._t_start = dt.datetime.now()
        self._t_now = self._t_start

    def stop(self, state: cabc.Mapping) -> bool:
        self._t_now = dt.datetime.now()
        return (self._t_now - self._t_start) > self._t_max

    def info(self) -> cabc.Mapping[str, float]:
        d = (self._t_now - self._t_start).total_seconds()
        return dict(duration=d)

    def clear(self):
        self._t_start = dt.datetime.now()
        self._t_now = self._t_start
