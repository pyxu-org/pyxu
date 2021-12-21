import collections.abc as cabc
import datetime as dt
import typing as typ

import numpy as np

import pycsou.abc.solver as pycs
import pycsou.util as pycu
import pycsou.util.ptype as pyct


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


class AbsMaxError(pycs.StoppingCriterion):
    """
    Stop iterative solver after absolute norm of a variable reaches threshold.
    """

    def __init__(
        self,
        eps: float,
        norm: float = 2,
        var: str = "primal",
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: float
            Positive threshold.
        norm: int | float
            Ln norm to use >= 0. (Default: L2.)
        var: str
            Variable in `Solver._mstate` to query.
        satisfy_all: bool
            If True (default) and `Solver._mstate[var]` is multi-dimensional, stop if all evaluation
            points lie below threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        try:
            assert norm >= 0
            self._norm = norm
        except:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._var = var
        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln norm(s) in stop().

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]
        if isinstance(x, pyct.Real):
            x = np.r_[x]
        xp = pycu.get_array_module(x)

        self._val = xp.linalg.norm(x, ord=self._norm, axis=-1, keepdims=True)
        f = xp.all if self._satisfy_all else xp.any
        return f(self._val <= self._eps)

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"AbsMax[{self._var}]": float(self._val[0])}
        else:
            data = {
                f"AbsMax[{self._var}]_min": float(self._val.min()),
                f"AbsMax[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class RelMaxError(pycs.StoppingCriterion):
    """
    Stop iterative solver after relative norm change of a variable reaches threshold.
    """

    def __init__(
        self,
        eps: float,
        norm: float = 2,
        var: str = "primal",
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: float
            Positive threshold.
        norm: int | float
            Ln norm to use >= 0. (Default: L2.)
        var: str
            Variable in `Solver._mstate` to query.
        satisfy_all: bool
            If True (default) and `Solver._mstate[var]` is multi-dimensional, stop if all evaluation
            points lie below threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        try:
            assert norm >= 0
            self._norm = norm
        except:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._var = var
        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln rel-norm(s) in stop().
        self._x_prev = None  # buffered var from last query.

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]
        xp = pycu.get_array_module(x)

        n = lambda _: xp.linalg.norm(_, ord=self._norm, axis=-1, keepdims=True)
        f = xp.all if self._satisfy_all else xp.any

        if self._x_prev is None:  # haven't seen enough past state -> don't stop yet.
            self._x_prev = x.copy()
            return False
        else:
            # Computing `_val` may fail in case x/0 (inf) or 0/0 (nan) occurs. For the purpose of
            # computing relative errors, we can safely assume all divide-by-zero computations lead
            # to np.inf.
            num, den = n(x - self._x_prev), n(self._x_prev)
            self._val = xp.zeros(x.shape)
            mask = xp.isclose(den, 0)
            self._val[mask] = np.inf
            self._val[~mask] = num[~mask] / den[~mask]

            self._x_prev = x.copy()
            return f(self._val <= self._eps)

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"RelMax[{self._var}]": float(self._val[0])}
        else:
            data = {
                f"RelMax[{self._var}]_min": float(self._val.min()),
                f"RelMax[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]
        self._x_prev = None
