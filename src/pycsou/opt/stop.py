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

    Tip
    ---
    If you want to add a grace period to a solver, i.e. for it to do *at least* N iterations before
    stopping based on the value of another criteria, you can AND `MaxIter` with the other criteria.

    Example
    -------

    >>> sc = MaxIter(n=5) & AbsErr(eps=0.1)
    # If N_iter < 5 -> never stop.
    # If N_iter >= 5 -> stop if AbsErr() decides to.
    """

    def __init__(self, n: int):
        """
        Parameters
        ----------
        n: int
            Max number of iterations allowed.
        """
        try:
            assert int(n) > 0
            self._n = int(n)
        except:
            raise ValueError(f"n: expected positive integer, got {n}.")
        self._i = 0

    def stop(self, state: cabc.Mapping) -> bool:
        self._i += 1
        return self._i > self._n

    def info(self) -> cabc.Mapping[str, float]:
        return dict(N_iter=self._i)

    def clear(self):
        self._i = 0


class ManualStop(pycs.StoppingCriterion):
    """
    Continue-forever criterion.

    This class is useful when calling `Solver.fit` with mode=MANUAL/ASYNC to defer the stopping
    decision to an explicit call by the user, i.e.:
    * mode=MANUAL: user must stop calling `next(solver.steps())`;
    * mode=ASYNC: user must call `Solver.stop`.
    """

    def stop(self, state: cabc.Mapping) -> bool:
        return False

    def info(self) -> cabc.Mapping[str, float]:
        return dict()

    def clear(self):
        pass


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


class AbsError(pycs.StoppingCriterion):
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
            data = {f"AbsError[{self._var}]": float(self._val[0])}
        else:
            data = {
                f"AbsError[{self._var}]_min": float(self._val.min()),
                f"AbsError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class RelError(pycs.StoppingCriterion):
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
            data = {f"RelError[{self._var}]": float(self._val[0])}
        else:
            data = {
                f"RelError[{self._var}]_min": float(self._val.min()),
                f"RelError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]
        self._x_prev = None
