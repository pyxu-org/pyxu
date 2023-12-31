import collections.abc as cabc
import datetime as dt
import warnings

import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.util as pxu
from pyxu.info.plugin import _load_entry_points

__all__ = [
    "AbsError",
    "ManualStop",
    "MaxDuration",
    "MaxIter",
    "Memorize",
    "RelError",
]

__all__ = _load_entry_points(globals(), group="pyxu.opt.stop", names=__all__)

SVFunction = cabc.Callable[[pxt.NDArray], pxt.NDArray]


def _norm(x: pxt.NDArray, ord: pxt.Integer, rank: pxt.Integer) -> pxt.NDArray:
    # x: (..., M1,...,MD) [rank=D]
    # n: (..., 1)         [`ord`-norm of `x`, computed over last `rank` axes]
    xp = pxu.get_array_module(x)
    axis = tuple(range(-rank, 0))
    if ord == 0:
        n = xp.sum(~xp.isclose(x, 0), axis=axis)
    elif ord == 1:
        n = xp.sum(xp.fabs(x), axis=axis)
    elif ord == 2:
        n = xp.sqrt(xp.sum(x**2, axis=axis))
    elif ord == np.inf:
        n = xp.max(xp.fabs(x), axis=axis)
    else:
        n = xp.power(xp.sum(x**ord, axis=axis), 1 / ord)
    return n[..., np.newaxis]


class MaxIter(pxa.StoppingCriterion):
    """
    Stop iterative solver after a fixed number of iterations.

    .. note::

       If you want to add a grace period to a solver, i.e. for it to do *at least* N iterations before stopping based
       on the value of another criteria, you can AND :py:class:`~pyxu.opt.stop.MaxIter` with the other criteria.

       .. code-block:: python3

          sc = MaxIter(n=5) & AbsError(eps=0.1)
          # If N_iter < 5  -> never stop.
          # If N_iter >= 5 -> stop if AbsError() decides to.
    """

    def __init__(self, n: pxt.Integer):
        """
        Parameters
        ----------
        n: Integer
            Max number of iterations allowed.
        """
        try:
            assert int(n) > 0
            self._n = int(n)
        except Exception:
            raise ValueError(f"n: expected positive integer, got {n}.")
        self._i = 0

    def stop(self, state: cabc.Mapping) -> bool:
        self._i += 1
        return self._i > self._n

    def info(self) -> cabc.Mapping[str, float]:
        return dict(N_iter=self._i)

    def clear(self):
        self._i = 0


class ManualStop(pxa.StoppingCriterion):
    """
    Continue-forever criterion.

    This class is useful when calling :py:meth:`~pyxu.abc.Solver.fit` with mode=MANUAL/ASYNC to defer the stopping
    decision to an explicit call by the user, i.e.:

    * mode=MANUAL: user must stop calling ``next(solver.steps())``;
    * mode=ASYNC: user must call :py:meth:`~pyxu.abc.Solver.stop`.
    """

    def stop(self, state: cabc.Mapping) -> bool:
        return False

    def info(self) -> cabc.Mapping[str, float]:
        return dict()

    def clear(self):
        pass


class MaxDuration(pxa.StoppingCriterion):
    """
    Stop iterative solver after a specified duration has elapsed.
    """

    def __init__(self, t: dt.timedelta):
        """
        Parameters
        ----------
        t: ~datetime.timedelta
            Max runtime allowed.
        """
        try:
            assert t > dt.timedelta()
            self._t_max = t
        except Exception:
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


class Memorize(pxa.StoppingCriterion):
    """
    Memorize a variable.  (Special :py:class:`~pyxu.abc.StoppingCriterion` mostly useful for tracking objective
    functions in :py:class:`~pyxu.abc.Solver`.)
    """

    def __init__(self, var: pxt.VarName):
        """
        Parameters
        ----------
        var: VarName
            Variable in :py:attr:`pyxu.abc.Solver._mstate` to query.  Must be a scalar or NDArray (1D).
        """
        self._var = var
        self._val = np.r_[0]  # last memorized value in stop().

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]
        if isinstance(x, pxt.Real):
            x = np.r_[x]
        assert x.ndim == 1

        self._val = pxu.compute(x)
        return False

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"Memorize[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"Memorize[{self._var}]_min": float(self._val.min()),
                f"Memorize[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class AbsError(pxa.StoppingCriterion):
    """
    Stop iterative solver after absolute norm of a variable (or function thereof) reaches threshold.
    """

    def __init__(
        self,
        eps: pxt.Real,
        var: pxt.VarName = "x",
        rank: pxt.Integer = 1,
        f: SVFunction = None,
        norm: pxt.Real = 2,
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: Real
            Positive threshold.
        var: VarName
            Variable in :py:attr:`pyxu.abc.Solver._mstate` to query.
            Must hold an NDArray.
        rank: Integer
            Array rank K of monitored variable **after** applying `f`. (See below.)
        f: ~collections.abc.Callable
            Optional function to pre-apply to ``_mstate[var]`` before applying the norm.  Defaults to the identity
            function. The callable should have the same semantics as :py:meth:`~pyxu.abc.Map.apply`:

              (..., M1,...,MD) -> (..., N1,...,NK)
        norm: Integer, Real
            Ln norm to use >= 0. (Default: L2.)
        satisfy_all: bool
            If True (default) and ``_mstate[var]`` is multi-dimensional, stop if all evaluation points lie below
            threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except Exception:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        self._var = var
        self._rank = int(rank)
        self._f = f if (f is not None) else (lambda _: _)

        try:
            assert norm >= 0
            self._norm = norm
        except Exception:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln norm(s) in stop().

    def stop(self, state: cabc.Mapping) -> bool:
        fx = self._f(state[self._var])  # (..., N1,...,NK)
        self._val = _norm(fx, ord=self._norm, rank=self._rank)  # (..., 1)

        xp = pxu.get_array_module(fx)
        rule = xp.all if self._satisfy_all else xp.any
        decision = rule(self._val <= self._eps)  # (..., 1)

        self._val, decision = pxu.compute(self._val, decision)
        return decision

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"AbsError[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"AbsError[{self._var}]_min": float(self._val.min()),
                f"AbsError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class RelError(pxa.StoppingCriterion):
    """
    Stop iterative solver after relative norm change of a variable (or function thereof) reaches threshold.
    """

    def __init__(
        self,
        eps: pxt.Real,
        var: pxt.VarName = "x",
        rank: pxt.Integer = 1,
        f: SVFunction = None,
        norm: pxt.Real = 2,
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: Real
            Positive threshold.
        var: VarName
            Variable in :py:attr:`pyxu.abc.Solver._mstate` to query.
            Must hold an NDArray
        rank: Integer
            Array rank K of monitored variable **after** applying `f`. (See below.)
        f: ~collections.abc.Callable
            Optional function to pre-apply to ``_mstate[var]`` before applying the norm.  Defaults to the identity
            function. The callable should have the same semantics as :py:meth:`~pyxu.abc.Map.apply`:

              (..., M1,...,MD) -> (..., N1,...,NK)
        norm: Integer, Real
            Ln norm to use >= 0. (Default: L2.)
        satisfy_all: bool
            If True (default) and ``_mstate[var]`` is multi-dimensional, stop if all evaluation points lie below
            threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except Exception:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        self._var = var
        self._rank = int(rank)
        self._f = f if (f is not None) else (lambda _: _)

        try:
            assert norm >= 0
            self._norm = norm
        except Exception:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln rel-norm(s) in stop().
        self._x_prev = None  # buffered var from last query.

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]  # (..., M1,...,MD)

        if self._x_prev is None:
            self._x_prev = x.copy()
            fx_prev = self._f(self._x_prev)  # (..., N1,...,NK)

            # force 1st .info() call to have same format as further calls.
            sh = fx_prev.shape[: -self._rank]
            self._val = np.zeros(shape=(*sh, 1))
            return False  # decision deferred: insufficient history to evaluate rel-err.
        else:
            xp = pxu.get_array_module(x)
            rule = xp.all if self._satisfy_all else xp.any

            fx_prev = self._f(self._x_prev)  # (..., N1,...,NK)
            numerator = _norm(self._f(x) - fx_prev, ord=self._norm, rank=self._rank)
            denominator = _norm(fx_prev, ord=self._norm, rank=self._rank)
            decision = rule(numerator <= self._eps * denominator)  # (..., 1)

            with warnings.catch_warnings():
                # Store relative improvement values for info(). Special care must be taken for the
                # problematic case 0/0 -> NaN.
                warnings.simplefilter("ignore")
                self._val = numerator / denominator  # (..., 1)
                self._val[xp.isnan(self._val)] = 0  # no relative improvement.
            self._x_prev = x.copy()

            self._x_prev, self._val, decision = pxu.compute(self._x_prev, self._val, decision)
            return decision

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"RelError[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"RelError[{self._var}]_min": float(self._val.min()),
                f"RelError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]
        self._x_prev = None
