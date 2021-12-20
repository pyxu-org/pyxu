import collections.abc as cabc
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
