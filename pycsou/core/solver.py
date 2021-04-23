# #############################################################################
# solver.py
# =========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
This module provides the base class for iterative algorithms.
"""

from pycsou.core.map import Map
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod
from copy import deepcopy


class GenericIterativeAlgorithm(ABC):
    r"""
    Base class for iterative algorithms.

    Any instance/subclass of this class must at least implement the abstract methods ``update_iterand``, ``print_diagnostics``
    ``update_diagnostics`` and ``stopping_metric``.
    """

    def __init__(self, objective_functional: Map, init_iterand: Any, max_iter: int = 500, min_iter: int = 10,
                 accuracy_threshold: float = 1e-3, verbose: Optional[int] = None):
        r"""
        Parameters
        ----------
        objective_functional: Map
            Objective functional to minimise.
        init_iterand: Any
            Initial guess for warm start.
        max_iter: int
            Maximum number of iterations.
        min_iter: int
            Minimum number of iterations.
        accuracy_threshold: float
            Accuracy threshold for stopping criterion.
        verbose: int
            Print diagnostics every ``verbose`` iterations. If ``None`` does not print anything.
        """
        self.objective_functional = objective_functional
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.accuracy_threshold = accuracy_threshold
        self.verbose = verbose
        self.diagnostics = None
        self.iter = 0
        self.iterand = None
        self.init_iterand = init_iterand
        self.converged = False
        super(GenericIterativeAlgorithm, self).__init__()

    def iterate(self) -> Any:
        r"""
        Run the algorithm.

        Returns
        -------
        Any
            Algorithm outcome.
        """
        self.old_iterand = deepcopy(self.init_iterand)
        while ((self.iter <= self.max_iter) and (self.stopping_metric() > self.accuracy_threshold)) or (
                self.iter <= self.min_iter):
            self.iterand = self.update_iterand()
            self.update_diagnostics()
            if self.verbose is not None:
                if self.iter % self.verbose == 0:
                    self.print_diagnostics()
            self.old_iterand = deepcopy(self.iterand)
            self.iter += 1
        self.converged = True
        self.iterand = self.postprocess_iterand()
        return self.iterand, self.converged, self.diagnostics

    def postprocess_iterand(self) -> Any:
        return self.iterand

    def reset(self):
        r"""
        Reset the algorithm.
        """
        self.iter = 0
        self.iterand = None

    def iterates(self, n: int) -> Tuple:
        r"""
        Generator allowing to loop through the n first iterates.

        Useful for debugging/plotting purposes.

        Parameters
        ----------
        n: int
            Max number of iterates to loop through.
        """
        self.reset()
        for i in range(n):
            self.iterand = self.update_iterand()
            self.iter += 1
            yield self.iterand

    @abstractmethod
    def update_iterand(self) -> Any:
        r"""
        Update the iterand.

        Returns
        -------
        Any
            Result of the update.
        """
        pass

    @abstractmethod
    def print_diagnostics(self):
        r"""
        Print diagnostics.
        """
        pass

    @abstractmethod
    def stopping_metric(self):
        r"""
        Stopping metric.
        """
        pass

    @abstractmethod
    def update_diagnostics(self):
        """Update the diagnostics."""
        pass
