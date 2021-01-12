import numpy as np
from pandas import DataFrame
from pycsou.core.map import DifferentiableMap
from pycsou.core.functional import DifferentiableFunctional, ProximableFunctional, NullDifferentiableFunctional, \
    NullProximableFunctional
from pycsou.core.linop import LinearOperator, IdentityOperator, NullOperator
from pycsou.core.map import Map
from typing import Optional, Tuple, Dict, Callable, Any
from numbers import Number
from abc import ABC, abstractmethod


class GenericIterativeAlgorithm(ABC):

    def __init__(self, objective_functional: Map, init_iterand: Any, max_iter: int = 500, min_iter: int = 10,
                 accuracy_threshold: float = 1e-3, verbose: int = 1):
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
        self.old_iterand = self.init_iterand
        while ((self.iter <= self.max_iter) and (self.stopping_metric() > self.accuracy_threshold)) or (
                self.iter <= self.min_iter):
            self.iterand = self.update_iterand()
            self.update_diagnostics()
            if self.iter % self.verbose == 0:
                self.print_diagnostics()
            self.old_iterand = self.iterand
            self.iter += 1
        self.converged = True
        return self.iterand, self.converged, self.diagnostics

    def reset(self):
        self.iter = 0
        self.iterand = None

    def iterates(self, n: int) -> Tuple:
        self.reset()
        for i in range(n):
            self.iterand = self.update_iterand()
            self.iter += 1
            yield self.iterand

    @abstractmethod
    def update_iterand(self) -> Any:
        pass

    @abstractmethod
    def print_diagnostics(self):
        pass

    @abstractmethod
    def stopping_metric(self):
        pass

    @abstractmethod
    def update_diagnostics(self):
        pass


class PrimalDualSplitting(GenericIterativeAlgorithm):

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 H: Optional[ProximableFunctional] = None, K: Optional[LinearOperator] = None,
                 tau: Optional[float] = None, sigma: Optional[float] = None, rho: Optional[float] = None,
                 beta: Optional[float] = None, x0: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
                 max_iter: int = 500, min_iter: int = 10, accuracy_threshold: float = 1e-3, verbose: int = 1,
                 store_diagnostics: bool = True):
        self.dim = dim
        self._H = True
        if isinstance(F, DifferentiableMap):
            if F.shape[1] != dim:
                raise ValueError(f'F does not have the proper dimension: {F.shape[1]}!={dim}.')
            else:
                self.F = F
            if F.diff_lipschitz_cst < np.infty:
                self.beta = self.F.diff_lipschitz_cst if beta is None else beta
            elif (beta is not None) and isinstance(beta, Number):
                self.beta = beta
            else:
                raise ValueError('F must be a differentiable functional with Lipschitz-continuous gradient.')
        elif F is None:
            self.F = NullDifferentiableFunctional(dim=dim)
            self.beta = 0
        else:
            raise TypeError(f'F must be of type {DifferentiableMap}.')

        if isinstance(G, ProximableFunctional):
            if G.dim != dim:
                raise ValueError(f'F does not have the proper dimension: {G.dim}!={dim}.')
            else:
                self.G = G
        elif G is None:
            self.G = NullProximableFunctional(dim=dim)
        else:
            raise TypeError(f'F must be of type {ProximableFunctional}.')

        if isinstance(K, LinearOperator) and isinstance(H, ProximableFunctional):
            if (K.shape[1] != dim) or (K.shape[0] != H.dim):
                raise ValueError(
                    f'Operator K with shape {K.shape} is inconsistent with functional H with dimension {H.dim}.')

        if isinstance(H, ProximableFunctional):
            self.H = H
            if isinstance(K, LinearOperator):
                self.K = K
            elif K is None:
                self.K = IdentityOperator(size=H.dim)
                self.K.lipschitz_cst = self.K.diff_lipschitz_cst = 1
            else:
                raise TypeError(f'K must be of type {LinearOperator}.')
        elif H is None:
            self.H = NullProximableFunctional(dim=dim)
            self._H = False
            self.K = NullOperator(shape=(dim, dim))
            self.K.lipschitz_cst = self.K.diff_lipschitz_cst = 0
        else:
            raise TypeError(f'H must be of type {ProximableFunctional}.')

        if (tau is not None) and (sigma is not None):
            self.tau, self.sigma = tau, sigma
        elif (tau is not None) and (sigma is None):
            self.tau = self.sigma = tau
        elif (tau is None) and (sigma is not None):
            self.tau = self.sigma = sigma
        else:
            self.tau, self.sigma = self.set_step_sizes()

        if rho is not None:
            self.rho = rho
        else:
            self.rho = self.set_momentum_term()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_primal_variable()

        if z0 is not None:
            self.z0 = np.asarray(z0)
        else:
            self.z0 = self.initialize_dual_variable()

        objective_functional = (self.F + self.G) + (self.H * self.K)
        init_iterand = {'primal_variable': self.x0, 'dual_variable': self.z0}
        super(PrimalDualSplitting, self).__init__(objective_functional=objective_functional, init_iterand=init_iterand,
                                                  max_iter=max_iter, min_iter=min_iter,
                                                  accuracy_threshold=accuracy_threshold, verbose=verbose)

    def set_step_sizes(self) -> Tuple[float, float]:
        if self.beta > 0:
            if self._H is False:
                tau = 2 / self.beta
                sigma = 0
            else:
                if self.K.lipschitz_cst < np.infty:
                    tau = sigma = (1 / (self.K.lipschitz_cst) ** 2) * (
                            (-self.beta / 4) + np.sqrt((self.beta ** 2 / 16) + self.K.lipschitz_cst ** 2))
                else:
                    raise ValueError(
                        'Please compute the Lipschitz constant of the linear operator K by calling its method "compute_lipschitz_cst()".')
        else:
            if self._H is False:
                tau = 1
                sigma = 0
            else:
                if self.K.lipschitz_cst < np.infty:
                    tau = sigma = 1 / self.K.lipschitz_cst
                else:
                    raise ValueError(
                        'Please compute the Lipschitz constant of the linear operator K by calling its method "compute_lipschitz_cst()".')
        return tau, sigma

    def set_momentum_term(self) -> float:
        if self.beta > 0:
            rho = 0.4
        else:
            rho = 1
        return rho

    def initialize_primal_variable(self) -> np.ndarray:
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def initialize_dual_variable(self) -> Optional[np.ndarray]:
        if self._H is False:
            return None
        else:
            return np.zeros(shape=(self.H.dim,), dtype=np.float)

    def update_iterand(self) -> Any:
        if self.iter == 0:
            x, z = self.init_iterand.values()
        else:
            x, z = self.iterand.values()
        x_temp = self.G.prox(x - self.tau * self.F.gradient(x) - self.tau * self.K.adjoint(z), tau=self.tau)
        if self._H is True:
            u = 2 * x_temp - x
            z_temp = self.H.fenchel_prox(z + self.sigma * self.K(u), sigma=self.sigma)
            z = self.rho * z_temp + (1 - self.rho) * z
        x = self.rho * x_temp + (1 - self.rho) * x
        iterand = {'primal_variable': x, 'dual_variable': z}
        return iterand

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        else:
            return self.diagnostics.loc[self.iter - 1, 'Relative Improvement (primal variable)']

    def update_diagnostics(self):
        if self._H is True:
            if self.iter == 0:
                self.diagnostics = DataFrame(
                    columns=['Iter', 'Relative Improvement (primal variable)', 'Relative Improvement (dual variable)'])
            self.diagnostics.loc[self.iter, 'Iter'] = self.iter
            self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.linalg.norm(
                self.old_iterand['primal_variable'] - self.iterand['primal_variable']) / np.linalg.norm(
                self.old_iterand['primal_variable'])
            self.diagnostics.loc[self.iter, 'Relative Improvement (dual variable)'] = np.linalg.norm(
                self.old_iterand['dual_variable'] - self.iterand['dual_variable']) / np.linalg.norm(
                self.old_iterand['dual_variable'])
        else:
            if self.iter == 0:
                self.diagnostics = DataFrame(
                    columns=['Iter', 'Relative Improvement (primal variable)'])
            self.diagnostics.loc[self.iter, 'Iter'] = self.iter
            self.diagnostics.loc[self.iter, 'Relative Improvement (primal variable)'] = np.linalg.norm(
                self.old_iterand['primal_variable'] - self.iterand['primal_variable']) / np.linalg.norm(
                self.old_iterand['primal_variable'])