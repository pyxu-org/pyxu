import numpy as np
from typing import Optional, Tuple
from numbers import Number
from pycsou.core.solver import GenericIterativeAlgorithm
from pycsou.core.map import DifferentiableMap
from pycsou.core.linop import LinearOperator
from pycsou.core.functional import ProximableFunctional
from pycsou.func.base import NullDifferentiableFunctional, NullProximableFunctional


class PMYULA(GenericIterativeAlgorithm):

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 tau: Optional[float] = None, gamma=Optional[None], x0: Optional[np.ndarray] = None,
                 max_iter: int = 1e5, min_iter: int = 100, beta: Optional[float] = None,
                 accuracy_threshold: float = 1e-5, burnin_percentage: float = 1., verbose: Optional[int] = 1,
                 seed: int = 0, linops: Optional[Tuple[LinearOperator, ...]] = None, store_mcmc_samples: bool = False):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.linops = linops
        self.burnin_percentage = burnin_percentage
        self.store_mcmc_samples = store_mcmc_samples
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
                raise ValueError(f'G does not have the proper dimension: {G.dim}!={dim}.')
            else:
                self.G = G
        elif G is None:
            self.G = NullProximableFunctional(dim=dim)
        else:
            raise TypeError(f'G must be of type {ProximableFunctional}.')

        if (tau is not None) and (gamma is not None):
            self.tau, self.gamma = tau, gamma
        else:
            self.tau, self.gamma = self.set_hyperparameters()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_mcmc()

        init_iterand = {'init_mcmc_sample': self.x0}
        super(PMYULA, self).__init__(objective_functional=F + G, init_iterand=init_iterand, max_iter=max_iter,
                                     min_iter=min_iter,
                                     accuracy_threshold=accuracy_threshold, verbose=verbose)

    def set_hyperparameters(self) -> Tuple[float, float]:
        tau = 1 / self.beta
        gamma = tau / 4
        return tau, gamma

    def initialize_mcmc(self) -> np.ndarray:
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def update_iterand(self) -> dict:
        if self.iter == 0:
            x = self.init_iterand['init_mcmc_sample']
            mmse_raw = 0
            mmse_linops = 0
            second_moment_raw = 0
            second_moment_linops = 0
        else:
            x, mmse_raw, mmse_linops, second_moment_raw, second_moment_linops = self.iterand.values()
        z = self.rng.standard_normal(size=self.dim)
        x = (1 - self.gamma / self.tau) * x - self.gamma * self.F.gradient(x) \
            + (self.gamma / self.tau) * self.G.prox(x, tau=self.tau) \
            + np.sqrt(2 * self.gamma) * z
        if self.iter > np.max(np.floor(self.burnin_percentage * self.max_iter), 5):
            mmse_raw += x
            second_moment_raw += x ** 2
            if self.linops is not None:
                for i, linop in enumerate(self.linops):
                    y = linop(x)
                    mmse_linops[i] += y
                    second_moment_linops[i] += y ** 2
        iterand = {'mcmc_sample': x}
        return iterand

    @classmethod
    def p2_algorithm(self, iter, next_sample, marker_heights, marker_positions):
        if iter < 6:
            return marker_heights.append(next_sample), list(np.arange(iter) +1)
        elif iter==5:
            marker_heights.append(next_sample)
            np.sort(marker_positions)
