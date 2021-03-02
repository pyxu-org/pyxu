import numpy as np
from pandas import DataFrame
from typing import Optional, Tuple
from numbers import Number

from pycsou.core.solver import GenericIterativeAlgorithm
from pycsou.core.map import DifferentiableMap
from pycsou.core.linop import LinearOperator
from pycsou.core.functional import ProximableFunctional
from pycsou.func.base import NullDifferentiableFunctional, NullProximableFunctional
from pycsou.util.stats import P2Algorithm


class PMYULA(GenericIterativeAlgorithm):

    def __init__(self, dim: int, F: Optional[DifferentiableMap] = None, G: Optional[ProximableFunctional] = None,
                 tau: Optional[float] = None, gamma: Optional[None] = None, x0: Optional[np.ndarray] = None,
                 max_iter: int = 1e5, min_iter: int = 200, beta: Optional[float] = None,
                 accuracy_threshold: float = 1e-5, nb_burnin_iterations: int = 1000, thinning_factor: int = 100,
                 verbose: Optional[int] = 1, seed: int = 0, linops: Optional[Tuple[LinearOperator, ...]] = None,
                 store_mcmc_samples: bool = False,
                 pvalues: Optional[Tuple[float, ...]] = (0.10, 0.9)):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.linops = linops
        self.nb_burnin_iterations = nb_burnin_iterations
        self.thinning_factor = thinning_factor
        self.store_mcmc_samples = store_mcmc_samples
        self.mcmc_samples = []
        self.pvalues = pvalues
        self.count = 0
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
        elif tau is not None:
            self.tau = tau
            self.gamma = tau / ((self.F.diff_lipschitz_cst * tau + 1))
        else:
            self.tau, self.gamma = self.set_hyperparameters()

        if x0 is not None:
            self.x0 = np.asarray(x0)
        else:
            self.x0 = self.initialize_mcmc()

        init_iterand = {'init_mcmc_sample': self.x0, 'mmse_raw': self.x0}
        super(PMYULA, self).__init__(objective_functional=self.F + self.G, init_iterand=init_iterand, max_iter=max_iter,
                                     min_iter=min_iter,
                                     accuracy_threshold=accuracy_threshold, verbose=verbose)

    def set_hyperparameters(self) -> Tuple[float, float]:
        if isinstance(self.G, NullProximableFunctional):
            tau = None
            gamma = 1 / self.beta
        else:
            tau = 2 / self.beta
            gamma = tau / ((self.beta * tau + 1))
        return tau, gamma

    def initialize_mcmc(self) -> np.ndarray:
        return np.zeros(shape=(self.dim,), dtype=np.float)

    def update_iterand(self) -> dict:
        if self.iter == 0:
            x = self.init_iterand['init_mcmc_sample']
            mmse_raw = 0
            second_moment_raw = 0
            if self.pvalues is None:
                p2_raw = None
            else:
                p2_raw = [P2Algorithm(pvalue=p) for p in self.pvalues]
            if self.linops is not None:
                mmse_linops = [0 for _ in self.linops]
                second_moment_linops = [0 for _ in self.linops]
                if self.pvalues is None:
                    p2_linops = [None for _ in self.linops]
                else:
                    p2_linops = [[P2Algorithm(pvalue=p) for p in self.pvalues] for _ in self.linops]
        else:
            if self.linops is None:
                x, mmse_raw, second_moment_raw, p2_raw = self.iterand.values()
            else:
                x, mmse_raw, second_moment_raw, p2_raw, mmse_linops, second_moment_linops, p2_linops = self.iterand.values()

        z = rng.standard_normal(size=self.dim)
        if isinstance(self.G, NullProximableFunctional):
            x = x - self.gamma * self.F.gradient(x) + np.sqrt(2 * self.gamma) * z
        else:
            x = (1 - self.gamma / self.tau) * x - self.gamma * self.F.gradient(x) \
                + (self.gamma / self.tau) * self.G.prox(x, tau=self.tau) \
                + np.sqrt(2 * self.gamma) * z

        if self.store_mcmc_samples:
            self.mcmc_samples.append(x)

        if self.iter > np.fmax(self.nb_burnin_iterations, 4):
            if (self.iter - self.nb_burnin_iterations) % self.thinning_factor == 0:
                self.count += 1
                mmse_raw += x
                second_moment_raw += x ** 2
                if self.pvalues is not None:
                    for pp in p2_raw: pp.add_sample(x)

                if self.linops is not None:
                    for i, linop in enumerate(self.linops):
                        y = linop(x)
                        mmse_linops[i] += y
                        second_moment_linops[i] += y ** 2
                        if self.pvalues is not None:
                            for pp in p2_linops[i]: pp.add_sample(y)

        if self.linops is None:
            iterand = {'mcmc_sample': x, 'mmse_raw': mmse_raw, 'second_moment_raw': second_moment_raw, 'p2_raw': p2_raw}
        else:
            iterand = {'mcmc_sample': x, 'mmse_raw': mmse_raw, 'second_moment_raw': second_moment_raw, 'p2_raw': p2_raw,
                       'mmse_linops': mmse_linops, 'second_moment_linops': second_moment_linops, 'p2_linops': p2_linops}
        return iterand

    def postprocess_iterand(self) -> dict:
        if self.linops is None:
            mmse_raw = self.iterand['mmse_raw'] / self.count
            second_moment_raw = self.iterand['second_moment_raw'] / self.count
            std_raw = np.sqrt(second_moment_raw - mmse_raw ** 2)
            if self.pvalues is not None:
                quantiles_raw = [pp.q for pp in self.iterand['p2_raw']]
            else:
                quantiles_raw = None
            iterand = {'mcmc_sample': self.iterand['mcmc_sample'], 'mmse': mmse_raw, 'std': std_raw,
                       'quantiles': quantiles_raw, 'pvalues': self.pvalues}

        else:
            mmse_raw = self.iterand['mmse_raw'] / self.count
            second_moment_raw = self.iterand['second_moment_raw'] / self.count
            std_raw = np.sqrt(second_moment_raw - mmse_raw ** 2)
            if self.pvalues is not None:
                quantiles_raw = [pp.q for pp in self.iterand['p2_raw']]
            else:
                quantiles_raw = None
            mmse_linops = []
            std_linops = []
            quantiles_linops = []
            for i, linop in enumerate(self.linops):
                mmse_linop = self.iterand['mmse_linops'][i] / self.count
                second_moment_linop = self.iterand['second_moment_linops'][i] / self.count
                std_linop = np.sqrt(second_moment_linop - mmse_linop ** 2)
                if self.pvalues is not None:
                    quantiles_linop = [pp.q for pp in self.iterand['p2_linops'][i]]
                else:
                    quantiles_linop = None
                mmse_linops.append(mmse_linop)
                std_linops.append(std_linop)
                quantiles_linops.append(quantiles_linop)
            iterand = {'mcmc_sample': self.iterand['mcmc_sample'], 'mmse_raw': mmse_raw, 'std_raw': std_raw,
                       'quantiles_raw': quantiles_raw,
                       'mmse_linops': mmse_linops, 'std_linops': std_linops, 'quantiles_linops': quantiles_linops,
                       'pvalues': self.pvalues}
        return iterand

    def stopping_metric(self):
        if self.iter == 0:
            return np.infty
        elif (self.iter - self.nb_burnin_iterations) % self.thinning_factor != 0:
            return np.infty
        else:
            return np.infty # self.diagnostics.loc[self.iter - 1, 'Relative Improvement (MMSE)']

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))

    def update_diagnostics(self):
        if self.iter == 0:
            self.diagnostics = DataFrame(
                columns=['Iter', 'Relative Improvement (MMSE)', 'Nb of samples'])
        self.diagnostics.loc[self.iter, 'Iter'] = self.iter
        self.diagnostics.loc[self.iter, 'Nb of samples'] = self.count
        if np.linalg.norm(self.old_iterand['mmse_raw']) == 0:
            self.diagnostics.loc[self.iter, 'Relative Improvement (MMSE)'] = np.infty
        else:
            self.diagnostics.loc[self.iter, 'Relative Improvement (MMSE)'] = np.linalg.norm(
                self.old_iterand['mmse_raw'] - self.iterand['mmse_raw']) / np.linalg.norm(
                self.old_iterand['mmse_raw'])


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pycsou.func.loss import SquaredL2Loss
    from pycsou.func.penalty import NonNegativeOrthant, L1Norm, SquaredL2Norm, Segment, L2Ball, L1Ball
    from pycsou.linop import Integration1D, MovingAverage1D, DownSampling
    from pycsou.opt.proxalgs import APGD

    rng = np.random.default_rng(0)
    N = 32
    x = np.repeat([0, 2, 1, 3, 0, 2, 0], N)
    S = DownSampling(size=x.size, downsampling_factor=4) * MovingAverage1D(window_size=8, shape=(x.size,))
    S.compute_lipschitz_cst()
    I = Integration1D(size=x.size)
    I.compute_lipschitz_cst()
    Gop = S * I
    y = S(x) + 0.05 * rng.standard_normal(size=S.shape[0])
    F = (1 / 2) * SquaredL2Loss(dim=S.shape[0], data=y) * Gop
    lam = 0.1 * np.max(np.abs(F.gradient(0 * x)))
    G = L2Ball(dim=x.size, radius=2)
    apgd = APGD(dim=x.size, F=F, G=G, min_iter=100, max_iter=1e4, accuracy_threshold=1e-4, verbose=1)
    out1, _, _ = apgd.iterate()
    pmyula = PMYULA(dim=x.size, F=F, G=G, max_iter=3e4, x0=out1['iterand'] + 0.05 * rng.standard_normal(x.size),
                    accuracy_threshold=1e-7, pvalues=(0.05, 0.95), tau=1e-4,
                    verbose=100, nb_burnin_iterations=200, thinning_factor=10, linops=(I,))
    out2, _, _ = pmyula.iterate()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(x)
    plt.title('Original Signal')
    plt.subplot(1, 3, 2)
    plt.plot(y)
    plt.title('Data')
    plt.subplot(1, 3, 3)
    plt.plot(I * out1['iterand'])
    plt.plot(out2['mmse_linops'][0])
    plt.fill_between(np.arange(x.size), out2['quantiles_linops'][0][0], out2['quantiles_linops'][0][1], alpha=0.3)
    plt.legend(['MAP', 'MMSE', '90% Credibility Intervals'])
