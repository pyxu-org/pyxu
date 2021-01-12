import numpy as np
import matplotlib.pyplot as plt
from pycsou.linop.sampling import Masking
from pycsou.linop.diff import Gradient
from pycsou.func.penalty import L1Norm, SquaredL2Norm, Segment
from pycsou.func.loss import SquaredL2Loss, L1Loss
from pycsou.core.solver import PrimalDualSplitting
from pycsou.core.linop import LinOpHStack
from pycsou.core.functional import ProxFuncHStack
from pycsou.linop.sampling import MappedDistanceMatrix

# Input signal:
x = np.linspace(-5, 15, 1000)
aperiodic = lambda t: t * np.sin(t) + t * np.cos(t)
scale = 10
periodic = lambda t: np.sin(scale * t) + np.sin(scale * 2 * t / 3)
period = 6 * np.pi / scale
x_per = np.linspace(0, period, 1000)
per2aper_ratio = 20
weight = np.linalg.norm(aperiodic(x)) * 10 ** (-per2aper_ratio / 10)
per_aper = lambda t: aperiodic(t) + weight * periodic(t)

plt.figure()
plt.plot(x, per_aper(x))

# Data
L = 64
rng = np.random.default_rng(seed=1)
p = (np.max(x) - np.min(x)) * rng.random(size=L) + np.min(x)
samples = per_aper(p)
plt.scatter(p, samples)

p_periodised = p % period
samples_aper = aperiodic(p)
samples_per = periodic(p_periodised)
plt.figure()
plt.plot(x, aperiodic(x))
plt.scatter(p, samples_aper)
plt.figure()
plt.plot(x_per, periodic(x_per))
plt.scatter(p_periodised, samples_per)
np.allclose(samples_aper + weight * samples_per, samples)

# Green kernels
N_per = 32
sigma_per = np.sqrt(2 - 2 * np.cos(2 * np.pi * (period / N_per) / period))
r = lambda t: np.sqrt(2 - 2 * np.cos(2 * np.pi * t / period))
kernel_per = lambda t: np.exp(- r(t) ** 2 / (2 * sigma_per ** 2))
knots_per = np.linspace(start=0, stop=period, num=N_per, endpoint=False)
N_aper = 32
sigma_aper = (np.max(x) - np.min(x)) / N_aper
kernel_aper = lambda t: np.exp(-t ** 2 / (2 * sigma_aper ** 2))
knots_aper = np.linspace(start=np.min(x), stop=np.max(x), num=N_aper, endpoint=True)

G_per = MappedDistanceMatrix(samples1=p_periodised, function=kernel_per, samples2=knots_per, mode='radial',
                             max_distance=5 * sigma_per, operator_type='dense')
G_aper = MappedDistanceMatrix(samples1=p, function=kernel_aper, samples2=knots_aper, mode='radial',
                              max_distance=5 * sigma_aper, operator_type='dense')
K = LinOpHStack(G_per, G_aper)
K.compute_lipschitz_cst(tol=1e-2)

plt.figure()
plt.imshow(G_per.mat)
plt.figure()
plt.imshow(G_aper.mat)

Psi_per = MappedDistanceMatrix(samples1=x_per, function=kernel_per, samples2=knots_per, mode='radial',
                               max_distance=5 * sigma_per, operator_type='dense')
Psi_per2 = MappedDistanceMatrix(samples1=x, function=kernel_per, samples2=knots_per, mode='radial',
                               max_distance=5 * sigma_per, operator_type='dense')
Psi_aper = MappedDistanceMatrix(samples1=x, function=kernel_aper, samples2=knots_aper, mode='radial',
                                max_distance=5 * sigma_aper, operator_type='dense')

plt.figure()
plt.plot(x_per, Psi_per.mat)
plt.figure()
plt.plot(x, Psi_aper.mat)

l22_loss = SquaredL2Loss(dim=K.shape[0], data=samples)
l22_cost = l22_loss * K
mu = 0.1 * np.max(l22_cost.gradient(np.zeros(K.shape[1])))
l1_norm = mu * L1Norm(dim=K.shape[1])
PDS = PrimalDualSplitting(dim=K.shape[1], F=l22_cost, G=l1_norm, rho=0.9)
PDS.iterate()

alpha_per, alpha_aper=PDS.iterand['primal_variable'][:N_per],PDS.iterand['primal_variable'][N_per:]
plt.figure()
plt.plot(x_per, periodic(x_per))
plt.scatter(p_periodised, samples_per)
plt.plot(x_per, Psi_per(alpha_per))

plt.figure()
plt.plot(x, aperiodic(x))
plt.scatter(p, samples_aper)
plt.plot(x, Psi_aper(alpha_aper))

plt.figure()
plt.plot(x, per_aper(x))
plt.scatter(p, samples)
plt.plot(x, Psi_per2(alpha_per) + Psi_aper(alpha_aper))