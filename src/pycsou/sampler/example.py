import matplotlib.pyplot as plt
import numpy as np

import pycsou.operator.func as pycof
from pycsou.sampler.sampler import ULA
from pycsou.sampler.statistics import OnlineMoment, OnlineVariance

f = pycof.SquaredL2Norm(dim=1) / 2  # To sample 1D normal distribution (mean 0, variance 1)
ula = ULA(f=f)  # Sampler with maximum step size
ula_lb = ULA(f=f, gamma=1e-1)  # Sampler with small step size

gen_ula = ula.samples(x0=np.zeros(1))
gen_ula_lb = ula_lb.samples(x0=np.zeros(1))
n_burn_in = int(1e3)  # Number of burn-in iterations
for i in range(n_burn_in):
    next(gen_ula)
    next(gen_ula_lb)

# Online statistics objects
mean_ula = OnlineMoment(order=1)
mean_ula_lb = OnlineMoment(order=1)
var_ula = OnlineVariance()
var_ula_lb = OnlineVariance()

n = int(1e4)  # Number of samples
samples_ula = np.zeros(n)
samples_ula_lb = np.zeros(n)
for i in range(n):
    sample = next(gen_ula)
    sample_lb = next(gen_ula_lb)
    samples_ula[i] = sample
    samples_ula_lb[i] = sample_lb
    mean = float(mean_ula.update(sample))
    var = float(var_ula.update(sample))
    mean_lb = float(mean_ula_lb.update(sample_lb))
    var_lb = float(var_ula_lb.update(sample_lb))

# Theoretical variances of biased stationary distributions of ULA
biased_var = 1 / (1 - ula._gamma / 2)
biased_var_lb = 1 / (1 - ula_lb._gamma / 2)

# Plots
grid = np.linspace(-4, 4, 1000)

plt.figure()
plt.title(
    f"ULA samples (large step size) \n Empirical mean: {mean:.3f} (theoretical: 0) \n "
    f"Empirical variance: {var:.3f} (theoretical: {biased_var:.3f})"
)
plt.hist(samples_ula, range=(min(grid), max(grid)), bins=100, density=True)
plt.plot(grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi), label=r"$p(x)$")
plt.plot(grid, np.exp(-(grid**2) / (2 * biased_var)) / np.sqrt(2 * np.pi * biased_var), label=r"$p_{\gamma_1}(x)$")
plt.legend()
plt.show()

plt.figure()
plt.title(
    f"ULA samples (small step size) \n Empirical mean: {mean_lb:.3f} (theoretical: 0) \n "
    f"Empirical variance: {var_lb:.3f} (theoretical: {biased_var_lb:.3f})"
)
plt.hist(samples_ula_lb, range=(min(grid), max(grid)), bins=100, density=True)
plt.plot(grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi), label=r"$p(x)$")
plt.plot(
    grid, np.exp(-(grid**2) / (2 * biased_var_lb)) / np.sqrt(2 * np.pi * biased_var_lb), label=r"$p_{\gamma_2}(x)$"
)
plt.legend()
plt.show()

# MYULA

# myula = MYULA(g=pycof.L1Norm(dim=1), lamb=1e-2, gamma=2 * 1e-2)  # Sampler of Laplace distribution (mean 0, variance 2)
#
# samples_myula = np.zeros(n)
# gen_myula = myula.sample(x0=np.zeros(1))
# mean_myula = OnlineMoment(order=1)
# var_myula = OnlineMoment(order=2)
# for i in range(n):
#     sample = next(gen_myula)
#     samples_myula[i] = sample
#     mean = mean_myula.update(sample)
#     var = var_myula.update(sample)
#
# print(f"Mean of MYULA: {mean}, variance: {var}")
#
# plt.figure()
# plt.hist(samples_myula, bins=100, density=True)
# grid_myula = np.linspace(samples_myula.min(), samples_myula.max(), 1000)
# plt.plot(grid_myula, np.exp(-np.abs(grid_myula)) / 2)
# plt.show()
