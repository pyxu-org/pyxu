import matplotlib.pyplot as plt
import numpy as np

import pycsou.operator.func as pycof
from pycsou.sampler.sampler import MYULA, ULA
from pycsou.sampler.statistics import OnlineMoment, OnlineVariance

f = pycof.SquaredL2Norm(dim=1) / 2
f.diff_lipschitz()
ula = ULA(f=f, gamma=1e-1)  # Sampler of normal distribution (mean 0, variance 1)

n = int(1e4)  # Number of samples
samples_ula = np.zeros(n)
gen_ula = ula.sample(x0=np.zeros(1))
mean_ula = OnlineMoment(order=1)
var_ula = OnlineVariance()
for i in range(n):
    sample = next(gen_ula)
    samples_ula[i] = sample
    mean = mean_ula.update(sample)
    var = var_ula.update(sample)

print(f"Mean of ULA: {mean}, variance: {var}")


plt.figure()
plt.hist(samples_ula, bins=100, density=True)
grid = np.linspace(samples_ula.min(), samples_ula.max(), 1000)
plt.plot(grid, np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi))
plt.show()

# MYULA

myula = MYULA(g=pycof.L1Norm(dim=1), lamb=1e-2, gamma=2 * 1e-2)  # Sampler of Laplace distribution (mean 0, variance 2)

samples_myula = np.zeros(n)
gen_myula = myula.sample(x0=np.zeros(1))
mean_myula = OnlineMoment(order=1)
var_myula = OnlineMoment(order=2)
for i in range(n):
    sample = next(gen_myula)
    samples_myula[i] = sample
    mean = mean_myula.update(sample)
    var = var_myula.update(sample)

print(f"Mean of MYULA: {mean}, variance: {var}")

plt.figure()
plt.hist(samples_myula, bins=100, density=True)
grid_myula = np.linspace(samples_myula.min(), samples_myula.max(), 1000)
plt.plot(grid_myula, np.exp(-np.abs(grid_myula)) / 2)
plt.show()
