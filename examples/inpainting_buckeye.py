import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from pycsou.linop.sampling import Masking
from pycsou.linop.diff import Gradient
from pycsou.func.penalty import L1Norm, SquaredL2Norm, Segment
from pycsou.func.loss import SquaredL2Loss, L1Loss
from pycsou.core.solver import PrimalDualSplitting
from pycsou.core.linop import LinOpVStack
from pycsou.core.functional import ProxFuncHStack

plt.rcParams["image.cmap"] = "cividis"
plt.rcParams["figure.figsize"] = [9, 6]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = False

rng = np.random.default_rng(seed=0)

input_image = skio.imread("examples/buckeye.jpg", as_gray=True).astype(np.float64)
plt.imshow(input_image)
plt.colorbar()

bool_mask = (
    rng.binomial(1, p=0.25, size=input_image.size)
        .astype(bool)
        .reshape(input_image.shape)
)

G = Masking(size=input_image.size, sampling_bool=bool_mask, dtype=np.float64)
G.compute_lipschitz_cst()
G.norm = G.lipschitz_cst
print(G.norm)

y = G(input_image.flatten())
y[rng.binomial(n=1, p=0.02, size=y.size).astype(bool)] = 1.0
upsampled_image = G.adjoint(y).reshape(input_image.shape)
plt.figure()
plt.imshow(upsampled_image)
plt.colorbar()

D = Gradient(shape=input_image.shape, edge=True, dtype='float64')
D.compute_lipschitz_cst(tol=1e-2)
D.norm = D.lipschitz_cst
mu = 0.1 * (G.norm ** 2 / D.norm ** 2)
l2_loss = (1 / 2) * SquaredL2Loss(dim=y.size, data=y)
l2_cost = l2_loss * G
l22_norm = SquaredL2Norm(dim=D.shape[0])
tikhonov_penalty = (mu / 2) * l22_norm * D
range = Segment(dim=input_image.size, a=0, b=1)
PDS = PrimalDualSplitting(dim=input_image.size, F=l2_cost + tikhonov_penalty, G=range, rho=1)
PDS.iterate()
plt.figure()
plt.imshow(PDS.iterand['primal_variable'].reshape(input_image.shape))

mu = 0.035 * np.max(D(G.adjoint(y)))
l1_norm = L1Norm(dim=D.shape[0])
PDS = PrimalDualSplitting(dim=input_image.size, F=l2_cost, G=range, H=mu * l1_norm, K=D, rho=0.9)
PDS.iterate()
plt.figure()
plt.imshow(PDS.iterand['primal_variable'].reshape(input_image.shape))

l1_loss = L1Loss(dim=y.size, data=y)
H = ProxFuncHStack(l1_loss, 0.6 * l1_norm)
K=LinOpVStack(G, D)
K.compute_lipschitz_cst(tol=1e-2)
PDS = PrimalDualSplitting(dim=input_image.size, F=None, G=range, H=H, K=K, rho=0.9)
PDS.iterate()
plt.figure()
plt.imshow(PDS.iterand['primal_variable'].reshape(input_image.shape))

surprise_image = skio.imread("examples/epfl.jpg", as_gray=True).astype(np.float64)
y = G(surprise_image.flatten())
y[rng.binomial(n=1, p=0.02, size=y.size).astype(bool)] = 1.0
l1_loss = L1Loss(dim=y.size, data=y)
H = ProxFuncHStack(l1_loss, 0.6 * l1_norm)
K=LinOpVStack(G, D)
K.compute_lipschitz_cst(tol=1e-2)
PDS = PrimalDualSplitting(dim=input_image.size, F=None, G=range, H=H, K=K, rho=0.9)
PDS.iterate()
plt.figure()
plt.imshow(PDS.iterand['primal_variable'].reshape(input_image.shape))
