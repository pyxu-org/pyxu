from pycsou.linop.sampling import Masking, Pooling, DownSampling
from pycsou.linop.diff import Laplacian, Gradient, GeneralisedLaplacian
from pycsou.linop.conv import Convolve2D
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import SquaredL2Norm, NonNegativeOrthant
from pycsou.core.solver import PrimalDualSplitting
from pycsou.core.functional import ProxFuncHStack
from pycsou.core.linop import LinOpVStack, IdentityOperator
import numpy as np
import scipy.misc as sp_misc
import matplotlib.pyplot as plt

# Input Image
input_image = sp_misc.face(gray=True).astype(np.float64)
plt.figure()
plt.imshow(input_image, cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.savefig(f'examples/exports/unpooling_gt.png', dpi=300, bbox_inches='tight', transparent=True)

# Define Forward Operator
# sampling_bool = np.random.binomial(n=1, p=0.2, size=input_image.shape)
# FwdOp = Masking(size=input_image.size, sampling_bool=sampling_bool, dtype=np.float64)
# FwdOp = Pooling(shape=input_image.shape, block_size=(9, 9), dtype=np.float64)
filter_size = 30
filter_support = np.linspace(-3, 3, filter_size)
gaussian_filter = np.exp(-(filter_support[:, None] ** 2 + filter_support[None, :] ** 2) / 2 * (0.5))
FilterOp = Convolve2D(size=input_image.size, filter=gaussian_filter, shape=input_image.shape)
DownSamplingOp = DownSampling(size=input_image.size, downsampling_factor=3, shape=input_image.shape)
FwdOp = DownSamplingOp * FilterOp

FwdOp.compute_lipschitz_cst(tol=1e-2)
data_image = (FwdOp * input_image.flatten())

plt.figure()
plt.imshow(FwdOp.adjoint(data_image).reshape(input_image.shape), cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.savefig(f'examples/exports/unpooling_data.png', dpi=300, bbox_inches='tight', transparent=True)

# Cost functionals
l22_loss = SquaredL2Loss(dim=data_image.size, data=data_image)
l22_cost = l22_loss * FwdOp

# Regularisation functionals
RegOp0 = None
RegOp1 = Gradient(shape=input_image.shape, dtype='float64')
RegOp1.compute_lipschitz_cst(tol=1e-2)
RegOp2 = Laplacian(shape=input_image.shape, dtype='float64')
RegOp2.compute_lipschitz_cst(tol=1e-2)
RegOps = {'identity': RegOp0, 'grad': RegOp1, 'laplacian': RegOp2}

for key, RegOp in RegOps.items():

    if RegOp is None:
        l22_norm = SquaredL2Norm(dim=input_image.size)
        lmbda = (15 * FwdOp.lipschitz_cst)
        l22_reg = lmbda * l22_norm
    else:
        l22_norm = SquaredL2Norm(dim=RegOp.shape[0])
        lmbda = 10 * FwdOp.lipschitz_cst / RegOp.lipschitz_cst
        l22_reg = lmbda * (l22_norm * RegOp)
    F = l22_cost + l22_reg
    PDS = PrimalDualSplitting(dim=input_image.size, F=F, G=NonNegativeOrthant(input_image.size), H=None, K=None,
                              verbose=1, accuracy_threshold=1e-3)
    primal_dual_variables, converged, diagnostics = PDS.iterate()

    plt.figure()
    plt.imshow(primal_dual_variables['primal_variable'].reshape(input_image.shape), cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'examples/exports/unpooling_{key}.png', dpi=300, bbox_inches='tight', transparent=True)
