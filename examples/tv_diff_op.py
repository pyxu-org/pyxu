from pycsou.linop.sampling import Masking
from pycsou.linop.diff import Laplacian, Gradient, GeneralisedLaplacian
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm, NonNegativeOrthant
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
plt.savefig(f'examples/exports/inpainting_gt.png', dpi=300, bbox_inches='tight', transparent=True)

# Define Forward Operator
sampling_bool = np.random.binomial(n=1, p=0.2, size=input_image.shape)
FwdOp = Masking(size=input_image.size, sampling_bool=sampling_bool, dtype=np.float64)
FwdOp.compute_lipschitz_cst(tol=1e-2)
data_image = (FwdOp * input_image.flatten())

plt.figure()
plt.imshow(FwdOp.adjoint(data_image).reshape(input_image.shape), cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.savefig(f'examples/exports/inpainting_data.png', dpi=300, bbox_inches='tight', transparent=True)

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
        l1_norm = L1Norm(dim=input_image.size)
        lmbda = (15 * FwdOp.lipschitz_cst)
        l1_reg = lmbda * l1_norm
    else:
        l1_norm = L1Norm(dim=RegOp.shape[0])
        lmbda = 10 * FwdOp.lipschitz_cst / RegOp.lipschitz_cst
        l1_reg = lmbda * l1_norm
    F = l22_cost
    PDS = PrimalDualSplitting(dim=input_image.size, F=F, G=NonNegativeOrthant(input_image.size), H=l1_reg, K=RegOp,
                              verbose=1, accuracy_threshold=1e-3)
    primal_dual_variables, converged, diagnostics = PDS.iterate()

    plt.figure()
    plt.imshow(primal_dual_variables['primal_variable'].reshape(input_image.shape), cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'examples/exports/inpainting_{key}.png', dpi=300, bbox_inches='tight', transparent=True)
