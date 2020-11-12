from pycsou.linop.sampling import DownSampling
from pycsou.linop.conv import Convolve2D
from pycsou.linop.diff import Laplacian, Gradient
from pycsou.func.loss import L2Loss, SquaredL2Loss, L1Loss, SquaredL1Loss, LInftyLoss, KLDivergence
from pycsou.func.penalty import L2Norm, SquaredL2Norm, L1Norm, SquaredL1Norm, ShannonEntropy, NonNegativeOrthant, \
    LogBarrier
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
plt.savefig(f'examples/exports/deblurring_gt.png', dpi=300, bbox_inches='tight', transparent=True)

# Define Forward Operator
filter_size = 30
filter_support = np.linspace(-3, 3, filter_size)
gaussian_filter = np.exp(-(filter_support[:, None] ** 2 + filter_support[None, :] ** 2) / 2 * (0.5))
FilterOp = Convolve2D(size=input_image.size, filter=gaussian_filter, shape=input_image.shape)
DownSamplingOp = DownSampling(size=input_image.size, downsampling_factor=3, shape=input_image.shape)
FwdOp = DownSamplingOp * FilterOp
FwdOp.compute_lipschitz_cst(tol=1e-2)
data_image = (FwdOp * input_image.flatten())

plt.figure()
plt.imshow(data_image.reshape(DownSamplingOp.output_shape), cmap='viridis')
plt.colorbar()
plt.axis('off')
plt.savefig(f'examples/exports/deblurring_data.png', dpi=300, bbox_inches='tight', transparent=True)

# Cost functionals
l22_loss = SquaredL2Loss(dim=data_image.size, data=data_image)
l22_cost = l22_loss * FwdOp
l1_loss = L1Loss(dim=data_image.size, data=data_image)
kl_divergence = KLDivergence(dim=data_image.size, data=data_image)

# Regularisation functionals
RegOp = Gradient(shape=input_image.shape, dtype='float64')
RegOp.compute_lipschitz_cst(tol=1e-2)
l1_norm = L1Norm(dim=RegOp.shape[0])
l22_norm = SquaredL2Norm(dim=RegOp.shape[0])
entropy = ShannonEntropy(dim=input_image.size)
nneg = NonNegativeOrthant(dim=input_image.size)

# Penalty strength
lmbda = 0.5 * np.max(2 * FwdOp.adjoint(data_image))

# F for each case:
Fs = {'l22_l22': l22_cost + 10 * lmbda * (l22_norm * RegOp),
      'l22_l1': l22_cost,
      'l22_entropy': l22_cost,
      'l1_l22': 100 * lmbda * (l22_norm * RegOp),
      # 'l1_l1': None,
      # 'l1_entropy': None,
      'kl_l22': 10 * lmbda * (l22_norm * RegOp),
      'kl_l1': None,
      # 'kl_entropy': None
      }

Gs = {'l22_l22': None,
      'l22_l1': None,
      'l22_entropy': None,
      'l1_l22': None,
      # 'l1_l1': None,
      # 'l1_entropy': None,
      'kl_l22': None,
      'kl_l1': None,
      # 'kl_entropy': None
      }

Hs = {'l22_l22': None,
      'l22_l1': 0.1 * lmbda * l1_norm,
      'l22_entropy': 0.1 * lmbda * entropy,
      'l1_l22': l1_loss,
      # 'l1_l1': ProxFuncHStack(l1_loss, 1e25 * lmbda * l1_norm),
      # 'l1_entropy': ProxFuncHStack(l1_loss, 1e25 * lmbda * entropy),
      'kl_l22': kl_divergence,
      'kl_l1': ProxFuncHStack(kl_divergence, 0.005 * lmbda * l1_norm),
      # 'kl_entropy': ProxFuncHStack(kl_divergence, 100 * lmbda * entropy)
      }

K_stack1 = LinOpVStack(FwdOp, RegOp)
K_stack1.lipschitz_cst = K_stack1.diff_lipschitz_cst = np.sqrt(
    np.sum(FwdOp.lipschitz_cst ** 2 + RegOp.lipschitz_cst ** 2))

K_stack2 = LinOpVStack(FwdOp, IdentityOperator(size=input_image.size, dtype=np.float64))
K_stack2.lipschitz_cst = K_stack2.diff_lipschitz_cst = np.sqrt(
    np.sum(FwdOp.lipschitz_cst ** 2 + 1))

Ks = {'l22_l22': None,
      'l22_l1': RegOp,
      'l22_entropy': None,
      'l1_l22': FwdOp,
      # 'l1_l1': K_stack1,
      # 'l1_entropy': K_stack2,
      'kl_l22': FwdOp,
      'kl_l1': K_stack1,
      # 'kl_entropy': K_stack2
      }

for key in Fs.keys():
    F, G, H, K = Fs[key], Gs[key], Hs[key], Ks[key]
    PDS = PrimalDualSplitting(dim=input_image.size, F=F, G=G, H=H, K=K, verbose=1, accuracy_threshold=1e-3)
    primal_dual_variables, converged, diagnostics = PDS.iterate()

    plt.figure()
    plt.imshow(primal_dual_variables['primal_variable'].reshape(input_image.shape), cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f'examples/exports/deblurring_{key}.png', dpi=300, bbox_inches='tight', transparent=True)
