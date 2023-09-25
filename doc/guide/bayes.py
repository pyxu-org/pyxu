import pyxu.abc as pxa
import pyxu.operator as pxo
import numpy as np
from pyxu.operator import Gradient, SquaredL2Norm, L1Norm, PositiveL1Norm
from PIL import Image

# Load the data
toucan = np.array(Image.open("../_static/favicon.png").convert("L"))
toucan = toucan.astype(float)
toucan /= toucan.max()

# Create noisy data
sigma = 0.2
Phi = pxo.IdentityOp(toucan.size) # This is not really necessary, but we will use it for consistency
data = Phi(toucan) + sigma * np.random.randn(*toucan.shape)
data = np.clip(data, 0, 1)
# TV prior
grad = Gradient(arg_shape=toucan.shape, accuracy=4,)
lambda_= 1000 / (2 * sigma**2)
l1_norm = lambda_ * L1Norm(grad.shape[0])

out = grad.unravel(grad(toucan.ravel()))

import matplotlib.pyplot as plt
plt.subplot(211)
plt.imshow(out[0])
plt.subplot(212)
plt.imshow(out[1])