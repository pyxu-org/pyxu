from pycsou.linop.conv import Convolve2D
from pycsou.linop.sampling import DownSampling, Pooling, Masking
import numpy as np
import scipy.misc as sp_misc
import matplotlib.pyplot as plt

input_image = sp_misc.face(gray=True)
plt.figure()
plt.imshow(input_image)
plt.axis('off')
filter_size = 40
filter_support = np.linspace(-3, 3, filter_size)
gaussian_filter = np.exp(-(filter_support[:, None] ** 2 + filter_support[None, :] ** 2) / 2 * (0.5))
plt.figure()
plt.imshow(gaussian_filter)
plt.axis('off')

FilterOp = Convolve2D(size=input_image.size, filter=gaussian_filter, shape=input_image.shape)
PoolingOp = Pooling(shape=input_image.shape, block_size=(24, 24))
DownSamplingOp = DownSampling(size=input_image.size, downsampling_factor=20, shape=input_image.shape)
BlurringOp = DownSamplingOp * FilterOp
sampling_bool = np.random.binomial(n=1, p=0.2, size=input_image.shape)
MaskingOp = Masking(size=input_image.size, sampling_bool=sampling_bool)
blurred_image = (FilterOp * input_image.flatten()).reshape(input_image.shape)
inpainting_image = MaskingOp.adjoint(MaskingOp * input_image.flatten()).reshape(input_image.shape)
pooled_image = (PoolingOp * input_image.flatten()).reshape(PoolingOp.output_shape)

plt.figure()
plt.imshow(blurred_image)
plt.axis('off')

plt.figure()
plt.spy(MaskingOp.sampling_bool.reshape(input_image.shape), cmap='gray_r')
plt.axis('off')

plt.figure()
plt.imshow(inpainting_image, cmap='viridis')
plt.axis('off')

plt.figure()
plt.pcolormesh(np.arange(PoolingOp.output_shape[1]), np.arange(PoolingOp.output_shape[0]), 0*pooled_image, cmap='gray_r', edgecolors='k', alpha=1)
plt.axis('off')

plt.figure()
plt.imshow(pooled_image, cmap='viridis')
plt.axis('off')

