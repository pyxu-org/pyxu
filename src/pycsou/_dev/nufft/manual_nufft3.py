import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

nufft_kwargs = dict(
    real=False,
    isign=-1,
    nthreads=0,
    modeord=1,
    upsampfac=1.25,
)

eps = 1e-7
rng = np.random.default_rng(1)
D, M, N = 1, 150, 250
x = rng.uniform(-1, 1, size=(M, D))
z = rng.uniform(-1, 1, size=(N, D))

xp = pycu.get_array_module(x)

A = nufft.NUFFT.type3(x, z, eps=eps, **nufft_kwargs)
A0 = nufft.NUFFT.type3(x, z, eps=0, **nufft_kwargs)

n_modes = A._fft_shape()
X, x0 = A._shift_coords(x)
Z, z0 = A._shift_coords(z)
gamma = xp.array(A._dilation_factor(), dtype=x.dtype)
A1 = nufft.NUFFT.type1(x / gamma, n_modes, eps=eps / 2, **nufft_kwargs)
mesh1 = A1.mesh(scale="source", upsampled=True)
A2 = nufft.NUFFT.type2(z * gamma * 2 * np.pi / xp.array(n_modes, dtype=x.dtype), n_modes, eps=eps / 2, **nufft_kwargs)

arr = rng.normal(10, size=M)
if not nufft_kwargs["real"]:
    arr = arr + 1j * rng.normal(10, size=arr.shape)

y0 = A0.apply(pycu.view_as_real(arr))
y_ref = A.apply(pycu.view_as_real(arr))
y_man = pycu.view_as_complex(A1.apply(pycu.view_as_real(arr)))
y_man = fft.ifft(
    y_man,
    norm="backward",
    workers=-1,
)
y_man = A2.apply(pycu.view_as_real(y_man))

print(np.linalg.norm(y_man - y_ref) / np.linalg.norm(y_ref))
# print(np.linalg.norm(y_man - y0) / np.linalg.norm(y0))
# print(np.linalg.norm(y0 - y_ref) / np.linalg.norm(y0))

plt.figure()
# plt.plot(y0)
plt.plot(y_ref)
plt.plot(y_man)
plt.show()

z_ref = A.adjoint(y_ref)
z_man = pycu.view_as_complex(A2.adjoint(y_ref))
z_man = fft.fft(
    z_man,
    norm="forward",
    workers=-1,
)
z_man = A1.adjoint(pycu.view_as_real(z_man))
print(np.linalg.norm(z_man - z_ref) / np.linalg.norm(z_ref))

plt.figure()
plt.plot(z_ref)
plt.plot(z_man)
plt.show()

plt.figure()
plt.subplot(2, 1, 1)
# plt.stem(x.squeeze() / gamma, np.abs(arr))
plt.scatter(
    x.squeeze() / gamma,
    np.abs(pycu.view_as_complex(z_ref)),
)
plt.scatter(x.squeeze() / gamma, np.abs(pycu.view_as_complex(z_man)))
plt.subplot(2, 1, 2)
plt.scatter(z.squeeze() * gamma * 2 * np.pi / xp.array(n_modes, dtype=x.dtype), np.abs(pycu.view_as_complex(y_ref)))
plt.scatter(z.squeeze() * gamma * 2 * np.pi / xp.array(n_modes, dtype=x.dtype), np.abs(pycu.view_as_complex(y_man)))
plt.show()
