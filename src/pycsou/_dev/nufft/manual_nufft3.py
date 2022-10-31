import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

import pycsou.operator.linop.nufft as nufft
import pycsou.runtime as pycrt
import pycsou.util as pycu

nufft_kwargs = dict(
    real=False,
    eps=1e-9,
    isign=-1,
    n_trans=5,
    nthreads=0,
    modeord=1,
    upsampfac=2,
)

rng = np.random.default_rng(1)
D, M, N = 2, 200, 50
x = rng.normal(size=(M, D))
z = rng.normal(size=(N, D))

A = nufft.NUFFT.type3(x, z, **nufft_kwargs)
n_modes = np.ceil(np.array(A.fft_size[0])).astype(int).tolist()
h = 2 * np.pi / np.array(n_modes)
gamma = A._dilationfac
A1 = nufft.NUFFT.type1(x / gamma, n_modes, **nufft_kwargs)
A2 = nufft.NUFFT.type2(z * gamma * h, n_modes, **nufft_kwargs)

arr = rng.normal(size=(34, nufft_kwargs["n_trans"], M)).reshape(-1, M)
if not nufft_kwargs["real"]:
    arr = arr + 1j * rng.normal(size=arr.shape)

y0 = pycu.view_as_real(A._nudft_apply(arr))
y_ref = A.apply(pycu.view_as_real(arr))
y_man = pycu.view_as_complex(A1.apply(pycu.view_as_real(arr)))
y_man = fft.ifftn(
    y_man.reshape(
        [
            -1,
        ]
        + n_modes
    ),
    axes=(-2, -1),
    norm="backward",
    workers=-1,
)
y_man = A2.apply(pycu.view_as_real(y_man).reshape(-1, A2.dim))

print(np.linalg.norm(y_man - y_ref) / np.linalg.norm(y_ref))
print(np.linalg.norm(y_man - y0) / np.linalg.norm(y0))
print(np.linalg.norm(y0 - y_ref) / np.linalg.norm(y0))

plt.plot(y0[0])
plt.plot(y_ref[0])
plt.plot(y_man[0])
