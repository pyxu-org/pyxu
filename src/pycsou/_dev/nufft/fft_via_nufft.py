import time as t

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import scipy.linalg as sclin

import pycsou.operator.linop.nufft as nufft
import pycsou.util as pycu

nufft_kwargs = dict(
    real=False,
    isign=-1,
    nthreads=0,
    modeord=1,
    upsampfac=2,
)


eps = 1e-5
rng = np.random.default_rng(1)
D, N = 2, 25
freqs = fft.fftfreq(N) * 2 * np.pi

A = nufft.NUFFT.type1(freqs, N, eps=eps, **nufft_kwargs)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(A.asarray())
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(pycu.view_as_real_mat(sclin.dft(N)))
plt.colorbar()


arr = rng.normal(size=N)
if not nufft_kwargs["real"]:
    arr = arr + 1j * rng.normal(size=arr.shape)

tic = t.time()
y_ref = fft.fft(
    arr,
    norm="backward",
    workers=-1,
)
print(f"FFT: {t.time() - tic} secs")
tic = t.time()
y_man = pycu.view_as_complex(A.apply(pycu.view_as_real(arr)))
print(f"NUFFT: {t.time() - tic} secs")
print(f"Acurracy: {np.linalg.norm(y_man - y_ref) / np.linalg.norm(y_ref)}")

plt.figure()
plt.plot(pycu.view_as_real(y_ref))
plt.plot(pycu.view_as_real(y_man))
plt.show()
