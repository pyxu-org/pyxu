import dask.array as da
import numpy as np

import pycsou.util as pycu

use_dask = True
kwargs = dict(real_output=False, real_input=False)
cmat = np.arange(6).reshape(2, 3) + 1j * (np.arange(6) + 2).reshape(2, 3)
if use_dask:
    cmat = da.from_array(cmat)
rmat = pycu.view_as_real_mat(cmat, **kwargs)
cmat2 = pycu.view_as_complex_mat(rmat, **kwargs)
print(pycu.compute(cmat), "\n")
print(pycu.compute(rmat), "\n")
print(pycu.compute(cmat2))
