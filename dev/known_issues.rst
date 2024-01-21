Known 3rd-Party Issues
======================

This file contains the list of 3rd-party bugs/limitations affecting Pyxu which require fixes
upstream. A link to an issue-tracker entry is provided if already reported.


* ``dask.array.tensordot(a, b, axes)`` returns wrong shape.

  .. code::

     import numpy as np
     import dask.array as da

     rng = np.random.default_rng()
     a = rng.normal(size=(30, 10))
     b = rng.normal(size=(10, 20))

     kwargs = dict(axes=[[-1], [0]])
     np_z = np.tensordot(a, b, **kwargs)
     da_z = da.tensordot(da.array(a), da.array(b), **kwargs)

     >>> np_z.shape, da_z.shape
     ((30, 20), (30, 1))

* ``cupyx.scipy.sparse.linalg.svds()`` returns wrong results (https://github.com/cupy/cupy/issues/6446).

  .. code::

     import cupy as cp
     import cupyx.scipy.sparse.linalg as cpsl
     import cupy.linalg as cpl

     N = 5
     x = cp.eye(N)
     y_dense = cpl.svd(x, compute_uv=False)[0]
     y_sparse = cpsl.svds(x, k=1, return_singular_vectors=False)  # sometimes gives ``LinAlgError("Eigenvalues did not converge")``

     >>> y_dense, y_sparse
     (array(1.), array([2.08290313]))
