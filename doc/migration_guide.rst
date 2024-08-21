.. _migration-guide:

Migrating from v1 to v2
=======================

With the release of `Pyxu v2`, several major improvements and changes have been introduced. This guide will help you smoothly transition your code from `v1` to `v2`.

The most significant change is that `Pyxu v2` no longer vectorizes **N-dimensional** signals. In `v1`, vectorizing **N-dimensional** arrays caused Dask arrays to rechunk into 1-dimensional chunks, which required computing the array in a single node, thus breaking the distributed nature of Dask. In `v2`, the arrays remain **N-dimensional** throughout, and Dask arrays are not "computed" at any point, preserving the benefits of distributed computing.


Key Changes
-----------
- **Signal Handling**: Operators and solvers now work directly with **N-dimensional** data without needing to flatten and reshape.
- **Functionals and Losses**: In `v1`, loss functionals could be defined from functionals with the `asloss` method. We have changed this method to `argshift` for clarity, avoiding ambiguity around sign usage.
- **Stopping Criteria**: The stopping criteria have been updated to use `dim_rank`, which specifies the rank of the signal dimensions.

Example Conversion
-------------------
Below is an example showing how to convert code from `v1` to `v2`.

**Common Setup for v1 and v2**:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import skimage
    from pyxu.operator import Convolve, L21Norm, Gradient, SquaredL2Norm, PositiveOrthant
    from pyxu.opt.solver import PD3O
    from pyxu.opt.stop import RelError

    # Load and preprocess the data
    data = skimage.data.cat()  # shape (300, 451, 3)
    data = np.asarray(data.astype("float32") / 255.0).transpose(2, 0, 1)  # shape (3, 300, 451)

    # Create the Gaussian blurring kernel
    sigma = 7
    width = 13
    mu = (width - 1) / 2
    gauss = lambda x: (1 / (2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x - mu) ** 2) / (sigma**2))
    kernel_1d = np.fromfunction(gauss, (width,)).reshape(1, -1)

    # The shape of the input array will be used to define operators
    dim_shape = data.shape


**v1 Code**:

.. code-block:: python

    # Applying the blurring and adding noise
    conv = Convolve(
        arg_shape=dim_shape,  # v1: using `arg_shape`
        kernel=[np.array([1]), kernel_1d, kernel_1d],
        center=[0, width // 2, width // 2],
    )
    y = conv(data.ravel()).reshape(dim_shape)  # Flattening and reshaping required in v1
    y = np.random.normal(loc=y, scale=0.05)

    # Setting up the MAP approach with total variation prior and positivity constraint
    sl2 = SquaredL2Norm(dim=y.size).asloss(y.ravel())  # v1: `dim` used with `.asloss()`
    loss = sl2 * conv

    l21 = L21Norm(arg_shape=(2, *dim_shape), l2_axis=(0, 1))  # v1: `arg_shape` used

    grad = Gradient(
        arg_shape=dim_shape,  # v1: `arg_shape`
        directions=(1, 2),
    )

    stop_crit = RelError(
        eps=1e-3,
    )

    positivity = PositiveOrthant(dim=y.size)  # v1: `dim` used
    solver = PD3O(f=loss, g=positivity, h=l21, K=grad)
    solver.fit(x0=y.ravel(), stop_crit=stop_crit)  # Flattening required in v1

    # Getting the deblurred image
    recons = solver.solution().reshape(dim_shape)
    recons /= recons.max()


**v2 Code**:

.. code-block:: python

    # Applying the blurring and adding noise
    conv = Convolve(
        dim_shape=dim_shape,  # v2: `dim_shape` replaces `arg_shape`
        kernel=[np.array([1]), kernel_1d, kernel_1d],
        center=[0, width // 2, width // 2],
    )
    y = conv(data)  # No need to flatten or reshape in v2
    y = np.random.normal(loc=y, scale=0.05)

    # Setting up the MAP approach with total variation prior and positivity constraint
    sl2 = SquaredL2Norm(dim_shape=dim_shape).argshift(-y)  # v2: `dim_shape` replaces `dim`, `.argshift()` replaces `.asloss()`
    loss = sl2 * conv

    l21 = L21Norm(dim_shape=(2, *dim_shape), l2_axis=(0, 1))  # v2: `dim_shape` replaces `arg_shape`

    grad = Gradient(
        dim_shape=dim_shape,  # v2: `dim_shape` replaces `arg_shape`
        directions=(1, 2),
    )

    stop_crit = RelError(
        eps=1e-3,
        dim_rank=len(dim_shape),  # v2: New `dim_rank` parameter for dimensional rank
    )

    positivity = PositiveOrthant(dim_shape=dim_shape)  # v2: `dim_shape` replaces `dim`
    solver = PD3O(f=loss, g=positivity, h=l21, K=grad)
    solver.fit(x0=y, stop_crit=stop_crit)  # No flattening required in v2


Migration Tips
--------------
- **dim_shape vs. dim**: In `v2`, wherever `dim` was used in `v1`, you now use `dim_shape` to work with the full **N-dimensional** structure of the data.
- **arg_shape vs. dim_shape**: Similarly, `arg_shape` is replaced by `dim_shape` to emphasize the full shape of the data.
- **argshift replaces asloss**: `argshift` is introduced in place of `asloss` to avoid ambiguity around signs and provide a more intuitive interface.
- **Flattening/Reshaping**: In `v2`, there is no need to flatten and reshape data when using operators like `Convolve` and solvers like `PD3O`. You can work directly with n-dimensional data.
- **dim_rank**: In stopping criteria, `dim_rank` now specifies the rank of the signal dimensions, which was not explicitly required in `v1`.

Further Help
------------
If you encounter any issues during your migration, please consult the `API Reference` and `Example Gallery` or reach out to the community via our support channels.
