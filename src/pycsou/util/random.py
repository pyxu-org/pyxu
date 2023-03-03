import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct


def random_generator(array_module: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY, seed: pyct.Integer = None):
    """Returns a random generator for a given array module."""
    if array_module == pycd.NDArrayInfo.DASK:
        from dask.array.random import RandomState

        return RandomState(seed)  # Dask does not yet have generator objects, but RandomState objects behave similarly?
    else:  # Numpy or Cupy
        xp = array_module.module()
        return xp.random.default_rng(seed)
