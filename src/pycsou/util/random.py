import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "default_rng",
]


def default_rng(
    ndi: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
    seed: pyct.Integer = None,
):
    """Returns a random generator for a given array module."""
    if ndi == pycd.NDArrayInfo.DASK:
        from dask.array.random import RandomState

        return RandomState(seed)  # Dask does not yet have generator objects, but RandomState objects behave similarly?
    else:  # Numpy or Cupy
        xp = ndi.module()
        return xp.random.default_rng(seed)
