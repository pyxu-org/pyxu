from itertools import product
from operator import add

import toolz


def unpad(arr, pad_width):
    r"""
    Reverse effect of np.pad given pad_width.
    See da.chunk.trim - https://github.com/dask/dask/blob/main/dask/array/chunk.py

    Note: Assuming pad_width is symmetric
    """
    return arr[tuple(slice(pad[0], -pad[1] if pad[0] else None) for pad in pad_width)]


def depth_to_pad(depth: dict, ndims: int = 0) -> list[tuple]:
    r"""
    Converts from dask depth to numpy pad inputs.

    **Note**
    Depth is assumed to be coerced using dask.array.overlap.coerce_depth. This means every dimension will have a depth
    key, even if that key is zero.

    Examples:
    -------
    >>> depth_to_pad({0:1, 1:2, 2:0})
    [(1,1), (2,2), (0,0)]
    """
    initial = [(0, 0)]*ndims
    initial.extend([(v, v) for _, v in depth.items()])
    return initial


def _cumsum(seq, initial):
    r"""
    Modified from dask.utils._cumsum - https://github.com/dask/dask/blob/main/dask/utils.py
    Can take an initial value other than zero.
    """
    return tuple(toolz.accumulate(add, seq, initial=initial))


def slices_from_chunks_with_overlap(chunks: tuple[tuple[int]], depth: dict):
    r"""
    Translates dask chunks tuples into a set of slices in product order. Takes into account padding and block overlaps.

    Modified from dask.array.core.slices_from_chunks - https://github.com/dask/dask/blob/main/dask/array/core.py

    **Remark 1:**
    The depth padding is assumed to be symmetric around each dimension. This is the convention dask uses, and we follow
    it.

    **Remark 2:**
    Depth is assumed to be coerced using dask.array.overlap.coerce_depth. This means every dimension will have a depth
    key, even if that key is zero.

    Examples:
    -------
    >>> slices_from_chunks_with_overlap(chunks=((2, 2), (3, 3, 3)), depth={0:1, 1:2})
     [(slice(0, 4, None), slice(0, 7, None)),
      (slice(0, 4, None), slice(3, 10, None)),
      (slice(0, 4, None), slice(6, 13, None)),
      (slice(2, 6, None), slice(0, 7, None)),
      (slice(2, 6, None), slice(3, 10, None)),
      (slice(2, 6, None), slice(6, 13, None))]
    """
    cumdims = [_cumsum(bds, initial=depth[i]) for i, bds in enumerate(chunks)]
    slices = [
        [slice(s - depth[i], s + dim + depth[i]) for s, dim in zip(starts, shapes)]
        for i, (starts, shapes) in enumerate(zip(cumdims, chunks))
    ]
    return list(product(*slices))
