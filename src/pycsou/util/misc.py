import collections.abc as cabc

import pycsou.util.ptype as pyct

__all__ = [
    "as_canonical_shape",
]


def as_canonical_shape(x: pyct.NDArrayShape) -> pyct.NDArrayShape:
    # Transform a lone integer into a valid tuple-based shape specifier.
    if not isinstance(x, cabc.Sequence):
        x = (x,)
    sh = tuple(map(int, x))
    return sh
