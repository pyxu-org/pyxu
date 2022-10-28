import collections.abc as cabc

import pycsou.util.ptype as pyct

__all__ = [
    "as_canonical_shape",
    "next_fast_len",
]


def as_canonical_shape(x: pyct.NDArrayShape) -> pyct.NDArrayShape:
    # Transform a lone integer into a valid tuple-based shape specifier.
    if not isinstance(x, cabc.Sequence):
        x = (x,)
    sh = tuple(map(int, x))
    return sh


def next_fast_len(N: pyct.Integer, even: bool = False) -> pyct.Integer:
    # Compute the next 5-smooth number greater-or-equal to `N`.
    # If `even=True`, then ensure the next 5-smooth number is even-valued.
    #
    # ref: https://en.wikipedia.org/wiki/Smooth_number

    # warning: scipy.fftpack.next_fast_len() != scipy.fft.next_fast_len()
    from scipy.fftpack import next_fast_len

    N5 = next_fast_len(int(N))

    is_even = lambda n: n % 2 == 0
    if (not is_even(N5)) and even:
        while not is_even(N5 := next_fast_len(N5)):
            N5 += 1

    return N5
