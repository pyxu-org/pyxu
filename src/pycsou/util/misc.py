import collections.abc as cabc

import numpy as np

import pycsou.util as pycu
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


def peaks(x: pyct.NDArray, y: pyct.NDArray) -> pyct.NDArray:
    r"""
    Matlab 2D peaks function.
    Peaks is a function of two variables, obtained by translating and scaling Gaussian distributions (see `Matlab's peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`_).
    This function is useful for testing purposes.
    Parameters
    ----------
    x: NDArray
        X coordinates.
    y: NDArray
        Y coordinates.
    Returns
    -------
    NDArray
        Values of the 2D function ``peaks`` at the points specified by the entries of ``x`` and ``y``.
    Examples
    --------
    .. plot::
       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import peaks
       x = np.linspace(-3,3, 1000)
       xx, yy = np.meshgrid(x,x)
       z = peaks(xx, yy)
       plt.figure()
       plt.imshow(z)
    """
    xp = pycu.get_array_module(x)
    z = (
        3 * ((1 - x) ** 2) * xp.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * xp.exp(-(x**2) - y**2)
        - (1 / 3) * xp.exp(-((x + 1) ** 2) - y**2)
    )
    return z


def star_like_sample(N: int, w: int, s: float, po: int, x0: float) -> pyct.NDArray:
    r"""
    Generates a NxN square image of a star-like object normalized between 0 and 1. Based on `GlobalBioIm's
    StarLikeSample function <https://www.mathworks.com/help/matlab/ref/peaks.html>`_. This function is useful for
    testing purposes as it contains high-frequency information.

    Parameters
    ----------
    N: int
        Size of the image (must be an even number).
    w: int
        The number of branches of the sample will be 4*w.
    s: float
        Slope of the sigmoid function :math:`\frac1{1+\exp(s*(x-x0))}` attenuating the boundaries.
    po: int
        Power-raising factor for the final image (to have smoother edges).
    x0: float
        Radial shift of the sigmoid function :math:`\frac1{1+\exp(s*(x-x0))}`.

    Returns
    -------
    im: NDArray
        Image of star-like sample.
    """
    grid = np.linspace(-1, 1, N)
    x, y = np.meshgrid(grid, grid)
    theta = np.arctan2(y, x)
    image = 1 + np.cos(4 * w * theta)
    image /= 1 + np.exp(s * (np.sqrt(x**2 + y**2) - x0))
    image = np.maximum(image, 0) / 2
    image **= po
    image /= np.amax(image)
    return image
