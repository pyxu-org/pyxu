import collections.abc as cabc

import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "as_canonical_shape",
    "next_fast_len",
    "peaks",
    "star_like_sample",
]


def as_canonical_shape(x: pyct.NDArrayShape) -> pyct.NDArrayShape:
    # Transform a lone integer into a valid tuple-based shape specifier.
    if isinstance(x, cabc.Iterable):
        x = tuple(x)
    else:
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

    Peaks is a function of two variables, obtained by translating and scaling Gaussian
    distributions.
    (See `Matlab's peaks function <https://www.mathworks.com/help/matlab/ref/peaks.html>`_.)

    This function is useful for testing purposes.

    Parameters
    ----------
    x: pyct.NDArray
        X coordinates.
    y: pyct.NDArray
        Y coordinates.

    Returns
    -------
    z: pyct.NDArray
        Values of the 2D function ``peaks`` at the points specified by the entries of ``x`` and
        ``y``.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import peaks

       x = np.linspace(-3, 3, 1000)
       xx, yy = np.meshgrid(x, x)
       z = peaks(xx, yy)
       plt.figure()
       plt.imshow(z)
    """
    ndi = pycd.NDArrayInfo.from_obj(x)
    xp = ndi.module()

    a = 3 * ((1 - x) ** 2) * xp.exp(-(x**2) - (y + 1) ** 2)
    b = -10 * ((x / 5) - x**3 - y**5) * xp.exp(-(x**2) - (y**2))
    c = -xp.exp(-((x + 1) ** 2) - (y**2)) / 3
    z = a + b + c
    return z


def star_like_sample(
    N: int,
    w: int,
    s: float,
    po: int,
    x0: float,
    ndi: pycd.NDArrayInfo = pycd.NDArrayInfo.NUMPY,
) -> pyct.NDArray:
    r"""
    Star-like test image.

    Generates a (N, N) square image of a star-like object normalized between 0 and 1.
    Based on `GlobalBioIm's StarLikeSample function
    <https://github.com/Biomedical-Imaging-Group/GlobalBioIm/blob/master/Util/StarLikeSample.m>`_.
    This function is useful for testing purposes as it contains high-frequency information.

    Parameters
    ----------
    N: int
        Size of the image (must be an even number).
    w: int
        The number of branches of the sample will be 4*w.
    s: float
        Slope of the sigmoid function :math:`\frac1{1+\exp[s (x-x_{0})]}` attenuating the boundaries.
    po: int
        Power-raising factor for the final image (to have smoother edges).
    x0: float
        Radial shift of the sigmoid function :math:`\frac1{1+\exp[s (x-x_{0})]}`.
    ndi: pycd.NDArrayInfo
        Desired array module for the output.

    Returns
    -------
    image: pyct.NDArray
        (N, N) image of star-like sample.

    Examples
    --------
    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.util.misc import star_like_sample

       star = star_like_sample(N=256, w=8, s=20, po=3, x0=0.7)
       plt.figure()
       plt.imshow(star)
    """
    assert N % 2 == 0
    xp = ndi.module()

    grid = xp.linspace(-1, 1, N)
    x, y = xp.meshgrid(grid, grid)
    theta = xp.arctan2(y, x)
    image = 1 + xp.cos(4 * w * theta)
    image /= 1 + xp.exp(s * (xp.sqrt(x**2 + y**2) - x0))
    image = xp.maximum(image, 0) / 2
    image **= po
    image /= xp.amax(image)
    return image
