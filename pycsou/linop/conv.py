# #############################################################################
# conv.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Convolution and smoothing operators for 1D, 2D and graph signals.

Many of the linear operator provided in this module are derived from linear operators from `PyLops <https://pylops.readthedocs.io/en/latest/api/index.html#linear-operators>`_.
"""

import numpy as np
import pylops.signalprocessing as pyconv
import pylops
from typing import Optional, Union
from pycsou.linop.base import PyLopLinearOperator, SparseLinearOperator


def Convolve1D(size: int, filter: np.ndarray, reshape_dims: Optional[tuple] = None, axis: int = 0,
               dtype: type = 'float64', method: Optional[str] = None) -> PyLopLinearOperator:
    r"""
    1D convolution operator.

    *This docstring was adapted from ``pylops.signalprocessing.Convolve1D``.*

    Convolve a multi-dimensional array along a specific ``axis`` with a one-dimensional compact ``filter``.

    Parameters
    ----------
    size: int
        Size of the input array.
    filter: np.ndarray
        1d compact filter. The latter should be real-valued and centered around its mid-size index.
    reshape_dims: Optional[tuple]
        Shape of the array to which the convolution should be applied.
    axis: int
        Axis along which to apply convolution.
    dtype: str
        Type of elements of the input array.
    method: Optional[str]
        Method used to calculate the convolution (``direct``, ``fft``,
        or ``overlapadd``). Note that only ``direct`` and ``fft`` are allowed
        when ``dims=None``, whilst ``fft`` and ``overlapadd`` are allowed
        when ``dims`` is provided.

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        Convolution operator.

    Raises
    ------
    NotImplementedError
        If ``method`` provided is not allowed.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.linop.conv import Convolve1D
       from scipy import signal

    .. doctest::

       >>> sig = np.repeat([0., 1., 0.], 10)
       >>> filter = signal.hann(5); filter[filter.size//2:] = 0
       >>> ConvOp = Convolve1D(size=sig.size, filter=filter)
       >>> filtered = ConvOp * sig
       >>> filtered_scipy = signal.convolve(sig, filter, mode='same', method='direct')
       >>> np.allclose(filtered, filtered_scipy)
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.conv import Convolve1D
       from scipy import signal
       sig = np.repeat([0., 1., 0.], 100)
       filter = signal.hann(50); filter[filter.size//2:] = 0
       ConvOp = Convolve1D(size=sig.size, filter=filter)
       filtered = ConvOp * sig
       correlated = ConvOp.H * sig
       backprojected = ConvOp.DomainGram * sig
       plt.figure()
       plt.subplot(2,2,1)
       plt.plot(sig); plt.plot(np.linspace(0, 50, filter.size), filter); plt.legend(['Signal', 'Filter'])
       plt.subplot(2,2,2)
       plt.plot(filtered); plt.title('Filtered Signal')
       plt.subplot(2,2,3)
       plt.plot(correlated); plt.title('Correlated Signal')
       plt.subplot(2,2,4)
       plt.plot(backprojected); plt.title('Backprojected Signal')
       plt.show()

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.conv import Convolve1D
       from scipy import signal
       sig = np.zeros(shape=(100,100))
       sig[sig.shape[0] // 2 - 2:sig.shape[0] // 2 + 3, sig.shape[1] // 2 - 2:sig.shape[1] // 2 + 3] = 1
       filter = signal.hann(50)
       ConvOp = Convolve1D(size=sig.size, filter=filter, reshape_dims=sig.shape, axis=0)
       filtered = (ConvOp * sig.reshape(-1)).reshape(sig.shape)
       plt.figure()
       plt.subplot(1,2,1)
       plt.imshow(sig, cmap='plasma'); plt.title('Signal')
       plt.subplot(1,2,2)
       plt.imshow(filtered, cmap='plasma'); plt.title('Filtered Signal')
       plt.show()

    Notes
    -----
    The ``Convolve1D`` operator applies convolution between the input signal
    :math:`x(t)` and a compact filter kernel :math:`h(t)` in forward model:

    .. math::
        y(t) = \int_{-\infty}^{\infty} h(t-\tau) x(\tau) d\tau

    This operation can be discretized as follows

    .. math::
        y[n] = \sum_{m\in\mathbb{Z}} h[n-m] x[m], \, n\in\mathbb{Z},

    as well as performed in the frequency domain:

    .. math::
        Y(f) = \mathscr{F} (h(t)) \times \mathscr{F} (x(t)),\; f\in\mathbb{R}.

    ``Convolve1D`` operator uses :py:func:`scipy.signal.convolve` that
    automatically chooses the best method for computing the convolution
    for one dimensional inputs. The FFT implementation
    :py:func:`scipy.signal.fftconvolve` is however enforced for signals in
    2 or more dimensions as this routine efficiently operates on
    multi-dimensional arrays. The method ``overlapadd`` uses :py:func:`scipy.signal.oaconvolve`.

    As the adjoint of convolution is correlation, ``Convolve1D`` operator applies
    correlation in the adjoint mode.

    In time domain:

    .. math::
        x(t) = \int_{-\infty}^{\infty} h(t+\tau) x(\tau) d\tau

    or in frequency domain:

    .. math::
        y(t) = \mathscr{F}^{-1} (H(f)^\ast \times X(f)).

    See Also
    --------
    :py:func:`~pycsou.linop.conv.Convolve2D`

    """
    if (filter.size % 2) == 0:
        offset = filter.size // 2 - 1
    else:
        offset = filter.size // 2
    PyLop = pyconv.Convolve1D(N=size, h=filter, dims=reshape_dims, dir=axis, dtype=dtype, method=method, offset=offset)
    return PyLopLinearOperator(PyLop)


def Convolve2D(size: int, filter: np.ndarray, shape: tuple, dtype: type = 'float64',
               method: str = 'fft') -> PyLopLinearOperator:
    r"""
    2D convolution operator.

    *This docstring was adapted from ``pylops.signalprocessing.Convolve2D``.*

    Convolve a two-dimensional array with a two-dimensional compact ``filter``.

    Parameters
    ----------
    size: int
        Size of the input array.
    filter: np.ndarray
        2d compact filter. The latter should be real-valued and centered around its central indices.
    shape: tuple
        Shape of the array to which the convolution should be applied.
    dtype: str
        Type of elements of the input array.
    method: str
        Method used to calculate the convolution (``direct`` or ``fft``).

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        Convolution operator.

    Raises
    ------
    ValueError
        If ``filter`` is not a 2D array.

    Examples
    --------
    .. testsetup::

       import numpy as np
       from pycsou.linop.conv import Convolve2D
       from scipy import signal

    .. doctest::

       >>> sig = np.zeros(shape=(100,100))
       >>> sig[sig.shape[0] // 2 - 2:sig.shape[0] // 2 + 3, sig.shape[1] // 2 - 2:sig.shape[1] // 2 + 3] = 1
       >>> filter = signal.hann(25); filter[filter.size//2:] = 0
       >>> filter = filter[None,:] * filter[:,None]
       >>> ConvOp = Convolve2D(size=sig.size, filter=filter, shape=sig.shape)
       >>> filtered = (ConvOp * sig.ravel()).reshape(sig.shape)
       >>> filtered_scipy = signal.convolve(sig, filter, mode='same', method='fft')
       >>> np.allclose(filtered, filtered_scipy)
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.conv import Convolve2D
       from scipy import signal
       sig = np.zeros(shape=(100,100))
       sig[sig.shape[0] // 2 - 2:sig.shape[0] // 2 + 3, sig.shape[1] // 2 - 2:sig.shape[1] // 2 + 3] = 1
       filter = signal.hann(50)
       filter = filter[None,:] * filter[:,None]
       ConvOp = Convolve2D(size=sig.size, filter=filter, shape=sig.shape)
       filtered = (ConvOp * sig.ravel()).reshape(sig.shape)
       correlated = (ConvOp.H * sig.ravel()).reshape(sig.shape)
       plt.figure()
       plt.subplot(1,3,1)
       plt.imshow(sig, cmap='plasma'); plt.title('Signal')
       plt.subplot(1,3,2)
       plt.imshow(filtered, cmap='plasma'); plt.title('Filtered Signal')
       plt.subplot(1,3,3)
       plt.imshow(correlated, cmap='plasma'); plt.title('Correlated Signal')
       plt.show()

    Notes
    -----
    The ``Convolve2D`` operator applies two-dimensional convolution
    between the input signal :math:`d(t,x)` and a compact filter kernel
    :math:`h(t,x)` in forward model:

    .. math::
        y(t,x) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
        h(t-\tau,x-\chi) d(\tau,\chi) d\tau d\chi

    This operation can be discretized as follows

    .. math::
        y[i,n] = \sum_{j=-\infty}^{\infty} \sum_{m=-\infty}^{\infty} h[i-j,n-m] d[j,m]


    as well as performed in the frequency domain:

    .. math::
        Y(f, k_x) = \mathscr{F} (h(t,x)) \times \mathscr{F} (d(t,x)).

    ``Convolve2D`` operator uses :py:func:`scipy.signal.convolve`
    that automatically chooses the best domain for the operation
    to be carried out.

    As the adjoint of convolution is correlation, ``Convolve2D`` operator
    applies correlation in the adjoint mode.

    In time domain:

    .. math::
        y(t,x) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
        h(t+\tau,x+\chi) d(\tau,\chi) d\tau d\chi

    or in frequency domain:

    .. math::
        y(t, x) = \mathscr{F}^{-1} (H(f, k_x)^\ast \times X(f, k_x)).

    See Also
    --------
    :py:func:`~pycsou.linop.conv.Convolve1D`

    """
    if (filter.shape[0] % 2) == 0:
        offset0 = filter.shape[0] // 2 - 1
    else:
        offset0 = filter.shape[0] // 2
    if (filter.shape[1] % 2) == 0:
        offset1 = filter.shape[1] // 2 - 1
    else:
        offset1 = filter.shape[1] // 2
    offset = (offset0, offset1)
    PyLop = pyconv.Convolve2D(N=size, h=filter, dims=shape, nodir=None, dtype=dtype, method=method, offset=offset)
    return PyLopLinearOperator(PyLop)


def MovingAverage1D(window_size: int, shape: tuple, axis: int = 0, dtype='float64'):
    r"""
    1D moving average.

    Apply moving average to a multi-dimensional array along a specific axis.

    Parameters
    ----------
    window_size: int
        Size of the window for moving average (must be *odd*).
    shape: tuple
        Shape of the input array.
    axis: int
        Axis along which moving average is applied.
    dtype: str
        Type of elements in input array.

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        1D moving average operator.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.conv import MovingAverage1D
       from scipy import signal
       sig = np.zeros(shape=(100,100))
       sig[sig.shape[0] // 2 - 2:sig.shape[0] // 2 + 3, sig.shape[1] // 2 - 2:sig.shape[1] // 2 + 3] = 1
       MAOp = MovingAverage1D(window_size=25, shape=sig.shape, axis=0)
       moving_average = (MAOp * sig.ravel()).reshape(sig.shape)
       plt.figure()
       plt.subplot(1,2,1)
       plt.imshow(sig, cmap='plasma'); plt.title('Signal')
       plt.subplot(1,2,2)
       plt.imshow(moving_average, cmap='plasma'); plt.title('Moving Average')
       plt.show()

    Notes
    -----
    The ``MovingAverage1D`` operator is a special type of convolution operator that
    convolves along a specific axis an array with a constant filter of size :math:`n_{smooth}`:

    .. math::
        \mathbf{h} = [ 1/n_{smooth}, 1/n_{smooth}, ..., 1/n_{smooth} ]

    For example, for a 3D array :math:`x`,  ``MovingAverage1D`` applied to the first axis yields:

    .. math::
        y[i,j,k] = 1/n_{smooth} \sum_{l=-(n_{smooth}-1)/2}^{(n_{smooth}-1)/2}
        x[l,j,k].

    Note that since the filter is symmetrical, the ``MovingAverage1D`` operator is
    self-adjoint.

    """
    PyLop = pylops.Smoothing1D(nsmooth=window_size, dims=shape, dir=axis, dtype=dtype)
    return PyLopLinearOperator(PyLop)


def MovingAverage2D(window_shape: Union[tuple, list], shape: tuple, dtype='float64'):
    r"""
    2D moving average.

    Apply moving average to a 2D array.

    Parameters
    ----------
    window_size: Union[tuple, list]
        Shape of the window for moving average (sizes in each dimension must be *odd*).
    shape: tuple
        Shape of the input array.
    dtype: str
        Type of elements in input array.

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        2D moving average operator.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.conv import MovingAverage2D
       from scipy import signal
       sig = np.zeros(shape=(100,100))
       sig[sig.shape[0] // 2 - 2:sig.shape[0] // 2 + 3, sig.shape[1] // 2 - 2:sig.shape[1] // 2 + 3] = 1
       MAOp = MovingAverage2D(window_shape=(50,25), shape=sig.shape)
       moving_average = (MAOp * sig.ravel()).reshape(sig.shape)
       plt.figure()
       plt.subplot(1,2,1)
       plt.imshow(sig, cmap='plasma'); plt.title('Signal')
       plt.subplot(1,2,2)
       plt.imshow(moving_average, cmap='plasma'); plt.title('Moving Average')
       plt.show()

    Notes
    -----
    The ``MovingAverage2D`` operator is a special type of convolution operator that
    convolves a 2D array with a constant 2d filter of size :math:`n_{smooth, 1} \quad \times \quad n_{smooth, 2}`:

    .. math::

        y[i,j] = \frac{1}{n_{smooth, 1} n_{smooth, 2}}
        \sum_{l=-(n_{smooth,1}-1)/2}^{(n_{smooth,1}-1)/2}
        \sum_{m=-(n_{smooth,2}-1)/2}^{(n_{smooth,2}-1)/2} x[l,m]

    Note that since the filter is symmetrical, the ``MovingAverage2D`` operator is
    self-adjoint.
    """

    PyLop = pylops.Smoothing2D(nsmooth=window_shape, dims=shape, nodir=None, dtype=dtype)
    return PyLopLinearOperator(PyLop)


if __name__ == '__main__':
    pass
