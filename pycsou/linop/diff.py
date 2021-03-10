# #############################################################################
# diff.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Discrete differential and integral operators.

This module provides differential operators for discrete signals defined over regular grids or arbitrary meshes (graphs).

Many of the linear operators provided in this module are derived from linear operators from `PyLops <https://pylops.readthedocs.io/en/latest/api/index.html#smoothing-and-derivatives>`_.
"""

import numpy as np
import pylops
from typing import Optional, Union, Tuple, Iterable, List
from pycsou.core.linop import LinearOperator
from pycsou.linop.base import PyLopLinearOperator, SparseLinearOperator, LinOpVStack, \
    DiagonalOperator, IdentityOperator, PolynomialLinearOperator
from numbers import Number


def FirstDerivative(size: int, shape: Optional[tuple] = None, axis: int = 0, step: float = 1.0, edge: bool = True,
                    dtype: str = 'float64', kind: str = 'forward') -> PyLopLinearOperator:
    r"""
    First derivative.

    *This docstring was adapted from ``pylops.FirstDerivative``.*

    Approximates the first derivative of a multi-dimensional array along a specific ``axis`` using finite-differences.

    Parameters
    ----------
    size: int
        Size of the input array.
    shape: tuple
        Shape of the input array.
    axis: int
        Axis along which to differentiate.
    step: float
        Step size.
    edge: bool
        For ``kind='centered'``, use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input array.
    kind: str
        Derivative kind (``forward``, ``centered``, or ``backward``).

    Returns
    -------
    :py:class:`~pycsou.linop.base.PyLopLinearOperator`
        First derivative operator.

    Raises
    ------
    ValueError
        If ``shape`` and ``size`` are not compatible.
    NotImplementedError
        If ``kind`` is not one of: ``forward``, ``centered``, or ``backward``.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import FirstDerivative

    .. doctest::

       >>> x = np.repeat([0,2,1,3,0,2,0], 10)
       >>> Dop = FirstDerivative(size=x.size)
       >>> y = Dop * x
       >>> np.sum(np.abs(y) > 0)
       6
       >>> np.allclose(y, np.diff(x, append=0))
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import FirstDerivative

       x = np.repeat([0,2,1,3,0,2,0], 10)
       Dop_bwd = FirstDerivative(size=x.size, kind='backward')
       Dop_fwd = FirstDerivative(size=x.size, kind='forward')
       Dop_cent = FirstDerivative(size=x.size, kind='centered')
       y_bwd = Dop_bwd * x
       y_cent = Dop_cent * x
       y_fwd = Dop_fwd * x
       plt.figure()
       plt.plot(np.arange(x.size), x)
       plt.plot(np.arange(x.size), y_bwd)
       plt.plot(np.arange(x.size), y_cent)
       plt.plot(np.arange(x.size), y_fwd)
       plt.legend(['Signal', 'Backward', 'Centered', 'Forward'])
       plt.title('First derivative')
       plt.show()

    Notes
    -----
    The ``FirstDerivative`` operator applies a first derivative along a given axis
    of a multi-dimensional array using either a *second-order centered stencil* or *first-order forward/backward stencils*.

    For simplicity, given a one dimensional array, the second-order centered
    first derivative is:

    .. math::
        y[i] = (0.5x[i+1] - 0.5x[i-1]) / \text{step}

    while the first-order forward stencil is:

    .. math::
        y[i] = (x[i+1] - x[i]) / \text{step}

    and the first-order backward stencil is:

    .. math::
        y[i] = (x[i] - x[i-1]) / \text{step}.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.SecondDerivative`, :py:func:`~pycsou.linop.diff.GeneralisedDerivative`

    """
    first_derivative = pylops.FirstDerivative(N=size, dims=shape, dir=axis, sampling=step, edge=edge, dtype=dtype,
                                              kind=kind)
    return PyLopLinearOperator(first_derivative)


def SecondDerivative(size: int, shape: Optional[tuple] = None, axis: int = 0, step: float = 1.0, edge: bool = True,
                     dtype: str = 'float64') -> PyLopLinearOperator:
    r"""
    Second derivative.

    *This docstring was adapted from ``pylops.SecondDerivative``.*

    Approximates the second derivative of a multi-dimensional array along a specific ``axis`` using finite-differences.

    Parameters
    ----------
    size: int
        Size of the input array.
    shape: tuple
        Shape of the input array.
    axis: int
        Axis along which to differentiate.
    step: float
        Step size.
    edge: bool
        Use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input array.

    Returns
    -------
    :py:class:`~pycsou.linop.base.PyLopLinearOperator`
        Second derivative operator.

    Raises
    ------
    ValueError
        If ``shape`` and ``size`` are not compatible.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import SecondDerivative

    .. doctest::

       >>> x = np.linspace(-2.5, 2.5, 100)
       >>> z = np.piecewise(x, [x < -1, (x >= - 1) * (x<0), x>=0], [lambda x: -x, lambda x: 3 * x + 4, lambda x: -0.5 * x + 4])
       >>> Dop = SecondDerivative(size=x.size)
       >>> y = Dop * z


    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import SecondDerivative

       x = np.linspace(-2.5, 2.5, 200)
       z = np.piecewise(x, [x < -1, (x >= - 1) * (x<0), x>=0], [lambda x: -x, lambda x: 3 * x + 4, lambda x: -0.5 * x + 4])
       Dop = SecondDerivative(size=x.size)
       y = Dop * z
       plt.figure()
       plt.plot(np.arange(x.size), z)
       plt.title('Signal')
       plt.show()
       plt.figure()
       plt.plot(np.arange(x.size), y)
       plt.title('Second Derivative')
       plt.show()

    Notes
    -----
    The ``SecondDerivative`` operator applies a second derivative to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the second-order centered
    second derivative is given by:

    .. math::
        y[i] = (x[i+1] - 2x[i] + x[i-1]) / \text{step}^2.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.FirstDerivative`, :py:func:`~pycsou.linop.diff.GeneralisedDerivative`

    """
    return PyLopLinearOperator(
        pylops.SecondDerivative(N=size, dims=shape, dir=axis, sampling=step, edge=edge, dtype=dtype))


def GeneralisedDerivative(size: int, shape: Optional[tuple] = None, axis: int = 0, step: float = 1.0, edge: bool = True,
                          dtype: str = 'float64', kind_op='iterated', kind_diff='centered', **kwargs) -> LinearOperator:
    r"""
    Generalised derivative.

    Approximates the generalised derivative of a multi-dimensional array along a specific ``axis`` using finite-differences.

    Parameters
    ----------
    size: int
        Size of the input array.
    shape: tuple
        Shape of the input array.
    axis: int
        Axis along which to differentiate.
    step: float
        Step size.
    edge: bool
        For ``kind_diff = 'centered'``, use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input array.
    kind_diff: str
        Derivative kind (``forward``, ``centered``, or ``backward``).
    kind_op: str
        Type of generalised derivative (``'iterated'``, ``'sobolev'``, ``'exponential'``, ``'polynomial'``).
        Depending on the cases, the ``GeneralisedDerivative`` operator is defined as follows:

        * ``'iterated'``: :math:`\mathscr{D}=D^N`,
        * ``'sobolev'``: :math:`\mathscr{D}=(\alpha^2 \mathrm{Id}-D^2)^N`, with :math:`\alpha\in\mathbb{R}`,
        * ``'exponential'``: :math:`\mathscr{D}=(\alpha \mathrm{Id} + D)^N`,  with :math:`\alpha\in\mathbb{R}`,
        * ``'polynomial'``: :math:`\mathscr{D}=\sum_{n=0}^N \alpha_n D^n`,  with :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`,

        where :math:`D` is the :py:func:`~pycsou.linop.diff.FirstDerivative` operator.

    kwargs: Any
        Additional arguments depending on the value of ``kind_op``:

        * ``'iterated'``: ``kwargs={order: int}`` where ``order`` defines the exponent :math:`N`.
        * ``'sobolev', 'exponential'``: ``kwargs={order: int, constant: float}`` where ``order`` defines the exponent :math:`N` and ``constant`` the scalar :math:`\alpha\in\mathbb{R}`.
        * ``kind_op='polynomial'``: ``kwargs={coeffs: Union[np.ndarray, list, tuple]}`` where ``coeffs`` is an array containing the coefficients :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`.

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
        A generalised derivative operator.

    Raises
    ------
    NotImplementedError
        If ``kind_op`` is not one of: ``'iterated'``, ``'sobolev'``, ``'exponential'``, ``'polynomial'``.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import GeneralisedDerivative, FirstDerivative

    .. doctest::

       >>> x = np.linspace(-5, 5, 100)
       >>> z = np.sinc(x)
       >>> Dop = GeneralisedDerivative(size=x.size, kind_op='iterated', order=1, kind_diff='forward')
       >>> D = FirstDerivative(size=x.size, kind='forward')
       >>> np.allclose(Dop * z, D * z)
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import GeneralisedDerivative

       x  = np.linspace(-2.5, 2.5, 500)
       z = np.exp(-x ** 2)
       Dop_it = GeneralisedDerivative(size=x.size, kind_op='iterated', order=4)
       Dop_sob = GeneralisedDerivative(size=x.size, kind_op='sobolev', order=2, constant=1e-2)
       Dop_exp = GeneralisedDerivative(size=x.size, kind_op='exponential', order=4, constant=-1e-2)
       Dop_pol = GeneralisedDerivative(size=x.size, kind_op='polynomial',
                                       coeffs= 1/2 * np.array([1e-8, 0, -2 * 1e-4, 0, 1]))
       y_it = Dop_it * z
       y_sob = Dop_sob * z
       y_exp = Dop_exp * z
       y_pol = Dop_pol * z
       plt.figure()
       plt.plot(x, z)
       plt.title('Signal')
       plt.show()
       plt.figure()
       plt.plot(x, y_it)
       plt.plot(x, y_sob)
       plt.plot(x, y_exp)
       plt.plot(x, y_pol)
       plt.legend(['Iterated', 'Sobolev', 'Exponential', 'Polynomial'])
       plt.title('Generalised derivatives')
       plt.show()

    Notes
    -----
    Problematic values at edges are set to zero.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.FirstDerivative`, :py:func:`~pycsou.linop.diff.SecondDerivative`,
    :py:func:`~pycsou.linop.diff.GeneralisedLaplacian`

    """
    D = FirstDerivative(size=size, shape=shape, axis=axis, step=step, edge=edge, dtype=dtype, kind=kind_diff)
    D.is_symmetric = False
    D2 = SecondDerivative(size=size, shape=shape, axis=axis, step=step, edge=edge, dtype=dtype)
    if kind_op == 'iterated':
        N = kwargs['order']
        Dgen = D ** N
        order = N
    elif kind_op == 'sobolev':
        I = IdentityOperator(size=size)
        alpha = kwargs['constant']
        N = kwargs['order']
        Dgen = ((alpha ** 2) * I - D2) ** N
        order = 2 * N
    elif kind_op == 'exponential':
        I = IdentityOperator(size=size)
        alpha = kwargs['constant']
        N = kwargs['order']
        Dgen = (alpha * I + D) ** N
        order = N
    elif kind_op == 'polynomial':
        coeffs = kwargs['coeffs']
        Dgen = PolynomialLinearOperator(LinOp=D, coeffs=coeffs)
        order = len(coeffs) - 1
    else:
        raise NotImplementedError(
            'Supported generalised derivative types are: iterated, sobolev, exponential, polynomial.')

    if shape is None:
        kill_edges = np.ones(shape=Dgen.shape[0])
    else:
        kill_edges = np.ones(shape=shape)

    if axis > 0:
        kill_edges = np.swapaxes(kill_edges, axis, 0)
    if kind_diff == 'forward':
        kill_edges[-order:] = 0
    elif kind_diff == 'backward':
        kill_edges[:order] = 0
    elif kind_diff == 'centered':
        kill_edges[-order:] = 0
        kill_edges[:order] = 0
    else:
        pass
    if axis > 0:
        kill_edges = np.swapaxes(kill_edges, 0, axis)
    KillEdgeOp = DiagonalOperator(kill_edges.reshape(-1))
    Dgen = KillEdgeOp * Dgen
    return Dgen


def FirstDirectionalDerivative(shape: tuple, directions: np.ndarray, step: Union[float, Tuple[float, ...]] = 1.,
                               edge: bool = True, dtype: str = 'float64',
                               kind: str = 'centered') -> PyLopLinearOperator:
    r"""
    First directional derivative.

    Computes the directional derivative of a multi-dimensional array (at least two dimensions are required)
    along either a single common direction or different ``directions`` for each entry of the array.


    Parameters
    ----------
    shape: tuple
        Shape of the input array.
    directions: np.ndarray
        Single direction (array of size :math:`n_{dims}`) or different directions for each entry (array of size :math:`[n_{dims} \times (n_{d_0} \times ... \times n_{d_{n_{dims}}})]`).
        Each column should be normalised.
    step: Union[float, Tuple[float, ...]]
        Step size in each direction.
    edge: bool
        For ``kind = 'centered'``, use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input vector.
    kind: str
        Derivative kind (``forward``, ``centered``, or ``backward``).

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        Directional derivative operator.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import FirstDirectionalDerivative, FirstDerivative
       from pycsou.util.misc import peaks

    .. doctest::

       >>> x = np.linspace(-2.5, 2.5, 100)
       >>> X,Y = np.meshgrid(x,x)
       >>> Z = peaks(X, Y)
       >>> direction = np.array([1,0])
       >>> Dop = FirstDirectionalDerivative(shape=Z.shape, directions=direction, kind='forward')
       >>> D = FirstDerivative(size=Z.size, shape=Z.shape, kind='forward')
       >>> np.allclose(Dop * Z.flatten(), D * Z.flatten())
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import FirstDirectionalDerivative, FirstDerivative
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       directions = np.zeros(shape=(2,Z.size))
       directions[0, :Z.size//2] = 1
       directions[1, Z.size//2:] = 1
       Dop = FirstDirectionalDerivative(shape=Z.shape, directions=directions)
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Directional derivatives')
       plt.show()


    Notes
    -----
    The ``FirstDirectionalDerivative`` applies a first-order derivative
    to a multi-dimensional array along the direction defined by the unitary
    vector :math:`\mathbf{v}`:

    .. math::
        d_\mathbf{v}f =
            \langle\nabla f, \mathbf{v}\rangle,

    or along the directions defined by the unitary vectors
    :math:`\mathbf{v}(x, y)`:

    .. math::
        d_\mathbf{v}(x,y) f =
            \langle\nabla f(x,y), \mathbf{v}(x,y)\rangle

    where we have here considered the 2-dimensional case.
    Note that the 2D case, choosing :math:`\mathbf{v}=[1,0]` or :math:`\mathbf{v}=[0,1]`
    is equivalent to the ``FirstDerivative`` operator applied to axis 0 or 1 respectively.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.SecondDirectionalDerivative`, :py:func:`~pycsou.linop.diff.FirstDerivative`

    """
    return PyLopLinearOperator(
        pylops.FirstDirectionalDerivative(dims=shape, v=directions, sampling=step, edge=edge, dtype=dtype, kind=kind))


def SecondDirectionalDerivative(shape: tuple, directions: np.ndarray, step: Union[float, Tuple[float, ...]] = 1.,
                                edge: bool = True, dtype: str = 'float64'):
    r"""
    Second directional derivative.

    Computes the second directional derivative of a multi-dimensional array (at least two dimensions are required)
    along either a single common direction or different ``directions`` for each entry of the array.


    Parameters
    ----------
    shape: tuple
        Shape of the input array.
    directions: np.ndarray
        Single direction (array of size :math:`n_{dims}`) or different directions for each entry (array of size :math:`[n_{dims} \times (n_{d_0} \times ... \times n_{d_{n_{dims}}})]`).
        Each column should be normalised.
    step: Union[float, Tuple[float, ...]]
        Step size in each direction.
    edge: bool
        Use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input vector.

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        Second directional derivative operator.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import SecondDirectionalDerivative
       from pycsou.util.misc import peaks

    .. doctest::

       >>> x = np.linspace(-2.5, 2.5, 100)
       >>> X,Y = np.meshgrid(x,x)
       >>> Z = peaks(X, Y)
       >>> direction = np.array([1,0])
       >>> Dop = SecondDirectionalDerivative(shape=Z.shape, directions=direction)
       >>> dir_d2 = (Dop * Z.reshape(-1)).reshape(Z.shape)

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import FirstDirectionalDerivative, SecondDirectionalDerivative
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       directions = np.zeros(shape=(2,Z.size))
       directions[0, :Z.size//2] = 1
       directions[1, Z.size//2:] = 1
       Dop = FirstDirectionalDerivative(shape=Z.shape, directions=directions)
       Dop2 = SecondDirectionalDerivative(shape=Z.shape, directions=directions)
       y = Dop * Z.flatten()
       y2 = Dop2 * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
       plt.colorbar(h)
       plt.title('First Directional derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y2.reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Second Directional derivatives')
       plt.show()



    Notes
    -----
    The ``SecondDirectionalDerivative`` applies a second-order derivative
    to a multi-dimensional array along the direction defined by the unitary
    vector :math:`\mathbf{v}`:

    .. math::
        d^2_\mathbf{v} f =
            - d_\mathbf{v}^\ast (d_\mathbf{v} f)

    where :math:`d_\mathbf{v}` is the first-order directional derivative
    implemented by :py:func:`~pycsou.linop.diff.FirstDirectionalDerivative`. The above formula generalises the well-known relationship:

    .. math::
        \Delta f= -\text{div}(\nabla f),

    where minus the divergence operator is the adjoint of the gradient.

    **Note that problematic values at edges are set to zero.**

    See Also
    --------
    :py:func:`~pycsou.linop.diff.FirstDirectionalDerivative`, :py:func:`~pycsou.linop.diff.SecondDerivative`

    """
    Pylop = PyLopLinearOperator(
        pylops.SecondDirectionalDerivative(dims=shape, v=directions, sampling=step, edge=edge, dtype=dtype))
    kill_edges = np.ones(shape=shape)
    for axis in range(len(shape)):
        kill_edges = np.swapaxes(kill_edges, axis, 0)
        kill_edges[-2:] = 0
        kill_edges[:2] = 0
        kill_edges = np.swapaxes(kill_edges, 0, axis)
    KillEdgeOp = DiagonalOperator(kill_edges.reshape(-1))
    DirD2 = KillEdgeOp * Pylop
    return DirD2


def DirectionalGradient(first_directional_derivatives: List[FirstDirectionalDerivative]) -> LinOpVStack:
    r"""
    Directional gradient.

    Computes the directional derivative of a multi-dimensional array (at least two dimensions are required)
    along multiple ``directions`` for each entry of the array.


    Parameters
    ----------
    first_directional_derivatives: List[FirstDirectionalDerivative]
        List of the directional derivatives to be stacked.

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
        Stack of first directional derivatives.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import FirstDirectionalDerivative, DirectionalGradient
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       directions1 = np.zeros(shape=(2,Z.size))
       directions1[0, :Z.size//2] = 1
       directions1[1, Z.size//2:] = 1
       directions2 = np.zeros(shape=(2,Z.size))
       directions2[1, :Z.size//2] = -1
       directions2[0, Z.size//2:] = -1
       Dop1 = FirstDirectionalDerivative(shape=Z.shape, directions=directions1)
       Dop2 = FirstDirectionalDerivative(shape=Z.shape, directions=directions2)
       Dop = DirectionalGradient([Dop1, Dop2])
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.quiver(x, x, directions1[1].reshape(X.shape), directions1[0].reshape(X.shape))
       plt.quiver(x, x, directions2[1].reshape(X.shape), directions2[0].reshape(X.shape), color='red')
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y[:Z.size].reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Directional derivatives in 1st direction')
       plt.figure()
       h = plt.pcolormesh(X,Y,y[Z.size:].reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Directional derivatives in 2nd direction')
       plt.show()


    Notes
    -----
    The ``DirectionalGradient`` of a multivariate function :math:`f(\mathbf{x})` is defined as:

    .. math::
        d_{\mathbf{v}_1(\mathbf{x}),\ldots,\mathbf{v}_N(\mathbf{x})} f =
            \left[\begin{array}{c}
            \langle\nabla f, \mathbf{v}_1(\mathbf{x})\rangle\\
            \vdots\\
            \langle\nabla f, \mathbf{v}_N(\mathbf{x})\rangle
            \end{array}\right],

    where :math:`d_\mathbf{v}` is the first-order directional derivative
    implemented by :py:func:`~pycsou.linop.diff.FirstDirectionalDerivative`.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.Gradient`, :py:func:`~pycsou.linop.diff.FirstDirectionalDerivative`

    """
    return LinOpVStack(*first_directional_derivatives)


def DirectionalLaplacian(second_directional_derivatives: List[SecondDirectionalDerivative],
                         weights: Optional[Iterable[float]] = None) -> LinearOperator:
    r"""
    Directional Laplacian.

    Sum of the second directional derivatives of a multi-dimensional array (at least two dimensions are required)
    along multiple ``directions`` for each entry of the array.


    Parameters
    ----------
    second_directional_derivatives: List[SecondDirectionalDerivative]
        List of the second directional derivatives to be summed.
    weights: Optional[Iterable[float]]
        List of optional positive weights with which each second directional derivative operator is multiplied.

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
         Directional Laplacian.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import SecondDirectionalDerivative, DirectionalLaplacian
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       directions1 = np.zeros(shape=(2,Z.size))
       directions1[0, :Z.size//2] = 1
       directions1[1, Z.size//2:] = 1
       directions2 = np.zeros(shape=(2,Z.size))
       directions2[1, :Z.size//2] = -1
       directions2[0, Z.size//2:] = -1
       Dop1 = SecondDirectionalDerivative(shape=Z.shape, directions=directions1)
       Dop2 = SecondDirectionalDerivative(shape=Z.shape, directions=directions2)
       Dop = DirectionalLaplacian([Dop1, Dop2])
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.quiver(x, x, directions1[1].reshape(X.shape), directions1[0].reshape(X.shape))
       plt.quiver(x, x, directions2[1].reshape(X.shape), directions2[0].reshape(X.shape), color='red')
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Directional Laplacian')
       plt.show()


    Notes
    -----
    The ``DirectionalLaplacian`` of a multivariate function :math:`f(\mathbf{x})` is defined as:

    .. math::
        d^2_{\mathbf{v}_1(\mathbf{x}),\ldots,\mathbf{v}_N(\mathbf{x})} f =
            -\sum_{n=1}^N
            d^\ast_{\mathbf{v}_n(\mathbf{x})}(d_{\mathbf{v}_n(\mathbf{x})} f).

    where :math:`d_\mathbf{v}` is the first-order directional derivative
    implemented by :py:func:`~pycsou.linop.diff.FirstDirectionalDerivative`.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.SecondDirectionalDerivative`, :py:func:`~pycsou.linop.diff.Laplacian`

    """
    directional_laplacian = second_directional_derivatives[0]
    if weights is None:
        weights = [1.] * len(second_directional_derivatives)
    else:
        if len(weights) != len(second_directional_derivatives):
            raise ValueError('The number of weights and operators provided differ.')
    for i in range(1, len(second_directional_derivatives)):
        directional_laplacian += weights[i] * second_directional_derivatives[i]
    return directional_laplacian


def Gradient(shape: tuple, step: Union[tuple, float] = 1., edge: bool = True, dtype: str = 'float64',
             kind: str = 'centered') -> PyLopLinearOperator:
    r"""
    Gradient.

    Computes the gradient of a multi-dimensional array (at least two dimensions are required).


    Parameters
    ----------
    shape: tuple
        Shape of the input array.
    step: Union[float, Tuple[float, ...]]
        Step size in each direction.
    edge: bool
        For ``kind = 'centered'``, use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input vector.
    kind: str
        Derivative kind (``forward``, ``centered``, or ``backward``).

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
        Gradient operator.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pycsou.linop.diff import Gradient, FirstDerivative
       from pycsou.util.misc import peaks

    .. doctest::

       >>> x = np.linspace(-2.5, 2.5, 100)
       >>> X,Y = np.meshgrid(x,x)
       >>> Z = peaks(X, Y)
       >>> Nabla = Gradient(shape=Z.shape, kind='forward')
       >>> D = FirstDerivative(size=Z.size, shape=Z.shape, kind='forward')
       >>> np.allclose((Nabla * Z.flatten())[:Z.size], D * Z.flatten())
       True

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import Gradient
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       Dop = Gradient(shape=Z.shape)
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.colorbar(h)
       plt.title('Signal')
       plt.figure()
       h = plt.pcolormesh(X,Y,y[:Z.size].reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Gradient (1st component)')
       plt.figure()
       h = plt.pcolormesh(X,Y,y[Z.size:].reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Gradient (2nd component)')
       plt.show()


    Notes
    -----
    The ``Gradient`` operator applies a first-order derivative to each dimension of
    a multi-dimensional array in forward mode.

    For simplicity, given a three dimensional array, the ``Gradient`` in forward
    mode using a centered stencil can be expressed as:

    .. math::
        \mathbf{g}_{i, j, k} =
            (f_{i+1, j, k} - f_{i-1, j, k}) / d_1 \mathbf{i_1} +
            (f_{i, j+1, k} - f_{i, j-1, k}) / d_2 \mathbf{i_2} +
            (f_{i, j, k+1} - f_{i, j, k-1}) / d_3 \mathbf{i_3}

    which is discretized as follows:

    .. math::
        \mathbf{g}  =
        \begin{bmatrix}
           \mathbf{df_1} \\
           \mathbf{df_2} \\
           \mathbf{df_3}
        \end{bmatrix}.

    In adjoint mode, the adjoints of the first derivatives along different
    axes are instead summed together.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.DirectionalGradient`, :py:func:`~pycsou.linop.diff.FirstDerivative`

    """
    return PyLopLinearOperator(pylops.Gradient(dims=shape, sampling=step, edge=edge, dtype=dtype, kind=kind))


def Laplacian(shape: tuple, weights: Tuple[float] = (1, 1), step: Union[tuple, float] = 1., edge: bool = True,
              dtype: str = 'float64') -> PyLopLinearOperator:
    r"""
    Laplacian.

    Computes the Laplacian of a 2D array.


    Parameters
    ----------
    shape: tuple
        Shape of the input array.
    weights: Tuple[float]
        Weight to apply to each direction (real laplacian operator if ``weights=[1,1]``)
    step: Union[float, Tuple[float, ...]]
        Step size in each direction.
    edge: bool
       Use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input vector.
    kind: str
        Derivative kind (``forward``, ``centered``, or ``backward``).

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
        Laplacian operator.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import Laplacian
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 25)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       Dop = Laplacian(shape=Z.shape)
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.colorbar(h)
       plt.title('Signal')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Laplacian')
       plt.show()


    Notes
    -----
    The Laplacian operator sums the second directional derivatives of a 2D array along the two canonical directions.

    It is defined as:

    .. math::
        y[i, j] =\frac{x[i+1, j] + x[i-1, j] + x[i, j-1] +x[i, j+1] - 4x[i, j]}
                  {dx\times dy}.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.DirectionalLaplacian`, :py:func:`~pycsou.linop.diff.SecondDerivative`

    """
    if isinstance(step, Number):
        step = [step] * len(shape)
    return PyLopLinearOperator(pylops.Laplacian(dims=shape, weights=weights, sampling=step, edge=edge, dtype=dtype))


def GeneralisedLaplacian(shape: Optional[tuple] = None, step: Union[tuple, float] = 1., edge: bool = True,
                         dtype: str = 'float64', kind='iterated', **kwargs) -> LinearOperator:
    r"""
    Generalised Laplacian operator.

    Generalised Laplacian operator for a 2D array.

    Parameters
    ----------
    shape: tuple
        Shape of the input array.
    step: Union[tuple, float] = 1.
        Step size for each dimension.
    edge: bool
        Use reduced order derivative at edges (``True``) or ignore them (``False``).
    dtype: str
        Type of elements in input array.
    kind: str
        Type of generalised differential operator (``'iterated'``, ``'sobolev'``, ``'polynomial'``).
        Depending on the cases, the ``GeneralisedLaplacian`` operator is defined as follows:

        * ``'iterated'``: :math:`\mathscr{D}=\Delta^N`,
        * ``'sobolev'``: :math:`\mathscr{D}=(\alpha^2 \mathrm{Id}-\Delta)^N`, with :math:`\alpha\in\mathbb{R}`,
        * ``'polynomial'``: :math:`\mathscr{D}=\sum_{n=0}^N \alpha_n \Delta^n`,  with :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`,

        where :math:`\Delta` is the :py:func:`~pycsou.linop.diff.Laplacian` operator.

    kwargs: Any
        Additional arguments depending on the value of ``kind``:

        * ``'iterated'``: ``kwargs={order: int}`` where ``order`` defines the exponent :math:`N`.
        * ``'sobolev'``: ``kwargs={order: int, constant: float}`` where ``order`` defines the exponent :math:`N` and ``constant`` the scalar :math:`\alpha\in\mathbb{R}`.
        * ``'polynomial'``: ``kwargs={coeffs: Union[np.ndarray, list, tuple]}`` where ``coeffs`` is an array containing the coefficients :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`.

    Returns
    -------
    :py:class:`pycsou.core.linop.LinearOperator`
        Generalised Laplacian operator.

    Raises
    ------
    NotImplementedError
        If ``kind`` is not one of: ``'iterated'``, ``'sobolev'``, ``'polynomial'``.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import GeneralisedLaplacian
       from pycsou.util.misc import peaks

       x  = np.linspace(-2.5, 2.5, 50)
       X,Y = np.meshgrid(x,x)
       Z = peaks(X, Y)
       Dop = GeneralisedLaplacian(shape=Z.shape, kind='sobolev', order=2, constant=0)
       y = Dop * Z.flatten()

       plt.figure()
       h = plt.pcolormesh(X,Y,Z, shading='auto')
       plt.colorbar(h)
       plt.title('Signal')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       plt.colorbar(h)
       plt.title('Sobolev')
       plt.show()

    Notes
    -----
    Problematic values at edges are set to zero.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.GeneralisedDerivative`, :py:func:`~pycsou.linop.diff.Laplacian`

    """
    Delta = Laplacian(shape=shape, step=step, edge=edge, dtype=dtype)
    Delta.is_symmetric = True
    if kind == 'iterated':
        N = kwargs['order']
        Dgen = Delta ** N
        order = 2 * N
    elif kind == 'sobolev':
        I = IdentityOperator(size=shape[0] * shape[1])
        alpha = kwargs['constant']
        N = kwargs['order']
        Dgen = ((alpha ** 2) * I - Delta) ** N
        order = 2 * N
    elif kind == 'polynomial':
        coeffs = kwargs['coeffs']
        Dgen = PolynomialLinearOperator(LinOp=Delta, coeffs=coeffs)
        order = 2 * (len(coeffs) - 1)
    else:
        raise NotImplementedError(
            'Supported generalised derivative types are: iterated, sobolev, polynomial.')

    kill_edges = np.ones(shape=shape)
    for axis in range(len(shape)):
        kill_edges = np.swapaxes(kill_edges, axis, 0)
        kill_edges[-order:] = 0
        kill_edges[:order] = 0
        kill_edges = np.swapaxes(kill_edges, 0, axis)

    KillEdgeOp = DiagonalOperator(kill_edges.reshape(-1))
    Dgen = KillEdgeOp * Dgen
    return Dgen


def Integration1D(size: int, shape: Optional[tuple] = None, axis: int = 0, step: float = 1.,
                  dtype='float64') -> PyLopLinearOperator:
    r"""
    1D integral/cumsum operator.

    Integrates a multi-dimensional array along a specific ``axis``.

    Parameters
    ----------
    size: int
        Size of the input array.
    shape: Optional[tuple]
        Shape of the input array if multi-dimensional.
    axis: int
        Axis along which integration is performed.
    step: float
        Step size.
    dtype: str
        Type of elements in input array.

    Returns
    -------
    :py:class:`pycsou.linop.base.PyLopLinearOperator`
        Integral operator.

    Examples
    --------

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.linop.diff import Integration1D

       x = np.array([0,0,0,1,0,0,0,0,0,2,0,0,0,0,-1,0,0,0,0,2,0,0,0,0])
       Int = Integration1D(size=x.size)
       y = Int * x
       plt.figure()
       plt.plot(np.arange(x.size), x)
       plt.plot(np.arange(x.size), y)
       plt.legend(['Signal', 'Integral'])
       plt.title('Integration')
       plt.show()

    Notes
    -----
    The ``Integration1D`` operator applies a causal integration to any chosen
    direction of a multi-dimensional array.

    For simplicity, given a one dimensional array, the causal integration is:

    .. math::
        y(t) = \int x(t) dt

    which can be discretised as :

    .. math::
        y[i] = \sum_{j=0}^i x[j] dt,

    where :math:`dt` is the ``sampling`` interval.

    See Also
    --------
    :py:func:`~pycsou.linop.diff.FirstDerivative`
    """
    return PyLopLinearOperator(
        pylops.CausalIntegration(N=size, dims=shape, dir=axis, sampling=step, halfcurrent=False, dtype=dtype))


if __name__ == "__main__":
    pass
