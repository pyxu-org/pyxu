# #############################################################################
# diff.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Discrete differential and integral operators.

This module provides differential operators for discrete signals defined over regular grids or arbitrary meshes (graphs).

Many of the linear operator provided in this module are derived from linear operators from `PyLops <https://pylops.readthedocs.io/en/latest/api/index.html#smoothing-and-derivatives>`_.
"""

import numpy as np
import pylops.signalprocessing as pyconv
import pylops
import pygsp
from typing import Optional, Union, Tuple
from pycsou.core.linop import PyLopLinearOperator, LinearOperator, IdentityOperator, DiagonalOperator


def FirstDerivative(size: int, shape: Optional[tuple] = None, axis=0, step: float = 1.0, edge: bool = True,
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
    :py:class:`~pycsou.core.linop.PyLopLinearOperator`
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


def SecondDerivative(size: int, shape: Optional[tuple] = None, axis=0, step: float = 1.0, edge: bool = True,
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
    :py:class:`~pycsou.core.linop.PyLopLinearOperator`
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


def GeneralisedDerivative(size: int, shape: Optional[tuple] = None, axis=0, step: float = 1.0, edge: bool = True,
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
    py:class:`pycsou.core.linop.LinearOperator`
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
                                    coeffs=1e-5 * (np.array([0, 1 / 4, -1 / 4, 1 / 4, -1 / 4])))
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
    :py:func:`~pycsou.linop.diff.FirstDerivative`, :py:func:`~pycsou.linop.diff.SecondDerivative`

    """
    D = FirstDerivative(size=size, shape=shape, axis=axis, step=step, edge=edge, dtype=dtype, kind=kind_diff)
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
        I = IdentityOperator(size=size)
        Dgen = coeffs[0] * I
        for i in range(1, len(coeffs)):
            Dgen += coeffs[i] * (D ** i)
        order = len(coeffs) - 1
    else:
        raise NotImplementedError(
            'Supported generalised derivative types are: iterated, sobolev, exponential, polynomial.')

    kill_edges = np.ones(shape=(Dgen.shape[0],))
    if kind_diff == 'forward':
        kill_edges[-order:] = 0
    elif kind_diff == 'backward':
        kill_edges[:order] = 0
    elif (kind_diff == 'centered'):
        kill_edges[-order:] = 0
        kill_edges[:order] = 0
    else:
        pass
    KillEdgeOp = DiagonalOperator(kill_edges)
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
        py:class:`pycsou.core.linop.PyLopLinearOperator`
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
           plt.pcolormesh(X,Y,Z, shading='auto')
           h = plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
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
            df_\mathbf{v} =
                \nabla f \mathbf{v},

        or along the directions defined by the unitary vectors
        :math:`\mathbf{v}(x, y)`:

        .. math::
            df_\mathbf{v}(x,y) =
                \nabla f(x,y) \mathbf{v}(x,y)

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
    py:class:`pycsou.core.linop.PyLopLinearOperator`
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
       plt.pcolormesh(X,Y,Z, shading='auto')
       h = plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
       plt.colorbar(h)
       plt.title('Signal and directions of derivatives')
       plt.figure()
       h = plt.pcolormesh(X,Y,y.reshape(X.shape), shading='auto')
       h = plt.quiver(x, x, directions[1].reshape(X.shape), directions[0].reshape(X.shape))
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
        d^2f_\mathbf{v} =
            - D_\mathbf{v}^\ast [D_\mathbf{v} f]

    where :math:`D_\mathbf{v}` is the first-order directional derivative
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
    kill_edges[-2:] = 0
    kill_edges[:2] = 0
    kill_edges[:, -2:] = 0
    kill_edges[:, :2] = 0
    KillEdgeOp = DiagonalOperator(kill_edges.reshape(-1))
    DirD2 = KillEdgeOp * Pylop
    return DirD2


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
