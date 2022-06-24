import collections.abc as cabc
import copy
import functools as ft
import types
import typing as typ
import warnings

import numpy as np
import scipy.linalg as spl
import scipy.sparse.linalg as spsl

import pycsou.abc.arithmetic as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

__all__ = [
    "DiffFunc",
    "DiffMap",
    "Func",
    "LinFunc",
    "LinOp",
    "Map",
    "NormalOp",
    "OrthProjOp",
    "PosDefOp",
    "ProjOp",
    "ProxDiffFunc",
    "ProxFunc",
    "SelfAdjointOp",
    "SquareOp",
    "UnitOp",
]

# Developer-Facing Types ------------------------------------------------------
class Property:
    r"""
    Abstract base class for Pycsou operators.
    """

    @classmethod
    def _property_list(cls) -> cabc.Set[str]:
        # List all possible properties of Pycsou's base operators.
        return frozenset(
            (
                "apply",
                "lipschitz",
                "jacobian",
                "diff_lipschitz",
                "single_valued",
                "grad",
                "prox",
                "adjoint",
            )
        )

    @classmethod
    def properties(cls) -> cabc.MutableSet[str]:
        r"""
        List class properties.
        """
        props = set(dir(cls))
        return props & cls._property_list()

    @classmethod
    def has(cls, prop: pyct.Name) -> bool:
        r"""
        Query class for specific property.

        Parameters
        ----------
        prop: pyct.Name
            Queried properties.

        Example
        -------

        >>> import pycsou.abc as pyca
        >>> pyca.LinOp.has(('adjoint', 'jacobian'))
        True
        >>> pyca.LinOp.has(('adjoint', 'prox'))
        False
        """
        if isinstance(prop, str):
            prop = (prop,)
        return frozenset(prop) <= cls.properties()

    def argscale(self, scalar: pyct.Real) -> pyct.MapT:
        r"""
        Scale the domain of a :py:class:`~pycsou.abc.operator.Map`.

        Parameters
        ----------
        scalar: pyct.Real
            Scale factor

        Returns
        -------
        op: pyct.MapT
            Domain-rescaled operator.

        Notes
        -----
        Calling ``self.argscale(scalar)`` is equivalent to precomposing ``self`` with
        :py:class:`~pycsou.operator.linop.base.HomotethyOp`:
        """
        return pyca.argscale(self, scalar)

    @pycrt.enforce_precision(i="shift", o=False)
    def argshift(self, shift: pyct.NDArray) -> pyct.MapT:
        r"""
        Shift the domain of a :py:class:`~pycsou.abc.operator.Map`.

        Parameters
        ----------
        shift: pyct.NDArray
            (N,) shift

        Returns
        -------
        op: pyct.MapT
            Domain-shifted operator.

        Notes
        -----
        The domain-shifted operator has the same type as ``self``, except if ``self`` is a
        :py:class:`~pycsou.abc.operator.LinFunc` or
        :py:class:`~pycsou.abc.operator.LinOp`
        since shifts do not preserve linearity.

        In the latter cases, the domain-shifted operator is of type
        :py:class:`~pycsou.abc.operator.DiffFunc` and
        :py:class:`~pycsou.abc.operator.DiffMap` respectively.
        """
        return pyca.argshift(self, shift)


class SingleValued(Property):
    r"""
    Mixin class defining the *single-valued* property.
    """

    def single_valued(self) -> bool:
        return True


class Apply(Property):
    r"""
    Mixin class defining the *apply* property.
    """

    @pycrt.enforce_precision(i="arr")
    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Alias for :py:meth:`~pycsou.abc.operator.Apply.apply`.
        """
        return self.apply(arr)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Evaluate operator at specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., M) input points.

        Returns
        -------
        out: pyct.NDArray
            (..., N) output points.


        .. Important::

           This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError

    def lipschitz(self, **kwargs) -> pyct.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A constant :math:`L_\mathbf{h}>0` is said to be a *Lipschitz constant* for a map
          :math:`\mathbf{h}:\mathbb{R}^N\to \mathbb{R}^M` if:

          .. math::

              \|\mathbf{h}(\mathbf{x})-\mathbf{h}(\mathbf{y})\|_{\mathbb{R}^M}
              \leq
              L_\mathbf{h} \|\mathbf{x}-\mathbf{y}\|_{\mathbb{R}^N},
              \qquad
              \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^N,

          where
          :math:`\|\cdot\|_{\mathbb{R}^M}` and
          :math:`\|\cdot\|_{\mathbb{R}^N}`
          are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of a map is called the *optimal Lipschitz constant*.
        """
        raise NotImplementedError


class Differential(Property):
    r"""
    Mixin class defining the *differentiability* properties.
    """

    def jacobian(self, arr: pyct.NDArray) -> pyct.MapT:
        r"""
        Evaluate the Jacobian of a vector-valued differentiable map at the specified point.

        Parameters
        ----------
        arr: pyct.NDArray
            (M,) evaluation point.

        Returns
        -------
        op: pyct.MapT
            (N, M) Jacobian operator at point ``arr``.

        Notes
        -----
        Let :math:`\mathbf{h}=[h_1, \ldots, h_N]: \mathbb{R}^M\to\mathbb{R}^N` be a differentiable
        multidimensional map.
        The *Jacobian* (or *differential*) of :math:`\mathbf{h}` at
        :math:`\mathbf{z}\in\mathbb{R}^M` is defined as the best linear approximator of
        :math:`\mathbf{h}` near :math:`\mathbf{z}`, in the sense that

        .. math::

           \mathbf{h}(\mathbf{x}) - \mathbf{h}(\mathbf{z})
           =
           \mathbf{J}_{\mathbf {h}}(\mathbf{z})(\mathbf{x} -\mathbf{z})+o(\|\mathbf{x} -\mathbf{z} \|)
           \quad
           \text{as} \quad \mathbf {x} \to \mathbf {z}.

        The Jacobian admits the following matrix representation:

        .. math::

           (\mathbf{J}_{\mathbf{h}}(\mathbf{x}))_{ij}
           :=
           \frac{\partial h_i}{\partial x_j}(\mathbf{x}),
           \qquad
           \forall (i,j)\in\{1,\cdots,N\} \times \{1,\cdots,M\}.
        """
        raise NotImplementedError

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        r"""
        Compute a Lipschitz constant of :py:meth:`~pycsou.abc.operator.Differential.jacobian`.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A Lipschitz constant :math:`L_{\mathbf{J}_{\mathbf{h}}}>0` of the Jacobian map
          :math:`\mathbf{J}_{\mathbf{h}}:\mathbf{R}^M\to \mathbf{R}^{N \times M}` is such that:

          .. math::

             \|\mathbf{J}_{\mathbf{h}}(\mathbf{x})-\mathbf{J}_{\mathbf{h}}(\mathbf{y})\|_{\mathbb{R}^{N \times M}}
             \leq
             L_{\mathbf{J}_{\mathbf{h}}} \|\mathbf{x}-\mathbf{y}\|_{\mathbb{R}^M},
             \qquad
             \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^M,

          where
          :math:`\|\cdot\|_{\mathbb{R}^{N \times M}}` and
          :math:`\|\cdot\|_{\mathbb{R}^M}`
          are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of the Jacobian is called the *optimal diff-Lipschitz
          constant*.
        """
        raise NotImplementedError


class Gradient(Differential):
    r"""
    Mixin class defining the *grad* property.
    """

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.MapT:
        r"""
        Construct the Jacobian linear functional associated with the gradient.

        Parameters
        ----------
        arr: pyct.NDArray
            (M,) evaluation point.

        Returns
        -------
        op: pyct.MapT
            (1, M) Jacobian linear functional.

        Notes
        -----
        The Jacobian of a functional :math:`f:\mathbb{R}^M\to \mathbb{R}` is given, for every
        :math:`\mathbf{x}\in\mathbb{R}^M`, by

        .. math::

           \mathbf{J}_f(\mathbf{x})(\mathbf{z})
           =
           \langle \mathbf{z}, \nabla f (\mathbf{x}) \rangle
           =
           \nabla f (\mathbf{x})^T\mathbf{z},
           \qquad
           \forall \mathbf{z}\in\mathbb{R}^M,

        where :math:`\nabla f (\mathbf{x})` denotes the *gradient* of :math:`f`
        (see :py:meth:`~pycsou.abc.operator.Gradient.grad`).

        The Jacobian matrix is hence given by the transpose of the gradient:
        :math:`\mathbf{J}_f(\mathbf{x})=\nabla f (\mathbf{x})^T`.
        """
        from pycsou.operator.linop.base import ExplicitLinFunc

        return ExplicitLinFunc(self.grad(arr))

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Evaluate operator gradient at specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., M) input points.

        Returns
        -------
        out: pyct.NDArray
            (..., N) gradients.

        Notes
        -----
        The gradient of a functional :math:`f:\mathbb{R}^M\to \mathbb{R}` is given, for every
        :math:`\mathbf{x}\in\mathbb{R}^M`, by

        .. math::

           \nabla f(\mathbf{x})
           :=
           \left[\begin{array}{c}
           \frac{\partial f}{\partial x_1}(\mathbf{x}) \\
           \vdots \\
           \frac{\partial f}{\partial x_M}(\mathbf{x})
           \end{array}\right].

        .. Important::

           This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError


class Adjoint(Property):
    r"""
    Mixin class defining the *adjoint* property.
    """

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Evaluate operator adjoint at specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., N) input points.

        Returns
        -------
        out: pyct.NDArray
            (..., M) adjoint evaluations.

        Notes
        -----
        The *adjoint* :math:`\mathbf{L}^\ast:\mathbb{R}^N\to \mathbb{R}^M` of a linear operator
        :math:`\mathbf{L}:\mathbb{R}^M\to \mathbb{R}^N` is defined as:

        .. math::

           \langle \mathbf{x}, \mathbf{L}^\ast\mathbf{y}\rangle_{\mathbb{R}^M}
           :=
           \langle \mathbf{L}\mathbf{x}, \mathbf{y}\rangle_{\mathbb{R}^N},
           \qquad
           \forall (\mathbf{x},\mathbf{y})\in \mathbb{R}^M \times \mathbb{R}^N.

        .. Important::

           This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError


class Proximal(Property):
    r"""
    Mixin class defining the *proximal* property.
    """

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        r"""
        Evaluate proximity operator of the ``tau``-scaled functional at specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., M) input points.
        tau: pyct.Real
            Positive scale factor.

        Returns
        -------
        out: pyct.NDArray
            (..., M) proximal evaluations.

        Notes
        -----
        For :math:`\tau>0`, the *proximity operator* of a ``tau``-scaled functional
        :math:`f:\mathbb{R}^M\to \mathbb{R}` is defined as:

        .. math::

           \mathbf{\text{prox}}_{\tau f}(\mathbf{z})
           :=
           \arg\min_{\mathbf{x}\in\mathbb{R}^M} f(x)+\frac{1}{2\tau} \|\mathbf{x}-\mathbf{z}\|_2^2,
           \quad
           \forall \mathbf{z}\in\mathbb{R}^M.

        .. Important::

           This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError

    @pycrt.enforce_precision(i=("arr", "sigma"))
    def fenchel_prox(self, arr: pyct.NDArray, sigma: pyct.Real) -> pyct.NDArray:
        r"""
        Evaluate proximity operator of the ``sigma``-scaled Fenchel conjugate of a functional at
        specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., M) input points.
        sigma: pyct.Real
            Positive scale factor

        Returns
        -------
        out: NDArray
            (..., M) proximal evaluations.

        Notes
        -----
        For :math:`\sigma>0`, the *Fenchel conjugate* is defined as:

        .. math::

           f^\ast(\mathbf{z})
           :=
           \max_{\mathbf{x}\in\mathbb{R}^M} \langle \mathbf{x},\mathbf{z} \rangle - f(\mathbf{x}).

        From **Moreau's identity**, its proximal operator is given by:

        .. math::

           \mathbf{\text{prox}}_{\sigma f^\ast}(\mathbf{z})
           =
           \mathbf{z} - \sigma \mathbf{\text{prox}}_{f/\sigma}(\mathbf{z}/\sigma).
        """
        return arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)


# User-instantiable Types -----------------------------------------------------
class Map(Apply):
    r"""
    Base class for real-valued maps :math:`\mathbf{M}:\mathbb{R}^M\to \mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply`.

    If the map is Lipschitz-continuous with known Lipschitz constant, the latter should be stored in
    the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: pyct.Shape
            (N, M) operator shape.
            Shapes of the form (N, None) denote domain-agnostic maps.
        """
        assert len(shape) == 2, f"shape: expected pyct.Shape, got {shape}."
        assert shape[0] is not None, "shape: codomain-agnostic maps are not supported."
        self._shape = tuple(shape)
        self._lipschitz = np.inf

    @property
    def shape(self) -> pyct.Shape:
        r"""
        Return (N, M) operator shape.
        """
        return self._shape

    @property
    def dim(self) -> pyct.Integer:
        r"""
        Return dimension of map's domain. (M)
        """
        return self.shape[1]

    @property
    def codim(self) -> pyct.Integer:
        r"""
        Return dimension of map's co-domain. (N)
        """
        return self.shape[0]

    def squeeze(self) -> pyct.MapT:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.Map` to the right sub-type given codomain dimension.

        Returns
        -------
        op : pycsou.abc.operator.Map | pycsou.abc.operator.Func
            Squeezed operator.

        Example
        -------
        Consider the :py:class:`Median` operator defined in the :ref:`developer-notes`.
        The latter was declared as a :py:class:`~pycsou.abc.operator.Map` subclass, but its
        co-domain has dimension 1.
        It is therefore preferable to see :py:class:`Median` objects as
        :py:class:`~pycsou.abc.operator.Func` objects.
        This recasting can be performed a-posteriori using :py:meth:`squeeze`:

        >>> m = Median()
        >>> type(m)
        pycsou.abc.operator.Map
        >>> type(m.squeeze())
        pycsou.abc.operator.Func
        """
        return self._squeeze(out=Func)

    def _squeeze(self, out: pyct.MapC) -> pyct.MapT:
        if self.codim == 1:
            obj = self.astype(out)
        else:
            obj = self
        return obj

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self._lipschitz

    def astype(self, cast_to: pyct.MapC) -> pyct.MapT:
        r"""
        Recast a :py:class:`~pycsou.abc.operator.Map` (or subclass thereof) to another
        :py:class:`~pycsou.abc.operator.Map` parent/subclass in the hierarchy.

        Parameters
        ----------
        cast_to: pyct.MapC
            Target type for the recast.

        Returns
        -------
        op: pyct.MapT
            Object with the new interface.
            Raises ValueError if `cast_to` is not an ancestor/child-class of ``self``.

        Notes
        -----
        If ``self`` does not implement all methods from ``cast_to``, then unimplemented methods will
        raise a ``NotImplementedError`` when called.
        """
        return self

    @classmethod
    def from_source(cls, shape: pyct.Shape, **kwargs) -> pyct.MapT:
        r"""
        Instantiate a :py:class:`~pycsou.abc.operator.Map` by directly defining the appropriate
        callables.

        Parameters
        ----------
        shape: pyct.Shape
            (N, M) operator shape.
            Shapes of the form (N, None) denote domain-agnostic maps.
        kwargs:
            kwargs corresponding to class callables.

        Returns
        -------
        op: pyct.MapT

        Example (simplified)
        --------------------
        >>> from pycsou.abc.operator import LinFunc
        >>> sh_map = (1, 5)
        >>> map_properties = {
        ...     "apply": lambda x: np.sum(x, axis=-1, keepdims=True),
        ...     "adjoint": lambda x: x * np.ones(sh_map[-1]),
        ...     "grad": lambda x: np.ones(shape=x.shape[:-1] + (sh_map[-1],)),
        ...     "prox": lambda x, tau: x - tau * np.ones(sh_map[-1]),
        ... }
        >>> func = LinFunc.from_source(shape=sh_map, **map_properties)
        >>> type(func)
        <class 'pycsou.abc.operator.LinFunc'>
        >>> func.apply(np.ones((1,5)))
        array([[5.]])
        >>> func.adjoint(1)
        array([1., 1., 1., 1., 1.])
        """
        prop, prop_op = set(kwargs.keys()), set(cls.properties())
        if cls in [LinOp, DiffFunc, ProxDiffFunc, LinFunc]:
            prop_op.discard("jacobian")
            prop_op.discard("single_valued")

        if prop_op == prop:
            out_op = cls(shape)
        else:
            raise ValueError(f"Cannot instantiate {cls.__name__} with provided properties.")

        for p in prop:
            if p in ("lipschitz", "diff_lipschitz"):
                setattr(out_op, "_" + p, kwargs["_" + p])
            elif p == "prox":
                f = lambda key, _, arr, tau: kwargs[key](arr, tau)
                method = types.MethodType(ft.partial(f, p), out_op)
                setattr(out_op, p, method)
            else:
                f = lambda key, _, arr: kwargs[key](arr)
                method = types.MethodType(ft.partial(f, p), out_op)
                setattr(out_op, p, method)
        return out_op

    def __add__(self, other: pyct.MapT) -> pyct.MapT:
        return pyca.add(self, other)

    def __sub__(self, other: pyct.MapT) -> pyct.MapT:
        return pyca.sub(self, other)

    def __neg__(self) -> pyct.MapT:
        return pyca.neg(self)

    def __mul__(self, other: typ.Union[pyct.MapT, pyct.Real]) -> pyct.MapT:
        return pyca.mul(self, other)

    def __pow__(self, k: pyct.Integer) -> pyct.MapT:
        return pyca.pow(self, k)

    def __matmul__(self, other: pyct.MapT) -> pyct.MapT:
        return NotImplemented

    def __rmatmul__(self, other: pyct.MapT) -> pyct.MapT:
        return NotImplemented


class DiffMap(Map, Differential):
    r"""
    Base class for real-valued differentiable maps :math:`\mathbf{M}:\mathbb{R}^M \to \mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply` and
    :py:meth:`~pycsou.abc.operator.Differential.jacobian`.

    If the map and/or its Jacobian are Lipschitz-continuous with known Lipschitz constants, the
    latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default),
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._diff_lipschitz = np.inf

    def squeeze(self) -> pyct.MapT:
        return self._squeeze(out=DiffFunc)

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        return self._diff_lipschitz


class Func(Map, SingleValued):
    r"""
    Base class for real-valued functionals :math:`f:\mathbb{R}^M \to \mathbb{R}\cup\{+\infty\}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply`.

    If the functional is Lipschitz-continuous with known Lipschitz constant, the latter should be
    stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        assert self.codim == 1, f"shape: expected (1, n), got {shape}."

    def asloss(self, data: pyct.NDArray = None) -> pyct.MapT:
        """
        Transform a functional into a loss functional.

        Parameters
        ----------
        data: pyct.NDArray
            (M,) input.

        Returns
        -------
        op: pyct.MapT
            (1, M) loss function.
            If `data = None`, then return `self`.
        """
        raise NotImplementedError


class ProxFunc(Func, Proximal):
    r"""
    Base class for real-valued proximable functionals
    :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}`.

    A functional :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}` is said *proximable* if its
    **proximity operator** (see :py:meth:`~pycsou.abc.operator.Proximal.prox` for a definition)
    admits
    a *simple closed-form expression*
    **or**
    can be evaluated *efficiently* and with *high accuracy*.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply` and
    :py:meth:`~pycsou.abc.operator.Proximal.prox`.

    If the functional is Lipschitz-continuous with known Lipschitz constant, the latter should be
    stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)

    @pycrt.enforce_precision(i="mu", o=False)
    def moreau_envelope(self, mu: pyct.Real) -> pyct.MapT:
        r"""
        Approximate proximable functional by its *Moreau envelope*.

        Parameters
        ----------
        mu: pyct.Real
            Positive regularization parameter.

        Returns
        -------
        op: pyct.MapT
            Differential Moreau envelope.

        Notes
        -----
        Consider a convex non-smooth proximable functional
        :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}`
        and a regularization parameter :math:`\mu>0`.
        Then, the :math:`\mu`-*Moreau envelope* (or *Moreau-Yoshida envelope*) of
        :math:`f` is given by

        .. math::

           f^\mu(\mathbf{x})
           =
           \min_{\mathbf{z}\in\mathbb{R}^M}
           f(\mathbf{z})
           \quad+\quad
           \frac{1}{2\mu}\|\mathbf{x}-\mathbf{z}\|^{2}.

        The parameter :math:`\mu` controls the trade-off between
        the regularity properties of :math:`f^\mu`
        and
        the approximation error incurred by the Moreau-Yoshida regularization.

        The Moreau envelope inherits the convexity of :math:`f` and is gradient Lipschitz (with
        Lipschitz constant :math:`\mu^{-1}`), even if :math:`f` is non-smooth.
        Its gradient is moreover given by:

        .. math::

           \nabla f^\mu(\mathbf{x})
           =
           \mu^{-1} \left(\mathbf{x} - \text{prox}_{\mu f}(\mathbf{x})\right).

        In addition, :math:`f^\mu` envelopes :math:`f` from below:
        :math:`f^\mu(\mathbf{x})\leq f(\mathbf{x})`.
        This envelope becomes tighter as :math:`\mu\to 0`:

        .. math::

           \lim_{\mu\to 0} f^\mu(\mathbf{x}) = f(\mathbf{x}).

        Finally, it can be shown that the minimizers of :math:`f` and :math:`f^\mu` coincide, and
        that the Fenchel conjugate of :math:`f^\mu` is strongly-convex.

        Example
        -------
        In the example below we construct and plot the Moreau envelope of the :math:`\ell_1`-norm:

        .. plot::

            import numpy as np
            import matplotlib. pyplot as plt
            from pycsou.abc import ProxFunc

            class L1Norm(ProxFunc):
                def __init__(self):
                    super().__init__(shape=(1, None))
                    self._lipschitz = 1
                def apply(self, arr):
                    return np.linalg.norm(arr, axis=-1, keepdims=True, ord=1)
                def prox(self, arr, tau):
                    return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)

            l1_norm = L1Norm()
            mus = [0.1, 0.5, 1]
            smooth_l1_norms = [l1_norm.moreau_envelope(mu) for mu in mus]

            x = np.linspace(-1,1, 512)[:, None]
            labels=['mu=0']
            labels.extend([f'mu={mu}' for mu in mus])
            plt.figure()
            plt.plot(x, l1_norm(x))
            for f in smooth_l1_norms:
                plt.plot(x, f(x))
            plt.legend(labels)
            plt.title('Moreau Envelope')

            labels=[f'mu={mu}' for mu in mus]
            plt.figure()
            for f in smooth_l1_norms:
                plt.plot(x, f.grad(x))
            plt.legend(labels)
            plt.title('Derivative of Moreau Envelope')
        """
        assert mu > 0, f"mu: expected positive, got {mu}"

        op = DiffFunc(self.shape)

        @pycrt.enforce_precision(i="arr")
        def op_apply(mu, self, _, arr):
            xp = pycu.get_array_module(arr)
            x = self.prox(arr, tau=mu)
            return self.apply(x) + (1 / (2 * mu)) * xp.linalg.norm(arr - x, axis=-1, keepdims=True) ** 2

        @pycrt.enforce_precision(i="arr")
        def op_grad(mu, self, _, arr):
            x = self.prox(arr, tau=mu)
            return (arr - x) / mu

        op.apply = types.MethodType(ft.partial(op_apply, mu, self), op)
        op.grad = types.MethodType(ft.partial(op_grad, mu, self), op)
        op._diff_lipschitz = 1 / mu
        return op


class DiffFunc(DiffMap, Func, Gradient):
    r"""
    Base class for real-valued differentiable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply` and
    :py:meth:`~pycsou.abc.operator.Gradient.grad`.

    If the functional and/or its derivative are Lipschitz-continuous with known Lipschitz constants,
    the latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default) and
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)


class ProxDiffFunc(ProxFunc, DiffFunc):
    r"""
    Base class for real-valued differentiable *and* proximable functionals
    :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply`,
    :py:meth:`~pycsou.abc.operator.Gradient.grad`, and
    :py:meth:`~pycsou.abc.operator.Proximal.prox`.

    If the functional and/or its derivative are Lipschitz-continuous with known Lipschitz constants,
    the latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default) and
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)


class LinOp(DiffMap, Adjoint):
    r"""
    Base class for real-valued linear operators :math:`L:\mathbb{R}^M\to\mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply` and
    :py:meth:`~pycsou.abc.operator.Adjoint.adjoint`.

    If known, the Lipschitz constant of the linear map should be stored in the private instance
    attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).

    The Jacobian of a linear map :math:`\mathbf{h}` is constant.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._diff_lipschitz = 0

    def squeeze(self) -> pyct.MapT:
        return self._squeeze(out=LinFunc)

    def jacobian(self, arr: pyct.NDArray) -> pyct.MapT:
        return self

    @property
    def T(self) -> pyct.MapT:
        r"""
        Return the (M, N) adjoint of the linear operator.
        """
        adj = copy.copy(self)
        adj._shape = self.dim, self.codim
        adj.apply = self.adjoint
        adj.adjoint = self.apply
        return adj

    def to_sciop(
        self,
        dtype: pyct.DType = None,
        gpu: bool = False,
    ) -> spsl.LinearOperator:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.LinOp` to a
        :py:class:`scipy.sparse.linalg.LinearOperator`, compatible with the matrix-free linear
        algebra routines of `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.

        Parameters
        ----------
        dtype: pyct.DType
            Working precision of the linear operator.
        gpu: bool
            Operate on CuPy inputs (True) vs. NumPy inputs (False).

        Returns
        -------
        op: [cupyx.]scipy.sparse.linalg.LinearOperator
            Linear operator object compliant with SciPy's interface.
        """

        matmat = lambda arr: self.apply(arr.T).T

        rmatmat = lambda arr: self.adjoint(arr.T).T

        if dtype is None:
            dtype = pycrt.getPrecision().value

        if gpu:
            assert pycd.CUPY_ENABLED
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = spsl
        return spx.LinearOperator(
            shape=self.shape,
            matvec=self.apply,
            rmatvec=self.adjoint,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=dtype,
        )

    def lipschitz(
        self,
        recompute: bool = False,
        algo: str = "svds",
        **kwargs,
    ) -> pyct.Real:
        r"""
        Return a (not necessarily optimal) Lipschitz constant of the operator.

        Parameters
        ----------
        recompute: bool
            If ``True``, forces re-estimation of the Lipschitz constant.
            If ``False``, use the last-computed Lipschitz constant.
        algo: 'svds', 'fro'
            Algorithm used for computing the Lipschitz constant.

            If ``algo==svds``, the Lipschitz constant is estimated as the spectral norm of :math:`L`
            via :py:func:`scipy.sparse.linalg.svds`. (Accurate, but compute-intensive.)

            If ``algo==fro``, the Lipschitz constant is estimated as the Froebenius norm of
            :math:`L` via :py:func:`pycsou.math.linalg.hutchpp`. (Cheaper & less accurate than
            SVD-based method.)
        kwargs:
            Optional kwargs passed on to :py:func:`scipy.sparse.linalg.svds` or
            :py:func:`pycsou.math.linalg.hutchpp`.

        Returns
        -------
        L : pyct.Real
            Value of the Lipschitz constant.

        Notes
        -----
        * The tightest Lipschitz constant is given by the spectral norm of the operator :math:`L`:
          :math:`\|L\|_2`.
          It can be computed via the SVD, which is compute-intensive task for large operators.
          In this setting, it may be advantageous to overestimate the Lipschitz constant with the
          Frobenius norm of :math:`L` since :math:`\|L\|_F \geq \|L\|_2`.

          :math:`\|L\|_F` can be efficiently approximated by computing the trace of :math:`L^\ast L`
          (or :math:`LL^\ast`) via the `Hutch++ stochastic algorithm <https://arxiv.org/abs/2010.09649>`_.

        * :math:`\|L\|_F` is upper bounded by :math:`\|L\|_F \leq \sqrt{n} \|L\|_2`, where the
          equality is reached (worst-case scenario) when the eigenspectrum of the linear operator is
          flat.
        """
        if recompute or (self._lipschitz == np.inf):
            if algo == "fro":
                import pycsou.math.linalg as pycl

                kwargs.update(m=kwargs.get("m", 126))
                op = self.gram() if (self.codim >= self.dim) else self.cogram()
                self._lipschitz = pycl.hutchpp(op, **kwargs)
            elif algo == "svds":
                kwargs.update(k=1, which="LM")
                self._lipschitz = self.svdvals(**kwargs).item()
            else:
                raise NotImplementedError
        return self._lipschitz

    def svdvals(
        self,
        k: pyct.Integer,
        which: str = "LM",
        gpu: bool = False,
        **kwargs,
    ) -> pyct.NDArray:
        r"""
        Compute the ``k`` largest or smallest singular values of the linear operator.

        Parameters
        ----------
        k: pyct.Integer
            Number of singular values to compute.
        which: 'LM' | 'SM'
            Which k singular values to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude
        gpu: bool
            If ``True`` the singular value decomposition is performed on the GPU.
        kwargs:
            Additional kwargs accepted by :py:func:`scipy.sparse.linalg.svds`.

        Returns
        -------
        D: pyct.NDArray
            (k,) singular values in ascending order.
        """

        def _dense_eval():
            if gpu:
                import cupy as xp
                import cupyx.scipy.linalg as spx
            else:
                xp, spx = np, spl
            op = self.asarray(xp=xp, dtype=pycrt.getPrecision().value)
            return spx.svdvals(op)

        def _sparse_eval():
            if gpu:
                assert pycd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx
            else:
                spx = spsl
            op = self.to_sciop(gpu=gpu, dtype=pycrt.getPrecision().value)
            kwargs.update(k=k, which=which, return_singular_vectors=False)
            return spx.svds(op, **kwargs)

        if k >= min(self.shape):
            msg = "Too many svdvals wanted: using matrix-based ops."
            warnings.warn(msg, UserWarning)
            D = _dense_eval()
        else:
            D = _sparse_eval()

        # Filter to k largest/smallest magnitude + sorted
        xp = pycu.get_array_module(D)
        D = D[xp.argsort(D)]
        return D[:k] if (which == "SM") else D[-k:]

    def asarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
    ) -> pyct.NDArray:
        r"""
        Matrix representation of the linear operator.

        Parameters
        ----------
        xp: pyct.ArrayModule
            Which array module to use to represent the output.
        dtype: pyct.DType
            Optional type of the array.

        Returns
        -------
        A: NDArray
            (codim, dim) array-representation of the operator.
        """
        if dtype is None:
            dtype = pycrt.getPrecision().value
        with pycrt.EnforcePrecision(False):
            E = xp.eye(self.dim, dtype=dtype)
            A = self.apply(E).T
        return A

    def __array__(self, dtype: pyct.DType = None) -> pyct.NDArray:
        r"""
        Coerce linear operator to a :py:class:`numpy.ndarray`.

        Parameters
        ----------
        dtype: pyct.DType
            Optional type of the array

        Returns
        -------
        A : numpy.ndarray
            (codim, dim) representation of the linear operator, stored as a NumPy array.

        Notes
        -----
        Functions like ``np.array`` or  ``np.asarray`` will check for the existence of the
        ``__array__`` protocol to know how to coerce the custom object fed as input into an array.
        """
        return self.asarray(xp=np, dtype=dtype)

    def gram(self) -> pyct.MapT:
        r"""
        Gram operator :math:`L^\ast L:\mathbb{R}^M\to \mathbb{R}^M`.

        Returns
        -------
        op: pyct.MapT
            (M, M) Gram operator.

        Notes
        -----
        By default the Gram is computed by the composition ``self.T * self``.
        This may not be the fastest way to compute the Gram operator.
        If the Gram can be computed more efficiently (e.g. with a convolution), the user should
        re-define this method.
        """
        return (self.T * self).astype(SelfAdjointOp)

    def cogram(self) -> pyct.MapT:
        r"""
        Co-Gram operator :math:`LL^\ast:\mathbb{R}^N\to \mathbb{R}^N`.

        Returns
        -------
        op: pyct.MapT
            (N, N) Co-Gram operator.

        Notes
        -----
        By default the co-Gram is computed by the composition ``self * self.T``.
        This may not be the fastest way to compute the co-Gram operator.
        If the co-Gram can be computed more efficiently (e.g. with a convolution), the user should
        re-define this method.
        """
        return (self * self.T).astype(SelfAdjointOp)

    @pycrt.enforce_precision(i=["arr", "damp"], allow_None=True)
    def pinv(
        self,
        arr: pyct.NDArray,
        damp: pyct.Real = None,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pyct.NDArray:
        r"""
        Evaluate the Moore-Penrose pseudo-inverse :math:`L^\dagger` at specified point(s).

        Parameters
        ----------
        arr: pyct.NDArray
            (..., N) input points.
        damp: pyct.Real
            Positive dampening factor regularizing the pseudo-inverse in case of ill-conditioning.
        kwargs_init: cabc.Mapping
            Optional kwargs to be passed to :py:meth:`pycsou.opt.solver.cg.CG.__init__`.
        kwargs_fit: cabc.Mapping
            Optional kwargs to be passed to :py:meth:`pycsou.opt.solver.cg.CG.fit`.

        Returns
        -------
        out: pyct.NDArray
            (..., M) pseudo-inverse(s).

        Notes
        -----
        The Moore-Penrose pseudo-inverse of an operator :math:`L:\mathbb{R}^M\to \mathbb{R}^N` is
        defined as the operator :math:`L^\dagger:\mathbb{R}^N\to \mathbb{R}^M` verifying the
        Moore-Penrose conditions:

            1. :math:`LL^\dagger L =L`,
            2. :math:`L^\dagger LL^\dagger =L^\dagger`,
            3. :math:`(L^\dagger L)^\ast=L^\dagger L`,
            4. :math:`(LL^\dagger)^\ast=LL^\dagger`.

        This operator exists and is unique for any finite-dimensional linear operator.
        The action of the pseudo-inverse :math:`L^\dagger \mathbf{y}` for every
        :math:`\mathbf{y}\in\mathbb{R}^N` can be computed in matrix-free fashion by solving the
        *normal equations*:

        .. math::

           L^\ast L \mathbf{x}= L^\ast \mathbf{y}
           \quad\Leftrightarrow\quad
           \mathbf{x}=L^\dagger \mathbf{y},
           \quad
           \forall (\mathbf{x},\mathbf{y})\in\mathbb{R}^M\times\mathbb{R}^N.

        In the case of severe ill-conditioning, it is also possible to consider the dampened normal
        equations for a numerically-stabler approximation of :math:`L^\dagger \mathbf{y}`:

        .. math::

            (L^\ast L + \tau I) \mathbf{x}= L^\ast \mathbf{y},

        where :math:`\tau>0` corresponds to the ``damp`` parameter.
        """
        from pycsou.operator.linop.base import IdentityOp
        from pycsou.opt.solver.cg import CG
        from pycsou.opt.stop import MaxIter

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init
        b = self.adjoint(arr)
        if damp is not None:
            A = self.gram() + (IdentityOp(shape=(self.dim, self.dim)) * damp)
        else:
            A = self.gram()
        kwargs_init.update(show_progress=kwargs_init.get("show_progress", False))
        cg = CG(A, **kwargs_init)
        if "stop_crit" not in kwargs_fit:
            # .pinv() may not have sufficiently converged given the default CG stopping criteria. To
            # avoid infinite loops, CG iterations are thresholded.
            sentinel = MaxIter(n=20 * A.dim)
            kwargs_fit["stop_crit"] = cg.default_stop_crit() | sentinel
        cg.fit(b=b, **kwargs_fit)
        return cg.solution()

    def dagger(
        self,
        damp: pyct.Real = None,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pyct.MapT:
        r"""
        Return the Moore-Penrose pseudo-inverse operator :math:`L^\dagger`.

        Parameters
        ----------
        damp: pyct.Real
            Positive dampening factor regularizing the pseudo-inverse in case of ill-conditioning.
        kwargs_init: cabc.Mapping
            Optional kwargs to be passed to :py:meth:`pycsou.opt.solver.cg.CG.__init__`.
        kwargs_fit: cabc.Mapping
            Optional kwargs to be passed to :py:meth:`pycsou.opt.solver.cg.CG.fit`.

        Returns
        -------
        op: pyct.MapT
            (M, N) Moore-Penrose pseudo-inverse operator.
        """
        dagger = LinOp(shape=(self.dim, self.codim))
        dagger.apply = types.MethodType(
            ft.partial(
                lambda damp, kwargs_init, kwargs_fit, _, arr: self.pinv(
                    arr,
                    damp,
                    kwargs_init,
                    kwargs_fit,
                ),
                damp,
                kwargs_init,
                kwargs_fit,
            ),
            dagger,
        )
        dagger.adjoint = types.MethodType(
            ft.partial(
                lambda damp, kwargs_init, kwargs_fit, _, arr: self.T.pinv(
                    arr,
                    damp,
                    kwargs_init,
                    kwargs_fit,
                ),
                damp,
                kwargs_init,
                kwargs_fit,
            ),
            dagger,
        )
        return dagger

    @classmethod
    def from_sciop(cls, sp_op: spsl.LinearOperator) -> pyct.MapT:
        r"""
        Cast a :py:class:`scipy.sparse.linalg.LinearOperator` to a
        :py:class:`~pycsou.abc.operator.LinOp`.

        Parameters
        ----------
        sp_op: [scipy|cupyx].sparse.linalg.LinearOperator
            (N, M) Linear operator compliant with SciPy's interface.

        Returns
        -------
        op: pyct.MapT

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`,
        :py:meth:`~pycsou.abc.operator.Map.from_source`,
        :py:meth:`~pycsou.abc.operator.LinOp.to_sciop`.
        """
        if sp_op.dtype not in [_.value for _ in pycrt.Width]:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)

        # [r]matmat only accepts 2D inputs -> reshape apply|adjoint inputs as needed.

        @pycrt.enforce_precision(i="arr")
        def apply(self, arr):
            if _1d := arr.ndim == 1:
                arr = arr.reshape((1, arr.size))
            out = sp_op.matmat(arr.T).T
            if _1d:
                out = out.squeeze(axis=0)
            return out

        @pycrt.enforce_precision(i="arr")
        def adjoint(self, arr):
            if _1d := arr.ndim == 1:
                arr = arr.reshape((1, arr.size))
            out = sp_op.rmatmat(arr.T).T
            if _1d:
                out = out.squeeze(axis=0)
            return out

        op = cls(shape=sp_op.shape)
        setattr(op, "apply", types.MethodType(apply, op))
        setattr(op, "adjoint", types.MethodType(adjoint, op))
        return op

    @classmethod
    def from_array(
        cls,
        A: pyct.NDArray,
        enable_warnings: bool = True,
    ) -> pyct.MapT:
        r"""
        Instantiate a :py:class:`~pycsou.abc.operator.LinOp` from its array representation.

        Parameters
        ----------
        A: pyct.NDArray
            (N, M) array

        Returns
        -------
        op: pyct.MapT
            (N, M) linear operator

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_sciop`,
        :py:meth:`~pycsou.abc.operator.Map.from_source`.
        """
        from pycsou.operator.linop.base import ExplicitLinOp

        return ExplicitLinOp(A, enable_warnings).astype(cls)


class LinFunc(ProxDiffFunc, LinOp):
    r"""
    Base class for real-valued linear functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply`,
    :py:meth:`~pycsou.abc.operator.Gradient.grad`,
    :py:meth:`~pycsou.abc.operator.Proximal.prox`, and
    :py:meth:`~pycsou.abc.operator.Adjoint.adjoint`.

    The Lipschitz constant of linear functionals should be stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).

    The Lipschitz constant of the gradient is 0 since the latter is constant-valued.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        ProxDiffFunc.__init__(self, shape)
        LinOp.__init__(self, shape)

    def lipschitz(self, **kwargs) -> pyct.Real:
        # 'fro' / 'svds' mode are identical for linfuncs.
        g = self.grad(np.ones(self.dim))
        self._lipschitz = float(np.linalg.norm(g))
        return self._lipschitz

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr - tau * self.grad(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr * self.grad(arr)

    def cogram(self) -> pyct.MapT:
        # Cannot auto-specialize LinFunc to SelfAdjointOp via .astype() since SelfAdjointOp lies
        # on a separate branch of the class hierarchy.
        from pycsou.operator.linop.base import HomothetyOp

        g = self.grad(np.zeros(self.dim))
        return HomothetyOp(cst=(g @ g).item(), dim=1)

    @classmethod
    def from_array(
        cls,
        A: pyct.NDArray,
        enable_warnings: bool = True,
    ) -> pyct.MapT:
        r"""
        Instantiate a :py:class:`~pycsou.abc.operator.LinFunc` from its vectorial representation.

        Parameters
        ----------
        A: pyct.NDArray
            (M,) array

        Returns
        -------
        op: pyct.MapT
            (M,) linear functional
        """
        from pycsou.operator.linop.base import ExplicitLinFunc

        return ExplicitLinFunc(A, enable_warnings).astype(cls)


class SquareOp(LinOp):
    r"""
    Base class for *square* linear operators, i.e. :math:`L:\mathbb{R}^M\to \mathbb{R}^M`
    (endomorphsisms).
    """

    def __init__(self, shape: pyct.Shape):
        shape = tuple(shape)
        super().__init__(shape=shape)
        assert self.dim == self.codim, f"shape: expected (M, M), got ({self.codim, self.dim})."

    def trace(self, **kwargs) -> pyct.Real:
        """
        Approximate trace of a linear operator.

        Parameters
        ----------
        kwargs: cabc.Mapping
            Optional kwargs passed to the algorithm used for computing the trace.

        Returns
        -------
        tr: pyct.Real
            Trace estimate.
        """
        import pycsou.math.linalg as pycl

        return pycl.hutchpp(self, **kwargs)


class NormalOp(SquareOp):
    r"""
    Base class for *normal* operators.

    Notes
    -----
    Normal operators commute with their adjoint, i.e. :math:`LL^\ast=L^\ast L`.
    It is `possible to show <https://www.wikiwand.com/en/Spectral_theorem#/Normal_matrices>`_ that
    an operator is normal iff it is *unitarily diagonalizable*, i.e. :math:`L=UDU^\ast`.
    """

    def _eigvals(
        self,
        k: pyct.Integer,
        which: str,
        gpu: bool,
        symmetric: bool,
        **kwargs,
    ) -> pyct.NDArray:
        def _dense_eval():
            if gpu:
                import cupy as xp
                import cupyx.scipy.linalg as spx
            else:
                xp, spx = np, spl
            op = self.asarray(xp=xp, dtype=pycrt.getPrecision().value)
            f = getattr(spx, "eigvalsh" if symmetric else "eigvals")
            return f(op)

        def _sparse_eval():
            if gpu:
                assert pycd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx
            else:
                spx = spsl
            op = self.to_sciop(pycrt.getPrecision().value, gpu)
            kwargs.update(k=k, which=which, return_eigenvectors=False)
            f = getattr(spx, "eigsh" if symmetric else "eigs")
            return f(op, **kwargs)

        if which not in ("LM", "SM"):
            raise NotImplementedError
        if k >= self.dim - 1:
            msg = "Too many eigvals wanted: performing via matrix-based ops."
            warnings.warn(msg, UserWarning)
            D = _dense_eval()
        else:
            D = _sparse_eval()

        # Filter to k largest/smallest magnitude + sorted
        xp = pycu.get_array_module(D)
        D = D[xp.argsort(xp.abs(D))]
        return D[:k] if (which == "SM") else D[-k:]

    def eigvals(
        self,
        k: pyct.Integer,
        which: str = "LM",
        gpu: bool = False,
        **kwargs,
    ) -> pyct.NDArray:
        r"""
        Find ``k`` eigenvalues of a normal operator.

        Parameters
        ----------
        k: pyct.Integer
            Number of eigenvalues to compute.
        which: ‘LM’ | ‘SM’
            Which ``k`` eigenvalues to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude
        gpu: bool
            If ``True`` the eigenvalue decomposition is performed on the GPU.
        kwargs: dict
            Additional kwargs accepted by :py:func:`scipy.sparse.linalg.eigs`.

        Returns
        -------
        D: NDArray
            (k,) eigenvalues in ascending magnitude order.
        """
        return self._eigvals(k, which, gpu, symmetric=False, **kwargs)

    def cogram(self) -> pyct.MapT:
        return self.gram().astype(cast_to=SelfAdjointOp)


class SelfAdjointOp(NormalOp):
    r"""
    Base class for *self-adjoint* operators, i.e. :math:`L^\ast=L`.
    """

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @property
    def T(self) -> pyct.MapT:
        return self

    def eigvals(
        self,
        k: pyct.Integer,
        which: str = "LM",
        gpu: bool = False,
        **kwargs,
    ) -> pyct.NDArray:
        return self._eigvals(k, which, gpu, symmetric=True, **kwargs)


class UnitOp(NormalOp):
    r"""
    Base class for *unitary* operators, i.e. :math:`LL^\ast=L^\ast L = I`.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self._lipschitz

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = self.adjoint(arr)
        if (damp := kwargs.pop("damp")) is not None:
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.MapT:
        op = self.T
        if (damp := kwargs.pop("damp")) is not None:
            from pycsou.operator.linop.base import HomothetyOp

            op = HomothetyOp(cst=1 / (1 + damp), dim=self.dim) * op
        return op

    def gram(self) -> pyct.MapT:
        from pycsou.operator.linop.base import IdentityOp

        return IdentityOp(shape=self.shape)

    def cogram(self) -> pyct.MapT:
        return self.gram()


class ProjOp(SquareOp):
    r"""
    Base class for *projection* operators.

    Projection operators are *idempotent*, i.e. :math:`L^2=L`.
    """
    pass


class OrthProjOp(ProjOp, SelfAdjointOp):
    r"""
    Base class for *orthogonal projection* operators.

    Orthogonal projection operators are *idempotent* and *self-adjoint*, i.e.
    :math:`L^2=L` and :math:`L^\ast=L`.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self._lipschitz

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = arr.copy()
        if (damp := kwargs.pop("damp")) is not None:
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.MapT:
        op = self
        if (damp := kwargs.pop("damp")) is not None:
            from pycsou.operator.linop.base import HomothetyOp

            op = HomothetyOp(cst=1 / (1 + damp), dim=self.dim) * op
        return op


class PosDefOp(SelfAdjointOp):
    r"""
    Base class for *positive-definite* operators.
    """
    pass


_base_operators = frozenset(
    {
        DiffFunc,
        DiffMap,
        Func,
        LinFunc,
        LinOp,
        Map,
        ProxDiffFunc,
        ProxFunc,
    }
)
