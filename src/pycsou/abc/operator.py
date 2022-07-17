import collections
import collections.abc as cabc
import copy
import enum
import functools as ft
import inspect
import types
import typing as typ
import warnings

import numpy as np
import scipy.linalg as spl
import scipy.sparse.linalg as spsl

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw


class Property(enum.Enum):
    """
    Mathematical properties.
    """

    CAN_EVAL = enum.auto()
    FUNCTIONAL = enum.auto()
    PROXIMABLE = enum.auto()
    DIFFERENTIABLE = enum.auto()
    DIFFERENTIABLE_FUNCTION = enum.auto()
    LINEAR = enum.auto()
    LINEAR_SQUARE = enum.auto()
    LINEAR_NORMAL = enum.auto()
    LINEAR_IDEMPOTENT = enum.auto()
    LINEAR_SELF_ADJOINT = enum.auto()
    LINEAR_POSITIVE_DEFINITE = enum.auto()
    LINEAR_UNITARY = enum.auto()
    QUADRATIC = enum.auto()

    def arithmetic_attributes(self) -> cabc.Set[str]:
        "Attributes affected by arithmetic operations."
        data = collections.defaultdict(list)
        data[self.CAN_EVAL].append("_lipschitz")
        data[self.DIFFERENTIABLE].append("_diff_lipschitz")

        attr = frozenset(data[self])
        return attr

    def arithmetic_methods(self) -> cabc.Set[str]:
        "Instance methods affected by arithmetic operations."
        data = collections.defaultdict(list)
        data[self.CAN_EVAL].extend(["apply", "__call__", "lipschitz"])
        data[self.PROXIMABLE].append("prox")
        data[self.DIFFERENTIABLE].extend(["jacobian", "diff_lipschitz"])
        data[self.DIFFERENTIABLE_FUNCTION].append("grad")
        data[self.LINEAR].append("adjoint")

        meth = frozenset(data[self])
        return meth


class Operator:
    """
    Abstract Base Class for Pycsou operators.

    Goals:

    * enable operator arithmetic.
    * cast operators to specialized forms.
    * attach tags encoding certain mathematical properties.
      Each core sub-class MUST have a unique set of :py:class:`~pycsou.abc.operator.Property`-ies to
      be distinguishable from its peers.
    """

    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: pyct.Shape
            (N, M) operator shape.
            Shapes of the form (N, None) denote domain-agnostic maps.
        """
        assert len(shape) == 2, f"shape: expected {pyct.Shape}, got {shape}."
        assert shape[0] is not None, "shape: codomain-agnostic operators are not supported."
        intify = lambda _: int(_) if (_ is not None) else _
        self._shape = tuple(map(intify, shape))

    # Public Interface --------------------------------------------------------
    @property
    def shape(self) -> pyct.Shape:
        r"""
        Return (N, M) operator shape.
        """
        return self._shape

    @property
    def dim(self) -> pyct.Integer:
        r"""
        Return dimension of operator's domain. (M)
        """
        return self.shape[1]

    @property
    def codim(self) -> pyct.Integer:
        r"""
        Return dimension of operator's co-domain. (N)
        """
        return self.shape[0]

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        "Mathematical properties of the operator."
        return frozenset()

    @classmethod
    def has(cls, prop: typ.Union[pyct.Property, cabc.Collection[pyct.Property]]) -> bool:
        """
        Verify if operator possesses supplied properties.
        """
        if isinstance(prop, Property):
            prop = (prop,)
        return frozenset(prop) <= self.properties()

    def asop(self, cast_to: pyct.OpC) -> pyct.OpT:
        r"""
        Recast an :py:class:`~pycsou.abc.operator.Operator` (or subclass thereof) to another
        :py:class:`~pycsou.abc.operator.Operator`.

        Users may call this method if the arithmetic API yields sub-optimal return types.

        This method is a no-op if `cast_to` is a parent class of ``self``.

        Parameters
        ----------
        cast_to: pyct.OpC
            Target type for the recast.

        Returns
        -------
        op: pyct.OpT
            Operator with the new interface.
            Fails when cast is forbidden. (Ex: Map -> Func if codim > 1)

        Implementation Notes
        --------------------
        * The interface of `cast_to` is provided via encapsulation + forwarding.
        * If ``self`` does not implement all methods from ``cast_to``, then unimplemented methods
          will raise ``NotImplementedError`` when called.
        """
        if cast_to not in _core_operators():
            raise ValueError(f"cast_to: expected a core base-class, got {cast_to}.")

        p_core = frozenset(self.properties())
        p_shell = frozenset(cast_to.properties())
        if p_shell <= p_core:
            # Trying to cast `self` to it's own class or a parent class.
            # Inheritance rules mean the target object already satisfies the intended interface.
            return self
        else:
            # (p_shell > p_core) -> specializing to a sub-class of ``self``
            # OR
            # len(p_shell ^ p_core) > 0 -> specializing to another branch of the class hierarchy.
            op = cast_to(shape=self.shape)
            op._core = self  # for debugging

            # Forward shared arithmetic fields from core to shell.
            for p in p_shell & p_core:
                for a in p.arithmetic_attributes():
                    a_core = getattr(self, a)
                    setattr(op, a, a_core)
                for m in p.arithmetic_methods():
                    m_core = getattr(self, m)
                    setattr(op, m, m_core)
            return op

    # Operator Arithmetic -----------------------------------------------------
    def __add__(self, other: pyct.OpT) -> pyct.OpT:
        """
        Add two operators.

        Parameters
        ----------
        self: pyct.OpT
            (A, B) Left operand.
        other: pyct.OpT
            (C, D) Right operand.

        Returns
        -------
        op: pyct.OpT
            Composite operator ``self + other``

        Notes
        -----
        Operand shapes must be `consistent`, i.e.:

            * have `identical shape`, or
            * be domain-agnostic, or
            * be `range-broadcastable`, i.e. functional + map works.

        .. todo::

           Add dispatch table here once stabilized.
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.add(self._squeeze(), other._squeeze())
        else:
            return NotImplemented

    def __sub__(self, other: pyct.OpT) -> pyct.OpT:
        """
        Subtract two operators.

        Parameters
        ----------
        self: pyct.OpT
            (A, B) Left operand.
        other: pyct.OpT
            (C, D) Right operand.

        Returns
        -------
        op: pyct.OpT
            Composite operator ``self - other``
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.add(self._squeeze(), -other._squeeze())
        else:
            return NotImplemented

    def __neg__(self) -> pyct.OpT:
        """
        Negate an operator.

        Returns
        -------
        op: pyct.OpT
            Composite operator ``-1 * self``.
        """
        import pycsou.abc.arithmetic as arithmetic

        return arithmetic.scale(other._squeeze(), cst=-1)

    def __mul__(self, other: typ.Union[pyct.Real, pyct.OpT]) -> pyct.OpT:
        """
        Compose two operators, or scale an operator by a constant.

        Parameters
        ----------
        self: pyct.OpT
            (A, B) Left operand.
        other: pyct.Real | pyct.OpT
            (1,) scalar, or
            (C, D) Right operand.

        Returns
        -------
        op: pyct.OpT
            (A, B) scaled operator, or
            (A, D) composed operator ``self * other``.

        Notes
        -----
        If called with two operators, their shapes must be `consistent`, i.e.:

            * B == C, or
            * B == None.

        .. todo::

           Add dispatch table here once stabilized.
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.compose(self._squeeze(), other._squeeze())
        elif isinstance(other, pyct.Real):
            return arithmetic.scale(self._squeeze(), cst=other)
        else:
            return NotImplemented

    def __rmul__(self, other: pyct.Real) -> pyct.OpT:
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, pyct.Real):
            return arithmetic.scale(self._squeeze(), cst=other)
        else:
            return NotImplemented

    def __truediv__(self, other: pyct.Real) -> pyct.OpT:
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, pyct.Real):
            return arithmetic.scale(self._squeeze(), cst=1 / other)
        else:
            return NotImplemented

    def __pow__(self, k: pyct.Integer) -> pyct.OpT:
        """
        Exponentiate an operator, i.e. compose it with itself.

        Parameters
        ----------
        k: pyct.Integer
            Number of times the operator is composed with itself.

        Returns
        -------
        op: pyct.OpT
            Exponentiated operator.

        Notes
        -----
        Exponentiation is only allowed for endomorphisms, i.e. square operators.
        Chaining domain-agnostic operators is moreover forbidden.
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(k, pyct.Integer):
            return arithmetic.pow(self._squeeze(), k)
        else:
            return NotImplemented

    def __matmul__(self, other) -> pyct.OpT:
        # (op @ NDArray) unsupported
        return NotImplemented

    def __rmatmul__(self, other) -> pyct.OpT:
        # (NDArray @ op) unsupported
        return NotImplemented

    def argscale(self, scalar: pyct.Real) -> pyct.OpT:
        """
        Scale operator's domain.

        Parameters
        ----------
        scalar: pyct.Real

        Returns
        -------
        op: pyct.OpT
            (N, M) domain-scaled operator.
        """
        import pycsou.abc.arithmetic as arithmetic

        assert isinstance(scalar, pyct.Real)
        return arithmetic.argscale(self._squeeze(), cst=scalar)

    def argshift(self, shift: pyct.NDArray) -> pyct.OpT:
        """
        Shift operator's domain.

        Parameters
        ----------
        shift: pyct.NDArray
            (M,) shift value

        Returns
        -------
        op: pyct.OpT
            (N, M) domain-shifted operator.
        """
        import pycsou.abc.arithmetic as arithmetic

        return arithmetic.argshift(self._squeeze(), cst=shift)

    # Internal Helpers --------------------------------------------------------
    @staticmethod
    def _infer_operator_type(prop: cabc.Collection[pyct.Property]) -> pyct.OpC:
        prop = frozenset(prop)
        for op in _core_operators():
            if op.properties() == prop:
                return op
        else:
            raise ValueError(f"No operator found with properties {prop}.")

    def _squeeze(self) -> pyct.OpT:
        r"""
        Cast an :py:class:`~pycsou.abc.operator.Operator` to the right core operator sub-type given
        codomain dimension.

        This function is meant for internal use only.
        If an end-user had to call it, then it is considered a bug.
        """
        p = set(self.properties())
        if self.codim == 1:
            p.add(Property.FUNCTIONAL)
            if Property.LINEAR in self.properties():
                for p_ in Property:
                    if p_.name.startswith("LINEAR_"):
                        p.discard(p_)
                p.add(Property.PROXIMABLE)
                p.add(Property.DIFFERENTIABLE_FUNCTION)
        elif self.codim == self.dim:
            if Property.LINEAR in self.properties():
                p.add(Property.LINEAR_SQUARE)
        klass = self._infer_operator_type(p)
        return self.asop(klass)

    def __repr__(self) -> str:
        klass = self.__class__.__name__
        return f"{klass}{self.shape}"


class Map(Operator):
    r"""
    Base class for real-valued maps :math:`\mathbf{M}:\mathbb{R}^M\to \mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply`.

    If the map is Lipschitz-continuous with known Lipschitz constant, the latter should be stored in
    the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.CAN_EVAL)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = np.inf

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        """
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

    def __call__(self, arr: pyct.NDArray) -> pyct.NDArray:
        """
        Alias for :py:meth:`~pycsou.abc.operator.Map.apply`.
        """
        return self.apply(arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A constant :math:`L_\mathbf{h}>0` is said to be a *Lipschitz constant* for a map
          :math:`\mathbf{h}:\mathbb{R}^M\to \mathbb{R}^N` if:

          .. math::

              \|\mathbf{h}(\mathbf{x})-\mathbf{h}(\mathbf{y})\|_{\mathbb{R}^N}
              \leq
              L_\mathbf{h} \|\mathbf{x}-\mathbf{y}\|_{\mathbb{R}^M},
              \qquad
              \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^M,

          where
          :math:`\|\cdot\|_{\mathbb{R}^M}` and
          :math:`\|\cdot\|_{\mathbb{R}^N}`
          are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of a map is called the *optimal Lipschitz constant*.
        """
        return self._lipschitz


class Func(Map):
    r"""
    Base class for real-valued functionals :math:`f:\mathbb{R}^M \to \mathbb{R}\cup\{+\infty\}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply`.

    If the functional is Lipschitz-continuous with known Lipschitz constant, the latter should be
    stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.FUNCTIONAL)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        assert self.codim == 1, f"shape: expected (1, n), got {shape}."

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        """
        Transform a functional into a loss functional.

        Parameters
        ----------
        data: pyct.NDArray
            (M,) input.

        Returns
        -------
        op: pyct.OpT
            (1, M) loss function.
            If `data = None`, then return `self`.
        """
        raise NotImplementedError


class DiffMap(Map):
    r"""
    Base class for real-valued differentiable maps :math:`\mathbf{M}:\mathbb{R}^M \to \mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply` and
    :py:meth:`~pycsou.abc.operator.DiffMap.jacobian`.

    If the map and/or its Jacobian are Lipschitz-continuous with known Lipschitz constants, the
    latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default),
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.DIFFERENTIABLE)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._diff_lipschitz = np.inf

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        r"""
        Evaluate the Jacobian of a vector-valued differentiable map at the specified point.

        Parameters
        ----------
        arr: pyct.NDArray
            (M,) evaluation point.

        Returns
        -------
        op: pyct.OpT
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
        Compute a Lipschitz constant of :py:meth:`~pycsou.abc.operator.DiffMap.jacobian`.

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
        return self._diff_lipschitz


class ProxFunc(Func):
    r"""
    Base class for real-valued proximable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}`.

    A functional :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}` is said *proximable* if its
    **proximity operator** (see :py:meth:`~pycsou.abc.operator.ProxFunc.prox` for a definition)
    admits
    a *simple closed-form expression*
    **or**
    can be evaluated *efficiently* and with *high accuracy*.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply` and
    :py:meth:`~pycsou.abc.operator.ProxFunc.prox`.

    If the functional is Lipschitz-continuous with known Lipschitz constant, the latter should be
    stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.PROXIMABLE)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)

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

    @pycrt.enforce_precision(i="mu", o=False)
    def moreau_envelope(self, mu: pyct.Real) -> pyct.OpT:
        r"""
        Approximate proximable functional by its *Moreau envelope*.

        Parameters
        ----------
        mu: pyct.Real
            Positive regularization parameter.

        Returns
        -------
        op: pyct.OpT
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


class DiffFunc(DiffMap, Func):
    r"""
    Base class for real-valued differentiable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply` and
    :py:meth:`~pycsou.abc.operator.DiffFunc.grad`.

    If the functional and/or its derivative are Lipschitz-continuous with known Lipschitz constants,
    the latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default) and
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        p.add(Property.DIFFERENTIABLE_FUNCTION)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        DiffMap.__init__(self, shape)
        Func.__init__(self, shape)

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        return LinFunc.from_array(self.grad(arr))

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
            (..., M) gradients.

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


class ProxDiffFunc(ProxFunc, DiffFunc):
    r"""
    Base class for real-valued differentiable *and* proximable functionals
    :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply`,
    :py:meth:`~pycsou.abc.operator.DiffFunc.grad`, and
    :py:meth:`~pycsou.abc.operator.ProxFunc.prox`.

    If the functional and/or its derivative are Lipschitz-continuous with known Lipschitz constants,
    the latter should be stored in the private instance attributes
    ``_lipschitz`` (initialized to :math:`+\infty` by default) and
    ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        ProxFunc.__init__(self, shape)
        DiffFunc.__init__(self, shape)


class LinOp(DiffMap):
    r"""
    Base class for real-valued linear operators :math:`L:\mathbb{R}^M\to\mathbb{R}^N`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Map.apply` and
    :py:meth:`~pycsou.abc.operator.LinOp.adjoint`.

    If known, the Lipschitz constant of the linear map should be stored in the attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).

    The Jacobian of a linear map :math:`\mathbf{h}` is constant.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._diff_lipschitz = 0

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

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        return self

    @property
    def T(self) -> pyct.OpT:
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
                from pycsou.math.linalg import hutchpp

                kwargs.update(m=kwargs.get("m", 126))
                op = self.gram() if (self.codim >= self.dim) else self.cogram()
                self._lipschitz = hutchpp(op, **kwargs)
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

    def gram(self) -> pyct.OpT:
        r"""
        Gram operator :math:`L^\ast L:\mathbb{R}^M\to \mathbb{R}^M`.

        Returns
        -------
        op: pyct.OpT
            (M, M) Gram operator.

        Notes
        -----
        By default the Gram is computed by the composition ``self.T * self``.
        This may not be the fastest way to compute the Gram operator.
        If the Gram can be computed more efficiently (e.g. with a convolution), the user should
        re-define this method.
        """
        return (self.T * self).asop(SelfAdjointOp)

    def cogram(self) -> pyct.OpT:
        r"""
        Co-Gram operator :math:`LL^\ast:\mathbb{R}^N\to \mathbb{R}^N`.

        Returns
        -------
        op: pyct.OpT
            (N, N) Co-Gram operator.

        Notes
        -----
        By default the co-Gram is computed by the composition ``self * self.T``.
        This may not be the fastest way to compute the co-Gram operator.
        If the co-Gram can be computed more efficiently (e.g. with a convolution), the user should
        re-define this method.
        """
        return (self * self.T).asop(SelfAdjointOp)

    @pycrt.enforce_precision(i=("arr", "damp"), allow_None=True)
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
        from pycsou.operator.linop import IdentityOp
        from pycsou.opt.solver import CG
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
            # .pinv() may not have sufficiently converged given the default CG stopping criteria.
            # To avoid infinite loops, CG iterations are thresholded.
            sentinel = MaxIter(n=20 * A.dim)
            kwargs_fit["stop_crit"] = cg.default_stop_crit() | sentinel
        cg.fit(b=b, **kwargs_fit)
        return cg.solution()

    def dagger(
        self,
        damp: pyct.Real = None,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pyct.OpT:
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
        op: pyct.OpT
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
    def from_sciop(cls, sp_op: spsl.LinearOperator) -> pyct.OpT:
        r"""
        Cast a :py:class:`scipy.sparse.linalg.LinearOperator` to a
        :py:class:`~pycsou.abc.operator.LinOp`.

        Parameters
        ----------
        sp_op: [scipy|cupyx].sparse.linalg.LinearOperator
            (N, M) Linear operator compliant with SciPy's interface.

        Returns
        -------
        op: pyct.OpT

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`,
        :py:meth:`~pycsou.abc.operator.LinOp.to_sciop`.
        """
        if sp_op.dtype not in [_.value for _ in pycrt.Width]:
            warnings.warn("Computation may not be performed at the requested precision.", pycuw.PrecisionWarning)

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
    ) -> pyct.OpT:
        r"""
        Instantiate a :py:class:`~pycsou.abc.operator.LinOp` from its array representation.

        Parameters
        ----------
        A: pyct.NDArray
            (N, M) array

        Returns
        -------
        op: pyct.OpT
            (N, M) linear operator

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_sciop`,
        """
        from pycsou.operator.linop import ExplicitLinOp

        return ExplicitLinOp(cls, A, enable_warnings)


class SquareOp(LinOp):
    r"""
    Base class for *square* linear operators, i.e. :math:`L:\mathbb{R}^M\to \mathbb{R}^M`
    (endomorphsisms).
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SQUARE)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        assert self.dim == self.codim, f"shape: expected (M, M), got {self.shape}."

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
        from pycsou.math.linalg import hutchpp

        tr = hutchpp(self, **kwargs)
        return tr


class NormalOp(SquareOp):
    r"""
    Base class for *normal* operators.

    Notes
    -----
    Normal operators commute with their adjoint, i.e. :math:`LL^\ast=L^\ast L`.
    It is `possible to show <https://www.wikiwand.com/en/Spectral_theorem#/Normal_matrices>`_ that
    an operator is normal iff it is *unitarily diagonalizable*, i.e. :math:`L=UDU^\ast`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_NORMAL)
        return frozenset(p)

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
            warnings.warn(msg, pycuw.DenseWarning)
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

    def cogram(self) -> pyct.OpT:
        return self.gram()


class SelfAdjointOp(NormalOp):
    r"""
    Base class for *self-adjoint* operators, i.e. :math:`L^\ast=L`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SELF_ADJOINT)
        return frozenset(p)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.apply(arr)

    @property
    def T(self) -> pyct.OpT:
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

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_UNITARY)
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self._lipschitz

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = self.adjoint(arr)
        if (damp := kwargs.pop("damp", None)) is not None:
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        op = self.T
        if (damp := kwargs.pop("damp", None)) is not None:
            op = op / (1 + damp)
        return op

    def gram(self) -> pyct.OpT:
        from pycsou.operator.linop import IdentityOp

        return IdentityOp(dim=self.dim)

    def cogram(self) -> pyct.OpT:
        return self.gram()


class ProjOp(SquareOp):
    r"""
    Base class for *projection* operators.

    Projection operators are *idempotent*, i.e. :math:`L^2=L`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_IDEMPOTENT)
        return frozenset(p)


class OrthProjOp(ProjOp, SelfAdjointOp):
    r"""
    Base class for *orthogonal projection* operators.

    Orthogonal projection operators are *idempotent* and *self-adjoint*, i.e.
    :math:`L^2=L` and :math:`L^\ast=L`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self._lipschitz

    def gram(self) -> pyct.OpT:
        return self

    def cogram(self) -> pyct.OpT:
        return self

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = arr.copy()
        if (damp := kwargs.pop("damp", None)) is not None:
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        op = self
        if (damp := kwargs.pop("damp", None)) is not None:
            op = op / (1 + damp)
        return op


class PosDefOp(SelfAdjointOp):
    r"""
    Base class for *positive-definite* operators.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_POSITIVE_DEFINITE)
        return frozenset(p)


class LinFunc(ProxDiffFunc, LinOp):
    r"""
    Base class for real-valued linear functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pycsou.abc.operator.Apply.apply`, and
    :py:meth:`~pycsou.abc.operator.Adjoint.adjoint`.

    If known, the Lipschitz constant of the linear functional should be stored in the attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).

    The Lipschitz constant of the gradient is 0 since the latter is constant-valued.
    """

    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        ProxDiffFunc.__init__(self, shape)
        LinOp.__init__(self, shape)
        assert self.dim != None, "shape: domain-agnostic LinFuncs are not supported."
        # Reason: `op.adjoint(arr).shape` cannot be inferred based on `arr.shape` and `op.dim`.

    def jacobian(self, arr: pyct.NDArray) -> pyct.OpT:
        return LinOp.jacobian(self, arr)

    def lipschitz(self, **kwargs) -> pyct.Real:
        # 'fro' / 'svds' mode are identical for linfuncs.
        g = self.grad(np.ones(self.dim))
        self._lipschitz = float(np.linalg.norm(g))
        return self._lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        x = xp.ones((*arr.shape[:-1], 1), dtype=arr.dtype)
        g = self.adjoint(x)
        return g

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return arr - tau * self.grad(arr)

    def cogram(self) -> pyct.OpT:
        from pycsou.operator.linop import HomothetyOp

        return HomothetyOp(cst=self.lipschitz() ** 2, dim=1)

    def svdvals(self, **kwargs) -> pyct.NDArray:
        if kwargs.pop("gpu", False):
            import cupy as xp
        else:
            xp = np
        D = xp.array([self.lipschitz()], dtype=pycrt.getPrecision().value)
        return D

    def asarray(
        self,
        xp: pyct.ArrayModule = np,
        dtype: pyct.DType = None,
    ) -> pyct.NDArray:
        if dtype is None:
            dtype = pycrt.getPrecision().value
        with pycrt.EnforcePrecision(False):
            x = xp.ones((1, 1), dtype=dtype)
            A = self.adjoint(x)
        return A

    @classmethod
    def from_array(
        cls,
        A: pyct.NDArray,
        enable_warnings: bool = True,
    ) -> pyct.OpT:
        A = A.reshape((1, -1))
        op = super().from_array(A, enable_warnings)
        return op


def _core_operators() -> cabc.Set[pyct.OpC]:
    # Operators which can be sub-classed by end-users and participate in arithmetic rules.
    ops = set()
    for _ in globals().values():
        if inspect.isclass(_) and issubclass(_, Operator):
            ops.add(_)
    ops.remove(Operator)
    return ops


__all__ = [
    "Operator",
    "Property",
    *map(lambda _: _.__name__, _core_operators()),
]
