import collections
import collections.abc as cabc
import copy
import enum
import inspect
import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse.linalg as spsl

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
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
        data[self.CAN_EVAL].extend(
            [
                "apply",
                "__call__",
                "lipschitz",
                "_expr",
            ]
        )
        data[self.PROXIMABLE].append("prox")
        data[self.DIFFERENTIABLE].extend(
            [
                "jacobian",
                "diff_lipschitz",
            ]
        )
        data[self.DIFFERENTIABLE_FUNCTION].append("grad")
        data[self.LINEAR].extend(
            [
                "adjoint",
                "asarray",
                "svdvals",
                "pinv",
                "gram",
                "cogram",
            ]
        )
        data[self.LINEAR_SQUARE].append("trace")
        data[self.LINEAR_NORMAL].append("eigvals")
        data[self.QUADRATIC].append("_hessian")

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
        return frozenset(prop) <= cls.properties()

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

        Notes
        -----
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

        See Also
        --------
        :py:class:`~pycsou.abc.arithmetic.AddRule`
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.AddRule(lhs=self, rhs=other).op()
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
            return arithmetic.AddRule(lhs=self, rhs=-other).op()
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

        return arithmetic.ScaleRule(op=self, cst=-1).op()

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

        See Also
        --------
        :py:class:`~pycsou.abc.arithmetic.ScaleRule`,
        :py:class:`~pycsou.abc.arithmetic.ChainRule`
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.ChainRule(lhs=self, rhs=other).op()
        elif isinstance(other, pyct.Real):
            return arithmetic.ScaleRule(op=self, cst=other).op()
        else:
            return NotImplemented

    def __rmul__(self, other: pyct.Real) -> pyct.OpT:
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, pyct.Real):
            return arithmetic.ScaleRule(op=self, cst=other).op()
        else:
            return NotImplemented

    def __truediv__(self, other: pyct.Real) -> pyct.OpT:
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(other, pyct.Real):
            return arithmetic.ScaleRule(op=self, cst=1 / other).op()

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

        See Also
        --------
        :py:class:`~pycsou.abc.arithmetic.PowerRule`
        """
        import pycsou.abc.arithmetic as arithmetic

        if isinstance(k, pyct.Integer) and (k >= 0):
            return arithmetic.PowerRule(op=self, k=k).op()
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

        See Also
        --------
        :py:class:`~pycsou.abc.arithmetic.ArgScaleRule`
        """
        import pycsou.abc.arithmetic as arithmetic

        assert isinstance(scalar, pyct.Real)
        return arithmetic.ArgScaleRule(op=self, cst=scalar).op()

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

        See Also
        --------
        :py:class:`~pycsou.abc.arithmetic.ArgShiftRule`
        """
        import pycsou.abc.arithmetic as arithmetic

        return arithmetic.ArgShiftRule(op=self, cst=shift).op()

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
            if Property.DIFFERENTIABLE in self.properties():
                p.add(Property.DIFFERENTIABLE_FUNCTION)
            if Property.LINEAR in self.properties():
                for p_ in Property:
                    if p_.name.startswith("LINEAR_"):
                        p.discard(p_)
                p.add(Property.PROXIMABLE)
        elif self.codim == self.dim:
            if Property.LINEAR in self.properties():
                p.add(Property.LINEAR_SQUARE)
        klass = self._infer_operator_type(p)
        return self.asop(klass)

    def __repr__(self) -> str:
        klass = self.__class__.__name__
        return f"{klass}{self.shape}"

    def _expr(self) -> tuple:
        """
        Show the expression-representation of the operator.

        If overridden, must return a tuple of the form

            (head, *tail),

        where `head` is the operator (ex: +/*), and `tail` denotes all the expression's terms.
        If an operator cannot be expanded further, then this method should return (self,).
        """
        return (self,)

    def expr(self, level: int = 0, strip: bool = True) -> str:
        """
        Pretty-Print the expression representation of the operator.

        Useful for debugging arithmetic-induced expressions.

        Example
        -------

        .. code-block:: python3

           >>> import numpy as np
           >>> import pycsou.abc as pyca

           >>> N = 5
           >>> op1 = pyca.LinFunc.from_array(np.arange(N))
           >>> op2 = pyca.LinOp.from_array(np.ones((N, N)))
           >>> op = ((2 * op1) + (op2 ** 3)).argshift(np.full(N, 4))

           >>> print(op.expr())
           [argshift, ==> DiffMap(5, 5)
           .[add, ==> SquareOp(5, 5)
           ..[scale, ==> LinFunc(1, 5)
           ...LinFunc(1, 5),
           ...2.0],
           ..[exp, ==> SquareOp(5, 5)
           ...LinOp(5, 5),
           ...3]],
           .(5,)]
        """
        fmt = lambda obj, lvl: ("." * lvl) + str(obj)
        lines = []

        head, *tail = self._expr()
        if len(tail) == 0:
            head = f"{repr(head)},"
        else:
            head = f"[{head}, ==> {repr(self)}"
        lines.append(fmt(head, level))

        for t in tail:
            if isinstance(t, Operator):
                lines += t.expr(level=level + 1, strip=False).split("\n")
            else:
                t = f"{t},"
                lines.append(fmt(t, level + 1))
        if len(tail) > 0:
            # Drop comma for last tail item, then close the sub-expression.
            lines[-1] = lines[-1][:-1]
            lines[-1] += "],"

        out = "\n".join(lines)
        if strip:
            out = out.strip(",")  # drop comma at top-level tail.
        return out


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
        out = self.prox(arr=arr / sigma, tau=1 / sigma)
        out = pycu.copy_if_unsafe(out)
        out *= -sigma
        out += arr
        return out

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

        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr):
            from pycsou.math.linalg import norm

            x = self.prox(arr, tau=_._mu)
            out = pycu.copy_if_unsafe(self.apply(x))
            out += (0.5 / _._mu) * norm(arr - x, axis=-1, keepdims=True) ** 2
            return out

        @pycrt.enforce_precision(i="arr")
        def op_grad(_, arr):
            x = arr.copy()
            x -= self.prox(arr, tau=_._mu)
            x /= _._mu
            return x

        def op_expr(_) -> tuple:
            return ("moreau_envelope", self, _._mu)

        assert mu > 0, f"mu: expected positive, got {mu}"
        op = DiffFunc(self.shape)
        op._mu = mu
        op._diff_lipschitz = 1 / mu
        op.apply = types.MethodType(op_apply, op)
        op.grad = types.MethodType(op_grad, op)
        op._expr = types.MethodType(op_expr, op)
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


class _QuadraticFunc(ProxDiffFunc):
    # Hidden alias of pycsou.operator.func.quadratic.QuadraticFunc to enable operator arithmetic.
    @classmethod
    def properties(cls) -> cabc.Set[pyct.Property]:
        p = set(super().properties())
        p.add(Property.QUADRATIC)
        return frozenset(p)

    def _hessian(self) -> pyct.OpT:
        # (M, M) Hessian matrix which may be useful for some arithmetic methods.
        # This function is NOT EXPOSED to the user on purpose: it is bad practice to try to compute
        # the Hessian in large-scale inverse problems due to its size.
        raise NotImplementedError


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

    # Internal Helpers ------------------------------------
    @staticmethod
    def _warn_vals_sparse_gpu():
        msg = "\n".join(
            [
                "Potential Error:",
                "Sparse GPU-evaluation of svdvals/eigvals() is known to produce incorrect results. (CuPy-specific + Matrix-Dependant.)",
                "It is advised to cross-check results with CPU-computed results.",
            ]
        )
        warnings.warn(msg, pycuw.BackendWarning)

    # -----------------------------------------------------

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

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.transpose`
        """
        return self.transpose()

    def transpose(self, klass: pyct.OpC = None) -> pyct.OpT:
        r"""
        Return the (M, N) adjoint of the linear operator.

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.T`
        """
        if klass is None:
            klass = self._infer_operator_type(self.properties())

        opT = klass(shape=(self.dim, self.codim))
        opT._op = self  # embed for introspection
        for p in opT.properties():
            for name in p.arithmetic_attributes():
                attr = getattr(self, name)
                setattr(opT, name, attr)
            for name in p.arithmetic_methods():
                func = getattr(self.__class__, name)
                setattr(opT, name, types.MethodType(func, opT))

        def opT_asarray(_, **kwargs) -> pyct.NDArray:
            A = self.asarray(**kwargs)
            return A.T

        def opT_eigvals(_, **kwargs) -> pyct.NDArray:
            D = self.eigvals(**kwargs)
            return D.conj()

        # Overwrite arithmetic methods with different implementations vs. encapsulated op.
        opT.apply = self.adjoint
        opT.__call__ = opT.apply
        opT.adjoint = self.apply
        opT.asarray = types.MethodType(opT_asarray, opT)
        opT.eigvals = types.MethodType(opT_eigvals, opT)
        opT.gram = self.cogram
        opT.cogram = self.gram
        return opT._squeeze()

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

        def matmat(arr):
            with pycrt.EnforcePrecision(False):
                return self.apply(arr.T).T

        def rmatmat(arr):
            with pycrt.EnforcePrecision(False):
                return self.adjoint(arr.T).T

        if dtype is None:
            dtype = pycrt.getPrecision().value

        if gpu:
            assert pycd.CUPY_ENABLED
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = spsl
        return spx.LinearOperator(
            shape=self.shape,
            matvec=matmat,
            rmatvec=rmatmat,
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
                kwargs.pop("gpu", None)
                op = self.gram() if (self.codim >= self.dim) else self.cogram()
                self._lipschitz = np.sqrt(hutchpp(op, **kwargs)).item()
            elif algo == "svds":
                kwargs.update(k=1, which="LM")
                kwargs.pop("xp", None)  # unsupported (if present) in svdvals()
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
                import cupy.linalg as spx
            else:
                import numpy as xp
                import scipy.linalg as spx
            op = self.asarray(xp=xp, dtype=pycrt.getPrecision().value)
            return spx.svd(op, compute_uv=False)

        def _sparse_eval():
            if gpu:
                assert pycd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx

                self._warn_vals_sparse_gpu()
            else:
                spx = spsl
            op = self.to_sciop(gpu=gpu, dtype=pycrt.getPrecision().value)
            kwargs.update(
                k=k,
                which=which,
                return_singular_vectors=False,
                # random_state=0,  # unsupported by CuPy
            )
            return spx.svds(op, **kwargs)

        if k >= min(self.shape) // 2:
            msg = "Too many svdvals wanted: using matrix-based ops."
            warnings.warn(msg, pycuw.DenseWarning)
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

        Note
        ----
        This generic implementation assumes the operator is backend-agnostic.
        Thus, when defining a new backend-specific operator, ``.asarray()`` may need to be
        overriden.
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
        op = self.T * self
        return op.asop(SelfAdjointOp)._squeeze()

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
        op = self * self.T
        return op.asop(SelfAdjointOp)._squeeze()

    @pycrt.enforce_precision(i=("arr", "damp"), allow_None=True)
    def pinv(
        self,
        arr: pyct.NDArray,
        damp: pyct.Real = 0,
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
        from pycsou.operator.linop import HomothetyOp
        from pycsou.opt.solver import CG
        from pycsou.opt.stop import MaxIter

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init
        b = self.adjoint(arr)
        if np.isclose(damp, 0):
            A = self.gram()
        else:
            A = self.gram() + HomothetyOp(cst=damp, dim=self.dim)
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
        damp: pyct.Real = 0,
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
        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init

        def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            return self.pinv(
                arr,
                damp=damp,
                kwargs_init=copy.copy(kwargs_init),
                kwargs_fit=copy.copy(kwargs_fit),
            )

        def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
            return self.T.pinv(
                arr,
                damp=damp,
                kwargs_init=copy.copy(kwargs_init),
                kwargs_fit=copy.copy(kwargs_fit),
            )

        def op_expr(_) -> tuple:
            return ("dagger", self, damp)

        klass = SquareOp if (self.dim == self.codim) else LinOp
        dagger = klass(shape=(self.dim, self.codim))
        dagger.apply = types.MethodType(op_apply, dagger)
        dagger.__call__ = dagger.apply
        dagger.adjoint = types.MethodType(op_adjoint, dagger)
        dagger._expr = types.MethodType(op_expr, dagger)
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

        Notes
        -----
        A :py:class:`~pycsou.abc.operator.LinOp` constructed via
        :py:meth:`~pycsou.abc.operator.LinOp.from_sciop` does not respect precision hints from
        pycsou's runtime environment. (Reason: this is just a thin layer around a SciOp to make it
        interoperable with Pycsou operators.)

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`,
        :py:meth:`~pycsou.abc.operator.LinOp.to_sciop`.
        """
        if sp_op.dtype not in [_.value for _ in pycrt.Width]:
            warnings.warn("Computation may not be performed at the requested precision.", pycuw.PrecisionWarning)

        # [r]matmat only accepts 2D inputs -> reshape apply|adjoint inputs as needed.

        def apply(_, arr):
            if _1d := arr.ndim == 1:
                arr = arr.reshape((1, arr.size))
            out = sp_op.matmat(arr.T).T
            if _1d:
                out = out.squeeze(axis=0)
            return out

        def adjoint(_, arr):
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
        A: typ.Union[pyct.NDArray, pyct.SparseArray],
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
        from pycsou.operator.linop.base import _ExplicitLinOp

        return _ExplicitLinOp(cls, A, enable_warnings)


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
                import cupy.linalg as spx
            else:
                import numpy as xp
                import scipy.linalg as spx
            op = self.asarray(xp=xp, dtype=pycrt.getPrecision().value)
            f = getattr(spx, "eigvalsh" if symmetric else "eigvals")
            return f(op)

        def _sparse_eval():
            if gpu:
                assert pycd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx

                self._warn_vals_sparse_gpu()
            else:
                spx = spsl
            op = self.to_sciop(pycrt.getPrecision().value, gpu)
            kwargs.update(k=k, which=which, return_eigenvectors=False)
            f = getattr(spx, "eigsh" if symmetric else "eigs")
            return f(op, **kwargs)

        if which not in ("LM", "SM"):
            raise NotImplementedError
        if k >= self.dim // 2:
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

    def transpose(self, **kwargs) -> pyct.OpT:
        return self._squeeze()

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
        if not np.isclose(damp := kwargs.get("damp", 0), 0):
            out = pycu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        op = self.T / (1 + kwargs.get("damp", 0))
        return op

    def gram(self) -> pyct.OpT:
        from pycsou.operator.linop import IdentityOp

        return IdentityOp(dim=self.dim)._squeeze()

    def svdvals(self, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
        D = xp.ones(kwargs.pop("k"), dtype=pycrt.getPrecision().value)
        return D


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
        return self._squeeze()

    def cogram(self) -> pyct.OpT:
        return self._squeeze()

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = self.apply(arr)
        if not np.isclose(damp := kwargs.get("damp", 0), 0):
            out = pycu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        op = self / (1 + kwargs.get("damp", 0))
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
        xp = kwargs.get("xp", np)
        g = self.grad(xp.ones(self.dim))
        self._lipschitz = float(xp.linalg.norm(g))
        return self._lipschitz

    @pycrt.enforce_precision(i="arr")
    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        x = xp.ones((*arr.shape[:-1], 1), dtype=arr.dtype)
        g = self.adjoint(x)
        return g

    @pycrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        # out = arr - tau * self.grad(arr)
        out = pycu.copy_if_unsafe(self.grad(arr))
        out *= -tau
        out += arr
        return out

    @pycrt.enforce_precision(i=("arr", "sigma"))
    def fenchel_prox(self, arr: pyct.NDArray, sigma: pyct.Real) -> pyct.NDArray:
        return self.grad(arr)

    def transpose(self, **kwargs) -> pyct.OpT:
        if self.dim == self.codim:
            opT = self
        else:
            opT = super().transpose(klass=LinOp)
        return opT

    def cogram(self) -> pyct.OpT:
        from pycsou.operator.linop import HomothetyOp

        return HomothetyOp(cst=self.lipschitz() ** 2, dim=1)

    def svdvals(self, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
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
        A: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True,
    ) -> pyct.OpT:
        if A.ndim == 1:
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
