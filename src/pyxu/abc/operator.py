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

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu


class Property(enum.Enum):
    """
    Mathematical property.

    See Also
    --------
    :py:class:`~pyxu.abc.Operator`
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

    def arithmetic_methods(self) -> cabc.Set[str]:
        "Instance methods affected by arithmetic operations."
        data = collections.defaultdict(list)
        data[self.CAN_EVAL].extend(
            [
                "apply",
                "__call__",
                "estimate_lipschitz",
                "_expr",
            ]
        )
        data[self.PROXIMABLE].append("prox")
        data[self.DIFFERENTIABLE].extend(
            [
                "jacobian",
                "estimate_diff_lipschitz",
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
        data[self.QUADRATIC].append("_quad_spec")

        meth = frozenset(data[self])
        return meth


class Operator:
    """
    Abstract Base Class for Pyxu operators.

    Goals:

    * enable operator arithmetic.
    * cast operators to specialized forms.
    * attach :py:class:`~pyxu.abc.Property` tags encoding certain mathematical properties.  Each core sub-class **must**
      have a unique set of properties to be distinguishable from its peers.
    """

    # For `(size-1 ndarray) * OpT` to work, we need to force NumPy's hand and call OpT.__rmul__() in
    # place of ndarray.__mul__() to determine how scaling should be performed.
    # This is achieved by increasing __array_priority__ for all operators.
    __array_priority__ = np.inf

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        r"""
        Parameters
        ----------
        dim_shape: NDArrayShape
            (M1,...,MD) operator input shape.
        codim_shape: NDArrayShape
            (N1,...,NK) operator output shape.
        """
        dim_shape = pxu.as_canonical_shape(dim_shape)
        assert all(ax >= 1 for ax in dim_shape)

        codim_shape = pxu.as_canonical_shape(codim_shape)
        assert all(ax >= 1 for ax in codim_shape)

        self._dim_shape = dim_shape
        self._codim_shape = codim_shape
        self._name = self.__class__.__name__

    # Public Interface --------------------------------------------------------
    @property
    def dim_shape(self) -> pxt.NDArrayShape:
        r"""
        Return shape of operator's domain. (M1,...,MD)
        """
        return self._dim_shape

    @property
    def dim_size(self) -> pxt.Integer:
        r"""
        Return size of operator's domain. (M1*...*MD)
        """
        return np.prod(self.dim_shape)

    @property
    def dim_rank(self) -> pxt.Integer:
        r"""
        Return rank of operator's domain. (D)
        """
        return len(self.dim_shape)

    @property
    def codim_shape(self) -> pxt.NDArrayShape:
        r"""
        Return shape of operator's co-domain. (N1,...,NK)
        """
        return self._codim_shape

    @property
    def codim_size(self) -> pxt.Integer:
        r"""
        Return size of operator's co-domain. (N1*...*NK)
        """
        return np.prod(self.codim_shape)

    @property
    def codim_rank(self) -> pxt.Integer:
        r"""
        Return rank of operator's co-domain. (K)
        """
        return len(self.codim_shape)

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        "Mathematical properties of the operator."
        return frozenset()

    @classmethod
    def has(cls, prop: typ.Union[Property, cabc.Collection[Property]]) -> bool:
        """
        Verify if operator possesses supplied properties.
        """
        if isinstance(prop, Property):
            prop = (prop,)
        return frozenset(prop) <= cls.properties()

    def asop(self, cast_to: pxt.OpC) -> pxt.OpT:
        r"""
        Recast an :py:class:`~pyxu.abc.Operator` (or subclass thereof) to another :py:class:`~pyxu.abc.Operator`.

        Users may call this method if the arithmetic API yields sub-optimal return types.

        This method is a no-op if `cast_to` is a parent class of ``self``.

        Parameters
        ----------
        cast_to: OpC
            Target type for the recast.

        Returns
        -------
        op: OpT
            Operator with the new interface.

            Fails when cast is forbidden.
            (Ex: :py:class:`~pyxu.abc.Map` -> :py:class:`~pyxu.abc.Func` if codim.size > 1)

        Notes
        -----
        * The interface of `cast_to` is provided via encapsulation + forwarding.
        * If ``self`` does not implement all methods from `cast_to`, then unimplemented methods will raise
          :py:class:`NotImplementedError` when called.
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
            op = cast_to(
                dim_shape=self.dim_shape,
                codim_shape=self.codim_shape,
            )
            op._core = self  # for debugging

            # Forward shared arithmetic fields from core to shell.
            for p in p_shell & p_core:
                for m in p.arithmetic_methods():
                    m_core = getattr(self, m)
                    setattr(op, m, m_core)

            # [diff_]lipschitz are not arithmetic methods, hence are not propagated.
            # We propagate them manually to avoid un-warranted re-evaluations.
            # Important: we write to _[diff_]lipschitz to not overwrite estimate_[diff_]lipschitz() methods.
            if cast_to.has(Property.CAN_EVAL) and self.has(Property.CAN_EVAL):
                op._lipschitz = self.lipschitz
            if cast_to.has(Property.DIFFERENTIABLE) and self.has(Property.DIFFERENTIABLE):
                op._diff_lipschitz = self.diff_lipschitz

            return op

    # Operator Arithmetic -----------------------------------------------------
    def __add__(self, other: pxt.OpT) -> pxt.OpT:
        """
        Add two operators.

        Parameters
        ----------
        self: OpT
            Left operand.
        other: OpT
            Right operand.

        Returns
        -------
        op: OpT
            Composite operator ``self + other``

        Notes
        -----
        Operand shapes must be consistent, i.e.:

            * have `same dimensions`, and
            * have `compatible co-dimensions` (after broadcasting).

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.AddRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.AddRule(lhs=self, rhs=other).op()
        else:
            return NotImplemented

    def __sub__(self, other: pxt.OpT) -> pxt.OpT:
        """
        Subtract two operators.

        Parameters
        ----------
        self: OpT
            Left operand.
        other: OpT
            Right operand.

        Returns
        -------
        op: OpT
            Composite operator ``self - other``
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.AddRule(lhs=self, rhs=-other).op()
        else:
            return NotImplemented

    def __neg__(self) -> pxt.OpT:
        """
        Negate an operator.

        Returns
        -------
        op: OpT
            Composite operator ``-1 * self``.
        """
        import pyxu.abc.arithmetic as arithmetic

        return arithmetic.ScaleRule(op=self, cst=-1).op()

    def __mul__(self, other: typ.Union[pxt.Real, pxt.OpT]) -> pxt.OpT:
        """
        Compose two operators, or scale an operator by a constant.

        Parameters
        ----------
        self: OpT
            Left operand.
        other: Real, OpT
            Scalar or right operand.

        Returns
        -------
        op: OpT
            Scaled operator or composed operator ``self * other``.

        Notes
        -----
        If called with two operators, their shapes must be `consistent`, i.e. ``self.dim_shape == other.codim_shape``.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ScaleRule`,
        :py:class:`~pyxu.abc.arithmetic.ChainRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.ChainRule(lhs=self, rhs=other).op()
        elif _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(other)).op()
        else:
            return NotImplemented

    def __rmul__(self, other: pxt.Real) -> pxt.OpT:
        import pyxu.abc.arithmetic as arithmetic

        if _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(other)).op()
        else:
            return NotImplemented

    def __truediv__(self, other: pxt.Real) -> pxt.OpT:
        import pyxu.abc.arithmetic as arithmetic

        if _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(1 / other)).op()
        else:
            return NotImplemented

    def __pow__(self, k: pxt.Integer) -> pxt.OpT:
        # (op ** k) unsupported
        return NotImplemented

    def __matmul__(self, other) -> pxt.OpT:
        # (op @ NDArray) unsupported
        return NotImplemented

    def __rmatmul__(self, other) -> pxt.OpT:
        # (NDArray @ op) unsupported
        return NotImplemented

    def argscale(self, scalar: pxt.Real) -> pxt.OpT:
        """
        Scale operator's domain.

        Parameters
        ----------
        scalar: Real

        Returns
        -------
        op: OpT
            Domain-scaled operator.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ArgScaleRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        assert _is_real(scalar)
        return arithmetic.ArgScaleRule(op=self, cst=float(scalar)).op()

    def argshift(self, shift: pxt.NDArray) -> pxt.OpT:
        r"""
        Shift operator's domain.

        Parameters
        ----------
        shift: NDArray
            Shift value :math:`c \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}`.

            `shift` must be broadcastable with operator's dimension.

        Returns
        -------
        op: OpT
            Domain-shifted operator :math:`g(x) = f(x + c)`.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ArgShiftRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        return arithmetic.ArgShiftRule(op=self, cst=shift).op()

    # Internal Helpers --------------------------------------------------------
    @staticmethod
    def _infer_operator_type(prop: cabc.Collection[Property]) -> pxt.OpC:
        prop = frozenset(prop)
        for op in _core_operators():
            if op.properties() == prop:
                return op
        else:
            raise ValueError(f"No operator found with properties {prop}.")

    def __repr__(self) -> str:
        klass = self._name
        return f"{klass}(dim={self.dim_shape}, codim={self.codim_shape})"

    def _expr(self) -> tuple:
        r"""
        Show the expression-representation of the operator.

        If overridden, must return a tuple of the form

            (head, \*tail),

        where `head` is the operator (ex: +/\*), and `tail` denotes all the expression's terms.  If an operator cannot
        be expanded further, then this method should return (self,).
        """
        return (self,)

    def _meta(self):
        # When using DASK inputs, it is sometimes necessary to pass extra information to Dask functions.  This function
        # serves this purpose: it lets class writers encode any such information and re-use it when processing DASK
        # inputs.  The action and return types of _meta() are at the sole discretion of the implementer.
        raise NotImplementedError

    def expr(self, level: int = 0, strip: bool = True) -> str:
        """
        Pretty-Print the expression representation of the operator.

        Useful for debugging arithmetic-induced expressions.

        Example
        -------

        .. code-block:: python3

           import numpy as np
           import pyxu.abc as pxa

           kwargs = dict(dim_shape=5, codim_shape=5)
           op1 = pxa.LinOp(**kwargs)
           op2 = pxa.DiffMap(**kwargs)
           op = ((2 * op1) + (op1 * op2)).argshift(np.r_[1])

           print(op.expr())
           # [argshift, ==> DiffMap(dim=(5,), codim=(5,))
           # .[add, ==> DiffMap(dim=(5,), codim=(5,))
           # ..[scale, ==> LinOp(dim=(5,), codim=(5,))
           # ...LinOp(dim=(5,), codim=(5,)),
           # ...2.0],
           # ..[compose, ==> DiffMap(dim=(5,), codim=(5,))
           # ...LinOp(dim=(5,), codim=(5,)),
           # ...DiffMap(dim=(5,), codim=(5,))]],
           # .(1,)]
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

    # Short-hands for commonly-used operators ---------------------------------
    def squeeze(self, axes: pxt.NDArrayAxis = None) -> pxt.OpT:
        """
        Drop size-1 axes from co-dimension.

        See Also
        --------
        :py:class:`~pyxu.operator.SqueezeAxes`
        """

        from pyxu.operator import SqueezeAxes

        sq = SqueezeAxes(
            dim_shape=self.codim_shape,
            axes=axes,
        )
        op = sq * self
        return op

    def transpose(self, axes: pxt.NDArrayAxis = None) -> pxt.OpT:
        """
        Permute co-dimension axes.

        See Also
        --------
        :py:class:`~pyxu.operator.TransposeAxes`
        """

        from pyxu.operator import TransposeAxes

        tr = TransposeAxes(
            dim_shape=self.codim_shape,
            axes=axes,
        )
        op = tr * self
        return op

    def reshape(self, codim_shape: pxt.NDArrayShape) -> pxt.OpT:
        """
        Reshape co-dimension shape.

        See Also
        --------
        :py:class:`~pyxu.operator.ReshapeAxes`
        """

        from pyxu.operator import ReshapeAxes

        rsh = ReshapeAxes(
            dim_shape=self.codim_shape,
            codim_shape=codim_shape,
        )
        op = rsh * self
        return op

    def broadcast_to(self, codim_shape: pxt.NDArrayShape) -> pxt.OpT:
        """
        Broadcast co-dimension shape.

        See Also
        --------
        :py:class:`~pyxu.operator.BroadcastAxes`
        """

        from pyxu.operator import BroadcastAxes

        bcast = BroadcastAxes(
            dim_shape=self.codim_shape,
            codim_shape=codim_shape,
        )
        op = bcast * self
        return op

    def subsample(self, *indices) -> pxt.OpT:
        """
        Sub-sample co-dimension.

        See Also
        --------
        :py:class:`~pyxu.operator.SubSample`
        """

        from pyxu.operator import SubSample

        sub = SubSample(self.codim_shape, *indices)
        op = sub * self
        return op

    def rechunk(self, chunks: dict) -> pxt.OpT:
        """
        Re-chunk core dimensions to new chunk size.

        See Also
        --------
        :py:func:`~pyxu.operator.RechunkAxes`
        """
        from pyxu.operator import RechunkAxes

        chk = RechunkAxes(
            dim_shape=self.codim_shape,
            chunks=chunks,
        )
        op = chk * self
        return op


class Map(Operator):
    r"""
    Base class for real-valued maps :math:`\mathbf{f}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1}
    \times\cdots\times N_{K}}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply`.

    If :math:`\mathbf{f}` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored
    in the :py:attr:`~pyxu.abc.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.CAN_EVAL)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.lipschitz = np.inf

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Evaluate operator at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) input points.

        Returns
        -------
        out: NDArray
            (..., N1,...,NK) output points.
        """
        raise NotImplementedError

    def __call__(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Alias for :py:meth:`~pyxu.abc.Map.apply`.
        """
        return self.apply(arr)

    @property
    def lipschitz(self) -> pxt.Real:
        r"""
        Return the last computed Lipschitz constant of :math:`\mathbf{f}`.

        Notes
        -----
        * If a Lipschitz constant is known apriori, it can be stored in the instance as follows:

          .. code-block:: python3

             class TestOp(Map):
                 def __init__(self, dim_shape, codim_shape):
                     super().__init__(dim_shape, codim_shape)
                     self.lipschitz = 2

             op = TestOp(2, 3)
             op.lipschitz  # => 2


          Alternatively the Lipschitz constant can be set manually after initialization:

          .. code-block:: python3

             class TestOp(Map):
                 def __init__(self, dim_shape, codim_shape):
                     super().__init__(dim_shape, codim_shape)

             op = TestOp(2, 3)
             op.lipschitz  # => inf, since unknown apriori

             op.lipschitz = 2  # post-init specification
             op.lipschitz  # => 2

        * :py:meth:`~pyxu.abc.Map.lipschitz` **never** computes anything:
          call :py:meth:`~pyxu.abc.Map.estimate_lipschitz` manually to *compute* a new Lipschitz estimate:

          .. code-block:: python3

             op.lipschitz = op.estimate_lipschitz()
        """
        if not hasattr(self, "_lipschitz"):
            self._lipschitz = self.estimate_lipschitz()

        return self._lipschitz

    @lipschitz.setter
    def lipschitz(self, L: pxt.Real):
        assert L >= 0
        self._lipschitz = float(L)

        # If no algorithm available to auto-determine estimate_lipschitz(), then enforce user's choice.
        if not self.has(Property.LINEAR):

            def op_estimate_lipschitz(_, **kwargs) -> pxt.Real:
                return _._lipschitz

            self.estimate_lipschitz = types.MethodType(op_estimate_lipschitz, self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Parameters
        ----------
        kwargs: ~collections.abc.Mapping
            Class-specific kwargs to configure Lipschitz estimation.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A constant :math:`L_{\mathbf{f}} > 0` is said to be a *Lipschitz constant* for a map :math:`\mathbf{f}:
          \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1} \times\cdots\times N_{K}}` if:

          .. math::

             \|\mathbf{f}(\mathbf{x}) - \mathbf{f}(\mathbf{y})\|_{\mathbb{R}^{N_{1} \times\cdots\times N_{K}}}
             \leq
             L_{\mathbf{f}} \|\mathbf{x} - \mathbf{y}\|_{\mathbb{R}^{M_{1} \times\cdots\times M_{D}}},
             \qquad
             \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^{M_{1} \times\cdots\times M_{D}},

          where :math:`\|\cdot\|_{\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}` and :math:`\|\cdot\|_{\mathbb{R}^{N_{1}
          \times\cdots\times N_{K}}}` are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of a map is called the *optimal Lipschitz constant*.
        """
        raise NotImplementedError


class Func(Map):
    r"""
    Base class for real-valued functionals :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}\cup\{+\infty\}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply`.

    If :math:`f` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.FUNCTIONAL)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        assert (self.codim_size == 1) and (self.codim_rank == 1)


class DiffMap(Map):
    r"""
    Base class for real-valued differentiable maps :math:`\mathbf{f}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{N_{1} \times\cdots\times N_{K}}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply` and :py:meth:`~pyxu.abc.DiffMap.jacobian`.

    If :math:`\mathbf{f}` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored
    in the :py:attr:`~pyxu.abc.Map.lipschitz` property.

    If :math:`\mathbf{J}_{\mathbf{f}}` is Lipschitz-continuous with known Lipschitz constant :math:`\partial L`, the
    latter should be stored in the :py:attr:`~pyxu.abc.DiffMap.diff_lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.DIFFERENTIABLE)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.diff_lipschitz = np.inf

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        r"""
        Evaluate the Jacobian of :math:`\mathbf{f}` at the specified point.

        Parameters
        ----------
        arr: NDArray
            (M1,...,MD) evaluation point.

        Returns
        -------
        op: OpT
            Jacobian operator at point `arr`.

        Notes
        -----
        Let :math:`\mathbf{f}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{N_{1} \times\cdots\times
        N_{K}}` be a differentiable multi-dimensional map.  The *Jacobian* (or *differential*) of :math:`\mathbf{f}` at
        :math:`\mathbf{z} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}` is defined as the best linear approximator of
        :math:`\mathbf{f}` near :math:`\mathbf{z}`, in the following sense:

        .. math::

           \mathbf{f}(\mathbf{x}) - \mathbf{f}(\mathbf{z}) = \mathbf{J}_{\mathbf{f}}(\mathbf{z}) (\mathbf{x} -
           \mathbf{z}) + o(\| \mathbf{x} - \mathbf{z} \|) \quad \text{as} \quad \mathbf{x} \to \mathbf{z}.

        The Jacobian admits the following matrix representation:

        .. math::

           [\mathbf{J}_{\mathbf{f}}(\mathbf{x})]_{ij} := \frac{\partial f_{i}}{\partial x_{j}}(\mathbf{x}), \qquad
           \forall (i,j) \in \{1,\ldots,N_{1}\cdots N_{K}\} \times \{1,\ldots,M_{1}\cdots M_{D}\}.
        """
        raise NotImplementedError

    @property
    def diff_lipschitz(self) -> pxt.Real:
        r"""
        Return the last computed Lipschitz constant of :math:`\mathbf{J}_{\mathbf{f}}`.

        Notes
        -----
        * If a diff-Lipschitz constant is known apriori, it can be stored in the instance as follows:

          .. code-block:: python3

             class TestOp(DiffMap):
                 def __init__(self, dim_shape, codim_shape):
                     super().__init__(dim_shape, codim_shape)
                     self.diff_lipschitz = 2

             op = TestOp(2, 3)
             op.diff_lipschitz  # => 2


          Alternatively the diff-Lipschitz constant can be set manually after initialization:

          .. code-block:: python3

             class TestOp(DiffMap):
                 def __init__(self, dim_shape, codim_shape):
                     super().__init__(dim_shape, codim_shape)

             op = TestOp(2, 3)
             op.diff_lipschitz  # => inf, since unknown apriori

             op.diff_lipschitz = 2  # post-init specification
             op.diff_lipschitz  # => 2

        * :py:meth:`~pyxu.abc.DiffMap.diff_lipschitz` **never** computes anything:
          call :py:meth:`~pyxu.abc.DiffMap.estimate_diff_lipschitz` manually to *compute* a new diff-Lipschitz estimate:

          .. code-block:: python3

             op.diff_lipschitz = op.estimate_diff_lipschitz()
        """
        if not hasattr(self, "_diff_lipschitz"):
            self._diff_lipschitz = self.estimate_diff_lipschitz()

        return self._diff_lipschitz

    @diff_lipschitz.setter
    def diff_lipschitz(self, dL: pxt.Real):
        assert dL >= 0
        self._diff_lipschitz = float(dL)

        # If no algorithm available to auto-determine estimate_diff_lipschitz(), then enforce user's choice.
        if not self.has(Property.QUADRATIC):

            def op_estimate_diff_lipschitz(_, **kwargs) -> pxt.Real:
                return _._diff_lipschitz

            self.estimate_diff_lipschitz = types.MethodType(op_estimate_diff_lipschitz, self)

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of :py:meth:`~pyxu.abc.DiffMap.jacobian`.

        Parameters
        ----------
        kwargs: ~collections.abc.Mapping
            Class-specific kwargs to configure diff-Lipschitz estimation.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A Lipschitz constant :math:`L_{\mathbf{J}_{\mathbf{f}}} > 0` of the Jacobian map
          :math:`\mathbf{J}_{\mathbf{f}}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}^{(N_{1}
          \times\cdots\times N_{K}) \times (M_{1} \times\cdots\times M_{D})}` is such that:

          .. math::

             \|\mathbf{J}_{\mathbf{f}}(\mathbf{x}) - \mathbf{J}_{\mathbf{f}}(\mathbf{y})\|_{\mathbb{R}^{(N_{1}
             \times\cdots\times N_{K}) \times (M_{1} \times\cdots\times M_{D})}}
             \leq
             L_{\mathbf{J}_{\mathbf{f}}} \|\mathbf{x} - \mathbf{y}\|_{\mathbb{R}^{M_{1} \times\cdots\times M_{D}}},
             \qquad
             \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}},

          where :math:`\|\cdot\|_{\mathbb{R}^{(N_{1} \times\cdots\times N_{K}) \times (M_{1} \times\cdots\times
          M_{D})}}` and :math:`\|\cdot\|_{\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}` are the canonical norms on their
          respective spaces.

          The smallest Lipschitz constant of the Jacobian is called the *optimal diff-Lipschitz constant*.
        """
        raise NotImplementedError


class ProxFunc(Func):
    r"""
    Base class for real-valued proximable functionals :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R} \cup \{+\infty\}`.

    A functional :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R} \cup \{+\infty\}` is said
    *proximable* if its **proximity operator** (see :py:meth:`~pyxu.abc.ProxFunc.prox` for a definition) admits a
    *simple closed-form expression* **or** can be evaluated *efficiently* and with *high accuracy*.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply` and :py:meth:`~pyxu.abc.ProxFunc.prox`.

    If :math:`f` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.PROXIMABLE)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        r"""
        Evaluate proximity operator of :math:`\tau f` at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) input points.
        tau: Real
            Positive scale factor.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) proximal evaluations.

        Notes
        -----
        For :math:`\tau >0`, the *proximity operator* of a scaled functional :math:`f: \mathbb{R}^{M_{1}
        \times\cdots\times M_{D}} \to \mathbb{R}` is defined as:

        .. math::

           \mathbf{\text{prox}}_{\tau f}(\mathbf{z})
           :=
           \arg\min_{\mathbf{x}\in\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}
           f(x)+\frac{1}{2\tau} \|\mathbf{x}-\mathbf{z}\|_{2}^{2},
           \quad
           \forall \mathbf{z} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}.
        """
        raise NotImplementedError

    def fenchel_prox(self, arr: pxt.NDArray, sigma: pxt.Real) -> pxt.NDArray:
        r"""
        Evaluate proximity operator of :math:`\sigma f^{\ast}`, the scaled Fenchel conjugate of :math:`f`, at specified
        point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) input points.
        sigma: Real
            Positive scale factor.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) proximal evaluations.

        Notes
        -----
        For :math:`\sigma > 0`, the *Fenchel conjugate* is defined as:

        .. math::

           f^{\ast}(\mathbf{z})
           :=
           \max_{\mathbf{x}\in\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}
           \langle \mathbf{x},\mathbf{z} \rangle - f(\mathbf{x}).

        From *Moreau's identity*, its proximal operator is given by:

        .. math::

           \mathbf{\text{prox}}_{\sigma f^{\ast}}(\mathbf{z})
           =
           \mathbf{z} - \sigma \mathbf{\text{prox}}_{f/\sigma}(\mathbf{z}/\sigma).
        """
        out = arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)
        return out

    def moreau_envelope(self, mu: pxt.Real) -> pxt.OpT:
        r"""
        Approximate proximable functional :math:`f` by its *Moreau envelope* :math:`f^{\mu}`.

        Parameters
        ----------
        mu: Real
            Positive regularization parameter.

        Returns
        -------
        op: OpT
            Differential Moreau envelope.

        Notes
        -----
        Consider a convex non-smooth proximable functional :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
        \mathbb{R} \cup \{+\infty\}` and a regularization parameter :math:`\mu > 0`.  The :math:`\mu`-*Moreau envelope*
        (or *Moreau-Yoshida envelope*) of :math:`f` is given by

        .. math::

           f^{\mu}(\mathbf{x})
           =
           \min_{\mathbf{z} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}}
           f(\mathbf{z})
           \quad + \quad
           \frac{1}{2\mu} \|\mathbf{x} - \mathbf{z}\|^{2}.

        The parameter :math:`\mu` controls the trade-off between the regularity properties of :math:`f^{\mu}` and the
        approximation error incurred by the Moreau-Yoshida regularization.

        The Moreau envelope inherits the convexity of :math:`f` and is gradient-Lipschitz (with Lipschitz constant
        :math:`\mu^{-1}`), even if :math:`f` is non-smooth.  Its gradient is moreover given by:

        .. math::

           \nabla f^{\mu}(\mathbf{x})
           =
           \mu^{-1} \left(\mathbf{x} - \text{prox}_{\mu f}(\mathbf{x})\right).

        In addition, :math:`f^{\mu}` envelopes :math:`f` from below: :math:`f^{\mu}(\mathbf{x}) \leq f(\mathbf{x})`.
        This envelope becomes tighter as :math:`\mu \to 0`:

        .. math::

           \lim_{\mu \to 0} f^{\mu}(\mathbf{x}) = f(\mathbf{x}).

        Finally, it can be shown that the minimizers of :math:`f` and :math:`f^{\mu}` coincide, and that the Fenchel
        conjugate of :math:`f^{\mu}` is strongly-convex.

        Example
        -------
        Construct and plot the Moreau envelope of the :math:`\ell_{1}`-norm:

        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from pyxu.abc import ProxFunc

           class L1Norm(ProxFunc):
               def __init__(self, dim: int):
                   super().__init__(dim_shape=dim, codim_shape=1)
               def apply(self, arr):
                   return np.linalg.norm(arr, axis=-1, keepdims=True, ord=1)
               def prox(self, arr, tau):
                   return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)

           mu = [0.1, 0.5, 1]
           f = [L1Norm(dim=1).moreau_envelope(_mu) for _mu in mu]
           x = np.linspace(-1, 1, 512).reshape(-1, 1)  # evaluation points

           fig, ax = plt.subplots(ncols=2)
           for _mu, _f in zip(mu, f):
               ax[0].plot(x, _f(x), label=f"mu={_mu}")
               ax[1].plot(x, _f.grad(x), label=f"mu={_mu}")
           ax[0].set_title('Moreau Envelope')
           ax[1].set_title("Derivative of Moreau Envelope")
           for _ax in ax:
               _ax.legend()
               _ax.set_aspect("equal")
           fig.tight_layout()
        """
        from pyxu.operator.interop import from_source

        assert mu > 0, f"mu: expected positive, got {mu}"

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            xp = pxu.get_array_module(arr)

            x = self.prox(arr, tau=_._mu)
            y = xp.sum(
                (arr - x) ** 2,
                axis=tuple(range(-self.dim_rank, 0)),
            )[..., np.newaxis]
            y *= 0.5 / _._mu

            out = self.apply(x) + y
            return out

        def op_grad(_, arr):
            out = arr - self.prox(arr, tau=_._mu)
            out /= _._mu
            return out

        op = from_source(
            cls=DiffFunc,
            dim_shape=self.dim_shape,
            codim_shape=self.codim_shape,
            embed=dict(
                _name="moreau_envelope",
                _mu=mu,
                _diff_lipschitz=float(1 / mu),
            ),
            apply=op_apply,
            grad=op_grad,
            _expr=lambda _: ("moreau_envelope", _, _._mu),
        )
        return op


class DiffFunc(DiffMap, Func):
    r"""
    Base class for real-valued differentiable functionals :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply` and :py:meth:`~pyxu.abc.DiffFunc.grad`.

    If :math:`f` and/or its derivative :math:`f'` are Lipschitz-continuous with known Lipschitz constants :math:`L` and
    :math:`\partial L`, the latter should be stored in the :py:attr:`~pyxu.abc.Map.lipschitz` and
    :py:attr:`~pyxu.abc.DiffMap.diff_lipschitz` properties.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        p.add(Property.DIFFERENTIABLE_FUNCTION)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        for klass in [DiffMap, Func]:
            klass.__init__(
                self,
                dim_shape=dim_shape,
                codim_shape=codim_shape,
            )

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        op = LinFunc.from_array(
            A=self.grad(arr)[np.newaxis, ...],
            dim_rank=self.dim_rank,
        )
        return op

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate operator gradient at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M1,...,MD) input points.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) gradients.

        Notes
        -----
        The gradient of a functional :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}` is given, for
        every :math:`\mathbf{x} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}`, by

        .. math::

           \nabla f(\mathbf{x})
           :=
           \left[\begin{array}{c}
           \frac{\partial f}{\partial x_{1}}(\mathbf{x}) \\
           \vdots \\
           \frac{\partial f}{\partial x_{M}}(\mathbf{x})
           \end{array}\right].
        """
        raise NotImplementedError


class ProxDiffFunc(ProxFunc, DiffFunc):
    r"""
    Base class for real-valued differentiable *and* proximable functionals :math:`f:\mathbb{R}^{M_{1} \times\cdots\times
    M_{D}} \to \mathbb{R}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply`, :py:meth:`~pyxu.abc.DiffFunc.grad`, and
    :py:meth:`~pyxu.abc.ProxFunc.prox`.

    If :math:`f` and/or its derivative :math:`f'` are Lipschitz-continuous with known Lipschitz constants :math:`L` and
    :math:`\partial L`, the latter should be stored in the :py:attr:`~pyxu.abc.Map.lipschitz` and
    :py:attr:`~pyxu.abc.DiffMap.diff_lipschitz` properties.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        for klass in [ProxFunc, DiffFunc]:
            klass.__init__(
                self,
                dim_shape=dim_shape,
                codim_shape=codim_shape,
            )


class QuadraticFunc(ProxDiffFunc):
    # This is a special abstract base class with more __init__ parameters than `dim/codim_shape`.
    r"""
    Base class for quadratic functionals :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R} \cup
    \{+\infty\}`.

    The quadratic functional is defined as:

    .. math::

       f(\mathbf{x})
       =
       \frac{1}{2} \langle\mathbf{x}, \mathbf{Q}\mathbf{x}\rangle
       +
       \langle\mathbf{c},\mathbf{x}\rangle
       +
       t,
       \qquad \forall \mathbf{x} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}},

    where :math:`Q` is a positive-definite operator :math:`\mathbf{Q}:\mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{M_{1} \times\cdots\times M_{D}}`, :math:`\mathbf{c} \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}}`,
    and :math:`t > 0`.

    Its gradient is given by:

    .. math::

       \nabla f(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}.

    Its proximity operator by:

    .. math::

       \text{prox}_{\tau f}(x)
       =
       \left(
           \mathbf{Q} + \tau^{-1} \mathbf{Id}
       \right)^{-1}
       \left(
           \tau^{-1}\mathbf{x} - \mathbf{c}
       \right).

    In practice the proximity operator is evaluated via :py:class:`~pyxu.opt.solver.CG`.

    The Lipschitz constant :math:`L` of a quadratic on an unbounded domain is unbounded.  The Lipschitz constant
    :math:`\partial L` of :math:`\nabla f` is given by the spectral norm of :math:`\mathbf{Q}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.QUADRATIC)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
        # required in place of `dim` to have uniform interface with Operator hierarchy.
        Q: "PosDefOp" = None,
        c: "LinFunc" = None,
        t: pxt.Real = 0,
    ):
        r"""
        Parameters
        ----------
        Q: ~pyxu.abc.PosDefOp
            Positive-definite operator. (Default: Identity)
        c: ~pyxu.abc.LinFunc
            Linear functional. (Default: NullFunc)
        t: Real
            Offset. (Default: 0)
        """
        from pyxu.operator import IdentityOp, NullFunc

        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )

        # Do NOT access (_Q, _c, _t) directly through `self`:
        # their values may not reflect the true (Q, c, t) parameterization.
        # (Reason: arithmetic propagation.)
        # Always access (Q, c, t) by querying the arithmetic method `_quad_spec()`.
        self._Q = IdentityOp(dim_shape=self.dim_shape) if (Q is None) else Q
        self._c = NullFunc(dim_shape=self.dim_shape) if (c is None) else c
        self._t = t

        # ensure dimensions are consistent if None-initialized
        assert self._Q.dim_shape == self.dim_shape
        assert self._Q.codim_shape == self.dim_shape
        assert self._c.dim_shape == self.dim_shape
        assert self._c.codim_shape == self.codim_shape

        self.diff_lipschitz = self._Q.lipschitz

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        Q, c, t = self._quad_spec()
        out = (arr * Q.apply(arr)).sum(axis=tuple(range(-self.dim_rank, 0)))[..., np.newaxis]
        out /= 2
        out += c.apply(arr)
        out += t
        return out

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        Q, c, _ = self._quad_spec()
        out = Q.apply(arr) + c.grad(arr)
        return out

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        from pyxu.operator import HomothetyOp
        from pyxu.opt.solver import CG
        from pyxu.opt.stop import MaxIter

        Q, c, _ = self._quad_spec()
        A = Q + HomothetyOp(cst=1 / tau, dim_shape=Q.dim_shape)
        b = arr.copy()
        b /= tau
        b -= c.grad(arr)

        slvr = CG(A=A, show_progress=False)

        sentinel = MaxIter(n=2 * A.dim_size)
        stop_crit = slvr.default_stop_crit() | sentinel

        slvr.fit(b=b, stop_crit=stop_crit)
        return slvr.solution()

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        Q, *_ = self._quad_spec()
        dL = Q.estimate_lipschitz(**kwargs)
        return dL

    def _quad_spec(self):
        """
        Canonical quadratic parameterization.

        Useful for some internal methods, and overloaded during operator arithmetic.
        """
        return (self._Q, self._c, self._t)


class LinOp(DiffMap):
    r"""
    Base class for real-valued linear operators :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{N_{1} \times\cdots\times N_{K}}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply` and :py:meth:`~pyxu.abc.LinOp.adjoint`.

    If known, the Lipschitz constant :math:`L` should be stored in the :py:attr:`~pyxu.abc.Map.lipschitz` property.

    The Jacobian of a linear map :math:`\mathbf{A}` is constant.
    """

    # Internal Helpers ------------------------------------
    @staticmethod
    def _warn_vals_sparse_gpu():
        msg = "\n".join(
            [
                "Potential Error:",
                "Sparse GPU-evaluation of svdvals() is known to produce incorrect results. (CuPy-specific + Matrix-Dependant.)",
                "It is advised to cross-check results with CPU-computed results.",
            ]
        )
        warnings.warn(msg, pxw.BackendWarning)

    # -----------------------------------------------------

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.diff_lipschitz = 0

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate operator adjoint at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., N1,...,NK) input points.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) adjoint evaluations.

        Notes
        -----
        The *adjoint* :math:`\mathbf{A}^{\ast}: \mathbb{R}^{N_{1} \times\cdots\times N_{K}} \to \mathbb{R}^{M_{1}
        \times\cdots\times M_{D}}` of :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
        \mathbb{R}^{N_{1} \times\cdots\times N_{K}}` is defined as:

        .. math::

           \langle \mathbf{x}, \mathbf{A}^{\ast}\mathbf{y}\rangle_{\mathbb{R}^{M_{1} \times\cdots\times M_{D}}}
           :=
           \langle \mathbf{A}\mathbf{x}, \mathbf{y}\rangle_{\mathbb{R}^{N_{1} \times\cdots\times N_{K}}},
           \qquad
           \forall (\mathbf{x},\mathbf{y})\in \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \times \mathbb{R}^{N_{1}
           \times\cdots\times N_{K}}.
        """
        raise NotImplementedError

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return self

    @property
    def T(self) -> pxt.OpT:
        r"""
        Return the adjoint of :math:`\mathbf{A}`.
        """
        import pyxu.abc.arithmetic as arithmetic

        return arithmetic.TransposeRule(op=self).op()

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Parameters
        ----------
        method: "svd" | "trace"

            * If `svd`, compute the optimal Lipschitz constant.
            * If `trace`, compute an upper bound. (Default)

        kwargs:
            Optional kwargs passed on to:

            * `svd`: :py:func:`~pyxu.abc.LinOp.svdvals`
            * `trace`: :py:func:`~pyxu.math.hutchpp`

        Notes
        -----
        * The tightest Lipschitz constant is given by the spectral norm of the operator :math:`\mathbf{A}`:
          :math:`\|\mathbf{A}\|_{2}`.  It can be computed via the SVD, which is compute-intensive task for large
          operators.  In this setting, it may be advantageous to overestimate the Lipschitz constant with the Frobenius
          norm of :math:`\mathbf{A}` since :math:`\|\mathbf{A}\|_{F} \geq \|\mathbf{A}\|_{2}`.

          :math:`\|\mathbf{A}\|_{F}` can be efficiently approximated by computing the trace of :math:`\mathbf{A}^{\ast}
          \mathbf{A}` (or :math:`\mathbf{A}\mathbf{A}^{\ast}`) via the `Hutch++ stochastic algorithm
          <https://arxiv.org/abs/2010.09649>`_.

        * :math:`\|\mathbf{A}\|_{F}` is upper-bounded by :math:`\|\mathbf{A}\|_{F} \leq \sqrt{n} \|\mathbf{A}\|_{2}`,
          where the equality is reached (worst-case scenario) when the eigenspectrum of the linear operator is flat.
        """
        method = kwargs.get("method", "trace").lower().strip()

        if method == "svd":
            # svdvals() may have alternative signature in specialized classes, but we must always use
            # the LinOp.svdvals() interface below for kwargs-filtering.
            func, sig_func = self.__class__.svdvals, LinOp.svdvals
            kwargs.update(k=1)
            estimate = lambda: func(self, **kwargs).item()
        elif method == "trace":
            from pyxu.math import hutchpp as func

            sig_func = func
            kwargs.update(
                op=self.gram() if (self.codim_size >= self.dim_size) else self.cogram(),
                m=kwargs.get("m", 126),
            )
            estimate = lambda: np.sqrt(func(**kwargs)).item()
        else:
            raise NotImplementedError

        # Filter unsupported kwargs
        sig = inspect.Signature.from_callable(sig_func)
        kwargs = {k: v for (k, v) in kwargs.items() if (k in sig.parameters)}

        L = estimate()
        return L

    def svdvals(
        self,
        k: pxt.Integer,
        gpu: bool = False,
        dtype: pxt.DType = None,
        **kwargs,
    ) -> pxt.NDArray:
        r"""
        Compute leading singular values of the linear operator.

        Parameters
        ----------
        k: Integer
            Number of singular values to compute.
        gpu: bool
            If ``True`` the singular value decomposition is performed on the GPU.
        dtype: DType
            Working precision of the linear operator.
        kwargs: ~collections.abc.Mapping
            Additional kwargs accepted by :py:func:`~scipy.sparse.linalg.svds`.

        Returns
        -------
        D: NDArray
            (k,) singular values in ascending order.
        """
        if dtype is None:
            dtype = pxrt.Width.DOUBLE.value

        def _dense_eval():
            if gpu:
                assert pxd.CUPY_ENABLED
                import cupy as xp
                import cupy.linalg as spx
            else:
                import numpy as xp
                import scipy.linalg as spx
            A = self.asarray(xp=xp, dtype=dtype)
            B = A.reshape(self.codim_size, self.dim_size)
            return spx.svd(B, compute_uv=False)

        def _sparse_eval():
            if gpu:
                assert pxd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx

                self._warn_vals_sparse_gpu()
            else:
                spx = spsl
            from pyxu.operator import ReshapeAxes
            from pyxu.operator.interop import to_sciop

            # SciPy's LinearOperator only understands 2D linear operators.
            # -> wrap `self` into 2D form for SVD computations.
            lhs = ReshapeAxes(dim_shape=self.codim_shape, codim_shape=self.codim_size)
            rhs = ReshapeAxes(dim_shape=self.dim_size, codim_shape=self.dim_shape)
            op = to_sciop(
                op=lhs * self * rhs,
                gpu=gpu,
                dtype=dtype,
            )

            which = kwargs.get("which", "LM")
            assert which.upper() == "LM", "Only computing leading singular values is supported."
            kwargs.update(
                k=k,
                which=which,
                return_singular_vectors=False,
                # random_state=0,  # unsupported by CuPy
            )
            return spx.svds(op, **kwargs)

        if k >= min(self.dim_size, self.codim_size) // 2:
            msg = "Too many svdvals wanted: using matrix-based ops."
            warnings.warn(msg, pxw.DenseWarning)
            D = _dense_eval()
        else:
            D = _sparse_eval()

        # Filter to k largest magnitude + sorted
        xp = pxu.get_array_module(D)
        return D[xp.argsort(D)][-k:]

    def asarray(
        self,
        xp: pxt.ArrayModule = None,
        dtype: pxt.DType = None,
    ) -> pxt.NDArray:
        r"""
        Matrix representation of the linear operator.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output. (Default: NumPy.)
        dtype: DType
            Precision of the array. (Default: current runtime precision.)

        Returns
        -------
        A: NDArray
            (*codim_shape, *dim_shape) array-representation of the operator.

        Note
        ----
        This generic implementation assumes the operator is backend-agnostic.  Thus, when defining a new
        backend-specific operator, :py:meth:`~pyxu.abc.LinOp.asarray` may need to be overriden.
        """
        if xp is None:
            xp = pxd.NDArrayInfo.default().module()
        if dtype is None:
            dtype = pxrt.Width.DOUBLE.value

        E = xp.eye(self.dim_size, dtype=dtype).reshape(*self.dim_shape, *self.dim_shape)
        A = self.apply(E)  # (*dim_shape, *codim_shape)

        axes = tuple(range(-self.codim_rank, 0)) + tuple(range(self.dim_rank))
        B = A.transpose(axes)  # (*codim_shape, *dim_shape)
        return B

    def gram(self) -> pxt.OpT:
        r"""
        Gram operator :math:`\mathbf{A}^{\ast} \mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
        \mathbb{R}^{M_{1} \times\cdots\times M_{D}}`.

        Returns
        -------
        op: OpT
            Gram operator.

        Note
        ----
        By default the Gram is computed by the composition ``self.T * self``.  This may not be the fastest way to
        compute the Gram operator.  If the Gram can be computed more efficiently (e.g. with a convolution), the user
        should re-define this method.
        """

        def op_expr(_) -> tuple:
            return ("gram", self)

        op = self.T * self
        op._expr = types.MethodType(op_expr, op)
        return op.asop(SelfAdjointOp)

    def cogram(self) -> pxt.OpT:
        r"""
        Co-Gram operator :math:`\mathbf{A}\mathbf{A}^{\ast}:\mathbb{R}^{N_{1} \times\cdots\times N_{K}} \to
        \mathbb{R}^{N_{1} \times\cdots\times N_{K}}`.

        Returns
        -------
        op: OpT
            Co-Gram operator.

        Note
        ----
        By default the co-Gram is computed by the composition ``self * self.T``.  This may not be the fastest way to
        compute the co-Gram operator.  If the co-Gram can be computed more efficiently (e.g. with a convolution), the
        user should re-define this method.
        """

        def op_expr(_) -> tuple:
            return ("cogram", self)

        op = self * self.T
        op._expr = types.MethodType(op_expr, op)
        return op.asop(SelfAdjointOp)

    def pinv(
        self,
        arr: pxt.NDArray,
        damp: pxt.Real,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pxt.NDArray:
        r"""
        Evaluate the Moore-Penrose pseudo-inverse :math:`\mathbf{A}^{\dagger}` at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., N1,...,NK) input points.
        damp: Real
            Positive dampening factor regularizing the pseudo-inverse.
        kwargs_init: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.CG`'s ``__init__()`` method.
        kwargs_fit: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.CG`'s ``fit()`` method.

        Returns
        -------
        out: NDArray
            (..., M1,...,MD) pseudo-inverse(s).

        Notes
        -----
        The Moore-Penrose pseudo-inverse of an operator :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}}
        \to \mathbb{R}^{N_{1} \times\cdots\times N_{K}}` is defined as the operator :math:`\mathbf{A}^{\dagger}:
        \mathbb{R}^{N_{1} \times\cdots\times N_{K}} \to \mathbb{R}^{M_{1} \times\cdots\times M_{D}}` verifying the
        Moore-Penrose conditions:

            1. :math:`\mathbf{A} \mathbf{A}^{\dagger} \mathbf{A} = \mathbf{A}`,
            2. :math:`\mathbf{A}^{\dagger} \mathbf{A} \mathbf{A}^{\dagger} = \mathbf{A}^{\dagger}`,
            3. :math:`(\mathbf{A}^{\dagger} \mathbf{A})^{\ast} = \mathbf{A}^{\dagger} \mathbf{A}`,
            4. :math:`(\mathbf{A} \mathbf{A}^{\dagger})^{\ast} = \mathbf{A} \mathbf{A}^{\dagger}`.

        This operator exists and is unique for any finite-dimensional linear operator.  The action of the pseudo-inverse
        :math:`\mathbf{A}^{\dagger} \mathbf{y}` for every :math:`\mathbf{y} \in \mathbb{R}^{N_{1} \times\cdots\times
        N_{K}}` can be computed in matrix-free fashion by solving the *normal equations*:

        .. math::

           \mathbf{A}^{\ast} \mathbf{A} \mathbf{x} = \mathbf{A}^{\ast} \mathbf{y}
           \quad\Leftrightarrow\quad
           \mathbf{x} = \mathbf{A}^{\dagger} \mathbf{y},
           \quad
           \forall (\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \times \mathbb{R}^{N_{1}
           \times\cdots\times N_{K}}.

        In the case of severe ill-conditioning, it is possible to consider the dampened normal equations for a
        numerically-stabler approximation of :math:`\mathbf{A}^{\dagger} \mathbf{y}`:

        .. math::

           (\mathbf{A}^{\ast} \mathbf{A} + \tau I) \mathbf{x} = \mathbf{A}^{\ast} \mathbf{y},

        where :math:`\tau > 0` corresponds to the `damp` parameter.
        """
        from pyxu.operator import HomothetyOp
        from pyxu.opt.solver import CG
        from pyxu.opt.stop import MaxIter

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init
        kwargs_init.update(show_progress=kwargs_init.get("show_progress", False))

        if np.isclose(damp, 0):
            A = self.gram()
        else:
            A = self.gram() + HomothetyOp(cst=damp, dim_shape=self.dim_shape)

        cg = CG(A, **kwargs_init)
        if "stop_crit" not in kwargs_fit:
            # .pinv() may not have sufficiently converged given the default CG stopping criteria.
            # To avoid infinite loops, CG iterations are thresholded.
            sentinel = MaxIter(n=20 * A.dim_size)
            kwargs_fit["stop_crit"] = cg.default_stop_crit() | sentinel

        b = self.adjoint(arr)
        cg.fit(b=b, **kwargs_fit)
        return cg.solution()

    def dagger(
        self,
        damp: pxt.Real,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pxt.OpT:
        r"""
        Return the Moore-Penrose pseudo-inverse operator :math:`\mathbf{A}^\dagger`.

        Parameters
        ----------
        damp: Real
            Positive dampening factor regularizing the pseudo-inverse.
        kwargs_init: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.CG`'s ``__init__()`` method.
        kwargs_fit: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.CG`'s ``fit()`` method.

        Returns
        -------
        op: OpT
            Moore-Penrose pseudo-inverse operator.
        """
        from pyxu.operator.interop import from_source

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            return self.pinv(
                arr,
                damp=_._damp,
                kwargs_init=_._kwargs_init,
                kwargs_fit=_._kwargs_fit,
            )

        def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
            return self.T.pinv(
                arr,
                damp=_._damp,
                kwargs_init=_._kwargs_init,
                kwargs_fit=_._kwargs_fit,
            )

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init

        dagger = from_source(
            cls=SquareOp if (self.dim_size == self.codim_size) else LinOp,
            dim_shape=self.codim_shape,
            codim_shape=self.dim_shape,
            embed=dict(
                _name="dagger",
                _damp=damp,
                _kwargs_init=copy.copy(kwargs_init),
                _kwargs_fit=copy.copy(kwargs_fit),
            ),
            apply=op_apply,
            adjoint=op_adjoint,
            _expr=lambda _: (_._name, _, _._damp),
        )
        return dagger

    @classmethod
    def from_array(
        cls,
        A: typ.Union[pxt.NDArray, pxt.SparseArray],
        dim_rank=None,
        enable_warnings: bool = True,
    ) -> pxt.OpT:
        r"""
        Instantiate a :py:class:`~pyxu.abc.LinOp` from its array representation.

        Parameters
        ----------
        A: NDArray
            (*codim_shape, *dim_shape) array.
        dim_rank: Integer
            Dimension rank :math:`D`. (Can be omitted if `A` is 2D.)
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.

        Returns
        -------
        op: OpT
            Linear operator
        """
        from pyxu.operator.linop.base import _ExplicitLinOp

        op = _ExplicitLinOp(
            cls,
            mat=A,
            dim_rank=dim_rank,
            enable_warnings=enable_warnings,
        )
        return op


class SquareOp(LinOp):
    r"""
    Base class for *square* linear operators, i.e. :math:`\mathbf{A}: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to
    \mathbb{R}^{M_{1} \times\cdots\times M_{D}}` (endomorphsisms).
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SQUARE)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        assert self.dim_size == self.codim_size

    def trace(self, **kwargs) -> pxt.Real:
        """
        Compute trace of the operator.

        Parameters
        ----------
        method: "explicit" | "hutchpp"

            * If `explicit`, compute the exact trace.
            * If `hutchpp`, compute an approximation. (Default)

        kwargs: ~collections.abc.Mapping
            Optional kwargs passed to:

            * `explicit`: :py:func:`~pyxu.math.trace`
            * `hutchpp`: :py:func:`~pyxu.math.hutchpp`

        Returns
        -------
        tr: Real
            Trace estimate.
        """
        from pyxu.math import hutchpp, trace

        method = kwargs.get("method", "hutchpp").lower().strip()

        if method == "explicit":
            func = sig_func = trace
            estimate = lambda: func(op=self, **kwargs)
        elif method == "hutchpp":
            func = sig_func = hutchpp
            estimate = lambda: func(op=self, **kwargs)
        else:
            raise NotImplementedError

        # Filter unsupported kwargs
        sig = inspect.Signature.from_callable(sig_func)
        kwargs = {k: v for (k, v) in kwargs.items() if (k in sig.parameters)}

        tr = estimate()
        return tr


class NormalOp(SquareOp):
    r"""
    Base class for *normal* operators.

    Normal operators satisfy the relation :math:`\mathbf{A} \mathbf{A}^{\ast} = \mathbf{A}^{\ast} \mathbf{A}`.  It can
    be `shown <https://www.wikiwand.com/en/Spectral_theorem#/Normal_matrices>`_ that an operator is normal iff it is
    *unitarily diagonalizable*, i.e.  :math:`\mathbf{A} = \mathbf{U} \mathbf{D} \mathbf{U}^{\ast}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_NORMAL)
        return frozenset(p)

    def cogram(self) -> pxt.OpT:
        return self.gram()


class SelfAdjointOp(NormalOp):
    r"""
    Base class for *self-adjoint* operators.

    Self-adjoint operators satisfy the relation :math:`\mathbf{A}^{\ast} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SELF_ADJOINT)
        return frozenset(p)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)


class UnitOp(NormalOp):
    r"""
    Base class for *unitary* operators.

    Unitary operators satisfy the relation :math:`\mathbf{A} \mathbf{A}^{\ast} = \mathbf{A}^{\ast} \mathbf{A} = I`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_UNITARY)
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.lipschitz = UnitOp.estimate_lipschitz(self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        return 1

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        if not np.isclose(damp, 0):
            out = pxu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / (1 + damp)
        return op

    def gram(self) -> pxt.OpT:
        from pyxu.operator import IdentityOp

        return IdentityOp(dim_shape=self.dim_shape)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)
        D = xp.ones(kwargs["k"], dtype=dtype)
        return D


class ProjOp(SquareOp):
    r"""
    Base class for *projection* operators.

    Projection operators are *idempotent*, i.e. :math:`\mathbf{A}^{2} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_IDEMPOTENT)
        return frozenset(p)


class OrthProjOp(ProjOp, SelfAdjointOp):
    r"""
    Base class for *orthogonal projection* operators.

    Orthogonal projection operators are *idempotent* and *self-adjoint*, i.e.  :math:`\mathbf{A}^{2} = \mathbf{A}` and
    :math:`\mathbf{A}^{\ast} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        self.lipschitz = OrthProjOp.estimate_lipschitz(self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        return 1

    def gram(self) -> pxt.OpT:
        return self

    def cogram(self) -> pxt.OpT:
        return self

    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.apply(arr)
        if not np.isclose(damp, 0):
            out = pxu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self / (1 + damp)
        return op


class PosDefOp(SelfAdjointOp):
    r"""
    Base class for *positive-definite* operators.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_POSITIVE_DEFINITE)
        return frozenset(p)


class LinFunc(ProxDiffFunc, LinOp):
    r"""
    Base class for real-valued linear functionals :math:`f: \mathbb{R}^{M_{1} \times\cdots\times M_{D}} \to \mathbb{R}`.

    Instances of this class must implement :py:meth:`~pyxu.abc.Map.apply`, and :py:meth:`~pyxu.abc.LinOp.adjoint`.

    If known, the Lipschitz constant :math:`L` should be stored in the :py:attr:`~pyxu.abc.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        for klass in [ProxDiffFunc, LinOp]:
            klass.__init__(
                self,
                dim_shape=dim_shape,
                codim_shape=codim_shape,
            )

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return LinOp.jacobian(self, arr)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        # Try all backends until one works.
        for ndi in pxd.NDArrayInfo:
            try:
                xp = ndi.module()
                g = self.grad(xp.ones(self.dim_shape))
                L = float(xp.sqrt(xp.sum(g**2)))
                return L
            except Exception:
                pass

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        ndi = pxd.NDArrayInfo.from_obj(arr)
        xp = ndi.module()

        sh = arr.shape[: -self.dim_rank]
        x = xp.ones((*sh, 1), dtype=arr.dtype)
        g = self.adjoint(x)

        if ndi == pxd.NDArrayInfo.DASK:
            # LinFuncs auto-determine [grad,prox,fenchel_prox]() via the user-specified adjoint().
            # Problem: cannot forward any core-chunk info to adjoint(), hence grad's core-chunks
            # may differ from `arr`. This is problematic since [grad,prox,fenchel_prox]() should
            # preserve core-chunks by default.
            if g.chunks != arr.chunks:
                g = g.rechunk(arr.chunks)
        return g

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        out = arr - tau * self.grad(arr)
        return out

    def fenchel_prox(self, arr: pxt.NDArray, sigma: pxt.Real) -> pxt.NDArray:
        return self.grad(arr)

    def cogram(self) -> pxt.OpT:
        from pyxu.operator import HomothetyOp

        L = self.estimate_lipschitz()
        return HomothetyOp(cst=L**2, dim_shape=1)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

        L = self.estimate_lipschitz()
        D = xp.array([L], dtype=dtype)
        return D

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.Width.DOUBLE.value)

        E = xp.ones((1, 1), dtype=dtype)
        A = self.adjoint(E)  # (1, *dim_shape)
        return A


def _core_operators() -> cabc.Set[pxt.OpC]:
    # Operators which can be sub-classed by end-users and participate in arithmetic rules.
    ops = set()
    for _ in globals().values():
        if inspect.isclass(_) and issubclass(_, Operator):
            ops.add(_)
    ops.remove(Operator)
    return ops


def _is_real(x) -> bool:
    if isinstance(x, pxt.Real):
        return True
    elif isinstance(x, pxd.supported_array_types()) and (x.size == 1):
        return True
    else:
        return False


__all__ = [
    "Operator",
    "Property",
    *map(lambda _: _.__name__, _core_operators()),
]
