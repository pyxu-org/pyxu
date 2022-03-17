import functools as ft
import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse.linalg as splin

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

if pycd.CUPY_ENABLED:
    import cupy as cp

__all__ = [
    "Property",
    "Apply",
    "Differential",
    "Gradient",
    "Adjoint",
    "SingleValued",
    "Proximal",
    "Map",
    "DiffMap",
    "Func",
    "DiffFunc",
    "ProxFunc",
    "ProxDiffFunc",
    "LinFunc",
    "LinOp",
    "NormalOp",
    "SelfAdjointOp",
    "SquareOp",
    "UnitOp",
    "ProjOp",
    "OrthProjOp",
    "PosDefOp",
]

MapLike = typ.Union["Map", "DiffMap", "Func", "DiffFunc", "ProxFunc", "ProxDiffFunc", "LinOp", "LinFunc"]
NonProxLike = typ.Union["Map", "DiffMap", "Func", "DiffFunc", "LinOp", "LinFunc"]


class Property:
    r"""
    Abstract base class for Pycsou's operators.
    """

    @classmethod
    def _property_list(cls) -> frozenset:
        r"""
        List all possible properties of Pycsou's base operators.

        Returns
        -------
        frozenset
            Set of properties.
        """
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
    def properties(cls) -> typ.Set[str]:
        r"""
        List the properties of the class.

        Returns
        -------
        typ.Set[str]
            Set of available properties.

        Examples
        --------

        >>> import pycsou.abc.operator as pycop
        >>> for op in pycop._base_operators:
        ...     print(op, op.properties())
        <class 'pycsou.abc.operator.Map'> {'apply'}
        <class 'pycsou.abc.operator.DiffMap'> {'apply', 'jacobian'}
        <class 'pycsou.abc.operator.DiffFunc'> {'grad', 'apply', 'single_valued', 'jacobian'}
        <class 'pycsou.abc.operator.LinOp'> {'apply', 'adjoint', 'jacobian'}
        <class 'pycsou.abc.operator.Func'> {'apply', 'single_valued'}
        <class 'pycsou.abc.operator.ProxFunc'> {'apply', 'single_valued', 'prox'}
        <class 'pycsou.abc.operator.ProxDiffFunc'> {'jacobian', 'single_valued', 'apply', 'grad', 'prox'}
        <class 'pycsou.abc.operator.LinFunc'> {'jacobian', 'single_valued', 'apply', 'grad', 'prox', 'adjoint'}

        """
        props = set(dir(cls))
        return set(props.intersection(cls._property_list()))

    @classmethod
    def has(cls, prop: pyct.VarName) -> bool:
        r"""
        Queries the class for certain properties.

        Parameters
        ----------
        prop: str | tuple(str)
            Queried properties.

        Returns
        -------
        bool
            ``True`` if the class has the queried properties, ``False`` otherwise.

        Examples
        --------

        >>> import pycsou.abc.operator as pycop
        >>> LinOp = pycop.LinOp
        >>> LinOp.has(('adjoint', 'jacobian'))
        True
        >>> LinOp.has(('adjoint', 'prox'))
        False
        """
        if isinstance(prop, str):
            prop = (prop,)
        return frozenset(prop) <= cls.properties()

    def __add__(self: MapLike, other: MapLike) -> MapLike:
        r"""
        Add two instances of :py:class:`~pycsou.abc.operator.Map` subclasses together (overloads the ``+`` operator).

        Parameters
        ----------
        self:  :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Left addend with shape (N,K).
        other: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Right addend with shape (M,L).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Sum of ``self`` and ``other``.

        Raises
        ------
        NotImplementedError
            If other is not an instance of :py:class:`~pycsou.abc.operator.Map`.

        ValueError
            If the addends' shapes are inconsistent (see below).

        Notes
        -----

        The addends' shapes must be `consistent` with one another, that is, they must:

            * be `range-broadcastable`, i.e. ``N=M or 1 in (N,M)``.
            * have `identical domain sizes` except if at least one of the two addends is `domain-agnostic`, i.e. ``K=L or None in (K,L)``.

        The addends `needs not be` instances of the same :py:class:`~pycsou.abc.operator.Map` subclass. The sum output is an
        instance of one of the following :py:class:`~pycsou.abc.operator.Map` (sub)classes: :py:class:`~pycsou.abc.operator.Map`,
        :py:class:`~pycsou.abc.operator.DiffMap`, :py:class:`~pycsou.abc.operator.Func`, :py:class:`~pycsou.abc.operator.DiffFunc`,
        :py:class:`~pycsou.abc.operator.ProxFunc`, :py:class:`~pycsou.abc.operator.ProxDiffFunc`, :py:class:`~pycsou.abc.operator.LinOp`,
        or :py:class:`~pycsou.abc.operator.LinFunc`.
        The output type is determined automatically by inspecting the shapes and common properties of the two addends as per the following table
        (the other half of the table can be filled by symmetry due to the commutativity of the addition):

        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        |              | Map | DiffMap | Func | DiffFunc | ProxFunc | LinOp   | LinFunc  | ProxDiffFunc |
        +==============+=====+=========+======+==========+==========+=========+==========+==============+
        | Map          | Map | Map     | Map  | Map      | Map      | Map     | Map      | Map          |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | DiffMap      |     | DiffMap | Map  | DiffMap  | Map      | DiffMap | DiffMap  | DiffMap      |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | Func         |     |         | Func | Func     | Func     | Map     | Func     | Func         |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | DiffFunc     |     |         |      | DiffFunc | Func     | DiffMap | DiffFunc | DiffFunc     |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | ProxFunc     |     |         |      |          | Func     | Map     | ProxFunc | Func         |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | LinOp        |     |         |      |          |          | LinOp   | LinOp    | DiffMap      |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | LinFunc      |     |         |      |          |          |         | LinFunc  | ProxDiffFunc |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+
        | ProxDiffFunc |     |         |      |          |          |         |          | DiffFunc     |
        +--------------+-----+---------+------+----------+----------+---------+----------+--------------+

        If the sum has one or more of the following properties ``[apply, jacobian, grad, adjoint, lipschitz, diff_lipschitz]``,
        the latter are defined as the sum of the corresponding properties of the addends. In the case ``ProxFunc/ProxDiffFunc/LinFunc + LinFunc``,
        the ``prox`` property is updated as described in the method ``__add__`` of the subclass :py:class:`~pycsou.abc.operator.ProxFunc`.

        .. Hint::

            Note that the case ``ProxFunc/ProxDiffFunc/LinFunc + LinFunc`` is handled in the method ``:py:meth:`~pycsou.abc.operator.ProxFunc.__add__``` of the subclass
            :py:class:`~pycsou.abc.operator.ProxFunc`.

        """
        if not isinstance(other, Map):
            raise NotImplementedError(f"Cannot add object of type {type(self)} with object of type {type(other)}.")
        try:
            out_shape = pycu.infer_sum_shape(self.shape, other.shape)
        except ValueError:
            raise ValueError(f"Cannot sum two maps with inconsistent shapes {self.shape} and {other.shape}.")
        shared_props = self.properties() & other.properties()
        shared_props.discard("prox")
        for Op in _base_operators:
            if Op.properties() == shared_props:
                break
        if Op in [LinOp, DiffFunc, ProxDiffFunc, LinFunc]:
            shared_props.discard("jacobian")
        shared_props.discard("single_valued")
        out_op = Op(out_shape)
        for prop in shared_props:
            if prop in ["lipschitz", "diff_lipschitz"]:
                setattr(out_op, "_" + prop, getattr(self, "_" + prop) + getattr(other, "_" + prop))
            else:

                @pycrt.enforce_precision(i="arr", o=False)  # Decorate composite method to avoid recasting [arr] twice.
                def composite_method(prop, _, arr: pyct.NDArray) -> typ.Union[pyct.NDArray, "LinOp"]:
                    return getattr(self, prop)(arr) + getattr(other, prop)(arr)

                setattr(out_op, prop, types.MethodType(ft.partial(composite_method, prop), out_op))
        return out_op.squeeze()

    def __mul__(self: MapLike, other: typ.Union[MapLike, pyct.Real]) -> MapLike:
        r"""
        Scale/compose one/two instance(s) of :py:class:`~pycsou.abc.operator.Map` subclasses respectively (overloads the ``*`` operator).

        Parameters
        ----------
        self: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Left factor with shape (N,K).
        other: numbers.Real | :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Right factor. Should be a real scalar or an instance of :py:class:`~pycsou.abc.operator.Map` subclasses with shape (M,L).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Product (scaling or composition) of ``self`` with ``other``, with shape (N,K) (for scaling) or (N,L) (for composition).

        Raises
        ------
        NotImplementedError
            If other is not an instance of :py:class:`~pycsou.abc.operator.Map` or :py:class:`numbers.Real`.

        ValueError
            If the factors' shapes are inconsistent (see below).

        Notes
        -----

        The factors' shapes must be `consistent` with one another, that is ``K=M``. If the left factor is domain-agnostic
        (i.e. ``K=None``) then ``K!=M`` is allowed. The right factor can also be domain-agnostic, in which case the output
        is also domain-agnostic and has shape (N, None).


        The factors `needs not be` instances of the same :py:class:`~pycsou.abc.operator.Map` subclass. The product output is an
        instance of one of the following :py:class:`~pycsou.abc.operator.Map` (sub)classes: :py:class:`~pycsou.abc.operator.Map`,
        :py:class:`~pycsou.abc.operator.DiffMap`, :py:class:`~pycsou.abc.operator.Func`, :py:class:`~pycsou.abc.operator.DiffFunc`,
        :py:class:`~pycsou.abc.operator.ProxFunc`, :py:class:`~pycsou.abc.operator.ProxDiffFunc`, :py:class:`~pycsou.abc.operator.LinOp`,
        or :py:class:`~pycsou.abc.operator.LinFunc`.
        The output type is determined automatically by inspecting the shapes and common properties of the two factors as per the following table:

        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        |              | Map     | DiffMap  | Func | DiffFunc | ProxFunc | LinOp       | LinFunc  | ProxDiffFunc |
        +==============+=========+==========+======+==========+==========+=============+==========+==============+
        | Map          | Map     | Map      | Map  | Map      | Map      | Map         | Map      | Map          |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | DiffMap      | Map     | DiffMap  | Map  | DiffMap  | Map      | DiffMap     | DiffMap  | DiffMap      |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | Func         | Func    | Func     | Func | Func     | Func     | Func        | Func     | Func         |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | DiffFunc     | Func    | DiffFunc | Func | DiffFunc | Func     | DiffFunc    | DiffFunc | DiffFunc     |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | ProxFunc     | Func    | Func     | Func | Func     | Func     | Func        | Func     | Func         |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | LinOp        | Map     | DiffMap  | Map  | DiffMap  | Map      | LinOp       | LinOp    | DiffMap      |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | LinFunc      | Func    | DiffFunc | Func | DiffFunc | Func     | LinFunc     | LinFunc  | DiffFunc     |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+
        | ProxDiffFunc | Func    | DiffFunc | Func | DiffFunc | Func     | DiffFunc    | DiffFunc | DiffFunc     |
        +--------------+---------+----------+------+----------+----------+-------------+----------+--------------+


        If the product output has one or more of the following properties ``[apply, jacobian, grad, adjoint, lipschitz, diff_lipschitz]``,
        the latter are defined from the corresponding properties of the factors as follows (the pseudocode below is mathematically equivalent to but does
        not necessarily reflect the actual implementation):

        .. code-block:: python3

            prod._lipschitz = self._lipschitz * other._lipschitz
            prod.apply = lambda x: self.apply(other.apply(x))
            prod.jacobian = lambda x: self.jacobian(other.apply(x)) * other.jacobian(x)
            prod.grad = lambda x: other.jacobian(x).adjoint(self.grad(other.apply(x)))
            prod.adjoint = lambda x: other.adjoint(self.adjoint(x))

        where ``prod = self * other`` denotes the product of ``self`` with ``other``.
        Moreover, the (potential) attribute :py:attr:`~pycsou.abc.operator.DiffMap._diff_lipschitz` is updated as follows:

        .. code-block:: python3

            if isinstance(self, LinOp):
                prod._diff_lipschitz = self._lipschitz * other._diff_lipschitz
            elif isinstance(other, LinOp):
                prod._diff_lipschitz = self._diff_lipschitz * (other._lipschitz) ** 2
            else:
                prod._diff_lipschitz = np.infty

        Unlike the other properties listed above, automatic update of the :py:attr:`~pycsou.abc.operator.DiffMap._diff_lipschitz` attribute is
        hence only possible when either ``self`` or ``other`` is a :py:class:`~pycsou.abc.operator.LinOp` (otherwise it is set to its default value ``np.infty``).

        .. Hint::

            The case ``ProxFunc * LinOp`` yields in general a :py:class:`~pycsou.abc.operator.Func` object except when either:

                * The :py:class:`~pycsou.abc.operator.ProxFunc` factor is of type :py:class:`~pycsou.abc.operator.LinFunc`,
                * The :py:class:`~pycsou.abc.operator.LinOp` factor is a scaling or of type :py:class:`~pycsou.abc.operator.UnitOp`.

            In which case, the product will be of type :py:class:`~pycsou.abc.operator.ProxFunc` (see :py:meth:`~pycsou.abc.operator.ProxFunc.__mul__` of :py:class:`~pycsou.abc.operator.ProxFunc` for more).
            This case, together with the case ``ProxFunc * scalar`` are handled separately in the method :py:meth:`~pycsou.abc.operator.ProxFunc.__mul__` of the subclass :py:class:`~pycsou.abc.operator.ProxFunc`,
            where the product's ``prox`` property update is described.
        """
        if isinstance(other, pyct.Real):
            from pycsou.linop.base import HomothetyOp

            hmap = HomothetyOp(other, dim=self.shape[0])
            return hmap.__mul__(self)
        elif not isinstance(other, Map):
            raise NotImplementedError(f"Cannot multiply object of type {type(self)} with object of type {type(other)}.")
        try:
            out_shape = pycu.infer_composition_shape(self.shape, other.shape)
        except ValueError:
            raise ValueError(f"Cannot compose two maps with inconsistent shapes {self.shape} and {other.shape}.")
        shared_props = self.properties() & other.properties()
        shared_props.discard("prox")
        if self.shape[0] == 1 and "jacobian" in shared_props:
            shared_props.update({"grad", "single_valued"})
        for Op in _base_operators:
            if Op.properties() == shared_props:
                break
        if Op in [LinOp, DiffFunc, ProxDiffFunc, LinFunc]:
            shared_props.discard("jacobian")
        shared_props.discard("single_valued")
        out_op = Op(out_shape)
        for prop in shared_props:  # ("apply", "lipschitz", "jacobian", "diff_lipschitz", "grad", "adjoint")
            if prop == "apply":
                out_op.apply = types.MethodType(lambda _, arr: self.apply(other.apply(arr)), out_op)
            elif prop == "lipschitz":
                out_op._lipschitz = self._lipschitz * other._lipschitz
            elif prop == "diff_lipschitz":
                if isinstance(self, LinOp):
                    out_op._diff_lipschitz = self._lipschitz * other._diff_lipschitz
                elif isinstance(other, LinOp):
                    out_op._diff_lipschitz = self._diff_lipschitz * (other._lipschitz) ** 2
                else:
                    out_op._diff_lipschitz = np.infty
            elif prop == "grad":

                @pycrt.enforce_precision(i="arr")
                def composite_grad(_, arr: pyct.NDArray) -> pyct.NDArray:
                    return other.jacobian(arr).adjoint(self.grad(other.apply(arr)))

                out_op.grad = types.MethodType(composite_grad, out_op)
            elif prop == "jacobian":

                @pycrt.enforce_precision(i="arr", o=False)
                def composite_jacobian(_, arr: pyct.NDArray) -> "LinOp":
                    return self.jacobian(other.apply(arr)) * other.jacobian(arr)

                out_op.jacobian = types.MethodType(composite_jacobian, out_op)
            elif prop == "adjoint":
                out_op.adjoint = types.MethodType(lambda _, arr: other.adjoint(self.adjoint(arr)), out_op)
        return out_op.squeeze()

    def __rmul__(self: MapLike, other: pyct.Real) -> MapLike:
        r"""
        Scale an instance of :py:class:`~pycsou.abc.operator.Map` subclasses by multiplying it from the left with a real number, i.e. ``prod = scalar * self``.

        Parameters
        ----------
        other: numbers.Real
            Left scaling factor.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Scaled operator.
        """
        return self.__mul__(other)

    def __pow__(self: MapLike, power: int) -> NonProxLike:
        r"""
        Exponentiate (i.e. compose with itself) an instance of :py:class:`~pycsou.abc.operator.Map` subclasses ``power`` times (overloads the ``**`` operator).

        Parameters
        ----------
        power: int
            Exponentiation power (number of times the operator is composed with itself).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Exponentiated operator.
        """
        if type(power) is int:
            if power == 0:
                from pycsou.linop.base import IdentityOp

                exp_map = IdentityOp(shape=self.shape)
            else:
                exp_map = self
                for i in range(1, power):
                    exp_map = self.__mul__(exp_map)
            return exp_map
        else:
            raise NotImplementedError

    def __neg__(self: MapLike) -> MapLike:
        r"""
        Negate a map (overloads the ``-`` operator). Alias for ``self.__mul__(-1)``.
        """
        return self.__mul__(-1)

    def __sub__(self: MapLike, other: MapLike) -> NonProxLike:
        r"""
        Take the difference between two instances of :py:class:`~pycsou.abc.operator.Map` subclasses. Alias for ``self.__add__(other.__neg__())``.

        Parameters
        ----------
        self: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Left term of the subtraction with shape (N,K).
        other: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Right term of the subtraction with shape (M,L).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Difference between ``self`` and ``other``.
        """
        return self.__add__(other.__neg__())

    def __truediv__(self: MapLike, scalar: pyct.Real) -> MapLike:
        r"""
        Divides a map by a real ``scalar`` (overloads the ``/`` operator). Alias for ``self.__mul__(1 / scalar)``.
        """
        if isinstance(scalar, pyct.Real):
            return self.__mul__(1 / scalar)
        else:
            raise NotImplementedError

    def argscale(self: MapLike, scalar: pyct.Real) -> MapLike:
        r"""
        Dilate/shrink the domain of an instance of :py:class:`~pycsou.abc.operator.Map` subclasses.

        Parameters
        ----------
        scalar: numbers.Real
            Dilation/shrinking factor.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Domain-rescaled operator.

        Raises
        ------
        NotImplementedError
            If scalar is not a real number.

        Notes
        -----
        Calling ``self.argscale(scalar)`` is equivalent to precomposing ``self`` with the (unitary) linear operator :py:class:`~pycsou.linop.base.HomotethyOp`:

        .. code-block:: python3

            # The two statements below are functionally equivalent
            out1 = self.argscale(scalar)
            out2 = self * HomotethyOp(scalar, dim=self.shape[1])

        """
        if isinstance(scalar, pyct.Real):
            from pycsou.linop.base import HomothetyOp

            hmap = HomothetyOp(scalar, dim=self.shape[1])
            return self.__mul__(hmap)
        else:
            raise NotImplementedError

    @pycrt.enforce_precision(i="arr", o=False)
    def argshift(self: MapLike, arr: pyct.NDArray) -> MapLike:
        r"""
        Domain-shift an instance of :py:class:`~pycsou.abc.operator.Map` subclasses by ``arr``.

        Parameters
        ----------
        arr: NDArray
            Shift vector with size (N,).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Domain-shifted operator.

        Raises
        ------
        ValueError
            If ``arr`` is not of type NDArray or has incorrect size, i.e. ``N != self.shape[1]``.

        Notes
        -----
        The output domain-shifted operator has either the same type of ``self`` or is of type
        :py:class:`~pycsou.abc.operator.DiffMap`/:py:class:`~pycsou.abc.operator.DiffFunc` when ``self`` is a
        :py:class:`~pycsou.abc.operator.LinOp`/:py:class:`~pycsou.abc.operator.LinFunc` object respectively (since shifting does not preserve linearity).
        Moreover, if the output has one or more of the following properties ``[apply, jacobian, grad, prox, lipschitz, diff_lipschitz]``,
        the latter are defined from the corresponding properties of ``self`` as follows (the pseudocode below is mathematically equivalent to but does
        not necessarily reflect the actual implementation):

        .. code-block:: python3

            out._lipschitz = self._lipschitz
            out._diff_lipschitz = self._diff_lipschitz
            out.apply = lambda x: self.apply(x + arr)
            out.jacobian = lambda x: self.jacobian(x + arr)
            out.grad = lambda x: self.grad(x + arr)
            out.prox = lambda x, tau: self.prox(x + arr, tau) - arr

        where ``out = self.argshift(arr)`` denotes the domain-shifted output.


        """
        try:
            arr = arr.copy().squeeze()
        except:
            raise ValueError("Argument [arr] must be of type NDArray.")
        if (self.shape[-1] is None) or (self.shape[-1] == arr.shape[-1]):
            out_shape = (self.shape[0], arr.shape[-1])
        else:
            raise ValueError(f"Invalid lag shape: {arr.shape[-1]} != {self.shape[-1]}")
        if isinstance(self, LinFunc):  # Shifting a linear map makes it an affine map.
            out_op = DiffFunc(shape=out_shape)
        elif isinstance(self, LinOp):  # Shifting a linear map makes it an affine map.
            out_op = DiffMap(shape=out_shape)
        else:
            out_op = self.__class__(shape=out_shape)
        props = out_op.properties()
        if out_op == DiffFunc:
            props.discard("jacobian")
        props.discard("single_valued")
        for prop in out_op.properties():
            if prop in ["lipschitz", "diff_lipschitz"]:
                setattr(out_op, "_" + prop, getattr(self, "_" + prop))
            elif prop == "prox":
                out_op.prox = types.MethodType(
                    ft.partial(lambda arr, _, x, tau: self.prox(x + arr, tau) - arr, arr), out_op
                )
            else:

                def argshifted_method(prop, arr, _, x: pyct.NDArray) -> typ.Union[pyct.NDArray, "LinOp"]:
                    return getattr(self, prop)(x + arr)

                setattr(out_op, prop, types.MethodType(ft.partial(argshifted_method, prop, arr), out_op))
        return out_op.squeeze()


class SingleValued(Property):
    r"""
    Mixin class defining the *single-valued* property.
    """

    def single_valued(self) -> typ.Literal[True]:
        r"""

        Returns
        -------
        bool
            True
        """
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
        Evaluate a map at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (...,M) input to the apply method.

        Returns
        -------
        NDArray
            (...,N) output to the apply method. Same number of dimensions as ``arr``.


        .. Important::

            This method should abide by the rules described in :ref:`developer-notes`.

        """
        raise NotImplementedError

    def lipschitz(self, **kwargs) -> float:
        r"""
        Compute the Lipschitz constant of the :py:meth:`~pycsou.abc.operator.Apply.apply` function.

        Returns
        -------
        float
            Lipschitz constant.

        Notes
        -----
        A constant :math:`L_\mathbf{h}>0` is said to be a *Lipschitz constant* for a map :math:`\mathbf{h}:\mathbb{R}^N\to \mathbb{R}^M` if:

        .. math::

            \|\mathbf{h}(\mathbf{x})-\mathbf{h}(\mathbf{y})\|_{\mathbb{R}^M} \leq L_\mathbf{h} \|\mathbf{x}-\mathbf{y}\|_{\mathbb{R}^N}, \qquad \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^N,

        where :math:`\|\cdot\|_{\mathbb{R}^M}`$` and :math:`\|\cdot\|_{\mathbb{R}^N}`$` are the canonical norms on their respective spaces. The smallest of all Lipschitz constants for a given map
        is called the *optimal Lipschitz constant*.
        """
        raise NotImplementedError


class Differential(Property):
    r"""
    Mixin class defining the *differentiability* property.
    """

    def jacobian(self, arr: pyct.NDArray) -> "LinOp":
        r"""
        Evaluate the Jacobian of a vector-valued multi-dimensional differentiable map at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (M,) input. Must be a 1-D array.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Jacobian operator at the requested point ``arr``.

        Notes
        -----
        Let :math:`\mathbf{h}=[h_1, \ldots, h_N]: \mathbb{R}^M\to\mathbb{R}^N` be a differentiable multidimensional map,
        then the *Jacobian* (or *differential*) of :math:`\mathbf{h}` at a given point :math:`\mathbf{z}\in\mathbb{R}^M` is defined as
        the best linear approximation of :math:`\mathbf{h}` near the point :math:`\mathbf{z}`, in the sense that

        .. math:: \mathbf{h}(\mathbf{x}) - \mathbf{h}(\mathbf{z})=\mathbf{J}_{\mathbf {h}}(\mathbf{z})(\mathbf{x} -\mathbf{z})+o(\|\mathbf{x} -\mathbf{z} \|)\quad \text{as} \quad \mathbf {x} \to \mathbf {z}.

        The Jacobian admits the following matrix representation:

        .. math::  (\mathbf{J}_{\mathbf{h}}(\mathbf{x}))_{ij}:=\frac{\partial h_i}{\partial x_j}(\mathbf{x}), \qquad \forall (i,j)\in\{1,\cdots,N\}\times \{1,\cdots,M\}.
        """
        raise NotImplementedError

    def diff_lipschitz(self, **kwargs) -> float:
        r"""
        Compute the Lipschitz constant of the :py:meth:`~pycsou.abc.operator.Differential.jacobian` function.

        Returns
        -------
        float
            Lipschitz constant.

        Notes
        -----
        A Lipschitz constant :math:`L_{\mathbf{J}_{\mathbf{h}}}>0` of the Jacobian map :math:`\mathbf{J}_{\mathbf{h}}:\mathbf{R}^N\to \mathbf{R}^{M\times N}` is such that:

        .. math::

            \|\mathbf{J}_{\mathbf{h}}(\mathbf{x})-\mathbf{J}_{\mathbf{h}}(\mathbf{y})\|_{\mathbb{R}^{M\times N}} \leq L_{\mathbf{J}_{\mathbf{h}}} \|\mathbf{x}-\mathbf{y}\|_{\mathbb{R}^N}, \qquad \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^N,

        where :math:`\|\cdot\|_{\mathbb{R}^{M\times N}}` and :math:`\|\cdot\|_{\mathbb{R}^N}` are the canonical norms on their respective spaces.
        The smallest of all Lipschitz constants for the Jacobian map is called its *optimal Lipschitz constant*.
        """
        raise NotImplementedError


class Gradient(Differential):
    r"""
    Mixin class defining the *grad* property.
    """

    @pycrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pyct.NDArray) -> "LinOp":
        r"""
        Construct the Jacobian linear functional associated with the gradient.

        Parameters
        ----------
        arr: NDArray
            (M,) input. Must be a 1-D array.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Jacobian linear functional.

        Notes
        -----
        The Jacobian of a functional :math:`f:\mathbb{R}^M\to \mathbb{R}` is given, for every :math:`\mathbf{x}\in\mathbb{R}^M`, by

        .. math:: \mathbf{J}_f(\mathbf{x})\mathbf{z}= \langle \mathbf{z}, \nabla f (\mathbf{x})\rangle_2 =  \nabla f (\mathbf{x})^T\mathbf{z}, \qquad \forall \mathbf{z}\in\mathbb{R}^M,

        where :math:`\nabla f (\mathbf{x})` denotes the *gradient* of :math:`f` (see :py:meth:`~pycsou.abc.operator.Gradient.grad`).
        The Jacobian matrix is hence given by the transpose of the gradient: :math:`\mathbf{J}_f(\mathbf{x})=\nabla f (\mathbf{x})^T`.
        """
        from pycsou.linop.base import ExplicitLinFunc

        return ExplicitLinFunc(self.grad(arr))

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Evaluate the gradient of a functional at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (...,M) input.

        Returns
        -------
        NDArray
            (...,N) gradient. Must have the same dimensions as ``arr``.

        Notes
        -----
        The gradient of a functional :math:`f:\mathbb{R}^M\to \mathbb{R}` is given, for every :math:`\mathbf{x}\in\mathbb{R}^M`, by

        .. math::

            \nabla f(\mathbf{x}):=\left[\begin{array}{c} \frac{\partial f}{\partial x_1}(\mathbf{x}) \\\vdots \\\frac{\partial f}{\partial x_M}(\mathbf{x})  \end{array}\right].


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
        Evaluate the adjoint of a linear operator at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (...,N) input.

        Returns
        -------
        NDArray
            (...,M) output. Must have the same dimensions as ``arr``.


        Notes
        -----
        The *adjoint* :math:`\mathbf{L}^\ast:\mathbb{R}^N\to \mathbb{R}^M` of a linear operator :math:`\mathbf{L}:\mathbb{R}^M\to \mathbb{R}^N` is defined as:

        .. math:: \langle \mathbf{x}, \mathbf{L}^\ast\mathbf{y}\rangle_{\mathbb{R}^M}:=\langle \mathbf{L}\mathbf{x}, \mathbf{y}\rangle_{\mathbb{R}^N}, \qquad\forall (\mathbf{x},\mathbf{y})\in \mathbb{R}^M\times \mathbb{R}^N.


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
        Evaluate the proximity operator of a ``tau``-scaled functional at the point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (...,M) input at which to evaluate the proximity operator.
        tau: numbers.Real
            Positive scaling parameter of the proximity step.
        Returns
        -------
        NDArray
            (...,M) output of the proximity operator.

        Notes
        -----
        The *proximity operator* of a ``tau``-scaled functional :math:`f:\mathbb{R}^M\to \mathbb{R}` is defined as:

        .. math:: \mathbf{\text{prox}}_{\tau f}(\mathbf{z}):=\arg\min_{\mathbf{x}\in\mathbb{R}^M} f(x)+\frac{1}{2\tau} \|\mathbf{x}-\mathbf{z}\|_2^2, \quad \forall \mathbf{z}\in\mathbb{R}^M.


        .. Important::

            This method should abide by the rules described in :ref:`developer-notes`.
        """
        raise NotImplementedError

    @pycrt.enforce_precision(i=("arr", "sigma"), o=True)
    def fenchel_prox(self, arr: pyct.NDArray, sigma: pyct.Real) -> pyct.NDArray:
        r"""
        Evaluate the proximity operator of the ``sigma``-scaled Fenchel conjugate of a functional at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (...,M) input at which to evaluate the proximity operator.
        sigma: numbers.Real
            Positive scaling parameter of the proximity step.

        Returns
        -------
        NDArray
            (...,M) output of the proximity operator.

        Notes
        -----
        The *Fenchel conjugate* is defined as:

        .. math::

           f^\ast(\mathbf{z}):=\max_{\mathbf{x}\in\mathbb{R}^M} \langle \mathbf{x},\mathbf{z} \rangle - f(\mathbf{x}).

        From **Moreau's identity**, its proximal operator is given by:

        .. math::

           \mathbf{\text{prox}}_{\sigma f^\ast}(\mathbf{z})= \mathbf{z}- \sigma \mathbf{\text{prox}}_{f/\sigma}(\mathbf{z}/\sigma).
        """
        return arr - sigma * self.prox(arr=arr / sigma, tau=1 / sigma)


class Map(Apply):
    r"""
    Base class for real-valued maps :math:`\mathbf{M}:\mathbb{R}^M\to \mathbb{R}^N`.

    Any instance/subclass of this class must implement the method :py:meth:`~pycsou.abc.operator.Apply.apply`. If the map
    is Lipschitz-continuous and the Lipschitz constant is known, the latter can be stored in the private instance attribute ``_lipschitz``
    (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete map, it is recommended to subclass :py:class:`~pycsou.abc.operator.Map` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import Map
    >>> class  ReLu(Map):
    ...    def __init__(self, shape):
    ...        super(ReLu, self).__init__(shape)
    ...        self._lipschitz = 1 # Lipschitz constant of the map
    ...    def apply(self, arr):
    ...        return np.clip(arr, a_min=0, a_max=None)
    >>> relu=ReLu((10,10)) # creates an instance
    >>> relu(np.arange(10)-2)
    array([0., 0., 0., 1., 2., 3., 4., 5., 6., 7.])

    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.Shape):
        r"""

        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.

        Raises
        ------
        An error is raised if the specified shape is invalid.

        Examples
        --------

        >>> from pycsou.abc import Map
        >>> m1 = Map((3,4)) # Map of shape (3,4): domain of size 4, co-domain of size 3.
        >>> m2 = Map((6,None)) # Domain-agnostic map of shape (6,None): domain of arbitrary size, co-domain of size 6.
        """
        if len(shape) > 2:
            raise NotImplementedError(
                "Shapes of map objects must be tuples of length 2 (tensorial maps not supported)."
            )
        elif shape[0] is None:
            raise ValueError("Codomain agnostic maps are not supported.")
        self._shape = shape
        self._lipschitz = np.infty

    @property
    def shape(self) -> typ.Tuple[int, typ.Union[int, None]]:
        r"""
        Returns
        -------
        tuple(int, [int|None])
            Shape (N, M) of the map.
        """
        return self._shape

    @property
    def dim(self) -> typ.Union[int, None]:
        r"""
        Returns
        -------
        int | None
            Dimension M of the map's domain.
        """
        return self.shape[1]

    @property
    def codim(self) -> int:
        r"""
        Returns
        -------
        int | None
            Dimension N of the map's co-domain.
        """
        return self.shape[0]

    def squeeze(self) -> typ.Union["Map", "Func"]:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.Map` object to the right type (:py:class:`~pycsou.abc.operator.Func` or :py:class:`~pycsou.abc.operator.Map`)
        given its co-domain's dimension.

        Returns
        -------
        pycsou.abc.operator.Map | pycsou.abc.operator.Func
            Output is a :py:class:`~pycsou.abc.operator.Func` object if ``self.codim==1`` and a :py:class:`~pycsou.abc.operator.Map` object otherwise.

        Examples
        --------

        Consider the :py:class:`Median` operator defined in the :ref:`developer-notes`. The latter was declared as a  :py:class:`~pycsou.abc.operator.Map` subclass
        but its co-domain has actually dimension 1. It would therefore be better indicated to see :py:class:`Median` objects as :py:class:`~pycsou.abc.operator.Func` objects.
        This recasting can be performed at posteriori using the method :py:meth:`squeeze`:

        >>> m = Median()
        >>> type(m)
        pycsou.abc.operator.Map
        >>> type(m.squeeze())
        pycsou.abc.operator.Func

        """
        return self._squeeze(out=Func)

    def _squeeze(
        self: MapLike,
        out: typ.Type[typ.Union["Func", "DiffFunc", "LinFunc"]],
    ) -> MapLike:
        if self.shape[0] == 1:
            obj = self.specialize(cast_to=out)
        else:
            obj = self
        return obj

    def lipschitz(self, **kwargs) -> float:
        r"""
        Return the map's Lipschitz constant.

        Returns
        -------
        float
            The map's Lipschitz constant.
        """
        return self._lipschitz

    def specialize(
        self: MapLike,
        cast_to: typ.Union[type, typ.Type["Property"]],
    ) -> "Property":
        r"""
        Recast an object of a :py:class:`~pycsou.abc.operator.Map` subclass to another :py:class:`~pycsou.abc.operator.Map` subclass
        lower in the class hierarchy.

        Parameters
        ----------
        cast_to: type
            Target type for the recast.

        Returns
        -------
        pycsou.abc.operator.Property
            Copy of the object with the new type.

        Raises
        ------
        ValueError
            If the ``cast_to`` type is higher in the class hierarchy than the current type of ``self``.

        Examples
        --------
        Consider the following operator implemented as a :py:class:`~pycsou.abc.operator.Map`:

        .. code-block::

            import pycsou.abc as pyca
            import numpy as np

            class Sum(pyca.Map):
                def __init__(self, dim):
                    super(Sum, self).__init__(shape=(1, dim))
                    self._lipschitz = np.sqrt(dim)
                    self._diff_lipschitz = 0

                def apply(self, arr):
                    return np.sum(arr, axis=-1, keepdims=True)

                def adjoint(self, arr):
                    return arr * np.ones(self.shape[-1])




        While being functionally equivalent to a linear operator (it has the methods :py:meth:`apply` and  :py:meth:`adjoint`)
        the :py:class:`Sum` operator does not possess all the convenience methods attached to the  :py:class:`~pycsou.abc.operator.LinOp` class.
        The method :py:meth:`specialize` allows the user to recast :py:class:`Sum` objects as a :py:class:`~pycsou.abc.operator.LinOp` objects:

        >>> s = Sum(5)
        >>> type(s.specialize(pyca.LinOp))
        pycsou.abc.operator.LinOp

        .. Warning::

            This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
            :ref:`developer-notes`.

        Notes
        -----
        If ``self`` does not implement all the methods of the ``cast_to`` target class, then unimplemented methods will raise a ``NotImplementedError``
        when called.
        """
        if cast_to == self.__class__:
            obj = self
        else:
            if self.properties() > cast_to.properties():
                raise ValueError(
                    f"Cannot specialize an object of type {self.__class__} to an object of type {cast_to}."
                )
            obj = cast_to(self.shape)
            for prop in self.properties():
                if prop == "jacobian" and cast_to.has("single_valued"):
                    obj.grad = types.MethodType(lambda _, x: self.jacobian(x).asarray().reshape(-1), obj)
                else:
                    setattr(obj, prop, getattr(self, prop))
        return obj

    @classmethod
    def from_source(cls, shape: typ.Tuple[int, typ.Union[int, None]], **kwargs) -> MapLike:
        r"""
        Create an instance of a :py:class:`~pycsou.abc.operator.Map` subclass by directly defining the appropriate
        callables to the constructor of this class.

        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
            If ``True`` the returned object is of type `~pycsou.abc.operator.SingleValued`.

        kwargs:
            Keyword arguments corresponding to the callables of this class. For example, to create an instance of
            :py:class:`~pycsou.abc.operator.DiffMap`, the callables for :py:meth:`~pycsou.abc.operator.Apply.apply` and
            :py:meth:`~pycsou.abc.operator.Jacobian.jacobian` must be supplied.

        Returns
        -------
        pycsou.abc.operator.Property
            Output is a :py:class:`~pycsou.abc.operator.Map` subclass object.

        Examples
        --------
        >>> from pycsou.abc.operator import LinFunc
        >>> map_shape = (1, 5)
        >>> map_properties = {
        ...     "apply": lambda x: np.sum(x, axis=-1, keepdims=True),
        ...     "adjoint": lambda x: x * np.ones(map_shape[-1]),
        ...     "grad": lambda x: np.ones(shape=x.shape[:-1] + (map_shape[-1],)),
        ...     "prox": lambda x, tau: x - tau * np.ones(map_shape[-1]),
        ... }
        >>> func = LinFunc.from_source(shape=map_shape,
        ...                           **map_properties)
        >>> type(func)
        <class 'pycsou.abc.operator.LinFunc'>
        >>> func.apply(np.ones((1,5)))
        array([[5.]])
        >>> func.adjoint(1)
        array([1., 1., 1., 1., 1.])


        .. Warning::
        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`, :py:meth:`~pycsou.abc.operator.LinOp.from_sciop`,
        :py:meth:`~pycsou.abc.operator.LinFunc.from_array`

        """

        properties = set(kwargs.keys())
        op_properties = set(cls.properties())
        if cls in [LinOp, DiffFunc, ProxDiffFunc, LinFunc]:
            op_properties.discard("jacobian")
            op_properties.discard("single_valued")
        if op_properties == properties:
            out_op = cls(shape)
        else:
            raise ValueError(f"Cannot create a {cls.__name__} object with the given properties.")
        for prop in properties:
            if prop in ["lipschitz", "diff_lipschitz"]:
                setattr(out_op, "_" + prop, kwargs["_" + prop])
            elif prop == "prox":
                f = lambda key, _, arr, tau: kwargs[key](arr, tau)
                setattr(out_op, prop, types.MethodType(ft.partial(f, prop), out_op))
            else:
                f = lambda key, _, arr: kwargs[key](arr)
                setattr(out_op, prop, types.MethodType(ft.partial(f, prop), out_op))
        return out_op


class DiffMap(Map, Differential):
    r"""
    Base class for real-valued differentiable maps :math:`\mathbf{M}:\mathbb{R}^M\to \mathbb{R}^N`.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply` and :py:meth:`~pycsou.abc.operator.Differential.jacobian`.
    If the map and/or its Jacobian are Lipschitz-continuous and the Lipschitz constants are known, the latter can be stored in the private instance attributes ``_lipschitz``
    and ``_diff_lipschitz`` respectively (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete differentiable map, it is recommended to subclass :py:class:`~pycsou.abc.operator.DiffMap` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import DiffMap
    >>> from pycsou.linop.base import ExplicitLinOp
    >>> class  Sin(DiffMap):
    ...    def __init__(self, shape):
    ...        super(Sin, self).__init__(shape)
    ...        self._lipschitz = self._diff_lipschitz = 1 # Lipschitz constants of the map and its derivative
    ...    def apply(self, arr):
    ...        return np.sin(arr)
    ...    def jacobian(self, arr):
    ...        return ExplicitLinOp(np.diag(np.cos(arr)))
    >>> sin = Sin((10,10)) # creates an instance


    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.Shape):
        r"""
        Parameters
        ----------
        shape: tuple(int, [int|None])
            Shape of the map (N,M). Shapes of the form (N, None) can be used to denote domain-agnostic maps.
        """
        super(DiffMap, self).__init__(shape)
        self._diff_lipschitz = np.infty

    def squeeze(self) -> typ.Union["DiffMap", "DiffFunc"]:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.DiffMap` object to the right type (:py:class:`~pycsou.abc.operator.DiffFunc` or :py:class:`~pycsou.abc.operator.DiffMap`)
        given its co-domain's dimension.

        Returns
        -------
        pycsou.abc.operator.DiffMap | pycsou.abc.operator.DiffFunc
            Output is a :py:class:`~pycsou.abc.operator.DiffFunc` object if ``self.codim==1`` and a :py:class:`~pycsou.abc.operator.DiffMap` object otherwise.

        See Also
        --------
        :py:meth:`pycsou.abc.operator.Map.squeeze`
        """
        return self._squeeze(out=DiffFunc)

    def diff_lipschitz(self, **kwargs) -> float:
        r"""
        Return the Jacobian's Lipschitz constant.

        Returns
        -------
        float
            The Jacobian's Lipschitz constant.
        """
        return self._diff_lipschitz


class Func(Map, SingleValued):
    r"""
    Base class for real-valued functionals :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}`.

    Any instance/subclass of this class must implement the method :py:meth:`~pycsou.abc.operator.Apply.apply`.
    If the functional is Lipschitz-continuous and the Lipschitz constant is known, the latter can be stored in the private instance attribute ``_lipschitz``
    (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete functional, it is recommended to subclass :py:class:`~pycsou.abc.operator.Func` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import Func
    >>> class Median(Func):
    ...    def __init__(self):
    ...        super(Median, self).__init__(shape=(1, None)) # The Median operator is domain-agnostic.
    ...    def apply(self, arr):
    ...        return np.median(arr, axis=-1, keepdims=True)
    >>> med = Median() # creates an instance



    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.ShapeOrDim):
        r"""

        Parameters
        ----------
        shape: int | None | tuple(1, int | None)
            Shape of the functional (1,M). Shapes of the form (1, None) can be used to denote domain-agnostic functionals.
            If a single integer is passed, it is interpreted as the domain dimension M.

        Raises
        ------
        ValueError
            If ``shape`` is specified as a tuple and the first entry of the latter is not 1.
        """
        if not isinstance(shape, tuple):
            shape = (1, int(shape))
        elif shape[0] > 1:
            raise ValueError("Functionals" " must be of the form (1,n).")
        super(Func, self).__init__(shape)

    def as_loss(self, data: typ.Optional[pyct.NDArray] = None) -> "Func":
        """
        Transform a Function into a loss function.

        Parameters
        ----------
        data: NDArray
            (N,) data terms.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Func`
            Loss function.
            If `data = None`, then should return `self`.
        """
        raise NotImplementedError


class ProxFunc(Func, Proximal):
    r"""
    Base class for real-valued proximable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}`.

    A functional :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}` is said *proximable* is its **proximity operator** (see :py:meth:`~pycsou.abc.operator.Proximal.prox` for a definition)
    admits a *simple closed-form expression* **or** can be evaluated *efficiently* and with *high accuracy*.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply` and :py:meth:`~pycsou.abc.operator.Proximal.prox`.
    If the functional is Lipschitz-continuous and the Lipschitz constant is known, the latter can be stored in the private instance attribute ``_lipschitz``
    (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete proximable functional, it is recommended to subclass :py:class:`~pycsou.abc.operator.ProxFunc` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import ProxFunc
    >>> class L1Norm(ProxFunc):
    ...    def __init__(self):
    ...        super(L1Norm, self).__init__(shape=(1, None)) # The L1Norm is domain-agnostic.
    ...        self._lipschitz = 1
    ...    def apply(self, arr):
    ...        return np.linalg.norm(arr, ord=1)
    ...    def prox(self, arr, tau):
    ...        return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)
    >>> l1 = L1Norm() # creates an instance


    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.ShapeOrDim):
        super(ProxFunc, self).__init__(shape)

    def __add__(self: "ProxFunc", other: MapLike) -> MapLike:
        r"""
        Add an instance of :py:class:`~pycsou.abc.operator.ProxFunc` with an instance of a :py:class:`~pycsou.abc.operator.Map` subclass together (overloads the ``+`` operator).

        Parameters
        ----------
        self:  :py:class:`~pycsou.abc.operator.ProxFunc`
            Left addend.
        other: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Right addend.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Sum of ``self`` and ``other``.

        Notes
        -----
        This method is identical to :py:meth:`pycsou.abc.operator.Property.__add__` except when ``other`` is a :py:class:`~pycsou.abc.operator.LinFunc` instance.
        In which case, the sum is a :py:class:`~pycsou.abc.operator.ProxFunc` instance, with proximity operator given by (the pseudocode below is equivalent to but
        does not necessarily reflect the actual implementation):

        .. code-block:: python3

            sum.prox = lambda x, tau: self.prox(x - tau * other.asarray(), tau)

        where ``sum = self + other`` denotes the sum of ``self`` with ``other``, and ``other.asarray()`` returns the vectorial representation of the :py:class:`~pycsou.abc.operator.LinFunc` instance.

        See Also
        --------
        :py:meth:`pycsou.abc.operator.Property.__add__`
        """
        f = Property.__add__(self, other)
        if isinstance(other, LinFunc):
            f = f.specialize(cast_to=self.__class__)
            f.prox = types.MethodType(lambda _, x, tau: self.prox(x - tau * other.asarray(), tau), f)
        return f.squeeze()

    def __mul__(self: "ProxFunc", other: MapLike) -> MapLike:
        r"""
        Scale a :py:class:`~pycsou.abc.operator.ProxFunc` instance or compose it with a :py:class:`~pycsou.abc.operator.Map` subclass instance (overloads the ``*`` operator).

        Parameters
        ----------
        self: :py:class:`~pycsou.abc.operator.ProxFunc`
            Left factor.
        other: numbers.Real | :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Right factor. Should be a real scalar or an instance of :py:class:`~pycsou.abc.operator.Map` subclasses.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
            Product (scaling or composition) of ``self`` with ``other``.

        Notes
        -----
        This method is identical to :py:meth:`pycsou.abc.operator.Property.__mul__` except when:

            * ``self`` and ``other`` are instances of the classes :py:class:`~pycsou.abc.operator.LinFunc` and :py:class:`~pycsou.abc.operator.LinOp` respectively,
            * ``other`` is a scalar or a :py:class:`~pycsou.abc.operator.UnitOp` instance.

        In which cases, the sum is a :py:class:`~pycsou.abc.operator.ProxFunc` instance, with proximity operator given by (the pseudocode below is equivalent to but
        does not necessarily reflect the actual implementation):

        .. code-block:: python3

            if isinstance(self, LinFunc) and isintance(other, LinOp):
                prod.prox = lambda arr, tau: arr - tau * other.adjoint(self.asarray())
            elif isinstance(other, numbers.Real):
                prod.prox = lambda arr, tau: (1 / other) * self.prox(other * arr, tau * (other ** 2))
            elif isinstance(other, UnitOp):
                prod.prox = lambda arr, tau: other.adjoint(self.prox(other.apply(arr), tau))

        where ``prod = self * other`` denotes the product of ``self`` with ``other``.

        See Also
        --------
        :py:meth:`pycsou.abc.operator.Property.__mul__`

        """
        from pycsou.linop.base import HomothetyOp

        f = Property.__mul__(self, other)
        if isinstance(other, pyct.Real):
            other = HomothetyOp(other, dim=self.shape[0])

        if isinstance(self, LinFunc) and isinstance(other, LinOp):
            f.specialize(cast_to=self.__class__)
            f.prox = types.MethodType(lambda _, arr, tau: arr - tau * other.adjoint(self.asarray()), f)
        elif isinstance(other, UnitOp):
            f.specialize(cast_to=self.__class__)
            f.prox = types.MethodType(lambda _, arr, tau: other.adjoint(self.prox(other.apply(arr), tau)), f)
        elif isinstance(other, HomothetyOp):
            f.specialize(cast_to=self.__class__)
            f.prox = types.MethodType(
                lambda _, arr, tau: (1 / other._cst) * self.prox(other._cst * arr, tau * (other._cst) ** 2), f
            )
        return f.squeeze()

    @pycrt.enforce_precision(i="mu", o=False)
    def moreau_envelope(self, mu: pyct.Real) -> "DiffFunc":
        r"""
        Approximate a non-smooth proximable functional by its smooth *Moreau envelope*.

        Parameters
        ----------
        mu: numbers.Real
            Positive regularization parameter (large values yield smoother envelopes).

        Returns
        -------
        :py:class:`~pycsou.abc.operator.DiffFunc`
            Moreau envelope.

        Notes
        -----
        Consider a convex, non-smooth proximable functional :math:`f:\mathbb{R}^M\to\mathbb{R}\cup\{+\infty\}` and a regularization
        parameter :math:`\mu>0`. Then, the :math:`\mu`-*Moreau envelope* (or *Moreau-Yosida envelope*) of :math:`f` is given by

        .. math:: f^\mu(\mathbf{x}) = \min_{\mathbf{z}\in\mathbb{R}^M} \left\{f(\mathbf{z}) \quad+\quad \frac{1}{2\mu}\|\mathbf{x}-\mathbf{z}\|^2\right\}.

        The parameter :math:`\mu` controls a trade-off between the regularity properties of :math:`f^\mu` and the approximation error incurred
        by the Moreau-Yosida regularization. The Moreau envelope inherits the convexity of :math:`f`
        and is gradient Lipschitz (with Lipschitz constant :math:`\mu^{-1}`), even if :math:`f` is non-smooth.
        Its gradient is moreover given by:

        .. math:: \nabla f^\mu(\mathbf{x}) = \mu^{-1} \left(\mathbf{x} - \text{prox}_{\mu f}(\mathbf{x})\right).

        In addition, :math:`f^\mu` envelopes :math:`f` from below: :math:`f^\mu(\mathbf{x})\leq f(\mathbf{x})`. This envelope becomes moreover
        tighter and tighter as :math:`\mu\to 0`:

        .. math:: \lim_{\mu\to 0}f^\mu(\mathbf{x})=f(\mathbf{x}).

        Finally, it can be shown that the minimisers of :math:`f` and :math:`f^\mu` coincide, and that the Fenchel conjugate of
        :math:`f^\mu` is strongly-convex.

        Examples
        --------
        In the example below we construct and plot the Moreau envelope of the :math:`\ell_1`-norm:

        .. plot::

            import numpy as np
            import matplotlib. pyplot as plt
            from pycsou.abc import ProxFunc

            class L1Norm(ProxFunc):
                def __init__(self):
                    super(L1Norm, self).__init__(shape=(1, None))
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
        moreau_envelope = DiffFunc(self.shape)

        @pycrt.enforce_precision(i="arr")
        def env_apply(mu, self, _, arr):
            xp = pycu.get_array_module(arr)
            return (
                self(self.prox(arr, tau=mu))
                + (1 / (2 * mu)) * xp.linalg.norm(arr - self.prox(arr, tau=mu), axis=-1, keepdims=True) ** 2
            )

        @pycrt.enforce_precision(i="arr")
        def env_grad(mu, self, _, arr):
            return (arr - self.prox(arr, tau=mu)) / mu

        moreau_envelope.apply = types.MethodType(ft.partial(env_apply, mu, self), moreau_envelope)
        moreau_envelope.grad = types.MethodType(ft.partial(env_grad, mu, self), moreau_envelope)
        moreau_envelope._diff_lipschitz = 1 / mu
        return moreau_envelope


class DiffFunc(DiffMap, Func, Gradient):
    r"""
    Base class for real-valued differentiable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply` and :py:meth:`~pycsou.abc.operator.Gradient.grad`.
    If the functional and/or its derivative are Lipschitz-continuous and the Lipschitz constants are known, the latter can be stored in the private instance attributes
    ``_lipschitz`` and ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete differentiable functional, it is recommended to subclass :py:class:`~pycsou.abc.operator.DiffFunc` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import DiffFunc
    >>> class LeastSquares(DiffFunc):
    ...    def __init__(self):
    ...        super(LeastSquares, self).__init__(shape=(1, None)) # The LeastSquares functional is domain-agnostic.
    ...        self._diff_lipschitz = 2
    ...    def apply(self, arr):
    ...        return np.linalg.norm(arr, axis=-1, keepdims=True) ** 2
    ...    def grad(self, arr):
    ...        return 2 * arr
    >>> l2 = LeastSquares() # creates an instance


    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.ShapeOrDim):
        super(DiffFunc, self).__init__(shape)


class ProxDiffFunc(ProxFunc, DiffFunc):
    r"""
    Base class for real-valued differentiable *and* proximable functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply`, :py:meth:`~pycsou.abc.operator.Gradient.grad`
    and :py:meth:`~pycsou.abc.operator.Proximal.prox`.
    If the functional and/or its derivative are Lipschitz-continuous and the Lipschitz constants are known, the latter can be stored in the private instance attributes
    ``_lipschitz`` and ``_diff_lipschitz`` (initialized to :math:`+\infty` by default).

    Examples
    --------
    To construct a concrete proximable and differentiable functional, it is recommended to subclass :py:class:`~pycsou.abc.operator.ProxDiffFunc` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import ProxDiffFunc
    >>> class LeastSquares(ProxDiffFunc):
    ...    def __init__(self):
    ...        super(LeastSquares, self).__init__(shape=(1, None)) # The LeastSquares functional is domain-agnostic.
    ...        self._diff_lipschitz = 2
    ...    def apply(self, arr):
    ...        return np.linalg.norm(arr, axis=-1, keepdims=True) ** 2
    ...    def grad(self, arr):
    ...        return 2 * arr
    ...    def prox(self, arr, tau):
    ...        return arr / (1 + 2 * tau)
    >>> l2 = LeastSquares() # creates an instance


    .. Warning::

        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

    """

    def __init__(self, shape: pyct.ShapeOrDim):
        super(ProxDiffFunc, self).__init__(shape)


class LinOp(DiffMap, Adjoint):
    r"""
    Base class for real-valued linear operators :math:`L:\mathbb{R}^M\to\mathbb{R}^N`.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply` and :py:meth:`~pycsou.abc.operator.Adjoint.adjoint`.
    If known, the Lipschitz constant of the linear map can be stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default).

    .. todo::

        Add examples of a concrete linear operator as well as example usage of its main methods.

    """

    def __init__(self, shape: pyct.NonAgnosticShape):
        r"""

        Parameters
        ----------
        shape: tuple(int, int)
           Shape of the linear operator (N,M).

        Notes
        -----
        Since linear maps have constant Jacobians (see :py:meth:`~pycsou.abc.operator.LinOp.jacobian`), the private instance attribute ``_diff_lipschitz`` is initialized to zero by this method.
        """
        super(LinOp, self).__init__(shape)
        self._diff_lipschitz = 0

    def squeeze(self) -> typ.Union["LinOp", "LinFunc"]:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.LinOp` object to the right type (:py:class:`~pycsou.abc.operator.LinFunc` or :py:class:`~pycsou.abc.operator.LinOp`)
        given its co-domain's dimension.

        Returns
        -------
        pycsou.abc.operator.LinOp | pycsou.abc.operator.LinFunc
            Output is a :py:class:`~pycsou.abc.operator.LinFunc` object if ``self.codim==1`` and a :py:class:`~pycsou.abc.operator.LinOp` object otherwise.

        See Also
        --------
        :py:meth:`pycsou.abc.operator.Map.squeeze`, :py:meth:`pycsou.abc.operator.DiffMap.squeeze`
        """
        return self._squeeze(out=LinFunc)

    def jacobian(self, arr: pyct.NDArray) -> "LinOp":
        r"""
        Return the Jacobian of the linear operator at a point ``arr``.

        Parameters
        ----------
        arr: NDArray
            (M,) input. Must be a 1-D array.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Jacobian operator at the requested point ``arr``.

        Notes
        -----
        Let :math:`\mathbf{h}=[h_1, \ldots, h_N]: \mathbb{R}^M\to\mathbb{R}^N` be a differentiable multidimensional map,
        then the *Jacobian* (or *differential*) of :math:`\mathbf{h}` at a given point :math:`\mathbf{z}\in\mathbb{R}^M` is defined as
        the best linear approximation of :math:`\mathbf{h}` near the point :math:`\mathbf{z}`, in the sense that

        .. math:: \mathbf{h}(\mathbf{x}) - \mathbf{h}(\mathbf{z})=\mathbf{J}_{\mathbf {h}}(\mathbf{z})(\mathbf{x} -\mathbf{z})+o(\|\mathbf{x} -\mathbf{z} \|)\quad \text{as} \quad \mathbf {x} \to \mathbf {z}.

        For a linear map :math:`\mathbf{h}`, we have hence trivially: :math:`\mathbf{J}_{\mathbf {h}}(\mathbf{z})=\mathbf{h}, \; \forall \mathbf{z}\in \mathbb{R}^M`,
        i.e. the Jacobian is constant (and hence trivially Lipschitz-continuous with Lipschitz constant 0).
        """
        return self

    @property
    def T(self) -> "LinOp":
        r"""
        Return the adjoint of the linear operator.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Adjoint of the linear operator.
        """
        adj = LinOp(shape=self.shape[::-1])
        adj.apply = self.adjoint
        adj.adjoint = self.apply
        adj._lipschitz = self._lipschitz
        return adj

    def to_sciop(self, dtype: typ.Optional[type] = None, cupyx: bool = False) -> splin.LinearOperator:
        r"""
        Cast a :py:class:`~pycsou.abc.operator.LinOp` instance into a :py:class:`scipy.sparse.linalg.LinearOperator` instance,
        compatible with the matrix-free linear algebra routines of `scipy.sparse.linalg <https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.

        Parameters
        ----------
        dtype: type | None
            Optional data type (i.e. working precision of the linear operator).
        cupyx: bool
            If ``True`` the returned object is of type  :py:class:`cupyx.scipy.sparse.linalg.LinearOperator` which accepts
            CuPy arrays as input (for GPU accelerated computations).

        Returns
        -------
        scipy.sparse.linalg.LinearOperator | cupyx.scipy.sparse.linalg.LinearOperator
            Linear operator object compliant with SciPy's interface.

        Notes
        -----
        This method defines the optional methods ``matmat`` and ``rmatmat`` of the output Scipy's linear operator by evaluating the
        methods :py:meth:`~pycsou.abc.operator.Apply.apply` and :py:meth:`~pycsou.abc.operator.Adjoint.adjoint` on well-chosen N-D arrays
        (see :ref:`developer-notes`). This can be more efficient (and is at the very least equally efficient)
        than the naive parallelization performed by Scipy when the two methods are not provided by the user.
        """

        def matmat(arr: pyct.NDArray) -> pyct.NDArray:
            return self.apply(arr.transpose())

        def rmatmat(arr: pyct.NDArray) -> pyct.NDArray:
            return self.adjoint(arr.transpose())

        if dtype is None:
            dtype = pycrt.getPrecision().value

        if (
            pycu.deps.CUPY_ENABLED and cupyx
        ):  # Scipy casts any input to the LinOp as a Numpy array so the cupyx version is needed.
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = splin
        return spx.LinearOperator(
            shape=self.shape, matvec=self.apply, rmatvec=self.adjoint, matmat=matmat, rmatmat=rmatmat, dtype=dtype
        )

    def lipschitz(self, recompute: bool = False, algo: str = "svds", gpu: bool = False, **kwargs) -> float:
        r"""
        Return the (not necessarily optimal) Lipschitz constant of the operator.

        Parameters
        ----------
        recompute: bool
            If ``True`` forces the recomputation of the Lipschitz constant. If ``False`` the value contained in the private
            instance attribute ``_lipschitz`` is used.
        gpu: bool
            If ``True`` the computation of the Lipschitz constant is performed on the GPU. Has no effect if ``recompute==False``.
        algo: 'svds', 'fro'
            Algorithm used for computing the Lipschitz constant. If ``algo==svds`` the Lipschitz constant is estimated as the
            spectral norm of :math:`L`, computed via Scipy's :py:func:`scipy.sparse.linalg.svds` routine (more accurate but more compute intensive).  If ``algo==fro`` the Lipschitz constant is estimated as the
            Froebenius norm of :math:`L`, computed via the matrix-free `Hutch++ stochastic trace estimation algorithm <https://arxiv.org/abs/2010.09649>`_ (less accurate but less compute intensive).
            Currently, only ``algo==svds`` is supported.
        kwargs:
            Optional arguments to be passed to the algorithm used for computing the Lipschitz constant.
            The parameter ``tol`` of Scipy's :py:func:`scipy.sparse.linalg.svds` routine can be particularly interesting to reduce
            the compute time of the Lipschitz constant.

        Returns
        -------
        float
            Value of the Lipschitz constant.

        Notes
        -----
        The tightest Lipschitz constant is given by the spectral norm of the operator :math:`L`: :math:`\beta=\|L\|_2`.
        It can be computed by means of truncated matrix-free SVD, which can be a compute-intensive task for large-scale
        operators. In which case, it can be advantageous to overestimate the Lipschtiz constant via the Frobenius norm of :math:`L`
        (we have indeed :math:`\|L\|_F \geq \|L\|_2`). The Frobenius norm of  :math:`L` can indeed be approximated efficiently by
        computing the trace of :math:`L^\ast L` (or  :math:`LL^\ast` depending on which is most advantageous) via the
        `Hutch++ stochastic algorithm <https://arxiv.org/abs/2010.09649>`_. Currently, only the SVD-based approach is supported.

        .. todo::
            Add support for trace-based estimate (@Joan).
        """
        if algo == "fro":
            raise NotImplementedError
        if recompute or (self._lipschitz == np.infty):
            kwargs.update(dict(k=1, which="LM", gpu=gpu))
            self._lipschitz = self.svdvals(**kwargs)
        return self._lipschitz

    @pycrt.enforce_precision(o=True)
    def svdvals(self, k: int, which="LM", gpu: bool = False, **kwargs) -> pyct.NDArray:
        r"""
        Compute the ``k`` largest or smallest singular values of the linear operator. The order of the singular values is not guaranteed.

        Parameters
        ----------
        k: int
            Number of singular values to compute. Must be ``1 <= k < min(self.shape)``.
        which: 'LM'|'SM'
            Which k singular values to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude

        gpu: bool
            If ``True`` the singular value decomposition is performed on the GPU.
        kwargs:
            Additional keyword arguments values accepted by Scipy’s function :py:func:`scipy.sparse.linalg.svds`.

        Returns
        -------
        NDArray
            Array containing the ``k`` requested singular values.

        Notes
        -----
        This function calls Scipy’s function: :py:func:`scipy.sparse.linalg.svds`. See the documentation of this function
        for more information on its behaviour and the underlying ARPACK/LOBPCG functions it relies on.
        """
        kwargs.update(dict(k=k, which=which, return_singular_vectors=False))
        SciOp = self.to_sciop(pycrt.getPrecision().value, cupyx=gpu)
        if pycu.deps.CUPY_ENABLED and gpu:
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = splin
        return spx.svds(SciOp, **kwargs)

    def asarray(self, xp: pyct.ArrayModule = np, dtype: typ.Optional[type] = None) -> pyct.NDArray:
        r"""
        Return the matrix representation of the linear operator.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output.
        dtype: type | None
            Optional type of the returned array.

        Returns
        -------
        NDArray
            Matrix representation stored as an array with shape (N,M).

        Notes
        -----
        The matrix representation is computed by evaluating the linear operator on the canonical basis vectors, which can
        be time consuming in large dimensions (and also require a lot of memory). Use this method with caution.
        """
        if dtype is None:
            dtype = pycrt.getPrecision().value
        return self.apply(xp.eye(self.shape[1], dtype=dtype)).transpose()

    def __array__(self, dtype: typ.Optional[type] = None) -> np.ndarray:
        r"""
        Numpy protocol allowing to coerce the linear operator into a :py:class:`numpy.ndarray` object.

        Parameters
        ----------
        dtype: type | None
            Optional ``dtype`` of the array.
        Returns
        -------
        numpy.ndarray
            Matrix representation of the linear operator stored as a Numpy array.

        Notes
        -----
        Functions like ``np.array`` or  ``np.asarray`` will check for the existence of the ``__array__`` protocol to know how to coerce the
        custom object fed as input into an array.
        """
        if dtype is None:
            dtype = pycrt.getPrecision().value
        return self.asarray(xp=np, dtype=dtype)

    def gram(self) -> "LinOp":
        r"""
        Gram operator :math:`L^\ast L:\mathbb{R}^M\to \mathbb{R}^M`.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Gram operator with shape (M,M).

        Notes
        -----
        By default the Gram is computed by the composition ``self.T * self``. This may not be the fastest
        way to compute the Gram operator. If the Gram can be computed more efficiently (e.g. with a convolution), the user should re-define this method.
        """
        return self.T * self

    def cogram(self) -> "LinOp":
        r"""
        Co-Gram operator :math:`LL^\ast:\mathbb{R}^N\to \mathbb{R}^N`.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            Co-Gram operator with shape (N,N).

        Notes
        -----
        By default the co-Gram is computed by the composition ``self * self.T``. This may not be the fastest
        way to compute the co-Gram operator. If the co-Gram can be computed more efficiently (e.g. with a convolution), the user should re-define this method.
        """
        return self * self.T

    @pycrt.enforce_precision(i="arr")
    def pinv(
        self, arr: pyct.NDArray, damp: typ.Optional[float] = None, verbose: typ.Optional[int] = None, **kwargs
    ) -> pyct.NDArray:  # Should we have a decorator that performs trivial vectorization like that for us?
        r"""
        Evaluate the Moore-Penrose pseudo-inverse :math:`L^\dagger` of the linear operator.

        Parameters
        ----------
        arr: NDArray
            Input 1-D array with shape (M,).
        damp: float | None
            Dampening factor for regularizing the pseudo-inverse in case of ill-conditioning.
        verbose: int | None
            Verbosity of the conjugate gradient algorithm used to evaluate the pseudo-inverse. If an integer ``n``, diagnostics
            are printed every ``n`` iterations. If ``None``, the algorithm is silent.
        kwargs:
            Additional keyword arguments values accepted by Scipy’s function :py:func:`scipy.sparse.linalg.cg`.

        Returns
        -------
        NDArray
            Output of the pseudo-inverse evaluated at ``arr``.

        Notes
        -----
        The Moore-Penros pseudo-inverse of an operator :math:`L:\mathbb{R}^N\to \mathbb{R}^M` is defined as the operator
        :math:`L^\dagger:\mathbb{R}^M\to \mathbb{R}^N` verifying the Moore-Penrose conditions:

            1. :math:`LL^\dagger L =L`,
            2. :math:`L^\dagger LL^\dagger =L^\dagger`,
            3. :math:`(L^\dagger L)^\ast=L^\dagger L`,
            4. :math:`(LL^\dagger)^\ast=LL^\dagger`.

        This operator exists and is unique for any finite-dimensional linear operator. The action of the pseudo-inverse
        :math:`L^\dagger \mathbf{y}` for every :math:`\mathbf{y}\in\mathbb{R}^M` can be computed in matrix-free fashion by
        solving the so-called *normal equations*:

        .. math::

            L^\ast L \mathbf{x}= L^\ast \mathbf{y} \quad\Leftrightarrow\quad \mathbf{x}=L^\dagger \mathbf{y}, \quad \forall (\mathbf{x},\mathbf{y})\in\mathbb{R}^N\times\mathbb{R}^M.

        In case of severe ill-conditioning, it is also possible to consider the dampened normal equations for a numerically stabler approximation of :math:`L^\dagger \mathbf{y}`:

        .. math::

            (L^\ast L + \tau I) \mathbf{x}= L^\ast \mathbf{y},

        where :math:`\tau>0` is the ``damp`` parameter from this routine, controlling the amount of regularization (the more the stabler/the less accurate).

        The normal equations can be solved via the conjugate gradient method (:py:func:`scipy.sparse.linalg.cg`) or the `LSQR <https://web.stanford.edu/group/SOL/software/lsqr/>`_ /`LSMR <https://web.stanford.edu/group/SOL/software/lsmr/>`_
        algorithms (see :py:func:`scipy.sparse.linalg.lsqr` and :py:func:`scipy.sparse.linalg.lsmr` respectively). The
        latter may converge faster when the operator is ill-conditioned and/or when there is no fast algorithm for ``self.gram()``
        (i.e. when ``self.gram()`` is trivially evaluated as the composition ``self.T * self``). The GPU implementation of LSQR
        (:py:func:`cupyx.scipy.sparse.linalg.lsqr`) seems however not matrix-free compatible. Currently, only the conjugate gradient
        method is supported.

        .. todo::

            Add support for LSQR/LSMR. **Add support for N-D inputs** (will require re-implementing cg ourselves).

        """
        if arr.ndim == 1:
            return self._pinv(arr=arr, damp=damp, verbose=verbose, **kwargs)
        else:
            xp = pycu.get_array_module(arr)
            pinv1d = lambda x: self._pinv(arr=x, damp=damp, verbose=verbose, **kwargs)
            return xp.apply_along_axis(func1d=pinv1d, arr=arr, axis=-1)

    def _pinv(
        self, arr: pyct.NDArray, damp: typ.Optional[float] = None, verbose: typ.Optional[int] = None, **kwargs
    ) -> pyct.NDArray:
        from pycsou.linop.base import IdentityOp

        b = self.adjoint(arr)
        if damp is not None:
            damp = np.array(damp, dtype=arr.dtype).item()  # cast to correct type
            A = self.gram() + damp * IdentityOp(shape=(self.shape[1], self.shape[1]))
        else:
            A = self.gram()
        if "x0" not in kwargs:
            kwargs["x0"] = 0 * arr
        if "atol" not in kwargs:
            kwargs["atol"] = 1e-16
        if verbose is not None:

            class CallBack:
                def __init__(self, verbose: int, A: LinOp, b: pyct.NDArray):
                    self.verbose = verbose
                    self.n = 0
                    self.A, self.b = A, b

                def __call__(self, x: pyct.NDArray):
                    if self.n % self.verbose == 0:
                        xp = pycu.get_array_module(x)
                        print(
                            f"Iteration: {self.n}, Relative residual norm:{xp.linalg.norm(self.b - self.A(x)) / xp.linalg.norm(self.b)}"
                        )
                        self.n += 1

            kwargs.update(dict(callback=CallBack(verbose, A, b)))

        xp = pycu.get_array_module(arr)
        if xp is np:
            spx = splin
        elif pycu.deps.CUPY_ENABLED and (xp is cp):
            import cupyx.scipy.sparse.linalg as spx
        else:
            raise NotImplementedError
        return spx.cg(A, b, **kwargs)[0]

    def dagger(self, damp: typ.Optional[float] = None, **kwargs) -> "LinOp":
        r"""
        Return the Moore-Penrose pseudo-inverse :math:`L^\dagger` as a :py:class:`~pycsou.abc.operator.LinOp` instance.

        Parameters
        ----------
        damp: float | None
            Dampening factor for regularizing the pseudo-inverse in case of ill-conditioning.
        kwargs:
            Additional keyword arguments values accepted by Scipy’s function :py:func:`scipy.sparse.linalg.cg`.

        Returns
        -------
        :py:class:`~pycsou.abc.operator.LinOp`
            The Moore-Penrose pseudo-inverse operator.
        """
        dagger = LinOp(self.shape[::-1])
        dagger.apply = types.MethodType(
            ft.partial(lambda damp, kwargs, _, x: self.pinv(x, damp, **kwargs), damp, kwargs), dagger
        )
        dagger.adjoint = types.MethodType(
            ft.partial(lambda damp, kwargs, _, x: self.T.pinv(x, damp, **kwargs), damp, kwargs), dagger
        )
        return dagger

    @classmethod
    def from_sciop(cls, scipy_operator) -> "LinOp":
        r"""
        Cast a :py:class:`scipy.sparse.linalg.LinearOperator` instance into a :py:class:`~pycsou.abc.operator.LinOp`
        instance.

        Parameters
        ----------
        scipy_operator:  scipy.sparse.linalg.LinearOperator | cupyx.scipy.sparse.linalg.LinearOperator
            Linear operator object compliant with SciPy's interface.

        Returns
        -------
        pycsou.abc.operator.LinOp
           Output is a :py:class:`~pycsou.abc.operator.LinOp` object.

        Examples
        --------
        >>> import numpy as np
        >>> from pycsou.abc.operator import LinOp
        >>> from scipy.sparse.linalg import aslinearoperator
        >>> mat = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> sciop = aslinearoperator(mat)
        >>> linop = LinOp.from_sciop(sciop)
        >>> x = np.arange(15, dtype=np.float32).reshape(5, 3)
        >>> np.allclose(linop(x), sciop.matmat(x.T).T) # transpose to satisfy numpy n-dim convention
        True


        .. Warning::
        This  is a simplified example for illustration puposes only. It may not abide by all the rules listed in the
        :ref:`developer-notes`.

        Notes
        -----
        Scipy linear operators by default only accept 2D inputs only. In order to use N-D arrays,
        :py:meth:`~pycsou.abc.operator.Map.from_sciop` reshapes the first N-1 stacking dimensions of the input array
        into a single stacking dimension. The output shape is matched to the input's stacking dimensions.

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`, :py:meth:`~pycsou.abc.operator.Map.from_source` and
        :py:meth:`~pycsou.abc.operator.LinOp.to_sciop`.
        """

        if scipy_operator.dtype not in [elem.value for elem in pycrt.Width]:
            warnings.warn("Computation may not be performed at the requested precision.", UserWarning)

        @pycrt.enforce_precision(i="arr")
        def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
            out_shape = arr.shape[:-1] + (-1,) if arr.ndim > 1 else (-1,)
            return scipy_operator.matmat(arr.reshape(-1, arr.shape[-1]).transpose()).transpose().reshape(*out_shape)

        @pycrt.enforce_precision(i="arr")
        def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
            out_shape = arr.shape[:-1] + (-1,) if arr.ndim > 1 else (-1,)
            return scipy_operator.rmatmat(arr.reshape(-1, arr.shape[-1]).transpose()).transpose().reshape(*out_shape)

        out_op = cls(scipy_operator.shape)
        setattr(out_op, "apply", types.MethodType(apply, out_op))
        setattr(out_op, "adjoint", types.MethodType(adjoint, out_op))
        return out_op

    @classmethod
    def from_array(cls, mat: typ.Union[pyct.NDArray, pyct.SparseArray], enable_warnings: bool = True) -> "LinOp":
        r"""
        Create an instance of a :py:class:`~pycsou.abc.operator.LinOp` from its matrix representation (see
        :py:class:`pycsou.linop.base.ExplicitLinOp`).

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_sciop`, :py:meth:`~pycsou.abc.operator.Map.from_source`,
        :py:class:`pycsou.linop.base.ExplicitLinOp`, :py:meth:`~pycsou.abc.operator.LinFunc.from_array`.

        """
        from pycsou.linop.base import ExplicitLinOp

        return ExplicitLinOp(mat, enable_warnings)


class LinFunc(ProxDiffFunc, LinOp):
    r"""
    Base class for real-valued linear functionals :math:`f:\mathbb{R}^M\to\mathbb{R}`.

    Any instance/subclass of this class must implement the methods :py:meth:`~pycsou.abc.operator.Apply.apply`, :py:meth:`~pycsou.abc.operator.Gradient.grad`,
    :py:meth:`~pycsou.abc.operator.Proximal.prox` and :py:meth:`~pycsou.abc.operator.Adjoint.adjoint`.
    The Lipschitz constant of the linear functional can be stored in the private instance attribute
    ``_lipschitz`` (initialized to :math:`+\infty` by default). The Lipschitz constant of the gradient is 0, since the latter
    is constant for a linear functional.

    Examples
    --------
    To construct a concrete linear functional, it is recommended to subclass :py:class:`~pycsou.abc.operator.LinFunc` as ilustrated
    in the following example:

    >>> import numpy as np
    >>> from pycsou.abc import LinFunc
    >>> class Sum(LinFunc):
    ...     def __init__(self, dim):
    ...         super(Sum, self).__init__(shape=(1, dim))
    ...         self._lipschitz = np.sqrt(dim)
    ...         self._diff_lipschitz = 0
    ...     def apply(self, arr):
    ...        return np.sum(arr, axis=-1, keepdims=True)
    ...     def adjoint(self, arr):
    ...        return arr * np.ones(self.shape[-1])
    ...     def grad(self, arr):
    ...        return np.ones(shape=arr.shape[:-1] + (self.dim,))
    ...     def prox(self, arr, tau):
    ...        return arr - tau * np.ones(self.shape[-1])

    >>> sum = Sum(10)

    It is also possible to use the class :py:class:`~pycsou.linop.base.ExplicitLinFunc`, which constructs a linear functional
    through its vectorial representation (i.e. :math:`f(\mathbf{x})=\langle\mathbf{x}, \mathbf{v}\rangle`):

    >>> from pycsou.linop.base import ExplicitLinFunc
    >>> sum = ExplicitLinFunc(vec=np.ones(10)) # Creates a LinFunc instance

    """

    def __init__(self, shape: pyct.ShapeOrDim):
        ProxDiffFunc.__init__(self, shape)
        LinOp.__init__(self, shape)

    __init__.__doc__ = DiffFunc.__init__.__doc__

    @classmethod
    def from_array(cls, vec: pyct.NDArray, enable_warnings: bool = True) -> "LinFunc":
        r"""
        Create an instance of a :py:class:`~pycsou.abc.operator.LinFunc` from its vectorial representation (see
        :py:class:`pycsou.linop.base.ExplicitLinFunc`).

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.from_array`, :py:meth:`~pycsou.abc.operator.Map.from_source`,
        :py:class:`pycsou.linop.base.ExplicitLinFunc`.

        """
        from pycsou.linop.base import ExplicitLinFunc

        return ExplicitLinFunc(vec, enable_warnings)


class SquareOp(LinOp):
    r"""
    Base class for *square* linear operators :math:`L:\mathbb{R}^N\to \mathbb{R}^N` (endomorphsisms).
    """

    def __init__(self, shape: pyct.SquareShape):
        r"""

        Parameters
        ----------
        shape: int | tuple(int, int)
            Shape of the operator.
        """
        if not isinstance(shape, tuple):
            shape = (int(shape), int(shape))
        elif shape[0] != shape[1]:
            raise ValueError(f"Inconsistent shape {shape} for operator of type {SquareOp}")
        super(SquareOp, self).__init__(shape=(shape[0], shape[0]))


class NormalOp(SquareOp):
    r"""
    Base class for *normal* operators.

    Notes
    -----
    Normal operators commute with their adjoint, i.e. :math:`LL^\ast=L^\ast L`. It is `possible to show <https://www.wikiwand.com/en/Spectral_theorem#/Normal_matrices>`_
    that an operator is normal iff it is *unitarily diagonalizable* :math:`L=UDU^\ast`.
    """

    @pycrt.enforce_precision(o=True)
    def eigvals(self, k: int, which="LM", gpu: bool = False, **kwargs) -> pyct.NDArray:
        r"""
        Find ``k`` eigenvalues of a normal operator.

        Parameters
        ----------
        k: int
            The number of eigenvalues and eigenvectors desired. ``k`` must be strictly smaller than the dimension of the operator.
            It is not possible to compute all eigenvectors of an operator.
        which: str, [‘LM’ | ‘SM’ | ‘LR’ | ‘SR’ | ‘LI’ | ‘SI’]
            Which ``k`` eigenvalues to find:

                * ‘LM’ : largest magnitude
                * ‘SM’ : smallest magnitude
                * ‘LR’ : largest real part
                * ‘SR’ : smallest real part
                * ‘LI’ : largest imaginary part
                * ‘SI’ : smallest imaginary part

        gpu: bool
            If ``True`` the singular value decomposition is performed on the GPU.
        kwargs: dict
            A dict of additional keyword arguments values accepted by Scipy's functions :py:func:`scipy.sparse.linalg.eigs`.

        Returns
        -------
        NDArray
            Array containing the ``k`` requested eigenvalues.

        Notes
        -----
        This function calls Scipy's function :py:func:`scipy.sparse.linalg.eigs`.
        See the documentation of this function for more information on its behaviour and the underlying ARPACK function it relies on.

        See Also
        --------
        :py:meth:`~pycsou.abc.operator.LinOp.svds`
        """
        kwargs.update(dict(k=k, which=which, return_eigenvectors=False))
        if pycu.deps.CUPY_ENABLED and gpu:
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = splin
        return spx.eigs(self.to_sciop(pycrt.getPrecision().value, cupyx=gpu), **kwargs)

    def cogram(self) -> "NormalOp":
        r"""
        Call the method ``self.gram()`` since the two are equivalent for normal operators.
        """
        return self.gram().specialize(cast_to=SelfAdjointOp)


class SelfAdjointOp(NormalOp):
    r"""
    Base class for *self-adjoint* operators :math:`L^\ast=L`.

    Self-adjoint operators need not implement the ``adjoint`` method.
    """

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Call the method ``self.apply(arr)`` since the two are equivalent for self-adjoint operators.
        """
        return self.apply(arr)

    @property
    def T(self) -> "SelfAdjointOp":
        r"""Return ``self``."""
        return self

    @pycrt.enforce_precision(o=True)
    def eigvals(self, k: int, which="LM", gpu: bool = False, **kwargs) -> pyct.NDArray:
        r"""
        Same as :py:meth:`~pycsou.abc.operator.NormalOp.eigvals`, but calls :py:func:`scipy.sparse.linalg.eigsh`
        in the background to leverage the self-adjointness and speed up the computation.
        """
        kwargs.update(dict(k=k, which=which, return_eigenvectors=False))
        if pycu.deps.CUPY_ENABLED and gpu:
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = splin
        return spx.eigsh(self.to_sciop(pycrt.getPrecision().value, cupyx=gpu), **kwargs)


class UnitOp(NormalOp):
    r"""
    Base class for *unitary* operators :math:`LL^\ast=L^\ast L =I`.
    """

    def __init__(self, shape: pyct.SquareShape):
        super(UnitOp, self).__init__(shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> float:
        r"""
        Always return 1, since the spectral norm of unitary operators is equal to 1.
        """
        return self._lipschitz

    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        r"""Call ``self.adjoint(arr)``, since the inverse (and hence pseudo-inverse) of a unitary operator is given by its adjoint."""
        return self.adjoint(arr)

    def dagger(self, **kwargs) -> "UnitOp":
        r"""Return ``self.T``, since the inverse (and hence pseudo-inverse) of a unitary operator is given by its adjoint."""
        return self.T


class ProjOp(SquareOp):
    r"""
    Base class for *projection* operators.

    Projection operators are *idempotent*, i.e. :math:`L^2=L`.
    """

    def __pow__(self, power: int) -> typ.Union["ProjOp", "UnitOp"]:
        r"""
        For ``power>0`` just return ``self`` as projection operators are idempotent.
        """
        if power == 0:
            from pycsou.linop.base import IdentityOp

            return IdentityOp(self.shape)
        else:
            return self


class OrthProjOp(ProjOp, SelfAdjointOp):
    r"""
    Base class for *orthogonal projection* operators.

    Orthogonal projection operators are *idempotent* and *self-adjoint*, i.e. :math:`L^2=L` and :math:`L^\ast=L`.
    """

    def __init__(self, shape: pyct.SquareShape):
        super(OrthProjOp, self).__init__(shape)
        self._lipschitz = 1

    def lipschitz(self, **kwargs) -> float:
        r"""
        Always return 1, since the spectral norm of orthogonal projection operators is equal to 1.
        """
        return self._lipschitz

    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        r"""
        Return ``self.apply(arr)`` since the pseudo-inverse of an orthogonal projection operator is itself:

            1. :math:`LLL=L`,
            2. :math:`(LL)^\ast=L^\ast L^\ast=LL`.

        """
        return self.apply(arr)

    def dagger(self, **kwargs) -> "OrthProjOp":
        r"""
        Return ``self`` since the pseudo-inverse of an orthogonal projection operator is itself.
        """
        return self


class PosDefOp(SelfAdjointOp):
    r"""
    Base class for *positive-definite* operators.
    """
    pass


_base_operators = frozenset([Map, DiffMap, Func, DiffFunc, ProxFunc, ProxDiffFunc, LinOp, LinFunc])
