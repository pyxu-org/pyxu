import copy
import functools as ft
import types
import typing as typ

import pycsou.abc.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


def add(lhs: pyct.MapT, rhs: pyct.MapT) -> pyct.MapT:
    raise NotImplementedError


def sub(lhs: pyct.MapT, rhs: pyct.MapT) -> pyct.MapT:
    raise NotImplementedError


def neg(op: pyct.MapT) -> pyct.MapT:
    raise NotImplementedError


def mul(lhs: pyct.MapT, rhs: typ.Union[pyct.MapT, pyct.Real]) -> pyct.MapT:
    raise NotImplementedError


def pow(lhs: pyct.MapT, k: pyct.Integer) -> pyct.MapT:
    raise NotImplementedError


def argscale(op: pyct.MapT, scale: pyct.Real) -> pyct.MapT:
    if isinstance(scalar, pyct.Real):
        from pycsou.operator.linop.base import HomothetyOp

        # If op's dim is agnostic (None), then operator arithmetic with a Homothety will fail.
        # Trick: since (op * Homothety).shape == op.shape, faking the Homothety's dim is OK.
        return mul(
            op,
            HomothetyOp(
                cst=scalar,
                dim=1 if (op.dim is None) else op.dim,
            ),
        )
    else:
        raise NotImplementedError


def argshift(op: pyct.MapT, shift: pyct.NDArray) -> pyct.MapT:
    try:
        pycu.get_array_module(shift)
    except:
        raise ValueError(f"shift: expected NDArray, got {type(shift)}.")

    if shift.ndim != 1:
        raise ValueError("Lag must be 1D.")

    if (op.dim is None) or (op.dim == shift.size):
        sh_out = (op.codim, shift.size)
    else:
        raise ValueError(f"Invalid lag shape: {shift.size} != {op.dim}.")

    if isinstance(op, pyco.LinFunc):  # Shifting a linear map makes it an affine map.
        op_out = pyco.DiffFunc(shape=sh_out)
    elif isinstance(op, pyco.LinOp):  # Shifting a linear map makes it an affine map.
        op_out = pyco.DiffMap(shape=sh_out)
    else:
        op_out = copy.copy(op)

    props = op_out.properties()
    if op_out == pyco.DiffFunc:
        props.discard("jacobian")
    props.discard("single_valued")

    for prop in op_out.properties():
        if prop in {"lipschitz", "diff_lipschitz"}:
            setattr(op_out, "_" + prop, getattr(op, "_" + prop))
        elif prop == "prox":

            @pycrt.enforce_precision(i=("arr", "tau"))
            def argshifted_prox(shift, _, arr, tau):
                shift = shift.astype(arr.dtype, copy=False)
                return op.prox(arr + shift, tau) - shift

            method = types.MethodType(ft.partial(argshifted_prox, shift), op_out)
            setattr(op_out, "prox", method)
        else:

            def argshifted_method(prop, shift, _, arr):
                shift = shift.astype(arr.dtype, copy=False)
                return getattr(op, prop)(arr + shift)

            method = types.MethodType(ft.partial(argshifted_method, prop, shift), op_out)
            setattr(op_out, prop, method)

    return op_out.squeeze()
