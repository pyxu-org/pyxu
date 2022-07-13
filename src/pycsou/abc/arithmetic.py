import pycsou.abc.operator as pyco
import pycsou.util.ptype as pyct


def add(lhs: pyct.OpT, rhs: pyct.OpT) -> pyct.OpT:
    pass


def scale(op: pyct.OpT, cst: pyct.Real) -> pyct.OpT:
    pass


def compose(lhs: pyct.OpT, rhs: pyct.OpT) -> pyct.OpT:
    pass


def pow(op: pyct.OpT, k: pyct.Integer) -> pyct.OpT:
    # check square
    pass


def argscale(op: pyct.OpT, cst: pyct.Real) -> pyct.OpT:
    pass


def argshift(op: pyct.OpT, cst: pyct.NDArray) -> pyct.OpT:
    pass
