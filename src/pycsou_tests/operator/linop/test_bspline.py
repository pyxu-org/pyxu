import itertools

import numpy as np
import pytest
import scipy.interpolate as sci

import pycsou.operator.linop as pycl
import pycsou.operator.linop.bspline as bsp
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class TestBSplineSampling(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            (np.array([0, 1.1, 2.1, 3]),),  # Evaluation grid
            (np.array([0, 0, 0, 0, 1.1, 1.9, 3, 3, 3, 3]),),  # Knots
            ([2, 3],),  # Degrees
            ([0, 1],),  # Differentiation orders
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        eval_grid, knots, degrees, deriv_orders, ndi, width = _spec
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = bsp.BSplineSampling(eval_grid, knots, degrees, deriv_orders, array_module=ndi)
        return op, ndi, width

    @pytest.fixture
    def dim(self, _spec):
        return (len(_spec[1]) - _spec[2][0] - 1) * (len(_spec[1]) - _spec[2][1] - 1)

    @pytest.fixture
    def codim(self, _spec):
        return len(_spec[0]) ** 2

    @pytest.fixture
    def data_shape(self, dim, codim):
        return codim, dim

    @pytest.fixture
    def data_apply(self, _spec, dim):
        arr = self._random_array(dim)
        tck = (_spec[1], _spec[1], arr, _spec[2][0], _spec[2][1])
        out = sci.bisplev(x=_spec[0], y=_spec[0], tck=tck, dx=_spec[3][0], dy=_spec[3][1]).ravel()
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


class TestPiecewiseCstInnos1D(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            (4,),  # dim
            (
                "not-a-knot",
                "periodic",
                "zero",
            ),  # Boundary conditions
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        op = bsp._PiecewiseCstInnos1D(_spec[0], _spec[1])
        return op, _spec[2], _spec[3]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0]

    @pytest.fixture
    def bc_type(self, _spec):
        return _spec[1]

    @pytest.fixture
    def codim(self, dim, bc_type):
        if bc_type == "not-a-knot":
            codim = dim - 1
        elif bc_type == "periodic":
            codim = dim
        elif bc_type == "zero":
            codim = dim + 1
        return codim

    @pytest.fixture
    def data_shape(self, dim, codim):
        return codim, dim

    @pytest.fixture
    def data_apply(self, dim, bc_type):
        arr = np.zeros(dim)
        arr[1] = 1
        out = np.zeros(dim)
        out[1] = 1
        out[2] = -1
        if bc_type == "not-a-knot":
            out = np.delete(out, 0)
        elif bc_type == "periodic":
            pass
        elif bc_type == "zero":
            out = np.append(out, 0)

        return dict(
            in_=dict(arr=arr),
            out=out,
        )


# Below is a test of the multidimensional-dimensional implementation PiecewiseCstInnos. We have chosen to support only
# the 1D case since the multidimensional case is not well understood, but we leave it here for future reference.
#
# class TestPiecewiseCstInnos(conftest.LinOpT):
#     @pytest.fixture(
#         params=itertools.product(
#             ((4, 5),),  # shape_in
#             (
#                 "not-a-knot",
#                 "periodic",
#                 "zero",
#             ),  # Boundary conditions
#             pycd.NDArrayInfo,
#             pycrt.Width,
#         )
#     )
#     def _spec(self, request):
#         return request.param
#
#     @pytest.fixture
#     def spec(self, _spec):
#         op = bsp._PiecewiseCstInnos(_spec[0], _spec[1])
#         return op, _spec[2], _spec[3]
#
#     @pytest.fixture
#     def shape_in(self, _spec):
#         return _spec[0]
#
#     @pytest.fixture
#     def bc_types(self, _spec):
#         bc_list, _ = bsp._convert_to_list(_spec[1], ndim=len(_spec[0]))
#         return bc_list
#
#     @pytest.fixture
#     def shape_out(self, shape_in, bc_types):
#         shape_out = ()
#         for i in range(len(shape_in)):
#             bc = bc_types[i]
#             if bc == "not-a-knot":
#                 shape_out = (*shape_out, shape_in[i] - 1)
#             elif bc == "periodic":
#                 shape_out = (*shape_out, shape_in[i])
#             elif bc == "zero":
#                 shape_out = (*shape_out, shape_in[i] + 1)
#         return shape_out
#
#     @pytest.fixture
#     def data_shape(self, shape_in, shape_out):
#         return np.prod(shape_out), np.prod(shape_in)
#
#     @pytest.fixture
#     def data_apply(self, shape_in, bc_types):
#         ndim = len(shape_in)
#         arr = np.zeros(shape_in)
#         arr[1, 1] = 1
#         out = np.zeros(shape_in)
#         out[1, 1] = 1
#         out[2, 1] = -1
#         out[1, 2] = -1
#         out[2, 2] = 1
#         for i in range(ndim):
#             if bc_types[i] == "not-a-knot":
#                 out = np.delete(out, 0, axis=i)
#             elif bc_types[i] == "periodic":
#                 out = out
#             elif bc_types[i] == "zero":
#                 sh = list(out.shape)
#                 sh[i] = 1
#                 out = np.append(out, np.zeros(sh), axis=i)
#
#         return dict(
#             in_=dict(arr=arr.ravel()),
#             out=out.ravel(),
#         )


class BSplinePeriodicIndFunc(conftest.ProxFuncT):
    disable_test = frozenset(
        conftest.ProxFuncT.disable_test
        | {
            "test_math_lipschitz",
        }
    )

    @pytest.fixture(
        params=itertools.product(
            (np.arange(8),),  # knots
            (3,),  # degrees
            (["not-a-knot", "periodic"],),  # bc_types
            (2,),  # ndim
            (1,),  # precond_weights
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        func = bsp.BSplinePeriodicIndFunc(_spec[0], _spec[1], _spec[2], _spec[3], _spec[4])
        return func, _spec[5], _spec[6]

    @pytest.fixture
    def ndim(self, _spec):
        return _spec[3]

    @pytest.fixture
    def knots(self, _spec, ndim):
        knots, _ = bsp._convert_to_list(_spec[0], ndim=ndim)
        return knots

    @pytest.fixture
    def degrees(self, _spec, ndim):
        degrees, _ = bsp._convert_to_list(_spec[1], ndim=ndim)
        return degrees

    @pytest.fixture
    def bc_types(self, _spec, ndim):
        bc_types, _ = bsp._convert_to_list(_spec[2], ndim=ndim)
        return bc_types

    @pytest.fixture
    def precond_weights(self, _spec, ndim):
        precond_weights, _ = bsp._convert_to_list(_spec[4], ndim=ndim)
        return precond_weights

    @pytest.fixture
    def shape_in(self, knots, degrees):
        sh = ()
        for i in range(len(knots)):
            sh = (*sh, knots[i].size - degrees[i] - 1)
        return sh

    @pytest.fixture
    def constraints(self, knots, degrees, bc_types, precond_weights, shape_in, ndim):
        A_list, P_list = [], []
        for i in range(ndim):
            # Only periodic boundary conditions leads to constraints for the corresponding dimension
            if bc_types[i] == "periodic":
                A, P = bsp._periodic_constraints(knots=knots[i], degree=degrees[i], precond_weights=precond_weights[i])
                A_list.append(A)
                P_list.append(P)
            else:
                # If non-periodic, there are no constraints and projector is identity
                A_list.append(pycl.NullOp(shape=(0, shape_in[-1])))
                P_list.append(pycl.IdentityOp(dim=shape_in[-1]))
        return A_list, P_list

    @pytest.fixture
    def data_shape(self, shape_in):
        return 1, np.prod(shape_in)

    @pytest.fixture
    def data_apply(self, shape_in):
        # Constant array along the periodic dimension -> periodic constraints are satisfied
        arr = np.ones(shape_in) * np.arange(shape_in[0]).reshape((shape_in[0], 1))
        return dict(
            in_=dict(arr=arr.ravel()),
            out=np.zeros((1,)),
        )

    @pytest.fixture
    def data_prox(self, shape_in, constraints, ndim):
        # Periodic constraints are satisfied -> prox is identity
        arr = np.ones(shape_in) * np.arange(shape_in[0]).reshape((shape_in[0], 1))
        return dict(
            in_=dict(arr=arr.ravel(), tau=1),
            out=arr.ravel(),
        )

    @pytest.fixture
    def data_math_lipschitz(self):
        pass


class BSplineInnos1D(conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            (np.arange(10)),  # Knots
            (2,),  # Degree
            (
                "not-a-knot",
                "periodic",
                "zero",
            ),  # Boundary conditions
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        knots, degree, bc_type, ndi, width = _spec
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = bsp.BSplineInnos1D(knots, degree, bc_type, array_module=ndi)
        return op, ndi, width

    @pytest.fixture
    def dim(self, _spec):
        return len(_spec[0]) - _spec[1] - 1

    @pytest.fixture
    def codim(self, _spec):
        knots, degree, bc_type, ndi, width = _spec
        n = len(knots) - 2 * degree - 1  # Number of spline coefficients of the degree-th derivative
        if bc_type == "not-a-knot":
            return n - 1
        elif bc_type == "periodic":
            return n
        elif bc_type == "zero":
            return n + 1
        return

    @pytest.fixture
    def data_shape(self, dim, codim):
        return codim, dim

    @pytest.fixture
    def data_apply(self, _spec, dim):
        knots, degree, bc_type, ndi, width = _spec
        arr = np.zeros(7)
        arr[3] = 1  # A single centered B-spline
        out = np.array([0, 1, -3, 3, -1])  # Innovation of a B-spline
        if bc_type == "not-a-knot":
            out = np.delete(out, 0)
        elif bc_type == "periodic":
            pass
        elif bc_type == "zero":
            out = np.append(out, 0)

        return dict(
            in_=dict(arr=arr),
            out=out,
        )
