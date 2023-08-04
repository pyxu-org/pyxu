import itertools

import numpy as np
import pytest

import pycsou.operator.diffusion as pycdiffusion
import pycsou.operator.linop.diff as pycdiff
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class SquareMapMixin(conftest.MapT):  # Base class to test balloon forces and diffusivities
    disable_test = frozenset(
        conftest.MapT.disable_test
        | {
            "test_math_lipschitz",
            "test_valueND_apply",
            "test_valueND_call",
        }
    )

    @pytest.fixture
    def map_klass(self):
        raise NotImplementedError

    @pytest.fixture
    def map_kwargs(self):
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self):
        raise NotImplementedError

    @pytest.fixture
    def data_math_lipschitz(self):  # Not tested
        pass

    @pytest.fixture(scope="session")
    def arg_shape(self):
        return 5, 6

    @pytest.fixture
    def dim(self, arg_shape):
        return np.prod(arg_shape)

    @pytest.fixture
    def data_shape(self, dim):
        return dim, dim

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def ndi(self, _spec):
        return _spec[0]

    @pytest.fixture
    def width(self, _spec):
        return _spec[1]

    @pytest.fixture
    def spec(self, arg_shape, map_klass, map_kwargs, ndi, width):
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = map_klass(arg_shape=arg_shape, **map_kwargs)
        return op, ndi, width


class TestDilation(SquareMapMixin):
    @pytest.fixture
    def map_klass(self):
        return pycdiffusion.Dilation

    @pytest.fixture(scope="session")
    def map_kwargs(self, arg_shape):
        return dict(gradient=pycdiff.Gradient(arg_shape=arg_shape))

    @pytest.fixture
    def data_apply(self, spec, arg_shape):  # TODO tests fail at the moment (output shape is weird)
        arr = self._random_array(arg_shape)  # TODO Replace with input-output pairs computed manually
        out = spec[0](arr.ravel())
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


# TODO Erosion, Mfi balloon term


class TestTikhonovDiffusivity(SquareMapMixin):
    @pytest.fixture
    def map_klass(self):
        return pycdiffusion.TikhonovDiffusivity

    @pytest.fixture
    def map_kwargs(self):
        return {}

    @pytest.fixture
    def data_apply(self, arg_shape):
        arr = self._random_array(arg_shape)
        out = np.ones(arg_shape)
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


class TestMfiDiffusivity(SquareMapMixin):
    @pytest.fixture
    def map_klass(self):
        return pycdiffusion.MfiDiffusivity

    @pytest.fixture(
        params=itertools.product(
            (2.0,),  # beta
            (True, False),  # tame
        )
    )
    def map_kwargs(self, request):
        return dict(beta=request.param[0], tame=request.param[1])

    @pytest.fixture
    def data_apply(self, arg_shape, map_kwargs):
        arr = np.ones(arg_shape) * 2

        if map_kwargs["tame"]:
            out = 1 / (1 + arr / map_kwargs["beta"])
        else:
            out = 1 / (arr / np.mean(arr) / map_kwargs["beta"])

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


# TODO: Perona-Malik, TV


class DiffusionCoeffMixin(conftest.LinOpT):
    @pytest.fixture
    def diffusion_coeff_klass(self):
        raise NotImplementedError

    @pytest.fixture
    def diffusion_coeff_kwargs(self):
        raise NotImplementedError

    @pytest.fixture
    def data_shape(self):  # Shape of the output LinOp of the apply() method
        raise NotImplementedError

    @pytest.fixture
    def arg_shape(self):  # Shape of the input argument of the apply() method
        return 2, 3

    @pytest.fixture
    def input_array(self, arg_shape, ndi):
        xp = ndi.module()
        return xp.arange(np.prod(arg_shape))

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def ndi(self, _spec):
        return _spec[0]

    @pytest.fixture
    def width(self, _spec):
        return _spec[1]

    @pytest.fixture
    def spec(self, diffusion_coeff_klass, diffusion_coeff_kwargs, input_array, ndi, width):
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = diffusion_coeff_klass(**diffusion_coeff_kwargs).apply(input_array)
        return op, ndi, width


class TestDiffusionCoeffIsotropic(DiffusionCoeffMixin):
    @pytest.fixture
    def diffusion_coeff_klass(self):
        return pycdiffusion.DiffusionCoeffIsotropic

    @pytest.fixture(
        params=itertools.product(
            (pycdiffusion.TikhonovDiffusivity,),  # diffusivity
            (True, False),  # trace_term
        )
    )
    def diffusion_coeff_kwargs(self, arg_shape, diffusivity, trace_term):
        return dict(arg_shape=arg_shape, diffusivity=diffusivity(arg_shape), trace_term=trace_term)

    @pytest.fixture
    def data_shape(self, arg_shape, trace_term):  # Shape of the output LinOp of the apply() method
        dim = np.prod(arg_shape)
        if trace_term:
            return dim, len(arg_shape) * dim
        else:
            return len(arg_shape) * dim, len(arg_shape) * dim

    @pytest.fixture
    def data_apply(self, spec, data_shape):
        arr = self._random_array(data_shape[1])  # TODO Replace with input-output pairs computed manually
        out = spec[0](arr)
        return dict(
            in_=dict(arr=arr),
            out=out,
        )


# TODO: DiffusionCoeffAnisotropic, ...


class DiffusionOpMixin(conftest.DiffFuncT):
    # Change to conftest.ProxDiffFunc to test prox (and uncomment data_prox fixture)
    disable_test = frozenset(
        conftest.MapT.disable_test
        | {
            "test_math_lipschitz",
            "test_valueND_apply",
            "test_valueND_call",
            "test_interface_asloss",
        }
    )

    @pytest.fixture
    def diffusion_op_klass(self):
        raise NotImplementedError

    @pytest.fixture
    def diffusion_op_kwargs(self):
        raise NotImplementedError

    @pytest.fixture
    def data_apply(self):  # If no apply method, then use DiffusionOpNoApplyMixin subclass
        raise NotImplementedError

    # @pytest.fixture
    # def data_prox(self):
    #     raise NotImplementedError

    @pytest.fixture
    def data_grad(self):
        raise NotImplementedError

    @pytest.fixture
    def data_math_lipschitz(self):  # Not tested
        pass

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 2
        return self._random_array((N_test, dim), seed=0)

    @pytest.fixture(scope="session")
    def arg_shape(self):
        return 2, 3

    @pytest.fixture(scope="session", params=[1])  # TODO tests currently fail with multiple channels (params=[1, 2])
    def nchannels(self, request):  # TODO remove if all subclasses do not have this argument (as is currently the case)
        return request.param

    @pytest.fixture
    def dim(self, arg_shape, nchannels):
        return np.prod(arg_shape) * nchannels

    @pytest.fixture
    def data_shape(self, dim):
        return 1, dim

    @pytest.fixture(
        scope="session",
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        ),
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(scope="session")
    def ndi(self, _spec):
        return _spec[0]

    @pytest.fixture(scope="session")
    def width(self, _spec):
        return _spec[1]

    @pytest.fixture(scope="session")  # Without session scope, stencils are instanciated at every test -> super slow
    def spec(self, arg_shape, nchannels, diffusion_op_klass, diffusion_op_kwargs, ndi, width):
        if ndi.module() is None:
            pytest.skip(f"{ndi} unsupported on this machine.")
        with pycrt.Precision(width):
            op = diffusion_op_klass(arg_shape=arg_shape, nchannels=nchannels, **diffusion_op_kwargs)
        return op, ndi, width


class DiffusionOpNoApplyMixin(DiffusionOpMixin):
    # Base class for _DiffusionOp objects who don't derive from a potential (with no apply() method)
    disable_test = frozenset(
        DiffusionOpMixin.disable_test
        | {
            "test_value1D_apply",
            "test_backend_apply",
            "test_prec_apply",
            "test_precCM_apply",
            "test_transparent_apply",
            "test_value1D_call",
            "test_backend_call",
            "test_prec_call",
            "test_precCM_call",
            "test_transparent_call",
            "test_interface_jacobian",  # Don't really understand this test, but it raises an error
        }
    )

    @pytest.fixture
    def data_apply(self):
        pass

    @pytest.fixture
    def _data_apply(self):
        pass


class TestMfiDiffusionOp(DiffusionOpMixin):
    @pytest.fixture(scope="session")
    def diffusion_op_klass(self):
        return pycdiffusion.MfiDiffusionOp

    @pytest.fixture(scope="session")
    def diffusion_op_kwargs(self):
        return dict(beta=1)

    @pytest.fixture
    def data_apply(self, arg_shape, nchannels, spec):  # Operator derives from potentional
        arr = self._random_array(arg_shape)  # TODO Replace with input-output pairs computed manually
        out = spec[0](arr.ravel())
        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )

    # @pytest.fixture
    # def data_prox(self, arg_shape, nchannels, spec):
    #     arr = self._random_array(arg_shape)  # TODO Replace with input-output pairs computed manually
    #     tau = 2
    #     out = spec[0].prox(arr.ravel(), tau=tau)
    #
    #     return dict(
    #         in_=dict(arr=arr.reshape(-1), tau=tau),
    #         out=out.reshape(-1),
    #     )

    @pytest.fixture
    def data_grad(self, arg_shape, nchannels, spec):
        arr = self._random_array(arg_shape)  # TODO Replace with input-output pairs computed manually
        out = spec[0].grad(arr.ravel())

        return dict(
            in_=dict(arr=arr.reshape(-1)),
            out=out.reshape(-1),
        )


# TODO all other diffusion ops
