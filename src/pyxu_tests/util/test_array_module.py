import dask.array as da
import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu


class TestGetArrayModule:
    def test_array(self, xp):
        x = xp.arange(5)
        assert pxu.get_array_module(x) is xp

    @pytest.mark.parametrize(
        ["obj", "fallback", "fail"],
        [
            [None, None, True],
            [None, np, False],
            [1, None, True],
            [1, np, False],
        ],
    )
    def test_fallback(self, obj, fallback, fail):
        # object is not an array type, so fail or return provided fallback
        if not fail:
            assert pxu.get_array_module(obj, fallback) is fallback
        else:
            with pytest.raises(ValueError):
                assert pxu.get_array_module(obj, fallback)


class TestCompute:
    @pytest.fixture(
        params=[
            1,
            [1, 2, 3],
            np.arange(5),
            da.arange(5),
        ]
    )
    def single_input(self, request):
        return request.param

    def equal(self, x, y):
        if any(type(_) in pxd.supported_array_types() for _ in [x, y]):
            return np.allclose(x, y)
        else:
            return x == y

    @pytest.fixture(params=["compute", "persist"])
    def mode(self, request):
        return request.param

    def test_single_inputs(self, single_input, mode):
        cargs = pxu.compute(single_input, mode=mode)
        assert self.equal(cargs, single_input)

    def test_multi_inputs(self, mode):
        x = da.arange(5)
        y = x + 1
        i_args = (1, [1, 2, 3], np.arange(5), x, y)
        o_args = (1, [1, 2, 3], np.arange(5), np.arange(5), np.arange(1, 6))
        cargs = pxu.compute(*i_args, mode=mode)

        assert len(cargs) == len(o_args)
        for c, o in zip(cargs, o_args):
            assert self.equal(c, o)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            pxu.compute(1, mode="test")

    def test_kwargs_does_not_fail(self):
        x = 1
        pxu.compute(x, optimize_graph=False)


class TestToNumPy:
    @pytest.fixture
    def arr(self) -> pxt.NDArray:
        return np.arange(-5, 5)

    @pytest.fixture
    def _arr(self, arr, xp):
        return xp.array(arr, dtype=arr.dtype)

    def test_backend_change(self, _arr):
        N = pxd.NDArrayInfo
        np_arr = pxu.to_NUMPY(_arr)

        assert N.from_obj(np_arr) == N.NUMPY
        if N.from_obj(_arr) == N.NUMPY:
            assert _arr is np_arr


class TestRedirect:
    def function_function():
        def f(x, y):
            return "f"

        def g(x, y):
            return "g"

        f = pxu.redirect("x", NUMPY=g)(f)
        return f, g

    def function_staticmethod():
        def f(x, y):
            return "f"

        class Klass:
            @staticmethod
            def g(x, y):
                return "g"

        f = pxu.redirect("x", NUMPY=Klass.g)(f)
        return f, Klass.g

    def staticmethod_function():
        class Klass:
            @staticmethod
            def f(x, y):
                return "f"

        def g(x, y):
            return "g"

        Klass.f = pxu.redirect("x", NUMPY=g)(Klass.f)
        return Klass.f, g

    def staticmethod_staticmethod():
        class Klass:
            @staticmethod
            def f(x, y):
                return "f"

            @staticmethod
            def g(x, y):
                return "g"

        Klass.f = pxu.redirect("x", NUMPY=Klass.g)(Klass.f)
        return Klass.f, Klass.g

    def method_method():
        class Klass:
            def f(self, x, y):
                return "f"

            def g(self, x, y):
                return "g"

        klass = Klass()
        klass.f = pxu.redirect("x", NUMPY=klass.g)(klass.f)
        return klass.f, klass.g

    @pytest.fixture(
        params=[
            function_function(),  # function -> function
            function_staticmethod(),  # function -> staticmethod
            staticmethod_function(),  # staticmethod -> function
            staticmethod_staticmethod(),  # staticmethod -> staticmethod
            method_method(),  # method -> method
        ]
    )
    def callables(self, request):
        return request.param

    def test_invalid_signature(self, callables):
        f_decorated, f_target = callables
        with pytest.raises(ValueError):
            f_decorated(z=1)

    def test_nonArray_input(self, callables):
        f_decorated, f_target = callables
        with pytest.raises(ValueError):
            f_decorated(x=1, y=1)

    def test_dispatch_to_default(self, callables):
        f_decorated, f_target = callables

        kwargs = dict(x=da.array([1]), y=1)
        assert f_decorated(**kwargs) == "f"

    def test_dispatch_to_custom(self, callables):
        f_decorated, f_target = callables

        kwargs = dict(x=np.array([1]), y=1)
        assert f_decorated(**kwargs) == f_target(**kwargs)


class TestCopyIfUnsafe:
    @pytest.fixture(
        params=[
            np.r_[1],
            np.ones((5,)),
            np.ones((5, 3, 4)),
        ]
    )
    def x(self, request):
        return request.param

    def test_no_copy(self, xp, x):
        x = xp.array(x)

        y = pxu.copy_if_unsafe(x)
        assert x is y

    @pytest.mark.parametrize("xp", set(pxd.supported_array_modules()) - {da})
    @pytest.mark.parametrize("mode", ["read_only", "view"])
    def test_copy(self, xp, x, mode):
        x = xp.array(x)
        if mode == "read_only":
            x.flags.writeable = False
        elif mode == "view":
            x = x.view()

        y = pxu.copy_if_unsafe(x)
        assert y.flags.owndata
        assert y.shape == x.shape
        assert xp.allclose(y, x)


class TestReadOnly:
    @pytest.fixture(
        params=[
            np.ones((1,)),  # contiguous
            np.ones((5,)),  # contiguous
            np.ones((5, 3, 4)),  # multi-dim, contiguous
            np.ones((5, 3, 4))[:, ::-1],  # multi-dim, view
        ]
    )
    def x(self, request):
        return request.param

    def test_transparent(self, x):
        x = da.array(x)
        y = pxu.read_only(x)
        assert y is x

    @pytest.mark.parametrize("xp", set(pxd.supported_array_modules()) - {da})
    def test_readonly(self, xp, x):
        x = xp.array(x)
        y = pxu.read_only(x)

        if hasattr(y.flags, "writeable"):
            assert not y.flags.writeable
        assert not y.flags.owndata
        assert y.shape == x.shape
        assert xp.allclose(y, x)
