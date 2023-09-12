import dask.array as da
import numpy as np
import pytest

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
