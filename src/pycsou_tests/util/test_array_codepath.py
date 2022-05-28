import dask.array as da
import numpy as np
import pytest

import pycsou.util as pycu


def function_function():
    def f(x, y):
        return "f"

    def g(x, y):
        return "g"

    f = pycu.redirect("x", NUMPY=g)(f)
    return f, g


def function_staticmethod():
    def f(x, y):
        return "f"

    class Klass:
        @staticmethod
        def g(x, y):
            return "g"

    f = pycu.redirect("x", NUMPY=Klass.g)(f)
    return f, Klass.g


def staticmethod_function():
    class Klass:
        @staticmethod
        def f(x, y):
            return "f"

    def g(x, y):
        return "g"

    Klass.f = pycu.redirect("x", NUMPY=g)(Klass.f)
    return Klass.f, g


def staticmethod_staticmethod():
    class Klass:
        @staticmethod
        def f(x, y):
            return "f"

        @staticmethod
        def g(x, y):
            return "g"

    Klass.f = pycu.redirect("x", NUMPY=Klass.g)(Klass.f)
    return Klass.f, Klass.g


def method_method():
    class Klass:
        def f(self, x, y):
            return "f"

        def g(self, x, y):
            return "g"

    klass = Klass()
    klass.f = pycu.redirect("x", NUMPY=klass.g)(klass.f)
    return klass.f, klass.g


class TestArrayCodePath:
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
