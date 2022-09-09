import concurrent.futures as cf
import functools as ft
import types
import typing as typ

import numpy as np

import pycsou.abc as pyca
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


def stack(
    maps: typ.Iterable[pyct.OpT],
    axis: typ.Literal[0, 1, -1] = 0,
    executor: typ.Union[
        None, cf.ThreadPoolExecutor, cf.ProcessPoolExecutor, typ.Literal["threads", "processes"]
    ] = None,
    max_workers: typ.Optional[int] = None,
) -> pyct.OpT:  # TODO: **kwargs instead of max_workers
    r"""
    Stack multiple instances of :py:class:`~pycsou.abc.operator.Map` subclasses together.

    Parameters
    ----------
    maps: :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
        Instances of :py:class:`~pycsou.abc.operator.Map` subclasses to be stacked. The various maps should have compatible shapes
        for the desired stacking mode (identical ``dim``/``codim`` for vertical/horizontal stacking respectively). For horizontal stacking,
        **no map should be domain-agnostic**, as the stacking of domain agnostic maps is ambiguous.
    axis: 0 | 1 | -1
        Determine whether the stacking is *vertical* (``axis=0``) or *horizontal* (``axis=1`` or ``axis =-1``).
    executor: None | :py:class:`concurrent.futures.ThreadPoolExecutor` | :py:class:`concurrent.futures.ProcessPoolExecutor` | 'threads' | 'processes'
        Define an :py:class:`concurrent.futures.Executor` subclass which will be used to execute calls to the individual methods of the stacked maps
        asynchronously using a pool of workers. If ``executor=None`` the executor will be of type  :py:class:`concurrent.futures.ThreadPoolExecutor`.
        If ``executor='threads'`` or ``executor='processes'`` then the executor will be of type  :py:class:`concurrent.futures.ThreadPoolExecutor` and
        :py:class:`concurrent.futures.ProcessPoolExecutor` with default arguments. The :py:class:`concurrent.futures.Executor` subclass can
        also be directly passed for finer control over the executor's parameters.
    max_workers: int
        Maximum number of workers in the pool. Set to one for serial execution (no multi[threading/processing]).
        If ``None`` defaults to ``min(len(maps), 32)``.

    Returns
    -------
    :py:class:`~pycsou.abc.operator.Map` | :py:class:`~pycsou.abc.operator.DiffMap` | :py:class:`~pycsou.abc.operator.Func` | :py:class:`~pycsou.abc.operator.DiffFunc` | :py:class:`~pycsou.abc.operator.ProxFunc` | :py:class:`~pycsou.abc.operator.ProxDiffFunc` | :py:class:`~pycsou.abc.operator.LinOp` | :py:class:`~pycsou.abc.operator.LinFunc`
    Stack map.

    Raises
    ------
    ValueError
        In case of incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from pycsou.operator.linop.base import ExplicitLinFunc, ExplicitLinOp
    >>> from pycsou.compound import stack
    >>> # Define a bunch of operators/functionals
    >>> vec1 = np.arange(10); f1 = ExplicitLinFunc(vec1)
    >>> vec2 = np.arange(5,15); f2 = ExplicitLinFunc(vec2)
    >>> mat1 = np.arange(20).reshape(2,10); m1 = ExplicitLinOp(mat1)
    >>> mat2 = np.arange(40).reshape(2,20); m2 = ExplicitLinOp(mat2)
    >>> # Stack them
    >>> stack1 = stack([f1, m1], axis=0); mstack1 = np.concatenate([vec1[None, :], mat1], axis=0)
    >>> np.allclose(stack1.asarray(), mstack1)
    True
    >>> stack2 = stack([f1, f2], axis=0); mstack2 = np.stack([vec1, vec2], axis=0)
    >>> np.allclose(stack2.asarray(), mstack2)
    True
    >>> stack3 = stack([m1, m2], axis=1); mstack3 = np.concatenate([mat1, mat2], axis=1)
    >>> np.allclose(stack3.asarray(), mstack3)
    True
    """

    @pycrt.enforce_precision(i="arr", o=False)
    def _method_vstack(
        _,
        arr: pyct.NDArray,
        methods: typ.List[typ.Callable],
        executor: typ.Union[cf.ThreadPoolExecutor, cf.ProcessPoolExecutor],
    ) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        fs = [executor.submit(method, arr) for method in methods]
        cf.wait(fs)
        out_list = []
        for f in fs:
            res = f.result()
            if res.ndim == 1:
                res = res[:, None]
            out_list.append(res)
        return xp.concatenate(out_list, axis=-1)

    @pycrt.enforce_precision(i="arr", o=False)
    def _method_separable_sum(
        _,
        arr: pyct.NDArray,
        methods: typ.List[typ.Callable],
        sections: np.ndarray,
        executor: typ.Union[cf.ThreadPoolExecutor, cf.ProcessPoolExecutor],
    ) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        arr_split = xp.split(arr, xp.asarray(sections), axis=-1)
        fs = [executor.submit(method, arr_split[i]) for i, method in enumerate(methods)]
        result = 0
        for f in cf.as_completed(fs):
            result += f.result()
        return result

    @pycrt.enforce_precision(i="arr", o=False)
    def _jacobian_stack(
        _,
        arr: pyct.NDArray,
        axis: typ.Literal[0, 1, -1],
        methods: typ.List[typ.Callable],
        executor: typ.Union[cf.ThreadPoolExecutor, cf.ProcessPoolExecutor],
        max_workers: typ.Optional[int],
    ) -> pyca.LinOp:
        fs = [executor.submit(method, arr) for method in methods]
        cf.wait(fs)
        out_list = [f.result() for f in fs]
        return stack(out_list, axis=axis, executor=executor, max_workers=max_workers)

    maps = list(maps)
    if abs(axis) > 1:
        ValueError("Axis must be one of {0, 1,-1}.")
    axis = int(axis)
    max_workers = min(len(maps), 32) if max_workers is None else max_workers
    if executor is None or executor == "threading":
        executor = cf.ThreadPoolExecutor(max_workers=max_workers)
    elif executor == "processes":
        executor = cf.ProcessPoolExecutor(max_workers=max_workers)
    else:
        try:
            assert isinstance(executor, cf.Executor)
        except:
            raise ValueError(f"Unknown executor type {executor.__class__}.")
    out_shape = pycu.infer_stack_shape(*[map_.shape for map_ in maps], axis=axis)
    sections = np.cumsum([map_.shape[axis] for map_ in maps])
    shared_props = maps[0].properties().intersection(*[m.properties() for m in maps[1:]])
    if axis == 0:
        shared_props.discard("prox")
        # if two functionals are stacked vertically, they become a multi-valued map so we discard the properties attached to single-valued maps:
        shared_props.discard("single_valued")
        stack_of_funcs = "grad" in shared_props
        shared_props.discard("grad")
    for Op in pyca._base_operators:
        if Op.properties() == shared_props:
            break
    if Op in [pyca.LinOp, pyca.DiffFunc, pyca.ProxDiffFunc, pyca.LinFunc]:
        shared_props.discard("jacobian")  # The method jacobian is implicitly defined for such objects.
    shared_props.discard("single_valued")  # Useful for determining the base class only, can discard now.
    out_op = Op(out_shape)
    for prop in shared_props:
        if prop in ["lipschitz", "diff_lipschitz"]:
            setattr(out_op, "_" + prop, np.linalg.norm(np.array([getattr(m, "_" + prop) for m in maps])))
        else:
            methods = [getattr(m, prop) for m in maps]
            if prop == "apply":
                if axis == 0:
                    multi_prop = ft.partial(_method_vstack, methods=methods, executor=executor)
                else:
                    multi_prop = ft.partial(
                        _method_separable_sum, methods=methods, sections=sections, executor=executor
                    )
            elif prop == "jacobian":
                if axis == 0 and stack_of_funcs:

                    @pycrt.enforce_precision(i="arr", o=False)
                    def _jacobian_from_grads(methods, _, arr: pyct.NDArray) -> pycl.ExplicitLinOp:
                        fs = [executor.submit(grad, arr) for grad in methods]
                        cf.wait(fs)
                        out_list = [f.result() for f in fs]
                        xp = pycu.get_array_module(arr)
                        return pycl.ExplicitLinOp(map=xp.stack(out_list, axis=0), enable_warnings=True)

                    multi_prop = ft.partial(_jacobian_from_grads, methods)
                else:
                    multi_prop = ft.partial(
                        _jacobian_stack, axis=axis, methods=methods, executor=executor, max_workers=max_workers
                    )
            elif prop == "grad":
                if axis == 0:
                    raise ValueError(f"[grad] method is not defined for map of type {Op}.")
                else:
                    multi_prop = ft.partial(
                        _method_separable_sum, methods=methods, sections=sections, executor=executor
                    )
            elif prop == "prox":
                if axis == 0:
                    raise ValueError(f"[prox] method is not defined for map of type {Op}.")
                else:

                    @pycrt.enforce_precision(i="arr", o=False)
                    def _component_wise_prox(methods, _, arr: pyct.NDArray) -> pyct.NDArray:
                        xp = pycu.get_array_module(arr)
                        arr_split = xp.split(arr, xp.asarray(sections))
                        fs = [executor.submit(method, arr_split[i]) for i, method in enumerate(methods)]
                        cf.wait(fs)
                        out_list = []
                        for f in fs:
                            res = f.result()
                            if res.ndim == 1:
                                res = res[:, None]
                            out_list.append(res)
                        return xp.concatenate(out_list, axis=-1)

                    multi_prop = ft.partial(_component_wise_prox, methods)
            elif prop == "adjoint":
                if axis == 0:
                    multi_prop = ft.partial(
                        _method_separable_sum, methods=methods, sections=sections, executor=executor
                    )
                else:
                    multi_prop = ft.partial(_method_vstack, methods=methods, executor=executor)
            else:
                raise ValueError(f"Unknown property {prop}!")
            setattr(out_op, prop, types.MethodType(multi_prop, out_op))
    return out_op.squeeze()


def hstack(
    maps: typ.Iterable[pyct.OpT],
    executor: typ.Union[
        None, cf.ThreadPoolExecutor, cf.ProcessPoolExecutor, typ.Literal["threads", "processes"]
    ] = None,
    max_workers: typ.Optional[int] = None,
) -> pyct.OpT:
    return stack(maps, axis=1, executor=executor, max_workers=max_workers)


def vstack(
    maps: typ.Iterable[pyct.OpT],
    executor: typ.Union[
        None, cf.ThreadPoolExecutor, cf.ProcessPoolExecutor, typ.Literal["threads", "processes"]
    ] = None,
    max_workers: typ.Optional[int] = None,
) -> pyct.OpT:
    return stack(maps, axis=0, executor=executor, max_workers=max_workers)
