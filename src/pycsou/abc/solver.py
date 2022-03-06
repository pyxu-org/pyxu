import collections.abc as cabc
import datetime as dt
import enum
import logging
import operator
import pathlib as plib
import shutil
import sys
import tempfile
import threading
import typing as typ

import numpy as np

import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct


@enum.unique
class Mode(enum.Enum):
    """
    Solver execution mode.
    """

    BLOCK = enum.auto()
    MANUAL = enum.auto()
    ASYNC = enum.auto()


class StoppingCriterion:
    """
    State(-less) machines (SM) which decide when to stop iterative solvers by examining their
    mathematical state.

    SM decisions are always accompanied by at least one numerical statistic. These stats may be
    queried by solvers via ``StoppingCriterion.info`` to provide diagnostic information to users.

    Composite stopping criteria can be implemented via the overloaded and[&]/or[|] operators.
    """

    def stop(self, state: cabc.Mapping) -> bool:
        """
        Compute a stop signal based on the current mathematical state.

        Parameters
        ----------
        state: Mapping[str]
            Full mathematical state of solver at some iteration, i.e. ``Solver._mstate``.
            Values from ``state`` may be cached inside the instance to form complex stopping
            conditions.

        Returns
        -------
        s: bool
            True if no further iterations should be performed, False otherwise.
        """
        raise NotImplementedError

    def info(self) -> cabc.Mapping[str, float]:
        """
        Get statistics associated with the last call to ``StoppingCriterion.stop``.

        Returns
        -------
        data: Mapping[str, float]
        """
        raise NotImplementedError

    def clear(self):
        """
        Clear SM state (if any).

        This method is useful when a ``StoppingCriterion`` instance must be reused in another call to
        ``Solver.fit``.
        """
        pass

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return _StoppingCriteriaComposition(lhs=self, rhs=other, op=operator.or_)

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return _StoppingCriteriaComposition(lhs=self, rhs=other, op=operator.and_)


class _StoppingCriteriaComposition(StoppingCriterion):
    def __init__(
        self,
        lhs: "StoppingCriterion",
        rhs: "StoppingCriterion",
        op: cabc.Callable[[bool, bool], bool],
    ):
        self._lhs = lhs
        self._rhs = rhs
        self._op = op

    def stop(self, state: cabc.Mapping) -> bool:
        return self._op(self._lhs.stop(state), self._rhs.stop(state))

    def info(self) -> cabc.Mapping[str, float]:
        return {**self._lhs.info(), **self._rhs.info()}

    def clear(self):
        self._lhs.clear()
        self._rhs.clear()


class Solver:
    r"""
    Iterative solver for minimization problems of the form

    .. math::

        \hat{x} = \arg\min_{x \in \mathbb{R}^{N}} \mathcal{F}(x),

    where the form of :math:`\mathcal{F}` is solver-dependent.

    Solver provides a versatile API for solving inverse problems, with the following features:

    * manual/automatic/background execution of solver iterations via parameters provided to
      `Solver.fit`. (See below.)
    * automatic checkpointing of solver progress, providing a safe restore point in case of faulty
      numerical code. Each solver instance backs its state and final output to a folder on disk for
      post-analysis. In particular ``Solver.fit`` will never crash: detailed exception information
      will always be available in a logfile for post-analysis.
    * arbitrary specification of complex stopping criteria via the `StoppingCriterion` class.
    * solve for multiple initial points in parallel.

    To implement a new iterative solver, users need to sub-class `Solver` and overwrite the methods
    below:

    * __init__()
    * fit()     # optional, for better signatures only.
    * m_init()  # i.e. math-init()
    * m_step()  # i.e. math-step()

    Advanced functionalities of `Solver` are automatically inherited by sub-classes.


    Examples
    --------
    Here are examples on how to solve minimization problems with this class:

    .. code-block:: python3

       slvr = Solver()

       ### 1. Blocking Mode: .fit() does not return until solver has stopped.
       >>> slvr.fit(mode=Mode.BLOCK, ...)
       >>> hist, data = slvr.stats()  # final output of solver.

       ### 2. Async Mode: solver iterations run in the background.
       >>> slvr.fit(mode=Mode.ASYNC, ...)
       >>> print('test')  # you can do something in between.
       >>> slvr.busy()  # or check whether the solver already stopped.
       >>> slvr.stop()  # and preemptively force it to stop.
       >>> hist, data = slvr.stats()  # then query the result after a (potential) force-stop.

       ### 3. Manual Mode: fine-grain control of solver data per iteration.
       >>> slvr.fit(mode=Mode.MANUAL, ...)
       >>> for data in slvr.steps():
       ...     # Do something with the logged variables after each iteration.
       ...     pass  # solver has stopped after the loop.
       >>> hist, data = slvr.stats()  # final output of solver.
    """

    def __init__(
        self,
        *,
        folder: typ.Optional[pyct.PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: pyct.VarName = frozenset(),
    ):
        """
        Parameters
        ----------
        folder: path-like
            Directory on disk where instance data should be stored. A location will be automatically
            chosen if unspecified. (Default: OS-dependent tempdir.)
        exist_ok: bool
            If ``folder`` is specified and ``exist_ok`` is false (the default), FileExistsError is
            raised if the target directory already exists.
        writeback_rate: int
            Rate at which solver checkpoints are written to disk. No checkpointing is done if
            unspecified: only the final solver output will be written back to disk.
        verbosity: int
            Rate at which stopping criteria statistics are logged to disk. If ``Solver.fit`` is run
            with mode=BLOCK, then statistics are also logged to stdout.
        log_var: VarName
            Variables from the solver's math-state (slvr._mstate) to be logged per iteration.
            These are the variables made available when calling ``Solver.stats``.
        """
        self._mstate = dict()  # mathematical state
        self._astate = dict(  # book-keeping (non-math) state
            history=None,  # stopping criteria values per iteration
            idx=0,  # iteration index
            log_rate=None,
            log_var=None,
            logger=None,
            stop_crit=None,
            wb_rate=None,
            workdir=None,
            # Execution-mode related -----------
            mode=None,
            active=None,
            worker=None,
        )

        try:
            if folder is None:
                folder = plib.Path(tempfile.mkdtemp(prefix="pycsou_"))
            elif (folder := plib.Path(folder).expanduser().resolve()).exists() and (not exist_ok):
                raise FileExistsError(f"{folder} already exists.")
            else:
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True)
            self._astate["workdir"] = folder
        except:
            raise Exception(f"folder: expected path-like, got {type(folder)}.")

        try:
            self._astate["wb_rate"] = writeback_rate
            if writeback_rate is not None:
                assert writeback_rate >= 1
                self._astate["wb_rate"] = int(writeback_rate)
        except:
            raise ValueError(f"writeback_rate must be positive, got {writeback_rate}.")

        try:
            assert verbosity >= 1
            self._astate["log_rate"] = int(verbosity)
        except:
            raise ValueError(f"verbosity must be positive, got {verbosity}.")

        try:
            if isinstance(log_var, str):
                log_var = (log_var,)
            self._astate["log_var"] = frozenset(log_var)
        except:
            raise ValueError(f"log_var: expected collection, got {type(log_var)}.")

    def fit(
        self,
        *args,
        stop_crit: StoppingCriterion,
        mode: Mode = Mode.BLOCK,
        **kwargs,
    ):
        r"""
        Solve minimization problem(s) defined in ``Solver.__init__``, with the provided run-specifc
        parameters.

        Parameterss
        ----------
        args[0]: NDArray
            (..., N) primal variable initial point(s).
        args[1]: NDArray
            (..., M) dual variable initial point(s). Only required for Primal-Dual solvers.
            ``*args`` must suitably broadcast along their leading dimensions.
        stop_crit: StoppingCriterion
            Stopping criterion to end solver iterations.
        mode: Mode
            Execution mode. See ``Solver`` for usage examples.
            Useful method pairs depending on the execution mode:
            * BLOCK: fit()
            * ASYNC: fit() + busy() + stop()
            * MANUAL: fit() + steps()
        **kwargs
            Extra solver-specific parameters, if applicable.

        Sub-classes may change the method signature of ``Solver.fit`` and ``Solver.m_init`` to improve
        clarity.
        """
        self._fit_init(mode, stop_crit)
        self.m_init(*args, **kwargs)
        self._fit_run()

    def m_init(self, *args, **kwargs):
        """
        Set solver's initial mathematical state based on (args, kwargs) provided to ``Solver.fit``.

        This method must only manipulate ``Solver._mstate``.

        After calling this method, the solver must be able to complete its 1st iteration via a call
        to ``Solver.m_step``.

        Sub-classes may change the method signature of ``Solver.fit`` and ``Solver.m_init`` to improve
        clarity.
        """
        raise NotImplementedError

    def m_step(self):
        """
        Perform one (mathematical) step.

        This method must only manipulate ``Solver._mstate``.
        """
        raise NotImplementedError

    def steps(self, n: typ.Optional[int] = None) -> cabc.Generator:
        """
        Generator of logged variables after each iteration.

        The i-th call to next() on this object returns the logged variables after the i-th solver
        iteration.

        This method is only usable after calling ``Solver.fit`` with mode=MANUAL. See ``Solver`` for
        usage examples.

        There is no guarantee that a checkpoint on disk exists when the generator is exhausted.
        (Reason: potential exceptions raised during solver's progress.) Users should invoke
        ``Solver.writeback`` afterwards if needed.

        Parameters
        ----------
        n: int
            Maximum number of next() calls allowed before exhausting the generator. Defaults to
            infinity if unspecified.

        The generator will terminate prematurely if the solver naturally stops before `n` calls to
        next() are made.
        """
        self._check_mode(Mode.MANUAL)
        i = 0
        while (n is None) or (i < n):
            if self._step():
                _, data = self.stats()
                yield data
                i += 1
            else:
                self._astate["mode"] = None  # force steps() to be call-once when exhausted.
                return

    def stats(self):
        """
        Query solver state.

        Returns
        -------
        history: np.ndarray | None
            (N_iter,) records of stopping-criteria values per iteration.
        data: dict[str, Real | NDArray | None]
            Value(s) of ``log_var``(s) after last iteration.

        Notes
        -----
        If any of the ``log_var``(s) and/or ``history`` are not (yet) known at query time, ``None`` is
        returned.
        """
        history = self._astate["history"]
        if history is not None:
            if len(history) > 0:
                history = np.concatenate(history, dtype=history[0].dtype, axis=0)
            else:
                history = None
        data = {k: self._mstate.get(k) for k in self._astate["log_var"]}
        return history, data

    @property
    def workdir(self) -> plib.Path:
        """
        Returns
        -------
        wd: plib.Path
            Absolute path to the directory on disk where instance data is stored.
        """
        return self._astate["workdir"]

    @property
    def logfile(self) -> plib.Path:
        """
        Returns
        -------
        lf: plib.Path
            Absolute path to the log file on disk where stopping criteria statistics are logged.
        """
        return self.workdir / "solver.log"

    @property
    def datafile(self) -> plib.Path:
        """
        Returns
        -------
        df: plib.Path
            Absolute path to the file on disk where ``log_var``(s) are stored during checkpointing or
            after solver has stopped.
        """
        return self.workdir / "data.npz"

    def busy(self) -> bool:
        """
        Test if an async-running solver has stopped.

        This method is only usable after calling ``Solver.fit`` with mode=ASYNC. See ``Solver`` for
        usage examples.

        Returns
        -------
        b: bool
            True if solver has stopped, False otherwise.
        """
        self._check_mode(Mode.ASYNC, Mode.BLOCK)
        return self._astate["active"].is_set()

    def solution(self):
        """
        Output the "solution" of the optimization problem.

        This is a helper method intended for novice users. The return type is sub-class dependent,
        so don't write an API using this: use ``Solver.stats`` instead.
        """
        raise NotImplementedError

    def stop(self):
        """
        Stop an async-running solver.

        This method is only usable after calling ``Solver.fit`` with mode=ASYNC. See ``Solver`` for
        usage examples.

        This method will block until the solver has stopped.

        There is no guarantee that a checkpoint on disk exists once halted. (Reason: potential
        exceptions raised during solver's progress.) Users should invoke ``Solver.writeback``
        afterwards if needed.

        Users must call this method to terminate an async-solver, even if ``Solver.busy`` is False.
        """
        self._check_mode(Mode.ASYNC, Mode.BLOCK)
        self._astate["active"].clear()
        self._astate["worker"].join()
        self._astate.update(
            mode=None,  # forces stop() to be call-once.
            active=None,
            worker=None,
        )

    def _fit_init(self, mode: Mode, stop_crit: StoppingCriterion):
        def _init_logger():
            log_name = str(self.workdir)
            logger = logging.getLogger(log_name)
            logger.handlers.clear()
            logger.setLevel("DEBUG")

            fmt = logging.Formatter(fmt="{levelname} -- {message}", style="{")
            handler = [logging.FileHandler(self.logfile, mode="w")]
            if mode is Mode.BLOCK:
                handler.append(logging.StreamHandler(sys.stdout))
            for h in handler:
                h.setLevel("DEBUG")
                h.setFormatter(fmt)
                logger.addHandler(h)

            return logger

        self._mstate.clear()
        stop_crit.clear()
        self._astate.update(  # suitable state for a new call to fit().
            history=None,
            idx=0,
            logger=_init_logger(),
            stop_crit=stop_crit,
            mode=mode,
            active=None,
            worker=None,
        )

    def _fit_run(self):
        self._m_persist()

        mode = self._astate["mode"]
        if mode is Mode.MANUAL:
            # User controls execution via steps().
            pass
        else:  # BLOCK / ASYNC
            self._astate.update(
                active=threading.Event(),
                worker=Solver._Worker(self),
            )
            self._astate["active"].set()
            self._astate["worker"].start()
            if mode is Mode.BLOCK:
                self._astate["worker"].join()
                self.stop()  # state clean-up
            else:
                # User controls execution via busy() + stop().
                pass

    def writeback(self):
        """
        Checkpoint state to disk.
        """
        history, data = self.stats()
        kwargs = {k: v for (k, v) in dict(history=history, **data).items() if (v is not None)}
        np.savez(self.datafile, **pycu.compute(kwargs))  # savez() requires NumPy arrays as input.
        # [TODO][Feature Request] Allow user to choose writeback format. Useful for large-scale
        # outputs which cannot be stored on one machine.

    def _check_mode(self, *modes: Mode):
        m = self._astate["mode"]
        if m in modes:
            pass  # ok
        else:
            if m is None:
                msg = "Illegal method call: invoke Solver.fit() first."
            else:
                msg = " ".join(
                    [
                        "Illegal method call: can only be used if Solver.fit() invoked with",
                        "mode=Any[" + ", ".join(map(lambda _: str(_.name), modes)) + "]",
                    ]
                )
            raise ValueError(msg)

    def _step(self) -> bool:
        ast = self._astate  # shorthand

        def _log(msg: str = None):
            if msg is None:  # report stopping-criterion values
                h = ast["history"][-1][0]
                msg = [f"[{dt.datetime.now()}] Iteration {ast['idx']:>_d}"]
                for field, value in zip(h.dtype.names, h):
                    msg.append(f"\t{field}: {value}")
                msg = "\n".join(msg)
            ast["logger"].info(msg)

        def _update_history():
            def _as_struct(data: dict[str, float]) -> np.ndarray:
                ftype = pycrt.getPrecision().value
                dtype = np.dtype([(k, ftype) for k in data])
                s = np.array(list(data.values()), dtype=ftype).view(dtype)
                return s

            h = _as_struct(ast["stop_crit"].info())
            ast["history"].append(h)

        try:
            if ast["stop_crit"].stop(self._mstate):
                _log(msg=f"[{dt.datetime.now()}] Stopping Criterion satisfied -> END")
                self.writeback()
                return False
            else:
                ast["idx"] += 1
                self.m_step()
                self._m_persist()
                _update_history()
                if ast["idx"] % ast["log_rate"] == 0:
                    _log()
                if (ast["wb_rate"] is not None) and (ast["idx"] % ast["wb_rate"] == 0):
                    self.writeback()
                return True
        except Exception as e:
            msg = f"[{dt.datetime.now()}] Something went wrong -> EXCEPTION RAISED"
            msg_xtra = f"More information: {self.logfile}."
            print("\n".join([msg, msg_xtra]), file=sys.stderr)
            if ast["wb_rate"] is not None:  # checkpointing enabled
                _, r = divmod(ast["idx"], ast["wb_rate"])
                idx_valid = ast["idx"] - r
                msg_idx = f"Last valid checkpoint done at idx={idx_valid}."
                msg = "\n".join([msg, msg_idx])
            ast["logger"].exception(msg, exc_info=e)
            return False

    def _m_persist(self):
        # Persist math state to avoid re-eval overhead.
        self._mstate.update(**pycu.compute(self._mstate, mode="persist"))

    class _Worker(threading.Thread):
        def __init__(self, solver: "Solver"):
            super().__init__()
            self.slvr = solver

        def run(self):
            while self.slvr.busy() and self.slvr._step():
                pass
            self.slvr._astate["active"].clear()
