import logging
import pathlib as plib
import shutil
import tempfile
import threading
import typing as typ

PathLike = typ.Union[str, plib.Path]
Data = typ.Union[nb.Number, np.ndarray]


class StoppingCriterion:
    # TODO
    pass


class Solver:
    # Q: what does user have to overwrite in subclass?
    # * m_init()
    # * m_step()
    # * __init__()

    def __init__(
        self,
        stop_crit: StoppingCriterion,
        *,
        folder: typ.Optional[PathLike] = None,
        exist_ok: bool = False,
        writeback_rate: typ.Optional[int] = None,
        verbosity: int = 1,
        log_var: typ.Union[str, cabc.Collection[str]] = frozenset(),
    ):
        self._mstate = dict()  # mathematical state
        self._astate = dict(  # book-keeping (non-math) state
            active=None,  # threading.Event
            worker=None,  # _Worker
            stop_crit=stop_crit,
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
        block: bool = True,
        manual: bool = False,
        **kwargs,
    ):
        self._fit_init()
        self.m_init(*args, **kwargs)

        if manual:
            if block:
                raise ValueError(f"Cannot run solver in non-blocking manual mode.")
            else:
                # User wants to execute step() himself.
                # active/worker are meaningless in this context, so stop() not required.
                pass
        else:
            self._astate["active"] = threading.Event()
            self._astate["active"].set()

            self._astate["worker"] = _Worker(self)
            self._astate["worker"].start()
            if block:
                self._astate["worker"].join()
                self._astate["worker"] = None
                self._astate["active"] = None
            else:
                # User wants to query solver status himself.
                # If busy() == False -> solver finished without manual intervention. No need to call stop().
                # If busy() == True -> stopping criteria not met, but user can force stop().
                pass

    def _fit_init(self):
        # reinit folder + internal non-math state
        # TODO
        pass

    def m_init(self, *args, **kwargs):
        # takes fit() params and sets required math-state in _mstate
        raise NotImplementedError

    def stop(self):
        if self.busy():
            self._astate["active"].clear()
            self._astate["worker"].join()
        else:
            pass

    def step(self) -> bool:
        # do a step of algo. Return True if another iteration is possible afterwards.
        # returns false and does nothing if stop-crit already met.
        # not step()'s job to manage active/worker
        # TODO
        pass

    def stats(self) -> typ.Tuple[np.ndarray, typ.Mapping[str, Data]]:
        # arg1: history ndarray
        # arg2: dict of log_var -> data pairs
        # (None, None) if not yet available
        # TODO
        pass

    @property
    def workdir(self) -> plib.Path:
        return self._astate["workdir"]

    def busy(self) -> bool:
        return self._astate["active"].is_set()

    @property
    def _log(self) -> logging.Logger:
        # Todo
        log_name = str(self.workdir)
        return logging.getLogger(log_name)

    class _Worker(threading.Thread):
        def __init__(self, solver: "Solver"):
            self.slvr = solver

        def run(self):
            while self.slvr.busy() and self.slvr.step():
                pass
            self.slvr._astate["active"].clear()
