import cmath
import collections
import collections.abc as cabc
import functools
import itertools
import math
import threading
import types
import typing as typ
import warnings

import dask
import numba
import numba.cuda
import numpy as np
import scipy.optimize as sopt
import scipy.spatial as spl

import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.operator.linop.select as pxs
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.util.operator import _array_ize, _dask_zip

finufft = pxu.import_module("finufft", fail_on_error=False)
if finufft is None:
    finufft_Plan = typ.TypeVar("finufft_Plan", bound="finufft.Plan")
else:
    finufft_Plan = finufft.Plan


__all__ = [
    "NUFFT",
]

SignT = typ.Literal[1, -1]
sign_default = 1
eps_default = 1e-4


@pxrt.enforce_precision(i=("z", "beta"))
def ES_kernel(z: pxt.NDArray, beta: pxt.Real) -> pxt.NDArray:
    r"""
    Evaluate the Exponential of Semi-Circle (ES) kernel.

    Parameters
    ----------
    z: NDArray
        (N,) evaluation points
    beta: Real
        cutoff-frequency

    Returns
    -------
    phi: NDArray
        (N,) kernel values at evaluation points.

    Notes
    -----
    The Exponential of Semi-Circle (ES) kernel is defined as (see [FINUFFT]_ eq. (1.8)):

    .. math::

       \phi_\beta(z)
       =
       \begin{cases}
           e^{\beta(\sqrt{1-z^2}-1)}, & |z|\leq 1,\\
           0,                         &\text{otherwise.}
       \end{cases}
    """
    assert beta > 0
    xp = pxu.get_array_module(z)

    phi = xp.zeros_like(z)
    mask = xp.fabs(z) <= 1
    phi[mask] = xp.exp(beta * (xp.sqrt(1 - z[mask] ** 2) - 1))

    return phi


class NUFFT(pxa.LinOp):
    r"""
    Non-Uniform Fast Fourier Transform (NUFFT) of Type 1/2/3 (for :math:`d=\{1,2,3\}`).

    The *Non-Uniform Fast Fourier Transform (NUFFT)* generalizes the FFT to off-grid data.  There are three types of
    NUFFTs proposed in the literature:

    * Type 1 (*non-uniform to uniform*),
    * Type 2 (*uniform to non-uniform*),
    * Type 3 (*non-uniform to non-uniform*).

    See the notes below, including [FINUFFT]_, for definitions of the various transforms and algorithmic details.

    The transforms should be instantiated via :py:meth:`~pyxu.operator.NUFFT.type1`,
    :py:meth:`~pyxu.operator.NUFFT.type2`, and :py:meth:`~pyxu.operator.NUFFT.type3` respectively.  (See each method for
    usage examples.)

    The dimension of each transform is inferred from the dimensions of the input arguments, with support for
    :math:`d=\{1,2,3\}`.

    Notes
    -----
    We adopt here the same notational conventions as in [FINUFFT]_.

    * **Mathematical Definition.**
      Let :math:`d\in\{1,2,3\}` and consider the mesh

      .. math::

         \mathcal{I}_{N_1,\ldots,N_d}
         =
         \mathcal{I}_{N_1} \times \cdots \times \mathcal{I}_{N_d}
         \subset \mathbb{Z}^d,

      where the mesh indices :math:`\mathcal{I}_{N_i}\subset\mathbb{Z}` are given for each dimension :math:`i=1,\dots,
      d` by:

      .. math::

         \mathcal{I}_{N_i}
         =
         \begin{cases}
             [[-N_i/2, N_i/2-1]], & N_i\in 2\mathbb{N} \text{ (even)}, \\
             [[-(N_i-1)/2, (N_i-1)/2]], & N_i\in 2\mathbb{N}+1 \text{ (odd)}.
         \end{cases}


      Then the NUFFT operators approximate, up to a requested relative accuracy :math:`\varepsilon \geq 0`, [#]_ the
      following exponential sums:

      .. math::

         \begin{align}
             (1)\; &u_{\mathbf{n}} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle}, \quad &\mathbf{n}\in \mathcal{I}_{N_1,\ldots, N_d},\qquad &\text{Type 1 (non-uniform to uniform)}\\
             (2)\; &w_{j} = \sum_{\mathbf{n}\in\mathcal{I}_{N_1,\ldots, N_d}} u_{\mathbf{n}} e^{\sigma i\langle \mathbf{n}, \mathbf{x}_{j} \rangle }, \quad &j=1,\ldots, M,\qquad  &\text{Type 2 (uniform to non-uniform)}\\
             (3)\; &v_{k} = \sum_{j=1}^{M} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \quad &k=1,\ldots, N, \qquad &\text{Type 3 (non-uniform to non-uniform)}
         \end{align}

      where :math:`\sigma \in \{+1, -1\}` defines the sign of the transform and :math:`u_{\mathbf{n}}, v_{k}, w_{j}\in
      \mathbb{C}`.  For the type-1 and type-2 NUFFTs, the non-uniform samples :math:`\mathbf{x}_{j}` are assumed to lie
      in :math:`[-\pi,\pi)^{d}`.  For the type-3 NUFFT, the non-uniform samples :math:`\mathbf{x}_{j}` and
      :math:`\mathbf{z}_{k}` are arbitrary points in :math:`\mathbb{R}^{d}`.

    * **Adjoint NUFFTs.**
      The type-1 and type-2 NUFFTs with opposite signs form an *adjoint pair*.  The adjoint of the type-3 NUFFT is
      obtained by flipping the transform's sign and switching the roles of :math:`\mathbf{z}_k` and
      :math:`\mathbf{x}_{j}` in (3).

    * **Lipschitz Constants.**
      We bound the Lipschitz constant by the Frobenius norm of the operators, which yields :math:`L \leq \sqrt{NM}`.
      Note that these Lipschitz constants are cheap to compute but may be pessimistic.  Tighter Lipschitz constants can
      be computed by calling :py:meth:`~pyxu.abc.Map.estimate_lipschitz`.

    * **Error Analysis.**
      Let :math:`\tilde{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
      :math:`\tilde{\mathbf{w}}\in\mathbb{C}^{M}` be the outputs of the type-1 and type-2 NUFFT algorithms which
      approximate the sequences :math:`{\mathbf{u}}\in\mathbb{C}^{\mathcal{I}_{N_1,\ldots, N_d}}` and
      :math:`{\mathbf{w}}\in\mathbb{C}^{M}` defined in (1) and (2) respectively.  Then [FINUFFT]_ shows that the
      relative errors :math:`\|\tilde{\mathbf{u}}-{\mathbf{u}}\|_2/\|{\mathbf{u}}\|_2` and
      :math:`\|\tilde{\mathbf{w}}-{\mathbf{w}}\|_2/\|{\mathbf{w}}\|_2` are *almost always similar to the user-requested
      tolerance* :math:`\varepsilon`, except when round-off error dominates (i.e. very small user-requested tolerances).
      The same holds approximately for the type-3 NUFFT.  Note however that this is a *typical error analysis*: some
      degenerate (but rare) worst-case scenarios can result in higher errors.

    * **Complexity.**
      Naive evaluation of the exponential sums (1), (2) and (3) above costs :math:`O(NM)`, where :math:`N=N_{1}\ldots
      N_{d}` for the type-1 and type-2 NUFFTs.  NUFFT algorithms approximate these sums to a user-specified relative
      tolerance :math:`\varepsilon` in log-linear complexity in :math:`N` and :math:`M`.  The complexity of the various
      NUFFTs are given by (see [FINUFFT]_):

      .. math::

         &\mathcal{O}\left(N \log(N) + M|\log(\varepsilon)|^d\right)\qquad &\text{(Type 1 and 2)}\\
         &\mathcal{O}\left(\Pi_{i=1}^dX_iZ_i\sum_{i=1}^d\log(X_iZ_i) + (M + N)|\log(\varepsilon)|^d\right)\qquad &\text{(Type 3)}

      where :math:`X_i = \max_{j=1,\ldots,M}|(\mathbf{x}_{j})_i|` and :math:`Z_i =
      \max_{k=1,\ldots,N}|(\mathbf{z}_k)_i|` for :math:`i=1,\ldots,d`.  The terms above correspond to the complexities
      of the FFT and spreading/interpolation steps respectively.

    * **Memory footprint.**
      The complexity and memory footprint of the type-3 NUFFT can be arbitrarily large for poorly-centered data, or for
      data with a large spread.  An easy fix consists in centering the data before/after the NUFFT via pre/post-phasing
      operations, as described in equation (3.24) of [FINUFFT]_.  This optimization is automatically carried out by
      FINUFFT if the compute/memory gains are non-negligible.

      Additionally the type-3 summation (eq. 3) can be evaluated block-wise by partitioning the non-uniform samples
      :math:`(x_{j}, z_{k})`:

      .. math::

         \begin{align}
             (4)\;\; &v_{k}
             =
             \sum_{p=1}^{P}
             \sum_{j\in \mathcal{M}_{p}} w_{j}
             e^{\sigma i\langle \mathbf{z}_{k}, \mathbf{x}_{j} \rangle },
             \quad &
             k\in \mathcal{N}_{q}, \quad q=1,\ldots, Q, \quad & \text{Type 3 (chunked)}
         \end{align}

      where :math:`\{\mathcal{M}_1, \ldots, \mathcal{M}_P\}` and :math:`\{\mathcal{N}_1, \ldots, \mathcal{N}_Q\}` are
      *partitions* of the sets :math:`\{1, \ldots, M\}` and  :math:`\{1, \ldots, N\}` respectively.  In the chunked
      setting, the partial sums in (4)

      .. math::

         \left\{\sum_{j\in \mathcal{M}_p} w_{j} e^{\sigma i\langle \mathbf{z}_k, \mathbf{x}_{j} \rangle }, \, k\in \mathcal{N}_q\right\}_{p,q}

      are each evaluated via a type-3 NUFFT involving a subset of the non-uniform samples.  Sub-problems can be
      evaluated in parallel.  Moreover, for wisely chosen data partitions, the memory budget of each NUFFT can be
      explicitly controlled and capped to a maximal value.  This allows one to perform out-of-core NUFFTs with
      (theoretically) no complexity overhead w.r.t a single large NUFFT. (See note below however.)

      Chunking of the type-3 NUFFT is activated by passing ``chunked=True`` to :py:meth:`~pyxu.operator.NUFFT.type3`
      (together with ``parallel=True`` for parallel computations).  Finally, :py:meth:`~pyxu.operator.NUFFT.auto_chunk`
      can be used to compute a good partition of the X/Z-domains.

      .. admonition:: Hint

         The current implementation of the chunked type-3 NUFFT has computational complexity:

         .. math::

            \mathcal{O}\left(
                \sum_{p,q=1}^{P}\Pi_{i=1}^{d} X^{(p)}_{i}Z^{(q)}_{i}
                \sum_{i=1}^{d} \log(X^{(p)}_{i} Z^{(q)}_{i})
                +
                (M_p + N_q)|\log(\varepsilon)|^{d}
            \right),

         where

         .. math::

            \begin{align*}
                X^{(p)}_{i} & = \max_{j\in \mathcal{M}_{p}}|(\mathbf{x}_{j})_{i}|, \quad i = \{1,\ldots,d\} \\
                Z^{(p)}_{i} & = \max_{k\in \mathcal{N}_{q}}|(\mathbf{z}_k)_{i}|, \quad i = \{1,\ldots,d\}
            \end{align*}


         and :math:`M_{p}, N_{q}` denote the number of elements in the sets :math:`\mathcal{M}_{p}, \mathcal{N}_{q}`,
         respectively.

         For perfectly balanced and uniform chunks (i.e. :math:`M_{p}=M/P`, :math:`N_{q}=N/Q`, :math:`X^{(p)}_{i} =
         X_{i}/P`, :math:`Z^{(q)}_{i} = Z_{i}/Q` and :math:`P=Q`) the complexity reduces to

         .. math::

            \mathcal{O}\left(
                \Pi_{i=1}^{d} X_{i}Z_{i}\sum_{i=1}^{d}\log(X_{i}Z_{i})
                +
                P(M + N)|\log(\varepsilon)|^{d}
            \right).

         This shows that the spreading/interpolation is :math:`P` times more expensive than in the non-chunked case.
         This overhead is usually acceptable if the number of chunks remains small.  With explicit control on the
         spreading/interpolation steps (which the Python interface to the FINUFFT backend does not currently offer), the
         spreading/interpolation overhead can be removed so as to obtain a computational complexity on par to that of
         the non-chunked type-3 NUFFT.  This will be implemented in a future release.

    * **Backend.**
      The NUFFT transforms are computed via Python wrappers to `FINUFFT <https://github.com/flatironinstitute/finufft>`_
      and `cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_.  (See also [FINUFFT]_ and [cuFINUFFT]_.) These
      librairies perform the expensive spreading/interpolation between nonuniform points and the fine grid via the
      "exponential of semicircle" kernel.

    * **Optional Parameters.**
      [cu]FINUFFT exposes many optional parameters to adjust the performance of the algorithms, change the output
      format, or provide debug/timing information.  While the default options are sensible for most setups, advanced
      users may overwrite them via the ``kwargs`` parameter of :py:meth:`~pyxu.operator.NUFFT.type1`,
      :py:meth:`~pyxu.operator.NUFFT.type2`, and :py:meth:`~pyxu.operator.NUFFT.type3`.  See the `guru interface
      <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ from FINUFFT and its `companion page
      <https://finufft.readthedocs.io/en/latest/opts.html#options-parameters>`_ for details.

      .. admonition:: Hint

         FINUFFT exposes a ``dtype`` keyword to control the precision (single or double) at which transforms are
         performed.  This parameter is ignored by :py:class:`~pyxu.operator.NUFFT`: use the context manager
         :py:class:`~pyxu.runtime.Precision` to control floating point precision.

      .. admonition:: Hint

         The NUFFT is performed in **batches of (n_trans,)**, where `n_trans` denotes the number of simultaneous
         transforms requested.  (See the ``n_trans`` parameter of `finufft.Plan
         <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.)

         Good performance is obtained when each batch fits easily in memory. This recommendation also applies to Dask
         inputs which are re-chunked internally to be `n_trans`-compliant.

         This parameter also affects performance of the ``eps=0`` case: increasing ``n_trans`` may improve performance
         when doing several transforms in parallel.

    .. Warning::

       Since FINUFFT plans cannot be shared among different processes, this class is **only compatible** with Dask's
       thread-based schedulers, i.e.:

       * ``scheduler='thread'``
       * ``scheduler='synchronous'``
       * ``distributed.Client(processes=False)``

       Batches are hence processed serially.  (Each batch is multi-threaded however.)

    .. [#] :math:`\varepsilon= 0` means that no approximation is performed: the exponential sums
           are naively computed by direct evaluation.

    See Also
    --------
    :py:class:`~pyxu.operator.FFT`
    """
    # The goal of this wrapper class is to sanitize __init__() inputs.

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)

    @staticmethod
    @pxrt.enforce_precision(i="x", o=False, allow_None=False)
    def type1(
        x: pxt.NDArray,
        N: typ.Union[pxt.Integer, tuple[pxt.Integer, ...]],
        isign: SignT = sign_default,
        eps: pxt.Real = eps_default,
        real: bool = False,
        plan_fw: bool = True,
        plan_bw: bool = True,
        enable_warnings: bool = True,
        **kwargs,
    ) -> pxt.OpT:
        r"""
        Type-1 NUFFT (non-uniform to uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in [-\pi,\pi)^{d}`.
        N: Integer, :py:class:`tuple` ( :py:attr:`~pyxu.info.ptype.Integer` )
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.

            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1, -1
            Sign :math:`\sigma` of the transform.
        eps: Real
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential sum using a Numba
            JIT-compiled kernel.
        real: bool
            If ``True``, assumes :py:func:`~pyxu.operator.NUFFT.apply` takes (..., M) inputs in :math:`\mathbb{R}^{M}`.

            If ``False``, then :py:func:`~pyxu.operator.NUFFT.apply` takes (..., 2M) inputs, i.e. :math:`\mathbb{C}^{M}`
            vectors viewed as bijections with :math:`\mathbb{R}^{2M}`.
        plan_fw/bw: bool
            If ``True``, allocate FINUFFT resources to do the forward (fw) and/or backward (bw) transform.  These are
            advanced options: use them with care.  Some public methods in the :py:class:`~pyxu.abc.LinOp` interface may
            not work if fw/bw transforms are disabled.  These options only take effect if ``eps > 0``.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.) Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: OpT
            (2N.prod(), M) or (2N.prod(), 2M) type-1 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pyxu.operator as pxo
           import pyxu.runtime as pxrt
           import pyxu.util as pxu

           rng = np.random.default_rng(0)
           D, M, N = 2, 200, 5  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)

           with pxrt.Precision(pxrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x.
               # Its precision is controlled by the context manager.
               N_trans = 5
               A = pxo.NUFFT.type1(
                       x, N,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-3,
                   )

               # Pyxu operators only support real inputs/outputs, so we use the functions
               # pxu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr =        rng.normal(size=(3, N_trans, M)) \
                     + 1j * rng.normal(size=(3, N_trans, M))
               A_out_fw = pxu.view_as_complex(A.apply(pxu.view_as_real(arr)))
               A_out_bw = pxu.view_as_complex(A.adjoint(pxu.view_as_real(A_out_fw)))
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x,
            N=N,
            isign=isign,
            eps=eps,
            real_in=real,
            real_out=False,
            plan_fw=plan_fw,
            plan_bw=plan_bw,
            enable_warnings=enable_warnings,
            **kwargs,
        )
        return _NUFFT1(**init_kwargs)

    @staticmethod
    @pxrt.enforce_precision(i="x", o=False, allow_None=False)
    def type2(
        x: pxt.NDArray,
        N: typ.Union[pxt.Integer, tuple[pxt.Integer, ...]],
        isign: SignT = sign_default,
        eps: pxt.Real = eps_default,
        real: bool = False,
        plan_fw: bool = True,
        plan_bw: bool = True,
        enable_warnings: bool = True,
        **kwargs,
    ) -> pxt.OpT:
        r"""
        Type-2 NUFFT (uniform to non-uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in [-\pi,\pi]^{d}`.
        N: Integer, :py:class:`tuple` ( :py:attr:`~pyxu.info.ptype.Integer` )
            ([d],) mesh size in each dimension :math:`(N_1, \ldots, N_d)`.

            If `N` is an integer, then the mesh is assumed to have the same size in each dimension.
        isign: 1, -1
            Sign :math:`\sigma` of the transform.
        eps: Real
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential sum using a Numba
            JIT-compiled kernel.
        real: bool
            If ``True``, assumes :py:func:`~pyxu.operator.NUFFT.apply` takes (..., N.prod()) inputs in
            :math:`\mathbb{R}^{N}`.

            If ``False``, then :py:func:`~pyxu.operator.NUFFT.apply` takes (..., 2N.prod()) inputs, i.e.
            :math:`\mathbb{C}^{N}` vectors viewed as bijections with :math:`\mathbb{R}^{2N}`.
        plan_fw/bw: bool
            If ``True``, allocate FINUFFT resources to do the forward (fw) and/or backward (bw) transform.  These are
            advanced options: use them with care.  Some public methods in the :py:class:`~pyxu.abc.LinOp` interface may
            not work if fw/bw transforms are disabled.  These options only take effect if ``eps > 0``.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.) Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: OpT
            (2M, N.prod()) or (2M, 2N.prod()) type-2 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pyxu.operator as pxo
           import pyxu.runtime as pxrt
           import pyxu.util as pxu

           rng = np.random.default_rng(0)
           D, M, N = 2, 200, 5  # D denotes the dimension of the data
           N_full = (N,) * D
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)

           with pxrt.Precision(pxrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x.
               # Its precision is controlled by the context manager.
               N_trans = 5
               A = pxo.NUFFT.type2(
                       x, N,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-3,
                   )

               # Pyxu operators only support real inputs/outputs, so we use the functions
               # pxu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr = np.reshape(
                          rng.normal(size=(3, N_trans, *N_full))
                   + 1j * rng.normal(size=(3, N_trans, *N_full)),
                   (3, N_trans, -1),
               )
               A_out_fw = pxu.view_as_complex(A.apply(pxu.view_as_real(arr)))
               A_out_bw = pxu.view_as_complex(A.adjoint(pxu.view_as_real(A_out_fw)))
        """
        init_kwargs = _NUFFT1._sanitize_init_kwargs(
            x=x,
            N=N,
            isign=-isign,
            eps=eps,
            real_in=False,
            real_out=real,
            plan_fw=plan_bw,  # note the reversal
            plan_bw=plan_fw,  # here
            enable_warnings=enable_warnings,
            **kwargs,
        )
        op_t1 = _NUFFT1(**init_kwargs)
        op_t2 = op_t1.T
        op_t2._name = "_NUFFT2"

        # not strictly necessary, but users will probably want to access it.
        op_t2.params = types.MethodType(_NUFFT1.params, op_t1)
        op_t2.mesh = types.MethodType(_NUFFT1.mesh, op_t1)
        return op_t2

    @staticmethod
    @pxrt.enforce_precision(i=("x", "z"), o=False, allow_None=False)
    def type3(
        x: pxt.NDArray,
        z: pxt.NDArray,
        isign: SignT = sign_default,
        eps: pxt.Real = eps_default,
        real: bool = False,
        plan_fw: bool = True,
        plan_bw: bool = True,
        enable_warnings: bool = True,
        chunked: bool = False,
        parallel: bool = False,
        **kwargs,
    ) -> pxt.OpT:
        r"""
        Type-3 NUFFT (non-uniform to non-uniform).

        Parameters
        ----------
        x: NDArray
            (M, [d]) d-dimensional sample points :math:`\mathbf{x}_{j} \in \mathbb{R}^{d}`.
        z: NDArray
            (N, [d]) d-dimensional query points :math:`\mathbf{z}_{k} \in \mathbb{R}^{d}`.
        isign: 1, -1
            Sign :math:`\sigma` of the transform.
        eps: Real
            Requested relative accuracy :math:`\varepsilon \geq 0`.

            If ``eps=0``, the transform is computed exactly via direct evaluation of the exponential sum using a Numba
            JIT-compiled kernel.
        real: bool
            If ``True``, assumes :py:func:`~pyxu.operator.NUFFT.apply` takes (..., M) inputs in :math:`\mathbb{R}^{M}`.

            If ``False``, then :py:func:`~pyxu.operator.NUFFT.apply` takes (..., 2M) inputs, i.e. :math:`\mathbb{C}^{M}`
            vectors viewed as bijections with :math:`\mathbb{R}^{2M}`.
        plan_fw/bw: bool
            If ``True``, allocate FINUFFT resources to do the forward (fw) and/or backward (bw) transform.  These are
            advanced options: use them with care.  Some public methods in the :py:class:`~pyxu.abc.LinOp` interface may
            not work if fw/bw transforms are disabled.  These options only take effect if ``eps > 0``.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        chunked: bool
            If ``True``, the transform is performed in small chunks. (See Notes for details.)
        parallel: bool
            This option only applies to chunked transforms.  If ``True``, evaluate chunks in parallel.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            (Illegal keywords are dropped silently.) Most useful are ``n_trans``, ``nthreads`` and ``debug``.

        Returns
        -------
        op: OpT
            (2N, M) or (2N, 2M) type-3 NUFFT.

        Examples
        --------

        .. code-block:: python3

           import numpy as np
           import pyxu.operator as pxo
           import pyxu.runtime as pxrt
           import pyxu.util as pxu

           rng = np.random.default_rng(0)
           D, M, N = 3, 200, 5  # D denotes the dimension of the data
           x = rng.normal(size=(M, D)) + 2000  # Poorly-centered data
           z = rng.normal(size=(N, D))
           with pxrt.Precision(pxrt.Width.SINGLE):
               # The NUFFT dimension (1/2/3) is inferred from the trailing dimension of x/z.
               # Its precision is controlled by the context manager.
               N_trans = 20
               A = pxo.NUFFT.type3(
                       x, z,
                       n_trans=N_trans,
                       isign=-1,
                       eps=1e-6,
                    )

               # Pyxu operators only support real inputs/outputs, so we use the functions
               # pxu.view_as_[complex/real] to interpret complex arrays as real arrays (and
               # vice-versa).
               arr =        rng.normal(size=(3, N_trans, M)) \
                     + 1j * rng.normal(size=(3, N_trans, M))
               A_out_fw = pxu.view_as_complex(A.apply(pxu.view_as_real(arr)))
               A_out_bw = pxu.view_as_complex(A.adjoint(pxu.view_as_real(A_out_fw)))

        .. rubric:: Notes (chunked-transform)

        * An extra initialization step is required before using a chunked-transform:

          .. code-block:: python3

             A = pxl.NUFFT.type3(
                     x, z
                     isign
                     chunked=True,   # with chunk specified
                     parallel=True,  # for extra speed (chunked-only)
                     **finufft_kwargs,
                  )
             x_chunks, z_chunks = A.auto_chunk()  # auto-determine a good x/z chunking strategy
             A.allocate(x_chunks, z_chunks)  # apply the chunking strategy.

          :py:meth:`~pyxu.operator.NUFFT.auto_chunk` is a helper method to auto-determine a good chunking strategy.

          Its runtime is significant when the number of sub-problems grows large. (1000+) In these contexts, assuming a
          good-enough x/z-split is known in advance, users can directly supply them to
          :py:meth:`~pyxu.operator.NUFFT.allocate`.

        * :py:func:`~pyxu.operator.NUFFT.apply` / :py:func:`~pyxu.operator.NUFFT.adjoint` runtime is minimized when x/z
          are well-ordered, i.e.  when sub-problems can sub-sample inputs to :py:func:`~pyxu.operator.NUFFT.apply` /
          :py:func:`~pyxu.operator.NUFFT.adjoint` via slice operators.

          To reduce runtime of chunked transforms, :py:meth:`~pyxu.operator.NUFFT.allocate` automatically re-orders x/z
          when appropriate.

          The side-effect is the cost of a permutation before/after calls to :py:func:`~pyxu.operator.NUFFT.apply` /
          :py:func:`~pyxu.operator.NUFFT.adjoint`.  This cost becomes significant when the number of non-uniform points
          x/z is large. (> 1e6)

          To avoid paying the re-ordering cost at each transform, it is recommended to supply x/z and apply/adjoint
          inputs in the "correct" order from the start.

          A good re-ordering is computed automatically by :py:meth:`~pyxu.operator.NUFFT.allocate` and can be used to
          initialize a new chunked-transform with better runtime properties as such:

          .. code-block:: python3

             ### Initialize a chunked transform (1st try; as above)
             A = pxl.NUFFT.type3(
                     x, z
                     isign
                     chunked=True,
                     parallel=True,
                     **finufft_kwargs,
                  )
             x_chunks, z_chunks = A.auto_chunk()  # auto-determine a good x/z chunking strategy
             A.allocate(x_chunks, z_chunks)  # will raise warning if bad x/z-order detected


             ### Now initialize a better transform (2nd try)
             x_idx, x_chunks = A.order("x")  # get a good x-ordering
             z_idx, z_chunks = A.order("z")  # get a good z-ordering
             A = pxl.NUFFT.type3(
                     x[x_idx], z[z_idx]  # re-order x/z accordingly
                     ...                 # same as before
                  )
             A.allocate(x_chunks, z_chunks)  # skip auto-chunking and apply
                                             # optimal x/z_chunks provided.

        See Also
        --------
        :py:meth:`~pyxu.operator.NUFFT.auto_chunk`,
        :py:meth:`~pyxu.operator.NUFFT.allocate`,
        :py:meth:`~pyxu.operator.NUFFT.diagnostic_plot`,
        :py:meth:`~pyxu.operator.NUFFT.stats`
        """
        init_kwargs = _NUFFT3._sanitize_init_kwargs(
            x=x,
            z=z,
            isign=isign,
            eps=eps,
            real=real,
            plan_fw=plan_fw,
            plan_bw=plan_bw,
            enable_warnings=enable_warnings,
            chunked=chunked,
            parallel=parallel,
            **kwargs,
        )

        if chunked := init_kwargs.pop("chunked", False):
            klass = _NUFFT3_chunked
        else:
            klass = _NUFFT3
            for k in [  # kwargs only valid for chunked-transforms
                "parallel",
            ]:
                init_kwargs.pop(k, None)

        op = klass(**init_kwargs)
        return op

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            - **Type 1 and 3:**
                * (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}` (real transform).
                * (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.  (See
                  :py:func:`~pyxu.util.view_as_real`.)
            - **Type 2:**
                * (...,  N.prod()) input weights :math:`\mathbf{u} \in \mathbb{R}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                  (real transform).
                * (..., 2N.prod()) input weights :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                  viewed as a real array.  (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                viewed as a real array.  (See :py:func:`~pyxu.util.view_as_real`.)
            - **Type 2:**
                (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.  (See
                :py:func:`~pyxu.util.view_as_real`.)
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.  (See
                :py:func:`~pyxu.util.view_as_real`.)
        """
        raise NotImplementedError

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            - **Type 1:**
                (..., 2N.prod()) output weights :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                viewed as a real array.  (See :py:func:`~pyxu.util.view_as_real`.)
            - **Type 2:**
                (..., 2M) output weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.  (See
                :py:func:`~pyxu.util.view_as_real`.)
            - **Type 3:**
                (..., 2N) output weights :math:`\mathbf{v} \in \mathbb{C}^{N}` viewed as a real array.  (See
                :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            - **Type 1 and 3:**
                * (...,  M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}` (real transform).
                * (..., 2M) input weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array.  (See
                  :py:func:`~pyxu.util.view_as_real`.)
            - **Type 2:**
                * (...,  N.prod()) input weights :math:`\mathbf{u} \in \mathbb{R}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                  (real transform).
                * (..., 2N.prod()) input weights :math:`\mathbf{u} \in \mathbb{C}^{\mathcal{I}_{N_1, \ldots, N_d}}`
                  viewed as a real array.  (See :py:func:`~pyxu.util.view_as_real`.)
        """
        raise NotImplementedError

    @staticmethod
    def _as_canonical_coordinate(x: pxt.NDArray) -> pxt.NDArray:
        if (N_dim := x.ndim) == 1:
            x = x.reshape((-1, 1))
        elif N_dim == 2:
            assert 1 <= x.shape[-1] <= 3, "Only (1,2,3)-D transforms supported."
        else:
            raise ValueError(f"Expected 1D/2D array, got {N_dim}-D array.")
        return x

    @staticmethod
    def _as_canonical_mode(N) -> pxt.NDArrayShape:
        N = pxu.as_canonical_shape(N)
        assert all(_ > 0 for _ in N)
        assert 1 <= len(N) <= 3, "Only (1,2,3)-D transforms supported."
        return N

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        # check init() params + put in standardized form
        raise NotImplementedError

    @staticmethod
    def _plan_fw(**kwargs) -> finufft_Plan:
        # create plan and set points
        raise NotImplementedError

    def _fw(self, arr: pxt.NDArray) -> pxt.NDArray:
        # apply forward operator.
        # input: (n_trans, Q1) complex-valued
        # output: (n_trans, Q2) complex-valued
        raise NotImplementedError

    @staticmethod
    def _plan_bw(**kwargs) -> finufft_Plan:
        # create plan and set points
        raise NotImplementedError

    def _bw(self, arr: pxt.NDArray) -> pxt.NDArray:
        # apply backward operator.
        # input: (n_trans, Q2) complex-valued
        # output: (n_trans, Q1) complex-valued
        raise NotImplementedError

    def _warn_cast(self, arr: pxt.NDArray) -> pxt.NDArray:
        W = pxrt.Width  # shorthand
        x_width = W(self._x.dtype)
        if (a_width := W(arr.dtype)) != x_width:
            if self._enable_warnings:
                msg = f"NUFFT was configured to run with {x_width.value} inputs, but provided {a_width.value} inputs."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=x_width.value)
        else:
            out = arr
        return out

    @staticmethod
    def _preprocess(
        arr: pxt.NDArray,
        n_trans: pxt.Integer,
        dim_out: pxt.Integer,
    ):
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # arr: pxt.NDArray
        #     (..., N1) complex-valued input of [apply|adjoint]().
        # n_trans: pxt.Integer
        #     n_trans parameter given to finufft.Plan()
        # dim_out: pxt.Integer
        #     Trailing dimension [apply|adjoint](arr) should have.
        #
        # Returns
        # -------
        # x: pxt.NDArray
        #     (N_blk, n_trans, N1) complex-valued blocks to input to [_fw|_bw](), suitably augmented as needed.
        # N: pxt.Integer
        #     Amount of "valid" data to extract from [_fw|_bw](). {For _postprocess()}
        # sh_out: pxt.NDArrayShape
        #     Shape [apply|adjoint](arr) should have. {For _postprocess()}
        sh_out = arr.shape[:-1] + (dim_out,)
        if arr.ndim == 1:
            arr = arr.reshape((1, -1))
        N, dim_in = np.prod(arr.shape[:-1]), arr.shape[-1]

        N_blk, r = divmod(N, n_trans)
        N_blk += 1 if (r > 0) else 0
        if r == 0:
            x = arr
        else:
            xp = pxu.get_array_module(arr)
            x = xp.concatenate(
                [
                    arr.reshape((N, dim_in)),
                    xp.zeros((n_trans - r, dim_in), dtype=arr.dtype),
                ],
                axis=0,
            )
        x = x.reshape((N_blk, n_trans, dim_in))
        return x, N, sh_out

    @staticmethod
    def _postprocess(
        blks: list[pxt.NDArray],
        N: pxt.Integer,
        sh_out: pxt.NDArrayShape,
    ) -> pxt.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # blks: list[pxt.NDArray]
        #     (N_blk,) complex-valued outputs of [_fw|_bw]().
        # N: pxt.Integer
        #     Amount of "valid" data to extract from [_fw|_bw]()
        # sh_out: pxt.NDArrayShape
        #     Shape [apply|adjoint](arr) should have.
        xp = pxu.get_array_module(blks[0])
        return xp.concatenate(blks, axis=0)[:N].reshape(sh_out)

    def ascomplexarray(
        self,
        xp: pxt.ArrayModule = pxd.NDArrayInfo.NUMPY.module(),
        dtype: pxt.DType = None,
    ) -> pxt.NDArray:
        r"""
        Matrix representation (complex-valued) of the linear operator.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output.
        dtype: DType
            Optional (complex-valued) type of the array.

        Returns
        -------
        A: NDArray
            Array representation of the operator (NUFFT type-dependant).

            - **Type 1:** (N.prod(), M)
            - **Type 2:** (M, N.prod())
            - **Type 3:** (N, M)
        """
        raise NotImplementedError

    def mesh(
        self,
        xp: pxt.ArrayModule = pxd.NDArrayInfo.NUMPY.module(),
        dtype: pxt.DType = None,
        scale: str = "unit",
        upsampled: bool = False,
    ) -> pxt.NDArray:
        r"""
        For type-1/2 NUFFT: compute the transform's meshgrid :math:`\mathcal{I}_{N_{1} \times \cdots \times N_{d}} =
        \mathcal{I}_{N_{1}} \times \cdots \times \mathcal{I}_{N_{d}}`.

        For type-3 NUFFT: compute the (shifted) meshgrid used for internal FFT computations.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output.
        dtype: DType
            Optional type of the array.
        scale: str
            Grid scale. Options are:

            - **Type1 and 2:**
                * ``unit``, i.e. :math:`\mathcal{I} = [[-N_{d}//2, \ldots, (N_{d}-1)//2 + 1))^{d}`
                * ``source``, i.e. :math:`\mathcal{I} \subset [-\pi, \pi)^{d}`
            - **Type 3:**
                * ``unit``, i.e. :math:`\mathcal{I} = [[-N_{d}//2, \ldots, (N_{d}-1)//2 + 1))^{d}`

                * ``source``, i.e. :math:`\mathcal{I}_{\text{source}} \subset x^{c} + [-X_{d}, X_{d})^{d}`

                * ``target``, i.e. :math:`\mathcal{I}_{\text{target}} \subset z^{c} + [-Z_{d}, Z_{d})^{d}`,

              where :math:`x^{c}`, :math:`z^{c}` denote the source/target centroids, and :math:`X`, :math:`Z` the
              source/target half-widths.

        upsampled: bool
            Use the upsampled meshgrid.  (See [FINUFFT]_ for details.)

        Returns
        -------
        grid: NDArray
            (N1, ..., Nd, d) grid.

        Examples
        --------
        .. code-block:: python3

           import numpy as np
           import pyxu.operator as pxo

           rng = np.random.default_rng(0)
           D, M, N = 1, 2, 3  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
           A = pxo.NUFFT.type1(
               x, N,
               isign=-1,
               eps=1e-3
           )
           A.mesh()  # array([[-1.],
                     #        [ 0.],
                     #        [ 1.]])
        """
        raise NotImplementedError

    def plot_kernel(self, ax=None, **kwargs):
        """
        Plot the spreading/interpolation kernel (along each dimension) on its support.

        Parameters
        ----------
        ax: :py:class:`~matplotlib.axes.Axes`
            Axes to draw on.  If :py:obj:`None`, a new axes is used.
        **kwargs
            Keyword arguments forwarded to :py:meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        ax : :py:class:`~matplotlib.axes.Axes`

        Examples
        --------

        .. plot::

           import numpy as np
           import pyxu.operator as pxo
           import matplotlib.pyplot as plt

           rng = np.random.default_rng(0)
           D, M, N = 1, 2, 3  # D denotes the dimension of the data
           x = np.fmod(rng.normal(size=(M, D)), 2 * np.pi)
           A = pxo.NUFFT.type1(
               x, N,
               isign=-1,
               eps=1e-9
           )
           A.plot_kernel()
           plt.show()

        Notes
        -----
        Requires `Matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        plt = pxu.import_module("matplotlib.pyplot")
        if ax is None:
            _, ax = plt.subplots()

        width = self._kernel_width()
        beta = self._kernel_beta()
        N = self._fft_shape()

        N_sample = 100
        z = np.linspace(-1, 1, N_sample)
        for d, n in zip(range(self._D), N):
            alpha = np.pi * width / n
            x = z / alpha
            phi = ES_kernel(x, beta)
            ax.plot(x, phi, label=rf"$\phi_{d}$", **kwargs)

        if self._D > 1:
            ax.legend()
        return ax

    def params(self) -> collections.namedtuple:
        r"""
        Compute internal parameters of the [FINUFFT]_ implementation.

        Returns
        -------
        p: :py:func:`~collections.namedtuple`
            Internal parameters of the FINUFFT implementation, with fields:

            * upsample_factor: :py:attr:`~pyxu.info.ptype.Real`
                FFT upsampling factor > 1
            * kernel_width: :py:attr:`~pyxu.info.ptype.Integer`
                Width of the spreading/interpolation kernels (in number of samples).
            * kernel_beta: :py:attr:`~pyxu.info.ptype.Real`
                Kernel decay parameter :math:`\beta > 0`.
            * fft_shape: (d,) [:py:attr:`~pyxu.info.ptype.Integer`]
                Size of the D-dimensional FFT(s) performed internally.
            * dilation_factor: (d,) [:py:attr:`~pyxu.info.ptype.Real`]
                Dilation factor(s) :math:`\gamma_{d}`. (Type-3 only)

        Notes
        -----
        When called from a chunked type-3 transform, :py:meth:`~pyxu.operator.NUFFT.params` returns parameters of the
        equivalent monolithic type-3 transform.  The monolithic transform is seldom instantiable due to its large memory
        requirements.  This method can hence be used to estimate the memory savings induced by chunking.
        """
        if self._direct_eval:
            p = None
        else:
            FINUFFT_PARAMS = collections.namedtuple(
                "finufft_params",
                [
                    "upsample_factor",
                    "kernel_width",
                    "kernel_beta",
                    "fft_shape",
                    "dilation_factor",
                ],
            )
            p = FINUFFT_PARAMS(
                upsample_factor=self._upsample_factor(),
                kernel_width=self._kernel_width(),
                kernel_beta=self._kernel_beta(),
                fft_shape=self._fft_shape(),
                dilation_factor=self._dilation_factor(),
            )
        return p

    def auto_chunk(
        self,
        max_mem: pxt.Real = 10,
        max_anisotropy: pxt.Real = 5,
    ) -> tuple[list[pxt.NDArray], list[pxt.NDArray]]:
        r"""
        (Only applies to chunked type-3 transforms.)

        Auto-determine chunk indices per domain.

        Use this function if you don't know how to optimally 'cut' x/z manually.

        Parameters
        ----------
        max_mem: Real
            Max FFT memory (MiB) allowed per sub-block. (Default = 10 MiB)
        max_anisotropy: Real
            Max tolerated (normalized) anisotropy ratio >= 1.

            * Setting close to 1 favors cubeoid-shaped partitions of x/z space.
            * Setting large allows x/z-partitions to be highly-rectangular.

        Returns
        -------
        x_chunks: list[NDArray[int]]
            (x_idx[0], ..., x_idx[A-1]) x-coordinate chunk specifier.
            `x_idx[k]` contains indices of `x` which participate in the k-th NUFFT sub-problem.
        z_chunks: list[NDArray[int]]
            (z_idx[0], ..., z_idx[B-1]) z-coordinate chunk specifier.
            `z_idx[k]` contains indices of `z` which participate in the k-th NUFFT sub-problem.

        Notes
        -----
        Chunks are identified by a custom hierarchical clustering method, with main steps:

        1. **Partition the NUFFT domains.**
           Given a maximum FFT memory budget :math:`B>0` and chunk anisotropy :math:`\alpha\geq 1`, partition the
           source/target domains into uniform rectangular cells.  The (half) widths of the source/target cells
           :math:`h_{k}>0` and :math:`\eta_{k}>0` in each dimension :math:`k=\{1,\ldots d\}` are chosen so as to:

           Minimize the total number of partitions:

           .. math::

              N_{c}
              =
              \underbrace{\prod_{k=1}^{d} \frac{X_k}{h_k}}_{\text{Source partition count}}
              \times
              \underbrace{\prod_{k=1}^{d} \frac{Z_k}{\eta_k}}_{\text{Target partition count}}

           subject to:

             (a) .. math::

                    \prod_{k=1}^{d} \eta_k h_k
                    \leq
                    (\pi/2\upsilon)^{d} \frac{B}{\delta \; \texttt{n_trans}},

                 where

                   * :math:`\upsilon` denotes the NUFFT's grid upsampling factor,
                   * :math:`\delta` the number of bytes occupied by a complex number,
                   * :math:`\texttt{n_trans}` the number of simultaneous transforms performed.
             (b) .. math::

                    \begin{align*}
                        h_{k} & \leq X_{k}, \quad k=\{1,\ldots,d\} \\
                        \eta_{k} & \leq Z_{k}, \quad k=\{1,\ldots,d\}
                    \end{align*}
             (c) .. math::

                    N_{c} \ge 1.
             (d) .. math::

                    \begin{align*}
                        \frac{1}{\alpha} & \leq \frac{h_{k}}{h_{j}} \frac{X_{j}}{X_{k}} \leq \alpha, \quad k \ne j, \\
                        \frac{1}{\alpha} & \leq \frac{\eta_{k}}{\eta_{j}} \frac{Z_{j}}{Z_{k}} \leq \alpha, \quad k \ne j.
                    \end{align*}
             (e) .. math::

                    \begin{align*}
                        \frac{1}{\alpha} & \leq \frac{h_{k}}{\eta_{j}} \frac{Z_{j}}{X_{k}} \leq \alpha, \quad (j, k) = \{1,\ldots,d\}^{2}.
                    \end{align*}

           Constraint (a) ensures type-3 NUFFTs performed in each sub-problem do not exceed the FFT memory budget.
           Constraints (b-c) ensure that partitions are non-degenerate/non-trivial respectively.
           Constraints (d-e) limit the anisotropy of the partitioning cells in each domain and across domains.
        2. **Data-independent Chunking.**
           Chunk the data by assigning non-uniform samples to their enclosing cell in the partition.
           Empty partitions are dropped.
        3. **Data-dependent Chunk Fusion.**
           Since (1) is data-independent, data chunks obtained in (2) may split clusters among adjacent chunks, which is
           undesirable.  Clusters whose joint-spread is small enough are hence fused hierarchically.

        .. Warning::

           This procedure yields a small number of memory-capped and well-separated data chunks in source/target
           domains.  However, it may result in unbalanced chunks, with some chunks containing significantly more
           data-points than others.  FINUFFT mitigates the unbalanced-chunk problem by spawning multiple threads to
           process dense clusters.

        See Also
        --------
        :py:meth:`~pyxu.operator.NUFFT.allocate`,
        :py:meth:`~pyxu.operator.NUFFT.diagnostic_plot`,
        :py:meth:`~pyxu.operator.NUFFT.stats`
        """
        raise NotImplementedError

    def allocate(
        self,
        x_chunks: list[typ.Union[pxt.NDArray, slice]],
        z_chunks: list[typ.Union[pxt.NDArray, slice]],
        direct_eval_threshold: pxt.Integer = 10_000,
    ):
        """
        (Only applies to chunked type-3 transforms.)

        Allocate NUFFT sub-problems based on chunk specification.

        Parameters
        ----------
        x_chunks: list[NDArray[int] | slice]
            (x_idx[0], ..., x_idx[A-1]) x-coordinate chunk specifier.
            `x_idx[k]` contains indices of `x` which participate in the k-th NUFFT sub-problem.
        z_chunks: list[NDArray[int] | slice]
            (z_idx[0], ..., z_idx[B-1]) z-coordinate chunk specifier.
            `z_idx[k]` contains indices of `z` which participate in the k-th NUFFT sub-problem.
        direct_eval_threshold: Integer
            If provided: lower bound on ``len(x) * len(z)`` below which an NUFFT sub-problem is replaced with
            direct-evaluation (eps=0) for performance reasons.

            (Defaults to 10k as per the `FINUFFT guidelines
            <https://finufft.readthedocs.io/en/latest/#do-i-even-need-a-nufft>`_.)

        See Also
        --------
        :py:meth:`~pyxu.operator.NUFFT.auto_chunk`,
        :py:meth:`~pyxu.operator.NUFFT.diagnostic_plot`,
        :py:meth:`~pyxu.operator.NUFFT.stats`
        """
        raise NotImplementedError

    def diagnostic_plot(self, domain: str):
        r"""
        (Only applies to chunked type-3 transforms.)

        Plot data + tesselation structure for diagnostic purposes.

        Parameters
        ----------
        domain: 'x', 'z'
            Plot x-domain or z-domain data.

        Returns
        -------
        fig: :py:class:`~matplotlib.figure.Figure`
            Diagnostic plot.

        Notes
        -----
        * This method can only be called after :py:meth:`~pyxu.operator.NUFFT.allocate`.
        * This method only works for 2D/3D domains.

        Examples
        --------

        .. plot::

           import numpy as np
           import pyxu.operator as pxo

           rng = np.random.default_rng(2)
           D, M, N = 2, 500, 200
           rnd_points = lambda _: rng.normal(scale=rng.uniform(0.25, 0.5, size=(D,)), size=(_, D))
           rnd_offset = lambda: rng.uniform(-1, 1, size=(D,))
           scale = 20
           x = np.concatenate(
               [
                   rnd_points(M) + rnd_offset() * scale,
                   rnd_points(M) + rnd_offset() * scale,
                   rnd_points(M) + rnd_offset() * scale,
                   rnd_points(M) + rnd_offset() * scale,
                   rnd_points(M) + rnd_offset() * scale,
               ],
               axis=0,
           )
           z = np.concatenate(
               [
                   rnd_points(N) + rnd_offset() * scale,
                   rnd_points(N) + rnd_offset() * scale,
                   rnd_points(N) + rnd_offset() * scale,
                   rnd_points(N) + rnd_offset() * scale,
                   rnd_points(N) + rnd_offset() * scale,
               ],
               axis=0,
           )

           kwargs = dict(
               x=x,
               z=z,
               isign=-1,
               eps=1e-3,
           )
           A = pxo.NUFFT.type3(**kwargs, chunked=True)
           x_chunks, z_chunks = A.auto_chunk(
               max_mem=.1,
               max_anisotropy=1,
           )
           A.allocate(x_chunks, z_chunks)
           fig = A.diagnostic_plot('x')
           fig.show()

        Notes
        -----
        Requires `Matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        raise NotImplementedError

    def stats(self):
        """
        (Only applies to chunked type-3 transforms.)

        Gather internal statistics about a chunked type-3 NUFFT.

        Returns
        -------
        p: :py:func:`~collections.namedtuple`
            Statistics on the NUFFT chunks, with fields:

            * blk_count: :py:attr:`~pyxu.info.ptype.Integer`
                Number of NUFFT chunks.
            * dEval_count: :py:attr:`~pyxu.info.ptype.Integer`
                Number of chunks directly evaluated via the NUDFT.
        """
        raise NotImplementedError

    def _upsample_factor(self) -> pxt.Real:
        raise NotImplementedError

    def _kernel_width(self) -> pxt.Integer:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/spreadinterp.cpp::setup_spreader()
        # [FINUFFT]_
        #     eq   3.2
        #     sect 4.2
        u = self._upsample_factor()
        if np.isclose(u, 2):
            w = np.ceil(-np.log10(self._eps) + 1)
        else:  # 1.25 Consistent with setup_spreader() but not sect 4.2 (safety factor gamma=1 instead of 0.976)
            scale = np.pi * np.sqrt(1 - (1 / u))
            w = np.ceil(-np.log(self._eps) / scale)
        w = max(2, int(w))
        return w

    def _kernel_beta(self) -> pxt.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/spreadinterp.cpp::setup_spreader()
        # [FINUFFT]_
        #     eq   3.2
        #     sect 4.2
        u = self._upsample_factor()
        w = self._kernel_width()
        if np.isclose(u, 2):
            scale = {
                2: 2.20,
                3: 2.26,
                4: 2.38,
            }.get(w, 2.30)
        else:  # 1.25
            gamma = 0.97  # 0.976 in paper
            scale = gamma * np.pi * (1 - (0.5 / u))
        beta = float(scale * w)
        return beta

    def _fft_shape(self) -> pxt.NDArrayShape:
        raise NotImplementedError

    def _dilation_factor(self) -> cabc.Sequence[pxt.Integer]:
        raise NotImplementedError


class _NUFFT1(NUFFT):
    def __init__(self, **kwargs):
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N = kwargs["N"]
        self._x = kwargs["x"]
        self._isign = kwargs["isign"]

        self._eps = kwargs.get("eps")
        self._direct_eval = not (self._eps > 0)
        self._enable_warnings = kwargs.pop("enable_warnings")
        self._real_in = kwargs.pop("real_in")
        self._real_out = kwargs.pop("real_out")
        self._upsampfac = kwargs.get("upsampfac")
        self._n = kwargs.get("n_trans", 1)
        self._modeord = kwargs.get("modeord", 0)
        if self._direct_eval:
            self._plan = None
        else:
            _pfw = kwargs.pop("plan_fw")
            _pbw = kwargs.pop("plan_bw")
            self._plan = dict(
                fw=self._plan_fw(**kwargs) if _pfw else None,
                bw=self._plan_bw(**kwargs) if _pbw else None,
            )

        sh_op = [2 * np.prod(self._N), 2 * self._M]
        sh_op[0] //= 2 if self._real_out else 1
        sh_op[1] //= 2 if self._real_in else 1
        super().__init__(shape=sh_op)
        self.lipschitz = np.sqrt(self._M * np.prod(self._N))  # analytical upper bound

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype"):
            kwargs.pop(k, None)
        x = kwargs["x"] = pxu.compute(cls._as_canonical_coordinate(kwargs["x"]))
        N = kwargs["N"] = cls._as_canonical_mode(kwargs["N"])
        kwargs["isign"] = int(np.sign(kwargs["isign"]))
        kwargs["eps"] = float(kwargs["eps"])
        kwargs["real_in"] = bool(kwargs["real_in"])
        kwargs["real_out"] = bool(kwargs["real_out"])
        kwargs["plan_fw"] = bool(kwargs["plan_fw"])
        kwargs["plan_bw"] = bool(kwargs["plan_bw"])
        kwargs["enable_warnings"] = bool(kwargs["enable_warnings"])
        if (D := x.shape[-1]) == len(N):
            pass
        elif len(N) == 1:
            kwargs["N"] = N * D
        else:
            raise ValueError("x vs. N: dimensionality mis-match.")
        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft_Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=N,
            dtype=pxrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], x.T[:N_dim])))
        return plan

    def _fw(self, arr: pxt.NDArray) -> pxt.NDArray:
        if self._direct_eval:
            # Computing the target each time is wasteful (in comparison to the type-3 case where it
            # is implicit.) We are ok with this since relying on NUDFT is a failsafe.
            target = self.mesh(
                xp=pxd.NDArrayInfo.from_obj(arr).module(),
                dtype=self._x.dtype,
                scale="unit",
                upsampled=False,
            ).reshape((-1, self._D))

            out = _nudft(
                weight=arr,
                source=self._x,
                target=target,
                isign=self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N1,..., Nd)
        return out.reshape((self._n, np.prod(self._N)))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft_Plan:
        kwargs = kwargs.copy()
        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=2,
            n_modes_or_dim=N,
            dtype=pxrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(**dict(zip("xyz"[:N_dim], x.T[:N_dim])))
        return plan

    def _bw(self, arr: pxt.NDArray) -> pxt.NDArray:
        if self._direct_eval:
            # Computing the target each time is wasteful (in comparison to the type-3 case where it
            # is implicit.) We are ok with this since relying on NUDFT is a failsafe.
            target = self.mesh(
                xp=pxd.NDArrayInfo.from_obj(arr).module(),
                dtype=self._x.dtype,
                scale="unit",
                upsampled=False,
            ).reshape((-1, self._D))

            out = _nudft(
                weight=arr,
                source=target,
                target=self._x,
                isign=-self._isign,
                dtype=arr.dtype,
            )
        else:
            arr = arr.reshape((self._n, *self._N))
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["bw"].execute(arr)  # ([n_trans], N1, ..., Nd) -> ([n_trans], M)
        return out.reshape((self._n, self._M))

    @pxrt.enforce_precision("arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)
        if self._real_in:
            r_width = pxrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pxu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, np.prod(self._N))
        blks = _dask_zip(
            func=(self._fw,) * len(data),
            data=data,
            out_shape=((self._n, np.prod(self._N)),) * len(data),
            out_dtype=(data.dtype,) * len(data),
            parallel=False,
        )
        out = self._postprocess(blks, N, sh)

        if self._real_out:
            return out.real
        else:
            return pxu.view_as_real(out)

    @pxrt.enforce_precision("arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)
        if self._real_out:
            r_width = pxrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pxu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._M)
        blks = _dask_zip(
            func=(self._bw,) * len(data),
            data=data,
            out_shape=((self._n, self._M),) * len(data),
            out_dtype=(data.dtype,) * len(data),
            parallel=False,
        )
        out = self._postprocess(blks, N, sh)

        if self._real_in:
            return out.real
        else:
            return pxu.view_as_real(out)

    def ascomplexarray(self, **kwargs) -> pxt.NDArray:
        # compute exact operator (using supported precision/backend)
        xp = pxu.get_array_module(self._x)
        mesh = self.mesh(
            xp=xp,
            dtype=self._x.dtype,
            scale="unit",
            upsampled=False,
        ).reshape((-1, self._D))
        _A = xp.exp(1j * self._isign * mesh @ self._x.T)

        # then comply with **kwargs()
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        c_dtype = kwargs.get("dtype", pxrt.getPrecision().complex.value)
        A = xp.array(pxu.to_NUMPY(_A), dtype=c_dtype)
        return A

    def mesh(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        scale = kwargs.get("scale", "unit")
        upsampled = kwargs.get("upsampled", False)

        N = self._fft_shape() if upsampled else self._N
        if scale == "unit":
            grid = xp.stack(  # (N1, ..., Nd, D)
                xp.meshgrid(
                    *[xp.arange(-(n // 2), (n - 1) // 2 + 1, dtype=dtype) for n in N],
                    indexing="ij",
                ),
                axis=-1,
            )
        elif scale == "source":
            # As per eq. 3.12, the source grid is of the form: 2*pi*l/n, l=0,...,n-1, that is n
            # points over [0, 2* pi[ (or [-pi, pi[ if shifted by pi).
            grid = xp.stack(  # (N1, ..., Nd, D)
                xp.meshgrid(
                    *[xp.linspace(-np.pi, np.pi, num=n, endpoint=False, dtype=dtype) for n in N],
                    indexing="ij",
                ),
                axis=-1,
            )
        else:
            raise NotImplementedError

        if self._modeord == 1:  # FFT-order
            grid = xp.fft.ifftshift(grid, axes=np.arange(len(N)))
        return grid

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        if (r_dtype := kwargs.get("dtype")) is None:
            c_dtype = pxrt.getPrecision().complex.value
        else:
            r_width = pxrt.Width(r_dtype)
            c_dtype = r_width.complex.value
        _A = self.ascomplexarray(xp=xp, dtype=c_dtype)

        A = pxu.view_as_real_mat(
            cmat=_A,
            real_input=self._real_in,
            real_output=self._real_out,
        )
        return A

    def _upsample_factor(self) -> pxt.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::FINUFFT_MAKEPLAN()
        if (u := self._upsampfac) is None:
            precQ = self._eps >= 1e-9
            dimQ = lambda d: self._D == d
            cutoffQ = lambda cutoff: np.prod(self._N) > int(cutoff)
            if precQ and dimQ(1) and cutoffQ(1e7):
                u = 1.25
            elif precQ and dimQ(2) and cutoffQ(3e5):
                u = 1.25
            elif precQ and dimQ(3) and cutoffQ(3e6):
                u = 1.25
            else:
                u = 2
        return u

    def _fft_shape(self) -> pxt.NDArrayShape:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::SET_NF_TYPE12()
        # [FINUFFT]_
        #     sect 3.1.1
        u = self._upsample_factor()
        w = self._kernel_width()
        shape = []
        for n in self._N:
            target = max(int(u * n), 2 * w)
            n_opt = pxu.next_fast_len(target, even=True)
            shape.append(n_opt)
        return tuple(shape)

    def _dilation_factor(self) -> cabc.Sequence[pxt.Integer]:
        # Undefined for type-1
        return None


class _NUFFT3(NUFFT):
    def __init__(self, **kwargs):
        self._M, self._D = kwargs["x"].shape  # Useful constants
        self._N, _ = kwargs["z"].shape
        self._x = kwargs["x"]
        self._z = kwargs["z"]
        self._isign = kwargs["isign"]

        self._eps = kwargs.get("eps")
        self._direct_eval = not (self._eps > 0)
        self._enable_warnings = kwargs.pop("enable_warnings")
        self._real = kwargs.pop("real")
        self._upsampfac = kwargs.get("upsampfac")
        self._n = kwargs.get("n_trans", 1)
        self._modeord = 0  # in case _NUFFT1 methods are called
        if self._direct_eval:
            self._plan = None
        else:
            _pfw = kwargs.pop("plan_fw")
            _pbw = kwargs.pop("plan_bw")
            self._plan = dict(
                fw=self._plan_fw(**kwargs) if _pfw else None,
                bw=self._plan_bw(**kwargs) if _pbw else None,
            )
        sh_op = [2 * self._N, 2 * self._M]
        sh_op[1] //= 2 if self._real else 1
        super().__init__(shape=sh_op)
        self.lipschitz = np.sqrt(self._M * np.prod(self._N))  # analytical upper bound

    @classmethod
    def _sanitize_init_kwargs(cls, **kwargs) -> dict:
        kwargs = kwargs.copy()
        for k in ("nufft_type", "n_modes_or_dim", "dtype", "modeord"):
            kwargs.pop(k, None)

        kwargs["isign"] = int(np.sign(kwargs["isign"]))
        kwargs["eps"] = float(kwargs["eps"])
        kwargs["real"] = bool(kwargs["real"])
        kwargs["plan_fw"] = bool(kwargs["plan_fw"])
        kwargs["plan_bw"] = bool(kwargs["plan_bw"])
        kwargs["enable_warnings"] = bool(kwargs["enable_warnings"])
        kwargs["parallel"] = bool(kwargs["parallel"])
        kwargs["chunked"] = bool(kwargs["chunked"])

        x = kwargs["x"] = cls._as_canonical_coordinate(kwargs["x"])
        z = kwargs["z"] = cls._as_canonical_coordinate(kwargs["z"])
        if not kwargs["chunked"]:
            x = kwargs["x"] = pxu.compute(kwargs["x"])
            z = kwargs["z"] = pxu.compute(kwargs["z"])
        assert x.shape[-1] == z.shape[-1], "x vs. z: dimensionality mis-match."
        assert pxu.get_array_module(x) == pxu.get_array_module(z)

        return kwargs

    @staticmethod
    def _plan_fw(**kwargs) -> finufft_Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pxrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    (*x.T[:N_dim], *z.T[:N_dim]),
                )
            ),
        )
        return plan

    def _fw(self, arr: pxt.NDArray) -> pxt.NDArray:
        if self._direct_eval:
            out = _nudft(
                weight=arr,
                source=self._x,
                target=self._z,
                isign=self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["fw"].execute(arr)  # ([n_trans], M) -> ([n_trans], N)
        return out.reshape((self._n, self._N))

    @staticmethod
    def _plan_bw(**kwargs) -> finufft_Plan:
        kwargs = kwargs.copy()
        x, z = [kwargs.pop(_) for _ in ("x", "z")]
        _, N_dim = x.shape

        plan = finufft.Plan(
            nufft_type=3,
            n_modes_or_dim=N_dim,
            dtype=pxrt.getPrecision().value,
            eps=kwargs.pop("eps"),
            n_trans=kwargs.pop("n_trans", 1),
            isign=-kwargs.pop("isign"),
            **kwargs,
        )
        plan.setpts(
            **dict(
                zip(
                    "xyz"[:N_dim] + "stu"[:N_dim],
                    (*z.T[:N_dim], *x.T[:N_dim]),
                )
            ),
        )
        return plan

    def _bw(self, arr: pxt.NDArray) -> pxt.NDArray:
        if self._direct_eval:
            out = _nudft(
                weight=arr,
                source=self._z,
                target=self._x,
                isign=-self._isign,
                dtype=arr.dtype,
            )
        else:
            if self._n == 1:  # finufft limitation: insists on having no
                arr = arr[0]  # leading-dim if n_trans==1.
            out = self._plan["bw"].execute(arr)  # ([n_trans], N) -> ([n_trans], M)
        return out.reshape((self._n, self._M))

    @pxrt.enforce_precision("arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)
        if self._real:
            r_width = pxrt.Width(arr.dtype)
            arr = arr.astype(r_width.complex.value)
        else:
            arr = pxu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._N)
        blks = _dask_zip(
            func=(self._fw,) * len(data),
            data=data,
            out_shape=((self._n, self._N),) * len(data),
            out_dtype=(data.dtype,) * len(data),
            parallel=False,
        )
        out = self._postprocess(blks, N, sh)

        return pxu.view_as_real(out)

    @pxrt.enforce_precision("arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        arr = self._warn_cast(arr)
        arr = pxu.view_as_complex(arr)

        data, N, sh = self._preprocess(arr, self._n, self._M)
        blks = _dask_zip(
            func=(self._bw,) * len(data),
            data=data,
            out_shape=((self._n, self._M),) * len(data),
            out_dtype=(data.dtype,) * len(data),
            parallel=False,
        )
        out = self._postprocess(blks, N, sh)

        if self._real:
            return out.real
        else:
            return pxu.view_as_real(out)

    def ascomplexarray(self, **kwargs) -> pxt.NDArray:
        # compute exact operator (using supported precision/backend)
        xp = pxu.get_array_module(self._x)
        _A = xp.exp(1j * self._isign * self._z @ self._x.T)

        # then comply with **kwargs()
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        c_dtype = kwargs.get("dtype", pxrt.getPrecision().complex.value)
        A = xp.array(pxu.to_NUMPY(_A), dtype=c_dtype)
        return A

    def mesh(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        scale = kwargs.get("scale", "unit")
        upsampled = True or kwargs.pop("upsampled", True)  # upsampled unsupported for type-3
        kwargs = dict(
            xp=xp,
            dtype=dtype,
            upsampled=upsampled,
        )

        if scale == "unit":
            grid = _NUFFT1.mesh(self, scale="unit", **kwargs)
        else:
            grid = _NUFFT1.mesh(self, scale="source", **kwargs)
            f = lambda _: xp.array(_, dtype=dtype)
            if scale == "source":  # Sect 3.3 Eq 3.18.
                s = f(self._dilation_factor()) * (1 - self._kernel_width() / f(self._fft_shape()))
                grid *= s
                _, center = self._shift_coords(self._x)
            else:  # target, Sect 3.3 Eq 3.22.
                s = f(self._dilation_factor()) / f(self._fft_shape())
                s *= f(2 * np.pi * self._upsample_factor())
                grid /= s
                _, center = self._shift_coords(self._z)
            grid += f(center)
        return grid

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        if (r_dtype := kwargs.get("dtype")) is None:
            c_dtype = pxrt.getPrecision().complex.value
        else:
            r_width = pxrt.Width(r_dtype)
            c_dtype = r_width.complex.value
        _A = self.ascomplexarray(xp=xp, dtype=c_dtype)

        A = pxu.view_as_real_mat(cmat=_A, real_input=self._real)
        return A

    def _upsample_factor(self) -> pxt.Real:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::FINUFFT_MAKEPLAN()
        if (u := self._upsampfac) is None:
            if self._eps >= 1e-9:
                u = 1.25
            else:
                u = 2
        return u

    def _fft_shape(self) -> pxt.NDArrayShape:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::set_nhg_type3()
        # [FINUFFT]_
        #     eq 3.23
        u = self._upsample_factor()
        w = self._kernel_width()
        X, _ = self._shift_coords(self._x)  # (D,)
        Z, _ = self._shift_coords(self._z)  # (D,)
        shape = []
        for d in range(self._D):
            n = (2 * u * max(1, X[d] * Z[d]) / np.pi) + (w + 1)
            target = max(int(n), 2 * w)
            n_opt = pxu.next_fast_len(target, even=True)
            shape.append(n_opt)
        return tuple(shape)

    def _dilation_factor(self) -> cabc.Sequence[pxt.Integer]:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/finufft.cpp::set_nhg_type3()
        # [FINUFFT]_
        #     eq 3.23
        u = self._upsample_factor()
        N = self._fft_shape()
        Z, _ = self._shift_coords(self._z)  # (D,)
        gamma = [n / (2 * u * s) for (n, s) in zip(N, Z)]
        return tuple(gamma)

    @staticmethod
    def _shift_coords(pts: pxt.NDArray) -> pxt.NDArray:
        # https://github.com/flatironinstitute/finufft/
        #     ./src/utils.cpp::arraywidcen()
        #     ./include/finufft/defs.h
        #
        # Parameters
        # ----------
        # pts: pxt.NDArray
        #     (Q, D) coordinates.
        #
        # Returns
        # -------
        # h_width: np.ndarray
        #     (D,) shifted half_width
        # center: np.ndarray
        #     (D,) shifted centroid
        low = pxu.to_NUMPY(pts.min(axis=0))
        high = pxu.to_NUMPY(pts.max(axis=0))
        h_width = (high - low) / 2
        center = (high + low) / 2
        grow_frac = 0.1

        mask = np.fabs(center) < h_width * grow_frac
        h_width[mask] += np.fabs(center[mask])
        center[mask] = 0
        return h_width, center


def _parallelize(func: cabc.Callable) -> cabc.Callable:
    # Parallelize execution of func() under conditions.
    #
    # * func() must be one of the arithmetic methods [apply,prox,grad,adjoint,pinv]()
    # * the `_parallel` attribute must be attached to the instance for it to parallelize execution
    #   over NUMPY inputs.

    @functools.wraps(func)
    def wrapper(*ARGS, **KWARGS):
        func_args = pxu.parse_params(func, *ARGS, **KWARGS)

        arr = func_args.get("arr", None)
        N = pxd.NDArrayInfo
        parallelize = ARGS[0]._parallel and (N.from_obj(arr) == N.NUMPY)

        # [2022.09.26] Sepand Kashani
        # Q: Why do we parallelize only for NUMPY inputs and not CUPY?
        # A: Given an arithmetic method `f`, there is no obligation (for DASK inputs) to satisfy the relation
        #    `f(arr).chunk_type == arr.chunk_type`.  In particular the relationship does not hold for LinFunc.[grad,
        #    adjoint]() unless the user provides a special implementation for DASK inputs.
        #    Consequence: the sum() / xp.concatenate() instructions in _COOBlock() may:
        #    * fail due to array-type mismatch; or
        #    * induce unnecessary CPU<>GPU transfers.

        if parallelize:
            xp = N.DASK.module()
            func_args.update(arr=xp.array(arr, dtype=arr.dtype))
        else:
            pass

        out = func(**func_args)
        f = {True: pxu.compute, False: lambda _: _}[parallelize]
        return f(out)

    return wrapper


class _NUFFT3_chunked(_NUFFT3):
    # Note:
    # * params() in chunked-context returns equivalent parameters of one single huge NUFFT3 block.
    #
    # TODO:
    #   What needs to be changed:
    #   * _tesselate() assumes x/z are in memory to find optimal chunks.  Do we want to support Dask arrays here so that
    #     users can auto-chunk disk-sized data?

    def __init__(self, **kwargs):
        self._disable_unsupported_methods()
        self._parallel = kwargs.pop("parallel")  # for _parallelize()

        kwargs.update(plan_fw=False, plan_bw=False)  # don't plan a huge NUFFT
        super().__init__(**kwargs)
        self._fail_on_small_problems()

        self._kwargs = kwargs.copy()  # extra FINUFFT planning args
        for k in ["x", "z"]:
            self._kwargs.pop(k, None)

        # variables set by allocate()
        self._initialized = False
        for attr in [
            "_plan_lock",  # FFTW planner lock; see _transform() for details
            "_up",  # up-sample ops
            "_down",  # down-sample ops
            "_dEval_threshold",  # direct-evaluation threshold
            "_x_reorder",  # .apply() input re-order op
            "_x_chunk",  # X-domain chunk slice-selectors
            "_z_reorder",  # .adjoint() input re-order op
            "_z_chunk",  # Z-domain chunk slice-selectors
        ]:
            setattr(self, attr, None)

    @_parallelize
    @pxrt.enforce_precision("arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        assert self._initialized
        arr = self._x_reorder.apply(arr)

        outer = []
        nufft = self._get_transform(arr)
        for i, up in self._up.items():
            inner = []
            for j, down in self._down.items():
                x = down.apply(arr)
                y = _array_ize(
                    data=nufft(j, i, "fw", x),
                    shape=(*arr.shape[:-1], up.dim),
                    dtype=arr.dtype,
                )
                inner.append(y)
            inner = self._tree_sum(inner)
            z = up.apply(inner)
            outer.append(z)
        out = self._tree_sum(outer)

        out = self._z_reorder.apply(out)
        return out

    @_parallelize
    @pxrt.enforce_precision("arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        assert self._initialized
        arr = self._z_reorder.adjoint(arr)

        outer = []
        nufft = self._get_transform(arr)
        for j, down in self._down.items():
            inner = []
            for i, up in self._up.items():
                z = up.adjoint(arr)
                y = _array_ize(
                    data=nufft(j, i, "bw", z),
                    shape=(*arr.shape[:-1], down.codim),
                    dtype=arr.dtype,
                )
                inner.append(y)
            inner = self._tree_sum(inner)
            x = down.adjoint(inner)
            outer.append(x)
        out = self._tree_sum(outer)

        out = self._x_reorder.adjoint(out)
        return out

    def _get_transform(self, arr: pxt.NDArray) -> cabc.Callable:
        NDI = pxd.NDArrayInfo
        dask_input = NDI.from_obj(arr) == NDI.DASK

        if dask_input:
            func = dask.delayed(
                self._transform,
                pure=True,
                traverse=False,
            )
        else:
            func = self._transform
        return func

    def _transform(
        self,
        x_idx: int,
        z_idx: int,
        mode: str,
        arr: pxt.NDArray,
    ) -> pxt.NDArray:
        # Internal method for apply/adjoint.
        #
        # Parameters
        # ----------
        # x_idx: int
        #     Index into _x_chunk.
        #     Identifies the X-region participating in the transform.
        # z_idx: int
        #     Index into _z_chunk.
        #     Identifies the Z-region participating in the transform.
        # mode: str
        #     Transform direction:
        #
        #     * 'fw': _NUFFT3.apply
        #     * 'bw': _NUFFT3.adjoint
        # arr: pxt.NDArray
        #     (...,) array fed to apply/adjoint().
        #
        # Returns
        # -------
        # out: pxt.NDArray
        #     (...,) output of _NUFFT3.[apply|adjoint](arr)
        kwargs = self._kwargs.copy()

        x = self._x[self._x_chunk[x_idx]]
        z = self._z[self._z_chunk[z_idx]]
        kwargs.update(x=x, z=z)

        if mode == "fw":
            kwargs.update(plan_fw=True, plan_bw=False)
            func = "apply"
        else:  # bw
            kwargs.update(plan_fw=False, plan_bw=True)
            func = "adjoint"

        if len(x) * len(z) <= self._dEval_threshold:
            kwargs.update(eps=0)

        # NUFFT.apply/adjoint() requires inputs to be C-contiguous.  This is not guaranteed in chunked context given
        # chain of operations preceding apply/adjoint().  Chunks are small however, so the overhead is minimal.
        xp = pxu.get_array_module(arr)
        arr = xp.require(arr, requirements="C")

        with self._plan_lock:
            # FINUFFT uses FFTW to compute FFTs.
            # FFTW's planner is not thread-safe. [http://www.fftw.org/fftw3_doc/Thread-safety.html]
            # Not coordinating the planning stage with other workers/tasks leads to segmentation faults.
            op = NUFFT.type3(**kwargs)

        # The complexity of _NUFFT3.[apply|adjoint]() is
        #     N_F \ln N_F   +   (N_x + N_z) w^d
        #        N_F = FFT size
        #        N_x = # x-domain points
        #        N_z = # z-domain points
        #        w = spread/interpolation kernel size
        #
        #
        # The complexity of _NUFFT3_chunked.[apply|adjoint]() is
        # (1) \sum_{i,j} N_F_ij \ln N_F_ij   +   N_z_blk N_x w^d   +   N_x_blk N_z w^d,
        #         N_F_ij = FFT size in (i,j)-th sub-problem
        #         N_x_blk = # x-domain chunks
        #         N_z_blk = # z-domain chunks
        # Assuming N_x_blk ~= N_z_blk, the term above simplifies to
        #     \sum_{i,j} N_F_ij \ln N_F_ij   +   (N_x w^d + N_z w^d) \sqrt(N_x_blk * N_z_blk),
        # i.e. the spread/interpolation cost grows proportional to \sqrt{# sub-problems}
        #
        #
        # It is possible to share the spread/interpolation costs amongst sub-problems so as to bring the complexity of
        # _NUFFT3_chunked.[apply|adjoint]() down to
        # (2) (N_x_blk * N_z_blk) * N_F \ln N_F   +   (N_x + N_z) w^d.
        #
        # The philosophies of computing via (1) and (2) differ:
        #   (1) optimal FFT-size per sub-problem + \sqrt{N_xz_blk} spread/interpolation overhead
        #   (2) largest FFT-size per sub-problem + no spread/interpolation overhead w.r.t. _NUFFT3.[apply|adjoint]()
        out = getattr(op, func)(arr)
        return out

    def auto_chunk(
        self,
        max_mem: pxt.Real = 10,
        max_anisotropy: pxt.Real = 5,
    ) -> tuple[list[pxt.NDArray], list[pxt.NDArray]]:
        max_mem = float(max_mem)
        assert max_mem > 0
        max_mem *= 2**20  # MiB -> B

        max_anisotropy = float(max_anisotropy)
        assert max_anisotropy >= 1

        T, B = self._box_dimensions(max_mem, max_anisotropy)
        x_chunks = self._tesselate(self._x, T)
        z_chunks = self._tesselate(self._z, B)
        return x_chunks, z_chunks

    def allocate(
        self,
        x_chunks: list[typ.Union[pxt.NDArray, slice]],
        z_chunks: list[typ.Union[pxt.NDArray, slice]],
        direct_eval_threshold: pxt.Integer = 10_000,
    ):
        def _to_slice(idx_spec):
            out = idx_spec
            if not isinstance(idx_spec, slice):
                lb = idx_spec.min()
                ub = idx_spec.max()
                if ub - lb + 1 == len(idx_spec):
                    out = slice(lb, ub + 1)
            return out

        def _preprocess(chunks, var: str):
            # Analyze chunk specifiers and return:
            #   * input re-ordering coordinates (if applicable)
            #   * slice() objects identifying each sub-chunk data
            chunks = list(map(_to_slice, chunks))
            if all(isinstance(chk, slice) for chk in chunks):
                # No re-ordering required: up/sub-sampling operators can slice-select themselves.
                reorder_spec = slice(0, sum(chk.stop - chk.start for chk in chunks))
                chunk_spec = chunks
            else:
                # Some chunks cannot be slice-indexed. To avoid expensive x/z copies when
                # initializing NUFFT sub-problems, x/z will be re-ordered before/after
                # down/up-sampling operators.
                reorder_spec = []
                for chk in chunks:
                    if isinstance(chk, slice):
                        _chk = np.arange(chk.start, chk.stop)
                    else:
                        _chk = chk
                    reorder_spec.append(_chk)
                reorder_spec = np.concatenate(reorder_spec)

                chunk_spec, start = [], 0
                for chk in chunks:
                    if isinstance(chk, slice):
                        _len = chk.stop - chk.start
                    else:
                        _len = len(chk)
                    s = slice(start, start + _len)
                    start += _len
                    chunk_spec.append(s)

                if self._enable_warnings:
                    msg = "\n".join(
                        [
                            f"'{var}' order is sub-optimal given provided chunk specifiers.",
                            f"'{var}' will be re-ordered internally to improve NUFFT performance.",
                            "The cost of re-ordering apply/adjoint inputs is significant when the number of non-uniform points x/z is large.",
                            f"It is recommended to re-initialize {self.__class__} where x/z [and apply/adjoint() inputs] are re-ordered.",
                            f"See notes/examples provided in docstring of {NUFFT.type3.__qualname__}() for how to achieve this.",
                        ]
                    )
                    warnings.warn(msg, pxw.PerformanceWarning)
            return reorder_spec, chunk_spec

        def _r2c(idx_spec):
            if isinstance(idx_spec, slice):
                idx = slice(2 * idx_spec.start, 2 * idx_spec.stop)
            else:
                idx = np.stack([2 * idx_spec, 2 * idx_spec + 1], axis=1).reshape(-1)
            return idx

        x_idx, x_chunks = _preprocess(x_chunks, var="x")
        self._x = self._x[x_idx]
        self._x_reorder = pxs.SubSample(  # Permutation
            (self.dim,),
            x_idx if self._real else _r2c(x_idx),
        )
        self._x_chunk = x_chunks
        self._down = {
            j: pxs.SubSample(
                (self.dim,),
                x_idx if self._real else _r2c(x_idx),
            )
            for (j, x_idx) in enumerate(x_chunks)
        }

        z_idx, z_chunks = _preprocess(z_chunks, var="z")
        self._z = self._z[z_idx]
        self._z_reorder = pxs.SubSample(
            (self.codim,),
            _r2c(z_idx),
        ).T
        self._z_chunk = z_chunks
        self._up = {
            i: pxs.SubSample(
                (self.codim,),
                _r2c(z_idx),
            ).T
            for (i, z_idx) in enumerate(z_chunks)
        }

        self._plan_lock = threading.Lock()
        self._dEval_threshold = float(direct_eval_threshold)
        self._initialized = True

    def stats(self) -> collections.namedtuple:
        BLOCK_STATS = collections.namedtuple(
            "block_stats",
            [
                "blk_count",
                "dEval_count",
            ],
        )

        N_up = [u.dim // 2 for u in self._up.values()]
        N_down = [d.codim // (1 if self._real else 2) for d in self._down.values()]
        blk_count = len(N_up) * len(N_down)
        dEval = sum(1 for (u, d) in itertools.product(N_up, N_down) if (u * d) <= self._dEval_threshold)

        p = BLOCK_STATS(blk_count=blk_count, dEval_count=dEval)
        return p

    def ascomplexarray(self, **kwargs) -> pxt.NDArray:
        cA = super().ascomplexarray(**kwargs)
        rA = pxu.view_as_real_mat(cA, real_input=self._real)
        with pxrt.Precision(pxrt.Width(rA.dtype)):
            rA = self._x_reorder.adjoint(rA)  # re-order rows
            rA = self._z_reorder.apply(rA.T).T  # re-order columns
        cA = pxu.view_as_complex_mat(rA, real_input=self._real)
        return cA

    def order(self, var: str) -> tuple:
        var = var.strip().lower()
        assert var in ("x", "z")

        def _c2r(idx_spec):
            if isinstance(idx_spec, slice):
                idx = slice(idx_spec.start // 2, idx_spec.stop // 2 + 1)
            else:
                idx = idx_spec[::2] // 2
            return idx

        if var == "x":
            idx = self._x_reorder._idx[0]
            idx = idx if self._real else _c2r(idx)
            chunks = self._x_chunk
        else:  # "z"
            idx = self._z_reorder._op._idx[0]  # _z_reorder = SubSampleOp.T
            idx = _c2r(idx)
            chunks = self._z_chunk

        return idx, chunks

    def _disable_unsupported_methods(self):
        # Despite being a child-class of _NUFFT3, some methods are not supported because they don't
        # make sense in the chunked context.
        # This method overrides problematic methods to avoid erroneous use.
        def unsupported(_, **kwargs):
            raise NotImplementedError

        unsupported_fields = [
            "mesh",
            "plot_kernel",
        ]
        for f in unsupported_fields:
            override = types.MethodType(unsupported, self)
            setattr(self, f, override)

    def _fail_on_small_problems(self):
        cst = 10
        if (len(self._x) < cst) and (len(self._z) < cst):
            msg = " ".join(
                [
                    "A chunked-NUFFT3 is sub-optimal for very small problems:",
                    "instantiate instead via `NUFFT.type3(x,z,eps=0)`.",
                ]
            )
            raise ValueError(msg)

    @classmethod
    def _tree_sum(cls, data: cabc.Sequence):
        # computes (data[0] + ... + data[N-1]) via a binary tree reduction.
        if (N := len(data)) == 1:
            return data[0]
        else:
            compressed = [data[2 * i] + data[2 * i + 1] for i in range(N // 2)]
            if N % 2 == 1:
                compressed.append(data[-1])
            return cls._tree_sum(compressed)

    def _box_dimensions(
        self,
        fft_bytes: float,
        alpha: float,
    ) -> tuple[tuple[pxt.Real], tuple[pxt.Real]]:
        r"""
        Find X box dimensions (T_1,...,T_D) and Z box dimensions (B_1,...,B_D) such that:

        * number of NUFFT sub-problems is minimized;
        * NUFFT sub-problems limited to user-specified memory budget;
        * box dimensions are not too rectangular, i.e. anisotropic.

        Parameters
        ----------
        fft_bytes: pxt.Real
            Max FFT memory (B) allowed per sub-block.
        alpha: pxt.Real
            Max tolerated (normalized) anisotropy ratio >= 1.

        Returns
        -------
        T: tuple[float]
            X-box dimensions.
        B: tuple[float]
            Z-box dimensions.

        Notes
        -----
        Given that

            FFT_memory / (element_itemsize * N_transform)
            \approx
            \prod_{k=1..D} (\sigma_k T_k B_k) / (2 \pi),

        we can solve an optimization problem to find the optimal (T_k, B_k) values.


        Mathematical Formulation
        ------------------------

        User input:
            1. FFT_memory: max memory budget per sub-problem
            2. alpha: max anisotropy >= 1

        minimize (objective_func)
            \prod_{k=1..D} T_k^{tot} / T_k                                                    # X-domain box-count
            *                                                                                 #       \times
            \prod_{k=1..D} B_k^{tot} / B_k                                                    # Z-domain box-count
        subject to
            1. \prod_{k=1..D} s_k T_k B_k <= FFT_mem / (elem_size * N_trans) * (2 \pi)^{D}    # sub-problem memory limit
            2. T_k <= T_k^{tot}                                                               # X-domain box size limited to X_k's spread
            3. B_k <= B_k^{tot}                                                               # Z-domain box size limited to Z_k's spread
            4. objective_func >= 1                                                            # at least 1 NUFFT sub-problem necessary
            5. 1/alpha <= (T_k / T_k^{tot}) / (T_q / T_q^{tot}) <= alpha                      # X-domain box size anisotropy limited
            6. 1/alpha <= (B_k / B_k^{tot}) / (B_q / B_q^{tot}) <= alpha                      # Z-domain box size anisotropy limited
            7. 1/alpha <= (T_l / T_l^{tot}) / (B_m / B_m^{tot}) <= alpha                      # XZ-domain box size cross-anisotropy limited

        Constraint (7) ensures (T_k, B_k) partitions the X/Z-domains uniformly. (Not including this term may give rise
        to solutions where X/Z-domains are partitioned finely/coarsely, or not at all.)

        The problem above can be recast as a small LP and easily solved.


        Mathematical Formulation (LinProg)
        ----------------------------------

        minimize
            c^{T} x
        subject to
            A x <= b
              x <= u
        where
            x = [ln(T_1) ... ln(T_D), ln(B_1) ... ln(B_D)] \in \bR^{2D}
            c = [-1 ... -1]
            u = [ln(T_1^{tot}) ... ln(T_D^{tot}), ln(B_1^{tot}) ... ln(B_D^{tot})]
            [A | b] = [
                    [  -c   | ln(FFT_mem / (elem_size * N_trans)) + \sum_{k=1..D} ln(2 \pi / s_k) ],  # sub-problem memory limit
                    [  -c   | \sum_{k=1..D} ln(T_k^{tot}) + \sum_{k=1..D} ln(B_k^{tot})           ],  # at least 1 NUFFT sub-problem necessary
               (L1) [ M1, Z | ln(alpha) + ln(T_k^{tot}) - ln(T_q^{tot})                           ],  # X-domain box size anisotropy limited (upper limit, vector form)
               (L2) [-M1, Z | ln(alpha) - ln(T_k^{tot}) + ln(T_q^{tot})                           ],  # X-domain box size anisotropy limited (lower limit, vector form)
               (L3) [ Z, M1 | ln(alpha) + ln(B_k^{tot}) - ln(B_q^{tot})                           ],  # Z-domain box size anisotropy limited (upper limit, vector form)
               (L4) [ Z,-M1 | ln(alpha) - ln(B_k^{tot}) + ln(B_q^{tot})                           ],  # Z-domain box size anisotropy limited (lower limit, vector form)
               (L5) [  M2   | ln(alpha) + ln(T_l^{tot}) - ln(B_m^{tot})                           ],  # XZ-domain box size cross-anisotropy limited (upper limit, vector form)
               (L6) [ -M2   | ln(alpha) - ln(T_l^{tot}) + ln(B_m^{tot})                           ],  # XZ-domain box size cross-anisotropy limited (lower limit, vector form)
            ]
            Z = zeros(D_choose_2, D)
            M1 = (D_choose_2, D) (M)ask containing:
                D = 1 => drop L1..4 in [A | b]
                D = 2 => [[ 1 -1]
                          [-1  1]]
                D = 3 => [[ 1 -1  0]
                          [ 1  0 -1]
                          [ 0  1 -1]]
            M2 = (D^{2}, 2 D) (M)ask containing:
                D = 1 => [1 -1]
                D = 2 => [[ 1  0 -1  0]
                          [ 1  0  0 -1]
                          [ 0  1 -1  0]
                          [ 0  1  0 -1]]
                D = 3 => [[ 1  0  0 -1  0  0]
                          [ 1  0  0  0 -1  0]
                          [ 1  0  0  0  0 -1]
                          [ 0  1  0 -1  0  0]
                          [ 0  1  0  0 -1  0]
                          [ 0  1  0  0  0 -1]
                          [ 0  0  1 -1  0  0]
                          [ 0  0  1  0 -1  0]
                          [ 0  0  1  0  0 -1]]
        """
        # NUFFT parameters
        D = self._D
        xp = pxu.get_array_module(self._x)
        T_tot = pxu.to_NUMPY(xp.ptp(self._x, axis=0))
        B_tot = pxu.to_NUMPY(xp.ptp(self._z, axis=0))
        sigma = np.array((self._upsample_factor(),) * D)
        c_width = pxrt.Width(self._x.dtype).complex
        c_itemsize = c_width.value.itemsize
        n_trans = self._n

        # (M)ask, (Z)ero and (R)ange arrays to simplify LinProg spec
        R = np.arange(D)
        Z = np.zeros((math.comb(D, 2), D))
        M1 = Z.copy()
        _k, _q = np.triu_indices(n=D, k=1)
        for i, (__k, __q) in enumerate(zip(_k, _q)):
            M1[i, __k] = 1
            M1[i, __q] = -1
        _l, _m = np.kron(R, np.ones(D, dtype=int)), np.tile(R, D)
        M2 = np.zeros((D**2, 2 * D))
        for i, (__l, __m) in enumerate(zip(_l, _m)):
            M2[i, __l] = 1
            M2[i, D + __m] = -1

        # LinProg parameters
        c = -np.ones(2 * D)  # maximize box volumes / minimize #sub-problems
        A = np.block(
            [
                [-c],  # memory limit
                [-c],  # at least 1 box
                [M1, Z],  # T_k anisotropy upper-bound
                [-M1, Z],  # T_k anisotropy lower-bound
                [Z, M1],  # B_k anisotropy upper-bound
                [Z, -M1],  # B_k anisotropy lower-bound
                [M2],  # T_k/B_k cross-anisotropy upper-bound
                [-M2],  # T_k/B_k cross-anisotropy lower-bound
            ]
        )
        b = np.r_[
            np.log(fft_bytes / (c_itemsize * n_trans)) + np.log(2 * np.pi / sigma).sum(),  # memory limit
            np.log(T_tot).sum() + np.log(B_tot).sum(),  # at least 1 box
            np.log(alpha) + np.log(T_tot)[_k] - np.log(T_tot)[_q],  # T_k anisotropy upper-bound
            np.log(alpha) - np.log(T_tot)[_k] + np.log(T_tot)[_q],  # T_k anisotropy lower-bound
            np.log(alpha) + np.log(B_tot)[_k] - np.log(B_tot)[_q],  # B_k anisotropy upper-bound
            np.log(alpha) - np.log(B_tot)[_k] + np.log(B_tot)[_q],  # B_k anisotropy lower-bound
            np.log(alpha) + np.log(T_tot)[_l] - np.log(B_tot)[_m],  # T_k/B_k cross-anisotropy upper-bound
            np.log(alpha) - np.log(T_tot)[_l] + np.log(B_tot)[_m],  # T_k/B_k cross-anisotropy lower-bound
        ]
        lb = -np.inf * np.ones(2 * D)  # T_k, B_k lower limit (None)
        ub = np.r_[np.log(T_tot), np.log(B_tot)]  # T_k, B_k upper limit

        res = sopt.linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=np.stack([lb, ub], axis=1),
            method="highs",
        )
        if res.success:
            T = np.exp(res.x[:D])
            B = np.exp(res.x[D:])
            return tuple(T), tuple(B)
        else:
            msg = "Auto-chunking failed given memory/anisotropy constraints."
            raise ValueError(msg)

    @staticmethod
    def _tesselate(
        data: pxt.NDArray,
        box_dim: tuple[pxt.Real],
    ) -> list[pxt.NDArray]:
        """
        Split point-cloud into disjoint rectangular regions.

        Parameters
        ----------
        data: pxt.NDArray
            (M, D) point cloud.
        box_dim: tuple[float]
            (D,) box dimensions.

        Returns
        -------
        chunks: list[NDArray[int]]
            (idx[0], ..., idx[C-1]) chunk specifiers.
            `idx[k]` contains indices of `data` which lie in the same box.
        """
        M, D = data.shape

        # Center data-points around origin
        data = pxu.to_NUMPY(data.copy())
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data -= (data_min + data_max) / 2

        # Compute optimal box_[dim, count]
        data_spread = data_max - data_min
        box_dim = np.array(box_dim, dtype=data.dtype)
        N_box = np.ceil(data_spread / box_dim).astype(int)
        box_dim = data_spread / N_box

        # Rescale data to have equal spread in each dimension.
        # Reason: KDTree only accepts scalar-valued radii.
        scale = box_dim
        data /= scale
        data_min /= scale
        data_max /= scale
        data_spread /= scale
        box_dim = np.ones_like(box_dim)

        # Compute gridded centroids
        range_spec = []
        for n in N_box:
            is_odd = n % 2 == 1
            lb, ub = -(n // 2), n // 2 + (1 if is_odd else 0)
            offset = 0 if is_odd else 1 / 2
            s = np.arange(lb, ub, dtype=data.dtype) + offset
            range_spec.append(s)
        centroid = np.meshgrid(*range_spec, indexing="ij")
        centroid = np.stack(centroid, axis=-1).reshape(-1, D)

        # Allocate data points to gridded centroids
        c_tree = spl.KDTree(centroid)  # centroid_tree
        dist, c_idx = c_tree.query(
            data,
            k=1,
            eps=1e-2,  # approximate NN-search for speed
            p=np.inf,  # L-infinity norm
        )
        idx = np.argsort(c_idx)
        count = collections.Counter(c_idx[idx])  # sort + count occurence
        chunks, start = [], 0
        for c_idx, step in sorted(count.items()):
            chk = idx[start : start + step]
            chunks.append(chk)
            start += step
        centroid = centroid[sorted(count.keys())]  # drop boxes with no data-points

        # Compute true centroids + tight box boundaries seen by FINUFFT
        tbox_dim = np.zeros((len(centroid), D))  # tight box_dim(s)
        for i in range(len(centroid)):
            _data = data[chunks[i]]
            _data_min = _data.min(axis=0)
            _data_max = _data.max(axis=0)
            centroid[i] = (_data_min + _data_max) / 2
            tbox_dim[i] = _data_max - _data_min

        # Fuse chunks which are closely-spaced & small-enough
        fuse_chunks = True
        while fuse_chunks:
            # Find fuseable centroid pairs
            c_tree = spl.KDTree(centroid)  # centroid_tree
            candidates = c_tree.query_pairs(
                r=box_dim[0] / 2,
                p=np.inf,
                output_type="ndarray",
            )
            _i, _j = candidates.T
            c_spacing = np.abs(centroid[_i] - centroid[_j])
            offset = (tbox_dim[_i] + tbox_dim[_j]) / 2
            fuseable = np.all(c_spacing + offset < box_dim, axis=1)
            candidates = candidates[fuseable]

            # If a centroid can be fused with multiple others, restrict choice to single pair
            seen, fuse = set(), set()
            for _i, _j in candidates:
                if (_i not in seen) and (_j not in seen):
                    seen |= {_i, _j}
                    fuse.add((_i, _j))

            if len(fuse) > 0:
                for _i, _j in fuse:
                    chunks[_i] = np.r_[chunks[_i], chunks[_j]]
                    _data = data[chunks[_i]]
                    _data_min = _data.min(axis=0)
                    _data_max = _data.max(axis=0)
                    centroid[_i] = (_data_min + _data_max) / 2
                    tbox_dim[_i] = _data_max - _data_min

                # Fuse cleanup: drop _j entries
                c_idx = np.setdiff1d(  # indices to keep
                    np.arange(len(centroid)),
                    [_j for (_i, _j) in fuse],  # indices to drop
                )
                centroid = centroid[c_idx]
                tbox_dim = tbox_dim[c_idx]
                chunks = [chk for (i, chk) in enumerate(chunks) if (i in c_idx)]
            else:
                fuse_chunks = False

        return chunks

    def diagnostic_plot(self, domain: str):
        plt = pxu.import_module("matplotlib.pyplot")
        mpl_p = pxu.import_module("matplotlib.patches")

        def _plot(
            points,  # (N, 2) data points
            chunks,  # (N_c,) chunk specifiers
            ax,
        ):
            # Compute chunk centroids, tight-box_dims, cvx hulls
            N_chk = len(chunks)
            centroid = np.zeros((N_chk, 2))
            tbox_dim = np.zeros((N_chk, 2))
            hull = []  # (N_c,) hulls
            for i, chk in enumerate(chunks):
                _pts = points[chk]
                _pts_min = _pts.min(axis=0)
                _pts_max = _pts.max(axis=0)
                centroid[i] = (_pts_min + _pts_max) / 2
                tbox_dim[i] = _pts_max - _pts_min
                _hull = _pts[spl.ConvexHull(_pts).vertices]
                hull.append(_hull)

            for _tbox, _c in zip(tbox_dim, centroid):
                data_rect = mpl_p.Rectangle(  # sub-problem bounding-box
                    xy=_c - _tbox / 2,
                    width=_tbox[0],
                    height=_tbox[1],
                    fill=True,
                    edgecolor="r",
                    facecolor="r",
                    alpha=0.5,
                    label="data box",
                )
                ax.add_patch(data_rect)
            for _hull in hull:
                data_poly = mpl_p.Polygon(  # sub-problem data support
                    xy=_hull,
                    fill=True,
                    edgecolor="b",
                    facecolor="b",
                    alpha=0.5,
                    label="data cvx hull",
                )
                ax.add_patch(data_poly)
            centroid_pts = ax.plot(  # sub-problem bounding-box centers
                centroid[:, 0],
                centroid[:, 1],
                "x",  # cross
                color="k",
                label="chunk centroid",
            )
            ax.axis("equal")
            ax.legend(
                handles=[
                    *centroid_pts,
                    data_poly,
                    data_rect,
                ],
                loc="upper right",
            )

        domain = domain.strip().lower()
        if domain == "x":
            data = self._x
            chunks = self._x_chunk
        elif domain == "z":
            data = self._z
            chunks = self._z_chunk
        else:
            raise ValueError(f"Unknown domain '{domain}'.")

        _, D = data.shape
        if D == 1:
            raise NotImplementedError

        fig = plt.figure()
        if D == 2:
            ax = fig.add_subplot(1, 1, 1)
            _plot(
                points=data,
                chunks=chunks,
                ax=ax,
            )
            ax.set_title("data/chunk distribution (XY-plane)")
        else:  # D == 3
            for i, idx, title in [
                (0, [0, 1], "data/chunk distribution (XY-plane)"),
                (1, [0, 2], "data/chunk distribution (XZ-plane)"),
                (2, [1, 2], "data/chunk distribution (YZ-plane)"),
            ]:
                ax = fig.add_subplot(1, 3, i + 1)
                _plot(
                    points=data[:, idx],
                    chunks=chunks,
                    ax=ax,
                )
                ax.set_title(title)
        return fig


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _nudft_NUMPY(
    weight: pxt.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pxt.NDArray,  # (M, D) sample points [float32/64]
    target: pxt.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pxt.DType,  # complex64/128
) -> pxt.NDArray:  # (Q, N) complex64/128
    Q = weight.shape[0]
    M = source.shape[0]
    N = target.shape[0]
    out = np.zeros(shape=(Q, N), dtype=dtype)
    for n in numba.prange(N):
        for m in range(M):
            scale = np.exp(isign * 1j * np.dot(source[m, :], target[n, :]))
            out[:, n] += weight[:, m] * scale
    return out


def _nudft_CUPY(
    weight: pxt.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pxt.NDArray,  # (M, D) sample points [float32/64]
    target: pxt.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pxt.DType,  # complex64/128
) -> pxt.NDArray:  # (Q, N) complex64/128
    @numba.cuda.jit(device=True)
    def _cexp(s, a, b):  # [(1,), (D,), (D,)] -> (1,)
        # np.exp(1j * s * (a @ b))
        D, c = len(a), 0
        for d in range(D):
            c += a[d] * b[d]
        out = cmath.exp(1j * s * c)
        return out

    @numba.cuda.jit(fastmath=True, opt=True, cache=True)
    def _kernel(weight, source, target, isign, out):
        Q, M = weight.shape[:2]
        N, D = target.shape[:2]
        q, n = numba.cuda.grid(2)
        if (q < Q) and (n < N):
            for m in range(M):
                scale = _cexp(isign, source[m, :], target[n, :])
                out[q, n] += weight[q, m] * scale

    Q = weight.shape[0]
    N = target.shape[0]
    xp = pxu.get_array_module(weight)
    out = xp.zeros((Q, N), dtype=dtype)

    ceil = lambda _: int(np.ceil(_))
    t_max = weight.device.attributes["MaxThreadsPerBlock"]
    tpb = [min(Q, t_max // 2), None]  # thread_per_block
    tpb[1] = t_max // tpb[0]
    bpg = [ceil(Q / tpb[0]), ceil(N / tpb[1])]  # block_per_grid

    config = _kernel[tuple(bpg), tuple(tpb)]
    config(weight, source, target, isign, out)
    return out


@pxu.redirect(i="weight", NUMPY=_nudft_NUMPY, CUPY=_nudft_CUPY)
def _nudft(
    weight: pxt.NDArray,  # (Q, M) weights (n_trans=Q) [complex64/128]
    source: pxt.NDArray,  # (M, D) sample points [float32/64]
    target: pxt.NDArray,  # (N, D) query points [float32/64]
    *,
    isign: SignT,
    dtype: pxt.DType,  # complex64/128
) -> pxt.NDArray:  # (Q, N) complex64/128
    pass
