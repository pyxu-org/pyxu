import functools
import itertools

import numpy as np
import pytest

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.operator.linop.fft._ffs as ffs
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.operator.conftest as conftest


class TestFFS(conftest.NormalOpT):
    @pytest.fixture(
        params=[
            # Specification:
            #     T (user-specified)
            #     T (canonical)
            #     Tc (user-specified)
            #     Tc (canonical)
            #     Nfs (user-specified)
            #     Nfs (canonical)
            #     Ns (user-specified)
            #     Ns (canonical)
            # 1D transforms ---------------------------------------------------
            (5, (5,), 0, (0,), 3, (3,), 3, (3,)),
            (5, (5,), 0.3, (0.3,), 5, (5,), 5, (5,)),  # Tc non-zero
            (5, (5,), 0.45, (0.45,), 5, (5,), 6, (6,)),  # Ns even
            (5, (5,), 0.45, (0.45,), 5, (5,), 7, (7,)),  # Ns odd
            # 2D transforms ---------------------------------------------------
            (5, (5, 5), 0, (0, 0), 3, (3, 3), (3, 3), (3, 3)),
            (  # all params distinct, Ns odd/odd
                (5, 0.1),
                (5, 0.1),
                (-1, 2.3),
                (-1, 2.3),
                (3, 5),
                (3, 5),
                (3, 5),
                (3, 5),
            ),
            (  # all params distinct, Ns odd/even
                (5, 0.1),
                (5, 0.1),
                (-1, 2.3),
                (-1, 2.3),
                (3, 5),
                (3, 5),
                (5, 6),
                (5, 6),
            ),
            (  # all params distinct, Ns even/odd
                (5, 0.1),
                (5, 0.1),
                (-1, 2.3),
                (-1, 2.3),
                (3, 5),
                (3, 5),
                (4, 7),
                (4, 7),
            ),
            (  # all params distinct, Ns even/even
                (5, 0.1),
                (5, 0.1),
                (-1, 2.3),
                (-1, 2.3),
                (3, 5),
                (3, 5),
                (4, 6),
                (4, 6),
            ),
        ]
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture(
        params=itertools.product(
            pxd.NDArrayInfo,
            pxrt.CWidth,
        )
    )
    def spec(self, _spec, request) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        T, Tc, Nfs, Ns = _spec[0], _spec[2], _spec[4], _spec[6]  # user-specified form
        op = ffs._FFS(
            T=T,
            Tc=Tc,
            Nfs=Nfs,
            Ns=Ns,
        )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, _spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        sh = _spec[7]
        return (*sh, 2)

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture
    def T(self, _spec) -> tuple[float]:
        return _spec[1]

    @pytest.fixture
    def Tc(self, _spec) -> tuple[float]:
        return _spec[3]

    @pytest.fixture
    def Nfs(self, _spec) -> tuple[int]:
        return _spec[5]

    @pytest.fixture
    def data_apply(
        self,
        op,
        T,
        Tc,
        Nfs,
        dim_shape,
    ) -> conftest.DataLike:
        # We test the transform on the multi-dim Dirichlet kernel of bandwidth Nfs.

        # Compute the space-samples
        location = op.sample_points(xp=np, dtype=pxrt.Width.DOUBLE.value, flatten=False)
        X = [dirichlet(loc, t, tc, nfs) for (loc, t, tc, nfs) in zip(location, T, Tc, Nfs)]
        x = functools.reduce(np.multiply, X).astype(complex)

        # Compute analytical FS coefficients
        Y = np.meshgrid(
            *[dirichlet_fs(nfs, t, tc) for (nfs, t, tc) in zip(Nfs, T, Tc)],
            indexing="ij",
            sparse=True,
        )
        y = np.zeros(dim_shape[:-1], dtype=complex)  # don't want the (2,) view dim.
        select = [slice(nfs) for nfs in Nfs]
        y[tuple(select)] = functools.reduce(np.multiply, Y)

        return dict(
            in_=dict(arr=pxu.view_as_real(x)),
            out=pxu.view_as_real(y),
        )


# Helper Functions ------------------------------------------------------------
def dirichlet(x, T, T_c, N_FS):
    r"""
    Return samples of a shifted Dirichlet kernel of period :math:`T` and
    bandwidth :math:`N_{FS} = 2 N + 1`:

    .. math::

       \phi(t) = \sum_{k = -N}^{N} \exp\left( j \frac{2 \pi}{T} k (t - T_{c}) \right)
               = \frac{\sin\left( N_{FS} \pi [t - T_{c}] / T \right)}{\sin\left( \pi [t - T_{c}]
               / T \right)}.

    Parameters
    ----------
    x : :py:class:`~numpy.ndarray`
        Sampling points.
    T : float
        Function period.
    T_c : float
        Period mid-point.
    N_FS : int
        Function bandwidth.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Function values.

    See Also
    --------
    :py:func:`~pyffs.func.dirichlet_fs`
    """
    y = x - T_c
    n, d = np.zeros((2, *x.shape))
    nan_mask = np.isclose(np.fmod(y, np.pi), 0)
    n[~nan_mask] = np.sin(N_FS * np.pi * y[~nan_mask] / T)
    d[~nan_mask] = np.sin(np.pi * y[~nan_mask] / T)
    n[nan_mask] = N_FS * np.cos(N_FS * np.pi * y[nan_mask] / T)
    d[nan_mask] = np.cos(np.pi * y[nan_mask] / T)
    return n / d


def dirichlet_fs(N_FS, T, T_c):
    """
    Return Fourier Series coefficients of a shifted Dirichlet kernel of period
    :math:`T` and bandwidth :math:`N_{FS} = 2 N + 1`.

    Parameters
    ----------
    N_FS : int
        Function bandwidth.
    T : float
        Function period.
    T_c : float
        Period mid-point.

    Returns
    -------
    vals : :py:class:`~numpy.ndarray`
        Fourier Series coefficients in ascending order.
    """
    N = (N_FS - 1) // 2
    return np.exp(-1j * (2 * np.pi / T) * T_c * np.arange(start=-N, stop=N + 1))
