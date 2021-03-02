import numpy as np
from typing import Tuple, Union
from numba import njit


class P2Algorithm:
    r"""
    P-Square Algorithm.

    The P-Square Algorithm is an heuristic algorithm  for dynamic calculation of empirical quantiles. The estimates
    are produced dynamically as the observations are generated. The observations are not stored; therefore, the algorithm has
    a very small and fixed storage requirement regardless of the number of observations. See [P2]_ for more details on the algorithm.

    Examples
    --------
    .. testsetup::

        import numpy as np
        import scipy.stats
        from pycsou.util import P2Algorithm


    .. doctest::

        >>> rng = np.random.default_rng(0)
        >>> population_quantile = scipy.stats.norm.ppf(0.95)
        >>> def generate_sample(n):
        ...     for i in range(n):
        ...         yield rng.standard_normal()
        >>> p2 = P2Algorithm(pvalue=0.95)
        >>> samples=[]
        >>> for sample in generate_sample(1000):
        ...     p2.add_sample(sample)
        ...     samples.append(sample)
        >>> print(f'P2 Quantile: {p2.q}, Empirical Quantile: {np.quantile(samples, 0.95)}, Population Quantile: {population_quantile}.')
        P2 Quantile: [1.51436338], Empirical Quantile: 1.514048975492714, Population Quantile: 1.6448536269514722.

    Notes
    -----
    The estimated quantile is stored in the attribute ``self.q``.
    Adding a new sample with the method ``add_sample`` will trigger an update of the estimated empirical quantile.
    For multidimensional distributions, the quantiles of the marginal empirical distributions are estimated.
    The P-Square Algorithm has **good accuracy**: above 10,000 samples, the relative error between the estimated empirical estimates
    and the actual population quantiles is typically *way below 1%*.

    Warnings
    --------
    The P-Square Algorithm cannot be vectorised and involves a ``for`` loop of size equal to the dimension of the samples.
    For computational efficiency in high dimensional settings, the ``for`` loop is therefore *jitted* (just-in-time compiled) using
    `Numba's decorator <https://numba.pydata.org/numba-doc/dev/index.html>`_ ``@njit``.

    """

    def __init__(self, pvalue: float):
        r"""

        Parameters
        ----------
        pvalue: float
            P-value of the desired quantile.
        """
        self.pvalue = pvalue
        self.count = 0
        self.marker_heights = None
        self.marker_positions = None
        self.q = None
        self.desired_marker_positions = np.array([1, 1 + 2 * self.pvalue, 1 + 4 * self.pvalue, 3 + 2 * self.pvalue, 5])
        self.increments = np.array([0, self.pvalue / 2, self.pvalue, (self.pvalue + 1) / 2, 1])

    def add_sample(self, sample: Union[float, np.ndarray]):
        r"""
        Update the estimate of the empirical quantile based on the new ``sample``.

        Parameters
        ----------
        sample: np.ndarray
            New empirical sample.
        """
        self.count += 1
        n = self.count
        sample = np.ascontiguousarray(sample)
        if n == 1:
            sample = sample.reshape(sample.size, 1)
            self.marker_heights = sample
        elif (n > 1) and (n <= 5):
            sample = sample.reshape(sample.size, 1)
            self.marker_heights = np.sort(np.concatenate((self.marker_heights, sample), axis=1), axis=1)
            if n == 5:
                self.marker_positions = np.broadcast_to(np.arange(5) + 1, (sample.size, 5)).copy()
        elif n > 5:
            self.desired_marker_positions += self.increments
            self.marker_heights, self.marker_positions = _p2_update(sample, self.marker_heights, self.marker_positions,
                                                                    self.desired_marker_positions)
            self.q = self.marker_heights[:, 2]


@njit
def _p2_update(sample: np.ndarray, marker_heights: np.ndarray, marker_positions: np.ndarray,
               desired_marker_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    for j in range(sample.size):
        q = list(marker_heights[j])
        nj = list(marker_positions[j])
        if sample[j] < q[0]:
            q[0] = sample[j]
            k = 0
        elif q[0] <= sample[j] < q[1]:
            k = 0
        elif q[1] <= sample[j] < q[2]:
            k = 1
        elif q[2] <= sample[j] < q[3]:
            k = 2
        elif q[3] <= sample[j] < q[4]:
            k = 3
        else:
            q[4] = sample[j]
            k = 3
        nj = [nj[t] + 1 * (t > k) for t in range(5)]
        for i in range(1, 4):
            d = desired_marker_positions[i] - nj[i]
            if ((d >= 1) and ((nj[i + 1] - nj[i]) > 1)) or ((d <= -1) and ((nj[i - 1] - nj[i]) < -1)):
                d = np.sign(d)
                q_temp = q[i] + (d / (nj[i + 1] - nj[i - 1])) * (
                        (nj[i] - nj[i - 1] + d) * (q[i + 1] - q[i]) / (nj[i + 1] - nj[i]) + (
                        nj[i + 1] - nj[i] - d) * (q[i] - q[i - 1]) / (nj[i] - nj[i - 1]))
                if q[i - 1] < q_temp < q[i + 1]:
                    q[i] = q_temp
                else:
                    q[i] = q[i] + d * (q[i + np.int64(d)] - q[i]) / (nj[i + np.int64(d)] - nj[i])
                nj[i] += d
        marker_heights[j] = np.array(q)
        marker_positions[j] = np.array(nj)
    return marker_heights, marker_positions

if __name__=='__main__':
    import numpy as np
    import scipy.stats
    from pycsou.util import P2Algorithm
    rng = np.random.default_rng(0)
    population_quantile = scipy.stats.norm.ppf(0.95)

    def generate_sample(n):
        for i in range(n):
            yield rng.standard_normal()

    p2 = P2Algorithm(pvalue=0.95)
    samples = []
    for sample in generate_sample(1000):
        p2.add_sample(sample)
        samples.append(sample)

    print(
    f'Approx. P2 Empirical Quantile: {p2.q}, Actual Empirical Quantile: {np.quantile(samples, 0.95)}, Population Quantile: {population_quantile}.')
