# Helper classes/functions related to clustering.
#
# These are low-level routines NOT meant to be imported by default via `import pyxu.math`.
# Import this module when/where needed only.

import concurrent.futures as cf

import numba
import numpy as np
import scipy.spatial as spl

BBoxDim = tuple[float]
PointIndex = np.ndarray[int]
ClusterMapping = dict[int, PointIndex]


def grid_cluster(
    x: np.ndarray,
    bbox_dim: BBoxDim,
) -> ClusterMapping:
    """
    Split D-dimensional points onto lattice-aligned clusters.
    Each cluster may contain arbitrary-many points.

    Parameters
    ----------
    x: ndarray[float]
        (M, D) point cloud.
    bbox_dim: BBoxDim
        (D,) box dimensions.

    Returns
    -------
    clusters: ClusterMapping
        (cluster_id, index) pairs.
        `index` contains indices of `x` which lie in cluster `cluster_id`.
    """
    (M, D), dtype = x.shape, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=dtype)
    assert (len(bbox_dim) == D) and (bbox_dim > 0).all()

    # Quick exit if only one point.
    if M == 1:
        clusters = {0: np.r_[0]}
        return clusters

    # Compute (multi,flat) cluster index of each point
    cM_idx, lattice_shape = _digitize(x, bbox_dim)  # multi-index
    cF_idx = np.ravel_multi_index(cM_idx.T, lattice_shape, mode="clip")  # flat-index

    # re-order & count points
    cl_count, idx = _count_sort(cF_idx, k=np.prod(lattice_shape))

    clusters, start = dict(), 0
    for i, step in enumerate(cl_count):
        if step > 0:
            clusters[i] = idx[start : start + step]
            start += step

    return clusters


def bisect_cluster(
    x: np.ndarray,
    clusters: ClusterMapping,
    N_max: int,
) -> ClusterMapping:
    """
    Hierarchically split clusters until each contains at most `N_max` points.

    Clusters are split in parallel.

    Parameters
    ----------
    x: ndarray[float]
        (M, D) point cloud.
    clusters: ClusterMapping
        (cluster_id, index) pairs.
    N_max: int
        Maximum number of points allocated per cluster.

    Returns
    -------
    bisected: ClusterMapping
        (cluser_id, index) pairs where `N_max` constraint is satisfied.
    """
    assert N_max > 0

    completed = []
    to_bisect = []
    for x_idx in clusters.values():
        if len(x_idx) <= N_max:
            completed.append(x_idx)
        else:
            to_bisect.append(x_idx)

    def _bisect(x_idx: np.ndarray) -> ClusterMapping:
        # Split point cloud
        _x = x[x_idx]
        bbox_dim = np.ptp(_x, axis=0) / 2
        _clusters = grid_cluster(_x, bbox_dim)

        # re-label output indices
        for cl_idx, sub_idx in _clusters.items():
            _clusters[cl_idx] = x_idx[sub_idx]
        return _clusters

    with cf.ThreadPoolExecutor() as executor:
        while (N_task := len(to_bisect)) > 0:
            fs = [executor.submit(_bisect, to_bisect.pop()) for _ in range(N_task)]

            # Schedule next hierarchy
            for f in cf.as_completed(fs):
                _clusters = f.result()
                for x_idx in _clusters.values():
                    if len(x_idx) <= N_max:
                        completed.append(x_idx)
                    else:
                        to_bisect.append(x_idx)

    bisected = {k: v for (k, v) in enumerate(completed)}
    return bisected


def fuse_cluster(
    x: np.ndarray,
    clusters: ClusterMapping,
    bbox_dim: BBoxDim,
) -> ClusterMapping:
    """
    Fuse neighboring clusters until aggregate bounding-boxes have at most size `bbox_dim`.

    Parameters
    ----------
    x: ndarray[float]
        (M, D) point cloud.
    clusters: ClusterMapping
        (cluster_id, index) pairs.
    bbox_dim: BBoxDim
        (D,) maximum (fused) box dimensions.

    Returns
    -------
    fused: ClusterMapping
        (cluster_id, index) fused pairs.
    """
    (_, D), dtype = x.shape, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=dtype)
    assert (len(bbox_dim) == D) and (bbox_dim > 0).all()

    # Center points around origin
    x = x.copy()
    x -= (x.min(axis=0) + x.max(axis=0)) / 2

    # Rescale points to have equal spread in each dimension.
    # Reason: KDTree only accepts scalar-valued radii.
    x /= bbox_dim
    bbox_dim = np.ones_like(bbox_dim)

    # Compute cluster centroids & tight box boundaries
    clusters = list(clusters.values())  # fusion doesn't care about cluster IDs.
    centroid = np.zeros((len(clusters), D), dtype=dtype)
    tbbox_dim = np.zeros((len(clusters), D), dtype=dtype)  # tight bbox_dim(s)
    for i, x_idx in enumerate(clusters):
        _x = x[x_idx]
        _x_min = _x.min(axis=0)
        _x_max = _x.max(axis=0)
        centroid[i] = (_x_min + _x_max) / 2
        tbbox_dim[i] = _x_max - _x_min

    # Fuse clusters which are closely-spaced & small-enough
    fuse_clusters = True
    while fuse_clusters:
        # Find fuseable centroid pairs
        c_tree = spl.KDTree(centroid)  # centroid_tree
        candidates = c_tree.query_pairs(
            r=bbox_dim[0] / 2,
            p=np.inf,
            output_type="ndarray",
        )
        _i, _j = candidates.T
        c_spacing = np.abs(centroid[_i] - centroid[_j])
        offset = (tbbox_dim[_i] + tbbox_dim[_j]) / 2
        fuseable = np.all(c_spacing + offset < bbox_dim, axis=1)
        candidates = candidates[fuseable]

        # If a centroid can be fused with multiple others, restrict choice to single pair
        seen, fuse = set(), set()
        for _i, _j in candidates:
            if (_i not in seen) and (_j not in seen):
                seen |= {_i, _j}
                fuse.add((_i, _j))
        fuse_clusters = len(fuse) > 0

        # (clusters,centroid,tbbox_dim): update _i entries.
        for _i, _j in fuse:
            clusters[_i] = np.r_[clusters[_i], clusters[_j]]
            _x = x[clusters[_i]]
            _x_min = _x.min(axis=0)
            _x_max = _x.max(axis=0)
            centroid[_i] = (_x_min + _x_max) / 2
            tbbox_dim[_i] = _x_max - _x_min

        # (clusters,centroid,tbbox_dim): drop _j entries.
        for _j in sorted({_j for (_i, _j) in fuse})[::-1]:
            clusters.pop(_j)
        c_idx = np.setdiff1d(  # indices to keep
            np.arange(len(centroid)),
            [_j for (_i, _j) in fuse],  # indices to drop
        )
        centroid = centroid[c_idx]
        tbbox_dim = tbbox_dim[c_idx]

    fused = {c_idx: x_idx for (c_idx, x_idx) in enumerate(clusters)}
    return fused


# Internal Helper functions ---------------------------------------------------
_nb_flags = dict(
    nopython=True,
    nogil=True,
    cache=True,
    forceobj=False,
    parallel=False,
    error_model="numpy",
    fastmath=True,
    locals={},
    boundscheck=False,
)


@numba.jit(**_nb_flags)
def _minmax(x: np.ndarray) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     (x.min(axis=0), x.max(axis=0))
    #
    # Parameters
    # ----------
    # x: ndarray[float32/64]
    #     (M, D)
    #
    # Returns
    # -------
    # x_min, x_max: ndarray[float32/64]
    #     (D,) axial min/max values
    M, D = x.shape
    x_min = np.full(D, fill_value=np.inf, dtype=x.dtype)
    x_max = np.full(D, fill_value=-np.inf, dtype=x.dtype)

    for m in range(M):
        for d in range(D):
            if x[m, d] < x_min[d]:
                x_min[d] = x[m, d]
            if x[m, d] > x_max[d]:
                x_max[d] = x[m, d]
    return x_min, x_max


@numba.jit(**_nb_flags)
def _digitize(x: np.ndarray, bbox_dim: np.ndarray) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     x_min, x_max = x.min(axis=0), x.max(axis=0)
    #     lattice_shape = np.maximum(1, np.ceil((x_max - x_min) / bbox_dim)).astype(int)
    #     c_idx = ((x - x_min) / bbox_dim).astype(int)  # (M, D)
    #
    # Parameters
    # ----------
    # x: ndarray[float32/64]
    #     (M, D)
    # bbox_dim: ndarray[float32/64]
    #     (D,)
    #
    # Returns
    # -------
    # c_idx: ndarray[int]
    #     (M, D) integer bins each element belongs to.
    # lattice_shape: ndarray[int]
    #     (D,) bin count per dimension.
    M, D = x.shape
    x_min, x_max = _minmax(x)

    lattice_shape = np.zeros(D, dtype=np.int64)
    for d in range(D):
        lattice_shape[d] = max(1, np.ceil((x_max[d] - x_min[d]) / bbox_dim[d]))

    c_idx = np.zeros((M, D), dtype=np.int64)
    for m in range(M):
        for d in range(D):
            c_idx[m, d] = (x[m, d] - x_min[d]) / bbox_dim[d]

    return c_idx, lattice_shape


@numba.jit(**_nb_flags)
def _count_sort(x: np.ndarray, k: int) -> tuple[np.ndarray]:
    # Computes code below more efficiently:
    #     idx = np.argsort(x)
    #     _, count = np.unique(x, return_counts=True)
    #
    # Parameters
    # ----------
    # x: ndarray[int]
    #     (N,) non-negative integers.
    # k: int
    #     Upper bound on values in `x`.
    #
    # Returns
    # -------
    # count: ndarray[int]
    #     (k,) counts of each element in `x`.
    # idx: ndarray[int]
    #     (N,) indices to sort x into ascending order.
    count = np.zeros(k, dtype=np.int64)
    for _x in x:
        count[_x] += 1

    # Write-index for each category
    w_idx = np.zeros(k, dtype=np.int64)
    w_idx[0] = 0
    w_idx[1:] = np.cumsum(count)[: k - 1]

    N = len(x)
    idx = np.zeros(N, dtype=x.dtype)
    for i, _x in enumerate(x):
        idx[w_idx[_x]] = i
        w_idx[_x] += 1

    return count, idx
