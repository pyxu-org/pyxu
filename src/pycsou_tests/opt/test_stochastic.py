import collections.abc as cabc
import functools

import dask.array as da
import numpy as np
import pytest

import pycsou.opt.stochastic as pystoc
from pycsou.operator.linop.base import HomothetyOp

# Want to test that the neighbors_map_overlap is working as expected
# test stack_dims stuff
# test indices
# test overlap


@pytest.mark.parametrize(
    ["x", "op", "ind", "overlap", "stack_dims", "block_info", "out"],
    [
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, 0),
            False,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # correct id, no overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, 1),
            False,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(1),
        ],  # incorrect id, no overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (-1, -1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # correct id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (-1, 0),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (-1, 1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, -1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, 0),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, 1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (1, -1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (1, 0),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (1, 1),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(2),
        ],  # overlap id, overlap
        [
            np.array(1),
            HomothetyOp(2, 1),
            (0, 2),
            True,
            (),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.array(1),
        ],  # incorrect id, overlap
        [
            np.ones(8).reshape(2, 2, 2),
            HomothetyOp(2, 2),
            (0, 0),
            True,
            (2, 2),
            {0: {"chunk-location": (0, 0), "array-location": [(0, 1), (0, 1)]}},
            np.ones(8).reshape(2, 2, 2) * 2,
        ],  # stack dims
    ],
)
def test_neighbors_map_overlap(x, op, ind, overlap, stack_dims, block_info, out):
    assert np.allclose(pystoc.neighbors_map_overlap(x, op, ind, overlap, stack_dims, block_info), out)


@pytest.mark.parametrize(
    ["arr", "op", "ind", "depth", "out"],
    [
        [da.ones((3), chunks=(1,)), HomothetyOp(2, 3), (0,), None, np.array([2, 1, 1])],  # no overlap
        [da.ones((3), chunks=(1,)), HomothetyOp(2, 3), (0,), {0: 1}, np.array([2, 2, 1])],  # overlap
        [da.ones((3), chunks=(1,)), HomothetyOp(2, 3), (1,), {0: 1}, np.array([2, 2, 2])],  # overlap
        [
            da.ones((2, 2, 3), chunks=(2, 2, 1)),
            HomothetyOp(2, 2),
            (0, 0, 0),
            None,
            np.array([[[2, 1, 1], [2, 1, 1]], [[2, 1, 1], [2, 1, 1]]]),
        ],  # stack dimensions
        [
            da.ones((2, 2, 3), chunks=(2, 2, 1)),
            HomothetyOp(2, 2),
            (0, 0, 0),
            {0: 0, 1: 0, 2: 1},
            np.array([[[2, 2, 1], [2, 2, 1]], [[2, 2, 1], [2, 2, 1]]]),
        ],  # stack dimensions, overlap
    ],
)
def test_dask_neighbors_map_overlap(arr, op, ind, depth, out):
    input_shape = arr.shape
    n_map_overlap = functools.partial(
        pystoc.neighbors_map_overlap,
        op=op,
        ind=ind,
        stack_dims=input_shape[:-1],
        overlap=bool(depth),
    )
    _out = da.map_overlap(
        n_map_overlap,
        arr,
        depth=depth,
        boundary=0,
        # allow_rechunk=False,- released 2022.6.1 DASK
        meta=np.array(()),
        dtype=arr.dtype,
    )
    assert np.allclose(_out, out)
