.. _extensions:

##########
Extensions
##########

Pycsou's core functionalities can be extended via the following specialised extension modules.

Pycsou-sphere
=============

`Pycsou-sphere <https://github.com/matthieumeo/pycsou-sphere>`_ is an extension module for solving linear inverse problems on the sphere.
The extension offers implementations of spherical zonal *convolution* operators as well as the spherical harmonic and Fourier-Legendre transforms (all compatible with Pycsou's interface for linear operators).
It also provides numerical routines for computing the Green kernels of common spherical pseudo-differential operators and generating spherical meshes/point sets.

This module heavily relies and follows similar conventions as the `healpy <https://healpy.readthedocs.io/en/latest/index.html>`_ package for spherical signal processing with Python.

The package (available on PyPI under the name `pycsphere <https://pypi.org/project/pycsphere>`_) is organised as follows:

1. The subpackage ``pycsphere.linop`` implements the following common spherical linear operators:

   * Convolution/pooling operators,
   * Spherical transforms and their inverses,
   * Finite-difference spherical differential operators.

2. The subpackage ``pycsphere.mesh`` provides routines for generating spherical meshes.
3. The subpackage ``pycsphere.green`` provides numerical routines for computing the Green  kernels of common spherical pseudo-differential operators.

The package's documentation is available at `this link <https://matthieumeo.github.io/pycsou-sphere/html/>`_.

Pycsou-gsp
==========

`Pycsou-gsp <https://github.com/matthieumeo/pycsou-gsp>`_ is the graph signal processing extension of Pycsou.
The extension offers implementations of graph *convolution* and *differential* operators, compatible with Pycsou's interface for linear operators.
Such tools can be useful when solving linear inverse problems involving signals defined on non Euclidean discrete manifolds.

Graphs in *Pycsou-gsp* are instances from the class ``pygsp.graphs.Graph`` from the `pygsp <https://github.com/epfl-lts2/pygsp>`_ library for graph signal processing with Python.

The package (available on PyPI under the name `pycgsp <https://pypi.org/project/pycgsp>`_) is organised as follows:

1. The subpackage ``pycgsp.linop`` implements the following common graph linear operators:

   * Graph convolution operators: ``GraphConvolution``
   * Graph differential operators: ``GraphLaplacian``, ``GraphGradient``, ``GeneralisedGraphLaplacian``.

2. The subpackage ``pycgsp.graph`` provides routines for generating graphs from discrete tessellations of continuous manifolds such as the sphere.

The package's documentation is available at this `link <https://matthieumeo.github.io/pycsou-gsp/html/>`_.
