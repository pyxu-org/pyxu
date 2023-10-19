Comparison with other Frameworks
================================

This document aims to provide a comprehensive comparison between Pyxu and its main contenders in the field of
computational imaging frameworks.  Various attributes such as features, maturity, ease-of-use, and support for
distributed and GPU computing are discussed.

Main Contenders
---------------

- `PyLops <https://github.com/PyLops/pylops>`_: PyLops is an open-source Python library focused on providing a
  backend-agnostic, idiomatic, matrix-free library of linear operators and related computations.
- `PyProximal <https://github.com/PyLops/pyproximal>`_: Proximal Operators and Algorithms in Python.
- `ODL <https://github.com/odlgroup/odl>`_: The *Operator Discretization Library (ODL)* is a Python library that enables
  research in inverse problems on realistic or real data.
- `SCICO <https://github.com/lanl/scico>`_: SCICO is a JAX-powered Python package for solving the inverse problems that
  arise in scientific imaging applications.
- `DeepInv <https://deepinv.github.io/deepinv/>`_: Deep Inverse is a Pytorch based library for solving imaging inverse
  problems with deep learning.
- `SigPy <https://github.com/mikgroup/sigpy>`_: SigPy is a package for signal processing, with emphasis on iterative
  methods. It is built on NumPy and CuPy.
- `GlobalBioIm <https://biomedical-imaging-group.github.io/GlobalBioIm/>`_ (MATLAB): A unifying Matlab framework for the
  development of reconstruction algorithms (solving inverse problems) in computational imaging.

.. note::

   This comparison excludes traditional medical imaging frameworks like `TomoPy
   <https://tomopy.readthedocs.io/en/latest/>`_ and `ASTRA <https://www.astra-toolbox.com/>`_, as well as
   general-purpose optimization frameworks (e.g., `scipy.optimize
   <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_, `MATLAB's Optimization
   <https://www.mathworks.com/help/optim/ug/optimization-toolbox.html>`_ or `CVX toolboxes <http://cvxr.com/cvx/>`_) and
   deep learning frameworks (e.g., `scikit-learn <https://scikit-learn.org/stable/>`_, `PyTorch
   <https://pytorch.org/>`_, `TensorFlow <https://www.tensorflow.org/>`_, `Keras <https://keras.io/>`_).  These
   exclusions are due to their architectural limitations or the narrow focus of their optimization methods, which are
   unsuitable for the diverse requirements of modern computational imaging.

Comparative Analysis
--------------------

Features and Maturity
+++++++++++++++++++++

Pyxu offers a comprehensive suite of algorithms, including the latest primal-dual splitting methods for hybrid
optimization.  The feature set is robust and mature, positioning it as a leader in the computational imaging arena.

.. include:: feature_maturity_comparison.rst

Ease-of-Use
+++++++++++

Pyxu stands out in its ease-of-use, providing well-documented code and a user-friendly interface.  Most of its
contenders either lack comprehensive documentation or have steeper learning curves.

Support for HPC Computing
+++++++++++++++++++++++++

Pyxu is unique in supporting both out-of-core and distributed computing. Additionally, it offers robust support for JIT
compilation and GPU computing via Numba and CuPy respectively. Most contenders either offer partial support or lack
these features altogether.

.. include:: hpc_features_comparison.rst

SCICO: A Closer Look
--------------------

Although SCICO is almost feature-complete, it relies on `JAX <https://github.com/google/jax>`_, which has some
drawbacks:

1. **Experimental Framework**: JAX is relatively young and is still officially considered experimental. If you're
   looking for a stable, long-term solution, this could be a concern.

2. **Expertise Required**: Working with JAX requires a deep understanding of functional programming to avoid pitfalls
   and debugging headaches.

3. **CPU Optimization**: JAX is not optimized for CPU computing, making it less versatile than NumPy in some scenarios.

4. **Platform Support**: JAX doesn't support Windows, limiting its adoption among those who use Windows-based systems.

DeepInv: A Note on Usability
----------------------------

DeepInv is based on `PyTorch <https://pytorch.org/>`_, making it less portable due to its numerous dependencies.
Moreover, it's primarily designed for deep learning users, making it less accessible for imaging scientists who may not
be as well-versed in deep learning paradigms.

Conclusion
----------

While all the frameworks discussed here have their merits, Pyxu appears to offer the most well-rounded set of features,
robustness, and ease-of-use.  Its support for distributed and GPU computing adds to its advantages, making it a leading
choice for computational imaging applications.
