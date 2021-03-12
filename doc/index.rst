.. image:: /images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://github.com/matthieumeo/pycsou

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


*Pycsou* is a Python 3 package for solving linear inverse problems with state-of-the-art proximal algorithms.
The software implements in a highly modular way the main building blocks -cost functionals, penalty terms and linear operators- of generic penalised convex optimisation problems.

This Python library is inspired by the MATLAB `GlobalBioIm <https://github.com/Biomedical-Imaging-Group/GlobalBioIm>`_ project. The ``LinearOperator`` interface is based on `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_  and `Pylops <https://pylops.readthedocs.io/en/latest/index.html>`_.

Inverse Problems in a Nutshell
------------------------------

.. seealso::
   See :ref:`theory` for a more in-depth discussion on the topic.

Most real-life approximation problems can be formulated as *linear inverse problems*: 

.. math:: 

   \text{Find  }\mathbf{\alpha}\in\mathbb{R}^N &\text{ from noisy measurements } \mathbf{y}\sim \mathbf{Y}\\ 
   &\text{ where } \mathbb{E}[\mathbf{Y}]=\mathbf{G}\mathbf{\alpha}\\
   &\text{ and } \mathbf{G}:\mathbb{R}^N\to \mathbb{R}^L \text{ is a linear operator.}

Such problems are in general **ill-posed**, i.e. attempting to directly solve the linear system :math:`\mathbf{y}=\mathbf{G}\alpha` may yield no solution, infinitely many solutions, or numerically unstable solutions. Therefore, linear inverse problems are commonly solved by means of a penalised optimisation problem, confronting the physical evidence to the analyst’s a priori beliefs about the solution (e.g. smoothness, sparsity) via a data-fidelity and regularisation term, respectively: 

.. math::
   :label: penalised_opt

   \min_{\mathbf{\alpha}\in\mathbb{R}^N} \,F(\mathbf{y}, \mathbf{G} \mathbf{\alpha})\quad+\quad \lambda\mathcal{R}(\mathbf{\alpha}).

* :math:`F:\mathbb{R}^L\times \mathbb{R}^L\rightarrow \mathbb{R}_+\cup\{+\infty\}` is called a **loss functional**, measuring the discrepancy between the observed and predicted measurements :math:`\mathbf{y}` and :math:`\mathbf{G}\mathbf{\alpha}` respectively.
* :math:`\mathcal{R}:\mathbb{R}^N\to \mathbb{R}_+\cup\{+\infty\}` is a **penalty functional** favouring simple and well-behaved solutions (typically with a finite number of degrees of freedom). 
* :math:`\lambda>0` is a **penalty parameter** which controls the amount of regularisation by putting the regularisation functional and the cost functional on a similar scale. 
  
.. warning::
   
   In this package, we assume the loss and penalty functionals to be proper, convex and  lower semi-continuous. 

Features
--------

Pycsou makes it very easy to construct and solve penalised optimisation problems à la :eq:`penalised_opt`:

1. It offers a rich collection of linear operators, loss functionals and penalty functionals commonly used in practice. [#f1]_ 
2. It implements *arithmetic operations* [#f2]_ for linear operators, loss functionals and penalty functionals, hence allowing to *add*, *substract*, *scale*, *compose*, *exponentiate* or *stack* [#f3]_ those various objects with one another and hence quickly design *custom complex optimisation problems*. 
3. It implements a rich collection of **state-of-the-art iterative proximal algorithms**, [#f4]_  including **efficient primal-dual splitting methods** which involve only gradient steps, proximal steps and simple linear evaluations. 
4. It supports *matrix-free linear operators*, [#f5]_ making it easy to work with large scale linear operators that *may not necessarily fit in memory*. Matrix-free linear operators can be implemented from scratch by subclassing the asbtract class :py:class:`~pycsou.core.linop.LinearOperator`, or built [#f6]_ from `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-classes>`_, `distributed Dask arrays <https://docs.dask.org/en/latest/array.html>`_ or `Pylops matrix-free operators <https://pylops.readthedocs.io/en/latest/api/index.html#linear-operators>`_ (which now support GPU computations).
5. It implements *automatic differentiation/proximation rules* [#f7]_, allowing to automatically compute the derivative/proximal operators of functionals constructed from arithmetic operations on common functionals shipped with Pycsou.
6. It leverages powerful rule-of-thumbs for **setting automatically** the hyper-parameters of the provided proximal algorithms. 
7. Pycsou is designed to easily interface with the packages `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_  and `Pylops <https://pylops.readthedocs.io/en/latest/index.html>`_. This allows to use the sparse linear algebra routines from ``scipy.sparse`` on Pycsou :py:class:`~pycsou.core.linop.LinearOperator`, and  benefit from the `large catalogue <https://pylops.readthedocs.io/en/latest/api/index.html>`_ of linear operators and solvers from ``Pylops``. 
   
Usage
-----

Example 1
~~~~~~~~~

Consider the following optimisation problem:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}_+^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda_1 \|\mathbf{D}\mathbf{x}\|_1\quad+\quad\lambda_2 \|\mathbf{x}\|_1,

with :math:`\mathbf{D}\in\mathbb{R}^{N\times N}` the discrete derivative operator and :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda_1,\lambda_2>0.`
This problem can be solved via the :py:class:`~pycsou.opt.proxalgs.PrimalDualSplitting` algorithm  with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2`, :math:`\mathcal{G}(\mathbf{x})=\lambda_2\|\mathbf{x}\|_1,`
:math:`\mathcal{H}(\mathbf{x})=\lambda \|\mathbf{x}\|_1` and :math:`\mathbf{K}=\mathbf{D}`.

.. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from pycsou.linop.diff import FirstDerivative
        from pycsou.func.loss import SquaredL2Loss
        from pycsou.func.penalty import L1Norm, NonNegativeOrthant
        from pycsou.linop.sampling import DownSampling
        from pycsou.opt.proxalgs import PrimalDualSplitting

        x = np.repeat([0, 2, 1, 3, 0, 2, 0], 10)
        D = FirstDerivative(size=x.size, kind='forward')
        D.compute_lipschitz_cst(tol=1e-3)
        rng = np.random.default_rng(0)
        G = DownSampling(size=x.size, downsampling_factor=3)
        G.compute_lipschitz_cst()
        y = G(x)
        l22_loss = (1 / 2) * SquaredL2Loss(dim=G.shape[0], data=y)
        F = l22_loss * G
        lambda_ = 0.1
        H = lambda_ * L1Norm(dim=D.shape[0])
        G = 0.01 * L1Norm(dim=G.shape[1])
        pds = PrimalDualSplitting(dim=G.shape[1], F=F, G=G, H=H, K=D, verbose=None)
        estimate, converged, diagnostics = pds.iterate()
        plt.figure()
        plt.stem(x, linefmt='C0-', markerfmt='C0o')
        plt.stem(estimate['primal_variable'], linefmt='C1--', markerfmt='C1s')
        plt.legend(['Ground truth', 'PDS Estimate'])
        plt.show()

Example 2
~~~~~~~~~

Consider the *LASSO problem*:

    .. math::

       \min_{\mathbf{x}\in\mathbb{R}^N}\frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2\quad+\quad\lambda \|\mathbf{x}\|_1,

with :math:`\mathbf{G}\in\mathbb{R}^{L\times N}, \, \mathbf{y}\in\mathbb{R}^L, \lambda>0.` This problem can be solved via :py:class:`~pycsou.opt.proxalgs.APGD` with :math:`\mathcal{F}(\mathbf{x})= \frac{1}{2}\left\|\mathbf{y}-\mathbf{G}\mathbf{x}\right\|_2^2` and :math:`\mathcal{G}(\mathbf{x})=\lambda \|\mathbf{x}\|_1`. We have:

    .. math::

       \mathbf{\nabla}\mathcal{F}(\mathbf{x})=\mathbf{G}^T(\mathbf{G}\mathbf{x}-\mathbf{y}), \qquad  \text{prox}_{\lambda\|\cdot\|_1}(\mathbf{x})=\text{soft}_\lambda(\mathbf{x}).

This yields the so-called *Fast Iterative Soft Thresholding Algorithm (FISTA)*, whose convergence is guaranteed for :math:`d>2` and :math:`0<\tau\leq \beta^{-1}=\|\mathbf{G}\|_2^{-2}`.

    .. plot::

       import numpy as np
       import matplotlib.pyplot as plt
       from pycsou.func.loss import SquaredL2Loss
       from pycsou.func.penalty import L1Norm
       from pycsou.linop.base import DenseLinearOperator
       from pycsou.opt.proxalgs import APGD

       rng = np.random.default_rng(0)
       G = DenseLinearOperator(rng.standard_normal(15).reshape(3,5))
       G.compute_lipschitz_cst()
       x = np.zeros(G.shape[1])
       x[1] = 1
       x[-2] = -1
       y = G(x)
       l22_loss = (1/2) * SquaredL2Loss(dim=G.shape[0], data=y)
       F = l22_loss * G
       lambda_ = 0.9 * np.max(np.abs(F.gradient(0 * x)))
       G = lambda_ * L1Norm(dim=G.shape[1])
       apgd = APGD(dim=G.shape[1], F=F, G=None, acceleration='CD', verbose=None)
       estimate, converged, diagnostics = apgd.iterate()
       plt.figure()
       plt.stem(x, linefmt='C0-', markerfmt='C0o')
       plt.stem(estimate['iterand'], linefmt='C1--', markerfmt='C1s')
       plt.legend(['Ground truth', 'LASSO Estimate'])
       plt.show()

Cite
----

For citing this package, please see: 

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


.. rubric:: Footnotes
   
.. [#f1] See :ref:`operators`, :ref:`losses` and :ref:`penalties` respectively.
.. [#f2] See for example the arithmetic methods from the abstract classes :py:class:`~pycsou.core.map.Map`, :py:class:`~pycsou.core.map.DifferentiableMap`, :py:class:`~pycsou.core.linop.LinearOperator`, :py:class:`~pycsou.core.functional.Functional`, :py:class:`~pycsou.core.functional.DifferentiableFunctional` and :py:class:`~pycsou.core.functional.ProximableFunctional`.
.. [#f3] See :py:class:`~pycsou.core.map.MapStack`, :py:class:`~pycsou.core.map.DiffMapStack`, :py:class:`~pycsou.linop.base.LinOpStack`,  or :py:class:`~pycsou.func.base.ProxFuncHStack`.
.. [#f4] See  :ref:`proxalgs`.
.. [#f5] See  :py:class:`~pycsou.linop.base.LinearOperator`.
.. [#f6] See  :py:class:`~pycsou.core.linop.SparseLinearOperator`, :py:class:`~pycsou.core.linop.DaskLinearOperator` and :py:class:`~pycsou.core.linop.PyLopLinearOperator`.
.. [#f7] See :py:class:`~pycsou.core.map.DifferentiableMap` and :py:class:`~pycsou.core.functional.ProximableFunctional` for more on the topic.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   general/install
   general/theory
   general/pycsou_classes
   general/extensions
   general/examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index
   api/other


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: More

   notes/index


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
