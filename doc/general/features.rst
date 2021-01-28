########
Features
########

Pycsou makes it very easy to construct and solve penalised optimisation problems à la :eq:`penalised_opt`:

1. It offers a rich collection of linear operators, loss functionals and penalty functionals commonly used in practice. [#f1]_ 
2. It implements *arithmetic operations* [#f2]_ for linear operators, loss functionals and penalty functionals, hence allowing to *add*, *substract*, *scale*, *compose*, *exponentiate* or *stack* [#f3]_ those various objects with one another and hence quickly design *custom complex optimisation problems*. 
3. It implements a rich collection of **state-of-the-art iterative proximal algorithms**, [#f4]_  including **efficient primal-dual splitting methods** which involve only gradient steps, proximal steps and simple linear evaluations. 
4. It supports *matrix-free linear operators*, [#f5]_ making it easy to work with large scale linear operators that *may not necessarily fit in memory*. Matrix-free linear operators can be implemented from scratch by subclassing the asbtract class :py:class:`~pycsou.core.linop.LinearOperator`, or built [#f6]_ from `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-classes>`_, `distributed Dask arrays <https://docs.dask.org/en/latest/array.html>`_ or `Pylops matrix-free operators <https://pylops.readthedocs.io/en/latest/api/index.html#linear-operators>`_ (which now support GPU computations).
5. It implements *automatic differentiation/proximation rules* [#f7]_, allowing to automatically compute the derivative/proximal operators of functionals constructed from arithmetic operations on common functionals shipped with Pycsou.
6. It leverages powerful rule-of-thumbs for **setting automatically** the hyper-parameters of the provided proximal algorithms. 
7. Pycsou is designed to easily interface with the packages `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_  and `Pylops <https://pylops.readthedocs.io/en/latest/index.html>`_. This allows to use the sparse linear algebra routines from ``scipy.sparse`` on Pycsou :py:class:`~pycsou.core.linop.LinearOperator`, and  benefit from the `large catalogue <https://pylops.readthedocs.io/en/latest/api/index.html>`_ of linear operators and solvers from ``Pylops``.
   

.. rubric:: Footnotes
   
.. [#f1] See :ref:`operators`, :ref:`losses` and :ref:`penalties` respectively.
.. [#f2] See for example the arithmetic methods from the abstract classes :py:class:`~pycsou.core.map.Map`, :py:class:`~pycsou.core.map.DifferentiableMap`, :py:class:`~pycsou.core.linop.LinearOperator`, :py:class:`~pycsou.core.functional.Functional`, :py:class:`~pycsou.core.functional.DifferentiableFunctional` and :py:class:`~pycsou.core.functional.ProximableFunctional`.
.. [#f3] See :py:class:`~pycsou.core.map.MapStack`, :py:class:`~pycsou.core.map.DiffMapStack`, :py:class:`~pycsou.linop.base.LinOpStack`,  or :py:class:`~pycsou.func.base.ProxFuncHStack`.
.. [#f4] See  :ref:`proxalgs`.
.. [#f5] See  :py:class:`~pycsou.linop.base.LinearOperator`.
.. [#f6] See  :py:class:`~pycsou.core.linop.SparseLinearOperator`, :py:class:`~pycsou.core.linop.DaskLinearOperator` and :py:class:`~pycsou.core.linop.PyLopLinearOperator`.
.. [#f7] See :py:class:`~pycsou.core.map.DifferentiableMap` and :py:class:`~pycsou.core.functional.ProximableFunctional` for more on the topic.
