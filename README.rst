.. image:: https://matthieumeo.github.io/pycsou/html/_images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://matthieumeo.github.io/pycsou/html/index


.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581

*Pycsou* is a Python 3 package for solving linear inverse problems with state-of-the-art proximal algorithms. The software implements in a highly modular way the main building blocks -cost functionals, penalty terms and linear operators- of generic penalised convex optimisation problems.

Pycsou's documentation is available at https://matthieumeo.github.io/pycsou/html/index

This Python library is inspired by the MATLAB `GlobalBioIm <https://github.com/Biomedical-Imaging-Group/GlobalBioIm>`_ project. The ``LinearOperator`` interface is based on `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_  and `Pylops <https://pylops.readthedocs.io/en/latest/index.html>`_.

Functionalities
===============

Pycsou makes it very easy to construct and solve penalised optimisation problems:

1. It offers a rich collection of linear operators, loss functionals and penalty functionals commonly used in practice.
2. It implements *arithmetic operations* for linear operators, loss functionals and penalty functionals, hence allowing to *add*, *substract*, *scale*, *compose*, *exponentiate* or *stack* those various objects with one another and hence quickly design *custom complex optimisation problems*. 
3. It implements a rich collection of **state-of-the-art iterative proximal algorithms**, including **efficient primal-dual splitting methods** which involve only gradient steps, proximal steps and simple linear evaluations. 
4. It supports *matrix-free linear operators*, making it easy to work with large scale linear operators that *may not necessarily fit in memory*. Matrix-free linear operators can be implemented from scratch by subclassing the asbtract class ``LinearOperator``, or built from `Scipy sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html#sparse-matrix-classes>`_, `distributed Dask arrays <https://docs.dask.org/en/latest/array.html>`_ or `Pylops matrix-free operators <https://pylops.readthedocs.io/en/latest/api/index.html#linear-operators>`_ (which now support GPU computations).
5. It implements *automatic differentiation/proximation rules*, allowing to automatically compute the derivative/proximal operators of functionals constructed from arithmetic operations on common functionals shipped with Pycsou.
6. It leverages powerful rule-of-thumbs for **setting automatically** the hyper-parameters of the provided proximal algorithms. 
7. Pycsou is designed to easily interface with the packages `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_  and `Pylops <https://pylops.readthedocs.io/en/latest/index.html>`_. This allows to use the sparse linear algebra routines from ``scipy.sparse`` on Pycsou ``LinearOperator``, and  benefit from the `large catalogue <https://pylops.readthedocs.io/en/latest/api/index.html>`_ of linear operators and solvers from ``Pylops``. 
   

Installation
============

Pycsou requires Python 3.6 or greater.It is developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------

The package dependencies are listed in the files ``requirements.txt`` and ``requirements-conda.txt``. 
It is recommended to install dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com/download/#linux>`_. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (a highly optimized BLAS library created by Intel).

To do so we create an environment named ``pycsou`` (you can change that if you wish) and equip it 
with the necessary requirements: 

.. code-block:: bash
   
   >> conda create -n pycsou python=3.6
   >> conda install -n pycsou --channel=conda-forge --file=requirements-conda.txt
   >> conda activate pycsou



Quick Install
-------------

Pycsou is available on `Pypi <https://pypi.org/project/pycsou/>`_. You can hence install it very simply via the command: 

.. code-block:: bash
   
   >> pip install pycsou

If you have previously activated your conda environment ``pip`` will install Pycsou in said environment. Otherwise it will install it in your base environment together with the various dependencies obtained from the file ``requirements.txt``.


Developper Install
------------------

It is also possible to install Pycsou from the source for developpers: 


.. code-block:: bash
   
   >> git clone https://github.com/matthieumeo/pycsou
   >> cd <repository_dir>/
   >> pip install -e .

The package documentation can be generated with: 

.. code-block:: bash
   
   >> conda install -n pycsou sphinx=='2.1.*'            \
                    sphinx_rtd_theme=='0.4.*'
   >> conda activate pycsou
   >> python3 setup.py build_sphinx  

You can verify that the installation was successful by running the package doctests: 

.. code-block:: bash
   
   >> conda activate pycsou
   >> python3 test.py

Cite
----
For citing this package, please see: http://doi.org/10.5281/zenodo.4486431

