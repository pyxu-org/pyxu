.. _installation:

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
