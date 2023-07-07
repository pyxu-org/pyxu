:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. raw:: html

    <!-- CSS overrides on the homepage only -->
    <style>
    .bd-main .bd-content .bd-article-container {
    max-width: 70rem; /* Make homepage a little wider instead of 60em */
    }
    /* Extra top/bottom padding to the sections */
    article.bd-article section {
    padding: 3rem 0 7rem;
    }
    /* Override all h1 headers except for the hidden ones */
    h1:not(.sd-d-none) {
    font-weight: bold;
    font-size: 48px;
    text-align: center;
    margin-bottom: 4rem;
    }
    /* Override all h3 headers that are not in hero */
    h3:not(#hero h3) {
    font-weight: bold;
    text-align: center;
    }

    p {
    text-align: justify;
    }
    </style>

.. raw:: html

    <div id="hero">
    <div id="hero-left">  <!-- Start Hero Left -->

Pyxu
====

.. raw:: html 

    <h2 style="font-size: 60px; font-weight: bold; display: inline"><span>Pyxu</span></h2>
    <h3 style="margin-top: 0; font-weight: bold; text-align: left; ">Modular Computational Imaging in Python</h3>
    <p>
    <b> Pyxu </b> (pronounced [piksu], formerly known as Pycsou) is an open-source software framework for Python
    allowing scientists at any level to quickly prototype/deploy <em> hardware accelerated and distributed </em> computational imaging pipelines at scale.
    <br>
    Thanks to its hardware-agnostic <b>microservice architecture </b> and its tight integration with the PyData ecosystem, 
    Pyxu supports a wide range of imaging applications, scales and compute architectures.
    </p>
    
    <div class="homepage-button-container">
    <div class="homepage-button-container-row">
        <a href="./getting_started/index.html" class="homepage-button primary-button">Get Started</a>
        <a href="./examples/index.html" class="homepage-button secondary-button">See Examples</a>
    </div>
    <div class="homepage-button-container-row">
        <a href="./api/index.html" class="homepage-button-link">See API Reference →</a>
    </div>
    </div>
    </div>  <!-- End Hero Left -->

.. raw:: html 

    <div id="hero-right">  <!-- Start Hero Right -->

.. image:: _static/microservice_hero.png


.. raw:: html

    </div>  <!-- End Hero Right -->
    </div>  <!-- End Hero -->
    <div style="padding-bottom: 60px;">

.. grid:: 4 4 8 8
    :gutter: 2

    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_denoising.png
    
    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0    
        :img-background: ./_static/grid_deblurring.png
    
    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_inpainting.png

    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_superresolution.png    

    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_demultiplexing.png    
    
    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_interferometry.png    
    
    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_fusion.png    
    
    .. grid-item-card::
        :shadow: none
        :class-card: sd-border-0
        :img-background: ./_static/grid_tomography.png    
    


.. raw:: html

    </div> 



Key Features & Capabilities
===========================

.. grid:: 2 2 2 3
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/microservice.png
               :scale: 5%
               
               **Microservice architecture**
                
               Loosely coupled software components composable via an advanced operator algebra. 


    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/pnp.png
               :scale: 5%
               
               **Plug-and-play API**
                
               Simple interface for beginners with a handful of easily interpretable parameters to set, 
               and *guru* interface for experts.

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/scope.png
               :scale: 4%
               
               **Application agnostic**
                
               Generic software components with wide applicability across modalities.
    
    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/hybrid.png
               :scale: 4%
               
               **Run anywhere**
                
               The same code executes on multiple backends, including CPU and GPU.


    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/hpc.png
               :scale: 5%
               
               **High performance computing**
                
               Just-in-time compilation, batch processing, automatic parallelization, out-of-core computing,
               and controllable compute precision.


    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/interop.png
               :scale: 4%
               
               **Interoperability**
                
               Pyxu is highly interoperable with the *PyData stack* and full-fledged zero-copy wrappers for `JAX <https://jax.readthedocs.io/en/latest/>`_ and `PyTorch <https://pytorch.org/>`_.

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/test.png
               :scale: 4%
               
               **Quality controlled**
                
               Extensive logical and functional testing of software components. Templated test classes for custom operators.



    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/git.png
               :scale: 4%
               
               **Community based**
                
               Pyxu is open source, version controlled and available to all on `PyPI <https://pypi.org/project/pycsou/>`_/`GitHub <https://github.com/matthieumeo/pycsou>`_.  

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/fair.png
               :scale: 4%
               
               **Extensible**
                
               Powerful plugin mechanism and community marketplace for discovering and sharing custom operators.

Ecosystem
=========

Pyxu is part of, looks and feels like, and is highly interoperable with the wider scientific Python ecosystem. It is indeed built on a minimal set of foundational and robust scientific computing 
librairies from the PyData stack. Pyxu supports notably multiple array backends --`NumPy <https://numpy.org/>`_, `Dask <https://www.dask.org/>`_, `Sparse <https://sparse.pydata.org/en/stable/>`_ and 
optionally `CuPy <https://cupy.dev/>`_ (for CUDA devices only), allowing users to choose array backends that work best for their application/computation.
Aside from `SciPy <https://scipy.org/>`_ and  `Numba <https://numba.pydata.org/>`_ --which we use for scientific computing and `JIT-compiling <https://numba.readthedocs.io/en/stable/user/5minguide.html#how-does-numba-work>`_ respectively--
these are Pyxu's **only** dependencies, making the software very easy to ship, install, deploy in production and sustain in the long-term. 

Pyxus is also interoperable with (but does not depend on) the major deep learning frameworks `JAX <https://jax.readthedocs.io/en/latest/>`_ and `PyTorch <https://pytorch.org/>`_, 
allowing users to benefit from the latest incursions of deep learning in the field of computational imaging (e.g., PnP methods, unrolled neural networks, deep generative priors). 
Our wrappers can moreover leverage the autograd engine to auto-infer gradients or adjoints operations. 


.. grid:: 2 2 4 4
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/numpy_logo.svg
               :width: 65%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> NumPy </b> <br/> 
                NumPy is the fundamental package for array computing with Python.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/dask_horizontal.svg
               :width: 65%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> Dask </b> <br/> 
                Distributed arrays and advanced parallelism for analytics, enabling performance at scale.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/cupy.png
               :width: 65%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> CuPy </b> <br/> 
                NumPy-compatible array library for GPU-accelerated computing with Python.
                </p>
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/sparse-logo.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> Sparse </b> <br/> 
                Sparse multi-dimensional arrays for the PyData ecosystem.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/Pytorch_logo.png
               :width: 65%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> PyTorch </b> <br/> 
                Tensors and dynamic neural networks in Python with strong GPU acceleration.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/jax_logo_250px.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: left;">
                <b> JAX </b> <br/> 
                Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
                </p>

.. figure:: ./_static/stack.png
    :width: 70%


Governance and Team
===================
Pyxu is an **open source project** developed and maintained primarily by members of the `EPFL Center for Imaging <https://imaging.epfl.ch/>`_, 
but the repo itself is public and we welcome external contributions. We are committed to keeping the project public and owned by the community.


.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/simeoni.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Matthieu Simeoni </b> <br/> 
                Creator, architect and technical lead.
                </p>

    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/kashani.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Sepand Kashani </b> <br/> 
                Architect and technical lead.
                </p>
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/rue_queralt.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Joan Rué-Queralt </b> <br/> 
                Maintainer and technical lead.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/debarre.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Thomas Debarre </b> <br/> 
                Maintainer and core contributor.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/hamm.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Daniele Hamm </b> <br/> 
                Core contributor.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/jarret.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Adrian Jarret </b> <br/> 
                Core contributor.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/salim.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Salim Najib </b> <br/> 
                Core contributor.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/okumus.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Kaan Okumus </b> <br/> 
                Contributor.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/flowers.png
               :width: 40%
               :align: center


            .. raw:: html 
              
                <p style="text-align: center;">
                <b> Alex Flowers </b> <br/> 
                Contributor.
                </p>

.. grid:: 1 2 4 4
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/EPFL_Logo_Digital_BLACK_PROD.png
               :width: 70%
               :align: center

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/EPFL_Unités_Center-for-imaging.svg
               :width: 70%
               :align: center
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/drawing.svg
               :width: 50%
               :align: center
        
    .. grid-item-card:: 
        :shadow: none
        :class-card: sd-border-0

        .. image:: _static/LCAV_LOGO.png
            :width: 50%
            :align: center

