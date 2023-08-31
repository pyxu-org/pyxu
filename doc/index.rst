:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. |br| raw:: html
   
   </br>

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

    a:visited {
    color: var(--pst-color-primary);
    }

    .homepage-button.primary-button:visited {
    color: var(--pst-color-background);
    }


    .sponsors-list-item {
    display: inline-flex;
    justify-content: center;
    opacity: 0.5;
    filter: brightness(0.5) grayscale(1);
    }

    @keyframes platformsSlideshow {
    100% {
        transform: translateX(-2000px);
    }
    }
    </style>

.. raw:: html

    <div id="hero">
    <div id="hero-left">  <!-- Start Hero Left -->

Pyxu
====

.. raw:: html 

    <h2 style="font-size: 60px; font-weight: bold; display: inline"><span>Pyxu</span></h2>
    <h3 style="margin-top: 0; font-weight: bold; text-align: left; ">Modular & Scalable Computational Imaging</h3>
    <p>
    <strong> Pyxu </strong> (pronounced [piksu], formerly known as Pycsou) is an open-source Python framework 
    allowing scientists at any level to quickly prototype/deploy <em> hardware accelerated and out-of-core </em> computational imaging pipelines at scale.
    Thanks to its hardware-agnostic <strong>microservice architecture </strong> and its tight integration with the PyData ecosystem, 
    Pyxu supports a wide range of imaging applications, scales, and computation architectures.
    </p>
    
    <div class="homepage-button-container">
    <div class="homepage-button-container-row">
        <a href="./intro/index.html" class="homepage-button primary-button">Get Started</a>
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
               :class: no-scaled-link
            
            .. raw:: html

                <p style="text-align: center;">
                <strong> Microservice architecture </strong> <br/> 
                Loosely coupled software components that are composable via an advanced <em> operator algebra</em>.
                </p>
               


    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/pnp.png
               :scale: 5%
               :class: no-scaled-link
               
            .. raw:: html

                <p style="text-align: center;">
                <strong> Plug-and-play API </strong> <br/> 
                Simple interface for beginners with theory-informed automatic hyperparameter selection.
                Experts may still fine-tune parameters via a <em> guru </em> interface.
                </p>


    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/scope.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Application agnostic </strong> <br/> 
                Generic software components with wide applicability across imaging modalities.
                </p>             
                    
    
    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/hybrid.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Flexible computation backends </strong> <br/> 
                The same code executes for multiple array backends, including CPU and GPU, with a unified, easily maintainable codebase.
                </p>             
                    

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/hpc.png
               :scale: 5%
               :class: no-scaled-link
               
            .. raw:: html

                <p style="text-align: center;">
                <strong> High-performance computing </strong> <br/> 
                Just-in-time compilation, batch processing, automatic parallelization, out-of-core computing,
                and controllable computation precision.
                </p>             
                                   

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/interop.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Interoperability </strong> <br/> 
                Pyxu is highly interoperable with the <em>PyData stack</em>, including full-fledged zero-copy wrappers for 
                <a href="https://jax.readthedocs.io/en/latest/">JAX</a> and <a href="https://pytorch.org/">PyTorch</a> operators.
                </p>             

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/test.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Quality controlled </strong> <br/> 
                Extensive logical and functional unit testing of software components. Templated test classes for custom operators.
                </p>             

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/git.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Community driven </strong> <br/> 
                Pyxu is open source, version controlled, and is available to all on 
                <a href="https://pypi.org/project/pyxu/">PyPI</a> and <a href="https://github.com/matthieumeo/pyxu">GitHub</a>.
                </p>             

    .. grid-item-card::
            :shadow: none
            :class-card: sd-border-0

            .. figure:: ./_static/fair.png
               :scale: 4%
               :class: no-scaled-link

            .. raw:: html

                <p style="text-align: center;">
                <strong> Extensible </strong> <br/> 
                Powerful plugin mechanism and community marketplace (<a href="./fair/index.html">Pyxu FAIR</a>) for discovering and sharing custom operators.
                </p>             
               
Ecosystem
=========

Pyxu is part of, looks and feels like, and is highly interoperable with the wider scientific Python ecosystem. It is indeed built on a minimal set of foundational and robust scientific computing 
librairies from the PyData stack. Pyxu notably supports multiple array backends --`NumPy <https://numpy.org/>`_, `Dask <https://www.dask.org/>`_, `Sparse <https://sparse.pydata.org/en/stable/>`_ and 
optionally `CuPy <https://cupy.dev/>`_--, allowing users to choose array backends that work best for their application/computation.
Aside from `SciPy <https://scipy.org/>`_ and  `Numba <https://numba.pydata.org/>`_ -- which we use for scientific computing and `JIT-compilation <https://numba.readthedocs.io/en/stable/user/5minguide.html#how-does-numba-work>`_ respectively--
these are Pyxu's **only** dependencies, making the software very easy to ship, install, deploy in production, and sustain in the long term.

Pyxu is also interoperable with (but does not depend on) the major deep learning frameworks `JAX <https://jax.readthedocs.io/en/latest/>`_ and `PyTorch <https://pytorch.org/>`_,
allowing users to benefit from the latest incursions of deep learning in the field of computational imaging (e.g., PnP methods, unrolled neural networks, deep generative priors). 
Our wrappers can moreover leverage the autograd engine to auto-infer gradients or adjoints operations. 


.. grid:: 2 2 4 4
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/numpy_logo.svg
               :width: 75%
               :align: center
               :alt: NumPy's logo
               :target: https://numpy.org/


            .. raw:: html 
              
                <p style="text-align: center;">
                NumPy is the fundamental package for array computing with Python.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/dask_horizontal.svg
               :width: 70%
               :align: center
               :alt: Dask's logo
               :target: https://www.dask.org/


            .. raw:: html 
              
                <p style="text-align: center;">
                NumPy-compatible distributed arrays and advanced parallelism for both in and out-of-core computing, enabling performance at scale.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/cupy.png
               :width: 75%
               :align: center
               :alt: CuPy's logo
               :target: https://cupy.dev/


            .. raw:: html 
              
                <p style="text-align: center;">
                NumPy-compatible array library for GPU-accelerated computing with Python.
                </p>
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/sparse-logo.png
               :width: 65%
               :align: center
               :alt: Sparse's logo
               :target: https://sparse.pydata.org/en/stable/


            .. raw:: html 
              
                <p style="text-align: center;">
                Sparse multi-dimensional arrays for the PyData ecosystem.
                </p>
    
    .. grid-item-card:: 
        :shadow: none
        :class-card: sd-border-0

        .. image:: _static/scipy.png
            :width: 70%
            :align: center
            :alt: SciPy's logo
            :target: https://scipy.org/


        .. raw:: html 
            
            <p style="text-align: center;">
            Fundamental algorithms for scientific computing in Python.
            </p>

    .. grid-item-card:: 
        :shadow: none
        :class-card: sd-border-0

        .. image:: _static/numba-blue-horizontal-rgb.svg
            :width: 85%
            :align: center
            :alt: Numba's logo
            :target: https://numba.pydata.org/


        .. raw:: html 
            
            <p style="text-align: center;">
            NumPy-aware dynamic Python compiler using <a href="https://llvm.org/">LLVM</a>.
            </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/Pytorch_logo.png
               :width: 75%
               :align: center
               :alt: PyTorch's logo
               :target: https://pytorch.org/


            .. raw:: html 
              
                <p style="text-align: center;">
                Tensors and dynamic neural networks in Python with strong GPU acceleration.
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/jax_logo_250px.png
               :width: 40%
               :align: center
               :alt: JAX's logo
               :target: https://jax.readthedocs.io/en/latest/


            .. raw:: html 
              
                <p style="text-align: center;">
                Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more.
                </p>

.. figure:: ./_static/stack.png
    :width: 70%
    :class: no-scaled-link


Governance and Team
===================
Pyxu is an **open-source project** developed and maintained primarily by members of the `EPFL Center for Imaging <https://imaging.epfl.ch/>`_, 
but the repository itself is public and we welcome external contributions. We are committed to keeping the project public and owned by the community through 
a meritocratic and consensus-based governance. Anyone with an interest in the project can join the community, contribute to the project design, 
and participate in the decision-making process.

.. grid:: 1 2 3 3
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/EPFL_Logo_Digital_BLACK_PROD.png
               :width: 60%
               :align: center
               :target: https://www.epfl.ch/en/

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/imaging.png
               :width: 60%
               :align: center
               :target: https://imaging.epfl.ch/
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/air.svg
               :width: 40%
               :align: center

Steering Council
----------------

The role of Pyxu's Steering Council is to ensure the long-term sustainability of the project, both technically and as a community. 
Pyxu's Steering Council meets regularly (every two weeks or so) and currently consists of the following members:

.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/simeoni.png
               :width: 40%
               :align: center
               :target: https://github.com/matthieumeo


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Matthieu Simeoni </strong> <br/> 
                Pyxu's creator/architect, project manager & team lead
                </p>

    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/kashani.png
               :width: 40%
               :align: center
               :target: https://github.com/SepandKashani


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Sepand Kashani </strong> <br/> 
                Technical lead, software architect & tests
                </p>
    
    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/rue_queralt.png
               :width: 40%
               :align: center
               :target: https://github.com/joanrue


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Joan Rué-Queralt </strong> <br/> 
                Technical lead, solution architect & plugins
                </p>
        
Contributors
------------
In addition to the steering council, the following people are currently (or have been in the past) 
core contributors to Pyxu's development and/or maintenance (alphabetical order, full list available on GitHub):

.. grid:: 2 2 3 3
    :gutter: 3

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/debarre.png
               :width: 40%
               :align: center
               :target: https://github.com/ThomasDeb


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Thomas Debarre </strong> <br>
                Core contributor (Emeritus)
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/flowers.png
               :width: 40%
               :align: center
               :target: https://github.com/alec-flowers


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Alec Flowers </strong> <br/> 
                Contributor (Emeritus)
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/hamm.png
               :width: 40%
               :align: center
               :target: https://github.com/dhamm97


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Daniele Hamm </strong> <br/> 
                Contributor
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/jarret.png
               :width: 40%
               :align: center
               :target: https://github.com/AdriaJ


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Adrian Jarret </strong><br/> 
                Contributor
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/salim.png
               :width: 40%
               :align: center
               :target: https://github.com/Dicedead


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Salim Najib </strong><br/> 
                Contributor
                </p>

    .. grid-item-card:: 
            :shadow: none
            :class-card: sd-border-0

            .. image:: _static/okumus.png
               :width: 40%
               :align: center
               :target: https://github.com/okumuskaan


            .. raw:: html 
              
                <p style="text-align: center;">
                <strong> Kaan Okumus </strong> <br/> 
                Contributor (Emeritus)
                </p>

Partners & Sponsors
===================

.. raw:: html 
   

    <div class="sponsors-inner" style="color: #fff;
    text-align: left;
    overflow: hidden;
    height: 92px;
    position: relative;
    transform: translate3d(0, 0, 0);
    z-index: 1;">
    <div class="sponsors-list" style="width: 4000px;
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    align-items: center;
    justify-content: space-between;
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    transform: translate3d(0, 0, 0);
    animation: platformsSlideshow 60s linear 0s infinite;
    padding: 0 40px;
    box-sizing: border-box;">
    
    <div class="sponsors-list-item">

    <img width="100" alt="EPFL Slider" src="_static/EPFL_Logo_Digital_BLACK_PROD.png">

    </div>
    <div class="sponsors-list-item">

    <img width="150" alt="Imaging Slider" src="_static/imaging.png">

    </div>
    <div class="sponsors-list-item">

    <img width="100" alt="LCAV Slider" src="_static/LCAV_LOGO.png">

    </div>
    <div class="sponsors-list-item">

    <img width="130" alt="Meta Slider" src="_static/Meta-Logo.png">

    </div>
    <div class="sponsors-list-item">

    <img width="130" alt="SKACH Slider" style="padding: 0px 0px 30px 0px;" src="_static/skach.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="ETH Slider" src="_static/ethr_en_rgb_black.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="SNF Slider" src="_static/SNF_logo_standard_office_color_pos_e.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="SPC Slider" src="_static/spc.png">
    </div>

    <div class="sponsors-list-item">

    <img width="100" alt="EPFL Slider" src="_static/EPFL_Logo_Digital_BLACK_PROD.png">

    </div>
    <div class="sponsors-list-item">

    <img width="150" alt="Imaging Slider" src="_static/imaging.png">

    </div>
    <div class="sponsors-list-item">

    <img width="100" alt="LCAV Slider" src="_static/LCAV_LOGO.png">

    </div>
    <div class="sponsors-list-item">

    <img width="130" alt="Meta Slider" src="_static/Meta-Logo.png">

    </div>
    <div class="sponsors-list-item">

    <img width="130" alt="SKACH Slider" style="padding: 0px 0px 30px 0px;" src="_static/skach.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="ETH Slider" src="_static/ethr_en_rgb_black.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="SNF Slider" src="_static/SNF_logo_standard_office_color_pos_e.png">

    </div>
    <div class="sponsors-list-item">

    <img width="170" alt="SPC Slider" src="_static/spc.png">
    </div>
    </div>
    </div>

.. toctree::
   :maxdepth: 1
   :hidden: 

   intro/index
   guide/index
   examples/index
   api/index
   fair/index
