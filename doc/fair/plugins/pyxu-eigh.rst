.. _plugins-page:

.. |br| raw:: html

   </br>

.. raw:: html

    <!-- CSS overrides on the pyxu-fair only -->
    <style>

    .summaryinfo {
        color: #000;
        font-size: 80%;
        margin-bottom: 12px;
        margin-top: 12px;
    }

    .entrypointraw {
        color: #777;
    }
    .badge {
        white-space: nowrap;
        display: inline-block;
        vertical-align: middle;
        /*vertical-align: baseline;*/
        font-family: "DejaVu Sans", Verdana, Geneva, sans-serif;
        /*font-size: 90%;*/
    }
    .currentstate {
        color: #666;
        font-size: 90%;
        margin-bottom: 12px;
    }
    span.badge-left {
        border-radius: .25rem;
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        color: #212529;
        background-color: #A2CBFF;
        /* color: #ffffff; */
        text-shadow: 1px 1px 1px rgba(0,0,0,0.3);

        padding: .25em .4em;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        float: left;
        display: block;
    }

    span.badge-right {
        border-radius: .25rem;
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;

        color: #fff;
        background-color: #343a40;

        padding: .25em .4em;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        float: left;
        display: block;
    }

    .badge-right.light-blue, .badge-left.light-blue {
        background-color: #A2CBFF;
        color: #212529;
    }

    .badge-right.light-red, .badge-left.light-red {
        background-color: rgb(255, 162, 162);
        color: rgb(43, 14, 14);
    }

    .badge-right.red, .badge-left.red {
        background-color: #e41a1c;
        color: #fff;
    }

    .badge-right.blue, .badge-left.blue {
        background-color: #377eb8;
        color: #fff;
    }

    .badge-right.green, .badge-left.green {
        background-color: #4daf4a;
        color: #fff;
    }

    .badge-right.purple, .badge-left.purple {
        background-color: #984ea3;
        color: #fff;
    }

    .badge-right.orange, .badge-left.orange {
        background-color: #ff7f00;
        color: #fff;
    }

    .badge-right.brown, .badge-left.brown {
        background-color: #a65628;
        color: #fff;
    }

    .badge-right.dark-gray, .badge-left.dark-gray {
        color: #fff;
        background-color: #343a40;
    }


    .badge a {
        text-decoration: none;
        padding: 0;
        border: 0;
        color: inherit;
    }

    .badge a:visited, .badge a:active {
        color: inherit;
      }

    .badge a:focus, .badge a:hover {
        color: rgba(255,255,255,0.5);
        mix-blend-mode: difference;
        text-decoration: none;
        /* background-color: rgb(192, 219, 255); */
    }


    .svg-badge {
        vertical-align: middle;
    }

    .tooltiptext {
        visibility: hidden;
        /* width: 120px; */
        background-color: rgb(255, 247, 175);
        color: #000;
        text-align: center;
        border-radius: 6px;
        padding: 5px;

        /* Position the tooltip */
        position: absolute;
        z-index: 1;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
    }
    </style>

*********
pyxu-eigh
*********


Description
===========

.. raw:: html
    Pyxu plugin to compute the eigenvectors and eigenvalues of a real symmetric matrix



General information
===================

.. raw:: html

    
    <p>
        <strong>Short description</strong>: Pyxu plugin to compute the eigenvectors and eigenvalues of a real symmetric matrix
    </p>
    

    <p class="currentstate">
        
        <img class="svg-badge" src="../../_static/plugins/status/status-planning-d9644d.svg" title="Planning: Not yet ready to use. Developers welcome!"></p>
    

    <p>
        <strong>Source code</strong>: <a href="https://github.com/AIR-Hub/pyxu-eigh" target="_blank">Go to the source code repository</a>
    </p>

    
    <p>
        <strong>Documentation</strong>: No documentation provided by the package author
    <p>
    


Detailed information
====================

.. raw:: html
    
    <p>
        <strong>Author(s)</strong>: Joan Rue Queralt
    </p>
    
    
    <p>
        <strong>Contact</strong>: <a href="mailto:joan.rue.q@gmail.com">joan.rue.q@gmail.com</a>
    </p>
    
    
    <p>
        <strong>Most recent version</strong>: 0.1.0
    </p>
    <p>
        <strong>Compatibility</strong>:
        <img class="svg-badge" title="Compatible with Pyxu 1" src="https://img.shields.io/badge/Pyxu-1-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII=">
    </p>
    


Components contributed
======================

.. raw:: html
    <p class="summaryinfo">
        
        <span class="badge">
            <span class="badge-left green">Math</span>
            <span class="badge-right">1</span>
        </span>
        
    </p>
    
    
    
      Math <span class="entrypointraw">(pyxu.math)</span>
    
    <ul>
    
    <li><code>eigh</code>
    
    </ul>
    
    