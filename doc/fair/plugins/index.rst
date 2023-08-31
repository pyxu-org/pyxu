:html_theme.sidebar_secondary.remove:
:sd_hide_title: true

.. |br| raw:: html

   </br>

.. raw:: html

    <!-- CSS overrides on the pyxu-fair only -->
    <style>
    .globalsummary-box {
        width:95%;
        padding:10px;
        margin: 20px;
        padding-left: 40px;
        padding-right: 40px;
        background-color: rgba(255,255,255,0.19);
        border:1px solid rgba(50,50,50,0.4);
        border-radius:4px;
        -webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.05);
        box-shadow:inset 0 1px 1px rgba(0,0,0,.05);
        line-height: 1.5;
    }
    .summaryinfo {
        color: #000;
        font-size: 80%;
        margin-bottom: 12px;
        margin-top: 12px;
    }
    .submenu-entry {
        width:90%;
        min-height:140px;
        padding:10px;
        margin: 20px;
        padding-left: 40px;
        padding-right: 40px;
        background-color:rgba(245,245,245,.19);
        border:1px solid rgba(227,227,227,.31);
        border-radius:4px;
        -webkit-box-shadow:inset 0 1px 1px rgba(0,0,0,.05);
        box-shadow:inset 0 1px 1px rgba(0,0,0,.05);
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

    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }

    .tooltip .tooltiptext {
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

**************
Plugin Catalog
**************


.. raw:: html

    <h2 style="font-size: 60px; font-weight: bold; display: inline"><span>Plugin Catalogue</span></h2>


**Total Registered Plugin Packages: 17**

.. raw:: html

   <div class='globalsummary-box'>
        <div style="display: table;">
            
            <span class="badge" style="display: table-row; line-height: 2;">
                <span style="display: table-cell; float: none; text-align: right;"><span class="badge-left blue" style="float: none; display: inline; text-align: right; border: none">Operator</span></span>
                <span style="display: table-cell; float: none; text-align: left;"><span class="badge-right" style="float: none; display: inline; text-align: left; border: none">17 plugins in 12 packages </span></span>
            </span>
        
            <span class="badge" style="display: table-row; line-height: 2;">
                <span style="display: table-cell; float: none; text-align: right;"><span class="badge-left brown" style="float: none; display: inline; text-align: right; border: none">Solver</span></span>
                <span style="display: table-cell; float: none; text-align: left;"><span class="badge-right" style="float: none; display: inline; text-align: left; border: none">14 plugins in 10 packages </span></span>
            </span>
        
            <span class="badge" style="display: table-row; line-height: 2;">
                <span style="display: table-cell; float: none; text-align: right;"><span class="badge-left purple" style="float: none; display: inline; text-align: right; border: none">Stop</span></span>
                <span style="display: table-cell; float: none; text-align: left;"><span class="badge-right" style="float: none; display: inline; text-align: left; border: none">18 plugins in 11 packages </span></span>
            </span>
        
            <span class="badge" style="display: table-row; line-height: 2;">
                <span style="display: table-cell; float: none; text-align: right;"><span class="badge-left green" style="float: none; display: inline; text-align: right; border: none">Math</span></span>
                <span style="display: table-cell; float: none; text-align: left;"><span class="badge-right" style="float: none; display: inline; text-align: left; border: none">16 plugins in 10 packages </span></span>
            </span>
        
            <span class="badge" style="display: table-row; line-height: 2;">
                <span style="display: table-cell; float: none; text-align: right;"><span class="badge-left orange" style="float: none; display: inline; text-align: right; border: none">Contrib</span></span>
                <span style="display: table-cell; float: none; text-align: left;"><span class="badge-right" style="float: none; display: inline; text-align: left; border: none">20 plugins in 12 packages </span></span>
            </span>
        
        </div>
    </div>

**Plugin in alphabetical order:**


.. grid:: 2 2 3 3
    :gutter: 3

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="CSEEG.html">CSEEG</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-planning-d9644d.svg" title="Planning: Not yet ready to use. Developers welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/lbrooks/CSEEG" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://lbrooks.github.io/CSEEG/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 69.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="DSP-Notebooks.html">DSP-Notebooks</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Production/Stable: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/sdunn/DSP-Notebooks" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://sdunn.github.io/DSP-Notebooks/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 83.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="EnvironTracker.html">EnvironTracker</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-alpha-d6af23.svg" title="Alpha: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/cedwards/EnvironTracker" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://cedwards.github.io/EnvironTracker/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 79.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="HoughDetector.html">HoughDetector</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-alpha-d6af23.svg" title="Alpha: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/abrick/HoughDetector" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://abrick.github.io/HoughDetector/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 56.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">4</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="HVOX.html">HVOX</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-planning-d9644d.svg" title="Pre-alpha: Not yet ready to use. Developers welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/snassar/HVOX" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://snassar.github.io/HVOX/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 80.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="OrientationPy.html">OrientationPy</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Mature: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/mmonteith/OrientationPy" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://mmonteith.github.io/OrientationPy/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 77.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="Palentologist.html">Palentologist</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Production/Stable: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/knaiman/Palentologist" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://knaiman.github.io/Palentologist/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 73.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="PhaseRet.html">PhaseRet</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-inactive-bbbbbb.svg" title="Inactive: No longer maintained.">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/esequeira/PhaseRet" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://esequeira.github.io/PhaseRet/html/index" target="_blank">Documentation</a>
                </li>
            

            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">3</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="PycGSP.html">PycGSP</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Production/Stable: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/doliphant/PycGSP" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://doliphant.github.io/PycGSP/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 77.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="pycNUFFT.html">pycNUFFT</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-alpha-d6af23.svg" title="Alpha: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/csincell/pycNUFFT" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://csincell.github.io/pycNUFFT/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 78.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="PycSphere.html">PycSphere</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-planning-d9644d.svg" title="Pre-alpha: Not yet ready to use. Developers welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/jhawk/PycSphere" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://jhawk.github.io/PycSphere/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 79.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">2</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="pycWavelet.html">pycWavelet</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-alpha-d6af23.svg" title="Alpha: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/cpark/pycWavelet" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://cpark.github.io/pycWavelet/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 68.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="PYFW.html">PYFW</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Mature: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/rshelton/PYFW" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://rshelton.github.io/PYFW/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 95.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">2</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="TokemakRec.html">TokemakRec</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-beta-d6af23.svg" title="Beta: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/msmith/TokemakRec" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://msmith.github.io/TokemakRec/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 91.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">3</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">2</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="TVDenoiser.html">TVDenoiser</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-inactive-bbbbbb.svg" title="Inactive: No longer maintained.">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/dbrown/TVDenoiser" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://dbrown.github.io/TVDenoiser/html/index" target="_blank">Documentation</a>
                </li>
            

            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">3</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="UncertaintyQuant.html">UncertaintyQuant</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-stable-4cc61e.svg" title="Production/Stable: Ready for production calculations. Bug reports welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/kdavis/UncertaintyQuant" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://kdavis.github.io/UncertaintyQuant/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 88.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    
    .. grid-item-card::

        .. raw:: html

            <h2 style="font-size: 30px;"><a href="WaveProp.html">WaveProp</a></h2>

            <p class="currentstate">
                
                <img class="svg-badge" src="../../_static/plugins/status/status-beta-d6af23.svg" title="Beta: Adds new functionality, not yet ready for production. Testing welcome!">&nbsp;<img class="svg-badge" title="Compatible with Pyxu plugin.pyxu_version }}" src="https://img.shields.io/badge/Pyxu-2.0.0-007ec6.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAAACXBIWXMAAAHpAAAB6QHzJ3XIAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAzpJREFUSImtlU1oXFUUx3/nzVemWq1pbQXXIpSaEiPUtiCSTdK4qAhVLFh0VXChKAbSVbCiUFFxoSAB3ZmKH2C0aivigBpR/AJxERS1hGATg9F2JhN999x7XMx7Ly/TaQmZHvhzLoc793e+eCNmxuVsYXT/9vLmwhNRNXqUqpSlGiFVSRQhFWldNAY2DZ7+vtMbUafgWx9/dt/cYt3mFutWFFkwz2jwVsYD3mj55JzmKHzXrA3bSm1oz2UhJ0/Vnvl5dtH29ve/AVAZHyb/sHnDMr96zpshX63UDsx1hEyeqtmeW3cf21TtYX7+XCs5b7nM233u3GaG3ZgHRQlgvlgoUC6VODh8Jw/efw+//n42ybStAgXUEiXn0BnUrA0NABST2I59t/UDcGz8ae69e4Tfzs6yM30oFkLdQwEoCVISpJIoXYCeVjzaWkCqkg7qW0Ciyfdr7+YzuH3vfmYXzlMulSASzEFoBsxoDTlVyPlcJeGCJyz5bFaNz0eujxAOXlws7Np58+rEDEhX3SwREJJWWW7L0p80WzHR8EsUQsA5pd5YXnNpZmaGl2463N7ovOasINNW4SMry5SU+JrSWpT9ZwhcK6+9fdpCCIQQODB4R3bhi+lpAJaW/uHv8xfYtvU6jh45JKzTlj+86wZEz0lFKKoqKSSOHeVyKemKIUBv7xZ6e7fgnM6vFwBw1cgH84CsfDq0L1JVVBV1yntnPskuRVGEmWHJDFTjuUs/eWmrDp75MvJeH1CnuAQ2+c4U3/zwI7v7+hARSEA9PdWBjUAAxMx4buJkMhdP8K3W+RCIm3VEJNPxsUfWPZO8FQFU3Z8h2PbgfQZIt04EQBK/MYsAxh4+skNdq10uNyPnHLFzOOdwsdswJP2s4FR/CsHvCiFkLVNVDOiiiNVKAJ4aPXpLWkG6cS6rSHHOxruGAGisY2sA6nHaatsrL4wfvyKQ5598/EQKWFluZBVpsL82CoDcTFJzzrsQfCmOYywEzIzXJ57d1g3kov94r3pYVfHqUe/p8cVrugFAh0qi5h9TjXA13nsolvtfnThRv+KQRqG3z7t/oeo2v/nyi41uAQDZRzDVoYcea7bHutX/c3lfgmmbsZoAAAAASUVORK5CYII="></p>

            <ul class="plugin-info">

            
                <li>
                    <a href="https://github.com/knix/WaveProp" target="_blank">Home page</a>
                </li>
            

            
                <li>
                    <a href="https://knix.github.io/WaveProp/html/index" target="_blank">Documentation</a>
                </li>
            

            
                <li>
                    <a> Score: 76.0 % </a>
                </li>
            
            </ul>

            <p class="summaryinfo">
                
                <span class="badge">
                    <span class="badge-left blue">Operator</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left brown">Solver</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left purple">Stop</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left green">Math</span>
                    <span class="badge-right">1</span>
                </span>
                
                <span class="badge">
                    <span class="badge-left orange">Contrib</span>
                    <span class="badge-right">1</span>
                </span>
                
            </p>

    



.. toctree::
   :maxdepth: 1
   :hidden:

   CSEEG
   DSP-Notebooks
   EnvironTracker
   HoughDetector
   HVOX
   OrientationPy
   Palentologist
   PhaseRet
   PycGSP
   pycNUFFT
   PycSphere
   pycWavelet
   PYFW
   TokemakRec
   TVDenoiser
   UncertaintyQuant
   WaveProp
   