:sd_hide_title: true

.. |br| raw:: html

   </br>

.. raw:: html

    <!-- CSS overrides on the Pyxu FAIR only -->
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



Extending Pyxu
##############


.. raw:: html

    <h2 style="font-size: 60px; font-weight: bold; display: inline"><span>Pyxu FAIR</span></h2>
    <h3 style="margin-top: 0; font-weight: bold; text-align: left; ">A marketplace for Pyxu-based plugins</h3>
    <p>While Pyxu offers flexibility and portability across various imaging domains, its general-purpose design might
    not cater to the specific needs of certain imaging communities. The <strong>Pyxu FAIR</strong> addresses this by
    offering a platform that allows for the development, sharing, and integration of specialized plugins to enhance the
    framework.
    </p>

In simple terms, the Pyxu FAIR is designed to supercharge your Pyxu experience:

Catalogue Website
~~~~~~~~~~~~~~~~~

A `one-stop shop <./plugins/index.html>`_ to discover exciting Pyxu plugins right from your browser. Imagine a shopping mall, but just for Pyxu plugins! 🛍️
Auto-discovery of plugins from the Python Package Index (PyPI). No more manual hunting! 🕵️
Relevant metadata display and a fuzzy-searchable index. Dive deeper into plugins, explore their features, and find the best fit for your needs. 🔎
Excitingly, you can even rank plugins based on their `Pyxu score <./score.html>`_. The higher the score, the better they gel with Pyxu's quality standards. 🏆

Meta-programming Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~

Kickstart your plugin development with the `Pyxu cookie-cutter <https://github.com/matthieumeo/cookiecutter-pycsou-plugin/>`_ 🖥. Want to share your Pyxu-based tools without the DevOps hassle? We've got you covered! 🛠️
This plugin-template generator ensures your plugins are easy to install, discoverable by Pyxu FAIR, and sync well with Pyxu's quality controls. 🌟

Interoperability Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~

Not just a directory, Pyxu FAIR ensures your plugins gel seamlessly with Pyxu's core framework. 🤝
Via Python's `entry points <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_, Pyxu FAIR sets guidelines on how plugins communicate with the main framework.
Excitingly, with our new and improved loading technique, interacting with user-contributed plugins feels like chatting with old friends. 🤖



.. toctree::
   :maxdepth: 2
   :hidden:

   howto
   contribute
   score
   plugins/index
