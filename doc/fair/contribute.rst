Contributing to the Pyxu-FAIR
=============================

**Want to contribute** your own Pyxu-based plugin and making it available in the `Pyxu FAIR <../plugins/index.html>`_,
we recommend using the `Pyxu cookiecutter <https://github.com/matthieumeo/cookiecutter-pyxu-plugin>`_.

This tutorial will guide you through the process of creating a new plugin using the cookiecutter.

Create your plugin package
--------------------------

Install `Cookiecutter <https://pypi.org/project/cookiecutter/>`_ if not installed:

.. code-block:: bash 

    python -m pip install cookiecutter

Generate a new Pyxu plugin project (a new folder will be created in your current working directory):

.. code-block:: bash 


    cookiecutter https://github.com/matthieumeo/cookiecutter-pyxu-plugin


The Cookiecutter prompts you for information regarding your plugin. Defaults are shown in brackets.

.. code-block:: bash

    full_name [Pyxu Developer]: Isaac Newton
    email [yourname@example.com]: inewton@trinity.uk
    github_username_or_organization [githubuser]: sirisaac

Note that for packages whose primary purpose is to be a Pyxu plugin, we recommend using the 'pyxu-' prefix in the 
package name. If your package provides functionality outside of Pyxu, you may choose to leave Pyxu out of the name.
    
.. code-block:: bash 

    plugin_name [pyxu-foobar]: pyxu-gradient-descent

    Select github_repository_url:
    1 - https://github.com/sirisaac/pyxu-gradient-descent
    2 - provide later
    Choose from 1, 2 [1]:

    module_name [pyxu_gradient_descent]:
    display_name [Pyxu FooBar Collection]: Gradient Descent
    short_description [A simple plugin to use the FooBar collection within Pyxu]: A simple gradient descent solver for Pyxu

You can choose from a variety of plugin template examples. These provide the foundational structure for a Pyxu plugin,
aiding in the development of your own plugin.

.. code-block:: bash

    include_math_plugin [y]: n
    include_operator_plugin [n]:
    include_solver_plugin [n]:y
    include_stop_plugin [n]: n
    include_contrib_plugin [n]: n


Next, you'll be prompted to decide between using git tags for versioning or managing package version numbers manually.
Using git tags offers a more straightforward approach and reduces potential errors.

.. code-block:: bash

    use_git_tags_for_versioning [n]:

Next, you'll be prompted to determine if you wish to install `pre-commit <https://pre-commit.com/>`_. This tool automates
tasks before each commit, ensuring code quality and consistency, reducing the likelihood of errors and oversights in
your codebase.

.. code-block:: bash

    install_precommit [n]:

Finally, you'll be prompted to select a license for your plugin. The default is the BSD-3 license.

.. code-block:: bash

    Select license:
    1 - BSD-3
    2 - MIT
    3 - Mozilla Public License 2.0
    4 - Apache Software License 2.0
    5 - GNU LGPL v3.0
    6 - GNU GPL v3.0
    Choose from 1, 2, 3, 4, 5, 6 [1]:

You just created the necessary structure for a funcitonal Pyxu plugin, completed with tests and ready for automatic
deployment!

For more detailed information on each prompt see the `prompts reference <https://github.com/matthieumeo/cookiecutter-pyxu-plugin/PROMPTS.md>`_.

.. code-block:: bash

    pyxu-gradient-descent
    ├── .git
    ├── .github
    │         └── workflows
    │             └── test_and_deploy.yml
    ├── .gitignore
    ├── __init__.py
    ├── LICENSE
    ├── MANIFEST.in
    ├── .pre-commit-config.yaml
    ├── .pyxu-gradient-descent
    │         ├── config.yml
    │         └── DESCRIPTION.md
    ├── pyproject.toml
    ├── README.md
    ├── setup.cfg
    ├── src
    │         ├── __init__.py
    │         ├── pyxu_gradient_descent
    │         │       ├── __init__.py
    │         │       └── opt
    │         │           ├── __init__.py
    │         │           └── solver
    │         │               └── __init__.py
    │         └── pyxu_gradient_descent_tests
    │             ├── __init__.py
    │             └── test_opt
    │                 ├── __init__.py
    │                 └── test_solver.py
    └── tox.ini

Initialize a git repository in your package
-------------------------------------------

This is important for version management.

.. code-block:: bash

    cd pyxu-gradient-descent
    git init
    git add .
    git commit -m 'initial commit'


Upload it to GitHub
-------------------

* Create a [new github repository] with the name ``github_repository_url`` you indicated.

* Add your newly created GitHub repo as a remote and push:

.. code-block:: bash

   git remote add origin https://github.com/sirisaac/pyxu-gradient-descent.git
   git push -u origin main


Setup a local environment
-------------------------

It is recommended to set up a local Python environment to develop and test your plugin. With `Conda <https://docs.conda.io/>`_, you can use:

.. code-block:: bash

   my_env=<CONDA ENVIRONMENT NAME>
   conda create --name "${my_env}" python=3.11
   conda activate "${my_env}"
   python -m pip install -e .

You will probably want to install your new package into this environment. ``Pyxu`` is already set as a dependency,
simply add the other required dependencies in the ``setup.cfg`` file and run the following commands.


.. code-block:: bash

   cd <your-repo-name>
   python -m pip install -e .

The ``-e .`` arguments install the package in editable mode, meaning that any changes you make to the source code, will
be reflected in the installed package.

Develop new features
--------------------

The cookiecutter offers a predefined hierarchy of classes and functions to aid novice Pyxu developers in creating
novel features. At this point, the developer can create new functionalities following the `Pyxu developer notes <https://github.com/matthieumeo/pycsou/blob/v2-dev/doc/dev_notes.rst>`_ and
structure predefined by the cookiecutter.

Continuous Integration
----------------------

This Pyxu-plugin generator repository provides you with already-parametrized continuous integration tools.

Pre-commit
~~~~~~~~~~

This template includes a default yaml configuration for `pre-commit <https://pre-commit.com/>`_.

Among other things, it includes checks for best practices in Pyxu plugins.

You may edit the config at ``.pre-commit-config.yaml``

To use it run:

.. code-block:: bash

    pip install pre-commit
    pre-commit install


You can also have these checks run automatically for you when you push to GitHub
by installing `pre-commit ci <https://pre-commit.ci/>`_ on your repository.


Running tests locally
~~~~~~~~~~~~~~~~~~~~~

You can run your tests locally with `pytest <https://docs.pytest.org/en/7.1.x/>`_.
You'll need to make sure that your package is installed in your environment,
along with testing requirements (specified in the setup.cfg `extras_require` section):

.. code-block:: bash

   pip install -e ".[testing]"
   pytest

Monitor testing and coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository is already setup to run your tests automatically each time you push an
update (configuration is in `.github/workflows/test_and_deploy.yml`). You can
monitor them in the "Actions" tab of your GitHub repository. If you're
following along, go have a look... they should be running right now!

When the tests are done, test coverage will be viewable at
`codecov.io <https://codecov.io/>`_) (assuming your repository is public):
`https://codecov.io/gh/<your-github-username>/<your-package-name>`

Set up automatic deployments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your new package is also nearly ready to automatically deploy to `PyPI <https://pypi.org/>`_.
(whenever you create a tagged release), so that your users can simply ``pip install`` your package. To do so, you just
need to create an `API token to authenticate with PyPi <https://pypi.org/help/#apitoken>`_, and then add it to your GitHub
repository:

1. If you don't already have one, `create an account <https://pypi.org/account/register/>`_ at PyPI.
2. Verify your email address with PyPI, (if you haven't already)
3. Generate an `API token <https://pypi.org/help/#apitoken>`_ at PyPI: In your
   `account settings <https://pypi.org/manage/account/>`_ go to the API tokens
   section and select "Add API token". Make sure to copy it somewhere safe!
4. `Create a new encrypted
   secret <https://help.github.com/en/actions/configuring-and-managing-workflows/creating-and-storing-encrypted-secrets#creating-encrypted-secrets>`_
   in your GitHub repository with the name "TWINE_API_KEY", and paste in your
   API token.

You are now setup for automatic deployment!

Automatic deployment and version management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time you want to deploy a new version, you just need to create a tagged
commit, and push it to your main branch on GitHub. Your package is set up to
use `setuptools_scm <https://github.com/pypa/setuptools_scm>`_ for version
management, meaning you don't need to hard-code your version anywhere in your
package. It will be inferred from the tag each time you release. The deployment
is also handled with the [github actions] using the same workflow file `.github/workflows/test_and_deploy.yml`.

The tag will be used as the version string for your package make it meaningful: https://semver.org/

.. code-block:: bash

    git tag -a v0.1.0 -m "v0.1.0"

Make sure to use follow-tags so that the tag also gets pushed to github

.. code-block:: bash

    git push --follow-tags

Monitor the "actions" tab on your GitHub repo for progress... and when the
"deploy" step is finished, your new version should be visible on PyPI:

`https://pypi.org/project/<your-package-name>/`

and available for pip install with:

.. code-block:: bash

    pip install pyxu-gradient-descent

Create your documentation
-------------------------

Documentation generation is not included in this template.
We recommend following the getting started guides for https://www.sphinx-doc.org/.