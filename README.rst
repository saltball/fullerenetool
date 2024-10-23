.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/fullerenetool.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/fullerenetool
    .. image:: https://readthedocs.org/projects/fullerenetool/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://fullerenetool.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/fullerenetool/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/fullerenetool
    .. image:: https://img.shields.io/pypi/v/fullerenetool.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/fullerenetool/
    .. image:: https://img.shields.io/conda/vn/conda-forge/fullerenetool.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/fullerenetool
    .. image:: https://pepy.tech/badge/fullerenetool/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/fullerenetool
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/fullerenetool

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=============
fullerenetool
=============


    Python package for fullerene and derivatives.


A longer description of your project goes here...


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd fullerenetool
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Development Note
================
This repository uses `setup.cfg` for configuration, and also `cython` for faster computation.
One needs to install `cython` before development, a simple command to install is::

    pip install cython

and one should use `setup.py` to build and install the package for development, a simple command to build is::

    python setup.py install build_ext --inplace
    pip install -e .

to install the package for development. You may use::

    pip uninstall fullerenetool -y && python setup.py install build_ext

to install the package to your environment directory.

You may need to install boost, numpy before installing the package, a simple command to install is::

    conda install numpy boost

We recommend to use dockerfile to build the package, a simple command to build is::

    docker build -t fullerenetool .

And the package can be run in the container.

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
