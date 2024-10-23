"""
Setup file for fullerenetool.
Use setup.cfg to configure your project.

This file was generated with PyScaffold 4.5.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""

from setuptools import find_packages, setup

try:
    import numpy
except ImportError:
    print("\n\nNumpy is required to build the project, please install it with:\n")
    print("   pip install numpy\n")
    print("or:\n")
    print("   conda install numpy\n")
    print("Ingnoring if you install the project in editable mode.")

try:
    from Cython.Build import cythonize
    from setuptools import Extension

    extensions = [
        Extension(
            "fullerenetool.algorithm.dual_graph",
            ["src/fullerenetool/algorithm/dual_graph.pyx"],
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
    ]
    setup(
        use_scm_version={"version_scheme": "no-guess-dev"},
        ext_modules=cythonize(extensions, language_level=3),
        include_dirs=[numpy.get_include()],
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
except ImportError:
    print(
        "Not using Cython, using pure python instead, make sure run \n",
        "    python setup.py build_ext --inplace\n",
        "to build the cython extension.",
    )
    setup(
        use_scm_version={"version_scheme": "no-guess-dev"},
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
except Exception as e:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U setuptools setuptools_scm wheel\n\n"
    )
    raise e
