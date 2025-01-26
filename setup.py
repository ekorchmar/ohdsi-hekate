"""This file is intended not for setuptools (it will read the pyproject.toml
file), but for Cython build configuration.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy  # PERF: Do we need numpy? May be removed later.


extensions = [
    Extension(
        "*", ["*.py[x]"],
        include_dirs=[numpy.get_include()]
    ),
]

Options.embed = "main"
Options.fast_fail = True
Options.annotate = True

setup(
    name="ohdsi-hekate",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "profile": False
        }
    ),
)
