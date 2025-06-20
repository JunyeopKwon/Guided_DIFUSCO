# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extension
extensions = [
    Extension(
        "alpha_2opt_cython", # Name of the module to import
        ["alpha_2opt_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

# Compile the extension
setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
)
