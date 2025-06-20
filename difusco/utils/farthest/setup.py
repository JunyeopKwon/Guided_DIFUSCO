# setup_farthest.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extension
extensions = [
    Extension(
        "farthest_cython", # This will be the name to import in Python
        ["farthest_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

# Use Cython to build the extension module
setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
)
