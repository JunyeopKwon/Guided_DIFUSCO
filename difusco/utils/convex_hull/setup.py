# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extension module
extensions = [
    Extension(
        "convex_hull_cython",  # This will be the name to import
        ["convex_hull_cython.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

# Use Cythonize to build the extension
setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
)
