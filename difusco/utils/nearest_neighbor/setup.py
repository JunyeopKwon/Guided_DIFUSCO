from setuptools import setup
from Cython.Build import cythonize
import numpy

# To compile, run this command in your terminal:
# python setup.py build_ext --inplace

setup(
    name="Nearest Neighborhood TSP Solver",
    ext_modules=cythonize(
        "nearest_neighbor_c.pyx",
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)