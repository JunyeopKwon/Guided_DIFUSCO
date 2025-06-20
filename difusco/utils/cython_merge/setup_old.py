# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

numpy_include_dir = numpy.get_include()
print(f"NumPy include directory: {numpy_include_dir}") # Add this line

ext = Extension("cython_merge", ["cython_merge.pyx"],
                include_dirs=[numpy.get_include()],
                )

setup(ext_modules=[ext], cmdclass={'build_ext': build_ext})
