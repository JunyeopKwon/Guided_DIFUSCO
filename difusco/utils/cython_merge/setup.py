from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys # Keep sys for printing executable if you want

numpy_include_dir = numpy.get_include()
print(f"--- Using Python executable: {sys.executable} ---")
print(f"--- Using NumPy version: {numpy.__version__} ---")
print(f"--- NumPy include directory from numpy.get_include(): {numpy_include_dir} ---")

ext = Extension(
    "cython_merge",
    ["cython_merge.pyx"],
    include_dirs=[numpy_include_dir],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

setup(
    name='CythonMergeModule',
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext}
)