from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("from_ndarray", sources=["from_ndarray.pyx", "test.c"], include_dirs=['.', get_include()])
setup(name="from_ndarray", ext_modules=cythonize([ext]))

