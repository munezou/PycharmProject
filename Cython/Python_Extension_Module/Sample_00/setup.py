from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("sample", sources=["sample.pyx", "csample.c"], include_dirs=['.', get_include()])
setup(name="sample", ext_modules=cythonize([ext]))