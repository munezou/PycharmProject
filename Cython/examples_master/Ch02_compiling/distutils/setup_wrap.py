from distutils.core import setup, Extension
from Cython.Build import cythonize

"""
------------------------------------------
C source code converted to a C extension so that it can be read in Python.
-------------------------------------------
"""

# First create an Extension object with the appropriate name and sources
ext = Extension(name="wrap_fib", sources=["cfib.c", "wrap_fib.pyx"])

# Use cythonize on the extension object.
setup(ext_modules=cythonize(ext))
