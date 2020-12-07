from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("for_loop_test_00.pyx"), )