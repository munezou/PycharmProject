from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("c_def2.pyx"), )