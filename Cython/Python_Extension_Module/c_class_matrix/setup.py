from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('c_class_matrix.pyx'), )