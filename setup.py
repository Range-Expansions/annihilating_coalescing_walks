from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy as np

extensions = [
    Extension("annihilating_coalescing_walks.linear",
              sources=["annihilating_coalescing_walks/linear.pyx"],
              language="c++", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()]),
    Extension("annihilating_coalescing_walks.inflation",
              sources=["annihilating_coalescing_walks/inflation.pyx"],
              language="c++", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()])
]

setup(
    name='annihilating-coalescing-walks',
    version='0.2',
    packages=['annihilating_coalescing_walks'],
    url='',
    license='',
    author='Bryan Weinstein',
    author_email='bweinstein@seas.harvard.edu',
    description='',
    include_dirs = [cython_gsl.get_include(), np.get_include()],
    ext_modules = cythonize(extensions, annotate=True)
)