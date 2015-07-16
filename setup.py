from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import cython_gsl
import numpy as np

extensions = [
    Extension("annihilating_coalescing_walks.annihilating_coalescing_walks", sources=["annihilating_coalescing_walks/annihilating_coalescing_walks.pyx"],
              language="c++", libraries = cython_gsl.get_libraries(),
              library_dirs = [cython_gsl.get_library_dir()],
              include_dirs = [cython_gsl.get_cython_include_dir(), np.get_include()]),
    Extension("annihilating_coalescing_walks.annihilating_coalescing_walks_inflation", sources=["annihilating_coalescing_walks/annihilating_coalescing_walks_inflation.pyx"],
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
    ext_modules = cythonize(extensions, annotate=True),
    py_modules = ['annihilating_coalescing_walks_utility']
)