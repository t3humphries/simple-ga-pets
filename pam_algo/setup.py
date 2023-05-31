from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension("_k_medoids_helper", ["_k_medoids_helper.pyx"], include_dirs=[numpy.get_include()])]),
    include_dirs=[numpy.get_include()],
)