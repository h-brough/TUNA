from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="tuna_integral",
        sources=["tuna_integral.pyx"],
        include_dirs=[numpy.get_include()], 
    )
]

setup(
    ext_modules=cythonize(extensions,compiler_directives={"language_level": "3"})
)