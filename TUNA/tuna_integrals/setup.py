from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform, os

# The compiler flags are dependent on the platform

if platform.system() == "Windows":

    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
    
elif platform.system() == "Darwin":

    os.environ["CC"] = "gcc-mp-15"
    os.environ["CXX"] = "g++-mp-15"
    
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"] 
    
else:

    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [

    Extension(
        name="tuna_integral",
        sources=["tuna_integral.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    
]

setup(

    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
    )
    
)
