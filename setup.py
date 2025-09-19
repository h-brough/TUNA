
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import scipy
import pathlib

scipy_include = pathlib.Path(scipy.__file__).parent / "special" / "cython"


extensions = [
    Extension(
        name="TUNA.tuna_integrals.tuna_integral",                # adjust name if your package dir is different
        sources=["TUNA/tuna_integrals/tuna_integral.pyx"],
        include_dirs=[numpy.get_include(), str(scipy_include)],
    )
]

setup(
    name="quantumtuna",
    version="0.8.0",
    packages=[
        "TUNA",
        "TUNA.tuna_integrals", ],    
    package_data={
        "TUNA": ["*.bat", "*.pdf"],
    },
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    include_package_data=True,
    zip_safe=False,
)
