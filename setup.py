
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import scipy
import pathlib


version_namespace = {}

with open("TUNA/__init__.py", "r", encoding="utf-8") as f:
    
    exec(f.read(), version_namespace)

version = version_namespace["__version__"]


scipy_include = pathlib.Path(scipy.__file__).parent / "special" / "cython"


extensions = [
    Extension(
        name="TUNA.tuna_integrals.tuna_integral",
        sources=["TUNA/tuna_integrals/tuna_integral.pyx"],
        include_dirs=[numpy.get_include(), str(scipy_include)],
    )
]

setup(
    name="QuantumTUNA",
    version=version, 
    packages=["TUNA", "TUNA.tuna_integrals"],
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    include_package_data=True,
    zip_safe=False,
)

#Need to include PDF next time too as well as bat