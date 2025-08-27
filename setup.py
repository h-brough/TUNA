
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="tuna_integral",                # adjust name if your package dir is different
        sources=["TUNA/tuna_integrals/tuna_integral.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="quantumtuna",
    version="0.7.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    include_package_data=True,
    zip_safe=False,
)
