from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os, platform, shutil, subprocess

if platform.system() == "Windows":

    extra_compile_args, extra_link_args = ["/O2", "/openmp"], []

elif platform.system() == "Darwin":

    libomp = os.environ.get("LIBOMP_PREFIX", "")

    if not libomp and shutil.which("brew"):

        try: libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        except subprocess.CalledProcessError: libomp = ""

    if not os.path.isdir(libomp):

        libomp = next((c for c in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp") if os.path.isdir(c)), None)

        if libomp is None: raise SystemExit("libomp not found - install it with `brew install libomp` or set LIBOMP_PREFIX")

    extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp", f"-I{libomp}/include"]
    extra_link_args = [f"-L{libomp}/lib", "-lomp", f"-Wl,-rpath,{libomp}/lib"]

else:

    extra_compile_args, extra_link_args = ["-O3", "-fopenmp"], ["-fopenmp"]


extension = Extension(
    "TUNA.tuna_integrals.tuna_integral",
    ["TUNA/tuna_integrals/tuna_integral.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=cythonize(extension, compiler_directives={"language_level": "3", "boundscheck": False, "wraparound": False, "cdivision": True, "nonecheck": False, "initializedcheck": False}))