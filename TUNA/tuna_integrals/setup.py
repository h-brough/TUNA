from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform, os, subprocess, shutil

# The compiler flags are dependent on the platform

if platform.system() == "Windows":

    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []

elif platform.system() == "Darwin":

    # Apple Clang needs the OpenMP flag passed to the frontend, and the runtime
    # comes from Homebrew's libomp (keg-only, so paths must be given explicitly).
    libomp = ""

    if shutil.which("brew"):
        try:
            libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        except subprocess.CalledProcessError:
            libomp = ""

    if not os.path.isdir(libomp):
        for candidate in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if os.path.isdir(candidate):
                libomp = candidate
                break
        else:
            raise SystemExit("libomp not found - install it with `brew install libomp`")

    extra_compile_args = ["-O3", "-Xpreprocessor", "-fopenmp", f"-I{libomp}/include"]
    extra_link_args = [f"-L{libomp}/lib", "-lomp", f"-Wl,-rpath,{libomp}/lib"]

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
