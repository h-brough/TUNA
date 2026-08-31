# TUNA

[![Tests](https://github.com/h-brough/TUNA/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/h-brough/TUNA/actions/workflows/build-wheels.yml)
[![PyPI version](https://img.shields.io/pypi/v/quantumtuna.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/QuantumTUNA)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/quantumtuna.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/QuantumTUNA)
[![License](https://img.shields.io/github/license/h-brough/TUNA.svg)](LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/quantumtuna.svg)](https://pypi.org/project/QuantumTUNA/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.01471-blue)](https://arxiv.org/abs/2604.01471)

Welcome to TUNA! A streamlined quantum chemistry program for atoms and diatomics. 

The program contains a collection of quantum chemistry methods, and considerable effort has been taken to document everything. The manual provides numerous examples and explanations for how TUNA works.

<br>
<p align="center"><img src="https://raw.githubusercontent.com/h-brough/TUNA/refs/heads/main/TUNA%20Logo.svg" alt="Fish swimming through a wavepacket." width=400 /></p>

## Using TUNA

### Prerequisites
The program requires Python 3.12 or higher and the following packages:

* numpy
* scipy
* matplotlib
* termcolor

### Installation

The simplest way to install TUNA and its dependencies is by running:

```
pip install quantumtuna
```

Then, in a new terminal, run ```TUNA --version``` which should print the current version if TUNA has installed correctly.

### Running

The syntax of the command to run a TUNA calculation is:

```
TUNA [Calculation] : [Atom A] [Atom B] [Distance] : [Method] [Basis]
```

For example, a geometry optimisation on dihydrogen, starting at 1.0 angstroms with B3LYP/6-31G is:

```
TUNA OPT : H H 1.0 : B3LYP 6-31G
```

And a single point energy calculation on the beryllium atom with minimal basis CCSDTQ is:

```
TUNA SPE : Be : CCSDTQ STO-3G
```


## Documentation

The <a href="https://github.com/h-brough/TUNA/blob/main/docs/TUNA%20Manual.pdf">TUNA Manual</a> can be found in this repository, and in the directory where the Python files are installed. Many copy-and-pasteable examples for how to use TUNA are found there. 

A concise description of the program can be found in the <a href="https://arxiv.org/abs/2604.01471">arXiv paper</a>.

## Citation

If TUNA is used in a publication, please consider citing it as follows:

H. Brough, TUNA: A streamlined quantum chemistry program for atoms and diatomics, arXiv preprint, 2026, <a href="https://arxiv.org/abs/2604.01471">arXiv:2604.01471</a>.
