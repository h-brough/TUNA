# TUNA

[![PyPI version](https://img.shields.io/pypi/v/quantumtuna.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/QuantumTUNA)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/quantumtuna.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/QuantumTUNA)
[![License](https://img.shields.io/github/license/h-brough/TUNA.svg)](LICENSE)
[![PyPI downloads](https://img.shields.io/pypi/dm/quantumtuna.svg)](https://pypistats.org/packages/QuantumTUNA)


Welcome to TUNA! A user-friendly quantum chemistry program for diatomics. The program contains a collection of quantum chemistry methods, and considerable effort has been taken to document everything. The accompanying manual provides numerous examples and explanations for how TUNA works.

<br>
<p align="center"><img src="https://raw.githubusercontent.com/h-brough/TUNA/9dd6ad4a5705cd38ffa64add120b2296aa9068ce/TUNA%20Logo.svg" alt="Fish swimming through a wavepacket." width=480 /></p>

## Contents

The repository includes:

* This README file
* The TUNA logo
* The file LICENSE with the MIT license
* The folder TUNA containing Python files
* A folder with the GitHub workflows for publishing
* The installation file pyproject.toml
* The installation file setup.py
* The TUNA manual
* A changelog

## Documentation

A copy of the <a href="https://github.com/h-brough/TUNA/blob/main/TUNA%20Manual.pdf">TUNA Manual</a> can be found in this repository, and in the directory where the Python files are installed.

## Using TUNA

### Prerequisites
The program requires Python 3.12 or higher and the following packages:

* numpy
* scipy
* matplotlib
* termcolor

### Installation

The simplest way to install TUNA and its dependencies is by running

```
pip install QuantumTUNA
```

Find the path to where TUNA is installed, `*/TUNA/`, with the other Python site packages.

On Windows, add this folder to PATH by editing the system environment variables.


On MacOS and Linux, find this folder's path and from a terminal, run

```
echo "alias tuna='noglob python3 /*/TUNA/tuna.py'" >> ~/.zshrc
echo "alias TUNA='noglob python3 /*/TUNA/tuna.py'" >> ~/.zshrc
source ~/.zshrc
```

Then, in a new terminal, run ```TUNA --version``` which should print the correct version if TUNA has installed correctly.

### Running

The syntax of the command to run a TUNA calculation is

```
TUNA [Calculation] : [Atom A] [Atom B] [Distance] : [Method] [Basis]
```

Read the manual for details!
