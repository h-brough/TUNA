# TUNA

Welcome to TUNA! A user-friendly quantum chemistry program for diatomics.

<br>
<p align="center"><img src="TUNA Logo.svg" alt="Fish swimming through a wavepacket" width=480 /></p>

## Contents

The repository includes:

* This README file
* The TUNA logo
* The file LICENSE with the MIT license
* The folder TUNA containing Python files
* The installation file pyproject.toml
* The TUNA manual
* A changelog

## Documentation

A copy of the <a href="./TUNA%20Manual.pdf" style="color:black;">TUNA Manual</a> can be found in this repository, and in the directory where the Python files are installed.

## Using TUNA

### Prerequisites
The program requires Python 3.12 or higher and the following packages:

* numpy
* scipy
* matplotlib
* plotly
* scikit-image
* termcolor

### Installation

The simplest way to install TUNA is by running

```
pip install QuantumTUNA
```

### Running

Add the folder where TUNA was installed to PATH, and then run ```TUNA --version``` which should print the correct version if TUNA has installed correctly.

The syntax of the command to run a TUNA calculation is:

```
TUNA [Calculation] : [Atom A] [Atom B] [Distance] : [Method] [Basis]
```

Read the manual for details!
