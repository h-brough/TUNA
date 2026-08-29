# Contributing to TUNA

Thanks for your interest in TUNA. Bug reports, questions and pull requests are all welcome.

## Ways to help

- **Bug reports or feature ideas** — open an [issue](https://github.com/h-brough/TUNA/issues).
- **Code** — new methods, keywords, performance work, or fixes, via a pull request.

If you are planning something substantial, please open an issue first so we can agree on the approach before you write the code.

## Reporting a bug

Please include:

- The output of `TUNA --version`, plus your OS and Python version.
- The exact input line, e.g. `TUNA OPT : H H 1.0 : B3LYP 6-31G`.
- What you expected, and what actually happened. For wrong numbers, give the reference value and where it came from.
- The relevant part of the output, or the full traceback if TUNA crashed.

## Development setup

TUNA needs Python 3.12 or higher, a C compiler with OpenMP, and `numpy`, `scipy`, `matplotlib` and `termcolor`.

```
git clone https://github.com/h-brough/TUNA.git
cd TUNA
pip install numpy scipy matplotlib termcolor cython
python setup.py build_ext --inplace
```

The last step compiles `TUNA/tuna_integrals/tuna_integral.pyx`. Rerun it whenever you change the `.pyx` file; pure Python changes need no rebuild.

Run from the checkout with:

```
python TUNA/tuna.py SPE : H H 0.74 : HF STO-3G
```

On macOS you will need `libomp` (`brew install libomp`), or set `LIBOMP_PREFIX` to point at your own build.

## Code style

Please try to match the style of TUNA functions:

- Four-space indents, and generous vertical whitespace — roughly ten blank lines between top-level functions, and a blank line after `def` before the docstring.
- Type hints on function signatures, e.g. `def calculate_coulomb_matrix(P: ndarray, ERI_AO: ndarray) -> ndarray:`.
- A docstring on every function, in the existing style:

```python
def calculate_coulomb_matrix(P: ndarray, ERI_AO: ndarray) -> ndarray:

    """

    Calculates the classical electron repulsion matrix.

    Args:
        P (array): Density matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis

    Returns:
        J (array): Coulomb matrix in AO basis

    """
```

- Descriptive names over short ones, and spaces around `=` in keyword arguments (`optimize = True`).
- If you add a new module, give it the same header docstring as the others: what it is for, and a numbered list of what it contains.

## Verifying changes

There is no automated test suite, so anything touching the numbers has to be checked by hand. Cross-check against an established program and quote the numbers in your pull request. For example:

```python
from pyscf import gto, scf

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
print(scf.RHF(mol).kernel())     # -1.1167593074
```

against

```
TUNA SPE : H H 0.74 : HF STO-3G  # Final single point energy: -1.1167593075
```

Some things worth checking, depending on what you changed:

- Energies and properties against PySCF, Psi4, ORCA or the literature, ideally in more than one basis and for both closed- and open-shell cases.
- Analytic derivatives against finite differences.
- That existing calculations still give the same answers — a change to shared machinery like the SCF or integral code can move results far from where you were working.

## Pull requests

- One logical change per pull request, kept as small as it can reasonably be.
- Say what you changed, why, and how you checked it.
- Add an entry to `CHANGELOG.md` under the current unreleased version, in the existing *Added / Changed / Fixed* format.
- New methods, keywords or defaults need documenting. The manual source is not in the repository, so note in your pull request what should be written up and it will be folded into the next release.

## Licence

TUNA is MIT licensed. By contributing, you agree that your contribution is released under the same licence.
