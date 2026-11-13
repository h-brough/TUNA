import contextlib
import importlib.util
import inspect
import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy import ndarray


"""

This is the shared configuration module for the TUNA test suite, written for version 0.11.2.

The suite runs against whichever TUNA is importable: an installed package if there is one, otherwise the
checkout this file sits in. That choice is made here, once, together with the helpers that the test modules
use to run a calculation and get numbers back out of it. Tests never import anything from TUNA themselves - they ask
for the "tuna" fixture and go through the runner object, so there is only one place to fix if the way
TUNA is driven ever changes.

The module contains:

1. Path setup and a quiet import of the TUNA modules
2. A function to undo TUNA's in-place edits of its global method table (reset_method_table)
3. Containers for the results of a calculation (EnergyResult, ExcitedStatesResult, QuasiparticleResult, FrequencyResult,
   CommandLineResult)
4. The runner object handed to tests by the fixture (TunaRunner)
5. The fixtures themselves

"""


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TUNA_DIRECTORY = REPOSITORY_ROOT / "TUNA"

# Only fall back to the checkout when TUNA is not already installed. Inserting it unconditionally would
# shadow an installed package with an unbuilt source tree, which is exactly what happens when cibuildwheel
# runs the suite against a freshly built wheel.

if importlib.util.find_spec("TUNA") is None:

    sys.path.insert(0, str(REPOSITORY_ROOT))


# TUNA prints its banner while being imported, and tuna.py reads sys.argv looking for "--version" at
# import time, so the arguments pytest was given are hidden from it here.

_saved_argv, sys.argv = sys.argv, sys.argv[:1]

try:

    with contextlib.redirect_stdout(io.StringIO()):

        import TUNA.tuna as tuna_main
        import TUNA.tuna_ci as configuration_interaction_module
        import TUNA.tuna_energy as energy_module
        import TUNA.tuna_freq as frequency_module
        import TUNA.tuna_mbpt as many_body_module
        import TUNA.tuna_opt as optimisation_module
        import TUNA.tuna_props as properties_module
        import TUNA.tuna_thermo as thermochemistry_module

        from TUNA.tuna_calc import Calculation
        from TUNA.tuna_util import TunaError, constants, electronic_structure_methods

except ImportError as import_error:

    raise ImportError(
        f"Could not import TUNA from {TUNA_DIRECTORY}: {import_error}.\n"
        "The compiled integral extension is probably missing - build it with "
        "\"python setup.py build_ext --inplace\" from the repository root."
    ) from import_error

finally:

    sys.argv = _saved_argv


# Several TUNA functions edit the Method objects in the global "electronic_structure_methods" list in place

_ORIGINAL_METHOD_STATE = [(method, method.name, method.unrestricted) for method in electronic_structure_methods]

ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*m")



def reset_method_table() -> None:

    """

    Restores the names and reference flags of TUNA's global electronic structure methods.

    """

    for method, name, unrestricted in _ORIGINAL_METHOD_STATE:

        method.name = name
        method.unrestricted = unrestricted

    return




@dataclass(frozen=True)
class EnergyResult:

    """

    Holds everything an energy evaluation produces that a test might want to look at.

    """

    # Final total energy in hartree, including any correlation or excited state contribution

    energy: float

    # The Output object from the self-consistent field cycle

    scf_output: object

    # The Molecule object the calculation was run on

    molecule: object

    # Final density matrix in the AO basis

    density: ndarray

    # The Calculation object, as TUNA left it - this records the method that actually ran, which TUNA may have simplified

    calculation: object

    @property
    def scf_energy(self) -> float:

        return self.scf_output.energy

    @property
    def correlation_energy(self) -> float:

        # For an excited state method this is the excitation energy instead

        return self.energy - self.scf_output.energy

    @property
    def integrals(self) -> object:

        return self.scf_output.integrals

    @property
    def orbital_energies(self) -> ndarray:

        return self.scf_output.epsilons




@dataclass(frozen=True)
class ExcitedStatesResult:

    """

    Holds every excited state an excited state calculation found, not only the one chosen with ROOT.

    """

    # Energy evaluation for the chosen state, whose energy is the total energy of that state

    result: EnergyResult

    # Every excitation energy in hartree, in the order TUNA reports them, lowest first

    excitation_energies: ndarray

    # Spin of each state ("singlet", "triplet" and so on), or an empty string where TUNA does not assign one

    state_types: ndarray

    # Oscillator strength of each transition from the ground state

    oscillator_strengths: ndarray

    # Expectation value of S^2 for each state, or None for the methods where TUNA does not calculate it

    spin_squared: ndarray | None

    @property
    def energy(self) -> float:

        return self.result.energy




@dataclass(frozen=True)
class QuasiparticleResult:

    """

    Holds the G0W0 quasiparticle energies alongside the orbital energies they correct.

    """

    # Quasiparticle energy of every orbital in hartree - spatial orbitals for a restricted reference, and spin orbitals in order
    # of increasing orbital energy for an unrestricted one

    quasiparticle_energies: ndarray

    # Orbital energies of the reference in the same order

    orbital_energies: ndarray

    # Number of occupied orbitals in that same order

    n_occupied: int

    @property
    def homo(self) -> float:

        return self.quasiparticle_energies[self.n_occupied - 1]

    @property
    def lumo(self) -> float:

        return self.quasiparticle_energies[self.n_occupied]




@dataclass(frozen=True)
class FrequencyResult:

    """

    Holds the results of a harmonic frequency calculation.

    """

    force_constant: float
    reduced_mass: float
    frequency_per_cm: float
    zero_point_energy: float




@dataclass(frozen=True)
class CommandLineResult:

    """

    Holds the results of running TUNA as a subprocess, with the terminal colour codes removed.

    """

    returncode: int
    output: str




class TunaRunner:

    """

    Drives TUNA from a test.

    Every method takes a TUNA input line in the usual "SPE : H H 0.74 : HF STO-3G" form. The line is
    upper-cased first, so tests may be written in whichever case reads best. All printing is suppressed.

    """

    # Exposed so that tests can assert on the errors TUNA raises without importing from TUNA themselves

    TunaError = TunaError

    @property
    def version(self) -> str:

        return tuna_main.VERSION


    def calculation(self, input_line: str) -> tuple:

        """

        Parses an input line and builds the Calculation object, without running anything.

        Args:
            input_line (str): TUNA input line

        Returns:
            calculation (Calculation): Calculation object
            atomic_symbols (list): List of atomic symbols
            coordinates (array): Atomic coordinates in bohr

        """

        reset_method_table()

        calculation_type, method_string, basis, atomic_symbols, coordinates, params = tuna_main.parse_input(input_line.upper())

        method = tuna_main.process_method(method_string)

        calculation = Calculation(calculation_type, method, 0.0, params, basis, atomic_symbols, True)

        return calculation, atomic_symbols, coordinates


    def run(self, input_line: str) -> EnergyResult:

        """

        Runs a single energy evaluation.

        Args:
            input_line (str): TUNA input line

        Returns:
            result (EnergyResult): Energy, output object, molecule and density

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        scf_output, molecule, energy, density = energy_module.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent = True)

        return EnergyResult(float(energy), scf_output, molecule, density, calculation)


    def energy(self, input_line: str) -> float:

        """

        Runs a single energy evaluation and returns only the final total energy.

        Args:
            input_line (str): TUNA input line

        Returns:
            energy (float): Final total energy in hartree

        """

        return self.run(input_line).energy


    def excited_states(self, input_line: str) -> ExcitedStatesResult:

        """

        Runs an excited state calculation and collects every state it found.

        TUNA only returns the state chosen with ROOT, so the full set is read from the arguments of the two functions every
        excited state method hands its states to for printing. Those are wrapped for the duration of the calculation only.

        Args:
            input_line (str): TUNA input line for an excited state method, or with the TD keyword

        Returns:
            result (ExcitedStatesResult): Chosen state, and the energy, spin and intensity of every state

        """

        captured_arguments = {}
        original_functions = {}


        def make_recorder(function_name: str):

            original_function = getattr(configuration_interaction_module, function_name)
            signature = inspect.signature(original_function)

            def recorder(*args, **kwargs):

                captured_arguments[function_name] = signature.bind(*args, **kwargs).arguments

                return original_function(*args, **kwargs)

            return original_function, recorder


        try:

            for function_name in ("print_excited_state_absorption_spectrum", "print_excited_state_contributions"):

                original_functions[function_name], recorder = make_recorder(function_name)

                setattr(configuration_interaction_module, function_name, recorder)

            result = self.run(input_line)

        finally:

            for function_name, original_function in original_functions.items():

                setattr(configuration_interaction_module, function_name, original_function)

        # Every excited state method reports its spectrum, while only the single-reference and EOM-CCSD methods report a spin for each state

        spectrum = captured_arguments["print_excited_state_absorption_spectrum"]
        spin_squared = captured_arguments.get("print_excited_state_contributions", {}).get("s_squared")

        return ExcitedStatesResult(result, np.array(spectrum["excitation_energies"], dtype = float), np.array(spectrum["state_types"]),
                                   np.array(spectrum["oscillator_strengths"], dtype = float), None if spin_squared is None else np.array(spin_squared, dtype = float))


    def quasiparticle_energies(self, input_line: str) -> QuasiparticleResult:

        """

        Calculates the G0W0 quasiparticle energies on top of a Hartree-Fock or Kohn-Sham reference.

        TUNA only runs G0W0 when it is printing, so it is called here directly on the converged reference, exactly as the
        "GW" keyword would call it.

        Args:
            input_line (str): TUNA input line for the reference calculation

        Returns:
            result (QuasiparticleResult): Quasiparticle energies, the orbital energies they correct, and the number occupied

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        scf_output, molecule, _, _ = energy_module.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent = True)

        quasiparticle_energies = many_body_module.calculate_G0W0(molecule, scf_output, calculation, silent = True)

        # An unrestricted reference works in spin orbitals, ordered by orbital energy

        if calculation.reference == "UHF":

            return QuasiparticleResult(np.array(quasiparticle_energies), np.sort(scf_output.epsilons_combined), molecule.n_occ)

        return QuasiparticleResult(np.array(quasiparticle_energies), np.array(scf_output.epsilons), molecule.n_doubly_occ)


    def optimise(self, input_line: str) -> tuple:

        """

        Optimises a geometry.

        Args:
            input_line (str): TUNA input line, with calculation type OPT

        Returns:
            molecule (Molecule): Optimised molecule object
            energy (float): Energy at the optimised geometry in hartree

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        molecule, energy = optimisation_module.optimise_geometry(calculation, atomic_symbols, coordinates)

        return molecule, float(energy)


    def gradient(self, input_line: str) -> float:

        """

        Calculates the derivative of the energy with respect to bond length.

        Args:
            input_line (str): TUNA input line

        Returns:
            gradient (float): Energy derivative in hartree per bohr

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        return float(optimisation_module.calculate_gradient(coordinates, calculation, atomic_symbols, silent = True))


    def frequency(self, input_line: str) -> FrequencyResult:

        """

        Calculates the harmonic vibrational frequency at the given geometry.

        Args:
            input_line (str): TUNA input line, with calculation type FREQ

        Returns:
            result (FrequencyResult): Force constant, reduced mass, frequency and zero-point energy

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        force_constant, reduced_mass, frequency_per_cm, zero_point_energy = frequency_module.calculate_harmonic_frequency(calculation, atomic_symbols = atomic_symbols, coordinates = coordinates)

        return FrequencyResult(float(force_constant), float(reduced_mass), float(frequency_per_cm), float(zero_point_energy))


    def charged_state_energies(self, input_line: str, charge_delta: int) -> tuple:

        """

        Calculates the reference and charged state energies used for ionisation potentials and
        electron affinities.

        Args:
            input_line (str): TUNA input line, with calculation type IP or EA
            charge_delta (int): Change in charge, +1 to ionise and -1 to attach

        Returns:
            reference_energy (float): Energy of the reference state in hartree
            charged_energy (float): Energy of the charged state in hartree

        """

        calculation, atomic_symbols, coordinates = self.calculation(input_line)

        reference_energy, charged_energy, _, _ = optimisation_module.calculate_charged_state_energies(calculation, atomic_symbols, coordinates, charge_delta = charge_delta)

        return float(reference_energy), float(charged_energy)


    def command_line(self, input_line: str) -> CommandLineResult:

        """

        Runs TUNA the way a user would, as a separate process.

        Args:
            input_line (str): Arguments to pass to tuna.py, separated by spaces

        Returns:
            result (CommandLineResult): Exit code, and the combined output with colour codes stripped

        """

        completed_process = subprocess.run([sys.executable, "-m", "TUNA.tuna"] + input_line.split(), capture_output = True, text = True, timeout = 300)

        output = ANSI_ESCAPE_PATTERN.sub("", completed_process.stdout + completed_process.stderr)

        return CommandLineResult(completed_process.returncode, output)




@pytest.fixture(scope = "session")
def tuna() -> TunaRunner:

    """

    Gives a test the runner used to drive TUNA.

    """

    return TunaRunner()




@pytest.fixture(autouse = True)
def _restore_global_method_state():

    """

    Puts TUNA's global method table back before and after every test, so that a calculation in one
    test cannot change the meaning of a method name in the next one.

    """

    reset_method_table()

    yield

    reset_method_table()