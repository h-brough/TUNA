import numpy as np
import pytest

from conftest import constants, thermochemistry_module


"""

This module tests the calculation types that sit on top of an energy evaluation: gradients, geometry
optimisation, harmonic frequencies and the thermochemical corrections that follow them.

These are the slowest tests in the suite, because each one runs many energy evaluations, so most of
them carry the "slow" marker. The reference bond length and frequency were obtained by fitting PySCF
energies on a grid of bond lengths and differentiating that fit, which is independent of how TUNA
takes its own derivatives.

"""


# Hydrogen fluoride, RHF/6-31G. Bond length from a quartic fit to PySCF energies, frequency from
# central differences of PySCF energies extrapolated to zero step size.

REFERENCE_BOND_LENGTH_IN_ANGSTROMS = 0.9208533
REFERENCE_ENERGY_AT_MINIMUM = -99.9834255057
REFERENCE_FREQUENCY_PER_CM = 4135.34




@pytest.mark.slow
def test_optimised_geometry_matches_an_independent_reference(tuna):

    """

    The optimiser has to find the minimum another program finds.

    """

    molecule, energy = tuna.optimise("OPT : H F 0.9 : HF 6-31G")

    bond_length_in_angstroms = molecule.bond_length * constants.bohr_radius_in_angstrom

    assert bond_length_in_angstroms == pytest.approx(REFERENCE_BOND_LENGTH_IN_ANGSTROMS, abs = 1e-4)
    assert energy == pytest.approx(REFERENCE_ENERGY_AT_MINIMUM, abs = 1e-7)




@pytest.mark.slow
def test_optimisation_lowers_the_energy(tuna):

    """

    Starting away from the minimum, the optimised energy has to come out below the starting one.

    """

    starting_energy = tuna.energy("SPE : H F 0.9 : HF 6-31G")

    _, optimised_energy = tuna.optimise("OPT : H F 0.9 : HF 6-31G")

    assert optimised_energy < starting_energy




@pytest.mark.slow
def test_gradient_vanishes_at_the_optimised_geometry(tuna):

    """

    A converged optimisation is a stationary point, so the gradient there is zero.

    """

    gradient = tuna.gradient(f"SPE : H F {REFERENCE_BOND_LENGTH_IN_ANGSTROMS} : HF 6-31G")

    assert gradient == pytest.approx(0.0, abs = 1e-5)




def test_gradient_points_downhill(tuna):

    """

    Compressed below the equilibrium bond length the energy falls as the bond lengthens, so the
    derivative with respect to bond length is negative, and the other way round when stretched.

    """

    assert tuna.gradient("SPE : H F 0.85 : HF 6-31G") < 0
    assert tuna.gradient("SPE : H F 1.00 : HF 6-31G") > 0




@pytest.mark.slow
def test_harmonic_frequency_matches_an_independent_reference(tuna):

    """

    The harmonic frequency at the minimum, against finite differences of PySCF energies.

    """

    frequency = tuna.frequency(f"FREQ : H F {REFERENCE_BOND_LENGTH_IN_ANGSTROMS} : HF 6-31G")

    assert frequency.frequency_per_cm == pytest.approx(REFERENCE_FREQUENCY_PER_CM, abs = 0.5)




@pytest.mark.slow
def test_frequency_follows_from_the_force_constant_and_the_reduced_mass(tuna):

    """

    The reported frequency has to be the square root of the force constant over the reduced mass, and
    the zero-point energy has to be half of it.

    """

    frequency = tuna.frequency(f"FREQ : H F {REFERENCE_BOND_LENGTH_IN_ANGSTROMS} : HF 6-31G")

    frequency_in_hartree = frequency.frequency_per_cm / constants.per_cm_in_hartree

    assert np.sqrt(frequency.force_constant / frequency.reduced_mass) == pytest.approx(frequency_in_hartree, rel = 1e-8)
    assert frequency.zero_point_energy == pytest.approx(frequency_in_hartree / 2, rel = 1e-8)




@pytest.mark.slow
def test_reduced_mass_comes_from_the_atomic_masses(tuna):

    """

    The reduced mass used for the vibration has to match the masses of the two atoms, which the
    Molecule object stores in electron masses.

    """

    frequency = tuna.frequency(f"FREQ : H F {REFERENCE_BOND_LENGTH_IN_ANGSTROMS} : HF 6-31G")

    masses = tuna.run(f"SPE : H F {REFERENCE_BOND_LENGTH_IN_ANGSTROMS} : HF 6-31G").molecule.masses

    assert frequency.reduced_mass == pytest.approx(masses[0] * masses[1] / (masses[0] + masses[1]), rel = 1e-10)

    # The masses themselves should be those of the most abundant isotopes, in unified mass units

    assert masses / constants.atomic_mass_unit_in_electron_mass == pytest.approx([1.00782503, 18.99840322], abs = 1e-6)




@pytest.mark.parametrize("temperature", [100.0, 298.15, 1000.0])
def test_translational_internal_energy_is_three_halves_kt(temperature):

    """

    Three translational degrees of freedom, each worth half of kT.

    """

    assert thermochemistry_module.calculate_translational_internal_energy(temperature) == pytest.approx(1.5 * constants.k * temperature, rel = 1e-12)




@pytest.mark.parametrize("temperature", [100.0, 298.15, 1000.0])
def test_rotational_internal_energy_is_kt_for_a_linear_molecule(temperature):

    """

    A diatomic has two rotational degrees of freedom, each worth half of kT.

    """

    assert thermochemistry_module.calculate_rotational_internal_energy(temperature) == pytest.approx(constants.k * temperature, rel = 1e-12)




def test_enthalpy_is_the_internal_energy_plus_kt():

    """

    The ideal gas relation between enthalpy and internal energy.

    """

    internal_energy, temperature = -99.5, 298.15

    assert thermochemistry_module.calculate_enthalpy(internal_energy, temperature) == pytest.approx(internal_energy + constants.k * temperature, rel = 1e-12)




def test_free_energy_is_the_enthalpy_less_the_entropy_term():

    """

    The definition of the Gibbs free energy.

    """

    enthalpy, temperature, entropy = -99.5, 298.15, 7.5e-5

    assert thermochemistry_module.calculate_free_energy(enthalpy, temperature, entropy) == pytest.approx(enthalpy - temperature * entropy, rel = 1e-12)