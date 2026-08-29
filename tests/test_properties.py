import pytest

from conftest import constants, properties_module


"""

This module tests the properties TUNA reports alongside an energy, and the table of physical constants
everything else is built on.

The dipole moment gets three separate tests, because there are three ways to arrive at it and they
should all agree: the analytic expression, a numerical derivative of the energy in a finite electric
field, and the value another program reports.

"""


INPUT_LINE = "SPE : H F 0.9168 : HF 6-31G"


# Hydrogen fluoride at 0.9168 angstroms, RHF/6-31G, from PySCF 2.14.0

REFERENCE_DIPOLE_MOMENT = -0.9031272
REFERENCE_ORBITAL_ENERGIES = [-26.27570599, -1.58968170, -0.73905181, -0.63104831, -0.63104831]


# CODATA 2022 values, so the constants are not being checked against themselves

CODATA_BOHR_RADIUS_IN_ANGSTROM = 0.529177210544
CODATA_HARTREE_IN_JOULES = 4.3597447222060e-18
CODATA_HARTREE_IN_EV = 27.211386245981
CODATA_HARTREE_IN_PER_CM = 219474.63136314
CODATA_ATOMIC_MASS_UNIT_IN_ELECTRON_MASS = 1822.888486209
CODATA_BOLTZMANN_CONSTANT_IN_HARTREE_PER_KELVIN = 3.1668115634564e-06




@pytest.fixture(scope = "module")
def result(tuna):

    """

    Runs the reference calculation once for every test in this module.

    """

    return tuna.run(INPUT_LINE)




def dipole_moment(result) -> tuple:

    """

    Calculates the analytic dipole moment of a finished calculation.

    Args:
        result (EnergyResult): Result of an energy evaluation

    Returns:
        moments (tuple): Total, nuclear and electronic dipole moments in atomic units

    """

    molecule = result.molecule

    return properties_module.calculate_analytical_dipole_moment(molecule.centre_of_mass, molecule.charges, molecule.coordinates, result.density, result.integrals.D)




def test_dipole_moment_matches_an_independent_reference(result):

    """

    The analytic dipole moment, against PySCF.

    """

    total, _, _ = dipole_moment(result)

    assert total == pytest.approx(REFERENCE_DIPOLE_MOMENT, abs = 1e-5)




def test_dipole_moment_is_the_sum_of_its_parts(result):

    """

    The total moment is the nuclear contribution plus the electronic one.

    """

    total, nuclear, electronic = dipole_moment(result)

    assert total == pytest.approx(nuclear + electronic, abs = 1e-12)




def test_a_homonuclear_molecule_has_no_dipole_moment(tuna):

    """

    Symmetry forbids it, so this checks the density and the dipole integrals together.

    """

    total, _, _ = dipole_moment(tuna.run("SPE : H H 0.74 : HF 6-31G"))

    assert total == pytest.approx(0.0, abs = 1e-10)




def test_electronic_dipole_moment_matches_a_finite_field_derivative(tuna, result):

    """

    An electric field enters the Hamiltonian through the electrons only, so minus the derivative of the
    energy with respect to the field is the electronic part of the dipole moment.

    """

    _, _, electronic = dipole_moment(result)

    field = 0.0005

    forward = tuna.energy(f"SPE : H F 0.9168 : HF 6-31G : EZ {field}")
    backward = tuna.energy(f"SPE : H F 0.9168 : HF 6-31G : EZ -{field}")

    assert -(forward - backward) / (2 * field) == pytest.approx(electronic, abs = 1e-6)




def test_orbital_energies_match_an_independent_reference(result):

    """

    The occupied orbital energies of hydrogen fluoride, against PySCF.

    """

    assert result.orbital_energies[:5] == pytest.approx(REFERENCE_ORBITAL_ENERGIES, abs = 1e-6)




def test_koopmans_ionisation_potential_is_minus_the_homo_energy(tuna, result):

    """

    Koopmans' theorem is exactly this identity, so it checks the plumbing rather than the physics.

    """

    calculation, _, _ = tuna.calculation(INPUT_LINE)

    ionisation_potential, _, _ = properties_module.calculate_koopmans_parameters(result.orbital_energies, result.molecule.n_doubly_occ, calculation)

    assert ionisation_potential == pytest.approx(-result.orbital_energies[result.molecule.n_doubly_occ - 1], abs = 1e-12)




def test_energy_components_add_up_to_the_total_energy(result):

    """

    The reported components have to account for the whole energy, with nothing left over but the
    nuclear repulsion.

    """

    scf_output = result.scf_output
    molecule = result.molecule

    components = (scf_output.kinetic_energy + scf_output.nuclear_electron_energy + scf_output.coulomb_energy
                  + scf_output.exchange_energy + scf_output.correlation_energy + scf_output.electric_field_energy
                  + scf_output.electric_field_gradient_energy)

    nuclear_repulsion = molecule.charges[0] * molecule.charges[1] / molecule.bond_length

    assert components + nuclear_repulsion == pytest.approx(scf_output.energy, abs = 1e-9)




@pytest.mark.slow
def test_ionisation_potential_matches_a_delta_scf_reference(tuna):

    """

    The vertical ionisation potential as the difference between two separately converged SCF energies,
    against the same difference computed by PySCF.

    """

    reference_ionisation_potential = 0.5238682278

    reference_energy, charged_energy = tuna.charged_state_energies("IP : H F 0.9168 : UHF 6-31G : VERTICAL", charge_delta = +1)

    assert charged_energy - reference_energy == pytest.approx(reference_ionisation_potential, abs = 1e-6)




@pytest.mark.parametrize("attribute, expected", [

    ("bohr_radius_in_angstrom", CODATA_BOHR_RADIUS_IN_ANGSTROM),
    ("hartree_in_joules", CODATA_HARTREE_IN_JOULES),
    ("eV_in_hartree", CODATA_HARTREE_IN_EV),
    ("per_cm_in_hartree", CODATA_HARTREE_IN_PER_CM),
    ("atomic_mass_unit_in_electron_mass", CODATA_ATOMIC_MASS_UNIT_IN_ELECTRON_MASS),
    ("k", CODATA_BOLTZMANN_CONSTANT_IN_HARTREE_PER_KELVIN),

    ])
def test_derived_constants_match_codata(attribute, expected):

    """

    TUNA builds its conversion factors from four fundamental constants rather than tabulating them, so
    this checks the arithmetic against the CODATA 2022 values it should reproduce.

    """

    assert getattr(constants, attribute) == pytest.approx(expected, rel = 1e-8)