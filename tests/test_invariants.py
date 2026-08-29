import pytest


"""

This module tests relations that have to hold whatever the numbers are.

Nothing here needs a reference value from another program, which makes these tests cheap to write and
hard to argue with. They catch a different class of bug from the reference energies: a change that
moves every energy by the same amount would slip past a comparison against PySCF made with the same
tolerance, but not past the variational and equivalence checks below.

"""




def test_energy_decreases_as_the_basis_set_grows(tuna):

    """

    Hartree-Fock is variational, so a larger basis can only lower the energy.

    """

    energies = [tuna.energy(f"SPE : H H 0.74 : HF {basis}") for basis in ["STO-3G", "6-31G", "CC-PVDZ"]]

    assert energies == sorted(energies, reverse = True)




def test_decontracting_the_basis_lowers_the_energy(tuna):

    """

    A decontracted basis spans everything the contracted one does and more.

    """

    contracted = tuna.energy("SPE : H F 0.9168 : HF 6-31G")
    decontracted = tuna.energy("SPE : H F 0.9168 : HF 6-31G : DECONTRACT")

    assert decontracted < contracted




@pytest.mark.parametrize("guess", ["COREGUESS", "SADGUESS", "SCFGUESS"])
def test_converged_energy_does_not_depend_on_the_guess(tuna, guess):

    """

    The starting density chooses the path to the solution, not the solution.

    """

    reference = tuna.energy("SPE : H F 0.9168 : HF 6-31G")

    assert tuna.energy(f"SPE : H F 0.9168 : HF 6-31G : {guess}") == pytest.approx(reference, abs = 1e-7)




def test_converged_energy_does_not_depend_on_diis(tuna):

    """

    DIIS accelerates convergence and must not move the converged energy.

    """

    with_diis = tuna.energy("SPE : H F 0.9168 : HF 6-31G")
    without_diis = tuna.energy("SPE : H F 0.9168 : HF 6-31G : NODIIS")

    assert without_diis == pytest.approx(with_diis, abs = 1e-7)




def test_unrestricted_matches_restricted_for_a_closed_shell(tuna):

    """

    UHF collapses onto the RHF solution for a closed shell molecule at its equilibrium geometry.

    """

    restricted = tuna.energy("SPE : H F 0.9168 : HF 6-31G")
    unrestricted = tuna.energy("SPE : H F 0.9168 : UHF 6-31G")

    assert unrestricted == pytest.approx(restricted, abs = 1e-7)




def test_energy_does_not_depend_on_the_order_of_the_atoms(tuna):

    """

    Writing the diatomic the other way round is the same molecule.

    """

    forwards = tuna.energy("SPE : H F 0.9168 : HF 6-31G")
    backwards = tuna.energy("SPE : F H 0.9168 : HF 6-31G")

    assert backwards == pytest.approx(forwards, abs = 1e-10)




@pytest.mark.parametrize("method", ["CISD", "QCISD", "CCSDT", "CCSDTQ"])
def test_two_electron_methods_are_all_exact(tuna, method):

    """

    With only two electrons there is nothing beyond doubles, so every method that includes singles and
    doubles is the full configuration interaction answer and they must all agree.

    """

    ccsd = tuna.energy("SPE : H H 0.74 : CCSD 6-31G")

    assert tuna.energy(f"SPE : H H 0.74 : {method} 6-31G") == pytest.approx(ccsd, abs = 1e-9)




def test_ghost_atom_lowers_the_atomic_energy(tuna):

    """

    The basis functions of a ghost atom are available to the real atom, so the counterpoise energy of
    an atom in the dimer basis lies below its energy in its own basis.

    """

    isolated = tuna.energy("SPE : H : HF 6-31G")
    with_ghost = tuna.energy("SPE : H XH 0.74 : HF 6-31G")

    assert with_ghost < isolated




@pytest.mark.parametrize("method", ["MP2", "MP3", "CISD", "CCSD"])
def test_correlation_energy_is_negative(tuna, method):

    """

    Correlation lowers the energy below the Hartree-Fock reference.

    """

    result = tuna.run(f"SPE : H F 0.9168 : {method} 6-31G")

    assert result.correlation_energy < 0




@pytest.mark.parametrize("first, second", [

    ("HF", "RHF"),
    ("CEPA", "CEPA0"),
    ("CEPA0", "CEPA[0]"),
    ("CEPA[0]", "CEPA(0)"),
    ("CEPA", "LCCSD"),
    ("MP4", "MP4[SDTQ]"),
    ("MP4[SDTQ]", "MP4(SDTQ)"),
    ("MP4[SDQ]", "MP4(SDQ)"),
    ("MP4[DQ]", "MP4(DQ)"),

    ])
def test_method_aliases_give_the_same_energy(tuna, first, second):

    """

    Two spellings of the same method have to produce the same number.

    """

    assert tuna.energy(f"SPE : H F 0.9168 : {second} 6-31G") == pytest.approx(tuna.energy(f"SPE : H F 0.9168 : {first} 6-31G"), abs = 1e-9)




@pytest.mark.parametrize("first, second", [

    ("CCSD[T]", "CCSD(T)"),
    ("QCISD[T]", "QCISD(T)"),
    ("CCSDT[Q]", "CCSDT(Q)"),

    ])

def test_round_bracket_aliases_give_the_same_energy(tuna, first, second):

    """

    The same alias check for the perturbative methods written with round brackets.
    
    """

    assert tuna.energy(f"SPE : H F 0.9168 : {second} 6-31G") == pytest.approx(tuna.energy(f"SPE : H F 0.9168 : {first} 6-31G"), abs = 1e-9)




@pytest.mark.slow
def test_density_functional_energy_converges_with_the_grid(tuna):

    """

    A meta-GGA is sensitive to the integration grid, so the difference from a well converged reference
    has to shrink as the grid is tightened. The reference is PySCF's level 9 grid.

    """

    reference = -100.3592140323

    errors = [abs(tuna.energy(f"SPE : H F 0.9168 : R2SCAN 6-31G : {grid}") - reference) for grid in ["LOOSEGRID", "MEDIUMGRID", "TIGHTGRID"]]

    assert errors == sorted(errors, reverse = True)
    assert errors[-1] < 5e-5