import pytest


"""

This module tests full configuration interaction (FCI) and its complete active space relatives, CASCI and CASSCF, together
with the FORCEMETHOD, NEL and NORB keywords that control them. All of these arrived in TUNA 0.12.0.

FCI is the exact answer in a given basis, which makes it unusually easy to test without a reference program. Any other method
that is also exact for the system at hand has to agree with it, and so does any active space large enough to hold every
orbital. The tests lean on those identities:

1. FCI against the methods that are exact for one, two, three and four electrons
2. FORCEMETHOD, which stops TUNA swapping FCI for one of those cheaper methods
3. Active spaces that are full, that leave a frozen core inactive, or that hold a single closed shell
4. The ordering that variational methods with nested wavefunction spaces must respect
5. Active spaces that make no sense, which must be refused

"""




@pytest.mark.parametrize("full_ci, exact_method, electrons", [

    ("SPE : H H 0.74 : FCI CC-PVDZ : FORCEMETHOD", "SPE : H H 0.74 : CCSD CC-PVDZ", "two"),
    ("SPE : LI : UFCI 6-31G : FORCEMETHOD", "SPE : LI : UCISDT 6-31G", "three"),
    ("SPE : BE : FCI 6-31G : FORCEMETHOD", "SPE : BE : CCSDTQ 6-31G", "four"),

    ])
def test_full_ci_agrees_with_methods_that_are_exact_for_the_system(tuna, full_ci, exact_method, electrons):

    """

    With N electrons there is nothing beyond N-fold excitations, so CCSD is exact for two electrons, CISDT for three and
    CCSDTQ for four. The determinant-based FCI shares no code with any of them.

    """

    assert tuna.energy(full_ci) == pytest.approx(tuna.energy(exact_method), abs = 1e-8), f"{electrons} electrons"




@pytest.mark.parametrize("full_ci, hartree_fock", [

    ("SPE : H : UFCI CC-PVDZ : FORCEMETHOD", "SPE : H : UHF CC-PVDZ"),
    ("SPE : H H 1.06 : UFCI 6-31G : CH 1 ML 2 FORCEMETHOD", "SPE : H H 1.06 : UHF 6-31G : CH 1 ML 2"),

    ])
def test_full_ci_of_one_electron_is_hartree_fock(tuna, full_ci, hartree_fock):

    """

    A single electron has no other electron to correlate with, so the exact energy is the Hartree-Fock energy.

    """

    assert tuna.energy(full_ci) == pytest.approx(tuna.energy(hartree_fock), abs = 1e-9)




@pytest.mark.parametrize("input_line, simplified_to", [

    ("SPE : H H 0.74 : FCI 6-31G", "CISD"),
    ("SPE : LI : UFCI 6-31G", "CISDT"),
    ("SPE : BE : FCI 6-31G", "CCSDTQ"),

    ])
def test_full_ci_is_simplified_unless_forcemethod_is_used(tuna, input_line, simplified_to):

    """

    TUNA replaces FCI with a cheaper method that is exact for the number of electrons, and FORCEMETHOD stops it. The two
    routes must still give the same energy.

    """

    simplified = tuna.run(input_line)
    forced = tuna.run(f"{input_line} : FORCEMETHOD")

    assert simplified.calculation.method.name.removeprefix("U") == simplified_to
    assert forced.calculation.method.name.removeprefix("U") == "FCI"

    assert forced.energy == pytest.approx(simplified.energy, abs = 1e-8)




@pytest.mark.parametrize("method", ["CASCI", "CASSCF"])
def test_a_full_active_space_is_full_ci(tuna, method):

    """

    Two electrons in all four orbitals of the 6-31G basis leaves nothing outside the active space. Every orbital rotation is
    then redundant, so CASSCF has nothing to optimise either.

    """

    full_ci = tuna.energy("SPE : H H 0.74 : FCI 6-31G : FORCEMETHOD")

    assert tuna.energy(f"SPE : H H 0.74 : {method} 6-31G : NEL 2 NORB 4") == pytest.approx(full_ci, abs = 1e-8)




@pytest.mark.parametrize("full_ci, casci, exact_method", [

    pytest.param("SPE : B H 1.2324 : FCI STO-3G : FREEZECORE FORCEMETHOD", "SPE : B H 1.2324 : CASCI STO-3G : NEL 4 NORB 5",
                 "SPE : B H 1.2324 : CCSDTQ STO-3G : FREEZECORE", id = "restricted"),

    pytest.param("SPE : B : UFCI 6-31G : FREEZECORE FORCEMETHOD", "SPE : B : UCASCI 6-31G : NEL 3 NORB 8",
                 "SPE : B : UCISDT 6-31G : FREEZECORE", id = "unrestricted"),

    # TUNA freezes nothing on beryllium by default, so the core is given explicitly. Spin orbital codes on a restricted reference
    # are meant to double the count, as OMP2 does, but FCI freezes only the alpha 1s spin orbital.

    pytest.param("SPE : BE : FCI 6-31G : FREEZECORE 1 FORCEMETHOD", "SPE : BE : CASCI 6-31G : NEL 2 NORB 8",
                 "SPE : BE : CCSD 6-31G : FREEZECORE 1", id = "restricted, FREEZECORE 1",
                 marks = pytest.mark.xfail(strict = True, reason = "FREEZECORE N freezes N spin orbitals rather than N spatial orbitals in restricted FCI")),

    ])
def test_frozen_core_full_ci_is_casci_with_an_inactive_core(tuna, full_ci, casci, exact_method):

    """

    Freezing the core in FCI keeps the core orbitals doubly occupied and correlates every other electron in every other orbital,
    which is a CASCI calculation with the core left inactive. With four, three or two electrons left to correlate, frozen core
    CCSDTQ, CISDT or CCSD is exact as well.

    """

    reference = tuna.energy(casci)

    assert tuna.energy(full_ci) == pytest.approx(reference, abs = 1e-8)
    assert tuna.energy(exact_method) == pytest.approx(reference, abs = 1e-8)




@pytest.mark.parametrize("method", ["CASCI", "CASSCF"])
def test_one_doubly_occupied_active_orbital_is_hartree_fock(tuna, method):

    """

    Two electrons in one orbital allow only the Hartree-Fock determinant. CASCI then reproduces the reference energy, and
    CASSCF has to stay where the SCF left it, up to the SCF's own convergence.

    """

    hartree_fock = tuna.energy("SPE : H F 0.9168 : HF 6-31G")

    assert tuna.energy(f"SPE : H F 0.9168 : {method} 6-31G : NEL 2 NORB 1") == pytest.approx(hartree_fock, abs = 1e-7)




def test_casci_energy_falls_as_the_active_space_grows(tuna):

    """

    Adding an orbital to the active space only adds determinants, so in the same orbitals the energy cannot go up.

    """

    energies = [tuna.energy(f"SPE : H F 0.9168 : CASCI 6-31G : NEL 2 NORB {n_orbitals}") for n_orbitals in range(1, 7)]

    assert all(larger <= smaller + 1e-10 for smaller, larger in zip(energies, energies[1:]))
    assert energies[-1] < energies[0]




@pytest.mark.parametrize("molecule", ["H H 0.74", "LI H 1.595"])
def test_casscf_lies_between_casci_and_full_ci(tuna, molecule):

    """

    CASSCF starts from the CASCI wavefunction in the SCF orbitals and only lowers the energy by rotating the orbitals, while
    never reaching below the exact answer in the basis. That exact answer comes from whichever method TUNA picks for FCI.

    """

    hartree_fock = tuna.energy(f"SPE : {molecule} : HF 6-31G")
    casci = tuna.energy(f"SPE : {molecule} : CASCI 6-31G : NEL 2 NORB 2")
    casscf = tuna.energy(f"SPE : {molecule} : CASSCF 6-31G : NEL 2 NORB 2")
    full_ci = tuna.energy(f"SPE : {molecule} : FCI 6-31G")

    assert full_ci < casscf < casci < hartree_fock




@pytest.mark.parametrize("input_line, problem", [

    ("SPE : H H 0.74 : CASCI 6-31G",                    "no active space given"),
    ("SPE : H H 0.74 : CASSCF 6-31G : NEL 2",           "no number of active orbitals"),
    ("SPE : H H 0.74 : CASCI 6-31G : NEL 4 NORB 4",     "more active electrons than the molecule has"),
    ("SPE : H H 0.74 : CASCI 6-31G : NEL 2 NORB 0",     "no active orbitals"),
    ("SPE : H F 0.9168 : CASCI 6-31G : NEL 4 NORB 1",   "too many electrons for the active orbitals"),
    ("SPE : H F 0.9168 : CASCI 6-31G : NEL 3 NORB 3",   "an odd number of inactive electrons"),
    ("SPE : H H 0.74 : CASCI STO-3G : NEL 2 NORB 3",    "more active orbitals than the basis has"),

    ])
def test_impossible_active_spaces_are_rejected(tuna, input_line, problem):

    """

    Each of these active spaces is impossible, and should be reported rather than run.

    """

    with pytest.raises(tuna.TunaError):

        tuna.energy(input_line)
