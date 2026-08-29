from typing import NamedTuple

import pytest


"""

This module holds the reference total energies for TUNA's electronic structure methods.

Each entry in one of the two tables below becomes one test, so adding coverage for a method means
adding a single line. The tables are:

1. REFERENCE_ENERGIES - energies checked against an independent program. Nothing goes in here without
   a reference from somewhere other than TUNA.

2. REGRESSION_ENERGIES - energies for methods with no convenient independent reference. These were
   produced by TUNA itself and only prove that a number has not moved. They are not evidence that the
   number is right, and any of them should be promoted to the table above once it has been checked
   against another program or the literature.

The tolerance on each entry is deliberately per-case rather than global. Most wavefunction methods
agree with PySCF to better than 1e-7 hartree, and the limit there is the SCF and amplitude convergence
thresholds. Density functional entries are looser because the integration grid, not the functional,
dominates the difference.

"""


# Default agreement required of a wavefunction method, in hartree

DEFAULT_TOLERANCE = 1e-6




class Case(NamedTuple):

    """

    One reference calculation.

    """

    # The TUNA input line to run

    line: str

    # Expected total energy in hartree

    energy: float

    # Absolute agreement required, in hartree

    tolerance: float = DEFAULT_TOLERANCE

    # Where the reference number came from

    source: str = ""

    # Should this case be skipped by "pytest -m 'not slow'"

    slow: bool = False




REFERENCE_ENERGIES = [

    # Restricted Hartree-Fock, references from PySCF 2.14.0

    Case("SPE : H H 0.74 : HF STO-3G",       -1.1167593074,   source = "PySCF RHF"),
    Case("SPE : H H 0.74 : HF 6-31G",        -1.1267553172,   source = "PySCF RHF"),
    Case("SPE : H H 0.74 : HF CC-PVDZ",      -1.1287000936,   source = "PySCF RHF"),
    Case("SPE : H F 0.9168 : HF 6-31G",     -99.9834071596,   source = "PySCF RHF"),
    Case("SPE : H F 0.9168 : HF CC-PVDZ",  -100.0194187031,   source = "PySCF RHF"),
    Case("SPE : LI H 1.595 : HF 6-31G",      -7.9792689484,   source = "PySCF RHF"),
    Case("SPE : B H 1.2324 : HF CC-PVDZ",   -25.1253318293,   source = "PySCF RHF"),

    # Atoms, and unrestricted Hartree-Fock

    Case("SPE : HE : HF CC-PVDZ",            -2.8551604772,   source = "PySCF RHF"),
    Case("SPE : BE : HF 6-31G",             -14.5667640335,   source = "PySCF RHF"),
    Case("SPE : H : UHF CC-PVDZ",            -0.4992784034,   source = "PySCF UHF"),
    Case("SPE : LI : UHF 6-31G",             -7.4312358111,   source = "PySCF UHF"),
    Case("SPE : O H 0.97 : UHF 6-31G : ML 2",       -75.3631682496, source = "PySCF UHF, doublet"),
    Case("SPE : H H 1.06 : UHF 6-31G : CH 1 ML 2",   -0.5840274596, source = "PySCF UHF, cation doublet"),

    # Moller-Plesset perturbation theory, configuration interaction and coupled cluster

    Case("SPE : H H 0.74 : MP2 6-31G",       -1.1441365741,   source = "PySCF MP2"),
    Case("SPE : H H 0.74 : CCSD 6-31G",      -1.1516726783,   source = "PySCF CCSD, exact for two electrons"),
    Case("SPE : H F 0.9168 : MP2 6-31G",   -100.1120906941,   source = "PySCF MP2"),
    Case("SPE : H F 0.9168 : CISD 6-31G",  -100.1103930551,   source = "PySCF CISD"),
    Case("SPE : H F 0.9168 : CCSD 6-31G",  -100.1146440660,   source = "PySCF CCSD"),
    Case("SPE : H F 0.9168 : CCSD[T] 6-31G",   -100.1152709313, source = "PySCF CCSD(T)"),
    Case("SPE : H F 0.9168 : MP2 CC-PVDZ",     -100.2231920685, source = "PySCF MP2"),
    Case("SPE : H F 0.9168 : CISD CC-PVDZ",    -100.2216772734, source = "PySCF CISD"),
    Case("SPE : H F 0.9168 : CCSD CC-PVDZ",    -100.2281541583, source = "PySCF CCSD"),
    Case("SPE : H F 0.9168 : CCSD[T] CC-PVDZ", -100.2300901337, source = "PySCF CCSD(T)"),

    # Density functional theory. The PySCF references use its level 9 grid; the difference from TUNA
    # is dominated by the grid rather than by the functional, which is why the tolerances vary so much.

    Case("SPE : H F 0.9168 : HFS 6-31G",    -99.0418810736, 1e-6, "PySCF RKS lda,"),
    Case("SPE : H F 0.9168 : SVWN3 6-31G",  -99.9458366080, 1e-6, "PySCF RKS lda,vwn3"),
    Case("SPE : H F 0.9168 : SVWN5 6-31G",  -99.7471409414, 1e-6, "PySCF RKS lda,vwn5"),
    Case("SPE : H F 0.9168 : BLYP 6-31G",  -100.3869743396, 1e-6, "PySCF RKS b88,lyp"),
    Case("SPE : H F 0.9168 : B3LYP 6-31G", -100.3647473753, 1e-6, "PySCF RKS b3lyp5, the VWN5 variant"),
    Case("SPE : H F 0.9168 : PBE 6-31G",   -100.3021864109, 5e-5, "PySCF RKS pbe,pbe"),
    Case("SPE : H F 0.9168 : PBE0 6-31G",  -100.3071104151, 5e-5, "PySCF RKS pbe0"),
    Case("SPE : H F 0.9168 : TPSS 6-31G",  -100.4051109671, 5e-5, "PySCF RKS tpss,tpss"),
    Case("SPE : H F 0.9168 : R2SCAN 6-31G : TIGHTGRID", -100.3592140323, 5e-5, "PySCF RKS r2scan,r2scan", slow = True),

    # Excited states. The reference is the ground state energy plus the first PySCF excitation energy.
    # TUNA reports the lowest root of either spin unless NOTRIPLETS asks for singlets only.

    Case("SPE : H F 0.9168 : CIS 6-31G",                 -99.5784039009, source = "PySCF RHF plus TDA triplet root 1"),
    Case("SPE : H F 0.9168 : CIS 6-31G : NOTRIPLETS",    -99.5458762503, source = "PySCF RHF plus TDA singlet root 1"),
    Case("SPE : H F 0.9168 : TDHF 6-31G",                -99.5830159585, source = "PySCF RHF plus RPA triplet root 1"),
    Case("SPE : H F 0.9168 : TDHF 6-31G : NOTRIPLETS",   -99.5483697328, source = "PySCF RHF plus RPA singlet root 1"),

    ]


REGRESSION_ENERGIES = [

    # None of these have been checked against another program. They are here to catch a number moving,
    # not to show that it is correct.

    Case("SPE : H F 0.9168 : MP3 CC-PVDZ",      -100.2258915241, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : MP4[SDTQ] 6-31G",  -100.1150789571, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : CEPA 6-31G",       -100.1145154328, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : QCISD[T] 6-31G",   -100.1153292557, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : CCSDT 6-31G",      -100.1153348254, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : CCSDT[Q] 6-31G",   -100.1156954865, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : CIS(D) 6-31G",      -99.6140477484, 1e-7, "TUNA 0.12.0"),
    Case("SPE : H F 0.9168 : B2PLYP 6-31G : TIGHTGRID", -100.3079373546, 1e-6, "TUNA 0.12.0", slow = True),

    ]




def build_parameters(cases: list) -> list:

    """

    Turns a list of cases into pytest parameters, labelled by input line and marked if slow.

    Args:
        cases (list): List of Case objects

    Returns:
        parameters (list): List of pytest parameters

    """

    return [pytest.param(case, id = case.line, marks = [pytest.mark.slow] if case.slow else []) for case in cases]




@pytest.mark.parametrize("case", build_parameters(REFERENCE_ENERGIES))
def test_energy_matches_independent_reference(tuna, case):

    """

    Checks a TUNA total energy against a value from another program.

    """

    energy = tuna.energy(case.line)

    assert energy == pytest.approx(case.energy, abs = case.tolerance), f"reference from {case.source}"




@pytest.mark.parametrize("case", build_parameters(REGRESSION_ENERGIES))
def test_energy_has_not_changed(tuna, case):

    """

    Checks a TUNA total energy against the value TUNA itself gave when the test was written.

    """

    energy = tuna.energy(case.line)

    assert energy == pytest.approx(case.energy, abs = case.tolerance), f"baseline from {case.source}"
