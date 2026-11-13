import numpy as np
import pytest


"""

This module tests the excited states added in TUNA 0.12.0: the higher roots of full and complete active space configuration
interaction, equation-of-motion coupled cluster (EOM-CCSD), TD-DFT with the (meta-)GGA exchange-correlation kernels, the spin of
CIS and TDA-KS states, and the perturbative doubles correction for double-hybrid functionals.

Excited states have few convenient independent references, so most of these tests compare two routes to the same states:

1. FCI, CASCI over the full space and EOM-CCSD are all exact for two electrons, so their spectra must be identical
2. The restricted and unrestricted TD-DFT kernels are separate implementations - one spin-adapted into singlet and triplet
   kernels, one fully spin-resolved - and must give the same states for a closed shell
3. With exact exchange only and no functional, TD-DFT has to reduce to TDHF, and its Tamm-Dancoff form to CIS
4. Spin is checked against what S^2 allows, which needs no reference at all
5. The doubles correction for a double hybrid is checked against the explicit "[D]" keyword and its own scaling

Two routes to the same states agree exactly only in exact arithmetic. A Davidson diagonalisation, a set of coupled cluster
amplitudes and a self-consistent field all stop at their own convergence thresholds, and the integrals underneath them are summed
in whatever order the OpenMP threads finish in, so the last digits move between runs. Comparisons across two such routes are made
to CROSS_METHOD_TOLERANCE rather than to the thresholds themselves, which sit at around 1e-8 and are the noise floor here. A real
disagreement between these routes would be orders of magnitude larger.

"""


TWO_ELECTRON_FULL_CI = "SPE : H H 0.74 : FCI 6-31G : TD"


# Agreement required between two routes to the same excited states, in hartree

CROSS_METHOD_TOLERANCE = 1e-6




def sum_over_degenerate_states(excitation_energies, values, threshold: float = 1e-6):

    """

    Adds up a quantity over each set of states that share an excitation energy.

    Exactly degenerate states can come out of a diagonalisation as any mixture of one another, so only the sum over the
    set is well defined.

    Args:
        excitation_energies (array): Excitation energy of each state
        values (array): Value of the quantity for each state
        threshold (float, optional): Largest gap between two states counted as degenerate

    Returns:
        sums (array): Sum of the quantity over each degenerate set, in order of increasing energy

    """

    order = np.argsort(excitation_energies)

    boundaries = np.flatnonzero(np.diff(excitation_energies[order]) > threshold) + 1

    return np.array([np.sum(group) for group in np.split(values[order], boundaries)])




@pytest.mark.parametrize("input_line", [

    "SPE : H H 0.74 : EOM-CCSD 6-31G",
    "SPE : H H 0.74 : CCSD 6-31G : TD",
    "SPE : H H 0.74 : CASCI 6-31G : NEL 2 NORB 4 TD",

    ])
def test_two_electron_spectra_agree_with_full_ci(tuna, input_line):

    """

    EOM-CCSD spans every singly and doubly excited determinant, which for two electrons is all of them, and an active space of
    every orbital is the full space. Each must reproduce the full CI spectrum state by state, spin included.

    """

    full_ci = tuna.excited_states(TWO_ELECTRON_FULL_CI)
    other = tuna.excited_states(input_line)

    n_states = len(other.excitation_energies)

    assert n_states >= 10
    assert other.excitation_energies == pytest.approx(full_ci.excitation_energies[:n_states], abs = CROSS_METHOD_TOLERANCE)
    assert list(other.state_types) == list(full_ci.state_types[:n_states])




@pytest.mark.parametrize("keywords", ["", "ROOT 2", "NOTRIPLETS", "NOSINGLETS ROOT 2"])
def test_root_and_spin_selection_pick_the_same_state(tuna, keywords):

    """

    ROOT counts upwards through whichever spins have not been left out, and the full CI and EOM-CCSD codes have to agree on
    which state that is.

    """

    full_ci = tuna.energy(f"{TWO_ELECTRON_FULL_CI} {keywords}")
    eom_ccsd = tuna.energy("SPE : H H 0.74 : EOM-CCSD 6-31G" + (f" : {keywords}" if keywords else ""))

    assert eom_ccsd == pytest.approx(full_ci, abs = CROSS_METHOD_TOLERANCE)




@pytest.mark.parametrize("input_line", [TWO_ELECTRON_FULL_CI, "SPE : H H 0.74 : EOM-CCSD 6-31G"])
def test_spin_forbidden_transitions_have_no_intensity(tuna, input_line):

    """

    The dipole operator cannot change the spin, so every transition from the singlet ground state to a triplet is dark.

    """

    states = tuna.excited_states(input_line)

    triplets = states.state_types == "triplet"

    assert np.any(triplets) and np.any(~triplets)
    assert np.all(states.oscillator_strengths[triplets] == 0)
    assert np.any(states.oscillator_strengths[~triplets] > 1e-3)




@pytest.mark.parametrize("functional, keywords", [

    pytest.param("PBE", "TD TDA", id = "GGA, Tamm-Dancoff"),
    pytest.param("B3LYP", "TD", id = "hybrid GGA, full TD"),
    pytest.param("SVWN5", "TD TDA", id = "LDA, Tamm-Dancoff", marks = pytest.mark.slow),
    pytest.param("PBE", "TD", id = "GGA, full TD", marks = pytest.mark.slow),
    pytest.param("BLYP", "TD", id = "GGA with LYP correlation, full TD", marks = pytest.mark.slow),
    pytest.param("TPSS", "TD TDA", id = "meta-GGA, Tamm-Dancoff", marks = pytest.mark.slow),
    pytest.param("R2SCAN", "TD TDA", id = "SCAN-family meta-GGA, Tamm-Dancoff", marks = pytest.mark.slow),

    ])
def test_restricted_and_unrestricted_kernels_give_the_same_spectrum(tuna, functional, keywords):

    """

    For a closed shell, the spin-conserving excitations of an unrestricted reference are exactly the singlets and the triplets
    of a restricted one. The two kernels share no code beyond the functional itself, so every excitation energy is compared.
    Excitation energies are used rather than total energies, as the two SCF solutions agree less tightly than the states do.

    """

    restricted = tuna.excited_states(f"SPE : H F 0.9168 : {functional} 6-31G : {keywords}")
    unrestricted = tuna.excited_states(f"SPE : H F 0.9168 : U{functional} 6-31G : {keywords}")

    assert set(restricted.state_types) == {"singlet", "triplet"}
    assert len(unrestricted.excitation_energies) == len(restricted.excitation_energies)

    assert np.sort(unrestricted.excitation_energies) == pytest.approx(np.sort(restricted.excitation_energies), abs = CROSS_METHOD_TOLERANCE)




@pytest.mark.parametrize("keywords, wavefunction_method", [("TD", "TDHF"), ("TD TDA", "CIS")])
def test_td_dft_without_a_functional_is_hartree_fock_response(tuna, keywords, wavefunction_method):

    """

    With all of the exchange exact and the functional switched off, the exchange-correlation kernel has to vanish and leave
    TDHF, or CIS in the Tamm-Dancoff approximation.

    """

    density_functional = tuna.excited_states(f"SPE : H F 0.9168 : B3LYP 6-31G : HFX 1 NOX NOC {keywords}")
    wavefunction = tuna.excited_states(f"SPE : H F 0.9168 : {wavefunction_method} 6-31G")

    assert density_functional.energy == pytest.approx(wavefunction.energy, abs = 1e-7)
    assert density_functional.excitation_energies == pytest.approx(wavefunction.excitation_energies, abs = 1e-7)




@pytest.mark.parametrize("input_line", [

    pytest.param("SPE : H F 0.9168 : UCIS 6-31G", id = "CIS"),
    pytest.param("SPE : H H 0.74 : EOM-CCSD 6-31G", id = "EOM-CCSD"),
    pytest.param("SPE : H F 0.9168 : UPBE 6-31G : TD TDA", id = "TDA-KS", marks = pytest.mark.slow),

    ])
def test_closed_shell_excited_states_are_singlets_and_triplets(tuna, input_line):

    """

    Excitations that keep the spin projection of a closed shell can only make singlets (S^2 = 0) and triplets (S^2 = 2). Exactly
    degenerate states can mix - the singlet and triplet Sigma-minus states from pi to pi* excitations are degenerate in CIS - so
    S^2 is summed over each degenerate set, which must then give twice the number of triplets in it.

    """

    states = tuna.excited_states(input_line)

    assert states.spin_squared is not None

    assert np.all(states.spin_squared > -1e-8) and np.all(states.spin_squared < 2 + 1e-8)

    triplets_in_each_set = sum_over_degenerate_states(states.excitation_energies, states.spin_squared) / 2

    assert triplets_in_each_set == pytest.approx(np.round(triplets_in_each_set), abs = 1e-6)




def test_cis_from_a_closed_shell_has_as_many_triplets_as_singlets(tuna):

    """

    Each occupied-virtual pair of spatial orbitals gives exactly one singlet and one triplet, so the average S^2 over every
    state of an unrestricted CIS calculation on a closed shell is exactly one.

    """

    states = tuna.excited_states("SPE : H F 0.9168 : UCIS 6-31G")

    assert np.mean(states.spin_squared) == pytest.approx(1.0, abs = 1e-8)




@pytest.mark.parametrize("input_line", [

    pytest.param("SPE : O H 0.97 : UCIS 6-31G : ML 2", id = "CIS, radical"),
    pytest.param("SPE : LI : UCIS 6-31G", id = "CIS, atom"),
    pytest.param("SPE : O H 0.97 : UPBE 6-31G : ML 2 TD TDA", id = "TDA-KS, radical", marks = pytest.mark.slow),

    ])
def test_doublet_excited_states_have_at_least_the_spin_of_a_doublet(tuna, input_line):

    """

    S^2 = S_z (S_z + 1) + S_- S_+, and S_- S_+ cannot be negative, so no state with S_z = 1/2 can have S^2 below 3/4. Spin
    contamination can only push it up.

    """

    states = tuna.excited_states(input_line)

    assert np.all(states.spin_squared > 0.75 - 1e-8)




@pytest.fixture(scope = "module")
def double_hybrid_excited_state_energies(tuna):

    """

    Runs the double-hybrid excited state calculations once for the tests that compare them.

    """

    base = "SPE : H F 0.9168 : B2PLYP 6-31G : TD TDA"

    return {keywords: tuna.energy(f"{base} {keywords}") for keywords in ["", "[D]", "MPC 0", "MPC 0.27", "MPC 0.54"]}




def test_double_hybrid_excited_states_include_the_doubles_correction(double_hybrid_excited_state_energies):

    """

    A double hybrid has to correct its TD-DFT states with the perturbative doubles, just as its ground state gets the MP2
    correlation, without needing "[D]" to ask for it.

    """

    energies = double_hybrid_excited_state_energies

    assert energies[""] == pytest.approx(energies["[D]"], abs = 1e-9)

    # Without any perturbative correlation there is nothing to correct, so the difference is the (negative) correction itself

    assert energies[""] < energies["MPC 0"] - 1e-3




def test_doubles_correction_is_scaled_by_the_perturbative_correlation(double_hybrid_excited_state_energies):

    """

    The amount of MP2 correlation in the functional does not change the Kohn-Sham orbitals, so the correction to the excited
    state is exactly proportional to it.

    """

    energies = double_hybrid_excited_state_energies

    single_correction = energies["MPC 0.27"] - energies["MPC 0"]
    double_correction = energies["MPC 0.54"] - energies["MPC 0"]

    assert energies["MPC 0.27"] == pytest.approx(energies[""], abs = 1e-9)
    assert double_correction == pytest.approx(2 * single_correction, abs = 1e-8)




@pytest.mark.parametrize("input_line", [

    "SPE : H : UHF CC-PVDZ : TD",
    "SPE : H H 1.06 : UCIS 6-31G : CH 1 ML 2",

    ])
def test_excited_states_of_one_electron_are_refused(tuna, input_line):

    """

    Asking for excited states of a one-electron system used to crash, and should now be reported as an error.

    """

    with pytest.raises(tuna.TunaError):

        tuna.energy(input_line)




def test_eom_excited_states_need_a_ccsd_ground_state(tuna):

    """

    EOM-CCSD is built on the CCSD ground state, so excited states from any other coupled cluster method are refused.

    """

    with pytest.raises(tuna.TunaError):

        tuna.energy("SPE : H F 0.9168 : QCISD 6-31G : TD")
