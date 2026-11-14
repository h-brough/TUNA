import numpy as np
import pytest


"""

This module tests the two methods built on the direct random phase approximation that arrived in TUNA 0.12.0: the RPA ground
state correlation energy, and one-shot G0W0 quasiparticle energies, each on a Hartree-Fock or Kohn-Sham reference.

TUNA finds the RPA correlation energy with the plasmon formula, from the eigenvalues of the RPA problem. The first test finds the
same energy a completely different way, by solving the ring coupled cluster doubles equations, which Scuseria, Henderson and
Sorensen (J. Chem. Phys. 129, 231101, 2008) showed are equivalent. Everything else compares the spin-adapted restricted code with
the spin orbital unrestricted code, or checks which way the corrections have to go.

"""


# Quasiparticle energies of hydrogen fluoride from TUNA 0.12.0 itself. These only prove that nothing has moved, and should be
# replaced with values from another program once they have been checked.

REGRESSION_QUASIPARTICLE_ENERGIES = {

    "SPE : H F 0.9168 : HF 6-31G": (-0.5653693012, 0.2066399301),
    "SPE : H F 0.9168 : PBE 6-31G": (-0.5286067231, 0.2161256079),

    }




def exact_exchange_energy(result) -> float:

    """

    Evaluates the Hartree-Fock energy expression on the converged density, whichever orbitals it came from.

    Args:
        result (EnergyResult): Result of a closed shell energy evaluation

    Returns:
        energy (float): Total energy with exact exchange and no correlation, in hartree

    """

    P = result.density
    ERI = result.integrals.ERI_AO

    J = np.einsum("pqrs,rs->pq", ERI, P, optimize = True)
    K = np.einsum("prqs,rs->pq", ERI, P, optimize = True)

    nuclear_repulsion = result.molecule.charges[0] * result.molecule.charges[1] / result.molecule.bond_length

    return float(np.einsum("pq,pq->", P, result.integrals.H_core + J / 2 - K / 4) + nuclear_repulsion)




def ring_coupled_cluster_doubles_energy(result) -> float:

    """

    Solves the ring coupled cluster doubles equations for a closed shell reference, by simple iteration.

    With the spin-adapted matrices A = D + 2K and B = 2K, where K is (ia|jb) and D holds the orbital energy differences, the
    amplitudes solve B + AT + TA + TBT = 0, and the correlation energy is Tr(BT) / 2.

    Args:
        result (EnergyResult): Result of a closed shell energy evaluation

    Returns:
        correlation_energy (float): Direct RPA correlation energy in hartree

    """

    n_occupied = result.molecule.n_doubly_occ

    orbitals = result.scf_output.molecular_orbitals
    orbital_energies = result.scf_output.epsilons

    occupied, virtual = orbitals[:, :n_occupied], orbitals[:, n_occupied:]

    K = np.einsum("pqrs,pi,qa,rj,sb->iajb", result.integrals.ERI_AO, occupied, virtual, occupied, virtual, optimize = True)
    K = K.reshape(n_occupied * virtual.shape[1], -1)

    D = (orbital_energies[n_occupied:][None, :] - orbital_energies[:n_occupied][:, None]).ravel()

    denominators = D[:, None] + D[None, :]

    T = np.zeros_like(K)

    for _ in range(1000):

        T_new = -2 * (K + K @ T + T @ K + T @ K @ T) / denominators

        converged = np.max(np.abs(T_new - T)) < 1e-12

        T = T_new

        if converged:

            break

    else:

        raise RuntimeError("The ring coupled cluster doubles equations did not converge")

    return float(np.trace(K @ T))




@pytest.mark.parametrize("reference", ["HF", "PBE"])
def test_rpa_matches_ring_coupled_cluster_doubles(tuna, reference):

    """

    The RPA total energy is the exact exchange energy of the reference density plus the direct RPA correlation energy, which is
    recalculated here from TUNA's integrals and orbitals by a different algorithm from the one TUNA uses.

    """

    result = tuna.run(f"SPE : H F 0.9168 : {reference} 6-31G : RPA")

    correlation_energy = ring_coupled_cluster_doubles_energy(result)

    assert correlation_energy < 0
    assert result.energy == pytest.approx(exact_exchange_energy(result) + correlation_energy, abs = 1e-7)




def test_rpa_method_and_keyword_are_the_same(tuna):

    """

    "RPA" as a method means RPA on a Hartree-Fock reference, which is what the keyword does to a Hartree-Fock calculation.

    """

    assert tuna.energy("SPE : H F 0.9168 : RPA 6-31G") == pytest.approx(tuna.energy("SPE : H F 0.9168 : HF 6-31G : RPA"), abs = 1e-12)




@pytest.mark.parametrize("reference", ["HF", "PBE"])
def test_restricted_and_unrestricted_rpa_agree_for_a_closed_shell(tuna, reference):

    """

    The unrestricted code builds the RPA matrices in spin orbitals rather than spin-adapting them, and has to give the same
    energy. Each is measured from its own SCF energy, so that the slightly different convergence of the two SCF solutions
    cancels.

    """

    restricted = tuna.run(f"SPE : H F 0.9168 : {reference} 6-31G : RPA")
    unrestricted = tuna.run(f"SPE : H F 0.9168 : U{reference} 6-31G : RPA")

    assert unrestricted.energy - unrestricted.scf_energy == pytest.approx(restricted.energy - restricted.scf_energy, abs = 1e-6)




@pytest.mark.parametrize("reference", [

    pytest.param("HF"),
    pytest.param("PBE", marks = pytest.mark.slow),

    ])
def test_restricted_and_unrestricted_g0w0_agree_for_a_closed_shell(tuna, reference):

    """

    Every spatial orbital of a closed shell is a degenerate pair of spin orbitals, which the unrestricted code must correct to the
    same quasiparticle energy as the restricted code does the spatial orbital.

    """

    restricted = tuna.quasiparticle_energies(f"SPE : H F 0.9168 : {reference} 6-31G")
    unrestricted = tuna.quasiparticle_energies(f"SPE : H F 0.9168 : U{reference} 6-31G")

    assert unrestricted.homo == pytest.approx(restricted.homo, abs = 1e-6)
    assert unrestricted.lumo == pytest.approx(restricted.lumo, abs = 1e-6)

    assert np.sort(unrestricted.quasiparticle_energies) == pytest.approx(np.sort(np.repeat(restricted.quasiparticle_energies, 2)), abs = 1e-6)




def test_frozen_core_orbitals_keep_their_orbital_energies(tuna):

    """

    A frozen orbital is left out of the self-energy altogether, so it has no quasiparticle correction, while every other
    orbital still gets one.

    """

    quasiparticles = tuna.quasiparticle_energies("SPE : H F 0.9168 : HF 6-31G : FREEZECORE")

    assert quasiparticles.quasiparticle_energies[0] == quasiparticles.orbital_energies[0]
    assert np.all(np.abs(quasiparticles.quasiparticle_energies[1:] - quasiparticles.orbital_energies[1:]) > 1e-6)




def test_g0w0_narrows_the_hartree_fock_gap_and_widens_the_kohn_sham_gap(tuna):

    """

    Hartree-Fock orbital energies have no correlation, which binds the occupied orbitals too tightly, while semi-local Kohn-Sham
    orbital energies suffer from self-interaction, which binds them far too weakly. G0W0 has to correct each towards the other,
    so a sign error in the self-energy would reverse at least one of these.

    """

    hartree_fock = tuna.quasiparticle_energies("SPE : H F 0.9168 : HF 6-31G")
    kohn_sham = tuna.quasiparticle_energies("SPE : H F 0.9168 : PBE 6-31G")

    def orbital_gap(result):

        return result.orbital_energies[result.n_occupied] - result.orbital_energies[result.n_occupied - 1]

    assert hartree_fock.homo > hartree_fock.orbital_energies[hartree_fock.n_occupied - 1]
    assert hartree_fock.lumo - hartree_fock.homo < orbital_gap(hartree_fock)

    assert kohn_sham.homo < kohn_sham.orbital_energies[kohn_sham.n_occupied - 1]
    assert kohn_sham.lumo - kohn_sham.homo > orbital_gap(kohn_sham)




@pytest.mark.parametrize("input_line", list(REGRESSION_QUASIPARTICLE_ENERGIES))
def test_quasiparticle_energies_have_not_changed(tuna, input_line):

    """

    Checks the HOMO and LUMO quasiparticle energies against the values TUNA gave when the test was written.

    """

    homo, lumo = REGRESSION_QUASIPARTICLE_ENERGIES[input_line]

    quasiparticles = tuna.quasiparticle_energies(input_line)

    assert quasiparticles.homo == pytest.approx(homo, abs = 1e-6)
    assert quasiparticles.lumo == pytest.approx(lumo, abs = 1e-6)
