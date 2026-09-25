import numpy as np
from math import comb

import TUNA.tuna_ci as ci
from TUNA.tuna_molecule import Molecule
from TUNA.tuna_calc import Calculation
from TUNA.tuna_util import error, log, log_spacer, Integrals, Output, constants, timer


"""

This is the TUNA module for complete active space methods, written for version 0.12.0.

The configuration interaction module is used here but with a defined active space, consisting of active occupied and virtual orbitals.

The module contains:

1. Functions for complete active space configuration interaction (validate_active_space, run_complete_active_space_configuration_interaction, etc.)

"""



def validate_active_space(molecule: Molecule, calculation: Calculation) -> tuple:

    """

    Checks the requested active space can be built, and counts the orbitals and electrons which go into it.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object

    Returns:
        n_active_orbitals (int): Number of active spatial orbitals
        n_active_alpha (int): Number of alpha electrons to distribute in the active space
        n_active_beta (int): Number of beta electrons to distribute in the active space
        n_inactive (int): Number of doubly occupied inactive spatial orbitals

    """

    n_active_electrons = calculation.n_active_electrons
    n_active_orbitals = calculation.n_active_orbitals

    if n_active_electrons is None or n_active_orbitals is None:

        error("Complete active space calculation requested without an active space! Define one with \"NELEC\" and \"NORB\".")

    if n_active_electrons < 1:

        error("At least one electron must be in the active space!")

    if n_active_orbitals < 1:

        error("At least one orbital must be in the active space!")

    if n_active_electrons > molecule.n_electrons:

        error(f"Active space of {n_active_electrons} electrons requested, but the molecule only has {molecule.n_electrons}!")

    if n_active_electrons > 2 * n_active_orbitals:

        error(f"Active space of {n_active_electrons} electrons cannot fit into {n_active_orbitals} orbitals!")

    # The electrons left outside the active space fill the inactive orbitals in pairs, so there must be an even number of them

    n_inactive_electrons = molecule.n_electrons - n_active_electrons

    if n_inactive_electrons % 2 != 0:

        error(f"An active space of {n_active_electrons} electrons leaves {n_inactive_electrons} electrons outside it, which cannot be paired up!")

    n_inactive = n_inactive_electrons // 2

    # The active electrons of each spin are whatever is left once the inactive orbitals have been doubly occupied

    n_active_alpha = molecule.n_alpha - n_inactive
    n_active_beta = molecule.n_beta - n_inactive

    if n_active_beta < 0:

        error(f"An active space of {n_active_electrons} electrons is not compatible with a multiplicity of {molecule.multiplicity}!")

    if n_active_alpha > n_active_orbitals:

        error(f"A multiplicity of {molecule.multiplicity} needs more than the {n_active_orbitals} active orbitals requested!")

    # There have to be enough orbitals in the basis set to hold the inactive and active spaces

    if n_inactive + n_active_orbitals > molecule.n_basis:

        error(f"An active space of {n_active_orbitals} orbitals on top of {n_inactive} inactive orbitals needs a larger basis set!")


    return n_active_orbitals, n_active_alpha, n_active_beta, n_inactive










def determine_active_space(molecule: Molecule, spin_labels: list, n_active_orbitals: int, n_inactive: int) -> tuple:

    """

    Works out which spin orbitals make up the inactive and active spaces of a complete active space calculation.

    Args:
        molecule (Molecule): Molecule object
        spin_labels (list): Spin of each spin orbital, in order of increasing orbital energy
        n_active_orbitals (int): Number of active spatial orbitals
        n_inactive (int): Number of doubly occupied inactive spatial orbitals

    Returns:
        active_alpha (list): Active alpha spin orbitals
        active_beta (list): Active beta spin orbitals
        inactive_orbitals (tuple): Spin orbitals occupied in every determinant, in ascending order

    """

    # Spin orbitals of each spin, in order of increasing orbital energy

    alpha_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "a"]
    beta_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "b"]

    # Both spins of an inactive orbital are occupied in every determinant

    inactive_orbitals = tuple(sorted(alpha_orbitals[:n_inactive] + beta_orbitals[:n_inactive]))

    # The active orbitals sit directly on top of the inactive ones

    active_alpha = alpha_orbitals[n_inactive:n_inactive + n_active_orbitals]
    active_beta = beta_orbitals[n_inactive:n_inactive + n_active_orbitals]


    return active_alpha, active_beta, inactive_orbitals










def run_complete_active_space_configuration_interaction(molecule: Molecule, integrals: Integrals, SCF_output: Output, calculation: Calculation, silent: bool = False) -> tuple:

    """

    Calculates the complete active space configuration interaction correlation energy, by full diagonalisation of the
    Hamiltonian in the basis of all the determinants which distribute the active electrons amongst the active orbitals.

    Args:
        molecule (Molecule): Molecule object
        integrals (Integrals): Molecular integrals
        SCF_output (Output): SCF output object
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_CASCI (float): Complete active space configuration interaction correlation energy
        density_matrices (tuple): Total, alpha and beta density matrices in AO basis

    """

    # Freezing core orbitals and choosing an active space are two ways of doing the same thing, so only one is allowed

    if calculation.freeze_core:

        error("Frozen core orbitals cannot be combined with an active space! Use the active space to choose which electrons are correlated.")

    # Checks the active space makes sense

    n_active_orbitals, n_active_alpha, n_active_beta, n_inactive = validate_active_space(molecule, calculation)

    n_active_electrons = n_active_alpha + n_active_beta

    # Every way of choosing the occupied active orbitals of each spin, independently of the other spin

    n_determinants = comb(n_active_orbitals, n_active_alpha) * comb(n_active_orbitals, n_active_beta)

    # Storing and diagonalising the Hamiltonian both scale very badly with the number of determinants

    if n_determinants > constants.MAX_N_DETERMINANTS:

        error(f"Active space of needs {n_determinants} determinants, which is too many to diagonalise! Try a smaller active space?")

    # Transforms the two-electron integrals into the antisymmetrised spin orbital basis, in physicists' notation

    g, C_spin_block, _, _, _, _, spin_labels, _, _ = ci.begin_spin_orbital_calculation(molecule, integrals.ERI_AO, SCF_output, calculation, silent = silent)

    timer("Complete active space CI", 0)

    # Transforms the core Hamiltonian into the spin orbital basis

    H_core_SO = ci.transform_matrix_AO_to_SO(ci.spin_block_core_Hamiltonian(integrals.H_core), C_spin_block)

    # Spin orbitals of each spin, in order of increasing orbital energy

    alpha_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "a"]
    beta_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "b"]

    log_spacer(calculation, 1, silent, start = "\n")
    log("   Complete Active Space Configuration Interaction", calculation, 1, silent, colour = "white")
    log_spacer(calculation, 1, silent)

    log(f"  Active electrons:                      {n_active_electrons:10}", calculation, 1, silent)
    log(f"  Active orbitals:                       {n_active_orbitals:10}", calculation, 1, silent)

    log(f"\n  Doubly occupied orbitals:              {n_inactive:10}", calculation, 1, silent)
    log(f"  Secondary orbitals:                    {molecule.n_basis - n_inactive - n_active_orbitals:10}\n", calculation, 1, silent)

    # Picks out which spin orbitals are inactive and which are active, now the orbital energies are known

    active_alpha, active_beta, inactive_orbitals = determine_active_space(molecule, spin_labels, n_active_orbitals, n_inactive)

    # The Hartree-Fock determinant occupies the lowest energy spin orbitals of each spin, so it is always in the expansion

    reference_determinant = tuple(sorted(alpha_orbitals[:molecule.n_alpha] + beta_orbitals[:molecule.n_beta]))

    log("  Building determinants...                   ", calculation, 1, silent, end = "")

    # Every way of distributing the active electrons of each spin amongst the active orbitals of that spin

    determinants = ci.build_FCI_determinants(active_alpha, active_beta, n_active_alpha, n_active_beta, inactive_orbitals)

    log("[Done]", calculation, 1, silent)

    log(f"\n  Number of determinants:                {n_determinants:10}\n", calculation, 1, silent)

    log("  Building CASCI Hamiltonian...              ", calculation, 1, silent, end = "")

    H = ci.build_FCI_Hamiltonian(determinants, H_core_SO, g, molecule.n_SO, molecule.n_electrons)

    log("[Done]", calculation, 1, silent)

    log("  Diagonalising CASCI Hamiltonian...         ", calculation, 1, silent, end = "")

    # The ground state is the lowest eigenvalue of the Hamiltonian, with the CI coefficients as its eigenvector

    energies, CI_vectors = np.linalg.eigh(H)

    log("[Done]", calculation, 1, silent)

    log("\n  Building CASCI density matrix...           ", calculation, 1, silent, end = "")

    # One-particle density matrix of the ground state, transformed into the AO basis for molecular properties

    P_SO = ci.calculate_FCI_density_matrix(determinants, CI_vectors[:, 0], molecule.n_SO, molecule.n_electrons)

    density_matrices = ci.transform_P_SO_to_AO(P_SO, C_spin_block, molecule.n_SO)

    log("[Done]", calculation, 1, silent)

    # The energy of the reference determinant is its diagonal element, so nuclear repulsion cancels in the correlation energy

    E_CASCI = energies[0] - ci.calculate_FCI_matrix_element(reference_determinant, reference_determinant, H_core_SO, g)

    timer("Complete active space CI", 1)

    return E_CASCI, density_matrices