import numpy as np
from numpy import ndarray
from math import comb
import scipy
import TUNA.tuna_ci as ci
from TUNA.tuna_molecule import Molecule
from TUNA.tuna_calc import Calculation
from TUNA.tuna_util import error, log, log_spacer, Integrals, Output, constants, timer


"""

This is the TUNA module for complete active space methods, written for version 0.12.0.

The configuration interaction module is used here but with a defined active space, consisting of active occupied and virtual orbitals. These orbitals
can either be fixed, in CAS configuration interaction, or optimised by the orbital gradient in CAS self-consistent field.

The module contains:

1. Functions shared by CASCI and CASSCF, which set up and solve the active space problem (validate_active_space, calculate_CASCI_ground_state, etc.)
2. Functions for complete active space configuration interaction (run_complete_active_space_configuration_interaction)
3. Functions for complete active space self-consistent field (calculate_orbital_rotation, run_complete_active_space_self_consistent_field)

"""



def validate_active_space(molecule: Molecule, calculation: Calculation) -> tuple:

    """

    Checks the requested active space can be built, and counts the orbitals, electrons and determinants which go into it.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object

    Returns:
        n_active_orbitals (int): Number of active spatial orbitals
        n_active_alpha (int): Number of alpha electrons to distribute in the active space
        n_active_beta (int): Number of beta electrons to distribute in the active space
        n_inactive (int): Number of doubly occupied inactive spatial orbitals
        n_determinants (int): Number of determinants in the active space

    """

    # Freezing core orbitals and choosing an active space are two ways of doing the same thing, so only one is allowed

    if calculation.freeze_core:

        error("Frozen core orbitals cannot be combined with an active space! Use the active space to choose which electrons are correlated.")

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

    # Every way of choosing the occupied active orbitals of each spin, independently of the other spin

    n_determinants = comb(n_active_orbitals, n_active_alpha) * comb(n_active_orbitals, n_active_beta)

    # Storing and diagonalising the Hamiltonian both scale very badly with the number of determinants

    if n_determinants > constants.MAX_N_DETERMINANTS:

        error(f"This active space needs {n_determinants} determinants, which is too many to diagonalise! Try a smaller active space?")


    return n_active_orbitals, n_active_alpha, n_active_beta, n_inactive, n_determinants










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










def build_active_space_determinants(molecule: Molecule, calculation: Calculation, n_active_orbitals: int, n_active_alpha: int, n_active_beta: int, n_inactive: int, n_determinants: int, title: str, silent: bool = False) -> tuple:

    """

    Prints the active space, then builds every determinant which distributes the active electrons amongst the active orbitals.

    The spin orbitals are in the same order as in build_spin_orbital_integrals, with the alpha and beta spin orbitals of each
    molecular orbital next to each other.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        n_active_orbitals (int): Number of active spatial orbitals
        n_active_alpha (int): Number of alpha electrons to distribute in the active space
        n_active_beta (int): Number of beta electrons to distribute in the active space
        n_inactive (int): Number of doubly occupied inactive spatial orbitals
        n_determinants (int): Number of determinants in the active space
        title (str): Heading printed above the active space
        silent (bool, optional): Should anything be printed

    Returns:
        determinants (list): Occupied spin orbitals of each determinant, in ascending order
        reference_determinant (tuple): Occupied spin orbitals of the SCF determinant

    """

    log_spacer(calculation, 1, silent, start = "\n")
    log(title, calculation, 1, silent, colour = "white")
    log_spacer(calculation, 1, silent)

    log(f"  Active electrons:                      {n_active_alpha + n_active_beta:10}", calculation, 1, silent)
    log(f"  Active orbitals:                       {n_active_orbitals:10}", calculation, 1, silent)

    log(f"\n  Doubly occupied orbitals:              {n_inactive:10}", calculation, 1, silent)
    log(f"  Secondary orbitals:                    {molecule.n_basis - n_inactive - n_active_orbitals:10}\n", calculation, 1, silent)

    # Even spin orbitals are alpha and odd spin orbitals are beta, both in the order of the molecular orbitals

    spin_labels = ["a", "b"] * molecule.n_basis

    alpha_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "a"]
    beta_orbitals = [p for p in range(molecule.n_SO) if spin_labels[p] == "b"]

    active_alpha, active_beta, inactive_orbitals = determine_active_space(molecule, spin_labels, n_active_orbitals, n_inactive)

    # The SCF determinant occupies the lowest spin orbitals of each spin, so it is always in the expansion

    reference_determinant = tuple(sorted(alpha_orbitals[:molecule.n_alpha] + beta_orbitals[:molecule.n_beta]))

    log("  Building determinants...                   ", calculation, 1, silent, end = "")

    # Every way of distributing the active electrons of each spin amongst the active orbitals of that spin

    determinants = ci.build_FCI_determinants(active_alpha, active_beta, n_active_alpha, n_active_beta, inactive_orbitals)

    log("[Done]", calculation, 1, silent)

    log(f"\n  Number of determinants:                {n_determinants:10}\n", calculation, 1, silent)

    return determinants, reference_determinant










def build_spin_orbital_integrals(molecular_orbitals_alpha: ndarray, molecular_orbitals_beta: ndarray, integrals: Integrals, calculation: Calculation, silent: bool = False) -> tuple:

    """

    Transforms the core Hamiltonian and two-electron integrals into the spin orbital basis, keeping the order of the molecular orbitals.

    Each molecular orbital keeps its place, with its alpha spin orbital directly before its beta spin orbital, so the inactive and
    active spin orbitals always come first however the orbitals have been rotated.

    Args:
        molecular_orbitals_alpha (array): Alpha molecular orbitals in AO basis
        molecular_orbitals_beta (array): Beta molecular orbitals in AO basis
        integrals (Integrals): Molecular integrals
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis
        H_core_SO (array): Core Hamiltonian in SO basis
        g (array): Antisymmetrised electron repulsion integrals in SO basis

    """

    log("\n Preparing transformation to spin orbital basis...", calculation, 1, silent)

    n_orbitals = molecular_orbitals_alpha.shape[1]

    # The orbital indices stand in for orbital energies, which interleaves the alpha and beta spin orbitals in the original order

    orbital_order = np.append(np.arange(n_orbitals), np.arange(n_orbitals) + 0.5)

    C_spin_block = ci.spin_block_molecular_orbitals(molecular_orbitals_alpha, molecular_orbitals_beta, orbital_order)

    # Spin blocks the integrals exactly as in begin_spin_orbital_calculation, then transforms them

    H_core_spin_block = ci.spin_block_core_Hamiltonian(integrals.H_core)
    ERI_spin_block = np.kron(np.eye(2), np.kron(np.eye(2), integrals.ERI_AO).T)

    H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)

    ERI_SO = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block, C_spin_block, calculation, silent)

    log(" Antisymmetrising two-electron integrals...  ", calculation, 1, silent, end = "")

    g = ci.antisymmetrise_integrals(ERI_SO)

    log("[Done]", calculation, 1, silent)

    return C_spin_block, H_core_SO, g










def calculate_CASCI_ground_state(determinants: list, H_core_SO: ndarray, g: ndarray, C_spin_block: ndarray, molecule: Molecule, calculation: Calculation, silent: bool = False) -> tuple:

    """

    Builds and diagonalises the Hamiltonian in the basis of the active space determinants, and calculates the density of the ground state.

    Args:
        determinants (list): Occupied spin orbitals of each determinant, in ascending order
        H_core_SO (array): Core Hamiltonian in SO basis
        g (array): Antisymmetrised electron repulsion integrals in SO basis
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_electronic (float): Electronic energy of the ground state
        CI_vector (array): Coefficient of each determinant in the ground state
        P_SO (array): One-particle density matrix of the ground state in SO basis
        density_matrices (tuple): Total, alpha and beta density matrices in AO basis

    """

    log("  Building CASCI Hamiltonian...              ", calculation, 1, silent, end = "")

    H = ci.build_FCI_Hamiltonian(determinants, H_core_SO, g, molecule.n_SO, molecule.n_electrons)

    log("[Done]", calculation, 1, silent)

    log("  Diagonalising CASCI Hamiltonian...         ", calculation, 1, silent, end = "")

    # The ground state is the lowest eigenvalue of the Hamiltonian, with the CI coefficients as its eigenvector

    energies, CI_vectors = np.linalg.eigh(H)

    E_electronic = energies[0]
    CI_vector = CI_vectors[:, 0]

    log("[Done]", calculation, 1, silent)

    log("\n  Building CASCI density matrix...           ", calculation, 1, silent, end = "")

    # One-particle density matrix of the ground state, transformed into the AO basis for molecular properties

    P_SO = ci.calculate_FCI_density_matrix(determinants, CI_vector, molecule.n_SO, molecule.n_electrons)

    density_matrices = ci.transform_P_SO_to_AO(P_SO, C_spin_block, molecule.n_SO)

    log("[Done]", calculation, 1, silent)

    return E_electronic, CI_vector, P_SO, density_matrices










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

    # Checks the active space makes sense

    n_active_orbitals, n_active_alpha, n_active_beta, n_inactive, n_determinants = validate_active_space(molecule, calculation)

    # Transforms the integrals into the spin orbital basis of the SCF orbitals, which can be different for each spin

    C_spin_block, H_core_SO, g = build_spin_orbital_integrals(SCF_output.molecular_orbitals_alpha, SCF_output.molecular_orbitals_beta, integrals, calculation, silent)

    timer("Complete active space CI", 0)

    determinants, reference_determinant = build_active_space_determinants(molecule, calculation, n_active_orbitals, n_active_alpha, n_active_beta, n_inactive, n_determinants, "   Complete Active Space Configuration Interaction", silent)

    E_electronic, _, _, density_matrices = calculate_CASCI_ground_state(determinants, H_core_SO, g, C_spin_block, molecule, calculation, silent)

    # The energy of the reference determinant is its diagonal element, so nuclear repulsion cancels in the correlation energy

    E_CASCI = E_electronic - ci.calculate_FCI_matrix_element(reference_determinant, reference_determinant, H_core_SO, g)

    timer("Complete active space CI", 1)

    return E_CASCI, density_matrices










def calculate_orbital_rotation(H_core_SO: ndarray, g: ndarray, P: ndarray, D: ndarray, o: slice, orbital_spaces: ndarray) -> tuple:

    """

    Calculates the orbital gradient and Hessian of the CASSCF energy for fixed CI coefficients, and the orbital rotation for an
    augmented Hessian Newton step, which is shifted to go downhill if the Hessian is not positive definite.

    Args:
        H_core_SO (array): Core Hamiltonian in SO basis
        g (array): Antisymmetrised electron repulsion integrals in SO basis
        P (array): One-particle density matrix in SO basis
        D (array): Two-particle density matrix over the inactive and active spin orbitals
        o (slice): Inactive and active spin orbitals
        orbital_spaces (array): Whether each spatial orbital is inactive (0), active (1) or secondary (2)

    Returns:
        kappa (array): Antisymmetric orbital rotation generator in the spatial orbital basis
        orbital_gradient (array): Orbital gradient in the spatial orbital basis

    """

    n_SO = H_core_SO.shape[0]
    n_orbitals = n_SO // 2

    # Generalised Fock matrix

    F = np.zeros_like(H_core_SO)

    F[:, o] = H_core_SO[:, o] @ P[o, o] + (1 / 2) * np.einsum("prst,stqr->pq", g[:, o, o, o], D, optimize = True)

    # Derivative of the generalised Fock matrix element F_tp with respect to the rotation X_ab, where the orbitals change by 1 + X

    dF_dX = np.einsum("bt,ap->tpab", np.eye(n_SO), F, optimize = True)

    dF_dX[:, o, :, o] += np.einsum("ta,bp->tpab", H_core_SO, P[o, o], optimize = True)
    dF_dX[:, o, :, o] += (1 / 2) * np.einsum("tars,pbrs->tpab", g[:, :, o, o], D, optimize = True)
    dF_dX[:, o, :, o] += np.einsum("tqra,pqrb->tpab", g[:, o, o, :], D, optimize = True)

    # The orbital gradient is 2 (F - F^T)

    dG_dX = 2 * (dF_dX - dF_dX.transpose(1, 0, 2, 3))
    dG_dX = dG_dX - dG_dX.transpose(0, 1, 3, 2)

    # The alpha and beta spin orbitals of a spatial orbital rotate together, so their gradients and Hessians add up

    orbital_gradient_SO = 2 * (F - F.T)

    orbital_gradient = orbital_gradient_SO[0::2, 0::2] + orbital_gradient_SO[1::2, 1::2]

    orbital_hessian = np.einsum("aibicjdj->abcd", dG_dX.reshape(n_orbitals, 2, n_orbitals, 2, n_orbitals, 2, n_orbitals, 2), optimize = True)

    # Rotations within the inactive, active or secondary spaces do not change the CASSCF energy, so only the rest are kept

    non_redundant = np.triu(orbital_spaces[:, None] != orbital_spaces[None, :])

    gradient = orbital_gradient[non_redundant]

    # Away from convergence the derivative of the gradient is not quite symmetric, and its symmetric part is the true Hessian

    hessian = orbital_hessian[non_redundant][:, non_redundant]
    hessian = (hessian + hessian.T) / 2

    # The augmented Hessian gives a Newton step, shifted just enough to go downhill if the Hessian is not positive definite

    augmented_hessian = np.block([[np.zeros((1, 1)), gradient[None, :]], [gradient[:, None], hessian]])

    _, eigenvectors = np.linalg.eigh(augmented_hessian)

    # The lowest eigenvector that involves the gradient is used, as symmetry can leave some directions with no gradient at all

    lowest = np.flatnonzero(np.abs(eigenvectors[0]) > 1e-3)[0]

    step = eigenvectors[1:, lowest] / eigenvectors[0, lowest]

    # Very large rotations are scaled back, as the Hessian is only accurate close to the current orbitals

    step *= constants.CASSCF_MAX_STEP / np.max(np.abs(step), initial = constants.CASSCF_MAX_STEP)

    # Builds the antisymmetric rotation generator from the independent rotations

    kappa = np.zeros((n_orbitals, n_orbitals))

    kappa[non_redundant] = step
    kappa = kappa - kappa.T

    orbital_gradient = np.where(non_redundant | non_redundant.T, orbital_gradient, 0)


    return kappa, orbital_gradient










def run_complete_active_space_self_consistent_field(molecule: Molecule, integrals: Integrals, SCF_output: Output, calculation: Calculation, V_NN: float, silent: bool = False) -> tuple:

    """

    Calculates the complete active space self-consistent field correlation energy, by alternating a CASCI calculation in the current
    orbitals with a Newton step for the orbitals, until the energy and orbital gradient are converged.

    Args:
        molecule (Molecule): Molecule object
        integrals (Integrals): Molecular integrals
        SCF_output (Output): SCF output object
        calculation (Calculation): Calculation object
        V_NN (float): Nuclear-nuclear repulsion energy
        silent (bool, optional): Should anything be printed

    Returns:
        E_CASSCF (float): Complete active space self-consistent field correlation energy
        density_matrices (tuple): Total, alpha and beta density matrices in AO basis

    """

    # Checks the active space makes sense

    n_active_orbitals, n_active_alpha, n_active_beta, n_inactive, n_determinants = validate_active_space(molecule, calculation)

    timer("Complete active space SCF", 0)

    # The same determinants are used throughout, as only the orbitals they are built from change

    determinants, _ = build_active_space_determinants(molecule, calculation, n_active_orbitals, n_active_alpha, n_active_beta, n_inactive, n_determinants, "     Complete Active Space Self-consistent Field", silent)

    # The orbital gradient uses the SCF commutator threshold, as both measure how far the orbitals are from converged

    gradient_convergence = calculation.SCF_conv["commutator"]

    log(f"  Energy convergence tolerance:    {calculation.energy_convergence:16.10f}", calculation, 1, silent)
    log(f"  Gradient convergence tolerance:  {gradient_convergence:16.10f}", calculation, 1, silent)

    # The orbitals are shared by both spins, so an unrestricted reference only provides its alpha orbitals as a starting guess

    molecular_orbitals = SCF_output.molecular_orbitals if calculation.reference == "RHF" else SCF_output.molecular_orbitals_alpha

    # The inactive and active spin orbitals come first, and each spatial orbital is inactive (0), active (1) or secondary (2)

    o = slice(0, 2 * (n_inactive + n_active_orbitals))

    orbital_spaces = np.array([0] * n_inactive + [1] * n_active_orbitals + [2] * (molecule.n_basis - n_inactive - n_active_orbitals))

    log("\n  Starting CASSCF iterations...\n", calculation, 1, silent)

    log_spacer(calculation, 1, silent)
    log("  Step          E                DE          Grad.", calculation, 1, silent)
    log_spacer(calculation, 1, silent)

    E_old = SCF_output.energy

    for iteration in range(1, calculation.correlated_max_iter + 1):

        # Solves the CASCI problem in the current orbitals, exactly as for CASCI but without printing

        C_spin_block, H_core_SO, g = build_spin_orbital_integrals(molecular_orbitals, molecular_orbitals, integrals, calculation, True)

        E_electronic, CI_vector, P_SO, density_matrices = calculate_CASCI_ground_state(determinants, H_core_SO, g, C_spin_block, molecule, calculation, True)

        E_CASSCF_total = E_electronic + V_NN

        # Two-particle density matrix of the ground state, which with the one-particle density matrix gives the orbital gradient

        D_SO = ci.calculate_FCI_two_particle_density_matrix(determinants, CI_vector, o.stop, molecule.n_electrons)

        kappa, orbital_gradient = calculate_orbital_rotation(H_core_SO, g, P_SO, D_SO, o, orbital_spaces)

        delta_E = E_CASSCF_total - E_old
        max_gradient = np.max(np.abs(orbital_gradient))

        log(f"  {iteration:3.0f}  {E_CASSCF_total:16.10f} {delta_E:16.10f}  {max_gradient:9.6f}", calculation, 1, silent)

        E_old = E_CASSCF_total

        # Converged orbitals need no further rotation, so the density matrices from this iteration belong to them

        if abs(delta_E) < calculation.energy_convergence and max_gradient < gradient_convergence:

            break

        elif iteration >= calculation.correlated_max_iter:

            error("Complete active space SCF failed to converge! Try increasing the maximum iterations with \"CORRMAXITER\"?")

        # Takes the Newton step, keeping the orbitals orthonormal

        molecular_orbitals = molecular_orbitals @ scipy.linalg.expm(kappa)

    log_spacer(calculation, 1, silent)

    log(f"\n  CASSCF energy:                   {E_CASSCF_total:16.10f}", calculation, 1, silent)

    # The correlation energy is measured from the SCF energy, as the orbitals have moved away from the reference

    E_CASSCF = E_CASSCF_total - SCF_output.energy

    timer("Complete active space SCF", 1)

    return E_CASSCF, density_matrices