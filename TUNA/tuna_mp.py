import numpy as np
import sys
import scipy
import tuna_scf as scf
import tuna_ci as ci
from tuna_util import *




def doubly_permute(array, idx_pair_1, idx_pair_2):

    """

    Incorporates symmetric permutation into an array. This is the definition of P from Molecular Electronic-Structure Theory.

    Args:
        array (array): Array wanted to be permuted
        idx_pair_1 (tuple): First index pair
        idx_pair_2 (tuple): Second index pair

    Returns:
        permuted_array (array): Symmetrically doubly permuted array

    """

    i, j = idx_pair_1
    a, b = idx_pair_2

    permuted_array = array + array.swapaxes(i, j).swapaxes(a, b)

    return permuted_array








def six_index_permute(array):

    """
    
    Incorporates symmetric permutation into an array. Definition from Molecular Electronic-Structure Theory.

    Args:
        array (array): Array wanted to be permuted
    
    Returns:
        permuted_array (array): Symmetrically permuted array

    """

    permuted_array = array + array.transpose(0,2,1,3,5,4) + array.transpose(1,0,2,4,3,5) + array.transpose(1,2,0,4,5,3) + array.transpose(2,0,1,5,3,4) + array.transpose(2,1,0,5,4,3)

    return permuted_array









def calculate_restricted_t_amplitude_energy(t_ijab, g):

    """

    Calculates the restricted MP2 energy.

    Args:
        t_ijab (array): Amplitude with shape ijab
        g (array): Electron repulsion integrals in spatial orbital basis

    Returns:
        E_MP2 (float): Contribution to MP2 energy

    """

    E_MP2 = np.einsum("ijab,ijab->", t_ijab, 2 * g - g.transpose(0, 1, 3, 2), optimize=True)

    return E_MP2










def calculate_t_amplitude_energy(t_ijab, ERI_SO):

    """

    Calculates the MP2 energy.

    Args:
        t_ijab (array): Amplitude with shape ijab
        ERI_SO (array): Electron repulsion integrals (optionally antisymmetrised) in SO basis

    Returns:
        E_MP2 (float): Contribution to MP2 energy

    """

    E_MP2 = (1 / 4) * np.einsum("ijab,ijab->", t_ijab, ERI_SO, optimize=True)

    return E_MP2








def build_t_amplitude_density_contribution(n_atomic_orbitals, t_ijab, o, v):

    """

    Calculates the MP2 energy.

    Args:
        n_atomic_orbitals (int): Number of atomic orbitals
        t_ijab (array): Amplitude with shape ijab
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice

    Returns:
        P_MP2 (array): Contribution to MP2 density matrix

    """

    P_MP2 = np.zeros((n_atomic_orbitals, n_atomic_orbitals))

    # Occupied-occupied and virtual-virtual contributions to MP2 density matrix
    P_MP2[v, v] +=  (1 / 2) * np.einsum('ijac,ijbc->ab', t_ijab, t_ijab, optimize=True)
    P_MP2[o, o] += - (1 / 2) * np.einsum('jkab,ikab->ij', t_ijab, t_ijab, optimize=True)

    return P_MP2








def spin_component_scale_MP2_energy(E_MP2_SS, E_MP2_OS, same_spin_scaling, opposite_spin_scaling, calculation, silent=False):
    
    """

    Scales the different spin components of the MP2 energy.

    Args:
        E_MP2_SS (float): Same-spin contribution to MP2 energy
        E_MP2_OS (float): Opposite-spin contribution to MP2 energy
        same_spin_scaling (float): Scaling of same-spin contribution
        opposite_spin_scaling (float): Scaling of opposite-spin contribution
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2_SS_scaled (float): Same-spin scaled contribution to MP2 energy
        E_MP2_OS_scaled (float): Opposite-spin scaled contribution to MP2 energy

    """

    # Scaling energy components
    E_MP2_SS_scaled = same_spin_scaling * E_MP2_SS 
    E_MP2_OS_scaled = opposite_spin_scaling * E_MP2_OS 

    log(f"  Same-spin scaling: {same_spin_scaling:.3f}", calculation, 1, silent=silent)
    log(f"  Opposite-spin scaling: {opposite_spin_scaling:.3f}\n", calculation, 1, silent=silent)
    

    return E_MP2_SS_scaled, E_MP2_OS_scaled
    








def calculate_natural_orbitals(P, X, calculation, silent=False):

    """

    Calculates the natural orbitals and occupancies of a density matrix.

    Args:
        P (array): Density matrix in AO basis
        X (array): Fock transformation matrix
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        natural_orbital_occupancies (array): Occupancies for natural orbitals
        natural_orbitals (array): Natural orbitals in AO basis

    """

    # Transforms density matrix to orthogonal basis
    P_orthogonal = np.linalg.inv(X) @ P @ np.linalg.inv(X)

    natural_orbital_occupancies, natural_orbitals = np.linalg.eigh(P_orthogonal)

    natural_orbital_occupancies = np.sort(natural_orbital_occupancies)[::-1]
    sum_of_occupancies = np.sum(natural_orbital_occupancies)

    natural_orbitals = natural_orbitals[:, natural_orbital_occupancies.argsort()] 

    # This ensures consistent spacing across UHF and correlated calculations
    if calculation.method != "UHF": log("", calculation, 2, silent=silent)

    log("  Natural orbital occupancies: \n", calculation, 2, silent=silent)

    # Prints out all the natural orbital occupancies, the sum and the trace of the density matrix
    for i in range(len(natural_orbital_occupancies)): 
        
        log(f"    {(i + 1):2.0f}. {natural_orbital_occupancies[i]:12.8f}", calculation, 2, silent=silent)

    log(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}", calculation, 2, silent=silent)
    log(f"  Trace of density matrix: {np.trace(P_orthogonal):.6f}", calculation, 2, silent=silent)


    return natural_orbital_occupancies, natural_orbitals









def run_restricted_Laplace_MP2(ERI_AO, molecular_orbitals, F, n_doubly_occ, calculation, P, silent=False):

    """

    Calculates the restricted Laplace transform MP2 energy.

    This is an implementation of the most efficient quadrature method, Euler-Maclaurin B, with the Chebyshev energy-weighted density matrix formula
    See M. Kobayashi and H. Nakai, Chem. Phys. Lett., 2006, 420, 250-255 for details. This calculates the MP2 energy as a functional of the RHF density matrix, without
    reference to the Fock matrix eigenvalues (so suitable for implementation in a linear scaling code).

    Args:
        ERI_AO (array): Electron repulsion integrals in AO basis
        molecular_orbitals (array): Molecular orbitals
        F (array): Fock matrix in AO basis
        n_doubly_occ (int): Number of doubly occupied orbitals
        calculation (Calculation): Calculation object
        P (array): Density matrix in AO basis
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): Restricted Laplace MP2 energy
        
    """

    # Removes the factor of two from the RHF density matrix to restore idempotency
    P /= 2

    log_spacer(calculation, silent=silent, start="\n")
    log("            Laplace Transform MP2 Energy", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log("  Constructing hole density matrix...        ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # This is the unoccupied orbital analogue to the HF density matrix
    Q = scf.construct_hole_density_matrix(molecular_orbitals, n_doubly_occ, 1)

    log("[Done]", calculation, 1, silent=silent)

    tau = calculation.n_MP2_grid_points

    log(f"  Building {tau} point integration grid...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    k = np.arange(1, tau + 1)

    r = k / (tau + 1)

    # Performs the change of variables for the integration
    s = (r ** 3 - 0.9 * r ** 4) / (1 - r) ** 2 + r ** 2 * np.tan(np.pi * r / 2) 

    # Analytical derivative from Wolfram Alpha
    ds_dr = -r / (1 - r) ** 3 * (r * (-1.8 * r ** 2 + 4.6 * r - 3) + 2 * (r - 1) ** 3 * np.tan(np.pi * r / 2) + np.pi / 2 * r * (r - 1) ** 3 * (1 / (np.cos(np.pi * r / 2) ** 2)))

    # Precomputes antisymmetrised ERI array
    L_AO = 2 * ERI_AO - ERI_AO.swapaxes(1, 3)

    log("[Done]", calculation, 1, silent=silent)

    log("\n  Calculating energy components...           ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    f = []

    # Construction of energy-weighted density matrices can not be easily vectorised, more efficient to calculate each e within a loop than through separate contraction

    for i in range(len(s)):

        X = scipy.linalg.expm(s[i] * P @ F) @ P
        Y = scipy.linalg.expm(-s[i] * Q @ F) @ Q

        e = np.einsum("mg,nd,kl,es,gdke,mnls->", X, Y, X, Y, ERI_AO, L_AO, optimize=True)

        f.append(e * ds_dr[i])

    log("[Done]", calculation, 1, silent=silent)

    log("\n  Integrating MP2 energy...                  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Uses the quadrature method to integrate the energy components as a functional of s(r).
    E_MP2 = -1 / (tau + 1) * np.sum(f) 

    log("[Done]", calculation, 1, silent=silent)

    log(f"\n  MP2 correlation energy:           {E_MP2:15.10f}", calculation, 1, silent=silent)


    return E_MP2









def run_iterative_restricted_MP2(ERI_MO, epsilons, molecular_orbitals, o, v, ERI_AO, n_doubly_occ, X, H_core, calculation, SCF_output, silent=False):
   
    """

    Calculates the iterative restricted MP2 energy based on minimising the Hyeralaas functional. Work in progress.

    This can have arbitraryly rotated virtual orbitals, but needs the same number of occupied orbitals. Do not mix occupied and virtual orbitals.
    The matrix indices, ab, may correspond to arbitrary non-orthogonal functions which span the virtual space.

    Args:
        ERI_MO (array): Electron repulsion integrals in spatial MO basis
        epsilons (array): Fock matrix eigenvalues
        molecular_orbitals (array): Molecular orbitals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        ERI_AO (array): Electron repulsion integrals in AO basis
        n_doubly_occ (int): Number of doubly occupied orbitals
        X (array): Fock transformation matrix
        H_core (array): Core Hamiltonian matrix in AO basis
        calculation (Calculation): Calculation object
        SCF_output (Output): Output object
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): Restricted Laplace MP2 energy
        P (array): MP2 density matrix in AO basis
        P_alpha (array): Alpha MP2 density matrix in AO basis
        P_beta (array): Beta MP2 density matrix in AO basis
        natural_orbital_occupancies (list): Natural orbital occupancies
        natural_orbitals (array): MP2 natural orbitals
        
    """

    # Converts to chemists' notation
    ERI_MO = ERI_MO.transpose(0, 2, 1, 3)

    # Builds atomic orbital density matrix and Fock matrix
    P = scf.construct_density_matrix(molecular_orbitals, n_doubly_occ, 2)

    F_AO, _, _ = scf.construct_restricted_Fock_matrix(H_core / 2, H_core / 2, ERI_AO, P, 1, np.zeros_like(H_core))

    # Converts the overlap and Fock matrices to the spatial MO basis
    S = molecular_orbitals.T @ SCF_output.S @ molecular_orbitals

    F = molecular_orbitals.T @ F_AO @ molecular_orbitals

    # Calculates the eigenvalues tensor
    epsilons, _ = scf.diagonalise_Fock_matrix(F_AO, X)
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)

    # Guess doubles amplitudes
    t_ijab = np.zeros_like(ERI_MO[o, o, v, v])

    E_MP2 = 0
    E_conv = calculation.MP2_conv

    log_spacer(calculation, silent=silent, start="\n")
    log("           Iterative MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    # Same convergence keyword as for OMP2
    log(f"\n  Tolerance for energy convergence:    {E_conv:.10f}", 1, silent=silent)
    log(f"\n  Starting MP2 iterations...\n", calculation, 1, end="", silent=silent)

    log_spacer(calculation, silent=silent, start="\n")
    log("  Step          Correlation E               DE", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)

    for step in range(1, calculation.MP2_max_iter + 1):

        E_MP2_old = E_MP2

        # Calculates this residual, minimises over time
        R_ijab = ERI_MO[o, o, v, v] + np.einsum("ap,ijpq,qb->ijab", F[v, v], t_ijab, S[v, v], optimize=True) + np.einsum("ap,ijpq,qb->ijab", S[v, v], t_ijab, F[v, v], optimize=True)
        R_ijab += - np.einsum("ap,ik,kjpq,qb->ijab", S[v, v], F[o, o], t_ijab, S[v, v], optimize=True) - np.einsum("ap,kj,ikpq,qb->ijab", S[v, v], F[o, o], t_ijab, S[v, v], optimize=True)

        # Calculates updated doubles amplitudes
        t_ijab += R_ijab * e_ijab

        # Calculates pair energies
        e_ij = np.einsum("ijab,ijab->ij", ERI_MO[o, o, v, v] + R_ijab, 4 * t_ijab - 2 * t_ijab.swapaxes(0,1), optimize=True) 

        # Total MP2 energy is a sum of pair correlation energies
        E_MP2 = (1 / 2) * np.einsum("ij->", e_ij, optimize=True) 

        # Change from last iteration
        delta_E = np.abs(E_MP2 - E_MP2_old)

        log(f"  {step:3.0f}           {E_MP2:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        if delta_E < E_conv: break 

        elif step > calculation.MP2_max_iter: 
            
            error("Iterative MP2 failed to converge! Try increasing the maximum iterations?")

    log_spacer(calculation, silent=silent)

    log(f"\n  MP2 correlation energy:             {E_MP2:.10f}", calculation, 1, silent=silent)

    log("\n  Constructing MP2 unrelaxed density...", calculation, 1, end="", silent=silent); sys.stdout.flush()

    P_MP2 = np.zeros_like(F)

    # This is the Hartree-Fock density in the MO basis
    P_MP2[o, o] = 2 * np.eye(n_doubly_occ)

    # Builds the MP2 contribution to density
    P_MP2[o, o] += -2 * np.einsum('ikab,kjab->ij', t_ijab, t_ijab, optimize=True)
    P_MP2[v, v] += 2 * np.einsum('ijac,ijcb->ab', t_ijab, t_ijab, optimize=True)

    # Converts from the spatial MO basis to the AO basis
    P = molecular_orbitals @ P_MP2 @ molecular_orbitals.T

    P_alpha = P_beta = P / 2

    log("      [Done]", calculation, 1, silent=silent)
    
    # If not, not requested, calculates and prints the natural orbitals
    if not calculation.no_natural_orbitals: 
        
        natural_orbital_occupancies, natural_orbitals = calculate_natural_orbitals(P, X, calculation, silent=silent)


    return E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals










def run_restricted_MP2(ERI_MO, epsilons, molecular_orbitals, o, v, n_atomic_orbitals, n_doubly_occ, X, calculation, molecule, silent=False):

    """

    Calculates the restricted (SCS-)MP2 energy and unrelaxed density.

    Args:
        ERI_MO (array): Spatial orbital electron repulsion integrals
        epsilons (array): Fock matrix eigenvalues
        molecular_orbitals (array): Molecular orbitals in AO basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        n_atomic_orbitals (int): Number of atomic orbitals
        n_doubly_occ (int): Number of doubly occupied orbitals
        X (array): Fock transformation matrix
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): MP2 correlation energy
        P (array): MP2 unrelaxed density matrix in AO basis
        P_alpha (array): MP2 unrelaxed density matrix for alpha orbitals in AO basis
        P_beta (array): MP2 unrelaxed density matrix for beta orbitals in AO basis

    """

    # Builds doubles epsilons tensor
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)

    # Initialises unscaled scaling factors
    same_spin_scale = 1
    opposite_spin_scale = 1

    log_spacer(calculation, silent=silent, start="\n")
    log("                MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log("  Calculating MP2 correlation energy... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Convert to chemists' notation
    ERI_MO = ERI_MO.transpose(0, 2, 1, 3)

    ERI_MO_ijab = ERI_MO[o, o, v, v]
    ERI_MO_ijab_ansym = ERI_MO_ijab - ERI_MO_ijab.swapaxes(2, 3)
    
    # Spin-components of energy
    E_MP2_OS = np.einsum("ijab,ijab,ijab->", ERI_MO_ijab, ERI_MO_ijab, e_ijab, optimize=True)
    E_MP2_SS = np.einsum("ijab,ijab,ijab->", ERI_MO_ijab, ERI_MO_ijab_ansym, e_ijab, optimize=True)

    log("     [Done]\n", calculation, 1, silent=silent)


    # Optionally scales the same- and opposite-spin contributions to energy
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"] or calculation.DFT_calculation and calculation.functional.functional_type == "spin-scaled double-hybrid": 
        
        E_MP2_SS, E_MP2_OS = spin_component_scale_MP2_energy(E_MP2_SS, E_MP2_OS, calculation.same_spin_scaling, calculation.opposite_spin_scaling, calculation, silent=silent)

    # Total MP2 energy
    E_MP2 = E_MP2_SS + E_MP2_OS

    log(f"  Same spin contribution:             {E_MP2_SS:13.10f}", calculation, 1, silent=silent)
    log(f"  Opposite spin contribution:         {E_MP2_OS:13.10f}", calculation, 1, silent=silent)
    log(f"\n  MP2 correlation energy:             {E_MP2:13.10f}", calculation, 1, silent=silent)

    log("\n  Constructing MP2 unrelaxed density... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Opposite and same-spin t-amplitudes
    t_ijab_OS = -2 * ERI_MO_ijab * e_ijab
    t_ijab_SS = ERI_MO_ijab_ansym * e_ijab

    P_MP2 = np.zeros((n_atomic_orbitals, n_atomic_orbitals))

    P_MP2_OS = P_MP2.copy()
    P_MP2_SS = P_MP2.copy()

    # Build OS-only MP2 density in MO-space 
    P_MP2_OS[o, o] += -0.5 * np.einsum('kiab,kjab->ij', t_ijab_OS, t_ijab_OS, optimize=True)
    P_MP2_OS[v, v] += 0.5 * np.einsum('ijbc,ijac->ab', t_ijab_OS, t_ijab_OS, optimize=True)

    # Build SS-only MP2 density in MO-space
    P_MP2_SS[o, o] += -1 * np.einsum('kiab,kjab->ij', t_ijab_SS, t_ijab_SS, optimize=True)
    P_MP2_SS[v, v] += np.einsum('ijbc,ijac->ab', t_ijab_SS, t_ijab_SS, optimize=True)

    # Optionally applies spin scaling to density
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"]:

        same_spin_scale = calculation.same_spin_scaling
        opposite_spin_scale = calculation.opposite_spin_scaling

    P_MP2[o, o] = 2 * np.eye(n_doubly_occ - molecule.n_core_orbitals) if calculation.freeze_core else 2 * np.eye(n_doubly_occ)

    double_hybrid_scale = calculation.MPC_prop if calculation.DFT_calculation else 1

    P_MP2 += (opposite_spin_scale * P_MP2_OS + same_spin_scale * P_MP2_SS) * double_hybrid_scale

    # Transform the (SCS-)MP2 MO-space density to AO basis
    P = molecular_orbitals @ P_MP2 @ molecular_orbitals.T
    P_alpha = P_beta = P / 2

    log("     [Done]", calculation, 1, silent=silent)
    
    # Calculates and prints natural orbital occupancies
    if not calculation.no_natural_orbitals: 
        
         natural_orbital_occupancies, natural_orbitals = calculate_natural_orbitals(P, X, calculation, silent=silent)

    return E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals











def run_unrestricted_MP2(molecule, calculation, SCF_output, n_SO, o, v, ERI_spin_block, X, silent=False):

    """

    Calculates the MP2 energy and unrelaxed density.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        SCF_output (Output): Output object
        n_SO (int): Number of spin orbitals
        ERI_spin_block (array): Spin-blocked electron repulsion integrals in AO basis
        X (array): Fock transformation matrix
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): MP2 correlation energy
        P (array): MP2 unrelaxed density matrix in AO basis
        P_alpha (array): MP2 unrelaxed density matrix for alpha orbitals in AO basis
        P_beta (array): MP2 unrelaxed density matrix for beta orbitals in AO basis

    """

    molecular_orbitals_alpha = SCF_output.molecular_orbitals_alpha
    molecular_orbitals_beta = SCF_output.molecular_orbitals_beta

    epsilons_alpha = SCF_output.epsilons_alpha
    epsilons_beta = SCF_output.epsilons_beta


    n_occ_alpha = molecule.n_alpha
    n_occ_beta = molecule.n_beta


    # Defines occupied and virtual slices for alpha and beta orbitals
    o_a = slice((o.start + 1) // 2, n_occ_alpha)
    o_b = slice(o.start // 2, n_occ_beta)

    v_a = slice(n_occ_alpha, n_SO // 2)
    v_b = slice(n_occ_beta, n_SO // 2)

    # Initialises unscaled scaling factors
    same_spin_scale = 1
    opposite_spin_scale = 1

    log_spacer(calculation, silent=silent, start="\n")
    log("                MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    # Spin-blocks alpha and beta orbitals separately
    C_spin_block_alpha = ci.spin_block_molecular_orbitals(molecular_orbitals_alpha, molecular_orbitals_alpha, epsilons_alpha)
    C_spin_block_beta = ci.spin_block_molecular_orbitals(molecular_orbitals_beta, molecular_orbitals_beta, epsilons_beta)
    
    # Transforms ERI for alpha, beta and alpha and beta spins
    ERI_SO_a = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_alpha, C_spin_block_alpha)
    ERI_SO_b = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_beta, C_spin_block_beta)
    ERI_SO_ab = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_alpha, C_spin_block_beta)

    # Antisymmetrises alpha and beta spins, but not alpha-beta spins
    g_a = ci.antisymmetrise_integrals(ERI_SO_a)
    g_b = ci.antisymmetrise_integrals(ERI_SO_b)

    log("  Calculating MP2 correlation energy... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    epsilons_alpha = np.sort(epsilons_alpha)
    epsilons_beta = np.sort(epsilons_beta)

    # Slicing out occupied and virtual parts of alpha-alpha, beta-beta and alpha-beta contributions to ERI
    ERI_SO_aa = g_a[o_a, o_a, v_a, v_a]
    ERI_SO_bb = g_b[o_b, o_b, v_b, v_b]
    ERI_SO_ab = ERI_SO_ab[o_a, o_b, v_a, v_b]

    # Epsilons tensor for alpha-alpha, beta-beta and alpha-beta spin pairs
    e_ijab_aa = ci.build_doubles_epsilons_tensor(epsilons_alpha, epsilons_alpha, o_a, o_a, v_a, v_a)
    e_ijab_bb = ci.build_doubles_epsilons_tensor(epsilons_beta, epsilons_beta, o_b, o_b, v_b, v_b)
    e_ijab_ab = ci.build_doubles_epsilons_tensor(epsilons_alpha, epsilons_beta, o_a, o_b, v_a, v_b)

    # MP2 amplitudes for alpha-alpha, beta-beta, alpha-beta and beta-alpha pairs 
    t_ijab_aa = ci.build_MP2_t_amplitudes(ERI_SO_aa, e_ijab_aa)
    t_ijab_bb = ci.build_MP2_t_amplitudes(ERI_SO_bb, e_ijab_bb)
    t_ijab_ab = ci.build_MP2_t_amplitudes(ERI_SO_ab, e_ijab_ab)
    t_ijab_ba = t_ijab_ab.transpose(1, 0, 3, 2) 
    
    # Calculates MP2 energy for alpha-alpha, beta-beta and alpha-beta pairs
    E_aa = calculate_t_amplitude_energy(t_ijab_aa, ERI_SO_aa)
    E_bb = calculate_t_amplitude_energy(t_ijab_bb, ERI_SO_bb)
    E_ab = 4 * calculate_t_amplitude_energy(t_ijab_ab, ERI_SO_ab)

    # Calculates same-spin and opposite-spin contributions
    E_MP2_SS = E_aa + E_bb

    E_MP2_OS = E_ab

    log("     [Done]\n", calculation, 1, silent=silent)

    # Optionally scales the same- and opposite-spin contributions to energy
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"] or calculation.DFT_calculation and calculation.functional.functional_type == "spin-scaled double-hybrid": 
        
        E_MP2_SS, E_MP2_OS = spin_component_scale_MP2_energy(E_MP2_SS, E_MP2_OS, calculation.same_spin_scaling, calculation.opposite_spin_scaling, calculation, silent=silent)


    E_MP2 = E_MP2_SS + E_MP2_OS

    log(f"  Energy from alpha-alpha pairs:      {E_aa:13.10f}", calculation, 1, silent=silent)
    log(f"  Energy from beta-beta pairs:        {E_bb:13.10f}", calculation, 1, silent=silent)
    log(f"  Energy from alpha-beta pairs:       {E_ab:13.10f}", calculation, 1, silent=silent)

    log(f"\n  Same spin contribution:             {E_MP2_SS:13.10f}", calculation, 1, silent=silent)
    log(f"  Opposite spin contribution:         {E_MP2_OS:13.10f}", calculation, 1, silent=silent)
    log(f"\n  MP2 correlation energy:             {E_MP2:13.10f}", calculation, 1, silent=silent)

    log("\n  Constructing MP2 unrelaxed density... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    P_MP2_a = np.zeros((n_SO // 2, n_SO // 2))
    P_MP2_b = np.zeros((n_SO // 2, n_SO // 2))

    # Fills up alpha and beta density matrices with occupied orbitals
    np.fill_diagonal(P_MP2_a[:n_occ_alpha, :n_occ_alpha], 1)
    np.fill_diagonal(P_MP2_b[:n_occ_beta, :n_occ_beta], 1)

    # Finds alpha-alpha and alpha-beta density contributions
    P_MP2_aa = build_t_amplitude_density_contribution(n_SO // 2, t_ijab_aa, o_a, v_a)
    P_MP2_ab = build_t_amplitude_density_contribution(n_SO // 2, t_ijab_ab, o_a, v_a) 

    # Finds beta-beta and beta-alpha density contributions
    P_MP2_bb = build_t_amplitude_density_contribution(n_SO // 2, t_ijab_bb, o_b, v_b)
    P_MP2_ba = build_t_amplitude_density_contribution(n_SO // 2, t_ijab_ba, o_b, v_b)

    # Optionally applies spin scaling to density
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"]:

        same_spin_scale = calculation.same_spin_scaling
        opposite_spin_scale = calculation.opposite_spin_scaling

    # Scales MP2 density if a double-hybrid functional is used
    double_hybrid_scale = calculation.MPC_prop if calculation.DFT_calculation else 1

    # Builds alpha and beta MP2 density matrices
    P_MP2_a += (same_spin_scale * P_MP2_aa + opposite_spin_scale * 2 * P_MP2_ab) * double_hybrid_scale
    P_MP2_b += (same_spin_scale * P_MP2_bb + opposite_spin_scale * 2 * P_MP2_ba) * double_hybrid_scale

    # Transform MP2 density back to AO basis
    P_alpha = molecular_orbitals_alpha @ P_MP2_a @ molecular_orbitals_alpha.T 
    P_beta = molecular_orbitals_beta @ P_MP2_b @ molecular_orbitals_beta.T 

    # Total AO density matrix
    P = P_alpha + P_beta

    log("     [Done]", calculation, 1, silent=silent)
    
    # Calculates and prints natural orbital occupancies
    if not calculation.no_natural_orbitals: 
        
         natural_orbital_occupancies, natural_orbitals = calculate_natural_orbitals(P, X, calculation, silent=silent)

    return E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals














def run_OMP2(molecule, calculation, g, C_spin_block, H_core, V_NN, n_SO, X, E_HF, ERI_spin_block, o, v, silent=False):

    """

    Calculates the orbital-optimised MP2 energy and relaxed density.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        g (array): Antisymmetrised electron repulsion integrals in SO basis
        C_spin_block (array): Spin-blocked molecular integrals in AO basis
        H_core (array): Core hamiltonian matrix in AO basis
        V_NN (float): Nuclear-nuclear repulsion energy
        n_SO (int): Number of spin orbitals
        X (array): Fock transformation matrix
        E_HF (float): Hartree-Fock total energy
        ERI_spin_block (array): Spin-blocked electron repulsion integrals in AO basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        silent (bool, optional): Should anything be printed

    Returns:
        E_OMP2 (float): OMP2 correlation energy
        P (array): OMP2 relaxed density matrix in AO basis
        P_alpha (array): OMP2 relaxed density matrix for alpha orbitals in AO basis
        P_beta (array): OMP2 relaxed density matrix for beta orbitals in AO basis

    """

    n_occ = molecule.n_occ
    n_virt = molecule.n_virt

    E_conv = calculation.MP2_conv
    OMP2_max_iter = calculation.MP2_max_iter


    log_spacer(calculation, silent=silent, start="\n")
    log("      Orbital-optimised MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)


    log(f"\n  Tolerance for energy convergence:    {E_conv:.10f}", 1, silent=silent)
    log("\n  Starting orbital-optimised MP2 iterations...\n", calculation, 1, end="", silent=silent)


    log_spacer(calculation, silent=silent, start="\n")
    log("  Step          Correlation E               DE", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)
    
    E_OMP2_old = 0

    n = np.newaxis

    # Spin blocks and transforms core Hamiltonian to spin-orbital basis
    H_core_spin_block = ci.spin_block_core_Hamiltonian(H_core)
    H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)

    # Sets up matrices based on number of spin orbitals
    P_corr = np.zeros((n_SO, n_SO))
    P_ref = np.zeros((n_SO, n_SO))
    R = np.zeros((n_SO, n_SO))
    D_corr = np.zeros((n_SO, n_SO, n_SO, n_SO))

    # Fills reference one-particle density matrix with ones, up to number of occupied spin orbitals
    P_ref[slice(0, n_occ), slice(0, n_occ)] = np.identity(n_occ)

    n_occ_corr = n_occ - molecule.n_core_spin_orbitals if calculation.freeze_core else n_occ

    # Sets up t amplitudes based on number of virtual and occupied orbitals
    t_abij = np.zeros((n_virt, n_virt, n_occ_corr, n_occ_corr))


    for iteration in range(1, OMP2_max_iter + 1):

        # Build Fock matrix from core Hamiltonian and two-electron integrals, in spin-orbital basis
        F = ci.build_spin_orbital_Fock_matrix(H_core_SO, g, slice(0, n_occ))
        
        # Build off-diagonal Fock matrix, epsilons obtained from diagonal elements
        F_prime = F.copy()
        np.fill_diagonal(F_prime, 0)
        epsilons_combined = F.diagonal()

        # Full t amplitudes for MP2, with permutations 
        t_1 = g[v, v, o, o]
        t_2 = np.einsum('ac,cbij->abij', F_prime[v, v], t_abij, optimize=True)
        t_3 = np.einsum('ki,abkj->abij', F_prime[o, o], t_abij, optimize=True)
        t_abij = t_1 + t_2 - t_2.transpose((1, 0, 2, 3)) - t_3 + t_3.transpose((0, 1, 3, 2))
        
        # Epsilons tensor built and transposed to abij shape, forms final t amplitudes by multiplication
        e_abij = ci.build_doubles_epsilons_tensor(epsilons_combined, epsilons_combined, o, o, v, v).transpose(2,3,0,1) 
        t_abij *= e_abij

        # Build one-particle reduced density matrix, using t_ijab
        P_corr = build_t_amplitude_density_contribution(n_SO, t_abij.transpose(2,3,0,1), o, v)

        # Add to reference P, which is diagonal of ones up to number of occupied spin orbitals
        P_OMP2 = P_corr + P_ref 

        # Forms two-particle density matrix from t amplitudes
        D_corr[v, v, o, o] = t_abij
        D_corr[o, o, v, v] = t_abij.transpose(2,3,0,1)

        # Forms other contributions to two-particle density matrix and their permutations 
        D_2 = np.einsum('rp,sq->rspq', P_corr, P_ref, optimize=True)
        D_3 = np.einsum('rp,sq->rspq', P_ref, P_ref, optimize=True)

        # Forms total two-particle density matrix
        D = D_corr + D_2 - D_2.transpose(1, 0, 2, 3) - D_2.transpose(0, 1, 3, 2) + D_2.transpose(1, 0, 3, 2) + D_3 - D_3.transpose(1, 0, 2, 3)

        # Forms generalised Fock matrix
        F = np.einsum('pr,rq->pq', H_core_SO, P_OMP2, optimize=True) + (1 / 2) * np.einsum('prst,stqr->pq', g, D, optimize=True)

        # Only consider rotations between occupied and virtual orbitals, as only these change the energy
        R[v, o] = (F - F.T)[v, o] / (epsilons_combined[n, o] - epsilons_combined[v, n])

        # Builds orbital rotation matrix by exponentiation
        U = scipy.linalg.expm(R - R.T)

        # Rotates orbitals
        C_spin_block = C_spin_block @ U

        # Uses new orbitals to form new core Hamiltonian and antisymmetrised two-electron integrals
        H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)
        ERI_SO = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block, C_spin_block)
        g = ci.antisymmetrise_integrals(ERI_SO)

        # Calculates total energy from one-electron and two-electron contractions with one- and two-electron Hamiltonians
        E_OMP2 = V_NN + np.einsum("ij,ij->", P_OMP2, H_core_SO, optimize=True) + (1 / 4) * np.einsum("ijkl,ijkl->", D, g, optimize=True)

        # Determines change in energy, and correlation energy by removing original reference (HF) energy
        E_OMP2 = E_OMP2 - E_HF
        delta_E = E_OMP2 - E_OMP2_old

        # Formats output lines nicely
        log(f"  {iteration:3.0f}           {E_OMP2:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)
        
        # Updates the "old" energy, to continue loop
        E_OMP2_old = E_OMP2

        # If convergence criteria has been reached, exit loop
        if (abs(delta_E)) < E_conv:

            break

        elif iteration >= OMP2_max_iter: error("Orbital-optimised MP2 failed to converge! Try increasing the maximum iterations?")


    log_spacer(calculation, silent=silent)

    log(f"\n  OMP2 correlation energy:            {E_OMP2:.10f}", calculation, 1, silent=silent)

    log("\n  Constructing OMP2 relaxed density...", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    P, P_alpha, P_beta = ci.transform_P_SO_to_AO(P_OMP2, C_spin_block, n_SO)

    log("       [Done]", calculation, 1, silent=silent)

    # Calculates natural orbitals from OMP2 density
    if not calculation.no_natural_orbitals: 
        
         natural_orbital_occupancies, natural_orbitals = calculate_natural_orbitals(P, X, calculation, silent=silent)

    return E_OMP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals










def run_restricted_MP3(calculation, g, epsilons, E_MP2, o, v, silent=False):

    """
    
    Calculated the (SCS-)MP3 energy from a spatial orbital basis.

    Args:
        calculation (Calculation): Calculation object
        g (array): Non-antisymmetrised electron repulsion integrals in spatial orbital basis
        epsilons (array): Fock matrix eigenvalues
        E_MP2 (float): MP2 correlation energy
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        silent (bool, optional): Should anything be printed
    
    Returns:
        E_MP3 (float): (SCS-)MP3 correlation energy
        e_ijab (array): Doubles epsilons tensor
        t_ijab (array): MP2 t-amplitudes
        t_tilde_ijab (array): MP3 t_tilde-amplitudes
        L (array): MP2 Lagrange multipliers    

    """
    
    log_spacer(calculation, silent=silent, start="\n")
    log("                      MP3 Energy  ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log("  Calculating amplitudes and multipliers...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Forms MP2 Lagrange multipliers
    L = 2 * g - g.transpose(0,3,2,1)

    # Builds epsilons tensor and t-amplitudes
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
    t_ijab = np.einsum("ijab,aibj->ijab", e_ijab, g[v, o, v, o], optimize=True)
    
    # Taken from Molecular Electronic-Structure Theory
    t_dash_ijab = 2 * np.einsum("ijab,iajb->ijab", e_ijab, L[o, v, o, v], optimize=True)
    t_tilde_ijab = t_dash_ijab
    
    log(f"[Done]", calculation, 1, silent=silent)

    # This should be zero if t_tilde_ijab is calculated correctly
    try:

        symmetry_diagnostic = np.max(t_tilde_ijab - t_tilde_ijab.swapaxes(0,1).swapaxes(2,3))
        log(f"  Symmetry diagnostic: {symmetry_diagnostic:13.10f}\n", calculation, 3,  silent=silent)

    except:
        pass

    log("  Calculating MP3 correlation energy...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Equations from Molecular Electronic-Structure Theory
    X_ijab = (1 / 2) * np.einsum("ijcd,acbd->ijab", t_ijab, g[v, v, v, v], optimize=True) + (1 / 2) * np.einsum("klab,kilj->ijab", t_ijab, g[o, o, o, o], optimize=True)
    X_ijab += np.einsum("ikac,bjkc->ijab", t_ijab, L[v, o, o, v], optimize=True) - np.einsum("kjac,bcki->ijab", t_ijab, g[v, v, o, o], optimize=True) - np.einsum("kiac,bjkc->ijab", t_ijab, g[v, o, o, v], optimize=True) 

    E_MP3 = np.einsum("ijab,ijab->", t_tilde_ijab, X_ijab, optimize=True)

    log(f"[Done]\n\n  MP3 correlation energy:             {E_MP3:13.10f}", calculation, 1, silent=silent)

    # Applies Grimme's default scaling to MP3 energy if SCS-MP3 is requested, then prints this information
    if calculation.method in ["SCS-MP3", "USCS-MP3"]:

        E_MP3 *= calculation.MP3_scaling
        
        log(f"\n  Scaling for MP3: {calculation.MP3_scaling:.3f}\n", calculation, 1, silent=silent)
        log(f"  Scaled MP3 correlation energy:    {E_MP3:15.10f}", calculation, 1, silent=silent)
        log(f"  SCS-MP3 correlation energy:       {(E_MP3 + E_MP2):15.10f}", calculation, 1, silent=silent)


    return E_MP3, e_ijab, t_ijab, t_tilde_ijab, L










def run_unrestricted_MP3(calculation, g, epsilons_sorted, E_MP2, o, v, silent=False):

    """

    Calculates the (SCS-)MP3 energy from a spin orbital basis.

    Args:
        calculation (Calculation): Calculation object
        g (array): Antisymmetrised electron repulsion integrals in SO basis
        epsilons_sorted (array): Sorted array of Fock matrix eigenvalues
        E_MP2 (float): MP2 correlation energy
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP3 (float): (SCS-)MP3 correlation energy

    """

    log_spacer(calculation, silent=silent, start="\n")
    log("                      MP3 Energy  ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    e_ijab = ci.build_doubles_epsilons_tensor(epsilons_sorted, epsilons_sorted, o, o, v, v)

    log("  Calculating MP3 correlation energy...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    E_MP3 = (1 / 8) * np.einsum('ijab,klij,abkl,ijab,klab->', g[o, o, v, v], g[o, o, o, o], g[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += (1 / 8) * np.einsum('ijab,abcd,cdij,ijab,ijcd->', g[o, o, v, v], g[v, v, v, v], g[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += np.einsum('ijab,kbcj,acik,ijab,ikac->', g[o, o, v, v], g[o, v, v, o], g[v, v, o, o], e_ijab, e_ijab, optimize=True)

    log(f"[Done]\n\n  MP3 correlation energy:             {E_MP3:13.10f}", calculation, 1, silent=silent)

    # Applies Grimme's default scaling to MP3 energy if SCS-MP3 is requested, then prints this information
    if calculation.method in ["SCS-MP3", "USCS-MP3"]:

        E_MP3 *= calculation.MP3_scaling
        
        log(f"\n  Scaling for MP3: {calculation.MP3_scaling:.3f}\n", calculation, 1, silent=silent)
        log(f"  Scaled MP3 correlation energy:    {E_MP3:15.10f}", calculation, 1, silent=silent)
        log(f"  SCS-MP3 correlation energy:       {(E_MP3 + E_MP2):15.10f}", calculation, 1, silent=silent)


    return E_MP3










def run_restricted_MP4(e_ijab, t_ijab, t_tilde_ijab, L, g, epsilons, o, v, calculation, silent=False):

    """
    
    Calculated the MP4 energy from a spatial orbital basis.
    

    Args:
        e_ijab (array): Doubles epsilons tensor
        t_ijab (array): MP2 t-amplitudes
        t_tilde_ijab (array): MP t_tilde-amplitudes
        L (array): MP2 Lagrange multipliers    
        g (array): Non-antisymmetrised electron repulsion integrals in spatial orbital basis
        epsilons (array): Fock matrix eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed
    
    Returns:
        E_MP4 (float): MP4 correlation energy

    """
    
    log_spacer(calculation, silent=silent, start="\n")
    log("                      MP4 Energy  ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)


    if calculation.no_singles: calculation.method = "MP4[DQ]"

    log("  Calculating amplitudes and multipliers...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Doesn't calculate singles contributions for MP4[DQ]
    if calculation.method != "MP4[DQ]":
        
        # Builds second-order singles epsilons tensor and singles t-amplitudes
        e_ia = ci.build_singles_epsilons_tensor(epsilons, o, v)

        t_ia_2 = np.einsum("klad,kild->ia", t_ijab, L[o, o, o, v], optimize=True) - np.einsum("kicd,adkc->ia", t_ijab, L[v, v, o, v], optimize=True)
        
        t_ia_2 *= -e_ia

    # Builds second-order doubles t_amplitudes for MP4
    t_ijab_2 = -1 * np.einsum("ijcd,acbd->ijab", t_ijab, g[v, v, v, v], optimize=True) - np.einsum("klab,kilj->ijab", t_ijab, g[o, o, o, o], optimize=True)
    t_ijab_2 += -1 * doubly_permute(np.einsum("ijkabc->ijab", np.einsum("ikac,bjkc->ijkabc", t_ijab, L[v, o, o, v], optimize=True) - np.einsum("kjac,bcki->ijkabc", t_ijab, g[v, v, o, o], optimize=True) - np.einsum("kiac,bjkc->ijkabc", t_ijab, g[v, o, o, v], optimize=True), optimize=True), (0, 1), (2, 3))
    
    # Minus sign here due to difference in definition of e_ijab in TUNA and Molecular Electronic-Structure Theory
    t_ijab_2 *= -e_ijab 

    # Only calculates triples contributions for MP4 and MP4[SDTQ]
    if calculation.method in ["MP4", "MP4[SDTQ]"]:

        # Builds second-order triples epsilons tensor and singles t-amplitudes
        e_ijkabc = ci.build_triples_epsilons_tensor(epsilons, o, v)

        t_ijkabc_2 = -1 * six_index_permute(np.einsum("ijad,ckbd->ijkabc", t_ijab, g[v, o, v, v], optimize=True) - np.einsum("ilab,cklj->ijkabc", t_ijab, g[v, o, o, o], optimize=True))
        
        # Minus sign here due to difference in definition of e_ijab in TUNA and Molecular Electronic-Structure Theory
        t_ijkabc_2 *= -e_ijkabc


    log(f"[Done]", calculation, 1, silent=silent)

    log("  Calculating MP4 correlation energy...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.method != "MP4[DQ]":

        # Singles contribution to MP4 energy
        S_ijab = np.einsum("jc,aibc->ijab", t_ia_2, g[v, o, v, v], optimize=True) - np.einsum("kb,aikj->ijab", t_ia_2, g[v, o, o, o], optimize=True)
    
    else:

        S_ijab = np.zeros_like(t_ijab)
    
    # Doubles contribution to MP4 energy
    D_ijab = (1 / 2) * np.einsum("ijcd,acbd->ijab", t_ijab_2, g[v, v, v, v], optimize=True) + (1 / 2) * np.einsum("klab,kilj->ijab", t_ijab_2, g[o, o, o, o], optimize=True)
    D_ijab += np.einsum("ikac,bjkc->ijab", t_ijab_2, L[v, o, o, v], optimize=True) - np.einsum("kjac,bcki->ijab", t_ijab_2, g[v, v, o, o], optimize=True) - np.einsum("kiac,bjkc->ijab", t_ijab_2, g[v, o, o, v], optimize=True)

    if calculation.method in ["MP4", "MP4[SDTQ]"]:

        # Triples contribution to MP4 energy (slowest step)
        T_ijab = np.einsum("ijkacd,bckd->ijab", t_ijkabc_2, L[v, v, o, v], optimize=True) - np.einsum("kjiacd,kdbc->ijab", t_ijkabc_2, g[o, v, v, v], optimize=True)
        T_ijab += -1 * np.einsum("iklabc,kjlc->ijab", t_ijkabc_2, L[o, o, o, v], optimize=True) + np.einsum("lkiabc,kjlc->ijab", t_ijkabc_2, g[o, o, o, v], optimize=True)

    else:

        T_ijab = np.zeros_like(t_ijab)

    # Quadruples contribution to MP4 energy 
    Q_ijab = (1 / 2) * np.einsum("klab,ijkl->ijab", t_ijab, np.einsum("ijcd,kcld->ijkl", t_ijab, g[o, v, o, v], optimize=True), optimize=True)
    Q_ijab += np.einsum("ikac,jkbc->ijab", t_ijab, np.einsum("jlbd,kcld->jkbc", t_ijab - t_ijab.swapaxes(0, 1), L[o, v, o, v], optimize=True), optimize=True)
    Q_ijab += (1 / 2) * np.einsum("kiac,jkbc->ijab", t_ijab, np.einsum("ljbd,kcld->jkbc", t_ijab, g[o, v, o, v], optimize=True), optimize=True)
    Q_ijab += (1 / 2) * np.einsum("kjad,ikbd->ijab", t_ijab, np.einsum("libc,kcld->ikbd", t_ijab, g[o, v, o, v], optimize=True), optimize=True)
    Q_ijab += -1 * np.einsum("ikab,jk->ijab", t_ijab, np.einsum("ljcd,lckd->jk", t_ijab, L[o, v, o, v], optimize=True), optimize=True)
    Q_ijab += -1 * np.einsum("ijac,bc->ijab", t_ijab, np.einsum("klbd,kcld->bc", t_ijab, L[o, v, o, v], optimize=True), optimize=True)

    # Final contractions to calculate the components of the MP4 energy
    E_MP4_S = np.einsum("ijab,ijab->", t_tilde_ijab, S_ijab, optimize=True)
    E_MP4_D = np.einsum("ijab,ijab->", t_tilde_ijab, D_ijab, optimize=True)
    E_MP4_T = np.einsum("ijab,ijab->", t_tilde_ijab, T_ijab, optimize=True)
    E_MP4_Q = np.einsum("ijab,ijab->", t_tilde_ijab, Q_ijab, optimize=True)

    E_MP4 = E_MP4_S + E_MP4_D + E_MP4_T + E_MP4_Q

    log(f"[Done]\n", calculation, 1, silent=silent)

    # Printiing information about what contributions were included in the MP4 energy
    if calculation.method == "MP4[SDQ]": log(f"  Triples are not included in MP4(SDQ).\n", calculation, 1, silent=silent)
    elif calculation.method == "MP4[DQ]": log(f"  Singles and triples are not included in MP4(DQ).\n", calculation, 1, silent=silent)
    else: log(f"  Triples are included in full MP4.\n", calculation, 1, silent=silent)

    log(f"  Singles correlation energy:         {E_MP4_S:13.10f}", calculation, 1, silent=silent)
    log(f"  Doubles correlation energy:         {E_MP4_D:13.10f}", calculation, 1, silent=silent)
    log(f"  Triples correlation energy:         {E_MP4_T:13.10f}", calculation, 1, silent=silent)
    log(f"  Quadruples correlation energy:      {E_MP4_Q:13.10f}", calculation, 1, silent=silent)

    log(f"\n  MP4 correlation energy:             {E_MP4:13.10f}", calculation, 1, silent=silent)


    return E_MP4










def calculate_Moller_Plesset(method, molecule, SCF_output, ERI_AO, calculation, X, H_core, V_NN, silent=False):

    """

    Calculates the requested Moller-Plesset perturbation theory energy and density.

    Args:
        method (str): Electronic structure method
        molecule (Molecule): Molecule object
        SCF_output (Output): SCF output object
        ERI_AO (array): Electron repulsion integrals in AO basis 
        calculation (Calculation): Calculation object
        X (array): Fock transformation matrix
        H_core (array): Core Hamiltonian matrix in AO basis
        V_NN (float): Nuclear-nuclear repulsion energy
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): (SCS-)MP2 correlation energy
        E_MP3 (float): (SCS-)MP3 correlation energy
        E_MP4 (float): MP4 correlation energy
        P (array): (SCS-)MP2 unrelaxed density matrix in AO basis
        P_alpha (array): (SCS-)MP2 unrelaxed alpha density matrix in AO basis
        P_beta (array): (SCS-)MP2 unrelaxed beta density matrix in AO basis

    """

    E_MP2 = 0
    E_MP3 = 0
    E_MP4 = 0

    P = SCF_output.P
    P_alpha = SCF_output.P_alpha
    P_beta = SCF_output.P_beta

    n_SO = molecule.n_SO
    n_occ = molecule.n_occ
    n_doubly_occ = molecule.n_doubly_occ

    natural_orbital_occupancies, natural_orbitals = None, None

    # Calculates useful quantities for all spin orbital or spatial orbital calculations

    if calculation.reference == "UHF" or "OMP2" in method:

        g, C_spin_block, epsilons_sorted, ERI_spin_block, o, v, _, _ = ci.begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=silent)
    
    else:

        ERI_MO, molecular_orbitals, epsilons, o, v = ci.begin_spatial_orbital_calculation(molecule, ERI_AO, SCF_output, n_doubly_occ, calculation, silent=silent)


    if method in ["OMP2", "UOMP2", "OOMP2", "UOOMP2"]: 
        
        E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals = run_OMP2(molecule, calculation, g, C_spin_block, H_core, V_NN, n_SO, X, SCF_output.energy, ERI_spin_block, o, v, silent=silent)
    
    elif method == "IMP2":

        E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals = run_iterative_restricted_MP2(ERI_MO, epsilons, molecular_orbitals, o, v, ERI_AO, n_doubly_occ, X, H_core, calculation, SCF_output, silent=silent)

    elif method == "LMP2":

        E_MP2 = run_restricted_Laplace_MP2(ERI_AO, molecular_orbitals, SCF_output.F, n_doubly_occ, calculation, SCF_output.P, silent=silent)


    elif method in ["MP2", "SCS-MP2", "UMP2", "USCS-MP2", "MP3", "UMP3", "SCS-MP3", "USCS-MP3", "MP4", "MP4[SDQ]", "MP4[SDTQ]", "MP4[DQ]"] or method in DFT_methods:
         
        if calculation.reference == "UHF":

            E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals = run_unrestricted_MP2(molecule, calculation, SCF_output, n_SO, o, v, ERI_spin_block, X, silent=silent)

        else:

            E_MP2, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals = run_restricted_MP2(ERI_MO, epsilons, molecular_orbitals, o, v, molecule.n_atomic_orbitals, n_doubly_occ, X, calculation, molecule, silent=silent)


        if method in ["MP3", "UMP3", "SCS-MP3", "USCS-MP3", "MP4", "MP4[SDQ]", "MP4[SDTQ]", "MP4[DQ]"]: 

            if calculation.reference == "UHF":
            
                E_MP3 = run_unrestricted_MP3(calculation, g, epsilons_sorted, E_MP2, o, v, silent=silent)

            else:

                E_MP3, e_ijab, t_ijab, t_tilde_ijab, L = run_restricted_MP3(calculation, ERI_MO, epsilons, E_MP2, o, v, silent=silent)

                if method in ["MP4", "MP4[SDQ]", "MP4[SDTQ]", "MP4[DQ]"]:
                    
                    E_MP4 = run_restricted_MP4(e_ijab, t_ijab, t_tilde_ijab, L, ERI_MO, epsilons, o, v, calculation, silent=silent)

    log_spacer(calculation, silent=silent)

    return E_MP2, E_MP3, E_MP4, P, P_alpha, P_beta, natural_orbital_occupancies, natural_orbitals
