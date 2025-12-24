import numpy as np
from tuna_util import log, error, symmetrise, log_big_spacer, Output, DFT_methods
import tuna_dft as dft


"""

This is the TUNA module for the self-consistent field (SCF) cycle, written first for version 0.1.0 and rewritten for version 0.9.0.

The module contains:

1. Some small utility functions (calculate_exchange_matrix, calculate_Coulomb_matrix, construct_density_matrix, etc.)
2. Functions to check the convergence of the SCF loop (calculate_SCF_changes, etc.)
3. Functions to calculate the energy for RHF and UHF references, and the respective Fock matrices (calculate_unrestricted_electronic_energy, etc.)
4. Functions for convergence acceleration (apply_DIIS, apply_damping)
5. The restricted and unrestricted SCF iteration functions, and the overall loop (run_self_consistent_field_cycle, etc.)

"""



def calculate_exchange_matrix(P, ERI_AO):

    """

    Calculates the Fock exchange matrix.

    Args:
        P (array): Density matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
    
    Returns:
        K (array): Fock exchange matrix in AO basis

    """

    K = np.einsum("ilkj,kl->ij", ERI_AO, P, optimize=True)

    return K






def calculate_Coulomb_matrix(P, ERI_AO):

    """

    Calculates the classical electron repulsion matrix.

    Args:
        P (array): Density matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
    
    Returns:
        J (array): Coulomb matrix in AO basis

    """

    J = np.einsum("ijkl,kl->ij", ERI_AO, P, optimize=True)

    return J







def format_output_line(E_total, delta_E, max_DP, RMS_DP, damping_factor, step, commutator, calculation, silent=False):

    """
    
    Prints an output line for a self-consistent field step.

    Args:
        E_total (float): Total molecular energy
        delta_E (float): Change in energy since last step
        max_DP (float): Maximum absolute change in the density matrix
        RMS_DP (float): Root-mean-square change in the density matrix
        step (int): Iteration of self-consistent field loop
        commutator (float): Error calculated from [F, PS]
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    """

    # Formats damping factor as a line if there's no damping
    damping_factor = f"{damping_factor:.3f}" if damping_factor != 0 else " ---"

    log(f"  {step:3.0f}  {E_total:16.10f}  {delta_E:16.10f} {RMS_DP:16.10f} {max_DP:16.10f} {commutator:16.10f}     {damping_factor}", calculation, 1, silent=silent)   

    return







def construct_density_matrix(molecular_orbitals, n_occ, n_electrons_per_orbital):

    """

    Builds the density matrix from molecular orbitals.

    Args:
        molecular_orbitals (array): Molecular orbitals in AO basis
        n_occ (int): Number of occupied molecular orbitals
        n_electrons_per_orbital (int): Number of electrons per molecular orbital (1 for UHF, 2 for RHF)
    
    Returns:
        P (array): Density matrix in AO basis

    """

    # Slices out occupied molecular orbitals
    occupied_mos = molecular_orbitals[:, :n_occ]

    # Builds density matrix
    P = n_electrons_per_orbital * occupied_mos @ occupied_mos.T

    # Symmetrises density matrix
    P = symmetrise(P)

    return P
    






def construct_hole_density_matrix(molecular_orbitals, n_occ, n_electrons_per_orbital):

    """

    Builds the hole density matrix from molecular orbitals.

    Args:
        molecular_orbitals (array): Molecular orbitals in AO basis
        n_occ (int): Number of occupied molecular orbitals
        n_electrons_per_orbital (int): Number of electrons per molecular orbital (1 for UHF, 2 for RHF)
    
    Returns:
        Q (array): Hole density matrix in AO basis

    """

    # Slices out unoccupied molecular orbitals
    unoccupied_mos = molecular_orbitals[:, n_occ:]

    # Builds hole density matrix
    Q = n_electrons_per_orbital * unoccupied_mos @ unoccupied_mos.T

    # Symmetrises hole density matrix
    Q = symmetrise(Q)

    return Q
    






def diagonalise_Fock_matrix(F, X):

    """

    Transforms and diagonalises Fock matrix for molecular orbitals and orbital energies.

    Args:
        F (array): Fock matrix in AO basis
        X (array): Fock transformation matrix

    Returns:
        epsilons (array): Fock matrix eigenvalues, orbital energies
        molecular_orbitals (array): Molecular orbitals in AO basis

    """

    # Orthonormalises the Fock matrix - symmetrisation improves convergence
    F_orthonormal = symmetrise(X.T @ F @ X)
    
    # Diagonalises the Fock matrix, assuming its Hermitian
    epsilons, eigenvectors = np.linalg.eigh(F_orthonormal)

    # Transforms molecular orbitals back to non-orthogonal AO basis
    molecular_orbitals = X @ eigenvectors

    return epsilons, molecular_orbitals







def calculate_SCF_changes(E, E_old, P, P_old):

    """

    Calculates changes to the energy and density matrix.

    Args:
        E (float): Energy
        E_old (float): Energy of last iteration
        P (array): Density matrix in AO basis
        P_old (array): Density matrix in AO basis of last iteration

    Returns:
        delta_E (float): Change in energy
        max_DP (float): Maximum absolute change in the density matrix
        RMS_DP (float): Root-mean-square change in the density matrix

    """

    delta_E = E - E_old
    delta_P = P - P_old
    
    # Calculates max changes in the density matrix
    max_DP = np.max(np.abs(delta_P))
    RMS_DP = np.mean(delta_P ** 2) ** (1 / 2)

    return delta_E, max_DP, RMS_DP








def check_convergence(SCF_conv, step, delta_E, max_DP, RMS_DP, commutator, calculation, silent=False):

    """

    Checks the convergence of the SCF loop.

    Args:
        SCF_conv (dict): Dictionary of SCF convergence thresholds
        step (int): Iteration of SCF
        delta_E (float): Change in energy since last step
        max_DP (float): Maximum change in density matrix
        RMS_DP (float): Root-mean-square change in density matrix
        commutator (float): Error calculates by root-mean-square of [F, PS]
        calculation (Calculation): Calculation object
        silent (bool, optional): Not silent by default

    Returns:
        converged (bool): Checks if the calculation has converged or not

    """
    
    converged = False

    # All factors must be below their thresholds to be converged
    if abs(delta_E) < SCF_conv["delta_E"] and abs(max_DP) < SCF_conv["max_DP"] and abs(RMS_DP) < SCF_conv["RMS_DP"] and abs(commutator) < SCF_conv["commutator"]: 

        log_big_spacer(calculation, silent=silent)
        log(f"\n Self-consistent field converged in {step} cycles!\n", calculation, 1, silent=silent)

        converged = True


    return converged   








def calculate_restricted_electronic_energy(T, V_NE, P, J, K, calculation, density, weights, e_X, e_C):

    """
    
    Calculates energy components for a restricted Hartree-Fock or restricted Kohn-Sham calculation.

    Args:
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        P (array): Density matrix in AO basis
        J (array): Coulomb matrix in AO basis
        K (array): Exchange matrix in AO basis
        calculation (Calculation): Calculation object
        density (array): Electron density
        weights (array): Integration weights
        e_X (array): Exchange energy density
        e_C (array): Correlation energy density
    
    Returns:
        electronic_energy (float): Total electronic energy
        energy_components (tuple): Tuple of energy components (kinetic, nuclear-electron, coulomb, exchange, correlation) 

    """

    # Calculates one-electron contributions to energy
    kinetic_energy = np.einsum("ij,ij->", P, T, optimize=True)
    nuclear_electron_energy = np.einsum("ij,ij->", P, V_NE, optimize=True)

    # Calculates classical electron-electron repulsion energy
    coulomb_energy = (1 / 2) * np.einsum("ij,ij->", P, J, optimize=True)

    # Calculates Fock exchange energy, multiplies by HFX proportion for hybrid functionals
    exchange_energy = -(1 / 4) * np.einsum("ij,ij->", P, K, optimize=True) * calculation.HFX_prop

    correlation_energy = 0

    # The following only applies if a DFT calculation is being performed
    if weights is not None:

        # Integrates exchange energy density on a grid, scales for hybrid functionals
        exchange_energy += dft.integrate_on_grid(e_X * density, weights) * calculation.DFX_prop if e_X is not None else 0

        # Integrates correlation energy density on a grid, scales for double-hybrid functionals
        correlation_energy += dft.integrate_on_grid(e_C * density, weights) * calculation.DFC_prop if e_C is not None else 0

    # Sums up total electronic energy
    electronic_energy = kinetic_energy + nuclear_electron_energy + coulomb_energy + exchange_energy + correlation_energy

    # Packages energy components into a tuple
    energy_components = kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, correlation_energy

    return electronic_energy, energy_components







def calculate_unrestricted_electronic_energy(T, V_NE, P_alpha, P_beta, J_alpha, J_beta, K_alpha, K_beta, calculation, alpha_density, beta_density, weights, e_X_alpha, e_X_beta, e_C):
    
    """
    
    Calculates energy components for an unrestricted Hartree-Fock or unrestricted Kohn-Sham calculation.

    Args:
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        J_alpha (array): Alpha coulomb matrix in AO basis
        J_beta (array): Beta coulomb matrix in AO basis
        K_alpha (array): Alpha exchange matrix in AO basis
        K_beta (array): Beta exchange matrix in AO basis
        calculation (Calculation): Calculation object
        alpha_density (array): Alpha electron density
        beta_density (array): Beta electron density
        weights (array): Integration weights
        e_X_alpha (array): Alpha exchange energy density
        e_X_beta (array): Beta exchange energy density
        e_C (array): Correlation energy density
    
    Returns:
        electronic_energy (float): Total electronic energy
        energy_components (tuple): Tuple of energy components (kinetic, nuclear-electron, coulomb, exchange, correlation) 

    """

    P = P_alpha + P_beta

    # Calculates one-electron contributions to energy
    kinetic_energy = np.einsum("ij,ij->", P, T, optimize=True)
    nuclear_electron_energy = np.einsum("ij,ij->", P, V_NE, optimize=True)

    # Calculates classical electron-electron repulsion energy
    coulomb_energy = (1 / 2) * np.einsum("ij,ij->",P, J_alpha + J_beta, optimize=True)

    # Calculates Fock exchange energy, multiplies by HFX proportion for hybrid functionals
    exchange_energy_alpha = -(1 / 2) * np.einsum("ij,ij->", P_alpha, K_alpha, optimize=True) * calculation.HFX_prop
    exchange_energy_beta = -(1 / 2) * np.einsum("ij,ij->", P_beta, K_beta, optimize=True) * calculation.HFX_prop

    correlation_energy = 0

    if weights is not None:

        # Integrates exchange energy density on a grid, scales for hybrid functionals
        exchange_energy_alpha += dft.integrate_on_grid(e_X_alpha * alpha_density, weights) * calculation.DFX_prop if e_X_alpha is not None else 0
        exchange_energy_beta += dft.integrate_on_grid(e_X_beta * beta_density, weights) * calculation.DFX_prop if e_X_beta is not None else 0
        
        # Integrates correlation energy density on a grid, scales for double-hybrid functionals
        correlation_energy += dft.integrate_on_grid(e_C * (alpha_density + beta_density), weights) * calculation.DFC_prop if e_C is not None else 0
    
    # Sums up exchange energy
    exchange_energy = exchange_energy_alpha + exchange_energy_beta

    # Sums up total electronic energy
    electronic_energy = kinetic_energy + nuclear_electron_energy + coulomb_energy + exchange_energy + correlation_energy

    # Packages energy components into a tuple
    energy_components = kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, correlation_energy

    return electronic_energy, energy_components




    



def construct_restricted_Fock_matrix(T, V_NE, ERI_AO, P, HFX_prop, V_XC):

    """
    
    Calculates the Fock matrix for a restricted Hartree-Fock or restricted Kohn-Sham calculation.

    Args:
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        P (array): Density matrix in AO basis
        HFX_prop (float): Proportion of Hartree-Fock exchange
        V_XC (array): Exchange-correlation matrix in AO basis
    
    Returns:
        F (array): Fock matrix in AO basis
        J (array): Coulomb matrix in AO basis
        K (array): Exchange matrix in AO basis
    
    """

    V_XC = V_XC if V_XC is not None else 0

    # Calculates the Coulomb and Fock exchange matrices
    J = calculate_Coulomb_matrix(P, ERI_AO)
    K = calculate_exchange_matrix(P, ERI_AO)

    # Builds the Fock matrix from one electron and two-electron contributions, scaling the exchange matrix for hybrid functionals
    F = T + V_NE + J - (1 / 2) * K * HFX_prop + V_XC

    # Symmetrises the Fock matrix
    F = symmetrise(F)

    return F, J, K








def construct_unrestricted_Fock_matrices(T, V_NE, ERI_AO, P_alpha, P_beta, HFX_prop, V_XC_alpha, V_XC_beta):

    """
    
    Calculates the Fock matrix for a restricted Hartree-Fock or restricted Kohn-Sham calculation.

    Args:
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        HFX_prop (float): Proportion of Hartree-Fock exchange
        V_XC_alpha (array): Alpha exchange-correlation matrix in AO basis
        V_XC_beta (array): Beta exchange-correlation matrix in AO basis
    
    Returns:
        F_alpha (array): Alpha Fock matrix in AO basis
        F_beta (array): Beta Fock matrix in AO basis
        J_alpha (array): Alpha Coulomb matrix in AO basis
        J_beta (array): Beta Coulomb matrix in AO basis
        K_alpha (array): Alpha exchange matrix in AO basis
        K_beta (array): Beta exchange matrix in AO basis
    
    """

    V_XC_alpha = V_XC_alpha if V_XC_alpha is not None else 0
    V_XC_beta = V_XC_beta if V_XC_beta is not None else 0

    # Calculates the Coulomb matrices
    J_alpha = calculate_Coulomb_matrix(P_alpha, ERI_AO)
    J_beta = calculate_Coulomb_matrix(P_beta, ERI_AO)

    # Calculates the Fock exchange matrices
    K_alpha = calculate_exchange_matrix(P_alpha, ERI_AO)
    K_beta = calculate_exchange_matrix(P_beta, ERI_AO)

    # Builds the Fock matrices from one electron and two-electron contributions, scaling the exchange matrix for hybrid functionals
    F_alpha = T + V_NE + J_alpha + J_beta - K_alpha * HFX_prop + V_XC_alpha
    F_beta = T + V_NE + J_alpha + J_beta - K_beta * HFX_prop + V_XC_beta

    # Symmetrises the Fock matrices
    F_alpha = symmetrise(F_alpha)
    F_beta = symmetrise(F_beta)

    return F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta








def calculate_restricted_exchange_correlation_matrix(P, bfs_on_grid, bf_gradients_on_grid, weights, calculation, exchange_functional, correlation_functional):

    """
    
    Calculates the exchange-correlation matrix for a restricted Kohn-Sham calculation.

    Args:
        P (array): Density matrix in AO basis
        bfs_on_grid (array): Basis functions evaluated on integration grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on integration grid
        weights (array): Integration weights
        calculation (Calculation): Calculation object
        exchange_functional (function): Exchange functional
        correlation_functional (function): Correlation functional

    Returns:
        V_XC (array): Exchange-correlation matrix in AO basis
        density (array): Electron density
        e_X (array): Exchange energy density
        e_C (array): Correlation energy density
    
    """

    # Constructs the electron density on a grid
    density = dft.construct_density_on_grid(P, bfs_on_grid)

    sigma, tau, density_gradient = None, None, None

    if calculation.functional.functional_class in ["GGA", "meta-GGA"]:

        # Calculates the density gradient for a GGA calculation
        sigma, density_gradient = dft.calculate_density_gradient(P, bfs_on_grid, bf_gradients_on_grid) 

        if calculation.functional.functional_class == "meta-GGA":

            tau = dft.calculate_kinetic_energy_density(P, bf_gradients_on_grid)

    # Calculates derivatives necessary for XC matrix
    df_dn_X, df_ds_X, df_dt_X, e_X = exchange_functional(density, sigma, tau, calculation) if exchange_functional is not None else (None, None, None, None)
    df_dn_C, df_ds_C, df_dt_C, e_C = correlation_functional(density, sigma, tau, calculation) if correlation_functional is not None else (None, None, None, None)
    
    # Builds the exchange and correlation matrices
    V_X = dft.calculate_V_X(weights, bfs_on_grid, df_dn_X, df_ds_X, df_dt_X, bf_gradients_on_grid, density_gradient) if df_dn_X is not None else np.zeros_like(P)
    V_C = dft.calculate_V_C(weights, bfs_on_grid, df_dn_C, df_ds_C, df_dt_C, bf_gradients_on_grid, density_gradient) if df_dn_C is not None else np.zeros_like(P)

    # Constructs exchange-correlation matrix considering hybrid functionals
    V_XC = V_X * calculation.DFX_prop + V_C * calculation.DFC_prop

    return V_XC, density, e_X, e_C








def calculate_unrestricted_exchange_correlation_matrix(P_alpha, P_beta, bfs_on_grid, bf_gradients_on_grid, weights, calculation, exchange_functional, correlation_functional):
    
    """
    
    Calculates the exchange-correlation matrix for an unrestricted Kohn-Sham calculation.

    Args:
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        bfs_on_grid (array): Basis functions evaluated on integration grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on integration grid
        weights (array): Integration weights
        calculation (Calculation): Calculation object
        exchange_functional (function): Exchange functional
        correlation_functional (function): Correlation functional

    Returns:
        V_XC_alpha (array): Alpha exchange-correlation matrix in AO basis
        V_XC_beta (array): Beta exchange-correlation matrix in AO basis
        alpha_density (array): Alpha electron density
        beta_density (array): Beta electron density
        density (array): Electron density
        e_X_alpha (array): Alpha exchange energy density
        e_X_beta (array): Beta exchange energy density
        e_C (array): Correlation energy density
    
    """

    # Constructs the electron density on a grid
    alpha_density = dft.construct_density_on_grid(P_alpha, bfs_on_grid)
    beta_density = dft.construct_density_on_grid(P_beta, bfs_on_grid)

    density = alpha_density + beta_density
    
    sigma_aa, sigma_bb, sigma_ab, density_gradient_alpha, density_gradient_beta, tau_alpha, tau_beta = None, None, None, None, None, None, None

    if calculation.functional.functional_class in ["GGA", "meta-GGA"]:

        # Calculates the density gradient for a GGA calculation
        sigma_aa, density_gradient_alpha = dft.calculate_density_gradient(P_alpha, bfs_on_grid, bf_gradients_on_grid) 
        sigma_bb, density_gradient_beta = dft.calculate_density_gradient(P_beta, bfs_on_grid, bf_gradients_on_grid) 

        # This sigma is cleaned here as the others are cleaned in calculate_density_gradient - do NOT clean this
        sigma_ab = np.einsum("akl,akl->kl", density_gradient_alpha, density_gradient_beta, optimize=True)

        if calculation.functional.functional_class == "meta-GGA":

            tau_alpha = dft.calculate_kinetic_energy_density(P_alpha, bf_gradients_on_grid)
            tau_beta = dft.calculate_kinetic_energy_density(P_beta, bf_gradients_on_grid)


    # Accounts for the spin scaling of the exact exchange functional
    alpha_density_scaled, beta_density_scaled = alpha_density * 2, beta_density * 2
    sigma_aa_scaled, sigma_bb_scaled = (sigma_aa * 4, sigma_bb * 4) if sigma_aa is not None else (None, None)
    tau_alpha_scaled, tau_beta_scaled = (tau_alpha * 2, tau_beta * 2) if tau_alpha is not None else (None, None)

    # Calculates derivatives necessary for XC matrix
    df_dn_X_alpha, df_ds_X_alpha, df_dt_X_alpha, e_X_alpha = exchange_functional(alpha_density_scaled, sigma_aa_scaled, tau_alpha_scaled, calculation) if exchange_functional is not None else (None, None, None, None)
    df_dn_X_beta, df_ds_X_beta, df_dt_X_beta, e_X_beta = exchange_functional(beta_density_scaled, sigma_bb_scaled, tau_beta_scaled, calculation) if exchange_functional is not None else (None, None, None, None)
    
    df_dn_C_alpha, df_dn_C_beta, df_ds_C_aa, df_ds_C_bb, df_ds_C_ab, df_dt_C_alpha, df_dt_C_beta, e_C = correlation_functional(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation) if correlation_functional is not None else (None, None, None, None, None, None, None, None)

    # This is for the spin scaling for exchange
    df_ds_X_alpha_scaled, df_ds_X_beta_scaled = (df_ds_X_alpha * 2, df_ds_X_beta * 2) if df_ds_X_alpha is not None else (None, None)

    # Builds the alpha and beta exchange matrices - factor of two accounts for spin scaling
    V_X_alpha = dft.calculate_V_X(weights, bfs_on_grid, df_dn_X_alpha, df_ds_X_alpha_scaled, df_dt_X_alpha, bf_gradients_on_grid, density_gradient_alpha) if df_dn_X_alpha is not None else np.zeros_like(P_alpha)
    V_X_beta = dft.calculate_V_X(weights, bfs_on_grid, df_dn_X_beta, df_ds_X_beta_scaled, df_dt_X_beta, bf_gradients_on_grid, density_gradient_beta) if df_dn_X_beta is not None else np.zeros_like(P_beta)

    # Builds the alpha and beta correlation matrices
    V_C_alpha = dft.calculate_V_C(weights, bfs_on_grid, df_dn_C_alpha, df_ds_C_aa, df_dt_C_alpha, bf_gradients_on_grid, density_gradient_alpha, density_gradient_other_spin=density_gradient_beta, df_ds_ab=df_ds_C_ab) if df_dn_C_alpha is not None else np.zeros_like(P_alpha)
    V_C_beta = dft.calculate_V_C(weights, bfs_on_grid, df_dn_C_beta, df_ds_C_bb, df_dt_C_beta, bf_gradients_on_grid, density_gradient_beta, density_gradient_other_spin=density_gradient_alpha, df_ds_ab=df_ds_C_ab) if df_dn_C_beta is not None else np.zeros_like(P_beta)

    # Constructs exchange-correlation matrices considering hybrid functionals
    V_XC_alpha = V_X_alpha * calculation.DFX_prop + V_C_alpha * calculation.DFC_prop
    V_XC_beta = V_X_beta * calculation.DFX_prop + V_C_beta * calculation.DFC_prop


    return V_XC_alpha, V_XC_beta, alpha_density, beta_density, density, e_X_alpha, e_X_beta, e_C 









def apply_damping(P_before_damping, P_old_damp, commutator, calculation, P_old_before_damping, P_very_old_damped, S, partition_ranges, atoms, step):

    """
    
    Applies damping to a density matrix, using the old density matrices.

    Args:
        P_before_damping (array): Density matrix from current iteration before damping
        P_old_damp (array): Density matrix from previous iteration after damping
        commutator (float): RMS([F,PS])
        calculation (Calculation): Calculation object
        P_old_before_damping (array): Density matrix from previous iteration before damping
        P_very_old_damped (array): Density matrix from two iterations ago after damping
        S (array): Overlap matrix in AO basis
        partition_ranges (list): List of number of atomic orbitals on each atom
        atoms (list): List of atoms
    
    Returns:
        P_damped (array): Damped density matrix for current iteration
        damping_factor (float): Damping factor, between zero and one
    
    """

    damping_factor = 0


    def calculate_gross_Mulliken_atomic_population(P):

        """
        
        Calculates the Mulliken gross atomic populations for a given density.
        
        """

        populations_Mulliken = [0, 0]
        PS = P @ S

        for atom in range(len(atoms)):

                # Sets up the lists for atomic_ranges
                if atom == 0: atomic_ranges = list(range(partition_ranges[0]))
                elif atom == 1: atomic_ranges = list(range(partition_ranges[0], partition_ranges[0] + partition_ranges[1]))

                for i in atomic_ranges:
                    
                    populations_Mulliken[atom] += PS[i, i]

        populations_Mulliken = np.array(populations_Mulliken)

        return populations_Mulliken
    

    # Only runs if NODAMP is not used
    if calculation.damping:
        
        # Runs if a damping value is specified
        if calculation.damping_factor != None: 

            try:
                
                # Tries to convert damping factor to a float
                damping_factor = float(calculation.damping_factor)

            except ValueError: pass

        else:

            if commutator > 0.01 and step > 1: 
                
                # Equations taken from Zerner and Hehenberger paper
                A_n_out = calculate_gross_Mulliken_atomic_population(P_before_damping)
                A_n1_in = calculate_gross_Mulliken_atomic_population(P_old_damp)
                A_n1_out = calculate_gross_Mulliken_atomic_population(P_old_before_damping)
                A_n2_in = calculate_gross_Mulliken_atomic_population(P_very_old_damped)
            
                denominator = A_n_out - A_n1_out - A_n1_in + A_n2_in

                alpha = (A_n_out - A_n1_out) / denominator if denominator.all() != 0 else [0, 0]

                if len(partition_ranges) == 2: damping_factor = (alpha[0] * partition_ranges[0] + alpha[1] * partition_ranges[1]) / (partition_ranges[0] + partition_ranges[1])
                else: damping_factor = alpha[0] * partition_ranges[0]

                # Clips the damping factor to be 0 if its negative
                damping_factor = max(damping_factor, 0)
                
                # Damping will never exceed this value
                damping_factor = damping_factor if damping_factor < min(calculation.max_damping, 1) else calculation.max_damping


    # Mixes old density with new, in proportion of damping factor
    P_damped = damping_factor * P_old_damp + (1 - damping_factor) * P_before_damping

    return P_damped, damping_factor
        









def calculate_DIIS_error(F_alpha, F_beta, P_alpha, P_beta, S, X, DIIS_error_vector, Fock_vector, calculation):

    """
    
    Calculates the root-mean-square commutator, [F, PS] for the Fock matrices, and updates the Fock and error vectors.

    Args:
        F_alpha (array): Alpha Fock matrix in AO basis
        F_beta (array): Beta Fock matrix in AO basis
        P_alpha (array): Alpha density matrix
        P_beta (array): Beta density matrix
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix
        DIIS_error_vector (array): Error vector for DIIS
        Fock_vector (array): Fock vector for DIIS
        calculation (Calculation): Calculation object
    
    Returns:
        commutator (float): Root-mean-square commutator, [F, PS]
        Fock_vector (array): Updated Fock vector
        DIIS_error_vector (array): Error vector for DIIS
        commutator_alpha (float): Root-mean-square commutator, [F_alpha, PS]
        commutator_beta (float): Root-mean-square commutator, [F_beta, PS]

    """


    def calculate_commutator(F, P):

        # Calculates the root mean square commutator 
        DIIS_error = F @ P @ S - S @ P @ F

        # Uses Fock transformation matrix to orthogonalise the DIIS error matrix
        orthogonalised_DIIS_error = X.T @ DIIS_error @ X

        # Root-mean-square of the orthogonalised error matrix
        commutator = np.mean(orthogonalised_DIIS_error * orthogonalised_DIIS_error) ** (1 / 2)

        return commutator, orthogonalised_DIIS_error

    # Calculates commutator for both spin channels
    commutator_alpha, orthogonalised_DIIS_error_alpha = calculate_commutator(F_alpha, P_alpha)
    commutator_beta, orthogonalised_DIIS_error_beta = calculate_commutator(F_beta, P_beta)

    # Commutator for convergence checking is largest of the two
    commutator = max(commutator_alpha, commutator_beta)

    # Updates DIIS error vector
    e_combined = np.concatenate((orthogonalised_DIIS_error_alpha.flatten(), orthogonalised_DIIS_error_beta.flatten()))
    DIIS_error_vector.append(e_combined)

    # Updates Fock vector
    Fock_vector.append((F_alpha, F_beta))
    
    # Clears old Fock matrices if Fock vector is too long
    if len(Fock_vector) > calculation.max_DIIS_matrices: 
        
        del Fock_vector[0]
        del DIIS_error_vector[0]


    return commutator, Fock_vector, DIIS_error_vector, commutator_alpha, commutator_beta








def apply_DIIS(commutator, step, P, P_alpha, P_beta, Fock_vector, DIIS_error_vector, n_alpha, n_beta, X, n_electrons_per_orbital, calculation, silent=False):
    
    """
    
    Sets up and solves the linear system of equations for DIIS to update the density matrices.

    Args:
        commutator (float): Root-mean-square [F, PS]
        P (array): Density matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        Fock_vector (array): Fock vector
        DIIS_error_vector (array): DIIS error vector
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        X (array): Fock transformation matrix
        n_electrons_per_orbital (int): One for UHF, two for RHF
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        P (array): Updated density matrix in AO basis
        Fock_vector (array): Updated Fock vector
        DIIS_error_vector (array): Updated DIIS error vector
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis

    """
    
    # Updates density matrix from DIIS extrapolated Fock matrix, applies it if the equations were solved successfully
    if step > 2 and calculation.DIIS and commutator < 0.3: 

        n_DIIS = len(DIIS_error_vector)
        
        # Convert list of DIIS error vectors to a 2D NumPy array for efficient computation
        DIIS_errors = np.array(DIIS_error_vector) 

        # Build the B matrix 
        B = np.empty((n_DIIS + 1, n_DIIS + 1))
        B[:n_DIIS, :n_DIIS] = DIIS_errors @ DIIS_errors.T 
        B[:n_DIIS, -1] = -1
        B[-1, :n_DIIS] = -1
        B[-1, -1] = 0

        # Right-hand side of the linear equations
        rhs = np.zeros(n_DIIS + 1)
        rhs[-1] = -1

        try:

            coeffs = np.linalg.solve(B, rhs)[:n_DIIS]  # Exclude the last coefficient which is for the constraint

            # Convert Fock_vector to separate alpha and beta lists
            F_alpha_list = np.array([fock[0] for fock in Fock_vector]) 
            F_beta_list = np.array([fock[1] for fock in Fock_vector]) 

            # Extrapolate Fock matrices for both alpha and beta spins using matrix multiplication
            F_alpha_DIIS = np.tensordot(coeffs, F_alpha_list, axes=(0, 0))
            F_beta_DIIS = np.tensordot(coeffs, F_beta_list, axes=(0, 0))

            # Diagonalize the extrapolated Fock matrices
            _, molecular_orbitals_alpha_DIIS = diagonalise_Fock_matrix(F_alpha_DIIS, X)
            _, molecular_orbitals_beta_DIIS = diagonalise_Fock_matrix(F_beta_DIIS, X)

            # Construct new density matrices
            P_alpha_DIIS = construct_density_matrix(molecular_orbitals_alpha_DIIS, n_alpha, n_electrons_per_orbital)
            P_beta_DIIS = construct_density_matrix(molecular_orbitals_beta_DIIS, n_beta, n_electrons_per_orbital)

        except np.linalg.LinAlgError:
            
            # Reset DIIS if equations cannot be solved
            Fock_vector.clear()
            DIIS_error_vector.clear()

            P_alpha_DIIS = None
            P_beta_DIIS = None

            log("\n                                       ~~~~~~ Resetting DIIS ~~~~~~", calculation, end="\n\n",silent=silent)

        if P_alpha_DIIS is not None and P_beta_DIIS is not None: 
            
            # If the DIIS equations were solved correctly, update the density matrices
            P_alpha = symmetrise(P_alpha_DIIS)
            P_beta = symmetrise(P_beta_DIIS)

            # Symmetrises the total density matrix
            P = symmetrise(P_alpha + P_beta) / 2
    
    return P, Fock_vector, DIIS_error_vector, P_alpha, P_beta








def run_restricted_SCF_cycle(step, E, P, P_old, P_before_damping, DIIS_error_vector, Fock_vector, calculation, molecule, T, V_NE, ERI_AO, S, X, n_doubly_occ, silent, bfs_on_grid, bf_gradients_on_grid, exchange_functional, correlation_functional, weights):

    """

    Performs the calculations for a single iteration of the restricted SCF loop.

    Args:
        step (int): Cycle step
        E (float): Electronic energy
        P (array): Density matrix in AO basis
        P_old (array): Density matrix from previous step
        P_before_damping (array): Density matrix before damping is applied
        DIIS_error_vector (array): Error vector for DIIS
        Fock_vector (array): Vector of Fock matrices for DIIS
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron energy matrix in AO basis
        ERI_AO (array): Two electron integrals in AO basis
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix in AO basis
        n_doubly_occ (int): Number of doubly occupied orbitals
        silent (bool): Should anything be printed
        bfs_on_grid (array): Basis functions evaluated on grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on grid
        exchange_functional (function): Exchange functional
        correlation_functional (function): Correlation functional
        weights (array): Integration weights
    
    Returns:
        E (float): Updated electronic energy
        E_old (float): Updated electronic energy from previous iteration
        P (array): Updated density matrix
        P_old (array): Updated density matrix from previous iteration
        commutator (array): DIIS error measure, rms[F, PS]
        damping_factor (float): Damping factor
        molecular_orbitals (array): Molecular orbitals
        epsilons (array): Molecular orbital eigenvalues
        energy_components (tuple): Components of electronic energy
        F (array): Fock matrix
        density (array): Electron density on grid

    """


    E_old = E
    P_very_old = P_old
    P_old_before_damping = P_before_damping
    P_old = P 

    # If a DFT calculation is requested, builds the exchange-correlation matrix
    V_XC, density, e_X, e_C = calculate_restricted_exchange_correlation_matrix(P, bfs_on_grid, bf_gradients_on_grid, weights, calculation, exchange_functional, correlation_functional) if calculation.DFT_calculation else (None, None, None, None)

    # Constructs the Fock matrix
    F, J, K = construct_restricted_Fock_matrix(T, V_NE, ERI_AO, P, calculation.HFX_prop, V_XC)

    # Calculates the DIIS error and updates the Fock and error vectors
    commutator, Fock_vector, DIIS_error_vector, _, _ = calculate_DIIS_error(F, F, P, P, S, X, DIIS_error_vector, Fock_vector, calculation)

    # Diagonalises Fock matrix
    epsilons, molecular_orbitals = diagonalise_Fock_matrix(F, X)
    
    # Constructs density matrix
    P = construct_density_matrix(molecular_orbitals, n_doubly_occ, n_electrons_per_orbital=2)

    # Calculates components of electronic energy
    E, energy_components = calculate_restricted_electronic_energy(T, V_NE, P, J, K, calculation, density, weights, e_X, e_C)    
    
    # Applies DIIS to calculate a new density matrix
    P, Fock_vector, DIIS_error_vector, _, _ = apply_DIIS(commutator, step, P, P/2, P/2,Fock_vector, DIIS_error_vector, n_doubly_occ, n_doubly_occ, X, 2, calculation, silent=silent)

    P_before_damping = P

    # Damping factor is applied to the density matrix
    P, damping_factor = apply_damping(P, P_old, commutator, calculation, P_old_before_damping, P_very_old, S, molecule.partition_ranges, molecule.atoms, step)
    

    return E, E_old, P, P_old, commutator, damping_factor, molecular_orbitals, epsilons, energy_components, F, density







def run_unrestricted_SCF_cycle(step, E, P_alpha, P_old_alpha, P_beta, P_old_beta, P, P_old, P_before_damping_alpha, P_before_damping_beta, DIIS_error_vector, Fock_vector, calculation, molecule, T, V_NE, ERI_AO, S, X, n_alpha, n_beta, silent, bfs_on_grid, bf_gradients_on_grid, exchange_functional, correlation_functional, weights):
    
    """

    Performs the calculations for a single iteration of the restricted SCF loop.

    Args:
        step (int): Cycle step
        E (float): Electronic energy
        P_alpha (array): Alpha density matrix in AO basis
        P_old_alpha (array): Alpha density matrix from previous step
        P_beta (array): Beta density matrix in AO basis
        P_old_beta (array): Beta density matrix from previous step
        P (array): Density matrix in AO basis
        P_old (array): Density matrix from previous step
        P_before_damping_alpha (array): Alpha density matrix before damping is applied
        P_before_damping_beta (array): Beta density matrix before damping is applied
        DIIS_error_vector (array): Error vector for DIIS
        Fock_vector (array): Vector of Fock matrices for DIIS
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron energy matrix in AO basis
        ERI_AO (array): Two electron integrals in AO basis
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix in AO basis
        n_alpha (int): Number of alpha orbitals
        n_beta (int): Number of beta orbitals
        silent (bool): Should anything be printed
        bfs_on_grid (array): Basis functions evaluated on grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on grid
        exchange_functional (function): Exchange functional
        correlation_functional (function): Correlation functional
        weights (array): Integration weights
    
    Returns:
        E (float): Updated electronic energy
        E_old (float): Updated electronic energy from previous iteration
        P (array): Updated density matrix
        P_old (array): Updated density matrix from previous iteration
        P_alpha (array): Updated alpha density matrix
        P_beta (array): Updated beta density matrix
        commutator (array): DIIS error measure, rms[F, PS]
        damping_factor (float): Damping factor
        molecular_orbitals_alpha (array): Alpha molecular orbitals
        molecular_orbitals_beta (array): Beta molecular orbitals
        epsilons_alpha (array): Alpha molecular orbital eigenvalues
        epsilons_beta (array): Beta molecular orbital eigenvalues
        energy_components (tuple): Components of electronic energy
        F_alpha (array): Alpha Fock matrix
        F_beta (array): Beta Fock matrix
        alpha_density (array): Alpha electron density on grid
        beta_density (array): Beta electron density on grid
        density (array): Electron density on grid

    """


    E_old = E

    P_very_old_alpha = P_old_alpha
    P_very_old_beta = P_old_beta
    
    P_old_before_damping_alpha = P_before_damping_alpha
    P_old_before_damping_beta = P_before_damping_beta

    P_old = P

    P_old_alpha = P_alpha
    P_old_beta = P_beta
    
    # If a DFT calculation is requested, builds the exchange-correlation matrices
    V_XC_alpha, V_XC_beta, alpha_density, beta_density, density, e_X_alpha, e_X_beta, e_C = calculate_unrestricted_exchange_correlation_matrix(P_alpha, P_beta, bfs_on_grid, bf_gradients_on_grid, weights, calculation, exchange_functional, correlation_functional) if calculation.DFT_calculation else (None, None, None, None, None, None, None, None)
    
    # Constructs the Fock matrices
    F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta = construct_unrestricted_Fock_matrices(T, V_NE, ERI_AO, P_alpha, P_beta, calculation.HFX_prop, V_XC_alpha, V_XC_beta)
    
    # Calculates the DIIS error and updates the Fock and error vectors
    commutator, Fock_vector, DIIS_error_vector, commutator_alpha, commutator_beta = calculate_DIIS_error(F_alpha, F_beta, P_alpha, P_beta, S, X, DIIS_error_vector, Fock_vector, calculation)

    # Diagonalises Fock matrices 
    epsilons_alpha, molecular_orbitals_alpha = diagonalise_Fock_matrix(F_alpha, X)
    epsilons_beta, molecular_orbitals_beta = diagonalise_Fock_matrix(F_beta, X)

    # Constructs density matrices
    P_alpha = construct_density_matrix(molecular_orbitals_alpha, n_alpha, n_electrons_per_orbital=1)
    P_beta = construct_density_matrix(molecular_orbitals_beta, n_beta, n_electrons_per_orbital=1)

    # Calculates components of electronic energy
    E, energy_components = calculate_unrestricted_electronic_energy(T, V_NE, P_alpha, P_beta, J_alpha, J_beta, K_alpha, K_beta, calculation, alpha_density, beta_density, weights, e_X_alpha, e_X_beta, e_C)
    
    # Applies DIIS to calculate new density matrices
    _, Fock_vector, DIIS_error_vector, P_alpha, P_beta = apply_DIIS(commutator, step, P, P_alpha, P_beta, Fock_vector, DIIS_error_vector, n_alpha, n_beta, X, 1, calculation, silent=silent)

    P_before_damping_alpha = P_alpha
    P_before_damping_beta = P_beta

    # Damping factor is applied to the density matrix
    P_alpha, damping_factor_alpha = apply_damping(P_alpha, P_old_alpha, commutator_alpha, calculation, P_old_before_damping_alpha, P_very_old_alpha, S, molecule.partition_ranges, molecule.atoms, step)
    P_beta, damping_factor_beta = apply_damping(P_beta, P_old_beta, commutator_beta, calculation, P_old_before_damping_beta, P_very_old_beta, S, molecule.partition_ranges, molecule.atoms, step)

    P = P_alpha + P_beta

    # For printing only, the damping factor is the largest used of alpha and beta streams
    damping_factor = max(damping_factor_alpha, damping_factor_beta)


    return E, E_old, P, P_old, P_alpha, P_beta, commutator, damping_factor, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons_alpha, epsilons_beta, energy_components, F_alpha, F_beta, alpha_density, beta_density, density









def run_self_consistent_field_cycle(molecule, calculation, T, V_NE, ERI_AO, V_NN, S, X, E, P, P_alpha, P_beta, bfs_on_grid, bf_gradients_on_grid, weights, silent):

    """

    Runs the self-consistent field cycle outer loop.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        V_NN (float): Nuclear-nuclear repulsion energy
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix
        E (float): Guess energy
        P (array): Guess density matrix in AO basis
        P_alpha (array): Guess alpha density matrix in AO basis
        P_beta (array): Guess beta density matrix in AO basis
        bfs_on_grid (array): Basis functions evaluated on integration grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on integration grid
        weights (array): Integration weights
        silent (bool): Should anything be printed

    Returns:
        SCF_output (Output): Output object containing converged SCF data

    """


    log_big_spacer(calculation, silent=silent)
    log("                                   Self-consistent Field Cycle Iterations", calculation, 1, silent=silent, colour="white")
    log_big_spacer(calculation, silent=silent)
    log("  Step          E                DE              RMS(DP)          MAX(DP)           Error       Damping", calculation, 1, silent=silent)
    log_big_spacer(calculation, silent=silent)


    # Unpacks useful calculation properties
    reference = calculation.reference

    if calculation.DFT_calculation:

        # The method strings from the functional definitions
        exchange_method = DFT_methods.get(calculation.method).x_functional
        correlation_method = DFT_methods.get(calculation.method).c_functional

        # The functions from the method strings
        exchange_functional = dft.exchange_functionals.get(exchange_method)
        correlation_functional = dft.correlation_functionals.get(correlation_method)

    else:
        
        exchange_functional, correlation_functional = None, None

    # Initialises various density matrices
    P_old, P_old_alpha, P_old_beta, P_before_damping, P_before_damping_alpha, P_before_damping_beta = np.zeros_like(P), np.zeros_like(P), np.zeros_like(P), np.zeros_like(P), np.zeros_like(P), np.zeros_like(P)

    commutator = 1

    # Initialises vectors for DIIS
    Fock_vector = []
    DIIS_error_vector = []

    for step in range(1, calculation.max_iter):
        
        if reference == "RHF":
            
            # Runs an SCF step
            E, E_old, P, P_old, commutator, damping_factor, molecular_orbitals, epsilons, energy_components, F, density = run_restricted_SCF_cycle(step, E, P, P_old, P_before_damping, DIIS_error_vector, Fock_vector, calculation, molecule, T, V_NE, ERI_AO, S, X, molecule.n_doubly_occ, silent, bfs_on_grid, bf_gradients_on_grid, exchange_functional, correlation_functional, weights)

            # For restricted references, the spin channels are just half the full matrices
            P_alpha = P_beta = P / 2
            F_alpha = F_beta = F / 2

            # Molecular orbitals and eigenvalues are all the same for alpha and beta spins
            molecular_orbitals_alpha = molecular_orbitals_beta = molecular_orbitals
            epsilons_alpha = epsilons_beta = epsilons

            # Alpha and beta density add up to the total density with equal weights
            alpha_density = beta_density = density / 2 if density is not None else None


        elif reference == "UHF":

            # Runs an SCF step
            E, E_old, P, P_old, P_alpha, P_beta, commutator, damping_factor, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons_alpha, epsilons_beta, energy_components, F_alpha, F_beta, alpha_density, beta_density, density = run_unrestricted_SCF_cycle(step, E, P_alpha, P_old_alpha, P_beta, P_old_beta, P, P_old, P_before_damping_alpha, P_before_damping_beta, DIIS_error_vector, Fock_vector, calculation, molecule, T, V_NE, ERI_AO, S, X, molecule.n_alpha, molecule.n_beta, silent, bfs_on_grid, bf_gradients_on_grid, exchange_functional, correlation_functional, weights)

            # Combines the molecular orbitals and eigenvalues into one array
            epsilons_combined = np.concatenate((epsilons_alpha, epsilons_beta)) if molecule.n_electrons > 1 else epsilons_alpha
            molecular_orbitals_combined = np.concatenate((molecular_orbitals_alpha, molecular_orbitals_beta), axis=1) if molecule.n_electrons > 1 else molecular_orbitals_alpha

            epsilons = epsilons_combined[np.argsort(epsilons_combined)]
            molecular_orbitals = molecular_orbitals_combined[:, np.argsort(epsilons_combined)]

        # Calculates the changes in energy and density
        delta_E, maxDP, rmsDP = calculate_SCF_changes(E, E_old, P, P_old)

        # Energy is sum of electronic and nuclear energies
        E_total = E + V_NN  

        # Data outputted to console
        format_output_line(E_total, delta_E, maxDP, rmsDP, damping_factor, step, commutator, calculation, silent=silent)

        # Check for convergence of energy and density
        if check_convergence(calculation.SCF_conv, step, delta_E, maxDP, rmsDP, commutator, calculation, silent=silent): 
            
            kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, correlation_energy = energy_components

            # Builds SCF Output object with useful quantities
            SCF_output = Output(E_total, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, correlation_energy, F_alpha=F_alpha, F_beta=F_beta, density=density, alpha_density=alpha_density, beta_density=beta_density)

            return SCF_output

            
    error(f"Self-consistent field not converged in {calculation.max_iter - 1} iterations! Increase maximum iterations or give up.")

