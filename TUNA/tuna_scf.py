import numpy as np
from tuna_util import *


def format_output_line(E, delta_E, max_DP, RMS_DP, damping_factor, step, orbital_gradient, calculation, silent=False):

    """

    Formats and logs SCF details.

    Args:
        E (float): Energy
        DE (float): Energy change
        maxDP (float): Maximum change in density matrix
        rmsDP (float): Root-mean-square change in density matrix
        damping_factor (float): Damping factor
        step (int): Iteration of SCF
        orbital_gradient (float): Root-mean-square orbital gradient
        calculation (Calculation): Calculation object
        silent (bool, optional): Not silent by default

    Returns:
        None: This function logs output but does not return a value.

    """
    
    damping_factor = f"{damping_factor:.3f}" if damping_factor != 0 else " ---"

    log(f"  {step:3.0f}  {E:16.10f}  {delta_E:16.10f} {RMS_DP:16.10f} {max_DP:16.10f} {orbital_gradient:16.10f}     {damping_factor}", calculation, 1, silent=silent)   





def construct_density_matrix(molecular_orbitals, n_occ, n_electrons_per_orbital):

    """

    Slices out occupied molecular orbitals to build one-particle reduced density matrix.

    Args:
        molecular_orbitals (array): Molecular orbitals in AO basis
        n_occ (int): Number of occupied molecular orbitals
        n_electrons_per_orbital (int): Number of electrons per orbital (RHF or UHF)

    Returns:
        P (array): One-particle reduced density matrix in AO basis

    """

    occupied_mos = molecular_orbitals[:, :n_occ]
    P = n_electrons_per_orbital * occupied_mos @ occupied_mos.T

    return P
    




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

    F_orthonormal = X.T @ F @ X
    
    epsilons, eigenvectors = np.linalg.eigh(F_orthonormal)

    # Transforms molecular orbitals back to non-orthogonal AO basis
    molecular_orbitals = X @ eigenvectors

    return epsilons, molecular_orbitals





def calculate_RHF_electronic_energy(P, H_core, G):

    """

    Forms Fock matrix and calculates RHF electronic energy.

    Args:
        P (array): Density matrix in AO basis
        H_core (array): Core Hamiltonian matrix in AO basis
        G (array): Two-electron part of Fock matrix

    Returns:
        electronic_energy (float): Electronic energy

    """

    F = H_core + G

    electronic_energy = calculate_one_electron_property(0.5 * P, F)
    
    return electronic_energy





def calculate_UHF_electronic_energy(P_alpha, P_beta, H_Core, F_alpha, F_beta):

    """

    Calculates UHF electronic energy.

    Args:
        P_alpha (array): Density matrix for alpha orbitals in AO basis
        P_beta (array): Density matrix for beta orbitals in AO basis
        H_core (array): Core Hamiltonian matrix in AO basis
        F_alpha (array): Fock matrix for alpha orbitals in AO basis
        F_beta (array): Fock matrix for beta orbitals in AO basis

    Returns:
        electronic_energy (float): Electronic energy

    """

    electronic_energy = 0.5 * (calculate_one_electron_property(P_alpha + P_beta, H_Core) + calculate_one_electron_property(P_alpha, F_alpha) + calculate_one_electron_property(P_beta, F_beta))
    
    return electronic_energy





def calculate_energy_components(P_alpha, P_beta, T, V_NE, J_alpha, J_beta, K_alpha, K_beta, P, J, K, reference):
    
    """

    Calculates UHF electronic energy components.

    Args:
        P_alpha (array): Density matrix for alpha orbitals in AO basis
        P_beta (array): Density matrix for beta orbitals in AO basis
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        J_alpha (array): Coulomb matrix for alpha orbitals in AO basis
        J_beta (array): Coulomb matrix for beta orbitals in AO basis
        K_alpha (array): Exchange matrix for alpha orbitals in AO basis
        K_beta (array): Exchange matrix for beta orbitals in AO basis
        P (array): Density matrix in AO basis
        J (array): Coulomb matrix in AO basis
        K (array): Exchange matrix in AO basis
        reference (string): Either UHF or RHF

    Returns:
        kinetic_energy (float): Kinetic energy
        nuclear_electron_energy (float): Nuclear-electron energy
        coulomb_energy (float): Coulomb energy
        exchange_energy (float): Exchange energy

    """

    kinetic_energy = calculate_one_electron_property(P, T)
    nuclear_electron_energy = calculate_one_electron_property(P, V_NE)

    if reference == "RHF":
            
        coulomb_energy = 0.5 * calculate_one_electron_property(P, J)
        exchange_energy = -0.5 * calculate_one_electron_property(P, K / 2)

    elif reference == "UHF":

        coulomb_energy = 0.5 * calculate_one_electron_property(P, J_alpha + J_beta)
        exchange_energy = -0.5 * calculate_one_electron_property(P_alpha, K_alpha) - 0.5 * calculate_one_electron_property(P_beta, K_beta)

    return kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy





def calculate_SCF_changes(E, E_old, P, P_old):

    """

    Calculates changes to energy and density matrix.

    Args:
        E (float): Energy
        E_old (float): Energy of last iteration
        P (array): Density matrix in AO basis
        P_old (array): Density matrix in AO basis of last iteration

    Returns:
        delta_E (float): Change in energy
        maxDP (float): Maximum absolute change in the density matrix
        rmsDP (float): Root-mean-square change in the density matrix

    """

    delta_E = E - E_old
    delta_P = P - P_old
    
    max_DP = np.max(np.abs(delta_P))
    RMS_DP = np.sqrt(np.mean(delta_P ** 2))

    return delta_E, max_DP, RMS_DP





def construct_RHF_Fock_matrix(H_core, ERI_AO, P):

    """

    Constructs RHF Fock matrix from density matrix and two-electron integrals.

    Args:
        H_core (array): Core Hamiltonian matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        P (array): Density matrix in AO basis

    Returns:
        F (array): RHF Fock matrix in AO basis
        J (array): Coulomb matrix in AO basis
        K (array): Exchange matrix in AO basis

    """


    J = np.einsum('ijkl,kl->ij', ERI_AO, P, optimize=True)
    K = np.einsum('ilkj,kl->ij', ERI_AO, P, optimize=True)

    # Two-electron part of Fock matrix   
    G = J - 0.5 * K
    
    F = H_core + G
    
    return F, J, K
    



def construct_UHF_Fock_matrices(H_core, ERI_AO, P_alpha, P_beta):

    """

    Constructs UHF Fock matrix from density matrices and two-electron integrals.

    Args:
        H_core (array): Core Hamiltonian matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        P_alpha (array): Density matrix for alpha orbitals in AO basis
        P_beta (array): Density matrix for beta orbitals in AO basis

    Returns:
        F_alpha (array): RHF Fock matrix for alpha orbitals in AO basis
        F_beta (array): RHF Fock matrix for beta orbitals in AO basis
        J_alpha (array): Coulomb matrix for alpha orbitals in AO basis
        J_beta (array): Coulomb matrix for beta orbitals in AO basis
        K_alpha (array): Exchange matrix for alpha orbitals in AO basis
        K_beta (array): Exchange matrix for beta orbitals in AO basis

    """

    J_alpha = np.einsum('ijkl,kl->ij', ERI_AO, P_alpha, optimize=True)
    J_beta = np.einsum('ijkl,kl->ij', ERI_AO, P_beta, optimize=True)

    K_alpha = np.einsum('ilkj,kl->ij', ERI_AO, P_alpha, optimize=True)
    K_beta = np.einsum('ilkj,kl->ij', ERI_AO, P_beta, optimize=True)

    # Builds separate Fock matrices for alpha and beta spins
    F_alpha = H_core + (J_alpha + J_beta) - K_alpha
    F_beta = H_core + (J_alpha + J_beta) - K_beta

    return F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta
    
    







def apply_damping(P_before_damping, P_old_damp, orbital_gradient, calculation, P_old_before_damping, P_very_old_damped, S, partition_ranges, atoms):

    """
    
    Applies damping to a density matrix, using the old density matrices.

    Args:
        P_before_damping (array): Density matrix from current iteration before damping
        P_old_damp (array): Density matrix from previous iteration after damping
        orbital_gradient (float): RMS([F,PS])
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
                    
                    populations_Mulliken[atom] += PS[i,i]

        return np.array(populations_Mulliken)
    


    if calculation.damping:

        if calculation.damping_factor != None: 

            try:
                
                # Tries to convert damping factor to a float
                damping_factor = float(calculation.damping_factor)

            except ValueError:

                pass

        else:

            if orbital_gradient > 0.01: 
                
                # Equations taken from Zerner and Hehenberger paper
                A_n_out = calculate_gross_Mulliken_atomic_population(P_before_damping)
                A_n1_in = calculate_gross_Mulliken_atomic_population(P_old_damp)
                A_n1_out = calculate_gross_Mulliken_atomic_population(P_old_before_damping)
                A_n2_in = calculate_gross_Mulliken_atomic_population(P_very_old_damped)
            
                denominator = A_n_out - A_n1_out - A_n1_in + A_n2_in


                alpha = (A_n_out - A_n1_out) / denominator if denominator.all() != 0 else [0, 0]

                if len(partition_ranges) == 2: damping_factor = (alpha[0] * partition_ranges[0] + alpha[1] * partition_ranges[1]) / (partition_ranges[0] + partition_ranges[1])
                else: damping_factor = alpha[0] * partition_ranges[0]

                if damping_factor < 0 or damping_factor > 1: 
                    
                    damping_factor = 0



    # Mixes old density with new, in proportion of damping factor
    P_damped = damping_factor * P_old_damp + (1 - damping_factor) * P_before_damping
    

    return P_damped, damping_factor
        








def apply_level_shift(F, P, calculation):

    """

    Applies level shift to Fock matrix.

    Args:
        F (array): Fock matrix in AO basis
        P (array): Density matrix in AO basis
        calculation (Calculation): Calculation object

    Returns:
        F_level_shifted (array): Level shifted Fock matrix in AO basis

    """

    F_level_shifted = F - calculation.level_shift_parameter * P

    return F_level_shifted
    



def calculate_DIIS_error(F, P, S, X):

    """

    Calculates the DIIS error and the root-mean-square of the orbital gradient.

    Args:
        F (array): Fock matrix in AO basis
        P (array): Density matrix in AO basis
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix in AO basis

    Returns:
        orthogonalised_DIIS_error (array): DIIS error matrix for this F and P
        orbital_gradient (float): Root-mean-square of DIIS error

    """

    # Orthogonalises DIIS error to remove issues from linear dependencies
    DIIS_error = F @ P @ S - S @ P @ F
    orthogonalised_DIIS_error = X.T @ DIIS_error @ X

    orbital_gradient = np.sqrt(np.mean(orthogonalised_DIIS_error ** 2))

    return orthogonalised_DIIS_error, orbital_gradient





def update_DIIS(DIIS_error_vector, Fock_vector, n_alpha, n_beta, X, n_electrons_per_orbital, calculation, silent=False):
    
    """
    Updates the Fock matrices using DIIS extrapolation.

    Args:
        DIIS_error_vector (array): List of DIIS error matrices
        Fock_vector (array): List of tuples containing Fock matrices for alpha and beta spins
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        X (array): Fock transformation orthonormalising matrix in AO basis
        n_electrons_per_orbital (int): Number of electrons per orbital
        calculation (Calculation): Calculation object
        silent (bool, optional): Not silent by default

    Returns:
        P_alpha (array): Updated alpha density matrix
        P_beta (array): Updated beta density matrix
        Fock_vector (array): Updated Fock vector
        DIIS_error_vector (array): Updated DIIS error vectors
    """

    n_DIIS = len(DIIS_error_vector)
    
    # Convert list of DIIS error vectors to a 2D NumPy array for efficient computation
    DIIS_errors = np.array(DIIS_error_vector) 

    # Build the B matrix 
    B = np.empty((n_DIIS + 1, n_DIIS + 1))
    B[:n_DIIS, :n_DIIS] = DIIS_errors @ DIIS_errors.T 
    B[:n_DIIS, -1] = -1
    B[-1, :n_DIIS] = -1
    B[-1, -1] = 0.0

    # Right-hand side of the linear equations
    rhs = np.zeros(n_DIIS + 1)
    rhs[-1] = -1.0

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
        P_alpha = construct_density_matrix(molecular_orbitals_alpha_DIIS, n_alpha, n_electrons_per_orbital)
        P_beta = construct_density_matrix(molecular_orbitals_beta_DIIS, n_beta, n_electrons_per_orbital)

    except np.linalg.LinAlgError:
        
        # Reset DIIS if equations cannot be solved
        Fock_vector.clear()
        DIIS_error_vector.clear()

        P_alpha = None
        P_beta = None

        log("\n                                       ~~~~~~ Resetting DIIS ~~~~~~", calculation, end="\n\n",silent=silent)

    return P_alpha, P_beta, Fock_vector, DIIS_error_vector




def check_convergence(SCF_conv_params, step, delta_E, max_DP, RMS_DP, orbital_gradient, calculation, silent=False):

    """

    Checks the convergence of the SCF loop.

    Args:
        SCF_conv (dict): Dictionary of SCF convergence thresholds
        step (int): Iteration of SCF
        delta_E (float): Change in energy since last step
        max_DP (float): Maximum change in density matrix
        RMS_DP (float): Root-mean-square change in density matrix
        orbital_gradient (float): Orbital gradient
        calculation (Calculation): Calculation object
        silent (bool, optional): Not silent by default

    Returns:
        converged (bool): Checks if the calculation has converged or not

    """
    
    converged = False

    if abs(delta_E) < SCF_conv_params["delta_E"] and abs(max_DP) < SCF_conv_params["max_DP"] and abs(RMS_DP) < SCF_conv_params["RMS_DP"] and abs(orbital_gradient) < SCF_conv_params["orbital_gradient"]: 

        log_big_spacer(calculation, silent=silent)
        log(f"\n Self-consistent field converged in {step} cycles!\n", calculation, 1, silent=silent)

        converged = True

    return converged   




def run_SCF(molecule, calculation, T, V_NE, ERI_AO, V_NN, S, X, E, P=None, P_alpha=None, P_beta=None, silent=False):

    """

    Runs the Hartree-Fock self-consistent field loop until convergence.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron attraction matrix in AO basis
        ERI_AO (array): Electron repulsion integrals in AO basis
        S (array): Overlap matrix in AO basis
        X (array): Fock transformation matrix in AO basis
        E (float): Guess energy
        P (array, optional): Guess density matrix in AO basis
        P_alpha (array, optional): Guess density matrix for alpha orbitals in AO basis
        P_beta (array, optional): Guess density matrix for beta orbitals in AO basis
        silent (bool, optional): Not silent by default

    Returns:
        scf_output (Output): Output object containing all SCF results

    Notes:
        - Either runs the RHF or UHF SCF loop
        - Throws an error if the SCF cycle does not converge in the maximum number of iterations

    """
    log_big_spacer(calculation, silent=silent)
    log("                                   Self-consistent Field Cycle Iterations", calculation, 1, silent=silent, colour="white")
    log_big_spacer(calculation, silent=silent)
    log("  Step          E                DE              RMS(DP)          MAX(DP)           Error       Damping", calculation, 1, silent=silent)
    log_big_spacer(calculation, silent=silent)


    # Build core Hamiltonian and initial guess Fock matrix
    H_core = T + V_NE
    F = H_core

    # Unpacks useful calculation properties
    reference = calculation.reference
    maximum_iterations = calculation.max_iter
    n_electrons_per_orbital = calculation.n_electrons_per_orbital
    level_shift_removed = False

    # Unpacks useful molecular quantities
    n_doubly_occ = molecule.n_doubly_occ

    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta

    P_old = np.zeros_like(P)
    P_old_alpha = np.zeros_like(P)
    P_old_beta = np.zeros_like(P)

    P_before_damping = np.zeros_like(P)
    P_before_damping_alpha = np.zeros_like(P)
    P_before_damping_beta = np.zeros_like(P)

    orbital_gradient = 1

    if reference == "RHF":

        # Initialises vectors for DIIS
        Fock_vector = []
        DIIS_error_vector = []

        for step in range(1, maximum_iterations):
            
            E_old = E
            P_very_old = P_old
            P_old_before_damping = P_before_damping
            P_old = P 
      
            # Constructs Fock matrix
            F, J, K = construct_RHF_Fock_matrix(H_core, ERI_AO, P)

            # Applies level shift
            if calculation.level_shift and not level_shift_removed:

                F = apply_level_shift(F, P, calculation)

                if orbital_gradient < 0.02:

                    level_shift_removed = True
                    log("   (Level Shift Off)", calculation, 1, end="", silent=silent)


            # Calculate DIIS error and append to error vector
            orthogonalised_DIIS_error, orbital_gradient = calculate_DIIS_error(F, P, S, X)

            e_combined = np.concatenate((orthogonalised_DIIS_error.flatten(), orthogonalised_DIIS_error.flatten()))
            DIIS_error_vector.append(e_combined)

            Fock_vector.append((F, F))
            
            # Diagonalises Fock matrix
            epsilons, molecular_orbitals = diagonalise_Fock_matrix(F, X)
            
            # Constructs density matrix
            P = construct_density_matrix(molecular_orbitals, n_doubly_occ, n_electrons_per_orbital)

            # Calculates electronic energy
            E = calculate_RHF_electronic_energy(P, H_core, F)

            # Calculates the changes in energy and density
            delta_E, maxDP, rmsDP = calculate_SCF_changes(E, E_old, P, P_old)

            # Clears old Fock matrices if Fock vector is too old
            if len(Fock_vector) > calculation.max_DIIS_matrices: 
                
                del Fock_vector[0]
                del DIIS_error_vector[0]
  
            # Updates density matrix from DIIS extrapolated Fock matrix, applies it if the equations were solved successfully
            if step > 2 and calculation.DIIS and orbital_gradient < 0.2: 
                
                P_alpha_DIIS, P_beta_DIIS, Fock_vector, DIIS_error_vector = update_DIIS(DIIS_error_vector, Fock_vector, n_alpha, n_beta, X, n_electrons_per_orbital, calculation, silent=silent)
                
                if P_alpha_DIIS is not None and P_beta_DIIS is not None: 
                    
                    P_alpha = P_alpha_DIIS
                    P_beta = P_beta_DIIS

                    P = (P_alpha + P_beta) / 2
                
            # Damping factor is applied to the density matrix
            P_before_damping = P

            P, damping_factor = apply_damping(P, P_old, orbital_gradient, calculation, P_old_before_damping, P_very_old, S, molecule.partition_ranges, molecule.atoms)

            # Energy is sum of electronic and nuclear energies
            E_total = E + V_NN  

            # Data outputted to console
            format_output_line(E_total, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient, calculation, silent=silent)

            # Check for convergence of energy and density
            if check_convergence(calculation.SCF_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, calculation, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_energy_components(None, None, T, V_NE, None, None, None, None, P, J, K, reference)

                SCF_output = Output(E_total, S, P, P / 2, P / 2, molecular_orbitals, molecular_orbitals, molecular_orbitals, epsilons, epsilons, epsilons, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy)
               
                return SCF_output


    elif reference == "UHF":

        # Initialises vectors for DIIS
        Fock_vector = []
        DIIS_error_vector = []

        P = P_alpha + P_beta 

        for step in range(1, maximum_iterations):

            E_old = E

            P_very_old_alpha = P_old_alpha
            P_very_old_beta = P_old_beta
            P_old_before_damping_alpha = P_before_damping_alpha
            P_old_before_damping_beta = P_before_damping_beta

            P_old = P

            P_old_alpha = P_alpha
            P_old_beta = P_beta
            

            # Constructs Fock matrices
            F_alpha, F_beta, J_alpha, J_beta, K_alpha, K_beta = construct_UHF_Fock_matrices(H_core, ERI_AO, P_alpha, P_beta)

            # Apply level shift only once
            if calculation.level_shift and not level_shift_removed:
                
                F_alpha = apply_level_shift(F_alpha, P_alpha, calculation)
                F_beta = apply_level_shift(F_beta, P_beta, calculation)

                if orbital_gradient < 0.02:

                    level_shift_removed = True
                    log("   (Level Shift Off)", calculation, 1, end="", silent=silent)

            # Calculate DIIS error and append to Fock and error vectors
            orthogonalised_DIIS_error_alpha, orbital_gradient_alpha = calculate_DIIS_error(F_alpha, P_alpha, S, X)
            orthogonalised_DIIS_error_beta, orbital_gradient_beta = calculate_DIIS_error(F_beta, P_beta, S, X)

            orbital_gradient = max(orbital_gradient_alpha, orbital_gradient_beta)

            e_combined = np.concatenate((orthogonalised_DIIS_error_alpha.flatten(), orthogonalised_DIIS_error_beta.flatten()))
            DIIS_error_vector.append(e_combined)

            Fock_vector.append((F_alpha, F_beta))

            # Diagonalises Fock matrices 
            epsilons_alpha, molecular_orbitals_alpha = diagonalise_Fock_matrix(F_alpha, X)
            epsilons_beta, molecular_orbitals_beta = diagonalise_Fock_matrix(F_beta, X)

            # Constructs density matrices
            P_alpha = construct_density_matrix(molecular_orbitals_alpha, n_alpha, n_electrons_per_orbital)
            P_beta = construct_density_matrix(molecular_orbitals_beta, n_beta, n_electrons_per_orbital)

            P = P_alpha + P_beta

            # Calculates electronic energy
            E = calculate_UHF_electronic_energy(P_alpha, P_beta, H_core, F_alpha, F_beta)

            # Calculates the changes in energy and density
            delta_E, maxDP, rmsDP = calculate_SCF_changes(E, E_old, P, P_old)

            # Clears old Fock matrices if Fock vector is too old
            if len(Fock_vector) > calculation.max_DIIS_matrices: 
                
                del Fock_vector[0]
                del DIIS_error_vector[0]

            # Update density matrices using DIIS extrapolated Fock matrices
            if step > 2 and calculation.DIIS and orbital_gradient < 0.2: 
                
                P_alpha_DIIS, P_beta_DIIS, Fock_vector, DIIS_error_vector = update_DIIS(DIIS_error_vector, Fock_vector, n_alpha, n_beta, X, n_electrons_per_orbital, calculation, silent=silent)
                
                if P_alpha_DIIS is not None: P_alpha = P_alpha_DIIS
                if P_beta_DIIS is not None: P_beta = P_beta_DIIS


            # Damping factor is applied to the density matrix
            P_before_damping_alpha = P_alpha
            P_before_damping_beta = P_beta

            P_alpha, damping_factor_alpha = apply_damping(P_alpha, P_old_alpha, orbital_gradient_alpha, calculation, P_old_before_damping_alpha, P_very_old_alpha, S, molecule.partition_ranges, molecule.atoms)
            P_beta, damping_factor_beta = apply_damping(P_beta, P_old_beta, orbital_gradient_beta, calculation, P_old_before_damping_beta, P_very_old_beta, S, molecule.partition_ranges, molecule.atoms)

            P = P_alpha + P_beta

            damping_factor = max(damping_factor_alpha, damping_factor_beta)

            # Energy is sum of electronic and nuclear energies
            E_total = E + V_NN  

            # Outputs useful information to console
            format_output_line(E_total, delta_E, maxDP, rmsDP, damping_factor, step, orbital_gradient, calculation, silent=silent)
            

            # Check for convergence of energy and density
            if check_convergence(calculation.SCF_conv, step, delta_E, maxDP, rmsDP, orbital_gradient, calculation, silent=silent): 

                kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = calculate_energy_components(P_alpha, P_beta, T, V_NE, J_alpha, J_beta, K_alpha, K_beta, P, None, None, reference)
                
                epsilons_combined = np.concatenate((epsilons_alpha, epsilons_beta))
                molecular_orbitals_combined = np.concatenate((molecular_orbitals_alpha, molecular_orbitals_beta), axis=1)

                epsilons = epsilons_combined[np.argsort(epsilons_combined)]
                molecular_orbitals = molecular_orbitals_combined[:, np.argsort(epsilons_combined)]

                # Builds SCF Output object with useful quantities
                SCF_output = Output(E_total, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy )

                return SCF_output
            

    error(f"Self-consistent field not converged in {maximum_iterations} iterations! Increase maximum iterations or give up.")

