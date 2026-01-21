
from tuna_util import *
import tuna_scf as scf


def rotate_molecular_orbitals(molecular_orbitals, n_occ, theta):
    
    """

    Rotates HOMO and LUMO of molecular orbitals by given angle theta to break the symmetry.

    Args:
        molecular_orbitals (array): Molecular orbital array in AO basis
        n_occ (int): Number of occupied molecular orbitals
        theta (float): Angle in radians to rotate orbitals

    Returns:
        rotated_molecular_orbitals (array): Molecular orbitals with HOMO and LUMO rotated

    """

    # Converts to radians
    theta *= np.pi / 180

    homo_index = n_occ - 1
    lumo_index = n_occ

    dimension = len(molecular_orbitals)
    rotation_matrix = np.eye(dimension)

    # Makes sure there is a HOMO and a LUMO to rotate, builds rotation matrix using sine and cosine of the requested angle, at the HOMO and LUMO indices
    try:
        
        rotation_matrix[homo_index:lumo_index + 1, homo_index:lumo_index + 1] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
    
    except: error("Basis set too small to rotate initial guess orbitals! Use a larger basis or the NOROTATE keyword.")

    # Rotates molecular orbitals with this matrix
    rotated_molecular_orbitals = molecular_orbitals @ rotation_matrix


    return rotated_molecular_orbitals









def setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, rotate_guess_mos, no_rotate_guess_mos, calculation, molecule, silent=False):

    """

    Either calculates or passes on the guess energy and density.

    Args:
        P_guess (array): Density matrix from previous step in AO basis
        P_guess_alpha (array): Alpha density matrix from previous step in AO basis
        P_guess_beta (array): Beta density matrix from previous step in AO basis
        E_guess (float): Final energy from previous step
        reference (str): Either RHF or UHF
        T (array): Kinetic energy integral matrix in AO basis
        V_NE (array): Nuclear-electron attraction integral matrix in AO basis
        X (array): Fock transformation matrix
        n_doubly_occ (int): Number of doubly occupied orbitals
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        rotate_guess_mos (bool): Force rotation of guess molecular orbitals
        no_rotate_guess_mos (bool): Force no rotation of guess molecular orbitals
        calculation (Calculation): Calculation object
        silent (bool, optional): Should output be printed

    Returns:
        E_guess (float): Guess energy
        P_guess (array): Guess density matrix in AO basis
        P_guess_alpha (array): Guess alpha density matrix in AO basis
        P_guess_beta (array): Guess beta density matrix in AO basis
        guess_epsilons (array): Guess one-electron Fock matrix eigenvalues
        guess_mos (array): Guess one-electron Fock matrix eigenvectors

    """
    
    H_core = T + V_NE
    
    guess_epsilons = []
    guess_mos = []


    if reference == "RHF":
        
        # If there's a guess density, just use that
        if P_guess is not None: 
            
            log("\n Using density matrix from previous step for guess. \n", calculation, 1, silent=silent)

        else:
            
            log("\n Calculating one-electron density for guess...  ", calculation, end="", silent=silent); sys.stdout.flush()

            # Diagonalise core Hamiltonian for one-electron guess, then build density matrix (2 electrons per orbital) from these guess molecular orbitals
            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

            P_guess = scf.construct_density_matrix(guess_mos, n_doubly_occ, 2)
            P_guess_alpha = P_guess_beta = P_guess / 2

            # Take lowest energy guess epsilon for guess energy
            E_guess = guess_epsilons[0]       

            log("[Done]\n", calculation, silent=silent)


    elif reference == "UHF":    

        # If there's a guess density, just use that
        if P_guess_alpha is not None and P_guess_beta is not None: log("\n Using density matrices from previous step for guess. \n", calculation, silent=silent)

        else:

            log("\n Calculating one-electron density for guess...   ", calculation, end="", silent=silent)

            # Only rotate guess MOs if there's an even number of electrons, and it hasn't been overridden by NOROTATE
            rotate_guess_mos = True if molecule.multiplicity == 1 and not no_rotate_guess_mos else False

            # Diagonalise core Hamiltonian for one-electron guess
            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

            # Rotate the alpha MOs if this is requested, otherwise take the alpha guess to equal the beta guess
            guess_mos_alpha = rotate_molecular_orbitals(guess_mos, n_alpha, calculation.theta) if rotate_guess_mos else guess_mos

            # Construct density matrices (1 electron per orbital) for the alpha and beta guesses
            P_guess_alpha = scf.construct_density_matrix(guess_mos_alpha, n_alpha, 1)
            P_guess_beta = scf.construct_density_matrix(guess_mos, n_beta, 1)

            # Take lowest energy guess epsilon for guess energy
            E_guess = guess_epsilons[0]

            # Add together alpha and beta densities for total density
            P_guess = P_guess_alpha + P_guess_beta

            log("[Done]\n", calculation, silent=silent)

            if rotate_guess_mos: 
                
                log(f" Initial guess density uses molecular orbitals rotated by {(calculation.theta):.1f} degrees.\n", calculation, silent=silent)


    return E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos



