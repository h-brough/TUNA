
from tuna_util import log, error, Calculation
import tuna_scf as scf
from scipy.linalg import block_diag
from tuna_integrals import tuna_integral as ints
from tuna_molecule import Molecule, Atom
from tuna_mp import calculate_natural_orbitals
import numpy as np
from numpy import ndarray
import sys


"""

This is the TUNA module for determining the initial guess to a self-consistent field calculation, written first for version 0.10.0 of TUNA.

The options for initial guess are (1) core Hamiltonian diagonalisation (one-electron energy), (2) superposition of atomic densities or (3) minimal basis
self-consistent field and projection, which is the best when combined with an initial SAD guess, so is the default. The atomic densities for SAD are stored in 
tuna_util within atomic_properties as spherically averaged minimal basis density matrices, which are projected onto larger basis sets. For unrestricted references, 
the spin-symmetry of the alpha and beta orbitals can be broken for symmetric molecules, or with the ROTATE keyword.

The module contains:

1. A function to mix the HOMO and LUMO by rotation (rotate_molecular_orbitals)
2. Functions used it the SAD guess procedure (form_minimal_superposition_density, break_spin_density_symmetry, etc.)
3. The main core guess and SAD guess functions (calculate_core_guess, calculate_superposition_guess)
4. The main guess function, which calls the relevant method to calculate the initial guess density matrices (setup_initial_guess)

"""



def rotate_molecular_orbitals(molecular_orbitals: ndarray, n_occ: int, theta: float) -> ndarray:
    
    """

    Rotates HOMO and LUMO of molecular orbitals by given angle theta to break the symmetry.

    Args:
        molecular_orbitals (array): Molecular orbital array in AO basis
        n_occ (int): Number of occupied molecular orbitals
        theta (float): Angle in degrees to rotate orbitals

    Returns:
        rotated_molecular_orbitals (array): Molecular orbitals with HOMO and LUMO rotated

    """
    
    # Converts to radians
    theta = np.deg2rad(theta)

    homo_index = n_occ - 1
    lumo_index = n_occ

    rotation_matrix = np.eye(len(molecular_orbitals))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Makes sure there is a HOMO and a LUMO to rotate, builds rotation matrix using sine and cosine of the requested angle, at the HOMO and LUMO indices

    try:
        
        rotation_matrix[homo_index:lumo_index + 1, homo_index:lumo_index + 1] = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    except: 
        
        error("Basis set too small to rotate initial guess orbitals! Use a larger basis or the NOROTATE keyword.")

    # Rotates molecular orbitals with this matrix
    rotated_molecular_orbitals = molecular_orbitals @ rotation_matrix


    return rotated_molecular_orbitals







def form_minimal_basis_superposition_density(atoms: list[Atom]) -> ndarray: 
    
    """

    Forms the superposition of atomic densities guess density matrix.

    Args:
        atoms (list): List of atoms

    Returns:
        P_minimal (array): Superposition of atomic densities guess density matrix in minimal basis

    """

    # The divisor by 2 ensures idempotency - forms block diagonal density
    P_minimal = block_diag(atoms[0].density, atoms[1].density) / 2

    return P_minimal







def build_minimal_basis_molecule(calculation: Calculation, molecule: Molecule, atomic_symbols: list[str]) -> Molecule:

    """
    
    Builds a molecule object with a minimal basis set.
    
    Args:
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        atomic_symbols (list): List of atomic symbols

    Returns:
        molecule_minimal (Molecule): Molecule object

    """

    # Stores the full basis
    old_basis = calculation.basis

    try:

        calculation.basis = "STO-3G"

        # Builds a minimal basis molecule object
        molecule_minimal = Molecule(atomic_symbols, molecule.coordinates, calculation, guess=True)

    finally:

        # Ensures restoration of full basis
        calculation.basis = old_basis
    

    return molecule_minimal







def break_density_spin_symmetry(P: ndarray, X: ndarray, n_occ: int, calculation: Calculation) -> ndarray:

    """
    
    Breaks the spin symmetry of a density matrix, by mixing the HONO and LUNO.
    
    Args:
        P (array): Spin-symmetric density matrix
        X (array): Fock orthogonalisation matrix
        n_occ (int): Number of occupied orbitals
        calculation (Calculation): Calculation object

    Returns:
        P_broken (array): Broken symmetry density matrix

    """

    # Diagonalise the density for the natural orbitals
    _, natural_orbitals = calculate_natural_orbitals(P, X, calculation, silent=True)

    # Rotate the HONO and LUNO by theta degrees
    rotated_orbitals_ortho = rotate_molecular_orbitals(natural_orbitals, n_occ, calculation.theta)

    # Orthogonalises the natural orbitals
    rotated_orbitals = X @ rotated_orbitals_ortho

    # Builds idempotent broken symmetry density matrix
    P_broken = scf.construct_density_matrix(rotated_orbitals, n_occ, 1)

    return P_broken








def project_density_matrix(P_to_project: ndarray, S_cross: ndarray, S_target_inverse: ndarray) -> ndarray:

    """
    
    Projects the input density matrix onto a larger basis set.
    
    Args:
        P_to_project (array): Density matrix to project
        S_cross (array): Overlap matrix between small and large basis sets
        S_target_inverse (int): Inverse overlap matrix of output, larger basis set

    Returns:
        P_target (array): Projected density matrix

    """

    # Projects the input density matrix onto the larger basis set of S inverse
    P_target = np.einsum("ip,pq,qr,sr,sj->ij", S_target_inverse, S_cross, P_to_project, S_cross, S_target_inverse, optimize=True)

    return P_target

 






def calculate_cross_basis_overlap_matrix(molecule_1: Molecule, molecule_2: Molecule) -> ndarray:
    
    """
    
    Calculates the overlap matrix between two different basis sets.
    
    Args:
        molecule_1 (array): Molecule object in one basis set
        molecule_2 (array): Molecule object in another basis set

    Returns:
        S_cross (array): Cross basis overlap matrix

    """

    S_cross = np.zeros((molecule_1.n_basis, molecule_2.n_basis))

    # Calculates full, non-square cross overlap matrix between basis sets

    for i in range(molecule_1.n_basis):

        for j in range(molecule_2.n_basis):

            S_cross[i, j] = ints.S(molecule_1.basis_functions[i], molecule_2.basis_functions[j])
    

    return S_cross







def calculate_energy_guess(H_core: ndarray, X: ndarray) -> float:
       
    """
    
    Calculates the guess energy for a given core Hamiltonian.
    
    Args:
        H_core (array): Core Hamiltonian matrix in AO basis
        X (array): Fock transformation matrix in AO basis

    Returns:
        E_guess (float): Guess energy

    """

    # Diagonalise Fock matrix
    eigenvalues, _ = scf.diagonalise_Fock_matrix(H_core, X)

    # The guess energy is the lowest one-electron eigenvalue in all cases
    E_guess = np.min(eigenvalues)

    return E_guess







def calculate_superposition_guess(S_inverse: ndarray, atomic_symbols: list[str], molecule: Molecule, calculation: Calculation, rotate_guess_mos: bool, X: ndarray, silent=False) -> tuple[ndarray, ndarray, ndarray]:

    """

    Calculates the guess density matrices using superposition of atomic densities.

    Args:
        S_inverse (array): Inverse overlap matrix in AO basis
        atomic_symbols (list): List of atomic symbols
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        rotate_guess_mos (bool): Should the guess orbitals be rotated
        X (array): Fock transformation matrix
        silent (bool, optional): Should anything be printed

    Returns:
        P_guess (array): Superposition of atomic densities total density matrix
        P_alpha (array): Superposition of atomic densities alpha density matrix
        P_beta (array): Superposition of atomic densities beta density matrix

    """

    log("\n Calculating superposition of atomic densities for guess...  ", calculation, end="", silent=silent); sys.stdout.flush()

    # Forms superposition of atomic densities density matrix
    P_minimal = form_minimal_basis_superposition_density(molecule.atoms)

    # Builds minimal basis molecule
    molecule_minimal = build_minimal_basis_molecule(calculation, molecule, atomic_symbols)

    # Forms cross basis overlap matrix between STO-3G and the chosen basis
    S_cross = calculate_cross_basis_overlap_matrix(molecule, molecule_minimal)
    
    # Projects onto the larger basis set
    P_guess_alpha = project_density_matrix(P_minimal, S_cross, S_inverse)
    P_guess_beta = project_density_matrix(P_minimal, S_cross, S_inverse)

    # If necessary, break the spin symmetry
    P_guess_alpha = break_density_spin_symmetry(P_guess_alpha, X, molecule.n_alpha, calculation) if rotate_guess_mos else P_guess_alpha

    # Form total density SAD guess
    P_guess = P_guess_alpha + P_guess_beta

    log("[Done]\n", calculation, silent=silent)


    return P_guess, P_guess_alpha, P_guess_beta








def calculate_core_guess(calculation: Calculation, H_core: ndarray, X: ndarray, molecule: Molecule, rotate_guess_mos: bool, silent=False) -> tuple[ndarray, ndarray, ndarray]:

    """

    Calculates the guess density matrices using one-electron Hamiltonian diagonalisation.

    Args:
        calculation (Calculation): Calculation object
        H_core (array): Core Hamiltonian matrix in AO basis
        X (array): Fock transformation matrix
        molecule (Molecule): Molecule object
        rotate_guess_mos (bool): Should the guess orbitals be rotated
        silent (bool, optional): Should anything be printed

    Returns:
        P_guess (array): Superposition of atomic densities total density matrix
        P_alpha (array): Superposition of atomic densities alpha density matrix
        P_beta (array): Superposition of atomic densities beta density matrix

    """

    log("\n Calculating one-electron density for guess...  ", calculation, end="", silent=silent); sys.stdout.flush()

    # Diagonalise core Hamiltonian for one-electron guess
    _, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)

    # Rotate the alpha MOs if this is requested, otherwise take the alpha guess to equal the beta guess
    guess_mos_alpha = rotate_molecular_orbitals(guess_mos, molecule.n_alpha, calculation.theta) if rotate_guess_mos else guess_mos

    # Construct density matrices (1 electron per orbital) for the alpha and beta guesses
    P_guess_alpha = scf.construct_density_matrix(guess_mos_alpha, molecule.n_alpha, 1)
    P_guess_beta = scf.construct_density_matrix(guess_mos, molecule.n_beta, 1)

    # Add together alpha and beta densities for total density
    P_guess = P_guess_alpha + P_guess_beta

    log("[Done]\n", calculation, silent=silent)


    return P_guess, P_guess_alpha, P_guess_beta








def setup_initial_guess(P_guess: ndarray, P_guess_alpha: ndarray, P_guess_beta: ndarray, E_guess: float, T: ndarray, V_NE: ndarray, X: ndarray, calculation: Calculation, molecule: Molecule, S_inverse: ndarray, atomic_symbols: list[str], silent=False) -> tuple[float, ndarray, ndarray, ndarray]:
    
    """

    Calculates the guess density matrices and guess energy.

    Args:
        P (array): Guess density matrix
        P_alpha (array): Guess alpha density matrix
        P_beta (array): Guess beta density matrix
        E_guess (array): Guess energy
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron energy matrix in AO basis
        X (array): Fock transformation matrix
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        S_inverse (array): Inverse overlap matrix in AO basis
        atomic_symbols (list): List of atomic symbols
        silent (bool, optional): Should anything be printed

    Returns:
        E_guess (float): Guess energy
        P_guess (array): Total guess density matrix
        P_alpha (array): Alpha guess density matrix
        P_beta (array): Beta guess density matrix

    """

    H_core = T + V_NE

    # Only rotate guess MOs if there's an even number of electrons, and it hasn't been overridden by NOROTATE
    rotate_guess_mos = True if molecule.multiplicity == 1 and not calculation.no_rotate_guess and calculation.reference == "UHF" else False 

    if calculation.reference == "RHF" and P_guess is not None:
            
        log("\n Using density matrix from previous step for guess. \n", calculation, 1, silent=silent)


    elif calculation.reference == "UHF" and P_guess_alpha is not None and P_guess_beta is not None: 
        
        log("\n Using density matrices from previous step for guess. \n", calculation, silent=silent)


    else:

        if calculation.core_guess:

            P_guess, P_guess_alpha, P_guess_beta = calculate_core_guess(calculation, H_core, X, molecule, rotate_guess_mos, silent=silent)

        else:

             P_guess, P_guess_alpha, P_guess_beta = calculate_superposition_guess(S_inverse, atomic_symbols, molecule, calculation, rotate_guess_mos, X, silent=silent)


        if rotate_guess_mos: 
            
            log(f" Initial guess density uses molecular orbitals rotated by {calculation.theta:.1f} degrees.\n", calculation, silent=silent)


    # Calculates the guess energy by diagonalisation of the core Hamiltonian
    E_guess = calculate_energy_guess(H_core, X) if E_guess is None else E_guess



    return E_guess, P_guess, P_guess_alpha, P_guess_beta
