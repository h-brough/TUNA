import numpy as np
import tuna_scf as scf
import sys
from tuna_util import *
import tuna_integral as integ
import tuna_postscf as postscf
import tuna_mpn as mpn
import tuna_plot as plot
import tuna_ci as ci


def calculate_nuclear_repulsion(charges, coordinates):
    
    """

    Calculates nuclear repulsion energy.

    Args:
        charges (array): Nuclear charges
        coordinates (array): Atomic coordinates

    Returns:
        V_NN (float): Nuclear-nuclear repulsion energy

    """

    V_NN = np.prod(charges) / np.linalg.norm(coordinates[1] - coordinates[0])
    
    return V_NN
    




def calculate_Fock_transformation_matrix(S):

    """

    Diagonalises the overlap matrix to find its square root, then inverts this as X = S^-1/2.

    Args:
        S (array): Overlap matrix in AO basis

    Returns:
        X (array): Fock transformation matrix

    """

    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    X = np.linalg.inv(S_sqrt)

    return X





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




def setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, rotate_guess_mos, no_rotate_guess_mos, calculation, silent=False):

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
        if P_guess is not None: log("\n Using density matrix from previous step for guess. \n", calculation, 1, silent=silent)

        else:
            
            log(" Calculating one-electron density for guess...   ", calculation, end="", silent=silent)

            # Diagonalise core Hamiltonian for one-electron guess, then build density matrix (2 electrons per orbital) from these guess molecular orbitals
            guess_epsilons, guess_mos = scf.diagonalise_Fock_matrix(H_core, X)
            P_guess = scf.construct_density_matrix(guess_mos, n_doubly_occ, 2)

            # Take lowest energy guess epsilon for guess energy
            E_guess = guess_epsilons[0]       

            log("[Done]\n", calculation, silent=silent)


    elif reference == "UHF":    

        # If there's a guess density, just use that
        if P_guess_alpha is not None and P_guess_beta is not None: log("\n Using density matrices from previous step for guess. \n", calculation, silent=silent)

        else:
            
            log(" Calculating one-electron density for guess...   ", calculation, end="", silent=silent)

            # Only rotate guess MOs if there's an even number of electrons, and it hasn't been overridden by NOROTATE
            rotate_guess_mos = True if (n_alpha + n_beta) % 2 == 0 and not no_rotate_guess_mos else False

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
                
                log(" Initial guess density uses rotated molecular orbitals.\n", calculation, silent=silent)


    return E_guess, P_guess, P_guess_alpha, P_guess_beta, guess_epsilons, guess_mos






def calculate_D2_energy(atoms, bond_length):

    """

    Calculates the D2 semi-empirical dispersion energy.

    Args:
        atoms (list): List of atomic symbols
        bond_length (float): Distance between two atoms

    Returns:
        E_D2 (float): D2 semi-empirical dispersion energy

    """

    # These parameters were chosen to match the implementation of Hartree-Fock in ORCA
    s6 = 1.2 
    damping_factor = 20
    
    # Makes sure there are two real atoms, then calculates the D2 dispersion energy from Grimme's equation
    if len(atoms) == 2 and not any("X" in atom for atom in atoms):

        C6 = np.sqrt(constants.atom_properties[atoms[0]]["C6"] * constants.atom_properties[atoms[1]]["C6"])
        vdw_sum = constants.atom_properties[atoms[0]]["vdw_radius"] + constants.atom_properties[atoms[1]]["vdw_radius"]

        f_damp = 1 / (1 + np.exp(-1 * damping_factor * (bond_length / (vdw_sum) - 1)))
        
        # Uses conventional dispersion energy expression, with damping factor to account for short bond lengths
        E_D2 = -1 * s6 * C6 / (bond_length ** 6) * f_damp
        
        return E_D2
        
    return 0






def calculate_one_electron_energy(method, reference, atomic_orbitals, charges, coordinates, centre_of_mass, calculation, silent=False):

    """

    Calculates the energy of a one-electron system.

    Args:
        method (str): Electronic structure method
        reference (str): Either UHF or RHF
        atomic_orbitals (array): Atomic orbitals
        charges (list): Nuclear charges
        coordinates (array): Atomic coordinates in 3D
        centre_of_mass (float): Distance from first atom of centre of mass
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E (float): One-electron energy
        P (array): One-electron density matrix in AO basis
        epsilons (array): Energy levels of one-electron system
        molecular_orbitals (array): Molecular orbitals of one-electron system in AO basis
        D (array): Dipole integral matrix
        S (array): Overlap matrix
        ERI_AO (array): Electron repulsion integrals in AO basis
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron matrix in AO basis

    """

    if method not in ["HF", "RHF", "UHF", "CIS", "UCIS", "CIS[D]", "UCIS[D]"]: error("A correlated calculation has been requested on a one-electron system!")
    elif method in ["CIS", "UCIS", "CIS[D]", "UCIS[D]"]: error("An excited state calculation has been requested on a one-electron system!")

    # Calculates one-electron integrals
    log(" Calculating one-electron integrals...    ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    S, T, V_NE, D, ERI_AO = integ.evaluate_integrals(atomic_orbitals, np.array(charges, dtype=np.float64), coordinates, centre_of_mass, two_electron_ints=False)
    
    log("[Done]", calculation, 1, silent=silent)     

    # Calculates Fock transformation matrix from overlap matrix
    log(" Constructing Fock transformation matrix...  ", calculation, 1, end="", silent=silent)
    
    X = calculate_Fock_transformation_matrix(S)
    
    log("[Done]", calculation, 1, silent=silent)

    # Builds initial guess, which is the final answer for the one-electron case
    E, P, P_alpha, P_beta, epsilons, molecular_orbitals = setup_initial_guess(None, None, None, None, reference, T, V_NE, X, 1, 1, 0, calculation.rotate_guess, calculation.no_rotate_guess, calculation, silent=silent)

    return E, P, epsilons, molecular_orbitals, D, S, ERI_AO, T, V_NE




def calculate_energy(calculation, atoms, coordinates, P_guess=None, P_guess_alpha=None, P_guess_beta=None, E_guess=None, terse=False, silent=False):
 
    """

    Calculates the energy of an atom or molecule, calling various modules. The main function in TUNA.

    Args:
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        coordinates (array): Atomic coordinates in 3D
        P_guess (array, optional): Guess density matrix in AO basis
        P_guess_alpha (array, optional): Guess alpha density matrix in AO basis
        P_guess_beta (array, optional): Guess beta density matrix in AO basis
        E_guess (float, optional): Guess energy
        terse (bool, optional): Should post-SCF output be printed
        silent (bool, optional): Should anything be printed

    Returns:
        SCF_output (Output): Output from SCF calculation
        molecule (Molecule): Molecule object
        final_energy (float): Final energy
        P (array): Final density matrix in AO basis

    """

    log("\n Setting up molecule...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Builds molecule object using calculation and atomic parameters
    molecule = Molecule(atoms, coordinates, calculation)
    
    # Unpacking of various useful calculation quantities
    method = calculation.method
    reference = calculation.reference

    # Unpacking of various useful molecular properties
    atoms = molecule.atoms
    charges = molecule.charges
    coordinates = molecule.coordinates
    bond_length = molecule.bond_length
    centre_of_mass = molecule.centre_of_mass
    atomic_orbitals = molecule.atomic_orbitals
    n_doubly_occ = molecule.n_doubly_occ
    n_occ = molecule.n_occ
    n_SO = molecule.n_SO
    n_virt = molecule.n_virt
    n_electrons = molecule.n_electrons
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta

    log("[Done]\n", calculation, 1, silent=silent)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("    Molecule and Basis Information", calculation, 1, silent=silent, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("  Molecular structure: " + molecule.molecular_structure, calculation, 1, silent=silent)
    log("  Number of atoms: " + str(len(atoms)), calculation, 1, silent=silent)
    log("  Number of basis functions: " + str(len(atomic_orbitals)), calculation, 1, silent=silent)
    log("  Number of primitive Gaussians: " + str(len(molecule.primitive_Gaussians)), calculation, 1, silent=silent)
    log("  Charge: " + str(molecule.charge), calculation, 1, silent=silent)
    log("  Multiplicity: " + str(molecule.multiplicity), calculation, 1, silent=silent)
    log("  Number of electrons: " + str(n_electrons), calculation, 1, silent=silent)
    log("  Number of alpha electrons: " + str(n_alpha), calculation, 1, silent=silent)
    log("  Number of beta electrons: " + str(n_beta), calculation, 1, silent=silent)
    log(f"  Point group: {molecule.point_group}", calculation, 1, silent=silent)
    if len(atoms) == 2: log(f"  Bond length: {bohr_to_angstrom(bond_length):.4f} ", calculation, 1, silent=silent)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1, silent=silent)


    # Nuclear repulsion and dispersion energy are only calculated if there are two real atoms present
    if len(atoms) == 2 and not any("X" in atom for atom in atoms):

        #Calculates nuclear repulsion energy
        log(" Calculating nuclear repulsion energy...  ", calculation, 1, end="", silent=silent)

        V_NN = calculate_nuclear_repulsion(charges, coordinates)

        log(f"[Done]\n\n Nuclear repulsion energy: {V_NN:.10f}\n", calculation, 1, silent=silent)
        
        # Calculates D2 dispersion energy if requested
        if calculation.D2:  

            log(" Calculating semi-empirical dispersion energy...  ", calculation, 1, end="", silent=silent)

            E_D2 = calculate_D2_energy(atoms, bond_length)

            log(f"[Done]\n\n Dispersion energy (D2): {E_D2:.10f}\n", calculation, 1, silent=silent)
            
        else: E_D2 = 0
        
    else: V_NN = 0; E_D2 = 0
        

    if n_electrons < 0: error("Negative number of electrons specified!")

    elif n_electrons == 0: 

        # If zero electrons are specified, the only energy is due to nuclear repulsion, which is printed and the calculation ends
        warning("Calculation specified with zero electrons!\n")
        log(f"Final energy: {V_NN:.10f}", calculation, 1, silent=silent)
        
        finish_calculation(calculation)


    elif n_electrons == 1: 
        
        # Calculates the one-electron energy, exact within the basis set
        E, P, epsilons, molecular_orbitals, D, S, ERI_AO, T, V_NE = calculate_one_electron_energy(method, reference, atomic_orbitals, charges, coordinates, centre_of_mass, calculation, silent=silent)

        J = np.einsum('ijkl,kl->ij', ERI_AO, P, optimize=True)
        K = np.einsum('ilkj,kl->ij', ERI_AO, P, optimize=True)

        final_energy = E + V_NN
        P_alpha = P / 2
        P_beta = P / 2

        kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy = scf.calculate_energy_components(P_alpha, P_beta, T, V_NE, None, None, None, None, P, J, K, "RHF")
        if not silent: postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation)

        # Builds energy output object with all the calculated quantities from the one-electron guess
        SCF_output = Output(final_energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals, None, epsilons, epsilons, None, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy)

        # No beta electrons for the one-electron case exists
        epsilons_alpha = epsilons
        epsilons_beta = None
        molecular_orbitals_alpha = molecular_orbitals
        molecular_orbitals_beta = None
        
        SCF_output.D = D


    elif n_electrons > 1:

        # Calculates one- and two-electron integrals
        log(" Calculating one- and two-electron integrals...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

        S, T, V_NE, D, ERI_AO = integ.evaluate_integrals(atomic_orbitals, np.array(charges, dtype=np.float64), coordinates, centre_of_mass)

        log("[Done]", calculation, 1, silent=silent)

        # Calculates Fock transformation matrix from overlap matrix
        log(" Constructing Fock transformation matrix...      ", calculation, 1, end="", silent=silent)

        X = calculate_Fock_transformation_matrix(S)

        log("[Done]", calculation, 1, silent=silent)

        # Calculates one-electron density for initial guess
        E_guess, P_guess, P_guess_alpha, P_guess_beta, _, _ = setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, reference, T, V_NE, X, n_doubly_occ, n_alpha, n_beta, calculation.rotate_guess, calculation.no_rotate_guess, calculation, silent=silent)

        log(" Beginning self-consistent field cycle...\n", calculation, 1, silent=silent)

        # Prints convergence criteria specified
        log(f" Using \"{calculation.scf_conv["name"]}\" SCF convergence criteria.", calculation, 1, silent=silent)

        # Prints the chosen SCF convergence acceleration options
        if calculation.DIIS and not calculation.damping: log(" Using DIIS for convergence acceleration.", calculation, 1, silent=silent)
        elif calculation.DIIS and calculation.damping: log(" Using initial dynamic damping and DIIS for convergence acceleration.", calculation, 1, silent=silent)
        elif calculation.damping and not calculation.slow_conv and not calculation.very_slow_conv: log(" Using permanent dynamic damping for convergence acceleration.", calculation, 1, silent=silent)  
        if calculation.slow_conv: log(" Using strong static damping for convergence acceleration.", calculation, 1, silent=silent)  
        elif calculation.very_slow_conv: log(" Using very strong static damping for convergence acceleration.", calculation, 1, silent=silent)  
        if calculation.level_shift: log(f" Using level shift for convergence acceleration with parameter {calculation.level_shift_parameter:.2f}.", calculation, 1, silent=silent)
        if not calculation.DIIS and not calculation.damping and not calculation.level_shift: log(" No convergence acceleration used.", calculation, 1, silent=silent)

        log("", calculation, 1, silent=silent)

        # Starts SCF cycle for two-electron energy
        SCF_output = scf.run_SCF(molecule, calculation, T, V_NE, ERI_AO, V_NN, S, X, E_guess, P=P_guess, P_alpha=P_guess_alpha, P_beta=P_guess_beta, silent=silent)

        # Extracts useful quantities from SCF output object
        molecular_orbitals = SCF_output.molecular_orbitals
        molecular_orbitals_alpha = SCF_output.molecular_orbitals_alpha  

        molecular_orbitals_beta = SCF_output.molecular_orbitals_beta   
        epsilons = SCF_output.epsilons
        epsilons_alpha = SCF_output.epsilons_alpha
        epsilons_beta = SCF_output.epsilons_beta
        P = SCF_output.P
        P_alpha = SCF_output.P_alpha
        P_beta = SCF_output.P_beta
        final_energy = SCF_output.energy
        kinetic_energy = SCF_output.kinetic_energy
        nuclear_electron_energy = SCF_output.nuclear_electron_energy
        coulomb_energy = SCF_output.coulomb_energy
        exchange_energy = SCF_output.exchange_energy

        # Packs dipole integrals into SCF output object
        SCF_output.D = D


        if reference == "UHF": 
            
            # Calculates UHF spin contamination and prints to the console
            postscf.calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S, calculation, reference, silent=silent)

        if not silent: 

            # Prints the individual components of the total SCF energy
            postscf.print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, V_NN, calculation)

        # If a correlated calculation is requested, calculates the energy and density matrices
        if method in ["MP2", "UMP2", "SCS-MP2", "MP3", "UMP3", "SCS-MP3", "USCS-MP2", "USCS-MP3", "OMP2", "UOMP2"]: 
            
            E_MP2, E_MP3, P, P_alpha, P_beta = mpn.calculate_Moller_Plesset(method, molecule, SCF_output, ERI_AO, calculation, X, T + V_NE, V_NN, silent=silent)
            postscf.calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S, calculation, "MP2", silent=silent)


    # Prints post SCF information, as long as its not an optimisation that hasn't finished yet
    if not terse and not silent: postscf.post_SCF_output(molecule, calculation, epsilons, molecular_orbitals, P, S, molecule.AO_ranges, D, P_alpha, P_beta, epsilons_alpha, epsilons_beta, molecular_orbitals_alpha, molecular_orbitals_beta)
    
    if method in ["CIS", "UCIS", "CIS[D]", "UCIS[D]"]:

        log("\n\n Beginning excited state calculation...", calculation, 1, silent=silent)

        if n_virt <= 0: error("Excited state calculation requested on system with no virtual orbitals!")

        E_CIS, E_transition, P_CIS, P_CIS_alpha, P_CIS_beta = ci.run_CIS(ERI_AO, n_occ, n_virt, n_SO, calculation, SCF_output, molecule, silent=silent)

        if calculation.additional_print: 
           
           # Optionally uses CIS density for dipole moment and population analysis
           postscf.post_SCF_output(molecule, calculation, epsilons, molecular_orbitals, P_CIS, S, molecule.AO_ranges, D, P_CIS_alpha, P_CIS_beta, epsilons_alpha, epsilons_beta, molecular_orbitals_alpha, molecular_orbitals_beta)

    # Prints Hartree-Fock energy
    if reference == "RHF": log("\n Final restricted Hartree-Fock energy: " + f"{final_energy:.10f}", calculation, 1, silent=silent)
    else: log("\n Final unrestricted Hartree-Fock energy: " + f"{final_energy:.10f}", calculation, 1, silent=silent)


    # Adds up and prints MP2 energies
    if method in ["MP2", "SCS-MP2", "UMP2", "USCS-MP2", "OMP2", "UOMP2"]: 
    
        final_energy += E_MP2

        log(f" Correlation energy from {method}: " + f"{E_MP2:.10f}\n", calculation, 1, silent=silent)
        log(" Final single point energy: " + f"{final_energy:.10f}", calculation, 1, silent=silent)


    # Adds up and prints MP3 energies
    elif method in ["MP3", "UMP3", "SCS-MP3", "USCS-MP3"]:
        
        final_energy += E_MP2 + E_MP3

        if method == "SCS-MP3":

            log(f" Correlation energy from SCS-MP2: " + f"{E_MP2:.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from SCS-MP3: " + f"{E_MP3:.10f}\n", calculation, 1, silent=silent)

        else:

            log(f" Correlation energy from MP2: " + f"{E_MP2:.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from MP3: " + f"{E_MP3:.10f}\n", calculation, 1, silent=silent)

        log(" Final single point energy: " + f"{final_energy:.10f}", calculation, 1, silent=silent)

    # Prints CIS energy of state of interest
    elif method in ["CIS", "UCIS", "CIS[D]", "UCIS[D]"]:

        final_energy = E_CIS

        d = "(D)" if "[D]" in method else ""

        log(f"\n Excitation energy to state {calculation.root} from CIS{d}: " + f"{E_transition:.10f}", calculation, 1, silent=silent)
        log(f"\n Final CIS{d} single point energy: {final_energy:.10f}", calculation, 1, silent=silent)

    # Adds on D2 energy, and prints this as dispersion-corrected final energy
    if calculation.D2:
    
        final_energy += E_D2

        log("\n Semi-empirical dispersion energy: " + f"{E_D2:.10f}", calculation, 1, silent=silent)
        log(" Dispersion-corrected final energy: " + f"{final_energy:.10f}", calculation, 1, silent=silent)
    
    # Calculates and plots electron density if this is requested
    if not silent and len(atoms) > 1:

        if calculation.dens_plot:

            plot.construct_electron_density(P, 0.07, molecule, calculation)

        if calculation.spin_dens_plot:
            
            if n_alpha != n_beta:
                
                R = P_alpha - P_beta if n_alpha + n_beta != 1 else P

                plot.construct_electron_density(R, 0.07, molecule, calculation)

            else: error("Spin density plot requested on singlet molecule!")

    return SCF_output, molecule, final_energy, P


    

def scan_coordinate(calculation, atoms, starting_coordinates):

    """

    Loops through a number of scan steps and increments bond length, calculating enery each time.

    Args:
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        starting_coordinates (array): Atomic coordinates to being coordinate scan calculation in 3D

    Returns:
        None: Nothing is returned

    """

    coordinates = starting_coordinates
    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

    # Unpacks useful quantities
    number_of_steps = calculation.scan_number
    step_size = calculation.scan_step

    log(f"Initialising a {number_of_steps} step coordinate scan in {step_size:.4f} angstrom increments.", calculation, 1) 
    log(f"Starting at a bond length of {bohr_to_angstrom(bond_length):.4f} angstroms.\n", calculation, 1)
    
    bond_lengths = [] 
    energies = []   
    
    P_guess = None
    E_guess = None 
    P_guess_alpha = None 
    P_guess_beta = None


    for step in range(1, number_of_steps + 1):
        
        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        log("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        log(f"Starting scan step {step} of {number_of_steps} with bond length of {bohr_to_angstrom(bond_length):.4f} angstroms...", calculation, 1)
        log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
        
        #Calculates the energy at the coordinates (in bohr) specified
        SCF_output, _, energy, _ = calculate_energy(calculation, atoms, coordinates, P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True)

        #If MOREAD keyword is used, then the energy and densities are used for the next calculation
        if calculation.MO_read: 
            
            P_guess = SCF_output.P
            E_guess = energy 
            P_guess_alpha = SCF_output.P_alpha 
            P_guess_beta = SCF_output.P_beta

        # Appends energies and bond lengths to lists
        energies.append(energy)
        bond_lengths.append(bond_length)

        # Builds new coordinates by adding step size on
        coordinates = np.array([coordinates[0], [0, 0, bond_length + step_size]])
        
    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)    
    
    log("\nCoordinate scan calculation finished!\n\n Printing energy as a function of bond length...\n", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("            Coordinate Scan", calculation, 1, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("    Bond Length           Energy", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    # Prints a table of bond lengths and corresponding energies
    for energy, bond_length in zip(energies, bond_lengths):
        
        log(f"      {bohr_to_angstrom(bond_length):.4f}          {energy:13.10f}", calculation, 1)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    
    # If SCANPLOT keyword is used, plots and shows a matplotlib graph of the data
    if calculation.scan_plot: 
        
        plot.scan_plot(calculation, bond_lengths, energies)

