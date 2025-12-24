import numpy as np
from tuna_util import *


def calculate_reduced_mass(masses): 

    """

    Calculates the reduced mass.

    Args:   
        masses (array): Mass array in atomic units

    Returns:
        reduced_mass (float): Reduced mass in atomic units

    """

    reduced_mass = np.prod(masses) / np.sum(masses) 

    return reduced_mass



 

def calculate_nuclear_dipole_moment(centre_of_mass, charges, coordinates): 

    """

    Calculates the nuclear dipole moment.

    Args:   
        centre_of_mass (float): Centre of mass in bohr 
        charges (array): Nuclear charges
        coordinates (array): Nuclear coordinates

    Returns:
        nuclear_dipole_moment (float): Nuclear dipole moment

    """

    nuclear_dipole_moment = 0

    for i in range(len(charges)): 

        nuclear_dipole_moment += (coordinates[i][2] - centre_of_mass) * charges[i]
    

    return nuclear_dipole_moment
   






def calculate_rotational_constant(masses, coordinates):

    """

    Calculates the rotational constant of a molecule.

    Args:   
        masses (float): Mass array in atomic units
        coordinates (array): Nuclear coordinates

    Returns:
        rotational_constant_per_cm (float): Rotational constant in per cm
        rotational_constant_GHz (float): Rotational constant in GHz

    """

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])
    reduced_mass = calculate_reduced_mass(masses)
    
    # Standard equation for linear molecule's rotational constant
    rotational_constant_hartree = 1 / (2 * reduced_mass * bond_length ** 2)

    # Various unit conversions  
    rotational_constant_per_bohr = rotational_constant_hartree / (constants.h * constants.c)
    rotational_constant_per_cm = rotational_constant_per_bohr / (100 * constants.bohr_in_metres)
    rotational_constant_GHz = constants.per_cm_in_GHz * rotational_constant_per_cm
    
    return rotational_constant_per_cm, rotational_constant_GHz







def calculate_Koopmans_parameters(epsilons, n_occ):

    """

    Calculates the Koopmans' theorem parameters of a system (ionisation energy, electron affinity and HOMO-LUMO gap).

    Args:   
        epsilons (array): Fock matrix eigenvalues
        n_occ (int): Number of occupied orbitals

    Returns:
        ionisation_energy (float): Ionisation energy
        electron_affinity (float): Electron affinity
        HOMO_LUMO_gap (float): Difference in energy between HOMO and LUMO

    """

    # IE = -HOMO
    ionisation_energy = -1 * epsilons[n_occ - 1]

    # As long as LUMO exists, EA = -LUMO    
    if len(epsilons) > n_occ: 
    
        electron_affinity = -1 * epsilons[n_occ]
        HOMO_LUMO_gap = ionisation_energy - electron_affinity
        
    else: 
    
        electron_affinity = "---"
        HOMO_LUMO_gap = "---"
        
        warning("Size of basis is too small for electron affinity calculation!")


    return ionisation_energy, electron_affinity, HOMO_LUMO_gap
 
 




def print_energy_components(nuclear_electron_energy, kinetic_energy, exchange_energy, coulomb_energy, correlation_energy, V_NN, calculation, silent=False):

    """

    Prints the various components of the Hartree-Fock energy to the console.

    Args:   
        nuclear_electron_energy (float): Nuclear-electron attraction energy
        kinetic_energy (float): Electronic kinetic energy
        exchange_energy (float): Exchange energy
        coulomb_energy (float): Coulomb energy
        correlation_energy (float): Correlation energy
        V_NN (float): Nuclear-nuclear repulsion energy
        calculation (Calculation): Calculation object

    Returns:
        None: Nothing is returned

    """

    # Adds up different energy components
    one_electron_energy = nuclear_electron_energy + kinetic_energy
    two_electron_energy = exchange_energy + coulomb_energy + correlation_energy
    electronic_energy = one_electron_energy + two_electron_energy

    total_energy = electronic_energy + V_NN
    
    virial_ratio = -1 * (total_energy - kinetic_energy) / kinetic_energy
           
    log_spacer(calculation, priority=2, silent=silent)
    log("                  Energy Components       ", calculation, 2, colour="white", silent=silent)
    log_spacer(calculation, priority=2, silent=silent)            
    log(f"  Kinetic energy:                   {kinetic_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Coulomb energy:                   {coulomb_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Exchange energy:                  {exchange_energy:15.10f}", calculation, 2, silent=silent)

    if calculation.method in DFT_methods:
        
        log(f"  Correlation energy:               {correlation_energy:15.10f}", calculation, 2, silent=silent)

    log(f"  Nuclear repulsion energy:         {V_NN:15.10f}", calculation, 2, silent=silent)
    log(f"  Nuclear attraction energy:        {nuclear_electron_energy:15.10f}\n", calculation, 2, silent=silent)      

    log(f"  One-electron energy:              {one_electron_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Two-electron energy:              {two_electron_energy:15.10f}", calculation, 2, silent=silent)

    if calculation.method in DFT_methods:
        
        log(f"  Exchange-correlation energy:      {exchange_energy + correlation_energy:15.10f}", calculation, 2, silent=silent)

    log(f"  Electronic energy:                {electronic_energy:15.10f}\n", calculation, 2, silent=silent)
    log(f"  Virial ratio:                     {virial_ratio:15.10f}\n", calculation, 2, silent=silent)
            
    log(f"  Total energy:                     {total_energy:15.10f}", calculation, 2, silent=silent)
    log_spacer(calculation, priority=2, silent=silent)






def calculate_spin_contamination(P_alpha, P_beta, n_alpha, n_beta, S, calculation, type, silent=False):

    """

    Calculates and prints theoretical spin squared operator and spin contamination.

    Args:
        P_alpha (array): Density matrix of alpha electrons in AO basis
        P_beta (array): Density matrix of beta electrons in AO basis
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        S (array): Overlap matrix in AO basis
        calculation (Calculation): Calculation object
        type (str): Either UHF or MP2
        silent (bool, optional): Should anything be printed

    Returns:
        None: Nothing is returned

    """

    s_squared_exact = ((n_alpha - n_beta) / 2) * ((n_alpha - n_beta) / 2 + 1)

    # Tensor contraction to calculate spin contamination
    spin_contamination = n_beta - np.trace(P_alpha.T @ S @ P_beta.T @ S)
    
    s_squared = s_squared_exact + spin_contamination

    space1 = "       "
    space2 = "            "
    title = type 

    if type in ["UHF", "UKS"]: priority = 2
    else: priority = 3

    if type == "Coupled cluster":

        space1 = ""
        space2 = ""
        title = type.title()


    
    log_spacer(calculation, silent=silent, priority=priority)
    log(f"   {space1}       {title} Spin Contamination       ", calculation, priority, silent=silent, colour="white")
    log_spacer(calculation, silent=silent, priority=priority)

    log(f"  Exact S^2 expectation value:            {s_squared_exact:9.6f}", calculation, priority, silent=silent)
    log(f"  {type} S^2 expectation value:  {space2}{s_squared:9.6f}", calculation, priority, silent=silent)
    log(f"\n  Spin contamination:                     {spin_contamination:9.6f}", calculation, priority, silent=silent)

    log_spacer(calculation, silent=silent, priority=priority)







def calculate_population_analysis(P, S, R, AO_ranges, atoms, charges):

    """

    Calculates the bond order, atomic charges and valences for Mulliken, Lowden and Mayer population analysis

    Args:   
        P (array): Density matrix in AO basis
        S (array): Overlap matrix in AO basis
        R (array): Spin density matrix in AO basis
        AO_ranges (array): Separates basis onto each atom
        atoms (list): Atomic symbols
        charges (array): Nuclear charges

    Returns:
        bond_order_Mulliken (float): Mulliken bond order
        charges_Mulliken (array): Mulliken atomic charges
        total_charges_Mulliken (float): Sum of Mulliken charges
        bond_order_Lowdin (float): Lowdin bond order
        charges_Lowdin (array): Lowdin atomic charges
        total_charges_Lowdin (float): Sum of Lowdin charges
        bond_order_Mayer (float): Mayer bond order
        free_valences (array): Mayer free valences
        total_valences (array): Mayer total valences

    """

    PS = P @ S
    RS = R @ S

    # Diagonalises overlap matrix to form density matrix in orthogonalised Lowdin basis
    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    P_Lowdin = S_sqrt @ P @ S_sqrt

    bond_order_Mayer, bond_order_Lowdin, bond_order_Mulliken = 0, 0, 0

    total_valences = [0, 0]
    populations_Mulliken = [0, 0]
    populations_Lowdin = [0, 0]
    charges_Mulliken = [0, 0]
    charges_Lowdin = [0, 0]


    # Sums over the ranges of each atomic orbital over atom A, then atom B to build the three bond orders
    for i in range(AO_ranges[0]):
        for j in range(AO_ranges[0], AO_ranges[0] + AO_ranges[1]):

            bond_order_Mayer += PS[i, j] * PS[j, i] + RS[i, j] * RS[j, i]
            bond_order_Lowdin += P_Lowdin[i,j] ** 2
            bond_order_Mulliken += 2 * P[i, j] * S[i, j]
    
    # Sums over atoms, then corresponding ranges of atomic orbitals in the density matrix, to build the valences and populations
    for atom in range(len(atoms)):

        if atom == 0: atomic_ranges = list(range(AO_ranges[0]))
        elif atom == 1: atomic_ranges = list(range(AO_ranges[0], AO_ranges[0] + AO_ranges[1]))

        for i in atomic_ranges:
            
            populations_Lowdin[atom] += P_Lowdin[i, i] 
            populations_Mulliken[atom] += PS[i, i]
            total_valences[atom] += np.einsum("j,j->", PS[i, atomic_ranges], PS[atomic_ranges, i], optimize=True)

        charges_Mulliken[atom] = charges[atom] - populations_Mulliken[atom]
        charges_Lowdin[atom] = charges[atom] - populations_Lowdin[atom]

        total_valences[atom] = 2 * populations_Mulliken[atom] - total_valences[atom]

    # Adds up total charges and calculates free valences from total and bonded valences
    total_charges_Mulliken = np.sum(charges_Mulliken)
    total_charges_Lowdin = np.sum(charges_Lowdin)

    free_valences = np.array(total_valences) - bond_order_Mayer


    return bond_order_Mulliken, charges_Mulliken, total_charges_Mulliken, bond_order_Lowdin, charges_Lowdin, total_charges_Lowdin, bond_order_Mayer, free_valences, total_valences






def calculate_dipole_moment(centre_of_mass, charges, coordinates, P, D):

    """

    Calculates the total dipole moment of a molecule.

    Args:   
        centre_of_mass (float): Centre of mass
        charges (array): Nuclear charges
        coordinates (array): Nuclear coordinates
        P (array): Density matrix in AO basis
        D (array): Dipole integral matrix in AO basis

    Returns:
        total_dipole_moment (float): Total molecular dipole moment in atomic units
        nuclear_dipole_moment (float): Nuclear dipole moment in atomic units
        electronic_dipole_moment (float): Electronic dipole moment in atomic units

    """

    nuclear_dipole_moment = calculate_nuclear_dipole_moment(centre_of_mass, charges, coordinates)        
    electronic_dipole_moment = -1 * np.einsum("ij,ij->", P, D, optimize=True)

    total_dipole_moment = nuclear_dipole_moment + electronic_dipole_moment


    return total_dipole_moment, nuclear_dipole_moment, electronic_dipole_moment







def print_molecular_orbital_eigenvalues(calculation, reference, n_doubly_occ, n_alpha, n_beta, epsilons, epsilons_alpha, epsilons_beta):

    """

    Prints the Fock matrix eigenvalues, separately for UHF references.

    Args:   
        calculation (Calculation): Calculation object
        reference (str): Either RHF or UHF
        n_doubly_occ (int): Number of doubly occupied orbitals
        n_alpha (int): Number of alpha orbitals
        n_beta (int): Number of beta orbitals
        epsilons (array): Fock matrix eigenvalues
        epsilons_alpha (array): Alpha Fock matrix eigenvalues
        epsilons_beta (array): Beta Fock matrix eigenvalues       

    Returns:
        None: Nothing is returned

    """

    log_spacer(calculation, 3, start="\n")
    log("           Molecular Orbital Eigenvalues", calculation, 3, colour="white")
    log_spacer(calculation, 3)


    # Prints alpha and beta eigenvalues separately
    if reference == "UHF":

        if n_beta > 0:

            log("\n Alpha eigenvalues:          Beta eigenvalues:\n", calculation, 3)
            
            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)
            log("   N    Occ.   Epsilon         N    Occ.   Epsilon  ", calculation, 3)
            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)
            
            # Occupied orbitals are alpha electrons only
            occupancies_alpha = [1] * n_alpha + [0] * int((len(epsilons_alpha) - n_alpha))
            occupancies_beta = [1] * n_beta + [0] * int((len(epsilons_beta) - n_beta))

            for i, (epsilon_alpha, epsilon_beta) in enumerate(zip(epsilons_alpha, epsilons_beta)):

                log(f"  {(i + 1):2.0f}     {occupancies_alpha[i]}   {epsilon_alpha:10.6f}       {(i + 1):2.0f}     {occupancies_beta[i]}   {epsilon_beta:10.6f}", calculation, 3)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 3)


        else:

            log("\n  Alpha eigenvalues:\n", calculation, 3)
            
            log("  ~~~~~~~~~~~~~~~~~~~~~~~  ", calculation, 3)
            log("    N    Occ.   Epsilon     ", calculation, 3)
            log("  ~~~~~~~~~~~~~~~~~~~~~~~   ", calculation, 3)
            
            # Occupied orbitals are alpha electrons only
            occupancies_alpha = [1] * n_alpha + [0] * int((len(epsilons_alpha) - n_alpha))

            for i, epsilon_alpha in enumerate(epsilons_alpha):

                log(f"   {(i + 1):2.0f}     {occupancies_alpha[i]}   {epsilon_alpha:10.6f}    ", calculation, 3)

            log("  ~~~~~~~~~~~~~~~~~~~~~~~  \n", calculation, 3)


    elif reference == "RHF":

        log("    N            Occupation             Epsilon ", calculation, 3)
        log_spacer(calculation, 3)

        # Occupied orbitals (doubly occupied) depend on number electron pairs
        occupancies = [2] * n_doubly_occ + [0] * int((len(epsilons) - n_doubly_occ))

        for i, epsilon in enumerate(epsilons):

            log(f"   {(i + 1):2.0f}                {occupancies[i]}                {epsilon:10.6f}", calculation, 3)









def print_molecular_orbital_coefficients(molecule, atoms, calculation, reference, epsilons, epsilons_alpha, n_alpha, n_beta, n_doubly_occ, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta):
    
    """

    Prints the coefficients of all molecular orbtials, for both alpha and beta spins for UHF.

    Args:   
        molecule (Molecule): Molecule object
        atoms (list): Atomic symbols
        calculation (Calculation): Calculation object
        reference (str): Either RHF or UHF
        epsilons (array): Fock matrix eigenvalues
        epsilons_alpha (array): Alpha Fock matrix eigenvalues
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        n_doubly_occ (int): Number of doubly-occupied molecular orbitals
        molecular_orbitals (array): Molecular orbitals in AO basis    
        molecular_orbitals_alpha (array): Alpha molecular orbitals in AO basis    
        molecular_orbitals_beta (array): Beta molecular orbitals in AO basis    

    """

    log_spacer(calculation, 3, start="")
    log("           Molecular Orbital Coefficients", calculation, 3, colour="white")
    log_spacer(calculation, 3)
    
    # Initialises various quantities
    symbol_list = []
    n_list = []
    switch_value = 0

    # Builds a list of atomic symbols and number of atomic orbitals per atom
    for i, atom in enumerate(molecule.partitioned_basis_functions):
        for j, _ in enumerate(atom):
            
            symbol_list.append(atoms[i])                  
            n_list.append(j + 1)
            
            # Determines the index at which the orbitals switch over from one atomic centre to the other
            if i == 1 and j == 0: 
                
                switch_value = len(symbol_list) - 1



    def format_angular_momentum_list(angular_momentum_list, AO_ranges):

        """
        
        Formats the angular momentum list with per-atom subshell numbering.

        Args:
            angular_momentum_list (list): List of angular momentum strings
            AO_ranges (array): Ranges of basis functions partitioned over atomic centres

        Returns:
            formatted (list): Formatted angular momentum list, with degeneracies considered
        
        """

        # Spectroscopic sequence without 'j'
        spectro_letters = "spdfghi"
        l_of = {ch: i for i, ch in enumerate(spectro_letters)}

        def min_n(letter):

            # The value of n at which the subshell first appears

            l = l_of.get(letter, None)

            return l + 1

        def degeneracy(letter):

            # Gives the degeneracy of each subshell (Cartesian)
            l = l_of.get(letter, None)

            return (l + 1) * (l + 2) / 2


        formatted = []
        idx = 0  # running index into angular_momentum_list

        for num_orbitals in AO_ranges:

            # Per-atom state: letter -> (current_n, used_in_current_n)
            state = {}

            for _ in range(num_orbitals):
                letter = str(angular_momentum_list[idx]).lower()
                
                n, used = state.get(letter, (min_n(letter), 0))
                cap = degeneracy(letter)

                if used >= cap:

                    # Advance principal quantum number when this shell is "full"
                    n += 1
                    used = 0

                used += 1
                state[letter] = (n, used)
                formatted.append(f"{n}{letter}")

                idx += 1

        return formatted
    



    formatted_angular_momentum_list = format_angular_momentum_list(molecule.angular_momentum_list, molecule.partition_ranges)

        

        
    def format_molecular_orbitals(symbol_list, k, switch_value, atoms, calculation, has_printed_1, has_printed_2):

        """
        
        Manages ghost atoms and formats the list of atoms to be printed.

        Args:
            symbol_list (list): List of atomic symbols
            k (int): Which element of list
            switch_value (int): Value of for loop at which the atom centre changes
            atoms (list): List of atoms
            calculation (Calculation): Calculation object
            has_printed_1 (bool): Has a gap printed
            has_printed_2 (bool): Has another gap printed
        
        """

        # Formats ghost atoms
        if "X" in symbol_list[k]: 

            symbol_list[k] = symbol_list[k].split("X")[1]
            symbol_list[k] = "X" + symbol_list[k].lower().capitalize()
            
        else: symbol_list[k] = symbol_list[k].lower().capitalize()

        if len(symbol_list[k]) == 1: symbol_list[k] = symbol_list[k] + " "

        # Manages when to print gaps
        if k < switch_value:

            if not has_printed_1: has_printed_1 = True
            else: symbol_list[k] = "  "

        else:

            if not has_printed_2: has_printed_2 = True
            else: symbol_list[k] = "  "

        if k == switch_value and len(atoms) == 2: 
            
            log("", calculation, 3)


        return symbol_list[k], has_printed_1, has_printed_2




    # Prints out coefficients for each orbital, as well as if each is occupied or virtual
    def print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals, eps, n, formatted_ang_mom):
        
        n_orbitals_to_print = min(len(eps), n_alpha + 10)

        for mo in range(n_orbitals_to_print):
            
            if n > mo: occ = "(Occupied)"
            else: occ = "(Virtual)"

            log(f"\n   MO {mo+1} {occ}\n", calculation, 3)
                
            has_printed_1 = False    
            has_printed_2 = False    
            
            n_orbitals_to_print = min([len(molecular_orbitals.T[mo]), 10])

            for k in range(n_orbitals_to_print):

                # Formats ghost atoms nicely, ignores decontracting basis
                try:
                    
                    symbol_list[k], has_printed_1, has_printed_2 = format_molecular_orbitals(symbol_list, k, switch_value, atoms, calculation, has_printed_1, has_printed_2)

                    log("    " + symbol_list[k] + f"  {formatted_ang_mom[k]}  :  " + f"{molecular_orbitals.T[mo][k]:7.4f}", calculation, 3)

                except: pass

    # For UHF calculations, do all of the above but separately for alpha and beta orbitals
    if reference == "UHF":

        if n_beta > 0:
            
            log("\n Alpha coefficients:          Beta coefficients:", calculation, 3)

            n_orbitals_to_print = min(len(epsilons_alpha), n_alpha + 10)

            for mo in range(n_orbitals_to_print):
                
                if n_alpha > mo: occ = "(Occupied)"
                else: occ = "(Virtual)"

                if n_beta > mo: occ_beta = "(Occupied)"
                else: occ_beta = "(Virtual)"

                log(f"\n  MO {mo+1} {occ}              MO {mo+1} {occ_beta}\n", calculation, 3)
                    
                has_printed_1 = False    
                has_printed_2 = False    
                

                for k in range(len(molecular_orbitals.T[mo])):

                    # Formats ghost atoms nicely, ignores decontracting basis
                    try:

                        symbol_list[k], has_printed_1, has_printed_2 = format_molecular_orbitals(symbol_list, k, switch_value, atoms, calculation, has_printed_1, has_printed_2)

                        log("   " + symbol_list[k] + f"  {formatted_angular_momentum_list[k]}  :  " + f"{molecular_orbitals_alpha.T[mo][k]:7.4f}" + "           " + symbol_list[k] + f"  {formatted_angular_momentum_list[k]}  :  " + f"{molecular_orbitals_beta.T[mo][k]:7.4f}", calculation, 3)

                    except: pass

        else:

            log("\n Alpha coefficients:         ", calculation, 3)

            n_orbitals_to_print = min(len(epsilons_alpha), n_alpha + 10)

            # If there's only one electron, just print the alpha coefficients
            for mo in range(n_orbitals_to_print):
                
                if n_alpha > mo: occ = "(Occupied)"
                else: occ = "(Virtual)"


                log(f"\n   MO {mo+1} {occ}      \n", calculation, 3)
                    
                has_printed_1 = False    
                has_printed_2 = False    

                for k in range(len(molecular_orbitals.T[mo])):

                    # Formats ghost atoms nicely, ignores decontracting basis
                    try:
                        
                        symbol_list[k], has_printed_1, has_printed_2 = format_molecular_orbitals(symbol_list, k, switch_value, atoms, calculation, has_printed_1, has_printed_2)

                        log("    " + symbol_list[k] + f"  {formatted_angular_momentum_list[k]}  :  " + f"{molecular_orbitals_alpha.T[mo][k]:7.4f}", calculation, 3)

                    except: pass


    # For RHF calculations, do all of the above for the combined doubly occupied orbitals
    else:
        
        print_coeffs(switch_value, calculation, symbol_list, molecular_orbitals, epsilons, n_doubly_occ, formatted_angular_momentum_list)


    log_spacer(calculation, 3, start="\n")








def post_SCF_output(molecule, calculation, epsilons, molecular_orbitals, P, S, AO_ranges, D, P_alpha, P_beta, epsilons_alpha, epsilons_beta, molecular_orbitals_alpha, molecular_orbitals_beta):

    """

    Calculates various TUNA properties and prints them to the console.

    Args:   
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        epsilons (array): Fock matrix eigenvalues
        molecular_orbitals (array): Molecular orbitals in AO basis    
        P (array): Density matrix in AO basis
        S (array): Overlap matrix in AO basis
        AO_ranges (array): Partition of basis set between atoms
        D (array): Dipole moment integral matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        epsilons_alpha (array): Alpha Fock matrix eigenvalues
        epsilons_beta (array): Beta Fock matrix eigenvalues   
        molecular_orbitals_alpha (array): Alpha molecular orbitals in AO basis    
        molecular_orbitals_beta (array): Beta molecular orbitals in AO basis    

    """

    log("\n Beginning calculation of TUNA properties... ", calculation, 3)

    # Unpacks useful quantities    
    method = calculation.method
    n_doubly_occ = molecule.n_doubly_occ
    n_alpha = molecule.n_alpha
    n_beta = molecule.n_beta
    reference = calculation.reference
    masses = molecule.masses
    coordinates = molecule.coordinates
    atoms = molecule.atomic_symbols
    charges = molecule.charges
    molecular_structure = molecule.molecular_structure


    # Specifies which density matrix is used for the property calculations
    if method in ["MP2", "SCS-MP2", "UMP2", "USCS-MP2"]: log("\n Using the MP2 unrelaxed density for property calculations.", calculation, 1)
    elif method in ["OMP2", "UOMP2", "OOMP2", "UOOMP2"]: log("\n Using the orbital-optimised MP2 relaxed density for property calculations.", calculation, 1)
    elif method in ["LMP2"]: warning("Using the Hartree-Fock density, not the MP2 density, for property calculations.")
    elif method in ["MP3", "SCS-MP3", "UMP3", "USCS-MP3", "MP4", "MP4[SDQ]"]: warning("Using the unrelaxed MP2 density for property calculations.")
    if "CC" in method or "CEPA" in method or "QC" in method: log("\n Using the linearised coupled cluster density for property calculations.", calculation, 1)
    if method in ["CCSD[T]", "UCCSD[T]"]: warning("Using the linearised CCSD density, not the CCSD(T) density, for property calculations.")
    if method in ["QCISD[T]", "UQCISD[T]"]: warning("Using the linearised QCISD density, not the QCISD(T) density, for property calculations.")

    # Prints molecular orbital eigenvalues and coefficients
    print_molecular_orbital_eigenvalues(calculation, reference, n_doubly_occ, n_alpha, n_beta, epsilons, epsilons_alpha, epsilons_beta)
    print_molecular_orbital_coefficients(molecule, atoms, calculation, reference, epsilons, epsilons_beta, n_alpha, n_beta, n_doubly_occ, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta)

    # Prints Koopmans' theorem parameters if RHF reference is used
    if calculation.reference == "RHF":
        
        ionisation_energy, electron_affinity, HOMO_LUMO_gap = calculate_Koopmans_parameters(epsilons, molecule.n_doubly_occ)

        if electron_affinity != "---": electron_affinity = f"{electron_affinity:9.6f}"
        if HOMO_LUMO_gap != "---": HOMO_LUMO_gap = f"{HOMO_LUMO_gap:9.6f}"
            
        log(f"\n Koopmans' theorem ionisation energy:   {ionisation_energy:9.6f}", calculation, 2)
        log(f" Koopmans' theorem electron affinity:   {electron_affinity}", calculation, 2)
        log(f" Energy gap between HOMO and LUMO:      {HOMO_LUMO_gap}", calculation, 2)

    # As long as there are two real atoms present, calculates rotational constant and dipole moment information
    if len(molecule.atoms) != 1 and not any(atom.ghost for atom in molecule.atoms):

        B_per_cm, B_GHz = calculate_rotational_constant(masses, coordinates)
                
        log(f"\n Rotational constant (GHz): {B_GHz:.3f}", calculation, 2)
        log(f" Rotational constant (per cm): {B_per_cm:.3f}", calculation, 2)

        # Calculates centre of mass for dipole moment calculations
        centre_of_mass = calculate_centre_of_mass(masses, coordinates)

        log(f"\n Dipole moment origin is the centre of mass, {bohr_to_angstrom(centre_of_mass):.5f} angstroms from the first atom.", calculation, 2)

        total_dipole, D_nuclear, D_electronic = calculate_dipole_moment(centre_of_mass, charges, coordinates, P, D)

        log_spacer(calculation, priority=2, start="\n")
        log("                    Dipole Moment", calculation, 2, colour="white")
        log_spacer(calculation, priority=2)

        log(f"  Nuclear: {D_nuclear:11.8f}      Electronic: {D_electronic:11.8f}\n", calculation, 2)
        log(f"  Total: {total_dipole:11.8f}     ", calculation, 2, end="")

        # Prints direction of dipole moment, where plus indicates positive side    
        if total_dipole > 0.00001:

            log("        " + molecular_structure, calculation, 2, end="")
            log("  +--->", calculation, 2)

        elif total_dipole < -0.00001:

            log("        " + molecular_structure, calculation, 2, end="")
            log("  <---+", calculation, 2)

        # If there's no dipole moment, just print out the molecular structure
        else: log(f"           {molecular_structure}", calculation, 2)

        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        
        # Builds spin density matrix
        R = P_alpha - P_beta if n_alpha + n_beta != 1 else P

        # Calculate population analysis and format all the data, then print to console
        bond_order_Mulliken, charges_Mulliken, total_charges_Mulliken, bond_order_Lowdin, charges_Lowdin, total_charges_Lowdin, bond_order_Mayer, free_valences, total_valences = calculate_population_analysis(P, S, R, AO_ranges, atoms, charges)
        
        atoms_formatted = []

        for atom in molecule.atomic_symbols:
        
            atom = atom.lower().capitalize()
            atom = atom + "  :" if len(atom) == 1 else atom + " :"
            atoms_formatted.append(atom)


        log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        log("      Mulliken Charges                Lowdin Charges                Mayer Free, Bonded, Total Valence", calculation, 2, colour="white")
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
        log(f"  {atoms_formatted[0]} {charges_Mulliken[0]:8.5f}                  {atoms_formatted[0]} {charges_Lowdin[0]:8.5f}                  {atoms_formatted[0]} {free_valences[0]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[0]:8.5f}", calculation, 2)
        log(f"  {atoms_formatted[1]} {charges_Mulliken[1]:8.5f}                  {atoms_formatted[1]} {charges_Lowdin[1]:8.5f}                  {atoms_formatted[1]} {free_valences[1]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[1]:8.5f}", calculation, 2)
        log(f"\n  Sum of charges: {total_charges_Mulliken:8.5f}       Sum of charges: {total_charges_Lowdin:8.5f}", calculation, 2) 
        log(f"  Bond order: {bond_order_Mulliken:8.5f}           Bond order: {bond_order_Lowdin:8.5f}           Bond order: {bond_order_Mayer:8.5f}", calculation, 2) 
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)