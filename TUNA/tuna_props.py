import numpy as np
from numpy import ndarray
from tuna_util import *
import tuna_molecule as mol
from tuna_molecule import Molecule
from tuna_calc import Calculation

 
"""

This is the TUNA module for post energy evaluation properties, written first for version 0.2.0 and rewritten for version 0.10.1.

Functions to calculate and print properties after the molecular energy and density matrix has been evaluated live here. These properties include population analysis,
multipole moments, and printing in a clearly formatted way the molecular orbital coefficients and eigenvalues.

Updated in version 0.11.0 to improve orbital printing, including support for natural orbitals.

This module contains:

1. Functions to calculate multipole moments (calculate_nuclear_dipole_moment, calculate_nuclear_quadrupole_moment, calculate_and_print_multipole_moments, etc.)
2. Functions to calculate population analysis, and print and format the orbitals (print_molecular_orbital_eigenvalues, print_molecular_orbital_coefficients, etc.)
3. The main function to calculate molecular properties at the end of an energy evaluation (calculate_molecular_properties)

"""



def calculate_nuclear_dipole_moment(dipole_origin: float, charges: ndarray, coordinates: ndarray) -> float: 

    """

    Calculates the nuclear dipole moment.

    Args:   
        dipole_origin (float): Dipole origin in bohr 
        charges (array): Nuclear charges
        coordinates (array): Nuclear coordinates

    Returns:
        nuclear_dipole_moment (float): Nuclear dipole moment

    """

    nuclear_dipole_moment = 0

    for i in range(len(charges)): 

        nuclear_dipole_moment += (coordinates[i][2] - dipole_origin) * charges[i]
    

    return nuclear_dipole_moment










def calculate_nuclear_quadrupole_moment(quadrupole_origin: float, charges: ndarray, coordinates: ndarray) -> float: 

    """

    Calculates the nuclear quadrupole moment.

    Args:   
        quadrupole_origin (float): Quadrupole origin in bohr 
        charges (array): Nuclear charges
        coordinates (array): Nuclear coordinates

    Returns:
        nuclear_quadrupole_moment (float): Nuclear quadrupole moment

    """

    nuclear_quadrupole_moment = 0

    for i in range(len(charges)): 

        nuclear_quadrupole_moment += (coordinates[i][2] - quadrupole_origin) ** 2 * charges[i]
    

    return nuclear_quadrupole_moment
   









def calculate_analytical_dipole_moment(centre_of_mass: float, charges: ndarray, coordinates: ndarray, P: ndarray, D: ndarray) -> tuple:

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

    # Extracts the z component of the dipole moment integrals

    nuclear_dipole_moment = calculate_nuclear_dipole_moment(centre_of_mass, charges, coordinates)        
    electronic_dipole_moment = -1 * np.einsum("ij,ij->", P, D[2], optimize = True)

    total_dipole_moment = nuclear_dipole_moment + electronic_dipole_moment


    return total_dipole_moment, nuclear_dipole_moment, electronic_dipole_moment










def calculate_analytical_quadrupole_moment(centre_of_mass: float, charges: ndarray, coordinates: ndarray, P: ndarray, Q: ndarray) -> tuple:

    """

    Calculates the total quadrupole moment of a molecule.

    Args:   
        centre_of_mass (float): Centre of mass
        charges (array): Nuclear charges
        coordinates (array): Nuclear coordinates
        P (array): Density matrix in AO basis
        Q (array): Quadrupole integral matrix in AO basis

    Returns:
        isotropic_quadrupole_moment (float): Isotropic quadrupole moment in atomic units
        nuclear_quadrupole_moment (float): Nuclear quadrupole moment in atomic units
        anisotropic_quadrupole_moment (float): Anisotropic quadrupole in atomic units

    """


    nuclear_quadrupole_moment = calculate_nuclear_quadrupole_moment(centre_of_mass, charges, coordinates)       
    
    # Extracts the xx and zz components of quadrupole moment integrals

    electronic_quadrupole_moment_xx = -1 * np.einsum("ij,ij->", P, Q[0], optimize = True)
    electronic_quadrupole_moment_zz = -1 * np.einsum("ij,ij->", P, Q[1], optimize = True)

    anisotropic_quadrupole_moment = electronic_quadrupole_moment_zz + nuclear_quadrupole_moment - electronic_quadrupole_moment_xx

    # Calculates the trace of the quadrupole moment tensor, leveraging diatomic symmetry

    isotropic_quadrupole_moment = (1 / 3) * (nuclear_quadrupole_moment + electronic_quadrupole_moment_zz + electronic_quadrupole_moment_xx * 2)

    return isotropic_quadrupole_moment, nuclear_quadrupole_moment, anisotropic_quadrupole_moment










def calculate_and_print_multipole_moments(P: ndarray, molecule: Molecule, SCF_output: Output, calculation: Calculation) -> None:

    """
    
    Calculates and prints the analytical dipole and quadrupole moments.

    Args:
        P (array): Density matrix in AO basis
        molecule (Molecule): Molecule object
        SCF_output (Output): Output from SCF calculation
        calculation (Calculation): Calculation object
    
    """

    # Calculates the centre of mass of the molecule

    centre_of_mass = calculate_centre_of_mass(molecule.masses, molecule.coordinates)

    log(f"\n Multipole moment origin is the centre of mass, {bohr_to_angstrom(centre_of_mass):.5f} angstroms from the first atom.", calculation, 2)

    # Calculates the analytical dipole moment

    total_dipole_moment, nuclear_dipole_moment, electronic_dipole_moment = calculate_analytical_dipole_moment(centre_of_mass, molecule.charges, molecule.coordinates, P, SCF_output.D)
    
    # Calculates the analytical quadrupole moment

    isotropic_quadrupole_moment, nuclear_quadrupole_moment, anisotropic_quadrupole_moment = calculate_analytical_quadrupole_moment(centre_of_mass, molecule.charges, molecule.coordinates, P, SCF_output.Q)

    # Only print a multipole orientation diagram if the multipole is significant


    def format_moment_structure(value: float, positive_diagram: str, negative_diagram: str) -> str:
        
        """
        
        Formats the multipole moment molecular structure diagram consistently.

        """

        if value > constants.MOMENT_THRESH:
            
            text = f"  {molecule.molecular_structure}  {positive_diagram}"
        
        elif value < -constants.MOMENT_THRESH:
            
            text = f"  {molecule.molecular_structure}  {negative_diagram}"

        else:
            
            text = f"      {molecule.molecular_structure}      "
        
        return text.center(25)


    # Creates consistent dipole and quadrupole moment diagrams

    dipole_molecular_structure = format_moment_structure(total_dipole_moment, "+--->   ", "<---+   ")

    quadrupole_molecular_structure = format_moment_structure(isotropic_quadrupole_moment, "+-> <-+   ", "<--+-->  ")


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("                    Dipole Moment                                        Quadrupole Moment", calculation, 2, colour = "white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  Nuclear: {nuclear_dipole_moment:11.7f}     Electronic: {electronic_dipole_moment:11.7f}       Nuclear: {nuclear_quadrupole_moment:11.7f}   Anisotropic: {anisotropic_quadrupole_moment:11.7f}\n", calculation, 2)
    
    log(f"  Total: {total_dipole_moment:11.7f}      {dipole_molecular_structure}      Isotropic: {isotropic_quadrupole_moment:11.7f}  {quadrupole_molecular_structure}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)


    return










def calculate_koopmans_parameters(epsilons: ndarray, n_occ: int, calculation: Calculation) -> tuple:

    """

    Calculates the Koopmans' theorem parameters of a system (ionisation energy, electron affinity and HOMO-LUMO gap).

    Args:   
        epsilons (array): Fock matrix eigenvalues
        n_occ (int): Number of occupied orbitals
        calculation (Calculation): Calculation object

    Returns:
        ionisation_potential (float): Ionisation energy
        electron_affinity (float): Electron affinity
        band_gap (float): Difference in energy between HOMO and LUMO

    """

    # IP = -HOMO

    ionisation_potential = -1 * epsilons[n_occ - 1]

    # As long as LUMO exists, EA = -LUMO    

    if len(epsilons) > n_occ: 
    
        electron_affinity = -1 * epsilons[n_occ]

        band_gap = ionisation_potential - electron_affinity
        
    else: 
    
        electron_affinity = band_gap = " --------"
        
        warning("Size of basis is too small for electron affinity calculation!")

    if not isinstance(electron_affinity, str): 
        
        electron_affinity = f"{electron_affinity:9.6f}"
        band_gap = f"{band_gap:9.6f}"
       
    log(f"\n Koopmans' theorem ionisation potential:  {ionisation_potential:9.6f}", calculation, 2)
    log(f" Koopmans' theorem electron affinity:     {electron_affinity}", calculation, 2)
    log(f" Energy gap between HOMO and LUMO:        {band_gap}", calculation, 2)


    return ionisation_potential, electron_affinity, band_gap
 
 








def print_energy_components(SCF_output: Output, V_NN: float, calculation: Calculation, silent: bool = False) -> None:

    """

    Prints the various components of the self-consistent field energy to the terminal.

    Args:   
        SCF_output (Output): Output object
        V_NN (float): Nuclear-nuclear repulsion energy
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    """

    # Adds up different energy components

    one_electron_energy = SCF_output.nuclear_electron_energy + SCF_output.kinetic_energy + SCF_output.electric_field_energy + SCF_output.electric_field_gradient_energy 
    two_electron_energy = SCF_output.exchange_energy + SCF_output.coulomb_energy + SCF_output.correlation_energy

    electronic_energy = one_electron_energy + two_electron_energy

    total_energy = electronic_energy + V_NN
    
    # Calculates Virial ratio between potential and kinetic energy

    virial_ratio = -1 * (total_energy - SCF_output.kinetic_energy) / SCF_output.kinetic_energy
           
    log_spacer(calculation, priority=2, silent = silent)
    log("                  Energy Components       ", calculation, 2, colour = "white", silent = silent)
    log_spacer(calculation, priority=2, silent = silent)    

    log(f"  Kinetic energy:                   {SCF_output.kinetic_energy:15.10f}", calculation, 2, silent = silent)
    log(f"  Coulomb energy:                   {SCF_output.coulomb_energy:15.10f}", calculation, 2, silent = silent)
    log(f"  Exchange energy:                  {SCF_output.exchange_energy:15.10f}", calculation, 2, silent = silent)

    if calculation.method.density_functional_method:
        
        log(f"  Correlation energy:               {SCF_output.correlation_energy:15.10f}", calculation, 2, silent = silent)

    log(f"  Nuclear repulsion energy:         {V_NN:15.10f}", calculation, 2, silent = silent)
    log(f"  Nuclear attraction energy:        {SCF_output.nuclear_electron_energy:15.10f}", calculation, 2, silent = silent)      

    if np.linalg.norm(calculation.electric_field) > 0:
    
        log(f"  Electric field energy:            {SCF_output.electric_field_energy:15.10f}", calculation, 2, silent = silent)
    
    if np.linalg.norm(calculation.electric_field_gradient) > 0:
    
        log(f"  Electric field gradient energy:   {SCF_output.electric_field_gradient_energy:15.10f}", calculation, 2, silent = silent)

    log(f"\n  One-electron energy:              {one_electron_energy:15.10f}", calculation, 2, silent = silent)
    log(f"  Two-electron energy:              {two_electron_energy:15.10f}", calculation, 2, silent = silent)

    if calculation.method.density_functional_method:
        
        log(f"  Exchange-correlation energy:      {SCF_output.exchange_energy + SCF_output.correlation_energy:15.10f}", calculation, 2, silent = silent)

    log(f"  Electronic energy:                {electronic_energy:15.10f}\n", calculation, 2, silent = silent)
    log(f"  Virial ratio:                     {virial_ratio:15.10f}\n", calculation, 2, silent = silent)
            
    log(f"  Total energy:                     {total_energy:15.10f}", calculation, 2, silent = silent)

    log_spacer(calculation, priority=2, silent = silent)

    return










def calculate_spin_contamination(P_alpha: ndarray, P_beta: ndarray, n_alpha: int, n_beta: int, S: ndarray, calculation: Calculation, kind: str, silent: bool = False) -> None:

    """

    Calculates and prints theoretical spin squared operator and spin contamination.

    Args:
        P_alpha (array): Density matrix of alpha electrons in AO basis
        P_beta (array): Density matrix of beta electrons in AO basis
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        S (array): Overlap matrix in AO basis
        calculation (Calculation): Calculation object
        kind (str): Either "UHF", "UKS", "MP2" or "Coupled cluster"
        silent (bool, optional): Should anything be printed

    """

    s_squared_exact = (n_alpha - n_beta) / 2 * ((n_alpha - n_beta) / 2 + 1)

    # Contraction to calculate spin contamination

    spin_contamination = n_beta - np.trace(P_alpha.T @ S @ P_beta.T @ S)
    
    s_squared = s_squared_exact + spin_contamination

    # Only print by default for unrestricted SCF

    priority = 2 if kind in ["UHF", "UKS"] else 3

    title = kind.title() if kind == "Coupled cluster" else kind

    space1, space2 = ("       ", "            ") if len(kind) == 3 else ("", "")
    
    log_spacer(calculation, silent = silent, priority = priority)
    log(f"   {space1}       {title} Spin Contamination       ", calculation, priority, silent = silent, colour = "white")
    log_spacer(calculation, silent = silent, priority = priority)

    log(f"  Exact S^2 expectation value:            {s_squared_exact:9.6f}", calculation, priority, silent = silent)
    log(f"  {kind} S^2 expectation value:  {space2}{s_squared:9.6f}", calculation, priority, silent = silent)
    log(f"\n  Spin contamination:                     {spin_contamination:9.6f}", calculation, priority, silent = silent)

    log_spacer(calculation, silent = silent, priority = priority, end="\n")

    return










def calculate_and_print_population_analysis(P: ndarray, S: ndarray, R: ndarray, partition_ranges: ndarray, atomic_symbols: list, charges: ndarray, calculation: Calculation) -> None:

    """

    Calculates the bond order, atomic charges and valences for Mulliken, Lowden and Mayer population analysis

    Args:   
        P (array): Density matrix in AO basis
        S (array): Overlap matrix in AO basis
        R (array): Spin density matrix in AO basis
        partition_ranges (array): Separates basis onto each atom
        atomic_symbols (list): Atomic symbols
        charges (array): Nuclear charges
        calculation (Calculation): Calculation object

    """

    PS = P @ S
    RS = R @ S

    # Diagonalises overlap matrix to form density matrix in orthogonalised Lowdin basis

    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    P_Lowdin = S_sqrt @ P @ S_sqrt

    # Slices of orbital indices on atoms A and B

    A = slice(0, partition_ranges[0])
    B = slice(partition_ranges[0], partition_ranges[0] + partition_ranges[1])

    # Sums over the ranges of each atomic orbital over atom A and B to build the three bond orders

    bond_order_Mayer = np.sum(PS[A, B] * PS[B, A].T + RS[A, B] * RS[B, A].T)
    bond_order_Lowdin = np.sum(P_Lowdin[A, B] ** 2)
    bond_order_Mulliken = 2 * np.sum(P[A, B] * S[A, B])

    # Sums over the corresponding ranges of atomic orbitals in the density matrix, to build the populations

    populations_Mulliken = [np.trace(PS[A, A]), np.trace(PS[B, B])]
    populations_Lowdin = [np.trace(P_Lowdin[A, A]), np.trace(P_Lowdin[B, B])]
    total_valences = [np.einsum("ij,ji->", PS[A, A], PS[A, A]), np.einsum("ij,ji->", PS[B, B], PS[B, B])]

    # Atomic charges are nuclear charges minus electronic populations

    charges_Mulliken = charges - populations_Mulliken
    charges_Lowdin = charges - populations_Lowdin
    total_valences = 2 * np.array(populations_Mulliken) - np.array(total_valences)

    # Adds up total charges and calculates free valences from total and bonded valences

    total_charges_Mulliken = np.sum(charges_Mulliken)
    total_charges_Lowdin = np.sum(charges_Lowdin)
    free_valences = np.array(total_valences) - bond_order_Mayer

    # Prints the population analysis information

    atoms_formatted = []

    for atomic_symbol in atomic_symbols:
    
        atomic_symbol = atomic_symbol.lower().capitalize()
        atomic_symbol = atomic_symbol + "  :" if len(atomic_symbol) == 1 else atomic_symbol + " :"
        atoms_formatted.append(atomic_symbol)

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("      Mulliken Charges                Lowdin Charges                Mayer Free, Bonded, Total Valence", calculation, 2, colour = "white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  {atoms_formatted[0]} {charges_Mulliken[0]:8.5f}                  {atoms_formatted[0]} {charges_Lowdin[0]:8.5f}                  {atoms_formatted[0]} {free_valences[0]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[0]:8.5f}", calculation, 2)
    log(f"  {atoms_formatted[1]} {charges_Mulliken[1]:8.5f}                  {atoms_formatted[1]} {charges_Lowdin[1]:8.5f}                  {atoms_formatted[1]} {free_valences[1]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[1]:8.5f}", calculation, 2)
    log(f"\n  Sum of charges: {total_charges_Mulliken:8.5f}       Sum of charges: {total_charges_Lowdin:8.5f}", calculation, 2) 
    log(f"  Bond order: {bond_order_Mulliken:8.5f}           Bond order: {bond_order_Lowdin:8.5f}           Bond order: {bond_order_Mayer:8.5f}", calculation, 2) 
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)


    return










def print_molecular_orbital_eigenvalues(calculation: Calculation, SCF_output: Output, occupancies: list, spin_labels: list) -> None:

    """

    Prints the Fock matrix eigenvalues.

    Args:   
        calculation (Calculation): Calculation object
        SCF_output (Output): Output object   
        occupancies (list): Orbital occupancies
        spin_labels (list): Orbital spin labels

    """

    priority = 1 if calculation.print_molecular_orbitals else 3

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, priority)
    log("                 Molecular Orbital Eigenvalues", calculation, priority = priority, colour = "white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, priority)
    log("   N        Occupancy           Spin                 Energy", calculation, priority)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, priority)

    orbital_energies = SCF_output.epsilons

    # Prints the spin for unrestricted calculations

    if calculation.reference == "RHF":
        
        spin_labels_words = ["----"] * len(orbital_energies)

    else:
        
        spin_labels_words = [{"a": "Alpha", "b": "Beta"}.get(item, item) for item in spin_labels]

    # Prints out the molecular orbital information

    for mo in range(len(orbital_energies)):

       log(f" {mo + 1:3.0f}         {occupancies[mo]:7.5f}            {spin_labels_words[mo]:<6}         {orbital_energies[mo]:16.10f}", calculation, priority)
     
    log(f"", calculation, priority)

    return









def print_molecular_orbital_coefficients(calculation: Calculation, molecule: Molecule, SCF_output: Output, occupancies: list, spin_labels: list, natural_orbitals: ndarray = None, natural_occupancies: ndarray = None) -> None:
    
    """
    Prints out molecular orbital coefficients, formatted very carefully.

    Args:
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        SCF_output (Output): Output object
        occupancies (list): Orbital occupancies
        spin_labels (list): Orbital spin labels
        natural_orbitals (array): Natural orbitals
        natural_occupancies (array): Natural occupancies

    """

    # Natural orbitals will be printed if calculated

    do_natorbs = natural_orbitals is not None

    priority = 1 if calculation.print_molecular_orbitals else 3

    if do_natorbs:
        
        log("                   Natural Orbital Coefficients", calculation, priority, colour = "white")

    else: 
        
        log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, priority)
        log("                 Molecular Orbital Coefficients", calculation, priority, colour = "white")
    
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, priority)

    # These are sorted by energy, so interleaved for unrestricted calculations - same as for plotting

    molecular_orbitals = SCF_output.molecular_orbitals
    orbital_energies = SCF_output.epsilons
    
    orbitals = natural_orbitals if do_natorbs else molecular_orbitals

    # Mappings from Cartesian to spherical harmonic counting

    starting_n = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5, "h": 6}
    capacity = {"s": 1, "p": 3, "d": 6, "f": 10, "g": 15, "h": 21}

    if calculation.cartesian_harmonics:   # Handles the "CARTHARM" case

        components = {

            "s": [""],
            "p": ["x", "y", "z"],
            "d": ["xx", "xy", "xz", "yy", "yz", "zz"],
            "f": ["xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"],
            "g": [f"c{i}" for i in range(1, 16)],
            "h": [f"c{i}" for i in range(1, 22)], 

        }

    else:   # Handles the usual case, in spherical harmonics

        components = {

            "s": [""],
            "p": ["x", "y", "z"],
            "d": ["xy", "xz", "yz", "xxyy", "zz"],
            "f": ["-3", "-2", "-1", "0", "+1", "+2", "+3"],
            "g": ["-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4"],
            "h": ["-5", "-4", "-3", "-2", "-1", "0", "+1", "+2", "+3", "+4", "+5"],

        }
    
    current_n = starting_n.copy()
    all_orbitals, all_components = [], []
    atom_1_cutoff = molecule.partition_ranges[0]
    
    i = 0

    # Loops through atomic orbitals, transforming lists to spherical harmonic counting if needed

    while i < len(molecule.angular_momentum_list):

        if len(all_orbitals) == atom_1_cutoff:

            current_n = starting_n.copy()
            
        l = molecule.angular_momentum_list[i]

        n = current_n[l]
        
        for comp in components[l]:

            # Creates orbital list as eg. "1s", "2s"

            all_orbitals.append(f"{n}{l}")
            all_components.append(comp)
            
        i += capacity[l] 

        current_n[l] += 1

    # Atomic orbital lists - "1s", "2s", "2p", "2p", "2p", etc.

    orbitals_on_atom_1 = all_orbitals[:atom_1_cutoff]
    orbitals_on_atom_2 = all_orbitals[atom_1_cutoff:]
    
    # Angular momenta - "", "", "x", "y", "z", etc.

    angular_momentum_on_atom_1 = all_components[:atom_1_cutoff]
    angular_momentum_on_atom_2 = all_components[atom_1_cutoff:]

    # Makes ranges, accounting for "NATORBS" and single atom calculations

    ao_range = max(molecule.partition_ranges[0], molecule.partition_ranges[1]) if len(molecule.atoms) > 1 else molecule.partition_ranges[0]
    mo_range = SCF_output.molecular_orbitals_alpha.shape[1] if do_natorbs else len(orbital_energies)

    # Changes occupancy list to words

    occupancies = ["Occupied" if occ in [1, 2] else "Virtual " for occ in occupancies]

    orbital_abbreviation = "NO" if do_natorbs else "MO"

    # Loops over molecular orbitals

    for mo in range(min(mo_range, calculation.n_orbitals_to_print)):

        log(f"\n  {orbital_abbreviation} {mo + 1} ", calculation, priority, end = "")

        if not do_natorbs:

            log(f"{"~~~" if mo + 1 < 10 else "~~"} {occupancies[mo]}", calculation, priority, end = "")

        else:
            
            log(f"{" " if mo + 1 < 10 else ""}", calculation, priority, end = "")
        
        # Prints occupancy of molecular orbitals

        if calculation.reference == "UHF" and not do_natorbs:

            if occupancies[mo] == "Occupied":
              
                log(f" ~~~ {"Alpha"}", calculation, priority, end = "") if spin_labels[mo] == "a" else log(f" ~~~~ {"Beta"}", calculation, priority, end = "")
            
            else:

                log(f"~~~~ {"Alpha"}", calculation, priority, end = "") if spin_labels[mo] == "a" else log(f"~~~~~ {"Beta"}", calculation, priority, end = "")
 
        else:
      
            log(f"          ", calculation, priority, end = "")

        if do_natorbs:   # Prints natural orbital occupancy
            
            log(f"                           N = {natural_occupancies[mo]:14.10f}", calculation, priority, end = "\n\n")

        else:   # Prints molecular orbital energy

            log(f"                E = {orbital_energies[mo]:14.10f}", calculation, priority, end = "\n\n")

        # Loops over atomic orbitals in molecular orbital

        for ao in range(ao_range):

            # Only look at the atomic orbitals on the first atom

            orbital_1_coeff = orbitals.T[mo][:molecule.partition_ranges[0]]

            first_atom = f"{molecule.atoms[0].symbol_formatted:<4}" if ao == 0 else "    "

            log(f"   {first_atom}", calculation, priority, end = "")
            
            if ao < molecule.partition_ranges[0]:   # Only print orbitals on first atom

                log(f"{orbitals_on_atom_1[ao]} {angular_momentum_on_atom_1[ao]:<4}  : ", calculation, priority, end = "")

                log(f"{orbital_1_coeff[ao]:11.5f}", calculation, priority, end = "")

            else:

                log("                    ", calculation, priority, end = "")
            
            # Allows atomic calculations
            
            if len(molecule.atoms) > 1:   
                
                # Only look at the atomic orbitals on the second atom

                orbital_2_coeff = orbitals.T[mo][molecule.partition_ranges[0]:]

                second_atom = f"{molecule.atoms[1].symbol_formatted:<4}" if ao == 0 else "    "

                log(f"        {second_atom}", calculation, priority, end = "")
        
                if ao < molecule.partition_ranges[1]:   # Only print orbitals on second atom

                    log(f"{orbitals_on_atom_2[ao]} {angular_momentum_on_atom_2[ao]:<4}  : ", calculation, priority, end = "")

                    log(f"{orbital_2_coeff[ao]:11.5f}", calculation, priority)

                else:

                    log("", calculation, priority)
            else:

                log("", calculation, priority)

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, priority)

    return










def print_density_information(calculation: Calculation) -> None:

    """
    
    Prints the type of density matrix used in property calculations.

    Args:
        calculation (Calculation): Calculation object

    """

    method = calculation.method

    density_type = "relaxed" if calculation.relaxed_density else "unrelaxed"

    # Specifies which density matrix is used for the property calculations
    
    match calculation.method.name:

        case "MP2" | "SCS-MP2": 

            log(f"\n Using the MP2 {density_type} density for property calculations.", calculation, 1)

        case "OMP2":

            log("\n Using the orbital-optimised MP2 relaxed density for property calculations.", calculation, 1)

        case "AO-MP2":

            warning("Using the Hartree-Fock density, not the MP2 density, for property calculations.")

        case "CCSD[T]" | "CCSD(T)":

            warning("Using the linearised CCSD density for property calculations.")

        case "QCISD[T]" | "QCISD(T)":

            warning("Using the linearised QCISD density for property calculations.")

    if method.method_base in ["MP3", "MP4"]:

        warning(f"Using the {density_type} MP2 density for property calculations.")

    elif method.coupled_cluster_method:
    
        log("\n Using the linearised coupled cluster density for property calculations.", calculation, 1)

    elif method.excited_state_method or calculation.time_dependent: 
    
        if method.density_functional_method:

            log(f"\n Using the unrelaxed TD-DFT density for property calculations.", calculation, 1)
        
        else:
            
            log(f"\n Using the unrelaxed TD-HF density for property calculations.", calculation, 1)

    if method.density_functional_method and calculation.MPC_prop != 0 and not calculation.time_dependent:

        log(f"\n Using the double-hybrid {density_type} density for property calculations.", calculation, 1)


    return










def calculate_molecular_properties(molecule: Molecule, calculation: Calculation, P: ndarray, S: ndarray, SCF_output: Output, P_alpha: ndarray, P_beta: ndarray, print_orbitals: bool = True, natural_orbitals: ndarray = None, natural_occupancies: ndarray = None) -> None:

    """

    Calculates various TUNA properties after an energy evaluation and prints them to the console.

    Args:   
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        P (array): Density matrix in AO basis
        S (array): Overlap matrix in AO basis
        SCF_output (Output): Output object
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        print_orbitals (bool): Should orbitals be printed
        natural_orbitals (array): Natural orbitals
        natural_occupancies (array): Natural orbital occupancies

    """

    log("\n Beginning calculation of TUNA properties... ", calculation, 3)

    # Prints information about the analytical density used

    print_density_information(calculation)

    if print_orbitals:

        # The list of spins for each spin orbital - only relevant for unrestricted references

        spin_labels = ["a"] * SCF_output.molecular_orbitals_alpha.shape[1] + ["b"] * SCF_output.molecular_orbitals_beta.shape[1]

        spin_labels_sorted = [spin_labels[i] for i in np.argsort(SCF_output.epsilons_combined)]

        # Builds a list of orbital occupancies

        if calculation.reference == "RHF":

            occupancies = [2] * molecule.n_doubly_occ + [0] * (len(SCF_output.epsilons) - molecule.n_doubly_occ)

        else:

            occupancies = [1] * molecule.n_occ + [0] * (len(SCF_output.epsilons_combined) - molecule.n_occ)

        # Prints molecular orbital eigenvalues and coefficients

        print_molecular_orbital_eigenvalues(calculation, SCF_output, occupancies, spin_labels_sorted)
        
        print_molecular_orbital_coefficients(calculation, molecule, SCF_output, occupancies, spin_labels_sorted)

        # Print natural orbital information

        if natural_orbitals is not None:
            
            print_molecular_orbital_coefficients(calculation, molecule, SCF_output, occupancies, spin_labels_sorted, natural_orbitals, natural_occupancies)

        # Prints Koopmans' theorem parameters if RHF reference is used

        if calculation.reference == "RHF":
            
            calculate_koopmans_parameters(SCF_output.epsilons, molecule.n_doubly_occ, calculation)

    # As long as there are two real atoms present, calculates rotational constant and dipole moment information

    if calculation.diatomic:

        # Prints the rotational constant

        mol.calculate_and_print_rotational_constant(molecule.reduced_mass, molecule.bond_length, calculation)

        # Prints the analytical dipole and quadrupole moments

        calculate_and_print_multipole_moments(P, molecule, SCF_output, calculation)

        # Builds spin density matrix

        R = P_alpha - P_beta if molecule.n_alpha + molecule.n_beta != 1 else P

        # Calculate population analysis, format all the data, then print to console

        calculate_and_print_population_analysis(P, S, R, molecule.partition_ranges, molecule.atomic_symbols, molecule.charges, calculation)
        

    return