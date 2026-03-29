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
    electronic_dipole_moment = -1 * np.einsum("ij,ij->", P, D[2], optimize=True)

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
        total_dipole_moment (float): Total molecular dipole moment in atomic units
        nuclear_dipole_moment (float): Nuclear dipole moment in atomic units
        electronic_dipole_moment (float): Electronic dipole moment in atomic units

    """


    nuclear_quadrupole_moment = calculate_nuclear_quadrupole_moment(centre_of_mass, charges, coordinates)       
    
    # Extracts the xx and zz components of quadrupole moment integrals

    electronic_quadrupole_moment_xx = -1 * np.einsum("ij,ij->", P, Q[0], optimize=True)
    electronic_quadrupole_moment_zz = -1 * np.einsum("ij,ij->", P, Q[1], optimize=True)

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

        width, boundary = 25, 1e-5

        if value > boundary:
            
            text = f"  {molecule.molecular_structure}  {positive_diagram}"
        
        elif value < -boundary:
            
            text = f"  {molecule.molecular_structure}  {negative_diagram}"

        else:
            
            text = f"      {molecule.molecular_structure}      "
        
        return text.center(width)


    # Creates consistent dipole and quadrupole moment diagrams

    dipole_molecular_structure = format_moment_structure(total_dipole_moment, "+--->   ", "<---+   ")

    quadrupole_molecular_structure = format_moment_structure(isotropic_quadrupole_moment, "+-> <-+   ", "<--+-->  ")


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("                    Dipole Moment                                        Quadrupole Moment", calculation, 2, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  Nuclear: {nuclear_dipole_moment:11.7f}     Electronic: {electronic_dipole_moment:11.7f}       Nuclear: {nuclear_quadrupole_moment:11.7f}   Anisotropic: {anisotropic_quadrupole_moment:11.7f}\n", calculation, 2)
    
    log(f"  Total: {total_dipole_moment:11.7f}      {dipole_molecular_structure}      Isotropic: {isotropic_quadrupole_moment:11.7f}  {quadrupole_molecular_structure}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)


    return










def calculate_Koopmans_parameters(epsilons: ndarray, n_occ: int, calculation: Calculation) -> tuple:

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

    one_electron_energy = SCF_output.nuclear_electron_energy + SCF_output.kinetic_energy + SCF_output.electric_field_energy
    two_electron_energy = SCF_output.exchange_energy + SCF_output.coulomb_energy + SCF_output.correlation_energy

    electronic_energy = one_electron_energy + two_electron_energy

    total_energy = electronic_energy + V_NN
    
    # Calculates Virial ratio between potential and kinetic energy

    virial_ratio = -1 * (total_energy - SCF_output.kinetic_energy) / SCF_output.kinetic_energy
           
    log_spacer(calculation, priority=2, silent=silent)
    log("                  Energy Components       ", calculation, 2, colour="white", silent=silent)
    log_spacer(calculation, priority=2, silent=silent)    

    log(f"  Kinetic energy:                   {SCF_output.kinetic_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Coulomb energy:                   {SCF_output.coulomb_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Exchange energy:                  {SCF_output.exchange_energy:15.10f}", calculation, 2, silent=silent)

    if calculation.method.density_functional_method:
        
        log(f"  Correlation energy:               {SCF_output.correlation_energy:15.10f}", calculation, 2, silent=silent)

    log(f"  Nuclear repulsion energy:         {V_NN:15.10f}", calculation, 2, silent=silent)
    log(f"  Nuclear attraction energy:        {SCF_output.nuclear_electron_energy:15.10f}", calculation, 2, silent=silent)      

    if np.linalg.norm(calculation.electric_field) > 0:
    
        log(f"  Electric field energy:            {SCF_output.electric_field_energy:15.10f}", calculation, 2, silent=silent)

    log(f"\n  One-electron energy:              {one_electron_energy:15.10f}", calculation, 2, silent=silent)
    log(f"  Two-electron energy:              {two_electron_energy:15.10f}", calculation, 2, silent=silent)

    if calculation.method.density_functional_method:
        
        log(f"  Exchange-correlation energy:      {SCF_output.exchange_energy + SCF_output.correlation_energy:15.10f}", calculation, 2, silent=silent)

    log(f"  Electronic energy:                {electronic_energy:15.10f}\n", calculation, 2, silent=silent)
    log(f"  Virial ratio:                     {virial_ratio:15.10f}\n", calculation, 2, silent=silent)
            
    log(f"  Total energy:                     {total_energy:15.10f}", calculation, 2, silent=silent)

    log_spacer(calculation, priority=2, silent=silent)

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

    s_squared_exact = ((n_alpha - n_beta) / 2) * ((n_alpha - n_beta) / 2 + 1)

    # Contraction to calculate spin contamination

    spin_contamination = n_beta - np.trace(P_alpha.T @ S @ P_beta.T @ S)
    
    s_squared = s_squared_exact + spin_contamination

    # Only print by default for unrestricted SCF

    priority = 2 if kind in ["UHF", "UKS"] else 3

    title = kind.title() if kind == "Coupled cluster" else kind

    space1, space2 = ("       ", "            ") if len(kind) == 3 else ("", "")
    
    log_spacer(calculation, silent=silent, priority=priority)
    log(f"   {space1}       {title} Spin Contamination       ", calculation, priority, silent=silent, colour="white")
    log_spacer(calculation, silent=silent, priority=priority)

    log(f"  Exact S^2 expectation value:            {s_squared_exact:9.6f}", calculation, priority, silent=silent)
    log(f"  {kind} S^2 expectation value:  {space2}{s_squared:9.6f}", calculation, priority, silent=silent)
    log(f"\n  Spin contamination:                     {spin_contamination:9.6f}", calculation, priority, silent=silent)

    log_spacer(calculation, silent=silent, priority=priority)


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
    log("      Mulliken Charges                Lowdin Charges                Mayer Free, Bonded, Total Valence", calculation, 2, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  {atoms_formatted[0]} {charges_Mulliken[0]:8.5f}                  {atoms_formatted[0]} {charges_Lowdin[0]:8.5f}                  {atoms_formatted[0]} {free_valences[0]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[0]:8.5f}", calculation, 2)
    log(f"  {atoms_formatted[1]} {charges_Mulliken[1]:8.5f}                  {atoms_formatted[1]} {charges_Lowdin[1]:8.5f}                  {atoms_formatted[1]} {free_valences[1]:8.5f},  {bond_order_Mayer:8.5f},  {total_valences[1]:8.5f}", calculation, 2)
    log(f"\n  Sum of charges: {total_charges_Mulliken:8.5f}       Sum of charges: {total_charges_Lowdin:8.5f}", calculation, 2) 
    log(f"  Bond order: {bond_order_Mulliken:8.5f}           Bond order: {bond_order_Lowdin:8.5f}           Bond order: {bond_order_Mayer:8.5f}", calculation, 2) 
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)


    return










def print_molecular_orbital_eigenvalues(calculation: Calculation, molecule: Molecule, SCF_output: Output) -> None:

    """

    Prints the Fock matrix eigenvalues, separately for UHF references.

    Args:   
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        SCF_output (Output): Output object   

    """

    log_spacer(calculation, 3, start="\n")
    log("           Molecular Orbital Eigenvalues", calculation, 3, colour="white")
    log_spacer(calculation, 3)

    epsilons, epsilons_alpha, epsilons_beta = SCF_output.epsilons, SCF_output.epsilons_alpha, SCF_output.epsilons_beta

    # Prints alpha and beta eigenvalues separately

    if calculation.reference == "UHF":

        if molecule.n_beta > 0:

            log("\n    Alpha Eigenvalues           Beta Eigenvalues\n", calculation, 3)
            
            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)
            log("   N    Occ.   Epsilon         N    Occ.   Epsilon  ", calculation, 3)
            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~", calculation, 3)
            
            # Occupied orbitals are alpha electrons only

            occupancies_alpha = [1] * molecule.n_alpha + [0] * int((len(epsilons_alpha) - molecule.n_alpha))
            occupancies_beta = [1] * molecule.n_beta + [0] * int((len(epsilons_beta) - molecule.n_beta))

            for i, (epsilon_alpha, epsilon_beta) in enumerate(zip(epsilons_alpha, epsilons_beta)):

                log(f"  {(i + 1):2.0f}     {occupancies_alpha[i]}   {epsilon_alpha:10.6f}       {(i + 1):2.0f}     {occupancies_beta[i]}   {epsilon_beta:10.6f}", calculation, 3)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 3)


        else:

            log("\n  Alpha eigenvalues:\n", calculation, 3)
            
            log("  ~~~~~~~~~~~~~~~~~~~~~~~  ", calculation, 3)
            log("    N    Occ.   Epsilon     ", calculation, 3)
            log("  ~~~~~~~~~~~~~~~~~~~~~~~   ", calculation, 3)
            
            # Occupied orbitals are alpha electrons only

            occupancies_alpha = [1] * molecule.n_alpha + [0] * int((len(epsilons_alpha) - molecule.n_alpha))

            for i, epsilon_alpha in enumerate(epsilons_alpha):

                log(f"   {(i + 1):2.0f}     {occupancies_alpha[i]}   {epsilon_alpha:10.6f}    ", calculation, 3)

            log("  ~~~~~~~~~~~~~~~~~~~~~~~  \n", calculation, 3)


    elif calculation.reference == "RHF":

        log("    N            Occupation             Epsilon ", calculation, 3)
        log_spacer(calculation, 3)

        # Occupied orbitals (doubly occupied) depend on number electron pairs

        occupancies = [2] * molecule.n_doubly_occ + [0] * int((len(epsilons) - molecule.n_doubly_occ))

        for i, epsilon in enumerate(epsilons):

            log(f"   {(i + 1):2.0f}                {occupancies[i]}                {epsilon:10.6f}", calculation, 3)


    return










def print_molecular_orbital_coefficients(calculation: Calculation, molecule: Molecule, SCF_output: Output) -> None:

    """

    Prints the coefficients of all molecular orbitals, for both alpha and beta spins for UHF.

    Args:
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        SCF_output (Output): Output object

    """

    log_spacer(calculation, 3, start="")
    log("           Molecular Orbital Coefficients", calculation, 3, colour="white")
    log_spacer(calculation, 3)

    # Build per-orbital atom symbol and subshell label lists

    symbol_list, n_list = [], []
    switch_value = 0

    for i, atom in enumerate(molecule.partitioned_basis_functions):

        for j, _ in enumerate(atom):

            symbol_list.append(molecule.atomic_symbols[i])
            n_list.append(j + 1)

            if i == 1 and j == 0:

                switch_value = len(symbol_list) - 1



    def format_angular_momentum_list(angular_momentum_list: list, partition_ranges: list) -> list:

        """

        Formats the angular momentum list with per-atom subshell numbering.

        Args:
            angular_momentum_list (list): List of angular momentum strings
            partition_ranges (array): Ranges of basis functions partitioned over atomic centres

        Returns:
            formatted (list): Formatted angular momentum list, with degeneracies considered

        """

        l_of = {ch: i for i, ch in enumerate("spdfghi")}

        def min_n(letter): 
            
            return l_of.get(letter, 0) + 1

        def degeneracy(letter):

            l = l_of.get(letter, 0)

            return (l + 1) * (l + 2) / 2

        formatted, idx = [], 0

        for num_orbitals in partition_ranges:

            state = {}

            for _ in range(num_orbitals):

                # Converts an index into s, p, d, etc.

                letter = str(angular_momentum_list[idx]).lower()

                n, used = state.get(letter, (min_n(letter), 0))

                if used >= degeneracy(letter):

                    n, used = n + 1, 0
                
                used += 1
                    
                state[letter] = (n, used)

                # Determines the orbital with the principal quantum number and angular momentum

                formatted.append(f"{n}{letter}")

                idx += 1

        return formatted




    def format_molecular_orbitals(symbol_list: list, k: int, switch_value: int, atoms: list, calculation: Calculation, has_printed_1: bool, has_printed_2: bool) -> tuple:

        """

        Manages ghost atoms and formats the list of atoms to be printed.

        """

        sym = symbol_list[k]

        sym = ("X" + sym.split("X")[1].lower().capitalize()) if "X" in sym else sym.lower().capitalize()

        if len(sym) == 1: 

            sym += " "

        if k < switch_value:

            if not has_printed_1: has_printed_1 = True

            else: sym = "  "

        else:

            if not has_printed_2: has_printed_2 = True

            else: sym = "  "

        if k == switch_value and len(atoms) == 2:

            log("", calculation, 3)

        symbol_list[k] = sym

        return sym, has_printed_1, has_printed_2


    # Format the orbital to "2p", "3d", etc.

    formatted_ang_mom = format_angular_momentum_list(molecule.angular_momentum_list, molecule.partition_ranges)


    def print_coeffs(molecular_orbitals: ndarray, eps: ndarray, n_occ: int) -> None:

        """
        
        Prints MO coefficients for RHF.
        
        """

        for mo in range(min(len(eps), molecule.n_alpha + 10)):

            occ = "(Occupied)" if n_occ > mo else "(Virtual)"

            log(f"\n   MO {mo+1} {occ}\n", calculation, 3)

            has_printed_1 = has_printed_2 = False

            for k in range(len(molecular_orbitals.T[mo])):

                try:
                    symbol_list[k], has_printed_1, has_printed_2 = format_molecular_orbitals(symbol_list, k, switch_value, molecule.atomic_symbols, calculation, has_printed_1, has_printed_2)
                    
                    log("    " + symbol_list[k] + f"  {formatted_ang_mom[k]}  :  {molecular_orbitals.T[mo][k]:7.4f}", calculation, 3)
                
                except: 
                    
                    pass

    # For unrestricted references, print the alpha and beta coefficients separately

    if calculation.reference == "UHF":

        show_beta = molecule.n_beta > 0

        header = "\n Alpha coefficients:          Beta coefficients:" if show_beta else "\n Alpha coefficients:         "
        log(header, calculation, 3)

        for mo in range(min(len(SCF_output.epsilons_alpha), molecule.n_alpha + 10)):

            occ_a = "(Occupied)" if molecule.n_alpha > mo else "(Virtual)"

            if show_beta:

                occ_b = "(Occupied)" if molecule.n_beta > mo else "(Virtual)"

                log(f"\n  MO {mo+1} {occ_a}              MO {mo+1} {occ_b}\n", calculation, 3)

            else:

                log(f"\n   MO {mo+1} {occ_a}      \n", calculation, 3)

            has_printed_1 = has_printed_2 = False

            for k in range(len(SCF_output.molecular_orbitals.T[mo])):

                try:

                    symbol_list[k], has_printed_1, has_printed_2 = format_molecular_orbitals(symbol_list, k, switch_value, molecule.atomic_symbols, calculation, has_printed_1, has_printed_2)

                    line = "   " + symbol_list[k] + f"  {formatted_ang_mom[k]}  :  {SCF_output.molecular_orbitals_alpha.T[mo][k]:7.4f}"
                    
                    if show_beta:

                        line += "           " + symbol_list[k] + f"  {formatted_ang_mom[k]}  :  {SCF_output.molecular_orbitals_beta.T[mo][k]:7.4f}"
                    
                    log(line, calculation, 3)

                except: 
                    
                    pass

    else:

        # Only one set of coefficients for restricted references

        print_coeffs(SCF_output.molecular_orbitals, SCF_output.epsilons, molecule.n_doubly_occ)

    log_spacer(calculation, 3, start="\n")

    return










def print_density_information(calculation: Calculation) -> None:

    """
    
    Prints the type of density matrix used in property calculations.

    Args:
        calculation (Calculation): Calculation object

    """

    method = calculation.method

    # Specifies which density matrix is used for the property calculations
    
    if method.name in ["MP2", "SCS-MP2"]: log("\n Using the MP2 unrelaxed density for property calculations.", calculation, 1)
    elif method.name == "OMP2": log("\n Using the orbital-optimised MP2 relaxed density for property calculations.", calculation, 1)
    elif method.name == "LMP2": warning("Using the Hartree-Fock density, not the MP2 density, for property calculations.")
    elif method.method_base == "MP3" or method.method_base == "MP4": warning("Using the unrelaxed MP2 density for property calculations.")
    
    if method.method_base == "CC": log("\n Using the linearised coupled cluster density for property calculations.", calculation, 1)
    if method.name == "CCSD[T]": warning("Using the linearised CCSD density, not the CCSD(T) density, for property calculations.")
    if method.name == "QCISD[T]": warning("Using the linearised QCISD density, not the QCISD(T) density, for property calculations.")

    return










def calculate_molecular_properties(molecule: Molecule, calculation: Calculation, P: ndarray, S: ndarray, SCF_output: Output, P_alpha: ndarray, P_beta: ndarray) -> None:

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

    """

    log("\n Beginning calculation of TUNA properties... ", calculation, 3)

    # Prints information about the analytical density used

    print_density_information(calculation)

    # Prints molecular orbital eigenvalues and coefficients

    print_molecular_orbital_eigenvalues(calculation, molecule, SCF_output)
    
    print_molecular_orbital_coefficients(calculation, molecule, SCF_output)

    # Prints Koopmans' theorem parameters if RHF reference is used

    if calculation.reference == "RHF":
        
        calculate_Koopmans_parameters(SCF_output.epsilons, molecule.n_doubly_occ, calculation)

    # As long as there are two real atoms present, calculates rotational constant and dipole moment information

    if calculation.diatomic:

        mol.calculate_and_print_rotational_constant(molecule.reduced_mass, molecule.bond_length, calculation)

        calculate_and_print_multipole_moments(P, molecule, SCF_output, calculation)

        # Builds spin density matrix

        R = P_alpha - P_beta if molecule.n_alpha + molecule.n_beta != 1 else P

        # Calculate population analysis, format all the data, then print to console

        calculate_and_print_population_analysis(P, S, R, molecule.partition_ranges, molecule.atomic_symbols, molecule.charges, calculation)
        

    return