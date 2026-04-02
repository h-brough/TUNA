import numpy as np
import time, sys
from termcolor import colored
from numpy import ndarray
from dataclasses import dataclass


"""

This is the TUNA module for constants and utility functions, written first for version 0.3.0 and rewritten for version 0.10.1.

Various widely useful functions are stored here, as are the defining constants which are used throughout the program. This module does not import any other 
TUNA module, and is imported by all other TUNA modules.

This module contains:

1. A class to contain all the defining and emergent constants used in TUNA
2. Dataclasses for the SCF output container, molecular integrals, electronic structure method and exchange-correlation functional
3. Various utility functions used throughout the program (clean_coordinates, log, error, etc.)
4. Dictionaries and lists containing all the atom types, calculation types, electronic structure methods (electronic_structure_methods, atomic_properties, etc.)

"""



class Constants:

    """

    Defines all the contants used in TUNA. Fundamental values are taken from the CODATA 2022 recommendations.
    
    Fundamental values are used to define various emergent constants and conversion factors.

    """

    # Fundamental constants to define Hartree land

    planck_constant_in_joules_seconds = 6.62607015e-34
    elementary_charge_in_coulombs = 1.602176634e-19
    electron_mass_in_kilograms = 9.1093837139e-31
    permittivity_in_farad_per_metre = 8.8541878188e-12

    # Non-quantum fundamental constants

    c_in_metres_per_second = 299792458
    k_in_joules_per_kelvin = 1.380649e-23
    avogadro = 6.02214076e23

    # Emergent unit conversions

    atomic_mass_unit_in_kg = 0.001 / avogadro
    reduced_planck_constant_in_joules_seconds = planck_constant_in_joules_seconds / (2 * np.pi)
    bohr_in_metres = 4 * np.pi * permittivity_in_farad_per_metre * reduced_planck_constant_in_joules_seconds ** 2 / (electron_mass_in_kilograms * elementary_charge_in_coulombs ** 2)
    hartree_in_joules = reduced_planck_constant_in_joules_seconds ** 2 / (electron_mass_in_kilograms * bohr_in_metres ** 2)
    atomic_time_in_seconds = reduced_planck_constant_in_joules_seconds / hartree_in_joules
    atomic_time_in_femtoseconds = atomic_time_in_seconds * 10 ** 15
    bohr_radius_in_angstrom = bohr_in_metres * 10 ** 10

    pascal_in_atomic_units = hartree_in_joules / bohr_in_metres ** 3
    per_cm_in_hartree = hartree_in_joules / (c_in_metres_per_second * planck_constant_in_joules_seconds * 10 ** 2)
    per_cm_in_GHz = hartree_in_joules / (planck_constant_in_joules_seconds * per_cm_in_hartree * 10 ** 9)
    atomic_mass_unit_in_electron_mass = atomic_mass_unit_in_kg / electron_mass_in_kilograms
    eV_in_hartree = hartree_in_joules / elementary_charge_in_coulombs

    # Emergent constants

    c = c_in_metres_per_second * atomic_time_in_seconds / bohr_in_metres
    k = k_in_joules_per_kelvin / hartree_in_joules
    h = 2 * np.pi

    # Gradient correct to 9 sf. for "EXTREME" and "TIGHT", 8 sf. for "MEDIUM" and 7 sf. for "LOOSE"

    FIRST_GEOM_DERIVATIVE_PROD = 0.00005   

    # Dipole moment correct to 9 sf. for "EXTREME" and "TIGHT", 7 sf. for "MEDIUM" and 6 sf. for "LOOSE"

    FIRST_ELEC_DERIVATIVE_PROD = 0.00001 

    # Hessian correct to 8 sf. for "EXTREME", 7 sf. for "TIGHT", 6 sf. for "MEDIUM" and 5 sf. for "LOOSE" - which is still correct frequency to 0.01 per cm

    SECOND_GEOM_DERIVATIVE_PROD = 0.01

    # Isotropic polarisability correct to 7 sf. for "EXTREME", 5 sf. for "TIGHT", 4 sf. for "MEDIUM" and 3 sf. for "LOOSE"

    SECOND_ELEC_DERIVATIVE_PROD = 0.001
    
    # Second-order VPT is correct to 8 sf. for "EXTREME" and "TIGHT", 5 sf. for "MEDIUM" and 2 sf. for "LOOSE" - a compromise between third and fourth derivative precision

    THIRD_GEOM_DERIVATIVE_PROD = 0.025

    # Parallel hyperpolarisability correct to 5 sf. for "EXTREME", 2sf for "TIGHT", "MEDIUM" and "LOOSE"

    THIRD_ELEC_DERIVATIVE_PROD = 0.0015

    # Constants for cleaning things on the DFT grid

    density_floor = 1e-23
    sigma_floor = density_floor ** 2

    # Convergence criteria for self-consistent field

    convergence_criteria_SCF = {

        "loose" : {"delta_E": 0.000001, "max_DP": 0.00001, "RMS_DP": 0.000001, "commutator": 0.0001, "name": "loose"},
        "medium" : {"delta_E": 0.0000001, "max_DP": 0.000001, "RMS_DP": 0.0000001, "commutator": 0.00001, "name": "medium"},
        "tight" : {"delta_E": 0.000000001, "max_DP": 0.00000001, "RMS_DP": 0.000000001, "commutator": 0.0000001, "name": "tight"},
        "extreme" : {"delta_E": 0.00000000001, "max_DP": 0.0000000001, "RMS_DP": 0.00000000001, "commutator": 0.000000001, "name": "extreme"}   
        
    }

    # Convergence criteria for geometry optimisation

    convergence_criteria_optimisation = {

        "loose" : {"gradient": 0.001, "step": 0.01, "name": "loose"},
        "medium" : {"gradient": 0.0001, "step": 0.0001, "name": "medium"},
        "tight" : {"gradient": 0.000001, "step": 0.00001, "name": "tight"},
        "extreme" : {"gradient": 0.00000001, "step": 0.0000001, "name": "extreme"}   

    }

    # Tightness criteria for the DFT grid
    
    convergence_criteria_grid = {

        "loose" : {"integral_accuracy": 3, "extent_multiplier": 0.7, "name": "loose"},
        "medium" : {"integral_accuracy": 4, "extent_multiplier": 0.9, "name": "medium"},
        "tight" : {"integral_accuracy": 5, "extent_multiplier": 1, "name": "tight"},
        "extreme" : {"integral_accuracy": 7, "extent_multiplier": 1.2, "name": "extreme"},

    }


constants = Constants()










@dataclass
class Integrals:

    """
    
    Stores the integrals needed for a self-consistent field calculation.

    """

    S: ndarray  # Overlap integrals
    T: ndarray  # Kinetic integrals
    V_NE: ndarray  # Nuclear-electron integrals
    D: ndarray  # Dipole integrals
    Q: ndarray  # Quadrupole integrals

    ERI_AO: ndarray  # Two-electron integrals
    
    F: ndarray | None = None # Total electric field integrals

    @property
    def H_core(self):

        if self.F is not None:

            return self.T + self.V_NE + self.F

        return self.T + self.V_NE
    
    @property
    def one_electron_integrals(self):

        return self.S, self.T, self.V_NE, self.D
    
    @property
    def two_electron_integrals(self):

        return self.ERI_AO
    
    @property
    def n_basis(self):

        return self.S.shape[0]










@dataclass
class Output:

    """
    
    Stores the useful output of a self-consistent field calculation.
    
    """

    # Components of energy

    energy: float

    kinetic_energy: float
    nuclear_electron_energy: float
    coulomb_energy: float
    exchange_energy: float
    correlation_energy: float
    electric_field_energy: float

    # Density and overlap matrices in AO basis

    P: ndarray
    P_alpha: ndarray
    P_beta: ndarray
    S: ndarray
    X: ndarray

    # Molecular orbitals

    molecular_orbitals: ndarray
    molecular_orbitals_alpha: ndarray
    molecular_orbitals_beta: ndarray

    # Orbital energies

    epsilons: ndarray
    epsilons_alpha: ndarray
    epsilons_beta: ndarray

    # Electron densities on grid

    density: ndarray
    alpha_density: ndarray
    beta_density: ndarray

    # Converged matrices in AO basis

    F_alpha: ndarray
    F_beta: ndarray
    T: ndarray
    V_NE: ndarray

    # Molecular integrals

    integrals: Integrals

    @property 
    def epsilons_combined(self):
        
        return np.append(self.epsilons_alpha, self.epsilons_beta)

    @property
    def F(self):

        return self.F_alpha + self.F_beta

    @property
    def exchange_correlation_energy(self):

        return self.exchange_energy + self.correlation_energy










@dataclass
class Method:

    """
    
    Defines an electronic structure method.

    """

    # The keyword for this method, ignoring "U" for unrestricted calculations, e.g. "HF"

    name: str

    # What is the human-readable method name, e.g. "Hartree-Fock theory"

    generic_name: str

    # Can this method be used with an unrestricted reference?

    unrestricted_available: bool = True

    # What kind of electronic structure method is this?
    
    method_base: bool = "HF"

    # Is this a method for excited states?

    excited_state_method: bool = False

    # Is this method unrestricted

    unrestricted: bool = False

    @property
    def long_name(self) -> str:
        
        if self.unrestricted:

            return "unrestricted " + self.generic_name 

        return self.generic_name

    @property
    def perturbative_method(self) -> bool:

        return self.method_base in ["MP2", "MP3", "MP4"] 

    @property
    def coupled_cluster_method(self) -> bool:

        return self.method_base == "CC"

    @property
    def correlated_method(self) -> bool:

        return self.coupled_cluster_method or self.perturbative_method

    @property
    def density_functional_method(self) -> bool:

        return self.method_base == "DFT"










@dataclass
class Functional:

    """

    Defines an exchange-correlation functional.

    """

    x_functional: callable
    c_functional: callable

    # Proportions of density-functional and Hartree-Fock exchange

    DFX: float = 1.0
    HFX: float = 0.0

    # Proportion of density-functional and second-order perturbative correlation

    DFC: float = 1.0
    MPC: float = 0.0

    # Same-spin scaling and opposite-spin scaling for double-hybrid functionals

    same_spin_scaling: float = 1.0
    opposite_spin_scaling: float = 1.0

    # Is the functional LDA, GGA or meta-GGA - which derivatives are needed for the exchange-correlation potential

    functional_class: str = "LDA"

    # Dispersion S6 value for D2 correction

    D2_S6: float = 1.2

    @property
    def functional_type(self) -> str:

        if self.MPC != 0:

            if self.same_spin_scaling != 1 and self.opposite_spin_scaling != 1:
                 
                return "spin-scaled double-hybrid"
            
            return "double-hybrid"

        if self.HFX != 0:

             return "hybrid"
        
        return "pure"










def bohr_to_angstrom(length_in_bohr: float | list | ndarray) -> float | ndarray: 
    
    """

    Converts length in bohr to length in angstroms.

    Args:   
        length_in_bohr (float | list | array): Length in bohr

    Returns:
        length_in_angstrom (float | array) : Length in angstrom

    """
    
    length_in_bohr = np.array(length_in_bohr)

    length_in_angstrom = length_in_bohr * constants.bohr_radius_in_angstrom

    return length_in_angstrom










def angstrom_to_bohr(length_in_angstrom: float | list | ndarray) -> float | ndarray: 
    
    """

    Converts length in angstrom to length in bohr.

    Args:   
        length (float | list | array): Length in angstrom

    Returns:
        length_in_bohr  (float | array) : Length in bohr

    """
        
    length_in_angstrom = np.array(length_in_angstrom)

    length_in_bohr = length_in_angstrom / constants.bohr_radius_in_angstrom

    return length_in_bohr










def one_dimension_to_three(coordinates_1D: ndarray) -> ndarray: 
    
    """

    Converts 1D coordinate array into 3D.

    Args:   
        coordinates (array): Coordinates in one dimension

    Returns:
        coordinates_3D (array) : Coordinates in three dimensions

    """

    coordinates_3D = np.array([[0, 0, coord] for coord in coordinates_1D])
    
    return coordinates_3D










def three_dimensions_to_one(coordinates_3D: ndarray) -> ndarray: 
    
    """

    Converts 3D coordinate array into 1D.

    Args:   
        coordinates_3D (array): Coordinates in three dimensions

    Returns:
        coordinates_1D (array) : Coordinates in one dimension

    """

    coordinates_1D = np.array([atom_coord[2] for atom_coord in coordinates_3D])
    
    return coordinates_1D
    









def calculate_bond_length(coordinates: ndarray) -> float:

    """
    
    Calculates the bond length of a molecule with 1D or 3D coordinates.

    Args:
        coordinates (array): Atomic coordinates in bohr
    
    Returns:
        bond_length (float): Bond length in bohr
    
    """

    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

    return bond_length










def calculate_first_derivative(F_m_1: float, F_p_1: float, dx: float) -> float:

    """

    Calculates the numerical first derivative of a function using the central differences method.

    This has error of O(dx^2).

    Args:
        F_m_1 (float): Value of the function at x - dx
        F_p_1 (float): Value of the function at x + dx
        dx (float): Step size

    Returns:
        dF_dx (float): First derivative of the function at x

    """

    dF_dx = (F_p_1 - F_m_1) / (2 * dx)

    return dF_dx










def calculate_second_derivative(F_m_2: float, F_m_1: float, F: float, F_p_1: float, F_p_2: float, dx: float) -> float:

    """
    
    Calculates the numerical second derivative of a function using the five-point stencil method.

    This has error of O(dx^4).

    Args:
        F_m_2 (float): Value of the function at x - 2dx
        F_m_1 (float): Value of the function at x - dx
        F (float): Value of the function at x
        F_p_1 (float): Value of the function at x + dx
        F_p_2 (float): Value of the function at x + 2dx
        dx (float): Step size

    Returns:
        d2F_dx2 (float): Second derivative of the function at x

    """

    # Equation from Wikipedia page on numerical second derivative methods, fairly noise-resistant formula

    d2F_dx2 = (-F_m_2 + 16 * F_m_1 - 30 * F + 16 * F_p_1 -  F_p_2) / (12 * dx ** 2)

    return d2F_dx2










def calculate_third_derivative(F_m_4: float, F_m_3: float, F_m_2: float, F_m_1: float, F_p_1: float, F_p_2: float, F_p_3: float, F_p_4: float, dx: float) -> float:
    
    """
    
    Calculates the numerical third derivative of a function using the eight-point stencil method.

    This has error of O(dx^6).

    Args:
        F_m_4 (float): Value of the function at x - 4dx
        F_m_3 (float): Value of the function at x - 3dx
        F_m_2 (float): Value of the function at x - 2dx
        F_m_1 (float): Value of the function at x - dx
        F_p_1 (float): Value of the function at x + dx
        F_p_2 (float): Value of the function at x + 2dx
        F_p_3 (float): Value of the function at x + 3dx
        F_p_4 (float): Value of the function at x + 4dx
        dx (float): Step size

    Returns:
        d3F_dx3 (float): Third derivative of the function at x

    """

    d3F_dx3 = (-7 * F_m_4 + 72 * F_m_3 - 338 * F_m_2 + 488 * F_m_1 - 488 * F_p_1 + 338 * F_p_2 - 72 * F_p_3 + 7 * F_p_4) / (240 * dx ** 3)

    return d3F_dx3










def calculate_fourth_derivative(F_m_4: float, F_m_3: float, F_m_2: float, F_m_1: float, F:float, F_p_1: float, F_p_2: float, F_p_3: float, F_p_4: float, dx: float) -> float:
    
    """
    
    Calculates the numerical fourth derivative of a function using the nine-point stencil method.

    This has error of O(dx^6).

    Args:
        F_m_4 (float): Value of the function at x - 4dx
        F_m_3 (float): Value of the function at x - 3dx
        F_m_2 (float): Value of the function at x - 2dx
        F_m_1 (float): Value of the function at x - dx
        F (float): Value of the function at x
        F_p_1 (float): Value of the function at x + dx
        F_p_2 (float): Value of the function at x + 2dx
        F_p_3 (float): Value of the function at x + 3dx
        F_p_4 (float): Value of the function at x + 4dx
        dx (float): Step size

    Returns:
        d4F_dx4 (float): Fourth derivative of the function at x

    """

    d4F_dx4 = (7 * F_m_4 - 96 * F_m_3 +676 * F_m_2 -1952 * F_m_1 +2730 * F - 1952 * F_p_1 + 676 * F_p_2 - 96 * F_p_3 + 7 * F_p_4) / (240 * dx ** 4)

    return d4F_dx4










def convert_boolean_to_string(boolean: bool) -> str:

    """

    Converts a boolean into either "Yes" or "No".

    Args:
        boolean (bool): A true of false object
    
    Returns
        convert_boolean_to_string (str): Either "Yes" or "No"

    """

    return "Yes" if boolean else "No"










def symmetrise(matrix: ndarray) -> ndarray:

    """
    
    Symmetrises a square matrix.

    Args:
        matrix (array): Square matrix
    
    Returns:
        matrix_symmetrised (array): Symmetrised square matrix

    """

    matrix_symmetrised = (1 / 2) * (matrix + matrix.T)

    return matrix_symmetrised










def calculate_centre_of_mass(masses: ndarray, coordinates: ndarray) -> float: 
    
    """

    Calculates the centre of mass of a coordinate and mass array.

    Args:   
        masses (array): Atomic masses
        coordinates (array): Atomic coordinates

    Returns:
        centre_of_mass (float) : The centre of mass in angstroms away from the first atom

    """

    centre_of_mass = np.einsum("i,ij->", masses, coordinates, optimize=True) / np.sum(masses)
    

    return centre_of_mass










def is_molecule_aligned_on_z_axis(molecule: any) -> bool:

    """
    
    Checks if a molecule lies only on the z axis.

    Args:
        molecule (Molecule): Molecule object

    Returns:
        is_molecule_aligned (bool): Is the molecule aligned on the z axis

    """

    is_molecule_aligned = True

    if len(molecule.atoms) == 2:

        # Iterates over both atoms, over the x and y coordinates

        for i in range(len(molecule.atoms)):

            for j in range(2):
                
                # Below 1e-14 is numerical noise and is irrelevant

                if np.abs(molecule.coordinates[i][j]) > 1e-14:
                    
                    is_molecule_aligned = False

    return is_molecule_aligned










def clean_coordinates(coordinates: ndarray) -> ndarray:

    """
    
    Makes sure the atomic coordinates are perfectly aligned along the z-axis.

    Args:
        coordinates (array): Atomic coordinates

    Returns:
        coordinates_cleaned (array): Perfectly aligned coordinates

    """

    # Handles the case for 3D coordinates of a molecule

    if coordinates.shape == (2, 3):

        coordinates_cleaned = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, calculate_bond_length(coordinates)]])
    
    # Handles the case for 1D coordinates of a molecule

    elif coordinates.shape == (1, 2):

        coordinates_cleaned = np.array([[0.0, calculate_bond_length(coordinates)]])
    
    # Handles the case for coordinates of an atom

    else:

        coordinates_cleaned = coordinates


    return coordinates_cleaned










def format_charge(charge: int) -> str:
    
    """
    
    Formats the charge as a string.

    Args:
        charge (int): Molecular charge
    
    Returns:
        formatted_charge (str): Charge stringwith prepended plus sign
    
    """

    formatted_charge = f"+{charge}" if charge > 0 else str(charge)

    return formatted_charge










def error(message: str) -> None: 

    """

    Closes TUNA and prints an error, in light red.

    Args:   
        message (string): Error message

    """
    
    print(colored(f"\nERROR: {message}  :(\n", "light_red"))

    # Exits the program

    sys.exit()

    return










def warning(message: str, space: int = 1) -> None: 
    
    """

    Prints a warning message, in light yellow.

    Args:   
        message (string): Error message
        space (int, optional): Number of indenting spaces from the left hand side

    """
    
    print(colored(f"\n{" " * space}WARNING: {message}", "light_yellow"))

    return










def log(message: str, calculation: any, priority: int = 1, end: str = "\n", silent: bool = False, colour: str = "light_grey") -> None:

    """

    Logs a message to the console.

    Args:   
        message (string): Error message
        calculation (Calculation): Calculation object
        priority (int, optional): Priority of message (1 to always appear, 2 to appear unless T keyword used, and 3 only to appear if P keyword used, 4 if DEBUG)
        end (string, optional): End of message
        silent (bool, optional): Specifies whether to print anything
        colour (str, optional): Colour for logging

    """

    if not silent:

        if priority == 1: 
            
            print(colored(message, colour), end=end)
        
        elif priority == 2 and not calculation.terse: 
            
            print(colored(message, colour, force_color = True), end=end)
        
        elif priority == 3 and calculation.additional_print: 
            
            print(colored(message, colour, force_color = True), end=end)
        
        elif priority == 4 and calculation.debug: 
            
            print(colored(message, colour, force_color = True), end=end)

    return










def log_spacer(calculation: any, priority: int = 1, start: str = "", end: str = "", space: str = " ", silent: bool = False) -> None:

    """
    
    Prints out a normal size wavey spacer.

    Args:
        calculation (Calculation): Calculation object
        priority (int, optional): Priority to print this
        start (str, optional): What to print at the start of the line
        end (str, optional): What to print at the end of the line
        space (str, optional): How many spaces to print at the start
        silent (bool, optional): Cancel logging
    
    """

    log(f"{start}{space}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{end}", calculation, priority=priority, silent=silent)

    return










def log_big_spacer(calculation: any, priority: int = 1, start: str = "", end: str = "", space: str = " ", silent: bool = False) -> None:

    """
    
    Prints out a very big wavey spacer.

    Args:
        calculation (Calculation): Calculation object
        priority (int, optional): Priority to print this
        start (str, optional): What to print at the start of the line
        end (str, optional): What to print at the end of the line
        space (str, optional): How many spaces to print at the start
        silent (bool, optional): Cancel logging
    
    """

    log(f"{start}{space}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{end}", calculation, priority=priority, silent=silent)

    return










def finish_calculation(calculation: any) -> None:

    """

    Finishes the calculation and exits the program.

    Args:   
        calculation (Calculation): Calculation object

    """

    # Calculates total time for the TUNA calculation

    end_time = time.perf_counter()
    total_time = end_time - calculation.start_time
    
    if calculation.additional_print:

        log(f"\n Time taken for molecular integrals:      {calculation.integrals_time - calculation.start_time:8.2f} seconds", calculation, 3)
        log(f" Time taken for SCF iterations:           {calculation.SCF_time - calculation.integrals_time:8.2f} seconds", calculation, 3)

        if calculation.method.correlated_method: 
            
            log(f" Time taken for correlated calculation:   {calculation.correlation_time - calculation.SCF_time:8.2f} seconds", calculation, 3)
        
        if calculation.method.excited_state_method:
        
            log(f" Time taken for excited state calculation:  {calculation.excited_state_time - calculation.SCF_time:6.2f} seconds", calculation, 3)

    if total_time > 120:

        minutes = total_time // 60
        seconds = total_time % 60

        if total_time > 7200:
            
            hours = total_time // 3600
            extra_minutes = (total_time % 3600) // 60

            log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {hours:.0f} hours, {extra_minutes:.0f} minutes and {seconds:.2f} seconds.  :)\n","white"), calculation, 1)

        else:
            
            log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {minutes:.0f} minutes and {seconds:.2f} seconds.  :)\n","white"), calculation, 1)

    else:
        
        log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n", "white"), calculation, 1)
    
    # Exits the program

    sys.exit()

    return










calculation_types = {

    "SPE"       :     "Single point energy",
    "OPT"       :     "Geometry optimisation",
    "FREQ"      :     "Harmonic frequency",
    "OPTFREQ"   :     "Optimisation and harmonic frequency",
    "SCAN"      :     "Coordinate scan",
    "MD"        :     "Ab initio molecular dynamics",
    "FORCE"     :     "Force",
    "ANHARM"    :     "Anharmonic frequency",
    "IP"        :     "Ionisation potential",
    "EA"        :     "Electron affinity",
    "BDE"       :     "Bond dissociation energy"
    
    }










electronic_structure_methods = [

    Method("H", "Hartree theory"),
    Method("HF", "Hartree-Fock theory"),
    Method("RHF", "Hartree-Fock theory"),

    Method("MP2", "MP2 theory", method_base = "MP2"),
    Method("OMP2", "orbital-optimised MP2 theory", method_base = "MP2"), 
    Method("IMP2", "iterative MP2 theory", unrestricted_available = False, method_base = "MP2"),
    Method("LMP2", "Laplace transform MP2 theory", unrestricted_available = False, method_base = "MP2"),
    Method("SCS-MP2", "spin-component-scaled MP2 theory", method_base = "MP2"),
    Method("DLPNO-MP2", "domain-based local pair natural orbital MP2 theory", method_base = "MP2"),
    
    Method("MP3", "MP3 theory", method_base = "MP3"),
    Method("SCS-MP3", "spin-component-scaled MP3 theory", method_base = "MP3"),
    
    Method("MP4", "MP4 theory", unrestricted_available = False, method_base = "MP4"),
    Method("MP4[SDTQ]", "MP4 theory", unrestricted_available = False, method_base = "MP4"),
    Method("MP4[SDQ]", "MP4 theory with singles, doubles and quadruples", unrestricted_available = False, method_base = "MP4"),
    Method("MP4[DQ]", "MP4 theory with doubles and quadruples", unrestricted_available = False, method_base = "MP4"),
    
    Method("CIS", "configuration interaction singles", excited_state_method = True),
    Method("CIS[D]", "configuration interaction singles with perturbative doubles", excited_state_method = True),

    Method("CCD", "coupled cluster doubles", method_base = "CC"),
    Method("CEPA", "coupled electron pair approximation", method_base = "CC"),
    Method("CEPA0", "coupled electron pair approximation", method_base = "CC"),
    Method("CEPA[0]", "coupled electron pair approximation", method_base = "CC"),
    Method("LCCD", "linearised coupled cluster doubles", method_base = "CC"),
    Method("LCCSD", "linearised coupled cluster singles and doubles", method_base = "CC"),
    Method("QCISD", "quadratic configuration interaction singles and doubles", method_base = "CC"),
    Method("QCISD[T]", "quadratic configuration interaction singles, doubles and perturbative triples", method_base = "CC"),
    Method("CC2", "approximate coupled cluster singles and doubles", unrestricted_available = False, method_base = "CC"),
    Method("CC3", "approximate coupled cluster singles, doubles and triples", unrestricted_available = False, method_base = "CC"),
    
    Method("CCSD", "coupled cluster singles and doubles", method_base = "CC"),
    Method("CCSD[T]", "coupled cluster singles, doubles and perturbative triples", method_base = "CC"),
    Method("CCSDT","coupled cluster singles, doubles and triples", method_base = "CC"),
    Method("CCSDT[Q]", "coupled cluster singles, doubles, triples and perturbative quadruples", unrestricted_available = False, method_base = "CC"),
    Method("CCSDTQ", "coupled cluster singles, doubles, triples and quadruples", unrestricted_available = False, method_base = "CC"),
    
    Method("HFS", "Hartree-Fock theory with Slater exchange", method_base = "DFT"),
    Method("LDA", "density functional theory via local density approximation", method_base = "DFT"),
    Method("LSDA", "density functional theory via local spin density approximation", method_base = "DFT"),
    Method("SVWN", "density functional theory with Slater exchange and VWN correlation", method_base = "DFT"),
    Method("SVWN3", "density functional theory with Slater exchange and VWN-III correlation", method_base = "DFT"),
    Method("SVWN5", "density functional theory with Slater exchange and VWN-V correlation", method_base = "DFT"),
    Method("SPW", "density functional theory with Slater exchange and Perdew-Wang correlation", method_base = "DFT"),
    
    Method("HFB", "Hartree-Fock theory with Becke exchange", method_base = "DFT"),
    Method("BVWN", "density functional theory with Becke exchange and VWN correlation", method_base = "DFT"),
    Method("BVWN3","density functional theory with Becke exchange and VWN-III correlation", method_base = "DFT"),
    Method("BVWN5", "density functional theory with Becke exchange and VWN-V correlation", method_base = "DFT"),
    Method("PBE", "density functional theory with PBE exchange and correlation", method_base = "DFT"),
    Method("BLYP", "density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("SLYP", "density functional theory with Slater exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("PWP", "density functional theory with Perdew-Wang exchange and Perdew 1986 correlation", method_base = "DFT"),
    Method("MPWPW", "density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation", method_base = "DFT"),
    Method("MPWLYP", "density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("BP86", "density functional theory with Becke exchange and Perdew 1986 correlation", method_base = "DFT"),
    
    Method("TPSS", "density functional theory with TPSS exchange and correlation", method_base = "DFT"),
    
    Method("PBE0", "hybrid density functional theory with PBE exchange and correlation", method_base = "DFT"),
    Method( "B1P86", "hybrid density functional theory with Becke exchange and Perdew 1986 correlation", method_base = "DFT"),
    Method("BHLYP", "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B1LYP", "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B3LYP", "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B3LYP/G", "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("MPW1LYP", "hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("PW1PW", "hybrid density functional theory with Perdew-Wang exchange and Perdew-Wang correlation", method_base = "DFT"),
    Method("MPW1PW", "hybrid density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation", method_base = "DFT"),
    Method("B3PW91", "hybrid density functional theory with Becke exchange and Perdew-Wang correlation", method_base = "DFT"),
    Method("B3P86", "hybrid density functional theory with Becke exchange and Perdew 1986 correlation", method_base = "DFT"),
    Method("TPSSH", "hybrid density functional theory with TPSS exchange and correlation", method_base = "DFT"),
    Method("TPSS0", "hybrid density functional theory with TPSS exchange and correlation", method_base = "DFT"),

    Method("PBE0-DH", "double-hybrid density functional theory with PBE exchange and correlation", method_base = "DFT"),
    Method("PBE-QIDH", "double-hybrid density functional theory with PBE exchange and correlation", method_base = "DFT"),
    Method("PBE0-2", "double-hybrid density functional theory with PBE exchange and correlation", method_base = "DFT"),
    Method("B2PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("DSD-BLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B2-PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B2K-PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B2T-PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B2G-PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("B2NC-PLYP", "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation", method_base = "DFT"),
    Method("MPW2PLYP", "double-hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation", method_base = "DFT"),

    ]










DFT_methods = {

    # At some point, want to handle the unrestricted cases together here, since exchange is always the same

    "HFS"         :     Functional("S", None, DFX=1, HFX=0, DFC=0, MPC=0, functional_class="LDA"),
    "UHFS"        :     Functional("S", None, DFX=1, HFX=0, DFC=0, MPC=0, functional_class="LDA"),
    "SVWN"        :     Functional("S", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "USVWN"       :     Functional("S", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "LSDA"        :     Functional("S", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "ULSDA"       :     Functional("S", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "LDA"         :     Functional("S", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "ULDA"        :     Functional("S", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "SVWN3"       :     Functional("S", "VWN3", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "USVWN3"      :     Functional("S", "UVWN3", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "SVWN5"       :     Functional("S", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "USVWN5"      :     Functional("S", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "SPW"         :     Functional("S", "PW", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "USPW"        :     Functional("S", "UPW", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="LDA"),
    "PBE"         :     Functional("PBE", "PBE", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=0.75),
    "UPBE"        :     Functional("PBE", "UPBE", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=0.75),
    "PBE0"        :     Functional("PBE", "PBE", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UPBE0"       :     Functional("PBE", "UPBE", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "PBE0-DH"     :     Functional("PBE", "PBE", DFX=0.50, HFX=0.50, DFC=0.875, MPC=0.125, functional_class="GGA"),
    "UPBE0-DH"    :     Functional("PBE", "UPBE", DFX=0.50, HFX=0.50, DFC=0.875, MPC=0.125, functional_class="GGA"),
    "PBE-QIDH"    :     Functional("PBE", "PBE", DFX=0.31, HFX=0.69, DFC=0.67, MPC=0.33, functional_class="GGA"),
    "UPBE-QIDH"   :     Functional("PBE", "UPBE", DFX=0.31, HFX=0.69, DFC=0.67, MPC=0.33, functional_class="GGA"),
    "PBE0-2"      :     Functional("PBE", "PBE", DFX=1-1/np.cbrt(2), HFX=1/np.cbrt(2), DFC=0.50, MPC=0.50, functional_class="GGA"),
    "UPBE0-2"     :     Functional("PBE", "UPBE", DFX=1-1/np.cbrt(2), HFX=1/np.cbrt(2), DFC=0.50, MPC=0.50, functional_class="GGA"),
    "HFB"         :     Functional("B", None, DFX=1, HFX=0, DFC=0, MPC=0, functional_class="GGA"),
    "UHFB"        :     Functional("B", None, DFX=1, HFX=0, DFC=0, MPC=0, functional_class="GGA"),
    "BVWN"        :     Functional("B", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UBVWN"       :     Functional("B", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "BVWN3"       :     Functional("B", "VWN3", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UBVWN3"      :     Functional("B", "UVWN3", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "BVWN5"       :     Functional("B", "VWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UBVWN5"      :     Functional("B", "UVWN5", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "BLYP"        :     Functional("B", "LYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.2),
    "UBLYP"       :     Functional("B", "ULYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.2),
    "BHLYP"       :     Functional("B", "LYP", DFX=0.50, HFX=0.50, DFC=1, MPC=0, functional_class="GGA"),
    "UBHLYP"      :     Functional("B", "ULYP", DFX=0.50, HFX=0.50, DFC=1, MPC=0, functional_class="GGA"),
    "B1LYP"       :     Functional("B", "LYP", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UB1LYP"      :     Functional("B", "ULYP", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "PWP"         :     Functional("PW", "P86", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UPWP"        :     Functional("PW", "UP86", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "SLYP"        :     Functional("S", "LYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "USLYP"       :     Functional("S", "ULYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "B3LYP"       :     Functional("B3", "3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "UB3LYP"      :     Functional("B3", "U3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "B3LYP/G"     :     Functional("B3", "3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "UB3LYP/G"    :     Functional("B3", "U3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "B2PLYP"      :     Functional("B", "LYP", DFX=0.47, HFX=0.53, DFC=0.73, MPC=0.27, functional_class="GGA", D2_S6=0.55),
    "UB2PLYP"     :     Functional("B", "ULYP", DFX=0.47, HFX=0.53, DFC=0.73, MPC=0.27, functional_class="GGA", D2_S6=0.55),
    "B2-PLYP"     :     Functional("B", "LYP", DFX=0.47, HFX=0.53, DFC=0.73, MPC=0.27, functional_class="GGA", D2_S6=0.55),
    "UB2-PLYP"    :     Functional("B", "ULYP", DFX=0.47, HFX=0.53, DFC=0.73, MPC=0.27, functional_class="GGA", D2_S6=0.55),
    "B2K-PLYP"    :     Functional("B", "LYP", DFX=0.28, HFX=0.72, DFC=0.58, MPC=0.42, functional_class="GGA"),
    "UB2K-PLYP"   :     Functional("B", "ULYP", DFX=0.28, HFX=0.72, DFC=0.58, MPC=0.42, functional_class="GGA"),
    "B2T-PLYP"    :     Functional("B", "LYP", DFX=0.40, HFX=0.60, DFC=0.69, MPC=0.31, functional_class="GGA"),
    "UB2T-PLYP"   :     Functional("B", "ULYP", DFX=0.40, HFX=0.60, DFC=0.69, MPC=0.31, functional_class="GGA"),
    "B2G-PLYP"    :     Functional("B", "LYP", DFX=0.35, HFX=0.65, DFC=0.64, MPC=0.36, functional_class="GGA"),
    "UB2G-PLYP"   :     Functional("B", "ULYP", DFX=0.35, HFX=0.65, DFC=0.64, MPC=0.36, functional_class="GGA"),
    "B2NC-PLYP"   :     Functional("B", "LYP", DFX=0.19, HFX=0.81, DFC=0.45, MPC=0.55, functional_class="GGA"),
    "UB2NC-PLYP"  :     Functional("B", "ULYP", DFX=0.19, HFX=0.81, DFC=0.45, MPC=0.55, functional_class="GGA"),
    "DSD-BLYP"    :     Functional("B", "LYP", DFX=0.25, HFX=0.75, DFC=0.53, MPC=1, same_spin_scaling=0.60, opposite_spin_scaling=0.46, functional_class="GGA"),
    "UDSD-BLYP"   :     Functional("B", "ULYP", DFX=0.25, HFX=0.75, DFC=0.53, MPC=1, same_spin_scaling=0.60, opposite_spin_scaling=0.46, functional_class="GGA"),
    "BP86"        :     Functional("B", "P86", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "UBP86"       :     Functional("B", "UP86", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA", D2_S6=1.05),
    "B1P86"       :     Functional("B", "P86", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UB1P86"      :     Functional("B", "UP86", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "TPSS"        :     Functional("TPSS", "TPSS", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="meta-GGA", D2_S6=1.0),
    "UTPSS"       :     Functional("TPSS", "UTPSS", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="meta-GGA", D2_S6=1.0),
    "TPSSH"       :     Functional("TPSS", "TPSS", DFX=0.90, HFX=0.10, DFC=1, MPC=0, functional_class="meta-GGA"),
    "UTPSSH"      :     Functional("TPSS", "UTPSS", DFX=0.90, HFX=0.10, DFC=1, MPC=0, functional_class="meta-GGA"),
    "TPSS0"       :     Functional("TPSS", "TPSS", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="meta-GGA"),
    "UTPSS0"      :     Functional("TPSS", "UTPSS", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="meta-GGA"),
    "MPWLYP"      :     Functional("MPW", "LYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UMPWLYP"     :     Functional("MPW", "ULYP", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "MPW1LYP"     :     Functional("MPW", "LYP", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UMPW1LYP"    :     Functional("MPW", "ULYP", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "MPW2PLYP"    :     Functional("MPW", "LYP", DFX=0.45, HFX=0.55, DFC=0.75, MPC=0.25, functional_class="GGA", D2_S6=0.4),
    "UMPW2PLYP"   :     Functional("MPW", "ULYP", DFX=0.45, HFX=0.55, DFC=0.75, MPC=0.25, functional_class="GGA", D2_S6=0.4),
    "MPWPW"       :     Functional("MPW", "PW91", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "UMPWPW"      :     Functional("MPW", "UPW91", DFX=1, HFX=0, DFC=1, MPC=0, functional_class="GGA"),
    "PW1PW"       :     Functional("PW", "PW91", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UPW1PW"      :     Functional("PW", "UPW91", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "MPW1PW"      :     Functional("MPW", "PW91", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "UMPW1PW"     :     Functional("MPW", "UPW91", DFX=0.75, HFX=0.25, DFC=1, MPC=0, functional_class="GGA"),
    "B3PW91"      :     Functional("B3", "3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA"),
    "UB3PW91"     :     Functional("B3", "U3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA"),
    "B3P86"       :     Functional("B3", "3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA"),
    "UB3P86"      :     Functional("B3", "U3P", DFX=0.80, HFX=0.20, DFC=1, MPC=0, functional_class="GGA"),

}










basis_types = {

    "CUSTOM" : "custom",
    "STO-2G" : "STO-2G",
    "STO-3G" : "STO-3G",
    "STO-4G" : "STO-4G",
    "STO-5G" : "STO-5G",
    "STO-6G" : "STO-6G",
    "3-21G" : "3-21G",
    "4-31G" : "4-31G",
    "6-31G" : "6-31G",
    "6-31+G" : "6-31+G",
    "6-31++G" : "6-31++G",
    "6-311G" : "6-311G",
    "6-311+G" : "6-311+G",
    "6-311++G" : "6-311++G",
    "6-31G*" : "6-31G*",
    "6-31G**" : "6-31G**",
    "6-311G*" : "6-311G*",
    "6-311G**" : "6-311G**",
    "6-31+G*" : "6-31+G*",
    "6-311+G*" : "6-311+G*",
    "6-31+G**" : "6-31+G**",
    "6-311+G**" : "6-311+G**",
    "6-31++G*" : "6-31++G*",
    "6-311++G*" : "6-311++G*",
    "6-31++G**" : "6-31++G**",
    "6-311++G**" : "6-311++G**",
    "CC-PVDZ" : "cc-pVDZ",
    "CC-PVTZ" : "cc-pVTZ",
    "CC-PVQZ" : "cc-pVQZ",
    "CC-PV5Z" : "cc-pV5Z",
    "CC-PV6Z" : "cc-pV6Z",
    "DEF2-SVP" : "def2-SVP",
    "DEF2-SVPD" : "def2-SVPD",
    "DEF2-TZVP" : "def2-TZVP",
    "DEF2-TZVPD" : "def2-TZVPD",
    "DEF2-TZVPP" : "def2-TZVPP",
    "DEF2-TZVPPD" : "def2-TZVPPD",
    "DEF2-QZVP" : "def2-QZVP",
    "DEF2-QZVPD" : "def2-QZVPD",
    "DEF2-QZVPP" : "def2-QZVPP",
    "DEF2-QZVPPD" : "def2-QZVPPD",
    "6-31G[D]" : "6-31G[d,p]",
    "6-31+G[D]" : "6-31+G[d,p]",
    "6-31++G[D]" : "6-31++G[d,p]",
    "6-311G[D]" : "6-311G[d,p]",
    "6-311+G[D]" : "6-311+G[d,p]",
    "6-311++G[D]" : "6-311++G[d,p]",
    "6-31G[D,P]" : "6-31G[d,p]",
    "6-31+G[D,P]" : "6-31+G[d,p]",
    "6-31++G[D,P]" : "6-31++G[d,p]",
    "6-311G[D,P]" : "6-311G[d,p]",
    "6-311+G[D,P]" : "6-311+G[d,p]",
    "6-311++G[D,P]" : "6-311++G[d,p]",
    "6-31G[2DF,P]" : "6-31G[2df,p]",
    "6-31G[3DF,3PD]" : "6-31G[3df,3pd]",
    "6-311G[D,P]" : "6-311G[d,p]",
    "6-311G[2DF,2PD]" : "6-311G[2df,2pd]",
    "6-311+G[2D,P]" : "6-311+G[2d,p]",
    "6-311++G[2D,2P]" : "6-311++G[2d,2p]",
    "6-311++G[3DF,3PD]" : "6-311++G[3df,3pd]",
    "PC-0" : "pc-0",
    "PC-1" : "pc-1",
    "PC-2" : "pc-2",
    "PC-3" : "pc-3",
    "PC-4" : "pc-4",
    "AUG-PC-0" : "aug-pc-0",
    "AUG-PC-1" : "aug-pc-1",
    "AUG-PC-2" : "aug-pc-2",
    "AUG-PC-3" : "aug-pc-3",
    "AUG-PC-4" : "aug-pc-4",
    "PCSEG-0" : "pcseg-0",
    "PCSEG-1" : "pcseg-1",
    "PCSEG-2" : "pcseg-2",
    "PCSEG-3" : "pcseg-3",
    "PCSEG-4" : "pcseg-4",
    "AUG-PCSEG-0" : "aug-pcseg-0",
    "AUG-PCSEG-1" : "aug-pcseg-1",
    "AUG-PCSEG-2" : "aug-pcseg-2",
    "AUG-PCSEG-3" : "aug-pcseg-3",
    "AUG-PCSEG-4" : "aug-pcseg-4",
    "AUG-CC-PVDZ" : "aug-cc-pVDZ",
    "AUG-CC-PVTZ" : "aug-cc-pVTZ",
    "AUG-CC-PVQZ" : "aug-cc-pVQZ",
    "AUG-CC-PV5Z" : "aug-cc-pV5Z",
    "AUG-CC-PV6Z" : "aug-cc-pV6Z",
    "D-AUG-CC-PVDZ" : "d-aug-cc-pVDZ",
    "D-AUG-CC-PVTZ" : "d-aug-cc-pVTZ",
    "D-AUG-CC-PVQZ" : "d-aug-cc-pVQZ",
    "D-AUG-CC-PV5Z" : "d-aug-cc-pV5Z",
    "D-AUG-CC-PV6Z" : "d-aug-cc-pV6Z",
    "T-AUG-CC-PVDZ" : "t-aug-cc-pVDZ",
    "T-AUG-CC-PVTZ" : "t-aug-cc-pVTZ",
    "T-AUG-CC-PVQZ" : "t-aug-cc-pVQZ",
    "T-AUG-CC-PV5Z" : "t-aug-cc-pV5Z",
    "T-AUG-CC-PV6Z" : "t-aug-cc-pV6Z",
    "CC-PCVDZ" : "cc-pCVDZ",
    "CC-PCVTZ" : "cc-pCVTZ",
    "CC-PCVQZ" : "cc-pCVQZ",
    "CC-PCV5Z" : "cc-pCV5Z",
    "AUG-CC-PCVDZ" : "aug-cc-pCVDZ",
    "AUG-CC-PCVTZ" : "aug-cc-pCVTZ",
    "AUG-CC-PCVQZ" : "aug-cc-pCVQZ",
    "AUG-CC-PCV5Z" : "aug-cc-pCV5Z",
    "CC-PWCVDZ" : "cc-pwCVDZ",
    "CC-PWCVTZ" : "cc-pwCVTZ",
    "CC-PWCVQZ" : "cc-pwCVQZ",
    "CC-PWCV5Z" : "cc-pwCV5Z",
    "AUG-CC-PWCVDZ" : "aug-cc-pwCVDZ",
    "AUG-CC-PWCVTZ" : "aug-cc-pwCVTZ",
    "AUG-CC-PWCVQZ" : "aug-cc-pwCVQZ",
    "AUG-CC-PWCV5Z" : "aug-cc-pwCV5Z",
    "ANO-PVDZ" : "ano-pVDZ",
    "ANO-PVTZ" : "ano-pVTZ",
    "ANO-PVQZ" : "ano-pVQZ",
    "ANO-PV5Z" : "ano-pV5Z",
    "AUG-ANO-PVDZ" : "aug-ano-pVDZ",
    "AUG-ANO-PVTZ" : "aug-ano-pVTZ",
    "AUG-ANO-PVQZ" : "aug-ano-pVQZ",
    "AUG-ANO-PV5Z" : "aug-ano-pV5Z",

}










atomic_properties = {
            
    "X" : {

        "charge" : 0,
        "mass" : 0,
        "C6" : 0,
        "vdw_radius" : 0,
        "real_vdw_radius" : 0,
        "core_orbitals": 0,
        "name" : "ghost",
        "density" : None

    },

    "H" : {

        "charge" : 1,
        "mass" : 1.007825,
        "C6" : 2.4283,
        "vdw_radius" : 1.8916,
        "real_vdw_radius" : 120,
        "core_orbitals": 0,
        "name" : "hydrogen",
        "density" : np.array([[1]])

    },

    "HE" : {

        "charge" : 2,
        "mass" : 4.002603,
        "C6" : 1.3876,
        "vdw_radius" : 1.9124,
        "real_vdw_radius" : 140,
        "core_orbitals": 0,
        "name" : "helium",
        "density" : np.array([[2]])

    },

    "LI" : {

        "charge" : 3,
        "mass" : 7.016004,
        "C6" : 27.92545,
        "vdw_radius" : 1.55902,
        "real_vdw_radius" : 182,
        "core_orbitals": 0,
        "name" : "lithium",
        "density" : np.array([[2.04424896, -0.22217986, 0, 0, 0], [-0.22217986, 1.06290242, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    },

    "BE" : {

        "charge" : 4,
        "mass" : 9.012182,
        "C6" : 27.92545,
        "vdw_radius" : 2.66073,
        "real_vdw_radius" : 153,
        "core_orbitals": 0,
        "name" : "beryllium",
        "density" : np.array([[2.14442543, -0.55651555, 0, 0, 0], [-0.55651555, 2.14442543, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    },

    "B" : {

        "charge" : 5,
        "mass" : 11.009305,
        "C6" : 54.28985,
        "vdw_radius" : 2.80624,
        "real_vdw_radius" : 192,
        "core_orbitals": 1,
        "name" : "boron",
        "density" : np.array([[2.15642129, -0.58078413, 0, 0, 0], [-0.58078413, 2.15642129, 0, 0, 0], [0, 0, 1/3, 0, 0], [0, 0, 0, 1/3, 0], [0, 0, 0, 0, 1/3]])

    },

    "C" : {

        "charge" : 6,
        "mass" : 12.000000,
        "C6" : 30.35375,
        "vdw_radius" : 2.74388,
        "real_vdw_radius" : 170,
        "core_orbitals": 1,
        "name" : "carbon",
        "density" : np.array([[2.13147782, -0.52937894, 0, 0, 0], [-0.5293789, 2.13147782, 0, 0, 0], [0, 0, 2/3, 0, 0], [0, 0, 0, 2/3, 0], [0, 0, 0, 0, 2/3]])

    },

    "N" : {

        "charge" : 7,
        "mass" : 14.003074,
        "C6" : 21.33435,
        "vdw_radius" : 2.63995,
        "real_vdw_radius" : 155,
        "core_orbitals": 1,
        "name" : "nitrogen",
        "density" : np.array([[2.11694591, -0.49756223, 0, 0, 0], [-0.49756223, 2.11694591, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

    },

    "O" : {

        "charge" : 8,
        "mass" : 15.994915,
        "C6" : 12.1415,
        "vdw_radius" : 2.53601,
        "real_vdw_radius" : 152,
        "core_orbitals": 1,
        "name" : "oxygen",
        "density" : np.array([[2.11870859, -0.50150663, 0, 0, 0], [-0.50150663, 2.11870859, 0, 0, 0], [0, 0, 4/3, 0, 0], [0, 0, 0, 4/3, 0], [0, 0, 0, 0, 4/3]])

    },

    "F" : {

        "charge" : 9,
        "mass" : 18.998403,
        "C6" : 13.00875,
        "vdw_radius" : 2.43208,
        "real_vdw_radius" : 147,
        "core_orbitals": 1,
        "name" : "fluorine",
        "density" : np.array([[2.12007958, -0.50455749, 0, 0, 0], [-0.50455749, 2.12007958, 0, 0, 0], [0, 0, 5/3, 0, 0], [0, 0, 0, 5/3, 0], [0, 0, 0, 0, 5/3]])

    },

    "NE" : {

        "charge" : 10,
        "mass" : 19.992440,
        "C6" : 10.92735,
        "vdw_radius" : 2.34893,
        "real_vdw_radius" : 154,
        "core_orbitals": 1,
        "name" : "neon",
        "density" : np.array([[2.12526962, -0.51597648, 0, 0, 0], [-0.51597648, 2.12526962, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])

    },

    "NA" : {

        "charge" : 11,
        "mass" : 22.989770,
        "C6" : 99.03995,
        "vdw_radius" : 2.16185,
        "real_vdw_radius" : 227,
        "core_orbitals": 1,
        "name" : "sodium",
        "density" : np.array([[2.17662152, -6.46761790e-01, 0, 0, 0, 1.12325209e-01, 0, 0, 0], [-6.46761790e-01, 2.37933229, 0, 0, 0, -5.32213795e-01, 0, 0, 0], [0, 0, 1.59827524, 0, 0, 0, 3.51284476e-01, 0, 0], [0, 0, 0, 1.59827524, 0, 0, 0, 3.51284476e-01, 0], [0, 0, 0, 0, 1.59827524, 0, 0, 0, 3.51284476e-01], [1.12325209e-01, -5.32213795e-01, 0, 0, 0, 1.27775127, 0, 0, 0], [0, 0, 3.51284476e-01, 0, 0, 0, 7.73009653e-02, 0, 0], [0, 0, 0, 3.51284476e-01, 0, 0, 0, 7.73009653e-02, 0], [0, 0, 0, 0, 3.51284476e-01, 0, 0, 0, 7.73009653e-02]])
   
    },

    "MG" : {

        "charge" : 12,
        "mass" : 23.985042,
        "C6" : 99.03995,
        "vdw_radius" : 2.57759,
        "real_vdw_radius" : 173,
        "core_orbitals": 1,
        "name" : "magnesium",
        "density" : np.array([[2.20235014, -0.71327382, 0, 0, 0, 0.21565979, 0, 0, 0], [-0.71327382, 2.52187183, 0, 0, 0, -0.88613634, 0, 0, 0], [0, 0, 1.7603467, 0, 0, 0, 0.26850121, 0, 0], [0, 0, 0, 1.7603467, 0, 0, 0, 0.26850121, 0], [0, 0, 0, 0, 1.7603467, 0, 0, 0, 0.26850121], [0.21565979, -0.88613634, 0, 0, 0, 2.31198223, 0, 0, 0], [0, 0, 0.26850121, 0, 0, 0, 0.04095381, 0, 0], [0, 0, 0, 0.26850121, 0, 0, 0, 0.04095381, 0], [0, 0, 0, 0, 0.26850121, 0, 0, 0, 0.04095381]])
    
    },

    "AL" : {

        "charge" : 13,
        "mass" : 26.981538,
        "C6" : 187.15255,
        "vdw_radius" : 3.09726,
        "real_vdw_radius" : 184,
        "core_orbitals": 5,
        "name" : "aluminium",
        "density": np.array([[2.21518153, -0.7210589, 0, 0, 0, 0.17765448, 0, 0, 0], [-0.7210589, 2.42119273, 0, 0, 0, -0.69637357, 0, 0, 0], [0, 0, 2.00715026, -0.00013103, -0.0624102, 0, -0.17674644, 0.00032326, 0.15397241], [0, 0, -0.00013103, 1.85304795, 5.307e-05, 0, 0.00032326, 0.20343991, -0.00013092], [0, 0, -0.0624102, 5.307e-05, 1.87832344, 0, 0.15397241, -0.00013092, 0.14108264], [0.17765448, -0.69637357, 0, 0, 0, 2.20073007, 0, 0, 0], [0, 0, -0.17674644, 0.00032326, 0.15397241, 0, 0.96020735, -0.00079744, -0.37983008], [0, 0, 0.00032326, 0.20343991, -0.00013092, 0, -0.00079744, 0.02233666, 0.00032296], [0, 0, 0.15397241, -0.00013092, 0.14108264, 0, -0.37983008, 0.00032296, 0.17616399]])
    
    },

    "SI" : {

        "charge" : 14,
        "mass" : 27.976927,
        "C6" : 160.09435,
        "vdw_radius" : 3.24277,
        "real_vdw_radius" : 210,
        "core_orbitals": 5,
        "name" : "silicon",
        "density": np.array([[2.23075361, -0.7411749, 0, 0, 0, 0.15812382, 0, 0, 0], [-0.7411749, 2.38444931, 0, 0, 0, -0.59611799, 0, 0, 0], [0, 0, 1.90557606, 0, 0, 0, 0.1529621, 0, 0], [0, 0, 0, 2.15540668, 0, 0, 0, -0.57876127, 0], [0, 0, 0, 0, 1.90557606, 0, 0, 0, 0.1529621], [0.15812382, -0.59611799, 0, 0, 0, 2.1494, 0, 0, 0], [0, 0, 0.1529621, 0, 0, 0, 0.01227839, 0, 0], [0, 0, 0, -0.57876127, 0, 0, 0, 2.15540668, 0], [0, 0, 0, 0, 0.1529621, 0, 0, 0, 0.01227839]])
    
    },

    "P" : {

        "charge" : 15,
        "mass" : 30.973762,
        "C6" : 135.9848,
        "vdw_radius" : 3.22198,
        "real_vdw_radius" : 180,
        "core_orbitals": 5,        
        "name" : "phosphorus",
        "density": np.array([[2.2491554, -0.77208347, 0, 0, 0, 0.16024315, 0, 0, 0], [-0.77208347, 2.3962711, 0, 0, 0, -0.58386231, 0, 0, 0], [0, 0, 1.9839067, -0.06015484, -0.05490761, 0, -0.05656304, 0.1850791, 0.16994956], [0, 0, -0.06015484, 2.0002586, 0.00511423, 0, 0.1850791, -0.10642059, -0.01632541], [0, 0, -0.05490761, 0.00511423, 2.1244592, 0, 0.16994956, -0.01632541, -0.49095309], [0.16024315, -0.58386231, 0, 0, 0, 2.1426492, 0, 0, 0], [0, 0, -0.05656304, 0.1850791, 0.16994956, 0, 0.58002822, -0.56941502, -0.52603041], [0, 0, 0.1850791, -0.10642059, -0.01632541, 0, -0.56941502, 0.73200918, 0.05206724], [0, 0, 0.16994956, -0.01632541, -0.49095309, 0, -0.52603041, 0.05206724, 1.9225545]])
    
    },

    "S" : {

        "charge" : 16,
        "mass" : 31.972071,
        "C6" : 96.61165,
        "vdw_radius" : 3.18041,
        "real_vdw_radius" : 180,
        "core_orbitals": 5,
        "name" : "sulfur",
        "density": np.array([[2.26601244, -0.799712, 0, 0, 0, 0.16211699, 0, 0, 0], [-0.799712, 2.40784748, 0, 0, 0, -0.57394585, 0, 0, 0], [0, 0, 2.13878051, -0.02469174, -0.00351481, 0, -0.54127214, 0.07973172, 0.01134962], [0, 0, -0.02469174, 1.94413911, -0.02814566, 0, 0.07973172, 0.08724145, 0.09088472], [0, 0, -0.00351481, -0.02814566, 2.13785753, 0, 0.01134962, 0.09088472, -0.53829176], [0.16211699, -0.57394585, 0, 0, 0, 2.13721465, 0, 0, 0], [0, 0, -0.54127214, 0.07973172, 0.01134962, 0, 2.10971252, -0.25746048, -0.03664888], [0, 0, 0.07973172, 0.08724145, 0.09088472, 0, -0.25746048, 0.08018887, -0.29347446], [0, 0, 0.01134962, 0.09088472, -0.53829176, 0, -0.03664888, -0.29347446, 2.10008862]])
   
    },

    "CL" : {

        "charge" : 17,
        "mass" : 34.968853,
        "C6" : 87.93915,
        "vdw_radius" : 3.09726,
        "real_vdw_radius" : 175,
        "core_orbitals": 5,
        "name" : "chlorine",
        "density" : np.array([[2.27846455, -0.81566933, 0, 0, 0, 0.14797638, 0, 0, 0], [-0.81566933, 2.39225803, 0, 0, 0, -0.51184572, 0, 0, 0], [0, 0, 2.04723353, -0.02649908, 0.0046819, 0, -0.24466305, 0.09752204, -0.01723034], [0, 0, -0.02649908, 2.10392172, 0.0018477, 0, 0.09752204, -0.45328717, -0.00679993], [0, 0, 0.0046819, 0.0018477, 2.11405309, 0, -0.01723034, -0.00679993, -0.49057268], [0.14797638, -0.51184572, 0, 0, 0, 2.10986634, 0, 0, 0], [0, 0, -0.24466305, 0.09752204, -0.01723034, 0, 1.20496033, -0.35890104, 0.06341118], [0, 0, 0.09752204, -0.45328717, -0.00679993, 0, -0.35890104, 1.97273974, 0.02502514], [0, 0, -0.01723034, -0.00679993, -0.49057268, 0, 0.06341118, 0.02502514, 2.10995807]])
   
    },

    "AR" : {

        "charge" : 18,
        "mass" : 39.962383,
        "C6" : 79.96045,
        "vdw_radius" : 3.01411,
        "real_vdw_radius" : 188,
        "core_orbitals": 5,
        "name" : "argon",
        "density" : np.array([[2.29450954, -0.84439346, 0, 0, 0, 0.16247279, 0, 0, 0], [-0.84439346, 2.42446138, 0, 0, 0, -0.55006745, 0, 0, 0], [0, 0, 2.12900247, 0, 0, 0, -0.52406734, 0, 0], [0, 0, 0, 2.12900247, 0, 0, 0, -0.52406734, 0], [0, 0, 0, 0, 2.12900247, 0, 0, 0, -0.52406734], [0.16247279, -0.55006745, 0, 0, 0, 2.12522405, 0, 0, 0], [0, 0, -0.52406734, 0, 0, 0, 2.12900247, 0, 0], [0, 0, 0, -0.52406734, 0, 0, 0, 2.12900247, 0], [0, 0, 0, 0, -0.52406734, 0, 0, 0, 2.12900247]])
   
    }

}