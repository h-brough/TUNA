from tuna_integrals import tuna_integral as ints
from tuna_util import *
from numpy import ndarray
import tuna_basis as bas
import numpy as np
from tuna_calc import Calculation



"""

This is the TUNA module for molecule and atom management, written first for version 0.5.0 and rewritten for version 0.10.1.

At the start of every energy evaluation, a Molecule object is made. This contains information inherited from Calculation, as well as structural information, 
data about the number of electrons, orbitals, etc. and is used to update the reference stored in Calculation (restricted or unrestricted).

This module contains:

1. A class to define an Atom
2. A class to define a Molecule
3. Functions to prepare the molecule, creating attributes stored in the Molecule class (prepare_molecule, etc.)
4. General functions which can be called from elsewhere in the program (calculate_and_print_rotational_constant, determine_point_group, etc.)

"""



@dataclass
class Atom:

    """

    An object representing data about an atom in a molecule.

    """

    # Relative nuclear charge for basis set formation

    basis_charge: int   

    # Atomic mass in AMU

    mass: float   

    # Coordinates

    origin: ndarray  

    # Parameters for D2 dispersion

    C6: float   
    vdw_radius: float  

    # Size of atom for DFT grid formation for heteronuclear diatomics

    real_vdw_radius: float  
    
    # Atomic symbol

    symbol: str  

    # How many orbitals are core

    core_orbitals: int   

    # Atomic spherically averaged HF/STO-3G density matrix

    density: ndarray  

    # Is the atom a ghost atom

    ghost: bool   

    @property
    def charge(self):

        return self.basis_charge if not self.ghost else 0

    @property
    def symbol_formatted(self):

        return "X" + self.symbol[1:].capitalize() if self.ghost else self.symbol.capitalize()










@dataclass
class Molecule:

    """
    
    Defines a molecule to be used in a TUNA calculation and calculates several commonly used parameters.

    Various default values for parameters are specified here. This object is created once per energy evaluation.
    
    """

    # List of atomic symbols

    atomic_symbols: list[str]

    # Nuclear coordinates

    coordinates: ndarray

    # The Calculation object underlying the calculation

    calculation: Calculation

    # Should this current energy evaluation finish after the SCF loop is finished (eg. in guess calculations)

    do_correlation: bool = True


    def __post_init__(self) -> None:

        # Stores the basis set, the charge, multiplicity and if this is a diatomic

        self.basis = self.calculation.basis

        self.charge = self.calculation.charge
        self.multiplicity = self.calculation.multiplicity

        # These inherit from Calculation, so as long as it's updated there it will be reflected here

        self.diatomic = self.calculation.diatomic
        self.monatomic = self.calculation.monatomic

        # Builds the Molecule object

        self.prepare_molecule(self.calculation)

        # Interprets whether we need a RHF/RKS or UHF/UKS calculation

        self.process_restricted_or_unrestricted_reference(self.calculation)

        # Initialises the bond length to None - it is calculated only for diatomics

        self.bond_length = 0.0

        if self.diatomic:
            
            # Determines structure based values if this is a molecule

            self.bond_length = calculate_bond_length(self.coordinates)

            self.reduced_mass = calculate_reduced_mass(self.masses)

            self.rotational_constant_per_cm, _ = calculate_and_print_rotational_constant(self.reduced_mass, self.bond_length, self.calculation, True)

            self.centre_of_mass = calculate_centre_of_mass(self.masses, self.coordinates)






    def prepare_molecule(self, calculation: Calculation) -> None:

        """
        
        Sets up an initial Molecule object.

        Args:
            self (Molecule): Molecule object
            calculation (Calculation): Calculation object
        
        """

        # Builds a list of Atom objects

        self.atoms = build_atom_list(self)

        # Creates arrays of atomic data for molecule

        self.basis_charges = np.array([atom.basis_charge for atom in self.atoms])
        self.charges = np.array([atom.charge for atom in self.atoms])

        self.masses = np.array([atom.mass for atom in self.atoms]) * constants.atomic_mass_unit_in_electron_mass
        self.total_mass = np.sum(self.masses)

        # Determines the basis dictionary for the first atom

        self.basis_data = bas.generate_basis(self.basis, self.basis_charges[0], calculation)

        # If two types of atom are present, generate data for both and combine them into one dictionary

        if len(self.atoms) == 2 and self.basis_charges[0] != self.basis_charges[1]: 
            
            self.basis_data = self.basis_data | bas.generate_basis(self.basis, self.basis_charges[1], calculation)

        # Number of, and list of, basis functions accounting for "DECONTRACT"

        self.n_basis, self.basis_functions = form_basis(self.atoms, self.basis_data, calculation.decontract)

        # Builds a list of all the primitive Gaussians
        
        self.primitive_Gaussians = [basis_function.num_exps for basis_function in self.basis_functions]

        # The partitioning of the basis set into atoms is done here, by checking which basis functions are located on which atoms

        self.partitioned_basis_functions = [[bf for bf in self.basis_functions if np.allclose(bf.origin, atom.origin)] for atom in self.atoms]
        self.partition_ranges = [len(group) for group in self.partitioned_basis_functions]
        self.angular_momentum_list = generate_angular_momentum_list(self.basis_functions)
        
        # Initialises the centre of mass to the origin for now

        self.centre_of_mass = 0

        # If custom masses are used, convert these to electron mass units before updating the mass array

        for i, mass in enumerate([calculation.custom_mass_1, calculation.custom_mass_2]):

            if mass is not None:

                self.masses[i] = mass * constants.atomic_mass_unit_in_electron_mass

        # Calculates number of electrons

        self.n_electrons = np.sum(self.charges) - self.charge

        if self.n_electrons < 0: 
            
            error("Negative number of electrons specified!")

        elif self.n_electrons == 0: 
            
            error("Zero electrons specified!")


        # Checks if there are any X-prefixed ghost atoms present, as a boolean

        self.ghost_atom_present = any(atom.ghost for atom in self.atoms)

        # Determines molecular point group and molecular structure

        self.point_group, self.homonuclear, self.heteronuclear = determine_point_group(self.atoms, self.ghost_atom_present)

        self.molecular_structure = determine_molecular_structure(self.atoms)
        

        return










    def process_restricted_or_unrestricted_reference(self, calculation: Calculation) -> None:
        
        """
        
        Processes the reference for a Molecule (restricted or unrestricted).

        Args:
            self (Molecule): Molecule object
            calculation (Calculation): Calculation object
        
        """

        # If multiplicity not specified but molecule has an odd number of electrons, set it to a doublet

        if calculation.default_multiplicity and self.n_electrons % 2 != 0: self.multiplicity = 2

        # Set the reference determinant to be used

        calculation.reference = "RHF" if self.multiplicity == 1 and not calculation.method.unrestricted else "UHF"

        # Sets information about alpha and beta electrons and occupied and virtual orbitals

        self.n_unpaired_electrons = self.multiplicity - 1
        self.n_alpha = (self.n_electrons + self.n_unpaired_electrons) // 2
        self.n_beta = self.n_electrons - self.n_alpha
        self.n_doubly_occ = min(self.n_alpha, self.n_beta)
        self.n_occ = self.n_alpha + self.n_beta
        self.n_SO = 2 * self.n_basis
        self.n_virt = self.n_SO - self.n_occ

        # This variable can be used for either RHF or UHF references

        self.n_orbitals = self.n_SO if calculation.reference == "UHF" else self.n_basis

        # Adds up total number of core orbitals from atomic data
        
        self.n_core_orbitals = sum(atom.core_orbitals for atom in self.atoms) if calculation.freeze_core else 0
        
        # Number of core electrons is the same as number of orbitals for UHF, double for RHF

        self.n_core_alpha_electrons = self.n_core_orbitals
        self.n_core_beta_electrons = self.n_core_orbitals
        self.n_core_spin_orbitals = self.n_core_orbitals * 2

        # If "FREEZECORE [N]" has been requested, update the number of core orbitals

        self.n_core_spin_orbitals = calculation.freeze_n_orbitals if isinstance(calculation.freeze_n_orbitals, int) else self.n_core_spin_orbitals
        self.n_core_orbitals = calculation.freeze_n_orbitals if isinstance(calculation.freeze_n_orbitals, int) else self.n_core_orbitals
        
        # Sets two electrons per orbital for RHF, one for UHF
        
        calculation.n_electrons_per_orbital = 2 if calculation.reference == "RHF" else 1

        # Determines whether the density should be read in between steps - stored in Calculation object

        calculation.MO_read = False if calculation.reference == "UHF" and self.multiplicity == 1 and not calculation.MO_read_requested and not calculation.no_rotate_guess or calculation.no_MO_read or calculation.rotate_guess else True

        # The OMP2 method is only implemented for spin-orbitals, so the number of core spin orbitals needs to be doubled

        if "OMP2" in calculation.method.name and calculation.reference == "RHF": 
            
            self.n_core_spin_orbitals *= 2
        
        # Makes sure the molecule is set up correctly

        self.assert_charge_and_multiplicity_errors(calculation)

        # Reduces the complexity of the method to full configuration interaction (CCSDT -> CCSD for helium, etc.)

        calculation.method = reduce_method_complexity(self, calculation)


        return










    def assert_charge_and_multiplicity_errors(self, calculation: Calculation) -> None:

        """
        
        Sends off errors and closes the program if anything is wrong with the molecular setup.

        Args:
            self (Molecule): Molecule object
            calculation (Calculation): Calculation object
        
        """

        # Sets off errors for invalid molecular configurations

        if self.n_electrons % 2 == 0 and self.multiplicity % 2 == 0: 
            
            error("Impossible charge and multiplicity combination (both even)!")
        
        if self.n_electrons % 2 != 0 and self.multiplicity % 2 != 0: 
            
            error("Impossible charge and multiplicity combination (both odd)!")

        if self.n_electrons - self.multiplicity < -1: 
            
            error("Multiplicity too high for number of electrons!")

        if self.multiplicity < 1: 
            
            error("Multiplicity must be at least 1!")

        if self.n_electrons > self.n_SO: 
            
            error("Too many electrons for size of basis set!")

        if calculation.reference == "UHF" and self.n_electrons > len(self.basis_functions) and self.n_electrons % 2 == 0 and self.multiplicity > self.n_electrons: 
            
            error("Too many electrons for size of basis set!")

        # Sets off errors for invalid use of restricted Hartree-Fock

        if calculation.reference == "RHF" or calculation.method.name == "RHF":

            if self.n_electrons % 2 != 0: 
                
                error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
            
            if self.multiplicity != 1: 
                
                error("Restricted Hartree-Fock is not compatible non-singlet states!")

        return
        









def build_atom_list(molecule: Molecule) -> list[Atom]:

    """
    
    Builds a list of Atom objects.

    Args:
        molecule (Molecule): Molecule object
    
    Returns:
        atoms (list): List of Atoms
    
    """

    atoms = []

    for i, symbol in enumerate(molecule.atomic_symbols):

        # Handles ghost atoms

        if "X" in symbol:
            
            if symbol == "X": 
                
                error("One or more atom types not recognised! Check the manual for available atoms.")

            atom_data = atomic_properties["X"]

            # This is a ghost atom - but which ghost atom is it?

            which_ghost = atomic_properties[symbol.split("X")[1]]

            atom = Atom(which_ghost["charge"], atom_data["mass"], molecule.coordinates[i], atom_data["C6"], atom_data["vdw_radius"], atom_data["real_vdw_radius"], symbol, atom_data["core_orbitals"], atom_data["density"], ghost = True)
        
        else:

            atom_data = atomic_properties[symbol]

            atom = Atom(atom_data["charge"], atom_data["mass"], molecule.coordinates[i], atom_data["C6"], atom_data["vdw_radius"], atom_data["real_vdw_radius"], symbol, atom_data["core_orbitals"], atom_data["density"], ghost = False)
            
        atoms.append(atom)

    return atoms 










def generate_angular_momentum_list(basis_functions: list) -> list:

    """
    
    Generates a list of angular momentum letters for the subshells in a basis set.

    Args:
        basis_functions (list): Array of basis functions
    
    Returns:
        angular_momentum_list (list): List of angular momentum letters

    """

    angular_momentum_to_letter = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}

    angular_momentum_list = []

    for basis_function in basis_functions:

        # Contracts all primitives with same value of L to one letter

        l = sum(basis_function.shell)

        angular_momentum_list.append(angular_momentum_to_letter[l])

    return angular_momentum_list




        





def form_basis(atoms: list, basis_data: dict, decontract: bool) -> tuple:

    """
    
    Builds the basis functions for the molecule.

    Args:
        atoms (list): List of Atoms
        basis_data (dict): Basis set data
        decontract (bool): Should the basis set be fully decontracted

    Returns:
        n_basis (int): Number of basis functions
        basis_functions (list): List of Basis objects

    """

    basis_functions = []

    try:

        for atom in atoms:

            for ang_mom, pgs in basis_data[atom.basis_charge]:

                exps = [e for e, c in pgs]
                coeffs = [c for e, c in pgs]

                # Considering all combinations of Cartesian harmonics

                for shell in convert_angular_momentum_to_subshell(ang_mom):

                    if decontract:

                        for e in exps:

                            # Coefficients of 1, length of 1 exponent per basis function

                            basis_functions.append(ints.Basis(atom.origin, shell, 1, [e], [1.0]))
                            
                    else: 
                        
                        basis_functions.append(ints.Basis(atom.origin, shell, len(exps), exps, coeffs))
    
    except:

        error("Basis set malformed! If using a custom basis set, check the file format carefully.")

    # Determines number of basis functions

    n_basis = len(basis_functions)


    return n_basis, basis_functions










def convert_angular_momentum_to_subshell(angular_momemtum_string: str) -> list:

    """
    
    Converts angular momentum string from basis data into array of subshells, for Cartesian harmonics.

    Args:
        angular_momemtum_string (str): Angular momentum, ie. "S", "P", "D", etc.
    
    Returns:
        exponent_list (list): List of triples of subshells

    """

    # These are the exponents on each Cartesian Gaussian x, y, z for each atomic orbital within each subshell

    subshells = {

        "S": [(0, 0, 0)],
        "P": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        "D": [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
        "F": [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
        "G": [(4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
        "H": [(5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1), (1, 0, 4), (0, 1, 4), (3, 2, 0), (3, 0, 2), (0, 3, 2), (2, 3, 0), (2, 0, 3), (0, 2, 3), (3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 1), (2, 1, 2), (1, 2, 2)],
        "I": [(6, 0, 0), (0, 6, 0), (0, 0, 6), (5, 1, 0), (5, 0, 1), (1, 5, 0), (0, 5, 1), (1, 0, 5), (0, 1, 5), (4, 2, 0), (4, 0, 2), (0, 4, 2), (2, 4, 0), (2, 0, 4), (0, 2, 4), (4, 1, 1), (1, 4, 1), (1, 1, 4), (3, 3, 0), (3, 0, 3), (0, 3, 3), (3, 2, 1), (3, 1, 2), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (2, 2, 2)]
    
    }
    
    exponent_list = subshells[str(angular_momemtum_string)]

    return exponent_list










def determine_point_group(atoms: list[Atom], ghost_atom_present: bool) -> tuple:

    """

    Determines the point group of a molecule.

    Args:
        atoms (list): Atoms
        ghost_atom_present (bool): Is there a ghost atom

    Returns:
        point_group (string) : Molecular point group
        homonuclear (bool) : The molecule is homonuclear
        heteronuclear (bool) : The molecule is heteronuclear

    """

    point_group = "K"

    # Two same atoms -> Dinfh, two different atoms -> Cinfv, single atom -> K

    if len(atoms) == 2 and not ghost_atom_present:

        point_group = "Dinfh" if atoms[0].symbol == atoms[1].symbol else "Cinfv"


    homonuclear = point_group == "Dinfh"
    heteronuclear = point_group == "Cinfv"

    return point_group, homonuclear, heteronuclear










def determine_molecular_structure(atoms: list[Atom]) -> str:

    """

    Determines molecular structure diagram for a molecule.

    Args:
        atoms (list): Atoms

    Returns:
        molecular_structure (string) : Molecular structure representation

    """

    molecular_structure = atoms[0].symbol_formatted

    if len(atoms) == 2:
        
        # Puts a line between two atoms if two atoms are given, formats symbols nicely

        if atoms[0].ghost:
            
            molecular_structure = atoms[1].symbol_formatted

        elif atoms[1].ghost:
           
            molecular_structure = atoms[0].symbol_formatted

        else: 
            
            molecular_structure = atoms[0].symbol_formatted + " --- " + atoms[1].symbol_formatted


    return molecular_structure










def calculate_reduced_mass(masses: ndarray) -> float: 

    """

    Calculates the reduced mass.

    Args:   
        masses (array): Mass array in atomic units

    Returns:
        reduced_mass (float): Reduced mass in atomic units

    """

    reduced_mass = np.prod(masses) / np.sum(masses) 

    return reduced_mass










def calculate_and_print_rotational_constant(reduced_mass: float, bond_length: float, calculation: Calculation, silent: bool = False) -> tuple:

    """

    Calculates the rotational constant of a molecule.

    Args:   
        reduced_mass (float): Reduced mass in atomic units
        bond_length (float): Bond length in bohr
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        rotational_constant_per_cm (float): Rotational constant in per cm
        rotational_constant_GHz (float): Rotational constant in GHz

    """
    
    # Standard equation for linear molecule's rotational constant

    rotational_constant_hartree = 1 / (2 * reduced_mass * bond_length ** 2)

    # Various unit conversions  
    
    rotational_constant_per_bohr = rotational_constant_hartree / (constants.h * constants.c)
    rotational_constant_per_cm = rotational_constant_per_bohr / (100 * constants.bohr_in_metres)
    rotational_constant_GHz = constants.per_cm_in_GHz * rotational_constant_per_cm
                    
    log(f"\n Rotational constant (GHz):            {rotational_constant_GHz:12.6f}", calculation, 2, silent=silent)
    log(f" Rotational constant (per cm):         {rotational_constant_per_cm:12.6f}", calculation, 2, silent=silent)
    
    return rotational_constant_per_cm, rotational_constant_GHz










def reduce_method_complexity(molecule: Molecule, calculation: Calculation) -> str:

    """
    
    In a situation where full configuration interaction can be done and a more complicated method requested, simplify it.
    
    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
    
    Returns:
        updated_method (str): Method with reduced complexity.


    """

    # Only adjusts a method if its correlated

    updated_method = calculation.method

    unrestricted = calculation.reference == "UHF"

    # Skips any correlation if this is a one-electron system - allows it for DFT

    if molecule.n_electrons == 1 and calculation.method.correlated_method:

        updated_method = Method("HF", "Hartree-Fock theory", unrestricted = unrestricted) 

    # Ignores triple excitations if this is a two-electron system

    elif molecule.n_electrons == 2:

        if calculation.method.name in ["CCSD[T]", "CCSDT", "CCSDT[Q]", "CCSDTQ"]: 
            
            updated_method = Method("CCSD", "coupled cluster singles and doubles", method_base = "CC", unrestricted = unrestricted) 


        if calculation.method in ["QCISD[T]"]: 
            
            updated_method = Method("QCISD", "quadratic configuration interaction singles and doubles", method_base = "CC", unrestricted = unrestricted) 
    
    # Ignores quadruple excitations if this is a two-electron system

    elif molecule.n_electrons == 3:

        if calculation.method.name in ["CCSDT[Q]", "CCSDTQ"]: 
            
            updated_method = Method("CCSDT","coupled cluster singles, doubles and triples", method_base = "CC", unrestricted = unrestricted) 
            

    return updated_method