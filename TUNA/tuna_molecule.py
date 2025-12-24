from tuna_integrals import tuna_integral as ints
from tuna_util import *
import tuna_basis as bas
import numpy as np



class Atom:

    def __init__(self, basis_charge, mass, origin, C6, vdw_radius, real_vdw_radius, symbol, core_orbitals, ghost=False):

        """

        Defines an atom to be used in a TUNA calculation.

        Args:
            basis_charge (int): Relative nuclear charge for basis set formation
            mass (float): Atomic mass in AMU
            origin (array): Coordinates in form [0, 0, z-coord]
            C6 (float): C6 parameter for D2 dispersion
            vdw_radius (float): Van der Waals radius for D2 dispersion
            symbol (str): Atomic symbol
            ghost (bool, optional): This is a ghost atom
        
        """

        # Defines key atomic parameters
        self.basis_charge = basis_charge
        self.mass = mass
        self.origin = origin

        # Defines D2 dispersion parameters
        self.C6 = C6
        self.vdw_radius = vdw_radius
        self.real_vdw_radius = real_vdw_radius

        # Defines capitalised atomic symbol
        self.symbol = symbol
        
        if "X" in symbol:

            if len(symbol) == 2: self.symbol_formatted = self.symbol.upper()
            elif len(symbol) == 3: self.symbol_formatted = "X" + self.symbol[1].upper() + self.symbol[2].lower()

        else:

            self.symbol_formatted = self.symbol.lower().capitalize() 


        # Defines whether this is a ghost atom
        self.ghost = ghost
        
        # Defines number of core orbitals
        self.core_orbitals = core_orbitals

        # The nuclear charge will be the same as the basis charge, unless this is a ghost atom with zero charge
        self.charge = self.basis_charge if not ghost else 0







class Molecule:


    def __init__(self, atomic_symbols, coordinates, calculation):

        """

        Defines a molecule to be used in a TUNA calculation and calculates several commonly used parameters.

        Args:
            atomic_symbols (list(str)): List of atomic symbols
            coordinates (array): Three-dimensional atomic coordinates, shaped as [[0, 0, 0], [0, 0, z-coord]]
            calculation (Calculation): Calculation object
        
        """

        # Defines key molecular parameters
        self.atomic_symbols = atomic_symbols
        self.basis = calculation.basis

        self.coordinates = coordinates
        self.charge = calculation.charge
        self.multiplicity = calculation.multiplicity

        # Builds list of one or two Atom objects, which contain all relevant atomic data
        self.atoms = []

        for i, symbol in enumerate(atomic_symbols):

            if "X" in symbol:
                
                if symbol == "X": error("One or more atom types not recognised! Check the manual for available atoms.")

                atom_data = atomic_properties["X"]
                which_ghost = atomic_properties[symbol.split("X")[1]]

                atom = Atom(which_ghost["charge"], atom_data["mass"], self.coordinates[i], atom_data["C6"], atom_data["vdw_radius"], atom_data["real_vdw_radius"], symbol, atom_data["core_orbitals"], ghost=True)
            
            else:

                atom_data = atomic_properties[symbol]

                atom = Atom(atom_data["charge"], atom_data["mass"], self.coordinates[i], atom_data["C6"], atom_data["vdw_radius"], atom_data["real_vdw_radius"], symbol, atom_data["core_orbitals"], ghost=False)
                
            self.atoms.append(atom)


        # Charge array in relative atomic charge units and mass array in electron mass units
        self.basis_charges = np.array([atom.basis_charge for atom in self.atoms], dtype=int)
        self.charges = np.array([atom.charge for atom in self.atoms], dtype=int)
        self.masses = np.array([atom.mass for atom in self.atoms], dtype=float) * constants.atomic_mass_unit_in_electron_mass

        # Generates basis data for one type of atom
        self.basis_data = bas.generate_basis(self.basis, self.basis_charges[0], calculation)

        # If two types of atom are present, generate data for both and combine them into one dictionary
        if len(self.atoms) == 2 and self.basis_charges[0] != self.basis_charges[1]: 
            
            self.basis_data = bas.generate_basis(self.basis, self.basis_charges[0], calculation) | bas.generate_basis(self.basis, self.basis_charges[1], calculation)

        # Number of, and list of, basis functions made using form_basis function, accounting for optional decontraction
        try:

            self.n_basis, self.basis_functions = self.form_basis(self.atoms, self.basis_data, calculation.decontract)

        except:

            error("Basis set malformed! If using a custom basis set, check the file format carefully.")

        # Decomposes basis functions list into array of number of exponents in each basis function
        self.primitive_Gaussians = [basis_function.num_exps for basis_function in self.basis_functions]

        # The partitioning of the basis set into atoms is done here, by checking which basis functions are located on which atoms
        self.partitioned_basis_functions = [[bf for bf in self.basis_functions if np.allclose(bf.origin, atom.origin)] for atom in self.atoms]
        self.partition_ranges = [len(group) for group in self.partitioned_basis_functions]
        self.angular_momentum_list = self.generate_angular_momentum_list(self.basis_functions)
        self.n_atomic_orbitals = len(self.basis_functions)

        # If custom masses are used, convert these to electron mass units before updating the mass array
        if calculation.custom_mass_1 is not None: self.masses[0] = calculation.custom_mass_1 * constants.atomic_mass_unit_in_electron_mass
        
        if calculation.custom_mass_2 is not None:
            
            if len(self.masses) > 1: 
                
                self.masses[1] = calculation.custom_mass_2 * constants.atomic_mass_unit_in_electron_mass

            else:

                error("Mass on second atom specified, but only one atom given!")

        # Calculates number of electrons
        self.n_electrons = np.sum(self.charges) - self.charge

        if self.n_electrons < 0: error("Negative number of electrons specified!")

        elif self.n_electrons == 0: error("Zero electrons specified!")



        # Determines molecular point group and molecular structure
        self.point_group = self.determine_point_group()
        self.molecular_structure = self.determine_molecular_structure()

        # Checks if there are any X-prefixed ghost atoms present, as a boolean
        self.ghost_atom_present = any("X" in symbol for symbol in self.atomic_symbols)

        # If a molecule is supplied, calculate the bond length and centre of mass
        if len(self.atoms) == 2: 
            
            self.bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

            if not self.ghost_atom_present:

                self.centre_of_mass = calculate_centre_of_mass(self.masses, self.coordinates)

            else: 
                
                self.centre_of_mass = 0

        else: 

            self.bond_length = "N/A"
            self.centre_of_mass = 0

        # If multiplicity not specified but molecule has an odd number of electrons, set it to a doublet
        if calculation.default_multiplicity and self.n_electrons % 2 != 0: self.multiplicity = 2

        # Set the reference determinant to be used
        calculation.reference = "RHF" if self.multiplicity == 1 and "U" not in calculation.method else "UHF"
    
        # Sets information about alpha and beta electrons for UHF
        self.n_unpaired_electrons = self.multiplicity - 1
        self.n_alpha = int((self.n_electrons + self.n_unpaired_electrons) / 2)
        self.n_beta = int(self.n_electrons - self.n_alpha)
        self.n_doubly_occ = min(self.n_alpha, self.n_beta)
        self.n_occ = self.n_alpha + self.n_beta
        self.n_SO = 2 * self.n_atomic_orbitals
        self.n_virt = self.n_SO - self.n_occ


        # This variable can be used for either RHF or UHF references
        self.n_orbitals = self.n_SO if calculation.reference == "UHF" else self.n_atomic_orbitals

        # Number of core electrons is the same as number of orbitals for UHF, double for RHF
        if calculation.freeze_core:
        
            self.n_core_orbitals = self.atoms[0].core_orbitals if len(self.atoms) == 1 else sum(atom.core_orbitals for atom in self.atoms)

        else:
            
            self.n_core_orbitals = 0

        self.n_core_alpha_electrons = self.n_core_orbitals
        self.n_core_beta_electrons = self.n_core_orbitals
        self.n_core_spin_orbitals = self.n_core_orbitals * 2
        self.n_core_spin_orbitals = calculation.freeze_n_orbitals if type(calculation.freeze_n_orbitals) == int else self.n_core_spin_orbitals
        self.n_core_orbitals = calculation.freeze_n_orbitals if type(calculation.freeze_n_orbitals) == int else self.n_core_orbitals

        if "OMP2" in calculation.method and calculation.reference == "RHF": self.n_core_spin_orbitals *= 2

        # Sets off errors for invalid molecular configurations
        if self.n_electrons % 2 == 0 and self.multiplicity % 2 == 0: error("Impossible charge and multiplicity combination (both even)!")
        if self.n_electrons % 2 != 0 and self.multiplicity % 2 != 0: error("Impossible charge and multiplicity combination (both odd)!")
        if self.n_electrons - self.multiplicity < -1: error("Multiplicity too high for number of electrons!")
        if self.multiplicity < 1: error("Multiplicity must be at least 1!")
        if self.n_electrons > self.n_SO: error("Too many electrons for size of basis set!")
        if calculation.reference == "UHF" and self.n_electrons > len(self.basis_functions) and self.n_electrons % 2 == 0: error("Too many electrons for size of basis set!")

        # Sets off errors for impossible correlated calculations
        if calculation.method in correlated_methods:

            if self.n_virt <= 0 and calculation.reference == "RHF" or self.n_virt <= 1 and calculation.reference == "UHF": 
                
                error("Correlated calculation requested on system with insufficient virtual orbitals!")
        
        if calculation.method in ["CCSD[T]", "UCCSD[T]", "CCSDT", "UCCSDT", "QCISD[T]", "UQCISD[T]"] and self.n_electrons == 2: error("Triple excitations have been requested on a two-electron system!")
        
        # Sets off errors for invalid use of restricted Hartree-Fock
        if calculation.reference == "RHF" or calculation.method == "RHF":

            if self.n_electrons % 2 != 0: error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
            if self.multiplicity != 1: error("Restricted Hartree-Fock is not compatible non-singlet states!")

        if calculation.method in ["MP4", "MP4[SDQ]", "MP4[SDTQ]", "MP4[DQ]", "LMP2", "IMP2"] and calculation.reference == "UHF":
                
            error(f"The {calculation.method} method is only implemented for spin-restricted references!")

        # Sets 2 electrons per orbital for RHF, otherwise 1 for UHF
        calculation.n_electrons_per_orbital = 2 if calculation.reference == "RHF" else 1


        # Determines whether the density should be read in between steps
        calculation.MO_read = False if calculation.reference == "UHF" and self.multiplicity == 1 and not calculation.MO_read_requested and not calculation.no_rotate_guess or calculation.no_MO_read or calculation.rotate_guess else True




    def generate_angular_momentum_list(self, basis_functions):

        """
        
        Generates a list of angular momentum letters for the subshells in a basis set.

        Args:
            basis_functions (array[Basis]): Array of basis functions
        
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




        


    def form_basis(self, atoms, basis_data, decontract):

        """
        
        Builds the basis functions for the molecule.

        Args:
            atoms (list(Atom)): List of Atom objects
            basis_data (dict): Basis set data
            decontract (bool): Should the basis set be fully decontracted

        Returns:
            n_basis (int): Number of basis functions
            basis_functions (array(Basis)): Array of basis functions

        """

        basis_functions = []

        for atom in atoms:

            for ang_mom, pgs in basis_data[atom.basis_charge]:

                exps = [e for e, c in pgs]
                coeffs = [c for e, c in pgs]

                # Considering all combinations of Cartesian harmonics
                for shell in self.convert_angular_momentum_to_subshell(ang_mom):

                    if decontract:

                        for e in exps:

                            # Coefficients of 1, length of 1 exponent per basis function
                            basis_functions.append(ints.Basis(np.asarray(atom.origin), np.asarray(shell), 1, np.asarray([e]), np.asarray([1.0])))
                            
                    else: 
                        
                        basis_functions.append(ints.Basis(np.asarray(atom.origin), np.asarray(shell), len(exps), np.asarray(exps), np.asarray(coeffs)))

        # Determines number of basis functions
        n_basis = len(basis_functions)
    
        return n_basis, basis_functions





    def convert_angular_momentum_to_subshell(self, ang_mom):

        """
        
        Converts angular momentum string from basis data into array of subshells, for Cartesian harmonics.

        Args:
            ang_mom (str): Angular momentum
        
        Returns:
            shells[str(ang_mom)] (array): Array of triples of subshells

        """

        subshells = {

            "S": [(0, 0, 0)],
            "P": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            "D": [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
            "F": [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1), (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
            "G": [(4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
            "H": [(5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1), (1, 0, 4), (0, 1, 4), (3, 2, 0), (3, 0, 2), (0, 3, 2), (2, 3, 0), (2, 0, 3), (0, 2, 3), (3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 1), (2, 1, 2), (1, 2, 2)],
            "I": [(6, 0, 0), (0, 6, 0), (0, 0, 6), (5, 1, 0), (5, 0, 1), (1, 5, 0), (0, 5, 1), (1, 0, 5), (0, 1, 5), (4, 2, 0), (4, 0, 2), (0, 4, 2), (2, 4, 0), (2, 0, 4), (0, 2, 4), (4, 1, 1), (1, 4, 1), (1, 1, 4), (3, 3, 0), (3, 0, 3), (0, 3, 3), (3, 2, 1), (3, 1, 2), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (2, 2, 2)]
        
        }
        
        return subshells[str(ang_mom)]
    





    def determine_point_group(self):

        """

        Determines the point group of a molecule.

        Returns:
            point_group (string) : Molecular point group

        """

        # Two same atoms -> Dinfh, two different atoms -> Cinfv, single atom -> K
        if len(self.atomic_symbols) == 2 and "X" not in self.atomic_symbols[0] and "X" not in self.atomic_symbols[1]:

            return "Dinfh" if self.atomic_symbols[0] == self.atomic_symbols[1] else "Cinfv"

        return "K"






    def determine_molecular_structure(self):

        """

        Determines molecular structure of a molecule.

        Returns:
            molecular_structure (string) : Molecular structure representation

        """

        if len(self.atomic_symbols) == 2:
            
            # Puts a line between two atoms if two atoms are given, formats symbols nicely
            if "X" not in self.atomic_symbols[0] and "X" not in self.atomic_symbols[1]: molecular_structure = f"{self.atomic_symbols[0].lower().capitalize()} --- {self.atomic_symbols[1].lower().capitalize()}"

            elif "X" in self.atomic_symbols[0]: molecular_structure = f"{self.atomic_symbols[1].lower().capitalize()}"
            elif "X" in self.atomic_symbols[1]: molecular_structure = f"{self.atomic_symbols[0].lower().capitalize()}"

        else:

            molecular_structure = self.atomic_symbols[0].lower().capitalize()

        return molecular_structure



