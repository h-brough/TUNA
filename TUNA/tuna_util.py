import numpy as np
import time, sys
import tuna_basis as basis_sets
from termcolor import colored


calculation_types = {

    "SPE": "Single point energy",
    "OPT": "Geometry optimisation",
    "FREQ": "Harmonic frequency",
    "OPTFREQ": "Optimisation and harmonic frequency",
    "SCAN": "Coordinate scan",
    "MD": "Ab initio molecular dynamics"
    
    }



method_types = {
    
    "HF": "Hartree-Fock theory", 
    "RHF": "restricted Hartree-Fock theory", 
    "UHF": "unrestricted Hartree-Fock theory", 
    "MP2": "MP2 theory", 
    "UMP2": "unrestricted MP2 theory", 
    "SCS-MP2": "spin-component-scaled MP2 theory", 
    "USCS-MP2": "unrestricted spin-component-scaled MP2 theory", 
    "MP3": "MP3 theory", 
    "UMP3": "unrestricted MP3 theory", 
    "SCS-MP3": "spin-component-scaled MP3 theory", 
    "USCS-MP3": "unrestricted spin-component-scaled MP3 theory", 
    "OMP2": "orbital-optimised MP2 theory", 
    "UOMP2": "unrestricted orbital-optimised MP2 theory",
    "CIS": "configuration interaction singles",        
    "UCIS": "unrestricted configuration interaction singles",
    "CIS[D]": "configuration interaction singles with perturbative doubles",
    "UCIS[D]": "unrestricted configuration interaction singles with perturbative doubles"

    }


basis_types = ["STO-3G", "STO-6G", "3-21G", "4-31G", "6-31G", "6-31+G", "6-31++G", "6-311G", "6-311+G", "6-311++G"]




class Constants:

    """

    Defines all the contants used in TUNA. Fundamental values are taken from the CODATA 2022 recommendations.
    
    Fundamental values are used to define various emergent constants and conversion factors.

    """

    def __init__(self):

        # Fundamental constants to define Hartree land
        self.planck_constant_in_joules_seconds = 6.62607015e-34
        self.elementary_charge_in_coulombs = 1.602176634e-19
        self.electron_mass_in_kilograms = 9.1093837139e-31
        self.permittivity_in_farad_per_metre = 8.8541878188e-12

        # Non-quantum fundamental constants
        self.c_in_metres_per_second = 299792458
        self.k_in_joules_per_kelvin = 1.380649e-23
        self.atomic_mass_unit_in_kg = 1.660539068911e-27
        self.avogadro = 6.02214076e23

        # Emergent unit conversions
        self.reduced_planck_constant_in_joules_seconds = self.planck_constant_in_joules_seconds / (2 * np.pi)
        self.bohr_in_metres = 4 * np.pi * self.permittivity_in_farad_per_metre * self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.elementary_charge_in_coulombs ** 2)
        self.hartree_in_joules = self.reduced_planck_constant_in_joules_seconds ** 2 / (self.electron_mass_in_kilograms * self.bohr_in_metres ** 2)
        self.atomic_time_in_seconds = self.reduced_planck_constant_in_joules_seconds /  self.hartree_in_joules
        self.atomic_time_in_femtoseconds = self.atomic_time_in_seconds * 10 ** 15
        self.bohr_radius_in_angstrom = self.bohr_in_metres * 10 ** 10
        self.pascal_in_atomic_units = self.hartree_in_joules / self.bohr_in_metres ** 3
        self.per_cm_in_hartree = self.hartree_in_joules / (self.c_in_metres_per_second * self.planck_constant_in_joules_seconds * 10 ** 2)
        self.per_cm_in_GHz = self.hartree_in_joules / (self.planck_constant_in_joules_seconds * self.per_cm_in_hartree * 10 ** 9)
        self.atomic_mass_unit_in_electron_mass = self.atomic_mass_unit_in_kg / self.electron_mass_in_kilograms
        self.eV_in_hartree = self.hartree_in_joules / self.elementary_charge_in_coulombs

        # Emergent constants
        self.c = self.c_in_metres_per_second * self.atomic_time_in_seconds / self.bohr_in_metres
        self.k = self.k_in_joules_per_kelvin / self.hartree_in_joules
        self.h = self.planck_constant_in_joules_seconds / (self.hartree_in_joules * self.atomic_time_in_seconds)

        self.atom_properties = {

            "H" : {
                "charge" : 1,
                "mass" : 1.00782503223 * self.atomic_mass_unit_in_electron_mass,
                "C6" : 2.4284,
                "vdw_radius" : 1.8916
            },

            "XH" : {
                "charge" : 0,
                "mass" :0,
                "C6" : 0,
                "vdw_radius" : 0
            },


            "HE" : {
                "charge" : 2,
                "mass" : 4.00260325413 * self.atomic_mass_unit_in_electron_mass,
                "C6" : 1.3876,
                "vdw_radius" : 1.9124
            },

            "XHE" : {
                "charge" : 0,
                "mass" :0,
                "C6" : 0,
                "vdw_radius" : 0
            }
        }


        self.convergence_criteria_SCF = {

            "loose" : {"delta_E": 0.000001, "max_DP": 0.00001, "RMS_DP": 0.000001, "orbital_gradient": 0.0001, "name": "loose"},
            "medium" : {"delta_E": 0.0000001, "max_DP": 0.000001, "RMS_DP": 0.0000001, "orbital_gradient": 0.00001, "name": "medium"},
            "tight" : {"delta_E": 0.000000001, "max_DP": 0.00000001, "RMS_DP": 0.000000001, "orbital_gradient": 0.0000001, "name": "tight"},
            "extreme" : {"delta_E": 0.00000000001, "max_DP": 0.0000000001, "RMS_DP": 0.00000000001, "orbital_gradient": 0.000000001, "name": "extreme"}   
            
        }


        self.convergence_criteria_optimisation = {

            "loose" : {"gradient": 0.001, "step": 0.01},
            "medium" : {"gradient": 0.0001, "step": 0.0001},
            "tight" : {"gradient": 0.000001, "step": 0.00001},
            "extreme" : {"gradient": 0.00000001, "step": 0.0000001}   

        }



constants = Constants()




class Calculation:

    """

    Processes and calculates from user-defined parameters specified at the start of a TUNA calculation.

    Various default values for parameters are specified here. This object is created once per TUNA calculation.
    
    """

    def __init__(self, calculation_type, method, start_time, params, basis):

        """

        Initialises calculation object.

        Args:   
            calculation_type (string): Type of calculation
            method (string): Electronic structure method
            start_time (float): Calculation start time
            params (list): List of user-specified parameters
            basis (string): Basis set

        Returns:
            None : This function does not return anything

        """

        # Key calculation parameters
        self.calculation_type = calculation_type
        self.method = method
        self.start_time = start_time
        self.basis = basis
        
        # Secondary important factors to begin a calculation
        self.no_rotate_guess = False
        self.rotate_guess = False
        self.theta = np.pi / 4
        self.level_shift = False
        self.level_shift_parameter = 0.2
        self.trajectory_path = "tuna-trajectory.xyz"
        
        # Process the user-defined parameters
        self.process_params(params)


    def process_params(self, params):
        
        """

        Processes user-defined parameters and sets default values.

        Args:   
            params (list): User-specified parameters

        Returns:
            None : This function does not return anything

        """

        # Processing of simple parameters, either on or off
        self.additional_print = True if "P" in params else False
        self.terse = True if "T" in params else False
        self.decontract = True if "DECONTRACT" in params else False

        self.DIIS = True if "DIIS" in params else True
        self.DIIS_requested = True if "DIIS" in params else False
        self.DIIS = False if "NODIIS" in params else True
        self.damping = True if "DAMP" in params else True
        self.damping = False if "NODAMP" in params else True
        self.slow_conv = True if "SLOWCONV" in params else False
        self.very_slow_conv = True if "VERYSLOWCONV" in params else False
        self.no_levelshift = True if "NOLEVELSHIFT" in params else False

        self.D2 = True if "D2" in params else False
        self.calc_hess = True if "CALCHESS" in params else False
        self.MO_read_requested = True if "MOREAD" in params else False
        self.no_MO_read = True if "NOMOREAD" in params else False 
        self.opt_max = True if "OPTMAX" in params else False
        self.trajectory = True if "TRAJ" in params else False
        self.no_trajectory = True if "NOTRAJ" in params else False

        self.dens_plot = True if "DENSPLOT" in params else False
        self.spin_dens_plot = True if "SPINDENSPLOT" in params else False
        self.scan_plot = True if "SCANPLOT" in params else False

        self.MO_read = False if self.no_MO_read else True

        # Convergence criteria for SCF
        if "LOOSE" in params or "LOOSESCF" in params: self.scf_conv = constants.convergence_criteria_SCF["loose"]
        elif "MEDIUM" in params or "MEDIUMSCF" in params: self.scf_conv = constants.convergence_criteria_SCF["medium"]
        elif "TIGHT" in params or "TIGHTSCF" in params: self.scf_conv = constants.convergence_criteria_SCF["tight"]  
        elif "EXTREME" in params or "EXTREMESCF" in params: self.scf_conv = constants.convergence_criteria_SCF["extreme"]

        elif self.calculation_type in ["OPT", "FREQ", "OPTFREQ", "MD"]: self.scf_conv = constants.convergence_criteria_SCF["tight"]  
        else: self.scf_conv = constants.convergence_criteria_SCF["medium"]

        if self.method in ["CIS", "CIS[D]", "UCIS", "UCIS[D]"]: self.scf_conv = constants.convergence_criteria_SCF["tight"]  

        # Convergence criteria for geometry optimisation
        if "LOOSEOPT" in params: self.geom_conv = constants.convergence_criteria_optimisation["loose"]
        elif "MEDIUMOPT" in params: self.geom_conv = constants.convergence_criteria_optimisation["medium"]
        elif "TIGHTOPT" in params: self.geom_conv = constants.convergence_criteria_optimisation["tight"]
        elif "EXTREMEOPT" in params: self.geom_conv = constants.convergence_criteria_optimisation["extreme"] 

        elif self.calculation_type == "OPTFREQ": self.geom_conv = constants.convergence_criteria_optimisation["tight"]  
        else: self.geom_conv = constants.convergence_criteria_optimisation["medium"]



        # Processing of parameters which have an optional value
        if "LEVELSHIFT" in params: 

            self.level_shift = True

            try:

                params.index("LEVELSHIFT")
                self.level_shift_parameter = float(params[params.index("LEVELSHIFT") + 1])
        
            except:
                pass
        

        if "ROTATE" in params: 

            self.rotate_guess = True

            try:

                params.index("ROTATE")
                self.theta = float(params[params.index("ROTATE") + 1]) * np.pi / 180
        
            except:
                pass


        elif "NOROTATE" in params: 

            self.no_rotate_guess = True
            self.MO_read = True     



        # Automates error messages for parameters with required variables
        def get_param_value(param_name, value_type):

            """

            Gets the requested parameter value, or throws an error if none is given.

            Args:   
                param_name (string): Parameter used could call
                value_type (type): Type of value expected after parameter in list

            Returns:
                None : This function does not return anything

            """

            if param_name in params:

                try: 
                    return value_type(params[params.index(param_name) + 1])
                   
                
                except IndexError: error(f"Parameter \"{param_name}\" requested but no value specified!")
                except ValueError: error(f"Parameter \"{param_name}\" must be of type {value_type.__name__}!")
            
            return 
        

        # Processes parameters with a mandatory value
        self.charge = get_param_value("CHARGE", int) if get_param_value("CHARGE", int) is not None else 0
        self.charge = get_param_value("CH", int) if get_param_value("CH", int) is not None else 0 
        self.multiplicity = get_param_value("MULTIPLICITY", int) if get_param_value("MULTIPLICITY", int) is not None else 1
        self.multiplicity = get_param_value("ML", int) if get_param_value("ML", int) is not None else 1
        self.default_multiplicity = False if "ML" in params or "MULTIPLICITY" in params else True

        # Optimisation, coordinate scan and MD parameters
        self.max_iter = get_param_value("MAXITER", int) or 100
        self.max_step = get_param_value("MAXSTEP", float) or 0.2
        self.default_Hessian = get_param_value("DEFAULTHESS", float) or 1 / 4
        self.geom_max_iter = get_param_value("GEOMMAXITER", int) or 30
        self.geom_max_iter = get_param_value("MAXGEOMITER", int) or 30
        self.scan_step = get_param_value("SCANSTEP", float) or None
        self.scan_number = get_param_value("SCANNUMBER", int) or None
        self.MD_number_of_steps = get_param_value("MDNUMBER", int) or 50
        self.timestep = get_param_value("TIMESTEP", float) or 0.1

        # Thermochemical parameters
        if self.calculation_type == "MD": self.temperature = 0
        else: self.temperature = 298.15

        self.temperature = get_param_value("TEMP", float) or self.temperature
        self.temperature = get_param_value("TEMPERATURE", float) or self.temperature
        self.pressure = get_param_value("PRES", float) or 101325
        self.pressure = get_param_value("PRESSURE", float) or self.pressure
        
        # Correlated calculation parameters
        self.same_spin_scaling = get_param_value("SSS", float) or 1 / 3
        self.opposite_spin_scaling = get_param_value("OSS", float) or 6 / 5
        self.MP3_scaling = get_param_value("MP3S", float) or 1 / 4
        self.OMP2_conv = get_param_value("OMP2CONV", float) or 1e-8
        self.OMP2_max_iter = get_param_value("OMP2MAXITER", int) or 20

        # Excited state parameters
        self.root = get_param_value("ROOT", int) or 1
        self.CIS_contribution_threshold = get_param_value("CISTHRESH", float) or 1
        self.n_states = get_param_value("NSTATES", int) or 10




class Molecule:

    """

    Stores and calculates various widely used molecular properties.

    This object can be created multiple times per TUNA calculation.
    
    """

    def __init__(self, atoms, coordinates, calculation):

        """

        Initialises Molecule object.

        Args:   
            atoms (list): Atom symbol list
            coordinates (array): Three-dimensional coordinate array
            calculation (Calculation): Calculation object

        Returns:
            None : This function does not return anything

        """

        self.atoms = atoms
        self.masses = np.array([constants.atom_properties[atom]["mass"] for atom in self.atoms])
        self.charges = np.array([constants.atom_properties[atom]["charge"]for atom in self.atoms])

        # Key molecular parameters
        self.coordinates = coordinates
        self.charge = calculation.charge
        self.multiplicity = calculation.multiplicity
        self.basis = calculation.basis

        self.n_electrons = np.sum(self.charges) - self.charge

        self.point_group = self.determine_point_group()
        self.molecular_structure = self.determine_molecular_structure()

        # Integral and related data
        self.mol = [basis_sets.generate_atomic_orbitals(atom, self.basis, coord) for atom, coord in zip(self.atoms, self.coordinates)]    
        self.AO_ranges = [len(basis_sets.generate_atomic_orbitals(atom, self.basis, coord)) for atom, coord in zip(self.atoms, self.coordinates)]
        self.atomic_orbitals = [orbital for atom_orbitals in self.mol for orbital in atom_orbitals] 
        self.primitive_Gaussians = [pg for atomic_orbital in self.atomic_orbitals for pg in atomic_orbital]

        # Decontracts orbitals if DECONTRACT keyword is requested
        if calculation.decontract: self.atomic_orbitals = [[pg] for pg in self.primitive_Gaussians]

        # If a molecule is supplied, calculate the bond length and centre of mass
        if len(self.atoms) == 2: 
            
            self.bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

            if not any("X" in atom for atom in self.atoms):

                self.centre_of_mass = calculate_centre_of_mass(self.masses, self.coordinates)

            else: self.centre_of_mass = 0

        else: 

            self.bond_length = "N/A"
            self.centre_of_mass = 0

        # If multiplicity not specified but molecule has an odd number of electrons, set it to a doublet
        if calculation.default_multiplicity and self.n_electrons % 2 != 0: self.multiplicity = 2

        # Set the reference determinant to be used
        if self.multiplicity == 1 and "U" not in calculation.method: calculation.reference = "RHF"
        else: calculation.reference = "UHF"
    
        # Sets information about alpha and beta electrons for UHF
        self.n_unpaired_electrons = self.multiplicity - 1
        self.n_alpha = int((self.n_electrons + self.n_unpaired_electrons) / 2)
        self.n_beta = int(self.n_electrons - self.n_alpha)
        self.n_doubly_occ = min(self.n_alpha, self.n_beta)
        self.n_occ = self.n_alpha + self.n_beta
        self.n_SO = 2 * len(self.atomic_orbitals)
        self.n_virt = self.n_SO - self.n_occ

        # Sets off errors for invalid molecular configurations
        if self.n_electrons % 2 == 0 and self.multiplicity % 2 == 0: error("Impossible charge and multiplicity combination (both even)!")
        if self.n_electrons % 2 != 0 and self.multiplicity % 2 != 0: error("Impossible charge and multiplicity combination (both odd)!")
        if self.n_electrons - self.multiplicity < -1: error("Multiplicity too high for number of electrons!")
        if self.multiplicity < 1: error("Multiplicity must be at least 1!")

        # Sets off errors for invalid use of restricted Hartree-Fock
        if calculation.reference == "RHF":

            if self.n_electrons % 2 != 0: error("Restricted Hartree-Fock is not compatible with an odd number of electrons!")
            if self.multiplicity != 1: error("Restricted Hartree-Fock is not compatible non-singlet states!")

        # Sets 2 electrons per orbital for RHF, otherwise 1 for UHF
        calculation.n_electrons_per_orbital = 2 if calculation.reference == "RHF" else 1


        calculation.MO_read = False if calculation.reference == "UHF" and self.multiplicity == 1 and not calculation.MO_read_requested and not calculation.no_rotate_guess or calculation.no_MO_read or calculation.rotate_guess else True



    def determine_point_group(self):

        """

        Determines point group of a molecule.

        Args:   
            None : This function does not require arguments

        Returns:
            point_group (string) : Molecular point group

        """

        # Two same atoms -> Dinfh, two different atoms -> Cinfv, single atom -> K
        if len(self.atoms) == 2 and "X" not in self.atoms[0] and "X" not in self.atoms[1]:

            point_group = "Dinfh" if self.atoms[0] == self.atoms[1] else "Cinfv"

        elif "X" in self.atoms[0] and "X" in self.atoms[1]: point_group = "None"

        else: point_group = "K"

        return point_group




    def determine_molecular_structure(self):

        """

        Determines molecular structure of a molecule.

        Args:   
            None : This function does not require arguments

        Returns:
            molecular_structure (string) : Molecular structure representation

        """

        if len(self.atoms) == 2:
            
            # Puts a line between two atoms if two atoms are given, formats symbols nicely
            if "X" not in self.atoms[0] and "X" not in self.atoms[1]: molecular_structure = f"{self.atoms[0].lower().capitalize()} --- {self.atoms[1].lower().capitalize()}"
            elif "X" in self.atoms[0] and "X" in self.atoms[1]: molecular_structure = "None" 

            elif "X" in self.atoms[0]: molecular_structure = f"{self.atoms[1].lower().capitalize()}"
            elif "X" in self.atoms[1]: molecular_structure = f"{self.atoms[0].lower().capitalize()}"

        else:

            molecular_structure = self.atoms[0].lower().capitalize()

        return molecular_structure



class Output:

    """

    Stores all the useful outputs of a converged SCF calculation.

    """

    def __init__(self, energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy):
       
        """

        Initialises Output object.

        Args:   
            energy (float): Total energy
            S (array): Overlap matrix in AO basis
            P (array): Density matrix in AO basis
            P_alpha (array): Density matrix for alpha orbitals in AO basis
            P_beta (array): Density matrix for beta orbitals in AO basis
            molecular_orbitals (array): Molecular orbitals in AO basis
            molecular_orbitals_alpha (array): Molecular orbitals for alpha electrons in AO basis
            molecular_orbitals_beta (array): Molecular orbitals for beta electrons in AO basis
            epsilons (array): Orbital eigenvalues
            epsilons_alpha (array): Alpha orbital eigenvalues
            epsilons_beta (array): Beta orbital eigenvalues
            kinetic_energy (float): Kinetic energy
            nuclear_electron_energy (float): Nuclear-electron energy
            coulomb_energy (float): Coulomb energy
            exchange_energy (float): Exchange energy

        Returns:
            None : This function does not return anything

        """

        # Key quantities
        self.energy = energy
        self.S = S

        # Density matrices
        self.P = P
        self.P_alpha = P_alpha
        self.P_beta = P_beta

        # Molecular orbitals
        self.molecular_orbitals = molecular_orbitals
        self.molecular_orbitals_alpha = molecular_orbitals_alpha
        self.molecular_orbitals_beta = molecular_orbitals_beta

        # Eigenvalues
        self.epsilons = epsilons
        self.epsilons_alpha = epsilons_alpha
        self.epsilons_beta = epsilons_beta
        self.epsilons_combined = np.append(self.epsilons_alpha, self.epsilons_beta)

        # Energy components
        self.kinetic_energy = kinetic_energy
        self.nuclear_electron_energy = nuclear_electron_energy
        self.coulomb_energy = coulomb_energy
        self.exchange_energy = exchange_energy




def rotate_coordinates_to_z_axis(difference_vector):

    """

    Calculates axis of rotation and rotates difference vector using Rodrigues' formula.

    Args:   
        difference_vector (array): Difference vector

    Returns:
        difference_vector_rotated (array) : Rotated difference vector on z axis
        rotation_matrix (array) : Rotation matrix

    """

    normalised_vector = difference_vector / np.linalg.norm(difference_vector)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Calculate the axis of rotation by the cross product
    rotation_axis = np.cross(normalised_vector, z_axis)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-10:

        # If the axis is too small, the vector is almost aligned with the z-axis
        rotation_matrix = np.eye(3)

    else:

        # Normalize the rotation axis
        rotation_axis /= axis_norm
        
        # Calculate the angle of rotation by the dot product
        cos_theta = np.dot(normalised_vector, z_axis)
        sin_theta = axis_norm
        
        # Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
        
        rotation_matrix = np.eye(3, dtype=np.float64) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    
    # Rotate the difference vector to align it with the z-axis
    difference_vector_rotated = np.dot(rotation_matrix, difference_vector)
    
    return difference_vector_rotated, rotation_matrix





def bohr_to_angstrom(length): 
    
    """

    Converts length in bohr to length in angstroms.

    Args:   
        length (float): Length in bohr

    Returns:
        constants.bohr_radius_in_angstrom * length (float) : Length in angstrom

    """
    
    return constants.bohr_radius_in_angstrom * length




def angstrom_to_bohr(length): 
    
    """

    Converts length in angstrom to length in bohr.

    Args:   
        length (float): Length in angstrom

    Returns:
        length / constants.bohr_radius_in_angstrom  (float) : Length in bohr

    """
    
    return length / constants.bohr_radius_in_angstrom 




def one_dimension_to_three(coordinates_1D): 
    
    """

    Converts 1D coordinate array into 3D.

    Args:   
        coordinates (array): Coordinates in one dimension

    Returns:
        coordinates_3D (array) : Coordinates in three dimensions

    """

    coordinates_3D = np.array([[0, 0, coord] for coord in coordinates_1D])
    
    return coordinates_3D





def three_dimensions_to_one(coordinates_3D): 
    
    """

    Converts 3D coordinate array into 1D.

    Args:   
        coordinates_3D (array): Coordinates in three dimensions

    Returns:
        coordinates_1D (array) : Coordinates in one dimension

    """

    coordinates_1D = np.array([atom_coord[2] for atom_coord in coordinates_3D])
    
    return coordinates_1D
    




def finish_calculation(calculation):

    """

    Finishes the calculation and exits the program.

    Args:   
        calculation (Calculation): Calculation object

    Returns:
        None : This function does not return anything

    """

    # Calculates total time for the TUNA calculation
    end_time = time.perf_counter()
    total_time = end_time - calculation.start_time

    # Prints the finale message
    log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n","white"), calculation, 1)
    
    # Exits the program
    sys.exit()



def calculate_centre_of_mass(masses, coordinates): 
    
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



def print_trajectory(molecule, energy, coordinates, trajectory_path):

    """

    Prints trajectory from optimisation or MD simulation to file.

    Args:   
        molecule (Molecule): Molecule object
        energy (float) : Final energy
        coordinates (array): Atomic coordinates
        trajectory_path (str): Path to file

    Returns:
        None : This function does not return anything

    """

    atoms = molecule.atoms

    with open(trajectory_path, "a") as file:
        
        # Prints energy and atoms
        file.write(f"{len(atoms)}\n")
        file.write(f"Coordinates from TUNA calculation, E = {energy:.10f}\n")

        coordinates_angstrom = bohr_to_angstrom(coordinates)

        # Prints coordinates
        for i in range(len(atoms)):

            file.write(f"  {atoms[i]}      {coordinates_angstrom[i][0]:6f}      {coordinates_angstrom[i][1]:6f}      {coordinates_angstrom[i][2]:6f}\n")

    file.close()





def calculate_one_electron_property(P, M):

    """

    Calculates a one-electron property.

    Args:   
        P (array): One-particle reduced density matrix
        M (array): Property matrix

    Returns:
        property (float) : Property defined by M

    """

    property = np.einsum('ij,ij->', P, M, optimize=True)

    return property





def calculate_two_electron_property(D, M):

    """

    Calculates a two-electron property.

    Args:   
        D (array): Two-particle reduced density matrix
        M (array): Property matrix

    Returns:
        property (float) : Property defined by M

    """

    property = (1 / 4) * np.einsum('ijkl,ijkl->', D, M, optimize=True)

    return property




def error(message): 

    """

    Closes TUNA and prints an error, in light red.

    Args:   
        message (string): Error message

    Returns:
        None : This function does not return anything

    """
    
    print(colored(f"\nERROR: {message}  :(\n", "light_red"))

    # Exits the program
    sys.exit()




def warning(message, space=1): 
    
    """

    Prints a warning message, in light yellow.

    Args:   
        message (string): Error message
        space (int, optional): Number of indenting spaces from the left hand side

    Returns:
        None: This function does not return anything

    """
    
    print(colored(f"\n{" " * space}WARNING: {message}", "light_yellow"))




def log(message, calculation, priority=1, end="\n", silent=False, colour="light_grey"):

    """

    Logs a message to the console.

    Args:   
        message (string): Error message
        calculation (Calculation): Calculation object
        priority (int, optional): Priority of message (1 to always appear, 2 to appear unless T keyword used, and 3 only to appear if P keyword used)
        end (string, optional): End of message
        silent (bool, optional): Specifies whether to print anything

    Returns:
        None : This function does not return anything

    """

    if not silent:

        if priority == 1: print(colored(message, colour), end=end)
        elif priority == 2 and not calculation.terse: print(colored(message, colour), end=end)
        elif priority == 3 and calculation.additional_print: print(colored(message, colour), end=end)



