import numpy as np
import time, sys
from termcolor import colored
import os
from functools import partial






def process_keyword(key_strings, default_value, params=None, boolean=True, check_next_space=False, mandatory_value=False, associated_keyword_default=None, value_type=None, plot_path=False):

    """
    
    Processes new keywords and their parameters.

    Args:
        key_strings (list): List of different optional keywords per keyword
        default_value (variable): Default value of keyword
        params (list, optional): List of parameters
        check_next_space (bool): Should the next space be checked
        mandatory_value (bool): Does the parameter have a mandatory value
        associated_keyword_default (variable): Secondary keyword's default value
        value_type (variable): Type of optional parameter
        plot_path (bool): Is this keyword to give a path for a plot

    Returns:
        keyword (variable): The value of the keyword
        associated_keyword (variable, optional): The value of the associated keyword
    
    """

    keyword = default_value
    associated_keyword = associated_keyword_default

    # If no type given, just use the default value
    if value_type is None:
        
        value_type = type(default_value)

    if params:

        for i, param in enumerate(params):

                if param in key_strings:
                    
                    # Simple boolean keywords
                    if boolean:

                        keyword = True

                    if check_next_space:
                        
                        # Makes sure a param is given
                        if i + 1 < len(params):

                            next_value = params[i + 1]

                            try:

                                next_value = value_type(next_value)

                                # Checks for string after keyword with certain extensions
                                if plot_path: 

                                    extensions = [".PNG", ".JPG", ".PDF", ".SVG", ".JPEG", ".TIF", ".TIFF", ".BMP", ".RAW", ".EPS", ".PS"]

                                    if any(params[params.index("SAVEPLOT") + 1].endswith(ext) for ext in extensions):
                                        
                                        associated_keyword = str(params[params.index("SAVEPLOT") + 1]).lower()
                                        keyword = True
                                    
                            # Error if wrong keyword value
                            except ValueError:
                                
                                if mandatory_value:

                                    error(f"Parameter \"{param}\" must be of type {value_type.__name__}, got {type(next_value).__name__} instead!")
                                
                                else:
                                    pass

                            if boolean and not plot_path:

                                associated_keyword = next_value

                                # Lower cases the path given for plotting or trajectory

                                if type(associated_keyword) == str:

                                    if "." in associated_keyword:

                                        associated_keyword = associated_keyword.lower()
                                
                            # Just takes the next value for certain parameters
                            elif not boolean and not plot_path:

                                keyword = next_value   

                        else:
                            
                            associated_keyword = associated_keyword_default

                            if mandatory_value:

                                error(f"Parameter \"{param}\" requested but no value specified!")
                        

    # Return either one or two keywords per keyword
    if check_next_space and boolean:

        return keyword, associated_keyword

    return keyword









class Calculation:

    """

    Processes and calculates from user-defined parameters specified at the start of a TUNA calculation.

    Various default values for parameters are specified here. This object is created once per TUNA calculation.
    
    """

    def __init__(self, calculation_type, method, start_time, params, basis, atoms):
        
        # Defines fundamental calculation parameters
        self.calculation_type = calculation_type
        self.method = method
        self.start_time = start_time
        self.params = params
        self.basis = basis
        self.original_basis = basis
        self.atoms = atoms

        # Prevents running "params" through every function call
        keyword = partial(process_keyword, params=params)

        # Simple boolean keywords
        self.additional_print = keyword(["P"], False)
        self.terse = keyword(["T"], False)
        self.decontract = keyword(["DECONTRACT"], False)

        self.natural_orbitals = keyword(["NATORBS"], False)
        self.no_natural_orbitals = keyword(["NONATORBS"], False)
        self.no_singles = keyword(["NOSINGLES"], False)
        self.MO_read_requested = keyword(["MOREAD"], False)
        self.MO_read = keyword(["MOREAD"], False)
        self.MO_read = not keyword(["NOMOREAD"], False)
        self.no_MO_read = keyword(["NOMOREAD"], False)
        self.D2 = keyword(["D2"], False)
        self.calc_hess = keyword(["CALCHESS"], False)
        self.opt_max = keyword(["OPTMAX"], False)
        self.no_trajectory = keyword(["NOTRAJ"], False)
        self.scan_plot = keyword(["SCANPLOT"], False)
        self.plot_dashed_lines = keyword(["DASH"], False)
        self.plot_dotted_lines = keyword(["DOT"], False)
        self.add_plot = keyword(["ADDPLOT"], False)
        self.extrapolate = keyword(["EXTRAPOLATE"], False)
        
        if keyword(["DELPLOT"], False): delete_saved_plot()

        # Convergence keywords with optional parameters
        self.DIIS, self.max_DIIS_matrices = keyword(["DIIS"], True, check_next_space=True, associated_keyword_default=6, value_type=int)
        if self.DIIS: self.DIIS = not keyword(["NODIIS"], False)

        self.level_shift, self.level_shift_parameter = keyword(["LEVELSHIFT"], False, check_next_space=True, associated_keyword_default=0.2, value_type=float)
        if self.level_shift: self.level_shift = not keyword(["NOLEVELSHIFT"], False)

        self.damping, self.damping_factor = keyword(["DAMP"], True, check_next_space=True, associated_keyword_default=None, value_type=float)
        if self.damping: self.damping = not keyword(["NODAMP"], False)
        if keyword(["SLOWCONV"], False): self.damping_factor = 0.5
        if keyword(["VERYSLOWCONV"], False): self.damping_factor = 0.85

        self.damping = keyword(["DAMP"], True)
        self.damping = not keyword(["NODAMP"], False)
        # Keywords with mandatory parameters
        self.charge = keyword(["CH", "CHARGE"], 0, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.multiplicity = keyword(["ML", "MULTIPLICITY"], 1, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.default_multiplicity = not keyword(["ML", "MULTIPLICITY"], False)
        self.S_eigenvalue_threshold = keyword(["STHRESH"], 1e-7, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.rotate_guess, self.theta = keyword(["ROTATE"], False, check_next_space=True, associated_keyword_default=45, value_type=float)[0], keyword(["ROTATE"], False, check_next_space=True, associated_keyword_default=45, value_type=float)[1] * np.pi / 180
        self.no_rotate_guess = keyword(["NOROTATE"], False)

        # Custom masses
        self.custom_mass_1 = keyword(["M1"], None, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.custom_mass_2 = keyword(["M2"], None, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)

        # Convergence and optimisation keywords
        self.max_iter = keyword(["MAXITER"], 100, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.max_step = keyword(["MAXSTEP"], 0.2, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.default_Hessian = keyword(["DEFAULTHESS"], 0.25, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.geom_max_iter = keyword(["GEOMMAXITER", "MAXGEOMITER"], 30, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.scan_step = keyword(["STEP", "SCANSTEP"], None, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.scan_number = keyword(["NUM", "SCANNUMBER"], None, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.timestep = keyword(["STEP", "TIMESTEP"], 0.1, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.MD_number_of_steps = keyword(["NUM", "MDNUMBER"], 50, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)

        # Thermochemistry keywords
        self.temperature = 0 if self.calculation_type == "MD" else 298.15
        self.temperature = keyword(["TEMP", "TEMPERATURE"], self.temperature, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.pressure = keyword(["PRES", "PRESSURE"], 101325, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)

        # Post-Hartree-Fock keywords
        self.same_spin_scaling = keyword(["SSS"], 1 / 3, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.opposite_spin_scaling = keyword(["OSS"], 6 / 5, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.MP3_scaling = keyword(["MP3S", "MP3SCALING", "MP3SCAL"], 1 / 4, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.OMP2_conv = keyword(["OMP2CONV"], 1e-8, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.OMP2_max_iter = keyword(["OMP2MAXITER"], 30, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.CC_conv = keyword(["CCCONV"], 1e-8, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.amp_conv = keyword(["AMPCONV"], 1e-7, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.CC_max_iter = keyword(["CCMAXITER"], 30, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.cc_damp_requested, self.coupled_cluster_damping_parameter = keyword(["CCDAMP"], False, check_next_space=True, value_type=float, associated_keyword_default=0.25)
        self.coupled_cluster_damping_parameter = self.coupled_cluster_damping_parameter if self.cc_damp_requested else 0
        self.freeze_core, self.freeze_n_orbitals = keyword(["FREEZECORE"], False, boolean=True, check_next_space=True, value_type=int, mandatory_value=False, associated_keyword_default=None)

        # Excited state keywords
        self.root = keyword(["ROOT"], 1, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.CIS_contribution_threshold = keyword(["CISTHRESH"], 1, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.n_states = keyword(["NSTATES"], 10, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)

        # External printing keywords
        self.trajectory, self.trajectory_path = keyword(["TRAJ"], False, boolean=True, check_next_space=True, mandatory_value=False, associated_keyword_default="tuna-trajectory.xyz", value_type=str)
        self.save_plot, self.save_plot_filepath = keyword(["SAVEPLOT"], False, check_next_space=True, associated_keyword_default="TUNA Plot.png", value_type=str, plot_path=True)
        self.scan_plot_colour = next((code for name, code in color_map.items() if name in params), "b")
        self.custom_basis_file = keyword(["BASIS"], None, boolean=False, check_next_space=True, mandatory_value=True, associated_keyword_default="tuna-trajectory.xyz", value_type=str)

        # Convergence keywords for SCF and optimisations
        self.SCF_conv_requested = True if "LOOSE" in params or "LOOSESCF" in params or "MEDIUM" in params or "MEDIUMSCF" in params or "TIGHT" in params or "TIGHTSCF" in params or "EXTREME" in params or "EXTREMESCF" in params else False
        self.geom_conv_requested = True if "LOOSEOPT" in params or "MEDIUMOPT" in params or "TIGHTOPT" in params or "EXTREMEOPT" in params else False

        self.SCF_conv = constants.convergence_criteria_SCF["loose"] if "LOOSE" in params or "LOOSESCF" in params else constants.convergence_criteria_SCF["medium"]
        self.SCF_conv = constants.convergence_criteria_SCF["medium"] if "MEDIUM" in params or "MEDIUMSCF"in params else self.SCF_conv
        self.SCF_conv = constants.convergence_criteria_SCF["tight"] if "TIGHT" in params or "TIGHTSCF" in params else self.SCF_conv
        self.SCF_conv = constants.convergence_criteria_SCF["extreme"] if "EXTREME" in params or "EXTREMESCF" in params else self.SCF_conv
        
        if not self.SCF_conv_requested:
        
            self.SCF_conv = constants.convergence_criteria_SCF["tight"] if self.calculation_type in ["OPT", "FREQ", "OPTFREQ", "MD"] and self.SCF_conv != constants.convergence_criteria_SCF["extreme"] else self.SCF_conv
            self.SCF_conv = constants.convergence_criteria_SCF["tight"] if "CIS" in self.method and self.SCF_conv != constants.convergence_criteria_SCF["extreme"] else self.SCF_conv

        self.geom_conv = constants.convergence_criteria_optimisation["loose"] if "LOOSEOPT" in params else constants.convergence_criteria_optimisation["medium"]
        self.geom_conv = constants.convergence_criteria_optimisation["medium"] if "MEDIUMOPT" in params else self.geom_conv
        self.geom_conv = constants.convergence_criteria_optimisation["tight"] if "TIGHTOPT" in params else self.geom_conv
        self.geom_conv = constants.convergence_criteria_optimisation["extreme"] if "EXTREMEOPT" in params else self.geom_conv
        
        if not self.geom_conv_requested:

            self.geom_conv = constants.convergence_criteria_optimisation["tight"] if self.calculation_type == "OPTFREQ" and self.geom_conv != constants.convergence_criteria_optimisation["extreme"] else self.geom_conv

        # Processes the NOSINGLES keyword
        self.method = process_no_singles_keyword(self.method, self.no_singles)



        




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
        self.avogadro = 6.02214076e23

        # Emergent unit conversions
        self.atomic_mass_unit_in_kg = 0.001 / self.avogadro
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




def delete_saved_plot():

    """
    
    Deletes a pickle plot, if it exists.
    
    """


    file_path = "TUNA-plot-temp.pkl"

    if os.path.exists(file_path):
        
        os.remove(file_path)
        warning(f"The file {file_path} has been deleted due to the DELPLOT keyword.\n",space=0)

    else:
        
        warning(f"Plot deletion requested but {file_path} could not be found!\n",space=0)






def process_no_singles_keyword(method, no_singles):

    """"
    
    Processes the NOSINGLES keyword.

    Args:
        method (str): Electronic structure method
        no_singles (bool): NOSINGLES keyword used?

    Returns:
        method (str): Updated electronic structure method
    
    """

    prefix = ""

    # Makes sure U is not lost o
    if method.startswith("U"):

        prefix = "U"
        method = method[1:]  

    if "CEPA" in method: 
        
        # CEPA0 defaults to LCCSD, but turns into LCCD if NOSINGLES is used
        method = "LCCSD"

    if no_singles:

        # CCSD methods become CCD if NOSINGLES is used
        if method == "LCCSD": method = "LCCD"
        elif method == "CCSD": method = "CCD"
        elif method == "CCSD[T]": method = "CCD"
        elif method == "CCSDT": method = "CCD"


    method = prefix + method  

    return method








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
    
    if total_time > 120:

        minutes = total_time // 60
        seconds = total_time % 60

        if total_time > 7200:
            
            hours = total_time // 3600
            extra_minutes = (total_time % 3600) // 60

            log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {hours:.0f} hours, {extra_minutes:.0f} minutes and {seconds:.2f} seconds.  :)\n","white"), calculation, 1)

        else:
            
            log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {minutes:.0f} minutes and {seconds:.2f} seconds.  :)\n","white"), calculation, 1)

    # Prints the finale message
    else:
        
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
    atomic_symbols = molecule.atomic_symbols
    
    with open(trajectory_path, "a") as file:
        
        # Prints energy and atomic_symbols
        file.write(f"{len(atomic_symbols)}\n")
        file.write(f"Coordinates from TUNA calculation, E = {energy:.10f}\n")

        coordinates_angstrom = bohr_to_angstrom(coordinates)

        # Prints coordinates
        for i in range(len(atomic_symbols)):

            file.write(f"  {atomic_symbols[i]}      {coordinates_angstrom[i][0]:6f}      {coordinates_angstrom[i][1]:6f}      {coordinates_angstrom[i][2]:6f}\n")

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
        elif priority == 2 and not calculation.terse: print(colored(message, colour, force_color = True), end=end)
        elif priority == 3 and calculation.additional_print: print(colored(message, colour, force_color = True), end=end)









def log_spacer(calculation, priority=1, start="", end="", space=" ", silent=False):

    log(f"{start}{space}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{end}", calculation, priority=priority, silent=silent)



def log_big_spacer(calculation, priority=1, start="", end="", space=" ", silent=False):

    log(f"{start}{space}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{end}", calculation, priority=priority, silent=silent)






def scan_plot(calculation, bond_lengths, energies):

    """

    Interfaces with matplotlib to plot energy as a function of bond length.

    Args:
        calculation (Calculation): Calculation object
        bond_lengths (array): List of bond lengths  
        energies (array): List of energies at each bond length

    Returns:
        None: Nothing is returned

    """

    log("\nPlotting energy profile diagram...      ", calculation, 1, end=""); sys.stdout.flush()
    
    import matplotlib.pyplot as plt
    import matplotlib
    import pickle
    from matplotlib import font_manager as fm

    plot_font = ["Consolas", "Liberation Mono", "DejaVu San"]

    matplotlib.rcParams['font.family'] = plot_font

    # Saves temporary file if ADDPLOT used
    if calculation.add_plot:

        try:
        
            with open("TUNA-plot-temp.pkl", "rb") as f:
                fig = pickle.load(f)
                ax = fig.axes[0]

        except:

            fig, ax = plt.subplots(figsize=(10,6))    
    
    else: 
        
        fig, ax = plt.subplots(figsize=(10,6))   


    def mag_then_sign(n):

        if n == 1: return '+'
        if n == -1: return '-'
        
        return f"{abs(n)}{'+' if n > 0 else '-'}"


    legend_label = f"{calculation.method}/{calculation.basis}" if "CIS" not in calculation.method else f"{calculation.method}/{calculation.basis}, ROOT {calculation.root}"


    charge = "" if calculation.charge == 0 else mag_then_sign(calculation.charge)
    
    linestyle = "--" if calculation.plot_dashed_lines else ":" if calculation.plot_dotted_lines else "-"

    font_prop = fm.FontProperties(family=plot_font, size=12)

    plt.plot(bond_lengths, energies, color=calculation.scan_plot_colour,linewidth=1.75, label=legend_label, linestyle=linestyle)
    plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily=plot_font,fontsize=14)
    plt.ylabel("Energy (Hartree)",labelpad=10, fontweight="bold", fontfamily=plot_font,fontsize=14)
    plt.legend(loc="upper right", fontsize=12, frameon=False, handlelength=4, prop=font_prop)
    plt.title(f"TUNA Calculation on "f"{calculation.atoms[0].capitalize()}—"f"{calculation.atoms[1].capitalize()}"rf"$^{{{charge}}}$ Molecule",fontweight="bold",fontsize=16,fontfamily=plot_font,pad=15)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')
    
    for spine in ax.spines.values(): spine.set_linewidth(1.25)
    
    plt.minorticks_on()
    plt.tight_layout() 

    log("[Done]", calculation, 1)

    if calculation.add_plot:

        with open("TUNA-plot-temp.pkl", "wb") as f:

            pickle.dump(fig, f)

    log("Saving energy profile diagram...      ", calculation, 1, end=""); sys.stdout.flush()
    
    if calculation.save_plot:

        plt.savefig(calculation.save_plot_filepath, dpi=1200, transparent=True)

    log("  [Done]", calculation, 1)

    log(f"\nSaved plot as \"{calculation.save_plot_filepath}\"", calculation, 1)    
    
    # Shows the coordinate scan plot
    plt.show()








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
    "MP4": "MP4 theory",
    "MP4[SDTQ]": "MP4 theory",
    "MP4[SDQ]": "MP4 theory with singles, doubles and quadruples",
    "MP4[DQ]": "MP4 theory with doubles and quadruples",
    "OMP2": "orbital-optimised MP2 theory", 
    "UOMP2": "unrestricted orbital-optimised MP2 theory",
    "OOMP2": "orbital-optimised MP2 theory", 
    "UOOMP2": "unrestricted orbital-optimised MP2 theory",
    "CIS": "configuration interaction singles",        
    "UCIS": "unrestricted configuration interaction singles",
    "CIS[D]": "configuration interaction singles with perturbative doubles",
    "UCIS[D]": "unrestricted configuration interaction singles with perturbative doubles",
    "CCD": "coupled cluster doubles",
    "UCCD": "unrestricted coupled cluster doubles",
    "CEPA0": "coupled electron pair approximation",
    "UCEPA0": "unrestricted coupled electron pair approximation",
    "LCCD": "linearised coupled cluster doubles",
    "ULCCD": "unrestricted linearised coupled cluster doubles",
    "LCCSD": "linearised coupled cluster singles and doubles",
    "ULCCSD": "unrestricted linearised coupled cluster singles and doubles",
    "CEPA": "coupled electron pair approximation",
    "UCEPA": "unrestricted coupled electron pair approximation",
    "CEPA[0]": "coupled electron pair approximation",
    "UCEPA[0]": "unrestricted coupled electron pair approximation",
    "CCSD": "coupled cluster singles and doubles",
    "UCCSD": "unrestricted coupled cluster singles and doubles",
    "CCSD[T]": "coupled cluster singles, doubles and perturbative triples",
    "UCCSD[T]": "unrestricted coupled cluster singles, doubles and perturbative triples",
    "CCSDT": "coupled cluster singles, doubles and triples",
    "UCCSDT": "unrestricted coupled cluster singles, doubles and triples"

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




color_map = {

    "RED": "r",
    "GREEN": "g",
    "BLUE": "b",
    "CYAN": "c",
    "MAGENTA": "m",
    "YELLOW": "y",
    "BLACK": "k",
    "WHITE": "w",
}






atomic_properties = {
            
    "X" : {
        "charge" : 0,
        "mass" : 0,
        "C6" : 0,
        "vdw_radius" : 0,
        "core_orbitals": 0,
        "name" : "ghost"
    },

    "H" : {
        "charge" : 1,
        "mass" : 1.007825,
        "C6" : 2.4283,
        "vdw_radius" : 1.8916,
        "core_orbitals": 0,
        "name" : "hydrogen"
    },

    "HE" : {
        "charge" : 2,
        "mass" : 4.002603,
        "C6" : 1.3876,
        "vdw_radius" : 1.9124,
        "core_orbitals": 0,
        "name" : "helium"
    },

    "LI" : {
        "charge" : 3,
        "mass" : 7.016004,
        "C6" : 27.92545,
        "vdw_radius" : 1.55902,
        "core_orbitals": 0,
        "name" : "lithium"
    },

    "BE" : {
        "charge" : 4,
        "mass" : 9.012182,
        "C6" : 27.92545,
        "vdw_radius" : 2.66073,
        "core_orbitals": 0,
        "name" : "beryllium"
    },

    "B" : {
        "charge" : 5,
        "mass" : 11.009305,
        "C6" : 54.28985,
        "vdw_radius" : 2.80624,
        "core_orbitals": 1,
        "name" : "boron"
    },

    "C" : {
        "charge" : 6,
        "mass" : 12.000000,
        "C6" : 30.35375,
        "vdw_radius" : 2.74388,
        "core_orbitals": 1,
        "name" : "carbon"
    },

    "N" : {
        "charge" : 7,
        "mass" : 14.003074,
        "C6" : 21.33435,
        "vdw_radius" : 2.63995,
        "core_orbitals": 1,
        "name" : "nitrogen"
    },

    "O" : {
        "charge" : 8,
        "mass" : 15.994915,
        "C6" : 12.1415,
        "vdw_radius" : 2.53601,
        "core_orbitals": 1,
        "name" : "oxygen"
    },

    "F" : {
        "charge" : 9,
        "mass" : 18.998403,
        "C6" : 13.00875,
        "vdw_radius" : 2.43208,
        "core_orbitals": 1,
        "name" : "fluorine"
    },

    "NE" : {
        "charge" : 10,
        "mass" : 19.992440,
        "C6" : 10.92735,
        "vdw_radius" : 2.34893,
        "core_orbitals": 1,
        "name" : "neon"
    },

    "NA" : {
        "charge" : 11,
        "mass" : 22.989770,
        "C6" : 99.03995,
        "vdw_radius" : 2.16185,
        "core_orbitals": 1,
        "name" : "sodium"
    },

    "MG" : {
        "charge" : 12,
        "mass" : 23.985042,
        "C6" : 99.03995,
        "vdw_radius" : 2.57759,
        "core_orbitals": 1,
        "name" : "magnesium"
    },

    "AL" : {
        "charge" : 13,
        "mass" : 26.981538,
        "C6" : 187.15255,
        "vdw_radius" : 3.09726,
        "core_orbitals": 5,
        "name" : "aluminium"
    },

    "SI" : {
        "charge" : 14,
        "mass" : 27.976927,
        "C6" : 160.09435,
        "vdw_radius" : 3.24277,
        "core_orbitals": 5,
        "name" : "silicon"
    },

    "P" : {
        "charge" : 15,
        "mass" : 30.973762,
        "C6" : 135.9848,
        "vdw_radius" : 3.22198,
        "core_orbitals": 5,        
        "name" : "phosphorus"
    },

    "S" : {
        "charge" : 16,
        "mass" : 31.972071,
        "C6" : 96.61165,
        "vdw_radius" : 3.18041,
        "core_orbitals": 5,
        "name" : "sulfur"
    },

    "CL" : {
        "charge" : 17,
        "mass" : 34.968853,
        "C6" : 87.93915,
        "vdw_radius" : 3.09726,
        "core_orbitals": 5,
        "name" : "chlorine"
    },

    "AR" : {
        "charge" : 18,
        "mass" : 39.962383,
        "C6" : 79.96045,
        "vdw_radius" : 3.01411,
        "core_orbitals": 5,
        "name" : "argon"
    }

}

