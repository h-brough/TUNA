import numpy as np
import time, sys
from termcolor import colored
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

    def __init__(self, calculation_type, method, start_time, params, basis, atomic_symbols):
        
        """

        Initialises the Calculation object.

        calculation_type (str): What kind of calculation - SPE, OPT, FREQ, etc.
        method (str): Electronic structure method
        start_time (time): Start time for calculation, after modules have been imported
        params (list): Requested parameters for calculation
        basis (str): Requested basis set 
        atomic_symbols (list): List of atomic symbols

        """

        # Defines fundamental calculation parameters
        self.calculation_type = calculation_type
        self.method = method
        self.start_time = start_time
        self.params = params
        self.basis = basis
        self.original_basis = basis
        self.atomic_symbols = atomic_symbols
        self.DFT_calculation = True if method in DFT_methods else False

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
        self.delete_plot = keyword(["DELPLOT"], False)
        self.plot_density = keyword(["DENSPLOT"], False)
        self.plot_spin_density = keyword(["SPINDENSPLOT"], False)
        self.plot_HOMO = keyword(["PLOTHOMO"], False)
        self.plot_LUMO = keyword(["PLOTLUMO"], False)
        self.plot_difference_density = keyword(["DIFFDENSPLOT"], False)
        self.plot_difference_spin_density = keyword(["DIFFSPINDENSPLOT"], False)
        self.no_DFT_exchange = keyword(["NOX"], False)
        self.no_DFT_correlation = keyword(["NOC"], False)

        # Convergence keywords with optional parameters
        self.DIIS, self.max_DIIS_matrices = keyword(["DIIS"], True, check_next_space=True, associated_keyword_default=6, value_type=int)
        if self.DIIS: self.DIIS = not keyword(["NODIIS"], False)

        self.damping, self.damping_factor = keyword(["DAMP"], True, check_next_space=True, associated_keyword_default=None, value_type=float)
        if self.damping: self.damping = not keyword(["NODAMP"], False)
        if keyword(["SLOWCONV"], False): self.damping_factor = 0.5
        if keyword(["VERYSLOWCONV"], False): self.damping_factor = 0.85

        self.damping = keyword(["DAMP"], True)
        self.damping = not keyword(["NODAMP"], False)
        self.max_damping = keyword(["MAXDAMP"], 0.700, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)

        # Keywords with mandatory parameters
        self.charge = keyword(["CH", "CHARGE"], 0, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.multiplicity = keyword(["ML", "MULTIPLICITY"], 1, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.default_multiplicity = not keyword(["ML", "MULTIPLICITY"], False)
        self.S_eigenvalue_threshold = keyword(["STHRESH"], 1e-7, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.rotate_guess, self.theta = keyword(["ROTATE"], False, check_next_space=True, associated_keyword_default=45, value_type=float)[0], keyword(["ROTATE"], False, check_next_space=True, associated_keyword_default=45, value_type=float)[1]
        self.no_rotate_guess = keyword(["NOROTATE"], False)

        # Custom masses
        self.custom_mass_1 = keyword(["M1"], None, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.custom_mass_2 = keyword(["M2"], None, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)

        # Convergence and optimisation keywords
        self.max_iter = keyword(["MAXITER"], 100, boolean=False, check_next_space=True, value_type=int, mandatory_value=True) + 1
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
        self.MP2_conv = keyword(["MPCONV"], 1e-8, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.MP2_max_iter = keyword(["MPMAXITER"], 30, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.CC_conv = keyword(["CCCONV"], 1e-8, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.amp_conv = keyword(["AMPCONV"], 1e-7, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.print_n_amplitudes = keyword(["PRINTAMPS"], 10, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.CC_max_iter = keyword(["CCMAXITER"], 30, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.cc_damp_requested, self.coupled_cluster_damping_parameter = keyword(["CCDAMP"], False, check_next_space=True, value_type=float, associated_keyword_default=0.25)
        self.coupled_cluster_damping_parameter = self.coupled_cluster_damping_parameter if self.cc_damp_requested else 0
        self.freeze_core, self.freeze_n_orbitals = keyword(["FREEZECORE"], False, boolean=True, check_next_space=True, value_type=int, mandatory_value=False, associated_keyword_default=None)
        self.n_MP2_grid_points = keyword(["MPGRID"], 20, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)

        # DFT keywords
        self.functional = DFT_methods.get(self.method)
        self.X_alpha = keyword(["XA"], 2 / 3, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.integral_accuracy_requested, self.integral_accuracy = keyword(["INTACC"], False, check_next_space=True, value_type=float, associated_keyword_default=4.0)
        self.DFX_requested, self.DFX_prop = keyword(["DFX"], False, check_next_space=True, mandatory_value=True, boolean=True, value_type=float, associated_keyword_default=100)
        self.DFC_requested, self.DFC_prop = keyword(["DFC"], False, check_next_space=True, mandatory_value=True, boolean=True, value_type=float, associated_keyword_default=100)
        self.MPC_requested, self.MPC_prop = keyword(["MPC"], False, check_next_space=True, mandatory_value=True, boolean=True, value_type=float, associated_keyword_default=0)

        # Checks for Hartree theory requested
        if self.method in ["H", "UH"]:
            
            self.HFX_requested, self.HFX_prop = (False, 0) 

        else: 
             
            self.HFX_requested, self.HFX_prop = keyword(["HFX"], False, check_next_space=True, mandatory_value=True, boolean=True, value_type=float, associated_keyword_default=100)
        
        if self.DFT_calculation: 
            
            # Only overwrites HFX, etc., if a DFT calculation is requested
            if not self.HFX_requested: self.HFX_prop = self.functional.HFX
            if not self.DFX_requested: self.DFX_prop = self.functional.DFX
            if not self.DFC_requested: self.DFC_prop = self.functional.DFC
            if not self.MPC_requested: self.MPC_prop = self.functional.MPC
        
        # Brings the percentage into a proportion
        self.HFX_prop, self.DFX_prop, self.DFC_prop, self.MPC_prop = self.HFX_prop / 100, self.DFX_prop / 100, self.DFC_prop / 100, self.MPC_prop / 100
        
        self.DFX_prop = 0 if self.no_DFT_exchange else self.DFX_prop
        self.DFC_prop = 0 if self.no_DFT_correlation else self.DFC_prop

        # Excited state keywords
        self.root = keyword(["ROOT"], 1, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)
        self.CIS_contribution_threshold = keyword(["CISTHRESH"], 1, boolean=False, check_next_space=True, value_type=float, mandatory_value=True)
        self.n_states = keyword(["NSTATES"], 10, boolean=False, check_next_space=True, value_type=int, mandatory_value=True)

        # External printing keywords
        self.trajectory, self.trajectory_path = keyword(["TRAJ"], False, boolean=True, check_next_space=True, mandatory_value=False, associated_keyword_default="tuna-trajectory.xyz", value_type=str)
        self.save_plot, self.save_plot_filepath = keyword(["SAVEPLOT"], False, check_next_space=True, associated_keyword_default="TUNA Plot.png", value_type=str, plot_path=True)
        self.scan_plot_colour = next((code for name, code in color_map.items() if name in params), "b")
        self.custom_basis_file = keyword(["BASIS"], None, boolean=False, check_next_space=True, mandatory_value=True, associated_keyword_default="tuna-trajectory.xyz", value_type=str)
        self.plot_molecular_orbital, self.molecular_orbital_to_plot = keyword(["PLOTMO"], False, check_next_space=True, value_type=int, associated_keyword_default=1)
        self.plot_natural_orbital, self.natural_orbital_to_plot = keyword(["PLOTNO"], False, check_next_space=True, value_type=int, associated_keyword_default=1)
        
        # Convergence keywords for SCF and optimisations
        self.SCF_conv_requested = True if "LOOSE" in params or "LOOSESCF" in params or "MEDIUM" in params or "MEDIUMSCF" in params or "TIGHT" in params or "TIGHTSCF" in params or "EXTREME" in params or "EXTREMESCF" in params else False
        self.geom_conv_requested = True if "LOOSEOPT" in params or "MEDIUMOPT" in params or "TIGHTOPT" in params or "EXTREMEOPT" in params else False
        self.grid_conv_requested = True if "LOOSEGRID" in params or "MEDIUMGRID" in params or "TIGHTGRID" in params or "EXTREMEGRID" in params else False

        self.SCF_conv = constants.convergence_criteria_SCF["loose"] if "LOOSE" in params or "LOOSESCF" in params else constants.convergence_criteria_SCF["medium"]
        self.SCF_conv = constants.convergence_criteria_SCF["medium"] if "MEDIUM" in params or "MEDIUMSCF"in params else self.SCF_conv
        self.SCF_conv = constants.convergence_criteria_SCF["tight"] if "TIGHT" in params or "TIGHTSCF" in params else self.SCF_conv
        self.SCF_conv = constants.convergence_criteria_SCF["extreme"] if "EXTREME" in params or "EXTREMESCF" in params else self.SCF_conv
        
        if not self.SCF_conv_requested:
        
            self.SCF_conv = constants.convergence_criteria_SCF["tight"] if self.calculation_type in ["OPT", "FREQ", "OPTFREQ", "MD"] and self.SCF_conv != constants.convergence_criteria_SCF["extreme"] else self.SCF_conv
            self.SCF_conv = constants.convergence_criteria_SCF["extreme"] if self.calculation_type in ["FREQ", "OPTFREQ"] else self.SCF_conv
            self.SCF_conv = constants.convergence_criteria_SCF["tight"] if "CIS" in self.method and self.SCF_conv != constants.convergence_criteria_SCF["extreme"] else self.SCF_conv

        self.geom_conv = constants.convergence_criteria_optimisation["loose"] if "LOOSEOPT" in params else constants.convergence_criteria_optimisation["medium"]
        self.geom_conv = constants.convergence_criteria_optimisation["medium"] if "MEDIUMOPT" in params else self.geom_conv
        self.geom_conv = constants.convergence_criteria_optimisation["tight"] if "TIGHTOPT" in params else self.geom_conv
        self.geom_conv = constants.convergence_criteria_optimisation["extreme"] if "EXTREMEOPT" in params else self.geom_conv
        
        if not self.geom_conv_requested:

            self.geom_conv = constants.convergence_criteria_optimisation["tight"] if self.calculation_type == "OPTFREQ" and self.geom_conv != constants.convergence_criteria_optimisation["extreme"] else self.geom_conv
        
        # Tightness criteria for DFT grid
        self.grid_conv = constants.convergence_criteria_grid["loose"] if "LOOSEGRID" in params else constants.convergence_criteria_grid["medium"]
        self.grid_conv = constants.convergence_criteria_grid["medium"] if "MEDIUMGRID" in params else self.grid_conv
        self.grid_conv = constants.convergence_criteria_grid["tight"] if "TIGHTGRID" in params else self.grid_conv
        self.grid_conv = constants.convergence_criteria_grid["extreme"] if "EXTREMEGRID" in params else self.grid_conv
        

        # Processes the NOSINGLES keyword
        self.method = process_no_singles_keyword(self.method, self.no_singles)
        self.plot_something = self.plot_density or self.plot_spin_density or self.plot_HOMO or self.plot_LUMO or self.plot_difference_density or self.plot_difference_spin_density or self.plot_molecular_orbital or self.plot_natural_orbital







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
        self.bohr_in_metres = 5.291772105443e-11
        
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


        # Convergence criteria for self-consistent field
        self.convergence_criteria_SCF = {

            "loose" : {"delta_E": 0.000001, "max_DP": 0.00001, "RMS_DP": 0.000001, "commutator": 0.0001, "name": "loose"},
            "medium" : {"delta_E": 0.0000001, "max_DP": 0.000001, "RMS_DP": 0.0000001, "commutator": 0.00001, "name": "medium"},
            "tight" : {"delta_E": 0.000000001, "max_DP": 0.00000001, "RMS_DP": 0.000000001, "commutator": 0.0000001, "name": "tight"},
            "extreme" : {"delta_E": 0.00000000001, "max_DP": 0.0000000001, "RMS_DP": 0.00000000001, "commutator": 0.000000001, "name": "extreme"}   
            
        }

        # Convergence criteria for geometry optimisation
        self.convergence_criteria_optimisation = {

            "loose" : {"gradient": 0.001, "step": 0.01, "name": "loose"},
            "medium" : {"gradient": 0.0001, "step": 0.0001, "name": "medium"},
            "tight" : {"gradient": 0.000001, "step": 0.00001, "name": "tight"},
            "extreme" : {"gradient": 0.00000001, "step": 0.0000001, "name": "extreme"}   

        }

        # Tightness criteria for the DFT grid
        self.convergence_criteria_grid = {

            "loose" : {"integral_accuracy": 3, "extent_multiplier": 0.7, "name": "loose"},
            "medium" : {"integral_accuracy": 4, "extent_multiplier": 0.9, "name": "medium"},
            "tight" : {"integral_accuracy": 5, "extent_multiplier": 1, "name": "tight"},
            "extreme" : {"integral_accuracy": 7, "extent_multiplier": 1.2, "name": "extreme"},


        }




constants = Constants()





class Output:

    """

    Stores all the useful outputs of a converged SCF calculation.

    """

    def __init__(self, energy, S, P, P_alpha, P_beta, molecular_orbitals, molecular_orbitals_alpha, molecular_orbitals_beta, epsilons, epsilons_alpha, epsilons_beta, kinetic_energy, nuclear_electron_energy, coulomb_energy, exchange_energy, correlation_energy, T=None, V_NE=None, J=None, K=None, F_alpha=None, F_beta=None, density=None, alpha_density=None, beta_density=None):
       
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
            F (array, optional): Fock matrix in AO basis
            T (array, optional): Kinetic matrix in AO basis
            V_NE (array, optional): Nuclear-electron matrix in AO basis
            J (array, optional): Coulomb matrix in AO basis

        """

        # Key quantities
        self.energy = energy
        self.S = S
        self.F = F_alpha + F_beta
        self.F_alpha = F_alpha
        self.F_beta = F_beta

        # Density matrices
        self.P = P
        self.P_alpha = P_alpha
        self.P_beta = P_beta
        self.density = density
        self.alpha_density = alpha_density
        self.beta_density = beta_density

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
        self.correlation_energy = correlation_energy
        self.exchange_correlation_energy = self.exchange_energy + self.correlation_energy

        # Optional matrices
        self.T = T
        self.V_NE = V_NE
        self.J = J
        self.K = K








class Functional:

    """

    Defines a DFT exchange-correlation functional.

    """

    def __init__(self, x_functional, c_functional, DFX=100, HFX=0, DFC=100, MPC=0, SSS=1, OSS=1, functional_class="LDA"):
        
        """

        Initialises Functional object.

        Args:   
            x_functional (function): Exchange functional    
            c_functional (function): Correlation functional    
            DFX (float, optional): Percentage of DFT exchange    
            HFX (float, optional): Percentage of Hartree-Fock exchange    
            DFC (float, optional): Percentage of DFC correlation    
            MPC (float, optional): Percentage of Moller-Plesset correlation 
            SSS (float, optional): Same spin scaling for MP2 correlation   
            OSS (float, optional): Opposite spin scaling for MP2 correlation   
            functional_class (str, optional): Either LDA, GGA or meta-GGA

        """
     
        # Exchange and correlation functional
        self.x_functional = x_functional
        self.c_functional = c_functional

        # Proportions of exchange and correlation
        self.DFX = DFX
        self.HFX = HFX
        self.DFC = DFC
        self.MPC = MPC

        # Same and opposite spin scaling
        self.SSS = SSS
        self.OSS = OSS

        # Either LDA, GGA or meta-GGA which determines how the exchange-correlation matrix in SCF is constructed
        self.functional_class = functional_class

        self.functional_type = self.process_functional()



    def process_functional(self):

        # Determines which "type" of functional this is

        self.functional_type = "pure"

        if self.HFX != 0: 
            
            self.functional_type = "hybrid"

            if self.MPC != 0: 

                self.functional_type = "double-hybrid"

                if self.SSS != 1 and self.OSS != 1:

                    self.functional_type = "spin-scaled double-hybrid"


        return self.functional_type








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
    
    if calculation.additional_print:

        log(f"\n Time taken for molecular integrals:      {calculation.integrals_time - calculation.start_time:8.2f} seconds", calculation, 3)
        log(f" Time taken for SCF iterations:           {calculation.SCF_time - calculation.integrals_time:8.2f} seconds", calculation, 3)

        if calculation.method in correlated_methods: 
            
            log(f" Time taken for correlated calculation:   {calculation.correlation_time - calculation.SCF_time:8.2f} seconds", calculation, 3)
        
        if calculation.method in excited_state_methods:
        
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

    # Prints the finale message
    else:
        
        log(colored(f"\n{calculation_types.get(calculation.calculation_type)} calculation in TUNA completed successfully in {total_time:.2f} seconds.  :)\n","white"), calculation, 1)
    

    # Exits the program
    sys.exit()






def symmetrise(M):

    """
    
    Symmetrises a square matrix.

    Args:
        M (array): Square matrix
    
    Returns:
        M_symmetrised (array): Symmetrised square matrix

    """

    M_symmetrised = (1 / 2) * (M + M.T)

    return M_symmetrised








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








calculation_types = {

    "SPE": "Single point energy",
    "OPT": "Geometry optimisation",
    "FREQ": "Harmonic frequency",
    "OPTFREQ": "Optimisation and harmonic frequency",
    "SCAN": "Coordinate scan",
    "MD": "Ab initio molecular dynamics",
    "FORCE": "Force"
    
    }




method_types = {
    
    "H": "Hartree theory", 
    "UH": "unrestricted Hartree theory", 
    "HF": "Hartree-Fock theory", 
    "RHF": "restricted Hartree-Fock theory", 
    "UHF": "unrestricted Hartree-Fock theory", 
    "IMP2": "iterative MP2 theory", 
    "LMP2": "Laplace transform MP2 theory", 
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
    "QCISD": "quadratic configuration interaction singles and doubles",
    "UQCISD": "unrestricted quadratic configuration interaction singles and doubles",
    "CCSD": "coupled cluster singles and doubles",
    "UCCSD": "unrestricted coupled cluster singles and doubles",
    "QCISD[T]": "quadratic configuration interaction singles, doubles and perturbative triples",
    "UQCISD[T]": "unrestricted quadratic configuration interaction singles, doubles and perturbative triples",
    "CCSD[T]": "coupled cluster singles, doubles and perturbative triples",
    "UCCSD[T]": "unrestricted coupled cluster singles, doubles and perturbative triples",
    "CCSDT": "coupled cluster singles, doubles and triples",
    "UCCSDT": "unrestricted coupled cluster singles, doubles and triples",
    "LDA": "density functional theory via local density approximation",
    "ULDA": "unrestricted density functional theory via local density approximation",
    "LSDA": "density functional theory via local spin density approximation",
    "ULSDA": "unrestricted density functional theory via local spin density approximation",
    "SVWN": "density functional theory with Slater exchange and VWN correlation",
    "USVWN": "unrestricted density functional theory with Slater exchange and VWN correlation",
    "SVWN3": "density functional theory with Slater exchange and VWN-III correlation",
    "USVWN3": "unrestricted density functional theory with Slater exchange and VWN-III correlation",
    "SVWN5": "density functional theory with Slater exchange and VWN-V correlation",
    "USVWN5": "unrestricted density functional theory with Slater exchange and VWN-V correlation",
    "BVWN": "density functional theory with Becke exchange and VWN correlation",
    "UBVWN": "unrestricted density functional theory with Becke exchange and VWN correlation",
    "BVWN3": "density functional theory with Becke exchange and VWN-III correlation",
    "UBVWN3": "unrestricted density functional theory with Becke exchange and VWN-III correlation",
    "BVWN5": "density functional theory with Becke exchange and VWN-V correlation",
    "UBVWN5": "unrestricted density functional theory with Becke exchange and VWN-V correlation",
    "SPW": "density functional theory with Slater exchange and Perdew-Wang correlation",
    "USPW": "unrestricted density functional theory with Slater exchange and Perdew-Wang correlation",
    "HFS": "Hartree-Fock theory with Slater exchange", 
    "RHFS": "restricted Hartree-Fock theory with Slater exchange", 
    "UHFS": "unrestricted Hartree-Fock theory with Slater exchange", 
    "HFB": "Hartree-Fock theory with Becke exchange", 
    "RHFB": "restricted Hartree-Fock theory with Becke exchange", 
    "UHFB": "unrestricted Hartree-Fock theory with Becke exchange", 
    "PBE": "density functional theory with PBE exchange and correlation",
    "UPBE": "unrestricted density functional theory with PBE exchange and correlation",
    "PBE0": "hybrid density functional theory with PBE exchange and correlation",
    "UPBE0": "unrestricted hybrid density functional theory with PBE exchange and correlation",
    "PBE0-DH": "double-hybrid density functional theory with PBE exchange and correlation",
    "UPBE0-DH": "unrestricted double-hybrid density functional theory with PBE exchange and correlation",
    "PBE-QIDH": "double-hybrid density functional theory with PBE exchange and correlation",
    "UPBE-QIDH": "unrestricted double-hybrid density functional theory with PBE exchange and correlation",
    "PBE0-2": "double-hybrid density functional theory with PBE exchange and correlation",
    "UPBE0-2": "unrestricted double-hybrid density functional theory with PBE exchange and correlation",
    "BLYP": "density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UBLYP": "unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "BP86": "density functional theory with Becke exchange and Perdew 1986 correlation",
    "UBP86": "unrestricted density functional theory with Becke exchange and Perdew 1986 correlation",
    "B1P86": "hybrid density functional theory with Becke exchange and Perdew 1986 correlation",
    "UB1P86": "hybrid unrestricted density functional theory with Becke exchange and Perdew 1986 correlation",
    "SLYP": "density functional theory with Slater exchange and Lee-Yang-Parr correlation",
    "USLYP": "unrestricted density functional theory with Slater exchange and Lee-Yang-Parr correlation",
    "BHLYP": "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UBHLYP": "hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B1LYP": "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB1LYP": "hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B3LYP": "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB3LYP": "hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B3LYP/G": "hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB3LYP/G": "hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "DSD-BLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UDSD-BLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2-PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2-PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2K-PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2K-PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2T-PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2T-PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2G-PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2G-PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "B2NC-PLYP": "double-hybrid density functional theory with Becke exchange and Lee-Yang-Parr correlation",
    "UB2NC-PLYP": "double-hybrid unrestricted density functional theory with Becke exchange and Lee-Yang-Parr correlation",   
    "TPSS": "density functional theory with TPSS exchange and correlation",
    "UTPSS": "unrestricted density functional theory with TPSS exchange and correlation",
    "TPSSH": "hybrid density functional theory with TPSS exchange and correlation",
    "UTPSSH": "unrestricted density functional theory with TPSS exchange and correlation",
    "TPSS0": "hybrid density functional theory with TPSS exchange and correlation",
    "UTPSS0": "hybrid unrestricted density functional theory with TPSS exchange and correlation",
    "PWP": "density functional theory with Perdew-Wang exchange and Perdew 1986 correlation",
    "UPWP": "unrestricted density functional theory with Perdew-Wang exchange and Perdew 1986 correlation",
    "MPWLYP": "density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "UMPWLYP": "unrestricted density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "MPW1LYP": "hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "UMPW1LYP": "unrestricted hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "MPW2PLYP": "double-hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "UMPW2PLYP": "unrestricted double-hybrid density functional theory with modified Perdew-Wang exchange and Lee-Yang-Parr correlation",
    "MPWPW": "density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation",
    "UMPWPW": "unrestricted density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation",
    "PW1PW": "hybrid density functional theory with Perdew-Wang exchange and Perdew-Wang correlation",
    "UPW1PW": "hybrid unrestricted density functional theory with Perdew-Wang exchange and Perdew-Wang correlation",
    "MPW1PW": "hybrid density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation",
    "UMPW1PW": "hybrid unrestricted density functional theory with modified Perdew-Wang exchange and Perdew-Wang correlation",
    "B3PW91": "hybrid density functional theory with Becke exchange and Perdew-Wang correlation",
    "UB3PW91": "hybrid unrestricted density functional theory with Becke exchange and Perdew-Wang correlation",
    "B3P86": "hybrid density functional theory with Becke exchange and Perdew 1986 correlation",
    "UB3P86": "hybrid unrestricted density functional theory with Becke exchange and Perdew 1986 correlation",

    }



correlated_methods = [
    
    "MP2", "UMP2", "SCS-MP2", "USCS-MP2", "MP3", "UMP3", "SCS-MP3", "USCS-MP3", "MP4", "MP4[SDTQ]", "MP4[SDQ]", "MP4[DQ]", "OMP2", "UOMP2", "OOMP2", "UOOMP2", "IMP2", "LMP2",
    "CCD", "UCCD", "CEPA0", "UCEPA0", "LCCD", "ULCCD",  "LCCSD", "ULCCSD", "CEPA", "UCEPA", "CEPA[0]", "UCEPA[0]", "QCISD", "UQCISD", "CCSD", "UCCSD", "QCISD[T]", "UQCISD[T]", "CCSD[T]", "UCCSD[T]", "CCSDT", "UCCSDT"

    ]





excited_state_methods = [
    
    "CIS", "UCIS", "UCIS[D]", "CIS[D]"

    ]





DFT_methods = {

    "HFS" : Functional("S", None, DFX=100, HFX=0, DFC=0, MPC=0, functional_class="LDA"),
    "UHFS" : Functional("S", None, DFX=100, HFX=0, DFC=0, MPC=0, functional_class="LDA"),
    "SVWN" : Functional("S", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "USVWN" : Functional("S", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "LSDA" : Functional("S", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "ULSDA" : Functional("S", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "LDA" : Functional("S", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "ULDA" : Functional("S", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "SVWN3" : Functional("S", "VWN3", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "USVWN3" : Functional("S", "UVWN3", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "SVWN5" : Functional("S", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "USVWN5" : Functional("S", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "SPW" : Functional("S", "PW", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "USPW" : Functional("S", "UPW", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="LDA"),
    "PBE" : Functional("PBE", "PBE", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UPBE" : Functional("PBE", "UPBE", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "PBE0" : Functional("PBE", "PBE", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UPBE0" : Functional("PBE", "UPBE", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "PBE0-DH" : Functional("PBE", "PBE", DFX=50, HFX=50, DFC=87.5, MPC=12.5, functional_class="GGA"),
    "UPBE0-DH" : Functional("PBE", "UPBE", DFX=50, HFX=50, DFC=87.5, MPC=12.5, functional_class="GGA"),
    "PBE-QIDH" : Functional("PBE", "PBE", DFX=31, HFX=69, DFC=67, MPC=33, functional_class="GGA"),
    "UPBE-QIDH" : Functional("PBE", "UPBE", DFX=31, HFX=69, DFC=67, MPC=33, functional_class="GGA"),
    "PBE0-2" : Functional("PBE", "PBE", DFX=100-100/np.cbrt(2), HFX=100/np.cbrt(2), DFC=50, MPC=50, functional_class="GGA"),
    "UPBE0-2" : Functional("PBE", "UPBE", DFX=100-100/np.cbrt(2), HFX=100/np.cbrt(2), DFC=50, MPC=50, functional_class="GGA"),
    "HFB" : Functional("B", None, DFX=100, HFX=0, DFC=0, MPC=0, functional_class="GGA"),
    "UHFB" : Functional("B", None, DFX=100, HFX=0, DFC=0, MPC=0, functional_class="GGA"),
    "BVWN" : Functional("B", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UBVWN" : Functional("B", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "BVWN3" : Functional("B", "VWN3", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UBVWN3" : Functional("B", "UVWN3", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "BVWN5" : Functional("B", "VWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UBVWN5" : Functional("B", "UVWN5", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "BLYP" : Functional("B", "LYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UBLYP" : Functional("B", "ULYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "BHLYP" : Functional("B", "LYP", DFX=50, HFX=50, DFC=100, MPC=0, functional_class="GGA"),
    "UBHLYP" : Functional("B", "ULYP", DFX=50, HFX=50, DFC=100, MPC=0, functional_class="GGA"),
    "B1LYP" : Functional("B", "LYP", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UB1LYP" : Functional("B", "ULYP", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "PWP" : Functional("PW", "P86", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UPWP" : Functional("PW", "UP86", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "SLYP" : Functional("S", "LYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "USLYP" : Functional("S", "ULYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "B3LYP" : Functional("B3", "3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "UB3LYP" : Functional("B3", "U3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "B3LYP/G" : Functional("B3", "3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "UB3LYP/G" : Functional("B3", "U3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "B2PLYP" : Functional("B", "LYP", DFX=47, HFX=53, DFC=73, MPC=27, functional_class="GGA"),
    "UB2PLYP" : Functional("B", "ULYP", DFX=47, HFX=53, DFC=73, MPC=27, functional_class="GGA"),
    "B2-PLYP" : Functional("B", "LYP", DFX=47, HFX=53, DFC=73, MPC=27, functional_class="GGA"),
    "UB2-PLYP" : Functional("B", "ULYP", DFX=47, HFX=53, DFC=73, MPC=27, functional_class="GGA"),
    "B2K-PLYP" : Functional("B", "LYP", DFX=28, HFX=72, DFC=58, MPC=42, functional_class="GGA"),
    "UB2K-PLYP" : Functional("B", "ULYP", DFX=28, HFX=72, DFC=58, MPC=42, functional_class="GGA"),
    "B2T-PLYP" : Functional("B", "LYP", DFX=40, HFX=60, DFC=69, MPC=31, functional_class="GGA"),
    "UB2T-PLYP" : Functional("B", "ULYP", DFX=40, HFX=60, DFC=69, MPC=31, functional_class="GGA"),
    "B2G-PLYP" : Functional("B", "LYP", DFX=35, HFX=65, DFC=64, MPC=36, functional_class="GGA"),
    "UB2G-PLYP" : Functional("B", "ULYP", DFX=35, HFX=65, DFC=64, MPC=36, functional_class="GGA"),
    "B2NC-PLYP" : Functional("B", "LYP", DFX=19, HFX=81, DFC=45, MPC=55, functional_class="GGA"),
    "UB2NC-PLYP" : Functional("B", "ULYP", DFX=19, HFX=81, DFC=45, MPC=55, functional_class="GGA"),
    "DSD-BLYP" : Functional("B", "LYP", DFX=25, HFX=75, DFC=53, MPC=100, SSS=0.60, OSS=0.46, functional_class="GGA"),
    "UDSD-BLYP" : Functional("B", "ULYP", DFX=25, HFX=75, DFC=53, MPC=100, SSS=0.60, OSS=0.46, functional_class="GGA"),
    "BP86" : Functional("B", "P86", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UBP86" : Functional("B", "UP86", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "B1P86" : Functional("B", "P86", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UB1P86" : Functional("B", "UP86", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "TPSS" : Functional("TPSS", "TPSS", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="meta-GGA"),
    "UTPSS" : Functional("TPSS", "UTPSS", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="meta-GGA"),
    "TPSSH" : Functional("TPSS", "TPSS", DFX=90, HFX=10, DFC=100, MPC=0, functional_class="meta-GGA"),
    "UTPSSH" : Functional("TPSS", "UTPSS", DFX=90, HFX=10, DFC=100, MPC=0, functional_class="meta-GGA"),
    "TPSS0" : Functional("TPSS", "TPSS", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="meta-GGA"),
    "UTPSS0" : Functional("TPSS", "UTPSS", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="meta-GGA"),
    "MPWLYP" : Functional("MPW", "LYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UMPWLYP" : Functional("MPW", "ULYP", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "MPW1LYP" : Functional("MPW", "LYP", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UMPW1LYP" : Functional("MPW", "ULYP", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "MPW2PLYP" : Functional("MPW", "LYP", DFX=45, HFX=55, DFC=75, MPC=25, functional_class="GGA"),
    "UMPW2PLYP" : Functional("MPW", "ULYP", DFX=45, HFX=55, DFC=75, MPC=25, functional_class="GGA"),
    "MPWPW" : Functional("MPW", "PW91", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "UMPWPW" : Functional("MPW", "UPW91", DFX=100, HFX=0, DFC=100, MPC=0, functional_class="GGA"),
    "PW1PW" : Functional("PW", "PW91", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UPW1PW" : Functional("PW", "UPW91", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "MPW1PW" : Functional("MPW", "PW91", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "UMPW1PW" : Functional("MPW", "UPW91", DFX=75, HFX=25, DFC=100, MPC=0, functional_class="GGA"),
    "B3PW91" : Functional("B3", "3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "UB3PW91" : Functional("B3", "U3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "B3P86" : Functional("B3", "3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
    "UB3P86" : Functional("B3", "U3P", DFX=80, HFX=20, DFC=100, MPC=0, functional_class="GGA"),
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
        "real_vdw_radius" : 0,
        "core_orbitals": 0,
        "name" : "ghost"
    },

    "H" : {
        "charge" : 1,
        "mass" : 1.007825,
        "C6" : 2.4283,
        "vdw_radius" : 1.8916,
        "real_vdw_radius" : 120,
        "core_orbitals": 0,
        "name" : "hydrogen"
    },

    "HE" : {
        "charge" : 2,
        "mass" : 4.002603,
        "C6" : 1.3876,
        "vdw_radius" : 1.9124,
        "real_vdw_radius" : 140,
        "core_orbitals": 0,
        "name" : "helium"
    },

    "LI" : {
        "charge" : 3,
        "mass" : 7.016004,
        "C6" : 27.92545,
        "vdw_radius" : 1.55902,
        "real_vdw_radius" : 182,
        "core_orbitals": 0,
        "name" : "lithium"
    },

    "BE" : {
        "charge" : 4,
        "mass" : 9.012182,
        "C6" : 27.92545,
        "vdw_radius" : 2.66073,
        "real_vdw_radius" : 153,
        "core_orbitals": 0,
        "name" : "beryllium"
    },

    "B" : {
        "charge" : 5,
        "mass" : 11.009305,
        "C6" : 54.28985,
        "vdw_radius" : 2.80624,
        "real_vdw_radius" : 192,
        "core_orbitals": 1,
        "name" : "boron"
    },

    "C" : {
        "charge" : 6,
        "mass" : 12.000000,
        "C6" : 30.35375,
        "vdw_radius" : 2.74388,
        "real_vdw_radius" : 170,
        "core_orbitals": 1,
        "name" : "carbon"
    },

    "N" : {
        "charge" : 7,
        "mass" : 14.003074,
        "C6" : 21.33435,
        "vdw_radius" : 2.63995,
        "real_vdw_radius" : 155,
        "core_orbitals": 1,
        "name" : "nitrogen"
    },

    "O" : {
        "charge" : 8,
        "mass" : 15.994915,
        "C6" : 12.1415,
        "vdw_radius" : 2.53601,
        "real_vdw_radius" : 152,
        "core_orbitals": 1,
        "name" : "oxygen"
    },

    "F" : {
        "charge" : 9,
        "mass" : 18.998403,
        "C6" : 13.00875,
        "vdw_radius" : 2.43208,
        "real_vdw_radius" : 147,
        "core_orbitals": 1,
        "name" : "fluorine"
    },

    "NE" : {
        "charge" : 10,
        "mass" : 19.992440,
        "C6" : 10.92735,
        "vdw_radius" : 2.34893,
        "real_vdw_radius" : 154,
        "core_orbitals": 1,
        "name" : "neon"
    },

    "NA" : {
        "charge" : 11,
        "mass" : 22.989770,
        "C6" : 99.03995,
        "vdw_radius" : 2.16185,
        "real_vdw_radius" : 227,
        "core_orbitals": 1,
        "name" : "sodium"
    },

    "MG" : {
        "charge" : 12,
        "mass" : 23.985042,
        "C6" : 99.03995,
        "vdw_radius" : 2.57759,
        "real_vdw_radius" : 173,
        "core_orbitals": 1,
        "name" : "magnesium"
    },

    "AL" : {
        "charge" : 13,
        "mass" : 26.981538,
        "C6" : 187.15255,
        "vdw_radius" : 3.09726,
        "real_vdw_radius" : 184,
        "core_orbitals": 5,
        "name" : "aluminium"
    },

    "SI" : {
        "charge" : 14,
        "mass" : 27.976927,
        "C6" : 160.09435,
        "vdw_radius" : 3.24277,
        "real_vdw_radius" : 210,
        "core_orbitals": 5,
        "name" : "silicon"
    },

    "P" : {
        "charge" : 15,
        "mass" : 30.973762,
        "C6" : 135.9848,
        "vdw_radius" : 3.22198,
        "real_vdw_radius" : 180,
        "core_orbitals": 5,        
        "name" : "phosphorus"
    },

    "S" : {
        "charge" : 16,
        "mass" : 31.972071,
        "C6" : 96.61165,
        "vdw_radius" : 3.18041,
        "real_vdw_radius" : 180,
        "core_orbitals": 5,
        "name" : "sulfur"
    },

    "CL" : {
        "charge" : 17,
        "mass" : 34.968853,
        "C6" : 87.93915,
        "vdw_radius" : 3.09726,
        "real_vdw_radius" : 175,
        "core_orbitals": 5,
        "name" : "chlorine"
    },

    "AR" : {
        "charge" : 18,
        "mass" : 39.962383,
        "C6" : 79.96045,
        "vdw_radius" : 3.01411,
        "real_vdw_radius" : 188,
        "core_orbitals": 5,
        "name" : "argon"
    }

}

