from __future__ import annotations
from tuna_util import error, DFT_methods, constants, Method
from dataclasses import dataclass
import numpy as np


"""

This is the TUNA module for keyword and calculation management, written first for version 0.10.1.

After the Calculation object is initialised, the simple input line is scanned through to process the keywords requested and set the others to their default
values. Afterwards, various post-processing which is done once and stored in the calculation object occurs.

This module contains:

1. A class to define a generic keyword
2. A list of keyword objects, with their variable types and defaults
3. A function to process the keywords and set attributes of the Calculation object (interpret_keywords)
4. The main Calculation object class, which is initialised once at the start of a calculation

"""



@dataclass
class Keyword:
    
    """
    
    Defines a keyword, with the variable types it corresponds to and their defaults.
    
    """

    # Aliases for this keyword, e.g. "TEMP", "TEMPERATURE"

    aliases: tuple | str

    # Attribute which the Calculation class will initialise

    attribute: str

    # Either "B" for boolean, "V" for value or "B+V" for boolean and value

    keyword_type: str = "B"

    # Default value for main attribute

    default: object = False

    # Type of secondary value for "B+V"

    value_type: type | None = None

    # Default for value attribute

    value_default: object = None
    
    # If there's a second attribute, like "DAMP 0.2" records is_damping = True and damping_factor = 0.2

    value_attribute: str | None = None

    # Is this a filepath?

    file_path_keyword: bool = False

    # This turns a single alias into a tuple

    def __post_init__(self):

        if isinstance(self.aliases, str):
            
            self.aliases = (self.aliases,)










KEYWORDS = [

    # These are boolean keywords

    Keyword("P", "additional_print"),
    Keyword("T", "terse"),
    Keyword("DEBUG", "debug"),
    Keyword("DECONTRACT", "decontract"),
    Keyword("CARTHARM", "cartesian_harmonics"),
    Keyword("EXTRAPOLATE", "extrapolate"),

    Keyword("NOROTATE", "no_rotate_guess"),
    Keyword("COREGUESS", "core_guess"),
    Keyword("SADGUESS", "superposition_guess"),
    Keyword("SCFGUESS", "self_consistent_guess", default = True),
    Keyword("SLOWCONV", "slow_conv"),
    Keyword("VERYSLOWCONV", "very_slow_conv"),
    Keyword("NODIIS", "no_DIIS"),
    Keyword("NODAMP", "no_damping"),
    Keyword("MOREAD", "MO_read_requested"),
    Keyword("NOMOREAD", "no_MO_read"),

    Keyword("NATORBS", "natural_orbitals"),
    Keyword("D2", "D2"),
    Keyword("CALCHESS", "calc_hess"),
    Keyword("OPTMAX", "opt_max"),
    Keyword("NOTRAJ", "no_trajectory"),
    Keyword("NOX", "no_DFT_exchange"),
    Keyword("NOC", "no_DFT_correlation"),
    Keyword("NOSINGLES", "no_singles"),
    Keyword("LOOSEPNO", "loose_DLPNO_criteria"),
    Keyword(("NORMALPNO", "MEDIUMPNO"), "medium_DLPNO_criteria"),
    Keyword("TIGHTPNO", "tight_DLPNO_criteria"),
    Keyword("EXTREMEPNO", "extreme_DLPNO_criteria"),

    Keyword("SCANPLOT", "scan_plot"),
    Keyword("DASH", "plot_dashed_lines"),
    Keyword("DOT", "plot_dotted_lines"),
    Keyword("ADDPLOT", "add_plot"),
    Keyword("DELPLOT", "delete_plot"),
    Keyword("DENSPLOT", "plot_density"),
    Keyword("SPINDENSPLOT", "plot_spin_density"),
    Keyword("PLOTHOMO", "plot_HOMO"),
    Keyword("PLOTLUMO", "plot_LUMO"),
    Keyword("DIFFDENSPLOT", "plot_difference_density"),
    Keyword("DIFFSPINDENSPLOT", "plot_difference_spin_density"),
    Keyword(("VIBPLOT", "PLOTVIB"), "plot_vibrational_wavefunctions"),

    Keyword("DIPOLE", "dipole"),
    Keyword("QUADRUPOLE", "quadrupole"),
    Keyword(("POLAR", "POLARISABILITY", "POLARIZABILITY"), "polarisability"),
    Keyword(("HYPER", "HYPERPOLARISABILITY", "HYPERPOLARIZABILITY"), "hyperpolarisability"),
    Keyword("VERTICAL", "vertical"),
    Keyword("VPT2", "perturbative_anharmonic"),
    Keyword("NOCP", "no_counterpoise_correction"),
    Keyword("ZPE", "do_ZPE_correction"),


    # These keywords give an attrribute for the value after the keyword

    Keyword(("CH", "CHARGE"), "charge", "V", 0, int),
    Keyword(("ML", "MULTIPLICITY"), "multiplicity", "V", 1, int),
    Keyword("BASIS", "custom_basis_file", "V", None, str),

    Keyword("XA", "X_alpha", "V", 2 / 3, float),
    Keyword("STHRESH", "S_eigenvalue_threshold", "V", 1e-7, float),
    Keyword("MAXITER", "max_iter", "V", 100, int),
    Keyword("MAXDAMP", "max_damping", "V", 0.7, float),
    Keyword("EX", "electric_field_x", "V", 0, float),
    Keyword("EY", "electric_field_y", "V", 0, float),
    Keyword("EZ", "electric_field_z", "V", 0, float),
    Keyword("NELEC", "n_electrons_for_ip_or_ea", "V", 1, int),
    Keyword("ROOT", "root", "V", 1, int),
    Keyword("CISTHRESH", "CIS_contribution_threshold", "V", 1, float),
    Keyword("NSTATES", "n_states", "V", 10, int),

    Keyword(("GEOMMAXITER", "MAXGEOMITER"), "geom_max_iter", "V", 30, int),
    Keyword("MAXSTEP", "max_step", "V", 0.2, float),
    Keyword("DEFAULTHESS", "default_hessian", "V", 0.25, float),
    Keyword("M1", "custom_mass_1", "V", None, float),
    Keyword("M2", "custom_mass_2", "V", None, float),
    Keyword(("TEMP", "TEMPERATURE"), "temperature", "V", None, float),
    Keyword(("PRES", "PRESSURE"), "pressure", "V", 101325, float),
    Keyword("ANHARMCONV", "anharm_convergence", "V", 0.01, float),
    Keyword("STEP", "step", "V", None, float),
    Keyword("NUM", "number_of_steps", "V", None, int),

    Keyword(("MP3S", "MP3SCALING", "MP3SCAL"), "MP3_scaling", "V", 1 / 4, float),
    Keyword("AMPCONV", "amp_conv", "V", 1e-8, float),
    Keyword("PRINTAMPS", "print_n_amplitudes", "V", 10, int),
    Keyword("MPGRID", "n_MP2_grid_points", "V", 20, int),
    Keyword("ECONV", "correlated_energy_convergence", "V", 1e-9, float),
    Keyword("CORRMAXITER", "correlated_max_iter", "V", 100, int),
    Keyword("TCUTDO", "TCutDO", "V", 1e-2, float),
    Keyword("TCUTPNO", "TCutPNO", "V", 1e-8, float),
    Keyword("TSCALEPNOCORE", "TScalePNOCore", "V", 1e-2, float),
    Keyword("FCUT", "FCut", "V", 1e-5, float),
    Keyword("TCUTPRE", "TCutPre", "V", 1e-6, float),
    Keyword("PAOSTHRESH", "PAOOverlapThresh", "V", 1e-8, float),

    
    # These keywords give two attributes, one boolean for "is this keyword requested", another for the value given
    
    Keyword("ROTATE", "rotate_guess", "B+V", False, float, 45, "theta"),
    Keyword("DIIS", "DIIS", "B+V", True, int, 6, "max_DIIS_matrices"),
    Keyword("DAMP", "damping", "B+V", True, float, None, "damping_factor"),
    Keyword("FREEZECORE", "freeze_core", "B+V", False, int, None, "freeze_n_orbitals"),
    Keyword("CORRDAMP", "correlated_damping_requested", "B+V", False, float, 0, "correlated_damping_parameter"),
    
    Keyword("INTACC", "integral_accuracy_requested", "B+V", False, float, 4, "integral_accuracy"),
    Keyword("DFX", "DFX_requested", "B+V", False, float, 1, "DFX_prop"),
    Keyword("DFC", "DFC_requested", "B+V", False, float, 1, "DFC_prop"),
    Keyword("MPC", "MPC_requested", "B+V", False, float, 0, "MPC_prop"),
    Keyword("HFX", "HFX_requested", "B+V", False, float, 1, "HFX_prop"),
    Keyword("SSS", "SSS_requested", "B+V", False, float, 1 / 3, "same_spin_scaling"),
    Keyword("OSS", "OSS_requested", "B+V", False, float, 6 / 5, "opposite_spin_scaling"), 

    Keyword("TRAJ", "trajectory", "B+V", False, str, "tuna-trajectory.xyz", "trajectory_path"),
    Keyword("SAVEPLOT", "save_plot", "B+V", False, str, "tuna-plot.pdf", "save_plot_filepath", True),
    Keyword("PLOTMO", "plot_molecular_orbital", "B+V", False, int, 1, "molecular_orbital_to_plot"),
    Keyword("PLOTNO", "plot_natural_orbital", "B+V", False, int, 1, "natural_orbital_to_plot"),

]










colour_map = {

    "RED": "r",
    "GREEN": "g",
    "BLUE": "b",
    "CYAN": "c",
    "MAGENTA": "m",
    "YELLOW": "y",
    "BLACK": "k",
    "WHITE": "w",
}










def interpret_keywords(calculation: Calculation, params: list) -> None:

    """
    
    Iterates through the given parameters, checks if they are keywords and sets the attributes of a Calculation object.

    Args:
        calculation (Calculation): Calculation object
        params (list): List of parameters on simple input line
    
    """
    
    ALIASES = {alias: keyword for keyword in KEYWORDS for alias in keyword.aliases}

    PLOT_EXTENSIONS = (".png", ".jpg", ".pdf", ".svg", ".jpeg", ".tif", ".tiff", ".bmp", ".raw", ".eps", ".ps")

    # Boolean keywords

    for keyword in KEYWORDS:

        setattr(calculation, keyword.attribute, keyword.default)

        if keyword.keyword_type == "B+V":

            setattr(calculation, keyword.value_attribute, keyword.value_default)

    i = 0

    while i < len(params):

        param = params[i]
        keyword = ALIASES.get(param)

        if keyword is None:

            i += 1
            continue
        
        # Boolean keywords

        if keyword.keyword_type == "B":

            setattr(calculation, keyword.attribute, True)

            i += 1

            continue

        # Check if there is a value after the keyword

        has_value = i + 1 < len(params) and params[i + 1] not in ALIASES

        if not has_value:

            if keyword.keyword_type == "V":

                error(f"Parameter \"{param}\" requested but no value specified!")

            elif keyword.keyword_type == "B+V":
                
                # Sets the boolean part of the keyword to true regardless of value being given

                setattr(calculation, keyword.attribute, True)

            i += 1

            continue

        # Can now safely read the value given for this parameter

        raw_value = params[i + 1]

        try:

            value = keyword.value_type(raw_value)

        except ValueError:

            error(f"Parameter \"{param}\" must be of type {keyword.value_type.__name__}!")

        if keyword.file_path_keyword:

            if not str(value).lower().endswith(PLOT_EXTENSIONS):

                error(f"Unsupported plot file extension in \"{value}\"!")

        if keyword.keyword_type == "V":

            setattr(calculation, keyword.attribute, value)

        elif keyword.keyword_type == "B+V":

            setattr(calculation, keyword.attribute, True)
            setattr(calculation, keyword.value_attribute, value)

        i += 2

    return










def process_complex_keywords(self: Calculation) -> None:

    """
    
    Processes the keywords which are not trivially interpreted.

    Args:
        self (Calculation): Calculation object
    
    """

    # Some keywords can override others
    
    self.MO_read = not self.no_MO_read
    self.DIIS = False if self.no_DIIS else self.DIIS
    self.damping = False if self.no_damping else self.damping
    
    # Checks if multiplicity has been overridden

    self.default_multiplicity = not any(param in ("ML", "MULTIPLICITY") for param in self.params)

    # Defines the "(VERY)SLOWCONV" keywords

    self.damping_factor = 0.85 if self.very_slow_conv else 0.5 if self.slow_conv else self.damping_factor

    if self.temperature is None:

        # Temperature default depends on calculation type

        self.temperature = 0 if self.calculation_type == "MD" else 298.15
    
    # Updates the method considering CEPA and the treatment of single excitations

    if self.method.name.startswith("U"):

        self.method.name = "U" + ("LCCSD" if "CEPA" in self.method.name[1:] else self.method.name[1:])

    else:

        self.method.name = "LCCSD" if "CEPA" in self.method.name else self.method.name

    # Useful geometric factors defined here

    self.ghost_atom_present = any("X" in symbol for symbol in self.atomic_symbols)

    self.monatomic = len(self.atomic_symbols) == 1 or self.ghost_atom_present
    self.diatomic = not self.monatomic

    # A core guess should be activated for atomic calculations, a SCF guess is disabled for BDE calculations where atoms are involved in the same Calculation object

    self.core_guess = True if self.monatomic else self.core_guess

    self.self_consistent_guess = False if (self.core_guess or self.superposition_guess or self.calculation_type == "BDE") else self.self_consistent_guess

    # The full three-dimensional electric field vector

    self.electric_field = np.array([self.electric_field_x, self.electric_field_y, self.electric_field_z])
    self.electric_field_gradient = np.zeros(3)

    # Does anything need to be plotted at the end of the calculation
    
    self.scan_plot_colour = next((code for name, code in colour_map.items() if name in self.params), "b")

    self.plot_something = self.plot_density or self.plot_spin_density or self.plot_HOMO or self.plot_LUMO or self.plot_difference_density or self.plot_difference_spin_density or self.plot_molecular_orbital or self.plot_natural_orbital

    # Accounts for Hartree theory being requested

    self.HFX_requested, self.HFX_prop = (False, 0)  if self.method.name in ["H", "UH"] else (self.HFX_requested, self.HFX_prop)

    # Manages the "NUM" keyword

    self.number_of_steps = 30 if self.number_of_steps is None and self.calculation_type == "MD" else self.number_of_steps

    if self.DFT_calculation: 
        
        # Only overwrites HFX, DFX, etc. if a DFT calculation is requested

        self.HFX_prop = self.functional.HFX if not self.HFX_requested else self.HFX_prop
        self.DFX_prop = self.functional.DFX if not self.DFX_requested else self.DFX_prop
        self.DFC_prop = self.functional.DFC if not self.DFC_requested else self.DFC_prop
        self.MPC_prop = self.functional.MPC if not self.MPC_requested else self.MPC_prop

        self.same_spin_scaling = self.functional.same_spin_scaling if not self.SSS_requested else self.same_spin_scaling 
        self.opposite_spin_scaling = self.functional.opposite_spin_scaling if not self.OSS_requested else self.opposite_spin_scaling 

    # Processes the "NOX" and "NOC" keywords

    self.DFX_prop = 0 if self.no_DFT_exchange else self.DFX_prop
    self.DFC_prop = 0 if self.no_DFT_correlation else self.DFC_prop

    # Determines the level of numerical derivative to be calculated

    self.third_derivative_requested = self.perturbative_anharmonic or self.hyperpolarisability
    self.second_derivative_requested = self.calculation_type in ["FREQ", "OPTFREQ", "ANHARM"] or self.polarisability or self.do_ZPE_correction or self.third_derivative_requested
    self.first_derivative_requested = self.calculation_type in ["OPT", "IP", "EA", "BDE", "MD"] or self.dipole or self.quadrupole or self.second_derivative_requested

    # Convergence criteria for self-consistent field calculation

    self.SCF_conv = constants.convergence_criteria_SCF["medium"]

    self.SCF_conv = constants.convergence_criteria_SCF["tight"] if self.first_derivative_requested else self.SCF_conv
    self.SCF_conv = constants.convergence_criteria_SCF["extreme"] if self.second_derivative_requested else self.SCF_conv

    self.SCF_conv = constants.convergence_criteria_SCF["loose"] if "LOOSE" in self.params or "LOOSESCF" in self.params else self.SCF_conv
    self.SCF_conv = constants.convergence_criteria_SCF["medium"] if "MEDIUM" in self.params or "MEDIUMSCF"in self.params else self.SCF_conv
    self.SCF_conv = constants.convergence_criteria_SCF["tight"] if "TIGHT" in self.params or "TIGHTSCF" in self.params else self.SCF_conv
    self.SCF_conv = constants.convergence_criteria_SCF["extreme"] if "EXTREME" in self.params or "EXTREMESCF" in self.params else self.SCF_conv
    
    # Convergence criteria for geometry optimisation

    self.geom_conv = constants.convergence_criteria_optimisation["medium"] 

    self.geom_conv = constants.convergence_criteria_optimisation["tight"] if self.second_derivative_requested else self.geom_conv
    
    self.geom_conv = constants.convergence_criteria_optimisation["loose"] if "LOOSEOPT" in self.params else self.geom_conv 
    self.geom_conv = constants.convergence_criteria_optimisation["medium"] if "MEDIUMOPT" in self.params else self.geom_conv
    self.geom_conv = constants.convergence_criteria_optimisation["tight"] if "TIGHTOPT" in self.params else self.geom_conv
    self.geom_conv = constants.convergence_criteria_optimisation["extreme"] if "EXTREMEOPT" in self.params else self.geom_conv

    # Tightness criteria for DFT grid

    self.grid_conv = constants.convergence_criteria_grid["medium"]

    self.grid_conv = constants.convergence_criteria_grid["loose"] if "LOOSEGRID" in self.params else self.grid_conv
    self.grid_conv = constants.convergence_criteria_grid["medium"] if "MEDIUMGRID" in self.params else self.grid_conv
    self.grid_conv = constants.convergence_criteria_grid["tight"] if "TIGHTGRID" in self.params else self.grid_conv
    self.grid_conv = constants.convergence_criteria_grid["extreme"] if "EXTREMEGRID" in self.params else self.grid_conv

    # The default energy convergence for correlated calculations is the same as the SCF convergence by default

    self.correlated_energy_convergence = self.SCF_conv.get("delta_E") if "ECONV" not in self.params else self.correlated_energy_convergence


    return










@dataclass
class Calculation:

    """
    
    Processes and calculates from user-defined parameters specified at the start of a TUNA calculation.

    Various default values for parameters are specified here. This object is created once per TUNA calculation.
    
    """

    # Type of calculation (SPE, OPT, FREQ, ANHARM, etc.)

    calculation_type: str

    # Electronic structure method object (B3LYP, HF, CCSD, etc.)

    method: Method

    # Start time for calculation, counted from after modules are imported
    
    start_time: float

    # Keywords in simple input line

    params: list[str]

    # Basis set (cc-pVDZ, 6-31G, ano-pVQZ, etc.)

    basis: str

    # List of atomic symbols,  e.g. ["C", "O"]

    atomic_symbols: list[str]


    def __post_init__(self) -> None:

        # Stores the basis set, and initialises the reference and functional

        self.original_basis = self.basis
        self.reference = "Undefined"

        self.functional = DFT_methods.get(self.method.name)
        self.DFT_calculation = self.method.density_functional_method

        # Initialises the time for different parts of the calculation

        self.SCF_time = 0.0
        self.integrals_time = 0.0
        self.correlation_time = 0.0
        self.excited_state_time = 0.0
        
        # Interprets the keyword list and sets the calculation attributes

        interpret_keywords(self, self.params)

        # Processes the non-trivial keywords

        process_complex_keywords(self)