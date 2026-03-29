import numpy as np
import tuna_scf as scf
import sys, time
from tuna_util import *
from tuna_calc import Calculation
import tuna_props as props
from numpy import ndarray
from tuna_molecule import Molecule
import tuna_dft as dft
import tuna_guess as guess
import tuna_kernel as kern
import tuna_out as out



"""

This is the TUNA module for calculating molecular energies, written first for version 0.1.0 and rewritten in version 0.10.0.

Any mathematical functions that don't call calculate_energy should be in tuna_kernel -- this is for wrappers only. Energy evaluations may
use extrapolation of the basis set, and begin by building the molecule then calculating the molecular integrals. The self-consistent field
cycle is then entered within tuna_scf before correlated or excited state calculations are performed. Finally, properties are calculated
in tuna_prop or numerical derivatives of the energy are calculated and the energy evaluation process is repeated.

Updated in version 0.10.1 to include numerical quadrupole moment calculations.

This module contains:

1. Functions to manage basis set extrapolation (evaluate_molecular_energy, extrapolate_energy)
2. The self-consistent guess function (calculate_self_consistent_guess)
3. Functions that calculate properties requiring numerical energy derivatives (calculate_polarisability, etc.)
4. Fundamental functions for energy evaluation (build_molecule_and_integrals, calculate_energy)
5. The coordinate scan function, which calls evaluate_molecular_energy several times (scan_coordinate)

"""




def evaluate_molecular_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, P_guess: ndarray = None, P_guess_alpha: ndarray = None, 
                              P_guess_beta: ndarray = None, E_guess: float = None, do_correlation: bool = True, silent: bool = False, terse: bool = False, integrals: Integrals = None) -> tuple:
    
    """
  
    Wrapper to evaluate the energy, either with or without extrapolation.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): Atomic symbols
        coordinates (array): Atomic coordinates
        P_guess (array, optional): Guess density matrix
        P_guess_alpha (array, optional): Guess alpha density matrix
        P_guess_beta (array, optional): Guess beta density matrix
        E_guess (float, optional): Guess energy
        silent (bool, optional): Cancel logging
        terse (bool, optional): Cancel post-SCF output
        integrals (Integrals, optional): Molecular integrals
        
    Returns:
        SCF_output (Output): Output object
        molecule (Molecule): Molecule object
        final_energy (float): Final energy
        P (array): Final density matrix

    """

    # The choice of function depends if basis set extrapolation is used

    energy_function = extrapolate_energy if calculation.extrapolate else calculate_energy

    return energy_function(calculation, atomic_symbols, coordinates, P_guess=P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, do_correlation=do_correlation, silent=silent, terse=terse, integrals=integrals)










def extrapolate_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, P_guess: ndarray = None, P_guess_alpha: ndarray = None,
                       P_guess_beta: ndarray = None, E_guess: float = None, do_correlation: bool = True, silent: bool = False, terse: bool = False, integrals: Integrals = None):
 
    """
    
    Calculates the extrapolated energy, from two energy calculations.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Coordinates
        P_guess (array, optional): Guess density matrix
        P_guess_alpha (array, optional): Guess alpha density matrix
        P_guess_beta (array, optional): Guess beta density matrix
        E_guess (float, optional): Guess energy
        silent (bool, optional): Should anything be printed
        terse (bool, optional): Should properties be calculated
        integrals (Integrals, optional): Molecular integrals

    Returns:
        SCF_output_small (Output): Small basis SCF output
        molecule_small (Molecule): Small basis Molecule object
        E_extrapolated (float): Extrapolated total energy
        P_small (array): Small basis density matrix

    """

    double_zeta_bases = ["CC-PVDZ", "AUG-CC-PVDZ", "D-AUG-CC-PVDZ", "T-AUG-CC-PVDZ", "PC-1", "DEF2-SVP", "DEF2-SVPD", "ANO-PVDZ", "AUG-ANO-PVDZ"]
    quadruple_zeta_bases = ["CC-PVQZ", "AUG-CC-PVQZ", "D-AUG-CC-PVQZ", "T-AUG-CC-PVQZ", "PC-3", "ANO-PVQZ", "AUG-ANO-PVQZ"]

    basis_pairs = {

        "CC-PVDZ" : "CC-PVTZ",
        "CC-PVTZ" : "CC-PVQZ",
        "CC-PVQZ" : "CC-PV5Z",
        "AUG-CC-PVDZ" : "AUG-CC-PVTZ",
        "D-AUG-CC-PVDZ" : "D-AUG-CC-PVTZ",
        "T-AUG-CC-PVDZ" : "T-AUG-CC-PVTZ",
        "AUG-CC-PVTZ" : "AUG-CC-PVQZ",
        "D-AUG-CC-PVTZ" : "D-AUG-CC-PVQZ",
        "T-AUG-CC-PVTZ" : "T-AUG-CC-PVQZ",
        "AUG-CC-PVQZ" : "AUG-CC-PV5Z",
        "D-AUG-CC-PVQZ" : "D-AUG-CC-PV5Z",
        "T-AUG-CC-PVQZ" : "T-AUG-CC-PV5Z",
        "PC-1": "PC-2",
        "PC-2": "PC-3",
        "PC-3": "PC-4",
        "DEF2-SVP" : "DEF2-TZVPP",
        "DEF2-TZVP" : "DEF2-QZVP",
        "DEF2-TZVPP" : "DEF2-QZVPP",
        "DEF2-SVPD" : "DEF2-TZVPPD",
        "DEF2-TZVPD" : "DEF2-QZVPD",
        "DEF2-TZVPPD" : "DEF2-QZVPPD",
        "ANO-PVDZ" : "ANO-PVTZ",
        "ANO-PVTZ" : "ANO-PVQZ",
        "ANO-PVQZ" : "ANO-PV5Z",
        "AUG-ANO-PVDZ" : "AUG-ANO-PVTZ",
        "AUG-ANO-PVTZ" : "AUG-ANO-PVQZ",
        "AUG-ANO-PVQZ" : "AUG-ANO-PV5Z"
    }

    # Takes out original and larger basis set

    small_basis = calculation.original_basis
    large_basis = basis_pairs.get(small_basis)

    small_basis_zeta = "double" if small_basis in double_zeta_bases else "quadruple" if small_basis in quadruple_zeta_bases else "triple"

    if not large_basis: 
        
        error(f"Basis set extrapolation is not available for \"{small_basis}\". Check the manual for compatible basis sets!")

    if small_basis_zeta == "double":

        log(f"\nBeginning basis set extrapolation with double- and triple-zeta basis sets...", calculation, 1, silent=silent)
        log(f"Double-zeta basis is {basis_types.get(small_basis)}, triple-zeta basis is {basis_types.get(large_basis)}.", calculation, 1, silent=silent)

        log_spacer(calculation, silent=silent, start="\n")
        log(f"               Double-zeta Calculation", calculation, 1, silent=silent, colour="white")
        log_spacer(calculation, silent=silent)
    
    elif small_basis_zeta == "triple":

        log(f"\nBeginning basis set extrapolation with triple- and quadruple-zeta basis sets...", calculation, 1, silent=silent)
        log(f"Triple-zeta basis is {basis_types.get(small_basis)}, quadruple-zeta basis is {basis_types.get(large_basis)}.", calculation, 1, silent=silent)

        log_spacer(calculation, silent=silent, start="\n")
        log(f"               Triple-zeta Calculation", calculation, 1, silent=silent, colour="white")
        log_spacer(calculation, silent=silent)

    else:

        log(f"\nBeginning basis set extrapolation with quadruple- and quintuple-zeta basis sets...", calculation, 1, silent=silent)
        log(f"Quadruple-zeta basis is {basis_types.get(small_basis)}, quintuple-zeta basis is {basis_types.get(large_basis)}.", calculation, 1, silent=silent)

        log_spacer(calculation, silent=silent, start="\n")
        log(f"              Quadruple-zeta Calculation", calculation, 1, silent=silent, colour="white")
        log_spacer(calculation, silent=silent)


    calculation.basis = small_basis

    # Calculates the energy with the first basis

    SCF_output_small, molecule_small, E_total_small, P_small  = calculate_energy(calculation, atomic_symbols, coordinates, P_guess=P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, silent=silent, do_correlation=do_correlation, terse=terse, integrals=integrals)

    calculation.basis = large_basis
    
    header = "               Triple-zeta Calculation" if small_basis_zeta == "double" else "             Quadruple-zeta Calculation" if small_basis_zeta == "triple" else "              Quintuple-zeta Calculation" 

    log_spacer(calculation, silent=silent, start="\n")
    log(header, calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    # Calculates the energy with the second basis

    SCF_output_large, _, E_total_large, _  = calculate_energy(calculation, atomic_symbols, coordinates, terse=terse, do_correlation=do_correlation, silent=silent)

    E_SCF_small = SCF_output_small.energy
    E_SCF_large = SCF_output_large.energy

    E_corr_small = E_total_small - E_SCF_small
    E_corr_large = E_total_large - E_SCF_large

    # Extrapolates the energies

    E_extrapolated = kern.calculate_extrapolated_energy(small_basis, E_SCF_small, E_SCF_large, E_corr_small, E_corr_large, calculation, silent, small_basis_zeta)
    
    # Uses the extrapolated energy as the central point in a polarisability calculation

    if not silent and calculation.dipole:

        calculate_numerical_dipole_moment(molecule_small, calculation, silent, atomic_symbols, coordinates, None)
    
    if not silent and calculation.quadrupole:

        calculate_numerical_quadrupole_moment(molecule_small, calculation, silent, atomic_symbols, coordinates, None)

    if not silent and calculation.polarisability:

        calculate_polarisability(molecule_small, calculation, E_extrapolated, silent, atomic_symbols, coordinates, None)
    
    if not silent and calculation.hyperpolarisability :

        calculate_hyperpolarisability(molecule_small, calculation, silent, atomic_symbols, coordinates, None)
        
        
    calculation.basis = small_basis
    
    return SCF_output_small, molecule_small, E_extrapolated, P_small
    









def calculate_self_consistent_guess(calculation: Calculation, atomic_symbols: list[str], coordinates: ndarray, molecule: Molecule, S_inverse: ndarray, silent: bool = False) -> tuple:

    """

    Calculates a minimal basis self-consistent field calculation for a guess density. This can't be in tuna_guess because it needs to call calculate_energy.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates  (array): Nuclear coordinates
        molecule (Molecule): Molecule object
        S_inverse (array): Inverse overlap matrix
        silent (bool, optional): Should anything be printed
    
    Returns:
        P_guess (array): Guess density matrix
        P_guess_alpha (array): Guess alpha density matrix
        P_guess_beta (array): Guess beta density matrix
        guess_energy (float): Energy guess

    """

    log("\n Calculating self-consistent density for guess...  ", calculation, end="", silent=silent); sys.stdout.flush()

    # Stores the full basis

    old_basis = calculation.basis

    calculation.basis = "STO-3G"
    
    # Performs a minimal basis SCF calculation

    SCF_output, molecule_minimal, guess_energy, _ = calculate_energy(calculation, atomic_symbols, coordinates, terse=True, silent=True, do_correlation=False)

    # Restores calculation to old basis

    calculation.basis = old_basis

    S_cross = guess.calculate_cross_basis_overlap_matrix(molecule, molecule_minimal)

    P_guess_alpha_minimal = SCF_output.P_alpha
    P_guess_beta_minimal = SCF_output.P_beta

    # Projects minimal density matrices onto larger basis with cross overlap matrix

    P_guess_alpha = guess.project_density_matrix(P_guess_alpha_minimal, S_cross, S_inverse)
    P_guess_beta = guess.project_density_matrix(P_guess_beta_minimal, S_cross, S_inverse)

    P_guess = P_guess_alpha + P_guess_beta

    log("[Done]", calculation, silent=silent)


    return P_guess, P_guess_alpha, P_guess_beta, guess_energy










def calculate_polarisability(molecule: Molecule, calculation: Calculation, energy: float, silent: bool, atomic_symbols: list, coordinates: ndarray, integrals: Integrals | None) -> float:

    """
    
    Calculates the dipole-dipole polarisability with finite electric fields.

    This requires the diatomic molecule to be aligned along the z-axis.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        energy (float): Molecular energy in zero field
        silent (bool): Cancel logging
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        integrals (Integrals): Molecular integrals
    
    Returns:
        isotropic_polarisability (float): Isotropic polarisability

    """

    original_electric_field = calculation.electric_field.copy()

    # This allows polarisability calculations within applied electric fields

    electric_field_x = np.array([constants.SECOND_ELEC_DERIVATIVE_PROD, 0.0, 0.0]) 
    electric_field_z = np.array([0.0, 0.0, constants.SECOND_ELEC_DERIVATIVE_PROD])

    log(f"\n Beginning dipole-dipole polarisability calculation... ", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent, start="\n")
    log(f"                    Polarisability", calculation, 1, silent=silent)
    log_spacer(calculation, 1, silent=silent)

    log(f"  Using a finite field magnitude of {constants.SECOND_ELEC_DERIVATIVE_PROD:.5f} au.", calculation, 1, silent=silent)


    def calculate_second_electric_field_derivative(electric_field: ndarray) -> tuple:

        # Performs second derivative of energy with respect to electric field along an axis

        calculation.electric_field = original_electric_field + electric_field * 2
        
        _, _, E_forward_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

        calculation.electric_field = original_electric_field + electric_field 

        _, _, E_forward, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
        
        calculation.electric_field = original_electric_field - electric_field

        _, _, E_backward, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
        
        calculation.electric_field = original_electric_field - electric_field * 2

        _, _, E_backward_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
        
        # Calculates numerical second derivative for component of polarisability

        polarisability_component = -1 * calculate_second_derivative(E_backward_far, E_backward, energy, E_forward, E_forward_far, constants.SECOND_ELEC_DERIVATIVE_PROD)

        return polarisability_component, E_backward, E_forward
    

    # Only two components of polarisability are indepdendent for diatomics

    log(f"\n  Calculating parallel derivative...         ", calculation, 1, silent=silent, end="")

    polarisability_parallel, E_backward_parallel, E_forward_parallel = calculate_second_electric_field_derivative(electric_field_z)
    
    # Calculates numerical dipole moment - this can be done for all electronic structure methods

    electronic_dipole_moment = -1 * calculate_first_derivative(E_backward_parallel, E_forward_parallel, constants.SECOND_ELEC_DERIVATIVE_PROD)

    log(f"[Done]", calculation, 1, silent=silent)

    log(f"  Calculating perpendicular derivative...    ", calculation, 1, silent=silent, end="")
    
    polarisability_perpendicular, _, _ = calculate_second_electric_field_derivative(electric_field_x)
    
    log(f"[Done]", calculation, 1, silent=silent)

    # Restores the electric field to baseline
    
    calculation.electric_field = original_electric_field

    # Calculates the two linearly independent components of polarisability for diatomics

    anisotropic_polarisability = polarisability_parallel - polarisability_perpendicular 
    isotropic_polarisability = (polarisability_perpendicular * 2 + polarisability_parallel) / 3

    nuclear_dipole_moment = props.calculate_nuclear_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates)

    total_dipole_moment = electronic_dipole_moment + nuclear_dipole_moment

    log(f"\n  Dipole moment:                         {total_dipole_moment:10.4f}", calculation, 1, silent=silent)

    log(f"\n  Parallel component:                    {polarisability_parallel:10.4f}", calculation, 3, silent=silent)
    log(f"  Perpendicular component:               {polarisability_perpendicular:10.4f}", calculation, 3, silent=silent) 

    log(f"\n  Ansotropic polarisability:             {anisotropic_polarisability:10.4f}", calculation, 1, silent=silent)
    log(f"  Isotropic polarisability:              {isotropic_polarisability:10.4f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent)

    return isotropic_polarisability










def calculate_hyperpolarisability(molecule: Molecule, calculation: Calculation, silent: bool, atomic_symbols: list, coordinates: ndarray, integrals: Integrals | None) -> tuple:

    """
    
    Calculates the dipole-dipole-dipole hyperpolarisability with finite electric fields.

    This requires the diatomic molecule to be aligned along the z-axis.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        integrals (Integrals): Molecular integrals
    
    Returns:
        parallel_hyperpolarisability (float): Parallel component of hyperpolarisability
        perpendicular_hyperpolarisability (float): Perpendicular component of hyperpolarisability

    """

    # For atoms, the numerical derivative displacement is better being higher

    original_electric_field = calculation.electric_field.copy()

    # This allows polarisability calculations within applied electric fields

    electric_field_x = np.array([constants.THIRD_ELEC_DERIVATIVE_PROD, 0.0, 0.0]) 
    electric_field_z = np.array([0.0, 0.0, constants.THIRD_ELEC_DERIVATIVE_PROD])

    log(f"\n Beginning dipole-dipole-dipole hyperpolarisability calculation... ", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent, start="\n")
    log(f"                 Hyperpolarisability", calculation, 1, silent=silent)
    log_spacer(calculation, 1, silent=silent)

    log(f"  Using a finite field magnitude of {constants.THIRD_ELEC_DERIVATIVE_PROD:.5f} au.", calculation, 1, silent=silent)

    # Only two components of hyperpolarisability are indepdendent for diatomics

    log(f"\n  Calculating parallel derivative...         ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

    # Performs third derivative of energy with respect to electric field along an axis
    
    calculation.electric_field = original_electric_field + electric_field_z * 3
    
    _, _, E_forward_very_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

    calculation.electric_field = original_electric_field + electric_field_z * 2
    
    _, _, E_forward_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

    calculation.electric_field = original_electric_field + electric_field_z 

    _, _, E_forward, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_z

    _, _, E_backward, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_z * 2

    _, _, E_backward_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_z * 3
    
    _, _, E_backward_very_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_z * 4
    
    _, _, E_backward_super_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

    calculation.electric_field = original_electric_field + electric_field_z * 4
    
    _, _, E_forward_super_far, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    
    parallel_hyperpolarisability = -1 * calculate_third_derivative(E_backward_super_far, E_backward_very_far, E_backward_far, E_backward, E_forward, E_forward_far, E_forward_very_far, E_forward_super_far, constants.THIRD_ELEC_DERIVATIVE_PROD)


    log(f"[Done]", calculation, 1, silent=silent)

    log(f"  Calculating perpendicular derivative...    ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    # Performs first derivative of energy with respect to electric field along an axis (z), of second derivative of energy wrt. field along another axis (x)

    calculation.electric_field = original_electric_field + electric_field_x + electric_field_z

    _, _, E_forward_plus, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

    calculation.electric_field = original_electric_field - electric_field_x + electric_field_z

    _, _, E_backward_plus, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field + electric_field_x - electric_field_z

    _, _, E_forward_minus, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_x - electric_field_z

    _, _, E_backward_minus, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    # Calculates numerical third derivative for perpendicular component of hyperpolarisability

    perpendicular_hyperpolarisability = -(E_backward_plus - 2 * E_forward + E_forward_plus - E_backward_minus + 2 * E_backward - E_forward_minus) / (2 * constants.THIRD_ELEC_DERIVATIVE_PROD ** 3)

    log(f"[Done]", calculation, 1, silent=silent)

    # Calculates numerical dipole moment - this can be done for all electronic structure methods

    electronic_dipole_moment = -1 * calculate_first_derivative(E_backward, E_forward, constants.THIRD_ELEC_DERIVATIVE_PROD)

    # Restores the electric field to baseline
    
    calculation.electric_field = original_electric_field

    # Calculates the two linearly independent components of hyperpolarisability for diatomics

    nuclear_dipole_moment = props.calculate_nuclear_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates)

    total_dipole_moment = electronic_dipole_moment + nuclear_dipole_moment

    log(f"\n  Dipole moment:                         {total_dipole_moment:10.4f}", calculation, 1, silent=silent)

    log(f"\n  Parallel hyperpolarisability:          {parallel_hyperpolarisability:10.4f}", calculation, 1, silent=silent)
    log(f"  Perpendicular hyperpolarisability:     {perpendicular_hyperpolarisability:10.4f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent)

    return parallel_hyperpolarisability, perpendicular_hyperpolarisability










def calculate_numerical_dipole_moment(molecule: Molecule, calculation: Calculation, silent: bool, atomic_symbols: list, coordinates: ndarray, integrals: Integrals | None) -> float:

    """
    
    Calculates the dipole moment with finite electric fields.

    This requires the diatomic molecule to be aligned along the z-axis.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        integrals (Integrals): Molecular integrals
    
    Returns:
        total_dipole_moment (float): Total dipole moment

    """

    original_electric_field = calculation.electric_field.copy()

    # This allows dipole moment calculations within applied electric fields

    electric_field_z = np.array([0.0, 0.0, constants.FIRST_ELEC_DERIVATIVE_PROD])

    log(f"\n Beginning dipole moment calculation... ", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent, start="\n")
    log(f"                    Dipole Moment", calculation, 1, silent=silent)
    log_spacer(calculation, 1, silent=silent)

    log(f"  Using a finite field magnitude of {constants.FIRST_ELEC_DERIVATIVE_PROD:.5f} au.", calculation, 1, silent=silent)

    log(f"\n  Calculating parallel derivative...         ", calculation, 1, silent=silent, end="")

    # Performs first derivative of energy with respect to electric field along the z-axis

    calculation.electric_field = original_electric_field + electric_field_z 

    _, _, E_forward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = original_electric_field - electric_field_z

    _, _, E_backward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    # Calculates numerical first derivative for dipole moment

    electronic_dipole_moment = -1 * calculate_first_derivative(E_backward_parallel, E_forward_parallel, constants.FIRST_ELEC_DERIVATIVE_PROD)

    log(f"[Done]", calculation, 1, silent=silent)

    # Restores the electric field to baseline
    
    calculation.electric_field = original_electric_field

    nuclear_dipole_moment = props.calculate_nuclear_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates)

    total_dipole_moment = electronic_dipole_moment + nuclear_dipole_moment

    log(f"\n  Nuclear dipole moment:                 {nuclear_dipole_moment:10.5f}", calculation, 1, silent=silent)
    log(f"  Electronic dipole moment:              {electronic_dipole_moment:10.5f}", calculation, 1, silent=silent)
    log(f"\n  Total dipole moment:                   {total_dipole_moment:10.5f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent)


    return total_dipole_moment










def calculate_numerical_quadrupole_moment(molecule: Molecule, calculation: Calculation, silent: bool, atomic_symbols: list, coordinates: ndarray, integrals: Integrals | None) -> float:

    """
    
    Calculates the quadrupole moment with finite electric field gradients.

    This requires the diatomic molecule to be aligned along the z-axis.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        integrals (Integrals): Molecular integrals
    
    Returns:
        isotropic_quadrupole (float): Isotropic quadrupole moment

    """

    electric_field_gradient_x = np.array([constants.FIRST_ELEC_DERIVATIVE_PROD, 0.0, 0.0])
    electric_field_gradient_z = np.array([0.0, 0.0, constants.FIRST_ELEC_DERIVATIVE_PROD])

    log(f"\n Beginning quadrupole moment calculation... ", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent, start="\n")
    log(f"                   Quadrupole Moment", calculation, 1, silent=silent)
    log_spacer(calculation, 1, silent=silent)

    log(f"  Using a finite gradient magnitude of {constants.FIRST_ELEC_DERIVATIVE_PROD:.5f} au.", calculation, 1, silent=silent)

    log(f"\n  Calculating parallel derivative...         ", calculation, 1, silent=silent, end="")

    # Performs first derivative of energy with respect to electric field gradient along the z-axis

    calculation.electric_field_gradient = electric_field_gradient_z

    _, _, E_forward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field_gradient = -electric_field_gradient_z

    _, _, E_backward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    # Calculates numerical first derivative for z-component of quadrupole moment

    electronic_quadrupole_moment_z = -1 * calculate_first_derivative(E_backward_parallel, E_forward_parallel, constants.FIRST_ELEC_DERIVATIVE_PROD)
    
    log(f"[Done]", calculation, 1, silent=silent)

    log(f"  Calculating perpendicular derivative...    ", calculation, 1, silent=silent, end="")

    # Performs first derivative of energy with respect to electric field gradient along the x-axis

    calculation.electric_field_gradient = electric_field_gradient_x

    _, _, E_forward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field_gradient = -electric_field_gradient_x

    _, _, E_backward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    # Calculates numerical first derivative for x-component of quadrupole moment

    electronic_quadrupole_moment_x = -1 * calculate_first_derivative(E_backward_parallel, E_forward_parallel, constants.FIRST_ELEC_DERIVATIVE_PROD)
    
    log(f"[Done]", calculation, 1, silent=silent)

    nuclear_quadrupole_moment = props.calculate_nuclear_quadrupole_moment(molecule.centre_of_mass, molecule.charges, coordinates)

    quadrupole_moment_z = electronic_quadrupole_moment_z + nuclear_quadrupole_moment

    anisotropic_quadrupole = quadrupole_moment_z - electronic_quadrupole_moment_x
    isotropic_quadrupole = (2 * electronic_quadrupole_moment_x + quadrupole_moment_z ) / 3

    log(f"\n  Nuclear quadrupole moment:             {nuclear_quadrupole_moment:10.5f}", calculation, 1, silent=silent)
    
    log(f"\n  Electronic quadrupole moment (x):      {electronic_quadrupole_moment_x:10.5f}", calculation, 1, silent=silent)
    log(f"  Electronic quadrupole moment (z):      {electronic_quadrupole_moment_z:10.5f}", calculation, 1, silent=silent)
    
    log(f"\n  Anisotropic quadrupole moment:         {anisotropic_quadrupole:10.5f}", calculation, 1, silent=silent)
    log(f"  Isotropic quadrupole moment:           {isotropic_quadrupole:10.5f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent)


    return isotropic_quadrupole










def build_molecule_and_integrals(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, silent: bool, guess_container: tuple, do_correlation: bool, integrals: Integrals = None) -> tuple:
    
    """
    
    Builds a molecule, calculates the molecular integrals and sets up the guess density.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        silent (bool): Should anything be printed
        guess_container (tuple): Tuple containing the guess density matrices and guess energy
        do_correlation (bool): Should exit after SCF
    
    Returns:
        molecule (Molecule): Molecule object
        integrals (Integrals): Integrals object containing the one- and two-electron integrals
        guess_container (tuple): Tuple containing the guess density matrices and guess energy
        grid_container (tuple): Tuple containing the basis functions on the grid, the grid weights and the basis function gradients on the grid
        X (array): Fock transformation matrix
        V_NN (float): Nuclear repulsion energy
        E_D2 (float): D2 dispersion energy

    """

    log("\n Setting up molecule...  ", calculation, 1, silent=silent, end="")

    # Builds molecule object using calculation and atomic parameters

    molecule = Molecule(atomic_symbols, coordinates, calculation, do_correlation=do_correlation)

    log("[Done]\n", calculation, 1, silent=silent)
    
    # Prints out the information about the molecule and calculation

    kern.print_molecule_information(molecule, calculation, silent)
    
    # Calculates nuclear repulsion energy

    V_NN = kern.calculate_nuclear_repulsion_energy(molecule.charges, coordinates, calculation, silent) if calculation.diatomic else 0
    
    # Calculates D2 dispersion energy if requested

    E_D2 = kern.calculate_D2_dispersion_energy(molecule, calculation, silent) if calculation.diatomic and calculation.D2 else 0
    
    # Prints "Beginning RHF/UHF/KS calculation..."

    kern.print_reference_type(calculation.method, calculation, silent)

    # Calculates the integrals between Gaussian basis functions

    integrals = kern.calculate_analytical_integrals(molecule, calculation, silent) if integrals is None else integrals

    # Calculates Fock transformation matrix from overlap matrix

    X, smallest_S_eigenvalue, S_inverse = kern.calculate_Fock_transformation_matrix(integrals.S, calculation, silent)

    # Makes sure there is no linear dependency in the basis set

    kern.check_overlap_eigenvalues(smallest_S_eigenvalue, calculation, silent=silent)

    P_guess, P_guess_alpha, P_guess_beta, E_guess = guess_container

    # Calls a minimal SCF calculation to get a self-consistent guess density

    if calculation.self_consistent_guess and do_correlation and P_guess is None and P_guess_alpha is None and P_guess_beta is None:

        P_guess, P_guess_alpha, P_guess_beta, E_guess = calculate_self_consistent_guess(calculation, atomic_symbols, coordinates, molecule, S_inverse, silent=silent)
    
    # Calculates initial guess

    E_guess, P_guess, P_guess_alpha, P_guess_beta = guess.setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, integrals, X, calculation, molecule, S_inverse, atomic_symbols, silent=silent)

    # Force the trace of the guess density to be correct

    P_guess, P_guess_alpha, P_guess_beta = kern.enforce_density_matrix_idempotency(P_guess_alpha, P_guess_beta, integrals.S, molecule.n_alpha, molecule.n_beta, calculation, silent)

    guess_container = P_guess, P_guess_alpha, P_guess_beta, E_guess

    # Sets up the integration grid for DFT calculations

    bfs_on_grid, weights, bf_gradients_on_grid = dft.set_up_integration_grid(molecule.basis_functions, molecule.atoms, molecule.bond_length, molecule.n_electrons, P_guess_alpha, P_guess_beta, calculation, silent=silent) if calculation.DFT_calculation else (None, None, None)

    grid_container = bfs_on_grid, weights, bf_gradients_on_grid


    return molecule, integrals, guess_container, grid_container, X, V_NN, E_D2










def calculate_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, P_guess: ndarray = None, P_guess_alpha: ndarray = None, P_guess_beta: ndarray = None,
                     E_guess: float = None, terse: bool = False, silent: bool = False, do_correlation: bool = True, integrals: Integrals = None) -> tuple:
    
    """
    
    Calculates the full molecular energy.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): Atomic symbols
        coordinates (array): Atomic coordinates
        P_guess (array, optional): Guess density matrix
        P_guess_alpha (array, optional): Guess alpha density matrix
        P_guess_beta (array, optional): Guess beta density matrix
        E_guess (array, optional): Guess energy
        terse (bool, optional): Cancel post-SCF output
        silent (bool, optional): Cancel logging
        do_correlation (bool, optional): Exit after SCF or not
        integrals (Integrals, optional): Molecular integrals
    
    Returns:
        SCF_output (Output): Output object
        molecule (Molecule): Molecule object
        final_energy (float): Final energy
        P (array): Final density matrix
    
    """

    guess_container = P_guess, P_guess_alpha, P_guess_beta, E_guess
    
    # Ensures the molecule is aligned on the z-axis

    coordinates = clean_coordinates(coordinates)

    # Builds the molecule, calculates molecular integrals and prepares the guess density

    molecule, integrals, guess_container, grid_container, X, V_NN, E_D2 = build_molecule_and_integrals(calculation, atomic_symbols, coordinates, silent, guess_container, do_correlation, integrals=integrals)

    # Updates the integral matrices if an electric field is applied

    integrals.F = kern.apply_electric_field(integrals.D, calculation.electric_field) if np.linalg.norm(calculation.electric_field) > 0 else np.zeros_like(integrals.S)

    # Updates the integral matrices if an electric field gradient is applied

    integrals.F = kern.apply_electric_field_gradient(integrals.Q, calculation.electric_field_gradient)  if np.linalg.norm(calculation.electric_field_gradient) > 0 else integrals.F

    # Runs the self-consistent field cycle, returning an Output object with the results

    SCF_output = scf.run_self_consistent_field_cycle(molecule, calculation, integrals, V_NN, X, guess_container, grid_container, silent)
    
    calculation.SCF_time = time.perf_counter()

    log(f" Time taken for SCF iterations:  {calculation.SCF_time - calculation.integrals_time:.2f} seconds\n", calculation, 3, silent=silent)

    if not do_correlation:

        return SCF_output, molecule, SCF_output.energy, SCF_output.P

    # Performs correlated calculations and prints the energy calculation output

    final_energy, P = kern.run_post_SCF_energy_calculation(molecule, integrals, SCF_output, grid_container, calculation, X, E_D2, V_NN, silent, terse)
    
    # Checking if "not silent" here ensures these functions only run once, not when multiple energy evaluations are needed for silent derivatives

    if not calculation.extrapolate and not silent:

        # Electric properties are calculated here

        if calculation.dipole:

            calculate_numerical_dipole_moment(molecule, calculation, False, atomic_symbols, coordinates, integrals)
        
        if calculation.quadrupole:

            calculate_numerical_quadrupole_moment(molecule, calculation, False, atomic_symbols, coordinates, integrals)

        if calculation.polarisability:

            calculate_polarisability(molecule, calculation, final_energy, False, atomic_symbols, coordinates, integrals)
        
        if calculation.hyperpolarisability:

            calculate_hyperpolarisability(molecule, calculation, False, atomic_symbols, coordinates, integrals)
        

    return SCF_output, molecule, final_energy, P

    








def scan_coordinate(calculation: Calculation, atomic_symbols: list, starting_coordinates: ndarray, silent: bool = False, reverse: bool = False) -> tuple:

    """

    Loops through a number of scan steps and increments bond length, calculating enery each time.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        starting_coordinates (array): Atomic coordinates to being coordinate scan 
        silent (bool, optional): Cancel logging
        reverse (bool, optional): Scan bond length in negative direction
    
    Returns:
        bond_lengths (list): Bond lengths at each scan step
        energies (list): Eenergies at each scan step 
        dipole_moments (list): Dipole moments at each scan step 

    """

    coordinates = starting_coordinates
    bond_length = calculate_bond_length(coordinates)

    step_size = angstrom_to_bohr(calculation.step)
    
    # Reverses step size if requested

    if reverse: 
        
        step_size = -1 * step_size   

    log(f"Initialising a {calculation.number_of_steps} step coordinate scan in {step_size:.4f} angstrom increments.", calculation, 1, silent=silent) 
    log(f"Starting at a bond length of {bohr_to_angstrom(bond_length):.4f} angstroms.\n", calculation, 1, silent=silent)
    
    bond_lengths, energies, dipole_moments = [], [], []
    P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None


    for step in range(1, calculation.number_of_steps + 1):
        
        # This is safe for molecules not stuck on the z axis

        bond_length = calculate_bond_length(coordinates)

        log_big_spacer(calculation, start="\n",space="", silent=silent)
        log(f"Starting scan step {step} of {calculation.number_of_steps} with bond length of {bohr_to_angstrom(bond_length):.5f} angstroms...", calculation, 1, silent=silent)
        log_big_spacer(calculation,space="", silent=silent)

        # Calculates the energy at the coordinates (in bohr) specified

        SCF_output, molecule, energy, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True, silent=silent)

        if calculation.dipole:

            dipole_moment = calculate_numerical_dipole_moment(molecule, calculation, True, atomic_symbols, coordinates, SCF_output.integrals)

        else:

            dipole_moment, _, _ = props.calculate_analytical_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates, SCF_output.P, SCF_output.D)
        
        dipole_moments.append(dipole_moment)

        #If "MOREAD" keyword is used, then the energy and densities are used for the next calculation

        if calculation.MO_read: 
            
            P_guess = SCF_output.P
            E_guess = energy 
            P_guess_alpha = SCF_output.P_alpha 
            P_guess_beta = SCF_output.P_beta


        energies.append(energy)
        bond_lengths.append(bond_length)

        # Builds new coordinates by adding step size on

        coordinates = np.array([coordinates[0], [0, 0, bond_length + step_size]]) 

        # Don't let the bond length get too small when doing a reverse scan

        if bond_length + step_size <= angstrom_to_bohr(0.2) and reverse: break

    log_big_spacer(calculation, start="\n",space="", silent=silent)    
    
    log("\nCoordinate scan calculation finished!\n\n Printing energy as a function of bond length...\n", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)
    log("                   Coordinate Scan", calculation, 1, colour="white", silent=silent)
    log_spacer(calculation, silent=silent)
    log("  Step         Bond Length               Energy", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)

    # Prints a table of bond lengths and corresponding energies

    for i, (energy, bond_length) in enumerate(zip(energies, bond_lengths)):
        
        log(f" {i + 1:4.0f}            {bohr_to_angstrom(bond_length):.5f}             {energy:13.10f}", calculation, 1, silent=silent)

    log_spacer(calculation, silent=silent)

    # If "DELPLOT" keyword is used, delete saved Pickle plot 

    if calculation.delete_plot:
        
        out.delete_saved_plot()
        
    # If "SCANPLOT" keyword is used, plots and shows a Matplotlib graph of the data

    if calculation.scan_plot: 
        
        out.plot_coordinate_scan(calculation, bohr_to_angstrom(bond_lengths), energies)


    return bond_lengths, energies, dipole_moments