import numpy as np
import tuna_scf as scf
import sys, time
from tuna_util import *
from tuna_util import Calculation, Integrals
import tuna_postscf as postscf
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
in tuna_postscf or numerical derivatives of the energy are calculated and the energy evaluation process is repeated.

This module contains:

1. Functions to manage basis set extrapolation (evaluate_molecular_energy, extrapolate_energy)
2. The self-consistent guess function (calculate_self_consistent_guess)
3. Functions that calculate properties requiring numerical energy derivatives (calculate_polarisability, etc.)
4. Fundamental functions for energy evaluation (build_molecule_and_integrals, calculate_energy)
5. The coordinate scan function, which calls evaluate_molecular_energy several times (scan_coordinate)

"""




def evaluate_molecular_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, P_guess: ndarray = None, P_guess_alpha: ndarray = None, 
                              P_guess_beta: ndarray = None, E_guess: float = None, silent: bool = False, terse: bool = False, integrals: Integrals = None) -> tuple:
    
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

    return energy_function(calculation, atomic_symbols, coordinates, P_guess=P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, silent=silent, terse=terse, integrals=integrals)









def extrapolate_energy(calculation, atomic_symbols, coordinates, P_guess=None, P_guess_alpha=None, P_guess_beta=None, E_guess=None, silent=False, terse=False, integrals=None):
    
    #Todo get this working with polarisability calculation and simplifty, type hints

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

    Returns:
        SCF_output_2 (Output): Double-zeta SCF output
        molecule_2 (Molecule): Double-zeta Molecule object
        E_extrapolated (float): Extrapolated total energy
        P_2 (array): Double-zeta density matrix

    """

    double_zeta_bases = ["CC-PVDZ", "AUG-CC-PVDZ", "PC-1", "DEF2-SVP", "DEF2-SVPD", "ANO-PVDZ", "AUG-ANO-PVDZ"]

    basis_pairs = {

        "CC-PVDZ" : "CC-PVTZ",
        "CC-PVTZ" : "CC-PVQZ",
        "AUG-CC-PVDZ" : "AUG-CC-PVTZ",
        "AUG-CC-PVTZ" : "AUG-CC-PVQZ",
        "PC-1": "PC-2",
        "PC-2": "PC-3",
        "DEF2-SVP" : "DEF2-TZVPP",
        "DEF2-TZVP" : "DEF2-QZVP",
        "DEF2-TZVPP" : "DEF2-QZVPP",
        "DEF2-SVPD" : "DEF2-TZVPPD",
        "DEF2-TZVPD" : "DEF2-QZVPD",
        "DEF2-TZVPPD" : "DEF2-QZVPPD",
        "ANO-PVDZ" : "ANO-PVTZ",
        "ANO-PVTZ" : "ANO-PVQZ",
        "AUG-ANO-PVDZ" : "AUG-ANO-PVTZ",
        "AUG-ANO-PVTZ" : "AUG-ANO-PVQZ"
    }

    # Takes out original and secondary basis set
    first_basis = calculation.original_basis
    second_basis = basis_pairs.get(first_basis)

    if not second_basis: error(f"Basis set extrapolation is not available for \"{first_basis}\". Check the manual for compatible basis sets!")

    if first_basis in double_zeta_bases:

        log(f"\nBeginning basis set extrapolation with double- and triple-zeta basis sets...", calculation, 1, silent=silent)
        log(f"Double-zeta basis is {basis_types.get(first_basis)}, triple-zeta basis is {basis_types.get(second_basis)}.", calculation, 1, silent=silent)

        log_spacer(calculation, silent=silent, start="\n")
        log(f"               Double-zeta Calculation", calculation, 1, silent=silent, colour="white")
        log_spacer(calculation, silent=silent)

    else:

        log(f"\nBeginning basis set extrapolation with triple- and quadruple-zeta basis sets...", calculation, 1, silent=silent)
        log(f"Triple-zeta basis is {basis_types.get(first_basis)}, quadruple-zeta basis is {basis_types.get(second_basis)}.", calculation, 1, silent=silent)

        log_spacer(calculation, silent=silent, start="\n")
        log(f"               Triple-zeta Calculation", calculation, 1, silent=silent, colour="white")
        log_spacer(calculation, silent=silent)


    calculation.basis = first_basis

    # Calculates the energy with the first basis, using the guess densities
    SCF_output_2, molecule_2, E_total_2, P_2  = calculate_energy(calculation, atomic_symbols, coordinates, P_guess=P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=terse, silent=silent, integrals=integrals)

    calculation.basis = second_basis
    
    log_spacer(calculation, silent=silent, start="\n")

    if first_basis in double_zeta_bases:

        log(f"               Triple-zeta Calculation", calculation, 1, silent=silent, colour="white")

    else:

        log(f"             Quadruple-zeta Calculation", calculation, 1, silent=silent, colour="white")

    log_spacer(calculation, silent=silent)

    # Calculates the energy with the second basis
    SCF_output_3, _, E_total_3, _  = calculate_energy(calculation, atomic_symbols, coordinates, terse=terse, silent=silent)

    E_SCF_2 = SCF_output_2.energy
    E_SCF_3 = SCF_output_3.energy

    E_corr_2 = E_total_2 - E_SCF_2
    E_corr_3 = E_total_3 - E_SCF_3

    # Extrapolates the energies
    E_extrapolated, E_SCF_extrapolated, E_corr_extrapolated = kern.calculate_extrapolated_energy(first_basis, E_SCF_2, E_SCF_3, E_corr_2, E_corr_3)

    log_spacer(calculation, silent=silent, start="\n")
    log(f"              Basis Set Extrapolation", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    if first_basis in double_zeta_bases:

        log(f"  Double-zeta SCF energy:          {E_SCF_2:16.10f}", calculation, 1, silent=silent)
        log(f"  Triple-zeta SCF energy:          {E_SCF_3:16.10f}", calculation, 1, silent=silent)

    else:

        log(f"  Triple-zeta SCF energy:          {E_SCF_2:16.10f}", calculation, 1, silent=silent)
        log(f"  Quadruple-zeta SCF energy:       {E_SCF_3:16.10f}", calculation, 1, silent=silent)

    if calculation.method not in ["RHF", "HF", "UHF"]:
        
        if first_basis in double_zeta_bases:

            log(f"\n  Double-zeta correlation energy:  {E_corr_2:16.10f}", calculation, 1, silent=silent)
            log(f"  Triple-zeta correlation energy:  {E_corr_3:16.10f}", calculation, 1, silent=silent)

        else:

            log(f"\n  Triple-zeta correlation energy:  {E_corr_2:16.10f}", calculation, 1, silent=silent)
            log(f"  Quadruple-zeta correlation energy: {E_corr_3:14.10f}", calculation, 1, silent=silent)

    log(f"\n  Extrapolated SCF energy:         {E_SCF_extrapolated:16.10f}", calculation, 1, silent=silent)

    if calculation.method not in ["RHF", "HF", "UHF"]:
        
        log(f"  Extrapolated correlation energy: {E_corr_extrapolated:16.10f}", calculation, 1, silent=silent)

    log(f"  Extrapolated total energy:       {E_extrapolated:16.10f}", calculation, 1, silent=silent)

    log_spacer(calculation, silent=silent)


    return SCF_output_2, molecule_2, E_extrapolated, P_2
    







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

    log(" Calculating self-consistent density for guess...  ", calculation, end="", silent=silent); sys.stdout.flush()

    # Stores the full basis

    old_basis = calculation.basis

    calculation.basis = "STO-3G"
    
    # Performs a minimal basis SCF calculation

    SCF_output, molecule_minimal, guess_energy = calculate_energy(calculation, atomic_symbols, coordinates, terse=True, silent=True, guess_calculation=True)

    # Restores calculation to old basis

    calculation.basis = old_basis

    S_cross = guess.calculate_cross_basis_overlap_matrix(molecule, molecule_minimal)

    P_guess_alpha_minimal = SCF_output.P_alpha
    P_guess_beta_minimal = SCF_output.P_beta

    # Projects minimal density matrices onto larger basis with cross overlap matrix

    P_guess_alpha = guess.project_density_matrix(P_guess_alpha_minimal, S_cross, S_inverse)
    P_guess_beta = guess.project_density_matrix(P_guess_beta_minimal, S_cross, S_inverse)

    P_guess = P_guess_alpha + P_guess_beta

    log("[Done]\n", calculation, silent=silent)


    return P_guess, P_guess_alpha, P_guess_beta, guess_energy










def calculate_polarisability(molecule: Molecule, calculation: Calculation, energy: float, silent: bool, atomic_symbols: list, coordinates: ndarray, integrals: Integrals) -> float:

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

    electric_field_x = np.array([constants.numerical_derivative_prod, 0, 0]) 
    electric_field_z = np.array([0, 0, constants.numerical_derivative_prod])

    log(f"\n Beginning dipole-dipole polarisability calculation... ", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent, start="\n")
    log(f"                    Polarisability", calculation, 1, silent=silent)
    log_spacer(calculation, 1, silent=silent)

    log(f"  Using a finite field magnitude of {constants.numerical_derivative_prod:.5f} au.", calculation, 1, silent=silent)
 
    log(f"\n  Calculating parallel derivative...         ", calculation, 1, silent=silent, end="")

    # Performs second derivative of energy with respect to electric field along the z axis

    calculation.electric_field = electric_field_z * 2
    
    _, _, E_forward_far_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)

    calculation.electric_field = electric_field_z 

    _, _, E_forward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = -1 * electric_field_z

    _, _, E_backward_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    calculation.electric_field = -1 * electric_field_z * 2

    _, _, E_backward_far_parallel, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
    
    log(f"[Done]", calculation, 1, silent=silent)

    polarisability_z = -1 * calculate_second_derivative(E_backward_far_parallel, E_backward_parallel, energy, E_forward_parallel, E_forward_far_parallel, constants.numerical_derivative_prod)

    polarisability_x = 0

    if calculation.diatomic:

        log(f"  Calculating perpendicular derivative...    ", calculation, 1, silent=silent, end="")

        # Performs second derivative of energy with respect to electric field along the x/y axis

        calculation.electric_field = electric_field_x * 2
        
        _, _, E_forward_far_perpendicular, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True, integrals=integrals)
        
        calculation.electric_field = electric_field_x 

        _, _, E_forward_perpendicular, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates,  silent=True, integrals=integrals)

        calculation.electric_field = -electric_field_x

        _, _, E_backward_perpendicular, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates,  silent=True, integrals=integrals)
        
        calculation.electric_field = -electric_field_x * 2

        _, _, E_backward_far_perpendicular, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates,  silent=True, integrals=integrals)

        log(f"[Done]", calculation, 1, silent=silent)

        polarisability_x = -1 * calculate_second_derivative(E_backward_far_perpendicular, E_backward_perpendicular, energy, E_forward_perpendicular, E_forward_far_perpendicular, constants.numerical_derivative_prod)
    
    # Calculates the two linearly independent components of polarisability for diatomics

    anisotropic_polarisability = polarisability_z - polarisability_x 
    isotropic_polarisability = (polarisability_x * 2 + polarisability_z) / 3 if calculation.diatomic else polarisability_z

    # Calculates numerical dipole moment - this can be done for all electronic structure methods

    nuclear_dipole_moment = postscf.calculate_nuclear_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates)

    electronic_dipole_moment = -1 * calculate_first_derivative(E_backward_parallel, E_forward_parallel, constants.numerical_derivative_prod)

    total_dipole_moment = electronic_dipole_moment + nuclear_dipole_moment

    log(f"\n  Dipole moment:                         {total_dipole_moment:10.5f}", calculation, 1, silent=silent)

    log(f"\n  Parallel component:                    {polarisability_z:10.5f}", calculation, 3, silent=silent)
    log(f"  Perpendicular component:               {polarisability_x:10.5f}", calculation, 3, silent=silent) 

    log(f"\n  Ansotropic polarisability:             {anisotropic_polarisability:10.5f}", calculation, 1, silent=silent)
    log(f"  Isotropic polarisability:              {isotropic_polarisability:10.5f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1, silent=silent)

    return isotropic_polarisability









def calculate_diagonal_born_oppenheimer_correction(calculation, atomic_symbols, coordinates, molecule, final_energy, silent=False):

    # Todo, simplify this, write a docstring, rotate the molecule onto the z axis to use diatomic parity always works, use arrays for the x,y z components
    # Just change the [1][z] coordinate to be the Pythagoreaj bond length  - should give same results
    prod = constants.numerical_derivative_prod
    
    n_occ = molecule.n_occ if calculation.reference == "UHF" else molecule.n_doubly_occ


    log_spacer(calculation, 1, start="\n")
    log("         Diagonal Born-Oppenheimer Correction  ", calculation, 1, silent=silent)
    log_spacer(calculation, 1)


    def calculate_DBOC_component(atom_idx, coord_idx):
        
        displacement = np.zeros_like(coordinates)
        displacement[atom_idx, coord_idx] = prod 

        coordinates_forward = coordinates + displacement
        coordinates_backward = coordinates - displacement

        SCF_output_back, molecule_back, _, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates_backward, silent=True)
        SCF_output_for, molecule_for, _, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates_forward, silent=True)

        C_back = SCF_output_back.molecular_orbitals[:, :n_occ]
        C_forward = SCF_output_for.molecular_orbitals[:, :n_occ]

        S_cross = guess.calculate_cross_basis_overlap_matrix(molecule_back, molecule_for)
        S_plus_minus = np.abs(np.linalg.det(C_back.T @ S_cross @ C_forward))

        axis_dboc = 1 / (4 * prod ** 2 * molecule.masses[atom_idx]) * (1 - S_plus_minus)

        return axis_dboc


    if calculation.monatomic:

        log("\n  Calculating energy on displaced geometry 1 of 1...       ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

        E_DBOC_first_atom_z = calculate_DBOC_component(0, 2)

        E_DBOC_first_atom_x = E_DBOC_first_atom_y = E_DBOC_first_atom_z

        E_DBOC_first_atom = E_DBOC_first_atom_x + E_DBOC_first_atom_y + E_DBOC_first_atom_z

        E_DBOC = E_DBOC_first_atom

        E_DBOC_second_atom = E_DBOC_second_atom_x = E_DBOC_second_atom_y = E_DBOC_second_atom_z = 0

    else:
        
        log("\n  Calculating energy on displaced geometry 1 of 4...       ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

        E_DBOC_first_atom_x = calculate_DBOC_component(0, 0) * 2

        E_DBOC_first_atom_y = E_DBOC_first_atom_x
        
        log("\n  Calculating energy on displaced geometry 2 of 4...       ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

        E_DBOC_first_atom_z = calculate_DBOC_component(0, 2) * 2

        if molecule.point_group == "Cinfv":
            
            log("\n  Calculating energy on displaced geometry 3 of 4...       ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

            E_DBOC_second_atom_x = calculate_DBOC_component(1, 0) * 2

            E_DBOC_second_atom_y = E_DBOC_second_atom_x
            
            log("\n  Calculating energy on displaced geometry 4 of 4...       ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

            E_DBOC_second_atom_z = calculate_DBOC_component(1, 2) * 2

        else:

            E_DBOC_second_atom_x, E_DBOC_second_atom_y, E_DBOC_second_atom_z = E_DBOC_first_atom_x, E_DBOC_first_atom_y, E_DBOC_first_atom_z



        E_DBOC_first_atom = E_DBOC_first_atom_x + E_DBOC_first_atom_y + E_DBOC_first_atom_z
        E_DBOC_second_atom = E_DBOC_second_atom_x + E_DBOC_second_atom_y + E_DBOC_second_atom_z

        E_DBOC = E_DBOC_first_atom + E_DBOC_second_atom

    log("[Done]", calculation, 1, silent=silent)

    log("\n        First Atom                Second Atom", calculation, 1, silent=silent)

    log(f"\n  X{E_DBOC_first_atom_x:16.10f}           {E_DBOC_second_atom_x:16.10f}", calculation, 1, silent=silent)
    log(f"  Y{E_DBOC_first_atom_y:16.10f}           {E_DBOC_second_atom_y:16.10f}", calculation, 1, silent=silent)
    log(f"  Z{E_DBOC_first_atom_z:16.10f}           {E_DBOC_second_atom_z:16.10f}", calculation, 1, silent=silent)

    log(f"\n   {E_DBOC_first_atom:16.10f}           {E_DBOC_second_atom:16.10f}", calculation, 1, silent=silent)


    log(f"\n  Total diagonal correction:       {E_DBOC:16.10f}", calculation, 1, silent=silent)

    log_spacer(calculation, 1)

    final_energy += E_DBOC

    log("\n Diagonal Born-Oppenheimer energy: " + f"{E_DBOC:16.10f}", calculation, 1, silent=silent)
    log(" DBOC-corrected final energy:      " + f"{final_energy:16.10f}", calculation, 1, silent=silent)


    return final_energy, E_DBOC










def build_molecule_and_integrals(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, silent: bool, guess_container: tuple, guess_calculation: bool, integrals: Integrals = None) -> tuple:
    
    """
    
    Builds a molecule, calculates the molecular integrals and sets up the guess density.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        silent (bool): Should anything be printed
        guess_container (tuple): Tuple containing the guess density matrices and guess energy
        guess_calculation (bool): Is this molecular build for a guess calculation
    
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

    molecule = Molecule(atomic_symbols, coordinates, calculation, guess=guess_calculation)

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

    # Calculates initial guess

    E_guess, P_guess, P_guess_alpha, P_guess_beta = guess.setup_initial_guess(P_guess, P_guess_alpha, P_guess_beta, E_guess, integrals.T, integrals.V_NE, X, calculation, molecule, S_inverse, atomic_symbols,  silent=silent)

    # Calls a minimal SCF calculation to get a self-consistent guess density

    if calculation.self_consistent_guess and not guess_calculation:

        P_guess, P_guess_alpha, P_guess_beta, E_guess = calculate_self_consistent_guess(calculation, atomic_symbols, coordinates, molecule, S_inverse, silent=silent)

    # Force the trace of the guess density to be correct

    P_guess, P_guess_alpha, P_guess_beta = kern.enforce_density_matrix_idempotency(P_guess_alpha, P_guess_beta, integrals.S, molecule.n_alpha, molecule.n_beta, calculation, silent)

    guess_container = P_guess, P_guess_alpha, P_guess_beta, E_guess

    # Redefines the calculation method, reapplies it

    calculation.method = "U" + calculation.method if calculation.reference == "UHF" and not calculation.method.startswith("U") else calculation.method

    # Sets up the integration grid for DFT calculations

    bfs_on_grid, weights, bf_gradients_on_grid = dft.set_up_integration_grid(molecule.basis_functions, molecule.atoms, molecule.bond_length, molecule.n_electrons, P_guess_alpha, P_guess_beta, calculation, silent=silent) if calculation.DFT_calculation else (None, None, None)

    grid_container = bfs_on_grid, weights, bf_gradients_on_grid


    return molecule, integrals, guess_container, grid_container, X, V_NN, E_D2










def calculate_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, P_guess: ndarray = None, P_guess_alpha: ndarray = None, P_guess_beta: ndarray = None,
                     E_guess: float = None, terse: bool = False, silent: bool = False, guess_calculation: bool = False, integrals: Integrals = None) -> tuple:
    
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
        guess_calculation (bool, optional): Is this a calculation to generate guess orbitals
        integrals (Integrals, optional): Molecular integrals
    
    Returns:
        SCF_output (Output): Output object
        molecule (Molecule): Molecule object
        final_energy (float): Final energy
        P (array): Final density matrix
    
    """

    guess_container = P_guess, P_guess_alpha, P_guess_beta, E_guess
    
    # Builds the molecule, calcualtes molecular integrals and prepares the guess density

    molecule, integrals, guess_container, grid_container, X, V_NN, E_D2 = build_molecule_and_integrals(calculation, atomic_symbols, coordinates, silent, guess_container, guess_calculation, integrals=integrals)
    
    # Updates the integral matrices if an electric field is applied

    integrals.Q = kern.apply_electric_field(integrals.D, calculation.electric_field) if calculation.electric_field is not None else np.zeros_like(integrals.D)

    # Runs the self-consistent field cycle, returning an Output object with the results

    SCF_output = scf.run_self_consistent_field_cycle(molecule, calculation, integrals, V_NN, X, guess_container, grid_container, silent)
    
    calculation.SCF_time = time.perf_counter()

    log(f" Time taken for SCF iterations:  {calculation.SCF_time - calculation.integrals_time:.2f} seconds\n", calculation, 3, silent=silent)

    if guess_calculation:

        return SCF_output, molecule, SCF_output.energy

    # Performs correlated calculations and prints the energy calculation output

    final_energy, P = kern.run_post_SCF_energy_calculation(molecule, integrals, SCF_output, grid_container[1], calculation, X, E_D2, V_NN, silent, terse)
    
    # Checking if "not silent" here ensures these functions only run once, not when multiple energy evaluations are needed for silent derivatives

    if not silent and calculation.polarisability:

        calculate_polarisability(molecule, calculation, final_energy, False, atomic_symbols, coordinates, integrals)


    if not silent and calculation.diagonal_born_oppenheimer_correction: 
        
        final_energy, E_DBOC = calculate_diagonal_born_oppenheimer_correction(calculation, atomic_symbols, coordinates, molecule, final_energy, silent=silent)
        

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
    bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

    number_of_steps = calculation.scan_number
    step_size = angstrom_to_bohr(calculation.scan_step)
    
    # Reverses step size if requested

    if reverse: 
        
        step_size = -1 * step_size   

    log(f"Initialising a {number_of_steps} step coordinate scan in {step_size:.4f} angstrom increments.", calculation, 1, silent=silent) 
    log(f"Starting at a bond length of {bohr_to_angstrom(bond_length):.4f} angstroms.\n", calculation, 1, silent=silent)
    
    bond_lengths, energies, dipole_moments = [], [], []
    P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None


    for step in range(1, number_of_steps + 1):
        
        # This is safe for molecules not stuck on the z axis

        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        log_big_spacer(calculation, start="\n",space="", silent=silent)
        log(f"Starting scan step {step} of {number_of_steps} with bond length of {bohr_to_angstrom(bond_length):.5f} angstroms...", calculation, 1, silent=silent)
        log_big_spacer(calculation,space="", silent=silent)

        # Calculates the energy at the coordinates (in bohr) specified

        SCF_output, molecule, energy, _ = evaluate_molecular_energy(calculation, atomic_symbols, coordinates, P_guess, P_guess_alpha, P_guess_beta, E_guess, terse=True, silent=silent)

        dipole_moment = postscf.calculate_dipole_moment(molecule.centre_of_mass, molecule.charges, coordinates, SCF_output.P, SCF_output.D)[0]

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

    # If "DELPLOT" keyword is used, delete saved pickle plot 

    if calculation.delete_plot:
        
        out.delete_saved_plot()
        
    # If SCANPLOT keyword is used, plots and shows a matplotlib graph of the data

    if calculation.scan_plot: 
        
        out.plot_coordinate_scan(calculation, bohr_to_angstrom(np.array(bond_lengths)), energies)


    return bond_lengths, energies, dipole_moments