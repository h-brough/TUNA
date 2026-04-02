
from tuna_integrals import tuna_integral as ints
from tuna_molecule import Molecule, Atom
from tuna_util import *
from tuna_calc import Calculation
import numpy as np
from scipy.linalg import block_diag
from numpy import ndarray
import tuna_dft as dft
import sys, time
import tuna_ci as ci
import tuna_props as props
import tuna_mp as mp
import tuna_cc as cc
import tuna_out as out


"""

This is the TUNA module for various low level calculations, written first for version 0.10.0.

Here live various fairly random functions, that are used within an energy calculation but are not needed in the high level tuna_energy module.

Updated in version 0.10.1 to begin implementation of D3 dispersion correction.
Updated in version 0.11.0 to enable calculations to be run with spherical, rather than Cartesian, harmonics.

This module contains:

1. Functions for printing output (print_molecule_information, print_reference_type)
2. Functions to calculate molecular integrals (calculate_one_electron_integrals, calculate_two_electron_integrals)
3. The main function for running an energy calculation after the SCF is completed (run_post_SCF_energy_calculation)

"""



def print_molecule_information(molecule: Molecule, calculation: Calculation, silent: bool = False) -> None:

    """

    Prints information about a molecule.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    """

    # Finds the number of occupied orbitals for RHF or UHF references

    n_occ_print, n_virt_print = (molecule.n_occ, molecule.n_virt) if calculation.reference == "UHF" else (molecule.n_occ // 2, molecule.n_virt // 2)

    # Prints various information about the molecule

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("    Molecule and Basis Information", calculation, 1, silent=silent, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)

    log("  Molecular structure: " + molecule.molecular_structure, calculation, 1, silent=silent)
    log("\n  Number of basis functions: " + str(len(molecule.basis_functions)), calculation, 1, silent=silent)
    log("  Number of primitive Gaussians: " + str(np.sum(molecule.primitive_Gaussians)), calculation, 1, silent=silent)
    log("\n  Charge: " + str(molecule.charge), calculation, 1, silent=silent)
    log("  Multiplicity: " + str(molecule.multiplicity), calculation, 1, silent=silent)
    log("  Number of electrons: " + str(molecule.n_electrons), calculation, 1, silent=silent)
    log("  Number of alpha electrons: " + str(molecule.n_alpha), calculation, 1, silent=silent)
    log("  Number of beta electrons: " + str(molecule.n_beta), calculation, 1, silent=silent)
    log("  Number of occupied orbitals: " + str(n_occ_print), calculation, 1, silent=silent)
    log("  Number of virtual orbitals: " + str(n_virt_print), calculation, 1, silent=silent)
    log(f"\n  Point group: {molecule.point_group}", calculation, 1, silent=silent)

    if calculation.diatomic:
        
        log(f"  Bond length: {bohr_to_angstrom(molecule.bond_length):.5f} ", calculation, 1, silent=silent)

    for atom in molecule.atoms:

        # If the same atom makes up both atoms in the molecule, print the basis data only once

        if len(molecule.atoms) == 2 and molecule.atoms[0].basis_charge == molecule.atoms[1].basis_charge and atom is molecule.atoms[1]: break

        log(f"\n  Basis set for {atom.symbol_formatted} :\n", calculation, 3, silent=silent)

        values = molecule.basis_data[atom.basis_charge]

        # Print the basis function data for the atoms

        for orbital, params in values:

            log(f"   {orbital}", calculation, 3, silent=silent)

            for exponent, coefficient in params:

                log(f"      {exponent:15.10f}     {coefficient:10.10f}", calculation, 3, silent=silent)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1, silent=silent)


    return










def enforce_density_matrix_idempotency(P_guess_alpha: ndarray, P_guess_beta: ndarray, S: ndarray, n_alpha: int, n_beta: int, calculation: Calculation, silent: bool = False) -> tuple[ndarray, ndarray, ndarray]:

    """

    Forces the trace of the guess density to be correct.

    Args:
        P_guess_alpha (array): Guess alpha density matrix
        P_guess_beta (array): Guess beta density matrix
        S (array): Overlap matrix
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed
    
    Returns:
        P_guess (array): Idempotent guess density
        P_guess_alpha (array): Idempotent alpha guess density
        P_guess_beta (array): Idempotent beta guess density

    """
    
    # Forces the trace of the guess density to be correct

    P_guess_alpha = dft.clean_density_matrix(P_guess_alpha, S, n_alpha)
    P_guess_beta = dft.clean_density_matrix(P_guess_beta, S, n_beta)
    
    P_guess = P_guess_alpha + P_guess_beta

    return P_guess, P_guess_alpha, P_guess_beta










def calculate_extrapolated_energy(small_basis: str, E_SCF_small: float, E_SCF_large: float, E_corr_small: float, E_corr_large: float, calculation: Calculation, silent: bool, small_basis_zeta: str) -> float:

    """
    
    Calculates the extrapolated energy, from input energies.

    Args:
        small_basis (str): Lower-zeta basis to extrapolate
        E_SCF_small (float): SCF Energy from lower-zeta basis
        E_SCF_large (float): SCF Energy from higher-zeta basis
        E_corr_small (float): Correlation energy from lower-zeta basis
        E_corr_large (float): Correlation energy from higher-zeta basis
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        small_basis_zeta (str): Small basis zeta type
    
    Returns:
        E_extrapolated (float): Extrapolated energy
    
    """

    # Values from ORCA manual or Neese2010 - the quadruple-quintuple extrapolation values are assumed the same as triple-quadruple

    alpha_values = {

        "CC-PVDZ" : 4.42, "CC-PVTZ" : 5.46, "CC-PVQZ" : 5.46,
        "AUG-CC-PVDZ" : 4.30, "AUG-CC-PVTZ" : 5.79, "AUG-CC-PVQZ" : 5.79,
        "D-AUG-CC-PVDZ" : 4.30, "D-AUG-CC-PVTZ" : 5.79, "D-AUG-CC-PVQZ" : 5.79,
        "T-AUG-CC-PVDZ" : 4.30, "T-AUG-CC-PVTZ" : 5.79, "T-AUG-CC-PVQZ" : 5.79,
        "PC-1" : 7.02, "PC-2" : 9.78, "PC-3" : 9.78,
        "DEF2-SVP" : 10.39, "DEF2-TZVPP" : 7.88, "DEF2-TZVP" : 7.88,
        "DEF2-SVPD" : 10.39, "DEF2-TZVPPD" : 7.88, "DEF2-TZVPD" : 7.88,
        "ANO-PVDZ" : 5.41, "ANO-PVTZ" : 4.48, "ANO-PVQZ" : 4.48,
        "AUG-ANO-PVDZ" : 5.12, "AUG-ANO-PVTZ" : 5.00, "AUG-ANO-PVQZ" : 5.00

    }


    # To match the ORCA implementation, we use the optimised alpha value but take fixed beta values

    alpha = alpha_values.get(small_basis)
    beta = 2.4 if small_basis_zeta == "double" else 3

    if alpha is None:

        error("Your chosen basis set is not parameterised for extrapolation!")

    exponent_small = 2 if small_basis_zeta == "double" else 3 if small_basis_zeta == "triple" else 4
    exponent_large = 3 if small_basis_zeta == "double" else 4 if small_basis_zeta == "triple" else 5

    # Same SCF extrapolation as used in ORCA

    E_SCF_extrapolated = E_SCF_small + (E_SCF_large - E_SCF_small) / (1 - np.exp(alpha * (np.sqrt(exponent_small) - np.sqrt(exponent_large))))

    # Same correlation energy extrapolation as used in ORCA

    E_corr_extrapolated = (exponent_small ** beta * E_corr_small - exponent_large ** beta  * E_corr_large) / (exponent_small ** beta - exponent_large ** beta)

    E_extrapolated = E_SCF_extrapolated + E_corr_extrapolated

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                Basis Set Extrapolation", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    if small_basis_zeta == "double":

        log(f"  Double-zeta SCF energy:          {E_SCF_small:16.10f}", calculation, 1, silent=silent)
        log(f"  Triple-zeta SCF energy:          {E_SCF_large:16.10f}", calculation, 1, silent=silent)
    
    if small_basis_zeta == "triple": 
        
        log(f"  Triple-zeta SCF energy:          {E_SCF_small:16.10f}", calculation, 1, silent=silent)
        log(f"  Quadruple-zeta SCF energy:       {E_SCF_large:16.10f}", calculation, 1, silent=silent)

    else:

        log(f"  Quadruple-zeta SCF energy:       {E_SCF_small:16.10f}", calculation, 1, silent=silent)
        log(f"  Quintuple-zeta SCF energy:       {E_SCF_large:16.10f}", calculation, 1, silent=silent)


    if calculation.method.correlated_method:
        
        if small_basis_zeta == "double":

            log(f"\n  Double-zeta correlation energy:  {E_corr_small:16.10f}", calculation, 1, silent=silent)
            log(f"  Triple-zeta correlation energy:  {E_corr_large:16.10f}", calculation, 1, silent=silent)
        
        elif small_basis_zeta == "triple": 

            log(f"\n  Triple-zeta correlation energy:  {E_corr_small:16.10f}", calculation, 1, silent=silent)
            log(f"  Quadruple-zeta correlation energy: {E_corr_large:14.10f}", calculation, 1, silent=silent)

        else:

            log(f"\n  Quadruple-zeta correlation energy:{E_corr_small:15.10f}", calculation, 1, silent=silent)
            log(f"  Quintuple-zeta correlation energy:{E_corr_large:15.10f}", calculation, 1, silent=silent)

    log(f"\n  Extrapolated SCF energy:         {E_SCF_extrapolated:16.10f}", calculation, 1, silent=silent)

    if calculation.method.correlated_method:
        
        log(f"  Extrapolated correlation energy: {E_corr_extrapolated:16.10f}", calculation, 1, silent=silent)

    log(f"  Extrapolated total energy:       {E_extrapolated:16.10f}", calculation, 1, silent=silent)

    log_spacer(calculation, silent=silent)

    return E_extrapolated










def print_reference_type(method: Method, calculation: Calculation, silent: bool) -> None:

    """
    
    Prints whether the calculation is an (un)restricted HF or KS calculation.

    Args:
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Whether to suppress output

    """

    reference_type = "Kohn-Sham" if method.density_functional_method else "Hartree-Fock"

    if calculation.reference == "RHF": 
        
        log(f"\n Beginning restricted {reference_type} calculation...  ", calculation, 1, silent=silent)

    else: 
        
        log(f"\n Beginning unrestricted {reference_type} calculation...  ", calculation, 1, silent=silent)

    return










def calculate_one_electron_integrals(atoms: list[Atom], n_basis: int, basis_functions: list, centre_of_mass: float) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:

    """"
    
    Calculates one-electron integrals in the Cartesian harmonic basis.

    Args:
        atoms (list): List of atoms
        n_basis (int): Number of basis functions
        basis_functions (array): Basis functions
        centre_of_mass (float): Z-coordinate of centre of mass

    Returns:
        S_cart (array): Overlap matrix in AO basis
        T_cart (array): Kinetic energy matrix in AO basis
        V_NE_cart (array): Nuclear-electron matrix in AO basis
        D_cart (array): Dipole integrals in AO basis
        Q_cart (array): Quadrupole integrals in AO basis

    """

    # Initialises the matrices

    S_cart = np.zeros((n_basis, n_basis)) 
    V_NE_cart = np.zeros((n_basis, n_basis)) 
    T_cart = np.zeros((n_basis, n_basis)) 
    Q_cart = np.zeros((2, n_basis, n_basis)) 
    D_cart = np.zeros((3, n_basis, n_basis)) 

    dipole_origin = np.array([0, 0, centre_of_mass])

    for i in range(n_basis):

        for j in range(i + 1):
            
            # Forms the overlap and kinetic matrices

            S_cart[i, j] = S_cart[j, i] = ints.calculate_overlap_integral(basis_functions[i], basis_functions[j])
            T_cart[i, j] = T_cart[j, i] = ints.calculate_kinetic_integral(basis_functions[i], basis_functions[j])

            # Forms the x, y and z components of the dipole moment matrix

            D_cart[0, i, j] = D_cart[0, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "x")
            D_cart[1, i, j] = D_cart[1, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "y")
            D_cart[2, i, j] = D_cart[2, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "z")
            
            # Forms the xx and zz components of the quadrupole moment matrix

            Q_cart[0, i, j] = Q_cart[0, j, i] = ints.calculate_quadrupole_integral(basis_functions[i], basis_functions[j], dipole_origin, "xx")
            Q_cart[1, i, j] = Q_cart[1, j, i] = ints.calculate_quadrupole_integral(basis_functions[i], basis_functions[j], dipole_origin, "zz")

            for atom in atoms:

                # Adds to the nuclear-electron attraction matrix

                V_NE_cart[i, j] += -atom.charge * ints.calculate_nuclear_electron_integral(basis_functions[i], basis_functions[j], atom.origin)

            V_NE_cart[j, i] = V_NE_cart[i, j]


    return S_cart, T_cart, V_NE_cart, D_cart, Q_cart










def calculate_two_electron_integrals(n_basis: int, basis_functions: list) -> ndarray:

    """"
    
    Calculates two-electron integrals in the Cartesian harmonic basis.

    Args:
        n_basis (int): Number of basis functions
        basis_functions (list): Basis functions
    
    Returns:
        ERI_AO_cart (array): Electron repulsion integrals in AO basis
        
    """

    ERI_AO_cart = np.zeros((n_basis, n_basis, n_basis, n_basis))  

    # Calculates electron repulsion integrals - diatomic parity skips over known zero values if molecule is aligned on the z axis

    ERI_AO_cart = ints.calculate_electron_repulsion_integrals(n_basis, ERI_AO_cart, basis_functions)

    ERI_AO_cart = np.asarray(ERI_AO_cart)

    return ERI_AO_cart










def calculate_analytical_integrals(molecule: Molecule, calculation: Calculation, silent: bool) -> Integrals:
    
    """

    Calculates the prints information about the one and two electron Gaussian integrals.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Should anything be printed

    Returns:
        integrals (Integrals): Integrals object containing the one- and two-electron integrals

    """

    n_basis = len(molecule.basis_functions)

    # Calculates the one-electron integrals

    log(" Calculating one-electron integrals...     ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    S_cart, T_cart, V_NE_cart, D_cart, Q_cart = calculate_one_electron_integrals(molecule.atoms, n_basis, molecule.basis_functions, molecule.centre_of_mass)

    log("[Done]", calculation, 1, silent=silent)

    # Makes sure the two-electron integrals can fit in memory, and calculate them

    log(" Calculating two-electron integrals...     ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    try:

        if not is_molecule_aligned_on_z_axis(molecule): 
            
            error("Molecule is incorrectly aligned! Unable to calculate two-electron integrals.")

        ERI_AO_cart = calculate_two_electron_integrals(n_basis, molecule.basis_functions)

    except MemoryError:

        error("Not enough memory to build two-electron integrals array! Uh oh!")
    
    log("[Done]", calculation, 1, silent=silent)

    # Transforms into the spherical harmonic basis from Cartesian harmonics

    S, T, V_NE, D, Q, ERI_AO = transform_to_spherical_harmonics(S_cart, T_cart, V_NE_cart, D_cart, Q_cart, ERI_AO_cart, molecule, calculation, silent)
    
    # Measure the time taken to calculate the integrals, print if requested

    calculation.integrals_time = time.perf_counter()

    log(f"\n Time taken for integrals:  {calculation.integrals_time - calculation.start_time:.2f} seconds", calculation, 3, silent=silent)

    # Packages up the one- and two-electron integrals

    integrals = Integrals(S, T, V_NE, D, Q, ERI_AO)

    return integrals










def transform_to_spherical_harmonics(S_cart: ndarray, T_cart: ndarray, V_NE_cart: ndarray, D_cart: ndarray, Q_cart: ndarray, ERI_AO_cart: ndarray, molecule: Molecule, calculation: Calculation, silent: bool) -> tuple:

    """
    
    Transforms the one- and two-electron integrals from Cartesian to spherical harmonic basis.

    Args:
        S_cart (array): Overlap matrix in Cartesian basis
        T_cart (array): Kinetic energy matrix in Cartesian basis
        V_NE_cart (array): Nuclear-electron matrix in Cartesian basis
        D_cart (array): Dipole integrals in Cartesian basis
        Q_cart (array): Quadrupole integrals in Cartesian basis
        ERI_AO_cart (array): Electron repulsion integrals in Cartesian basis
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
    
    Returns:
        S (array): Overlap matrix in spherical harmonic basis
        T (array): Kinetic energy matrix in spherical harmonic basis
        V_NE (array): Nuclear-electron matrix in spherical harmonic basis
        D (array): Dipole integrals in spherical harmonic basis
        Q (array): Quadrupole integrals in spherical harmonic basis
        ERI_AO (array): Electron repulsion integrals in spherical harmonic basis
    
    """

    if calculation.cartesian_harmonics:

        return S_cart, T_cart, V_NE_cart, D_cart, Q_cart, ERI_AO_cart
    
    log("\n Transforming to spherical harmonics...    ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Builds the Cartesian to spherical transformation matrix

    U = build_spherical_harmonic_transformation_matrix(molecule, calculation)
    
    # Transforms the one-electron integrals

    S = U @ S_cart @ U.T
    T = U @ T_cart @ U.T
    V_NE = U @ V_NE_cart @ U.T

    # These integral matrices have multiple Cartesian components

    D = np.einsum("mw,awx,nx->amn", U, D_cart, U, optimize=True)
    Q = np.einsum("mw,awx,nx->amn", U, Q_cart, U, optimize=True)

    # Transforms the two-electron integrals - first index of U needs to be Cartesian, second spherical

    ERI_AO = np.einsum("mw,nx,wxyz,ky,lz->mnkl", U, U, ERI_AO_cart, U, U, optimize=True)
    
    log("[Done]\n", calculation, 1, silent=silent)

    return S, T, V_NE, D, Q, ERI_AO










def build_spherical_harmonic_transformation_matrix(molecule: Molecule, calculation: Calculation) -> ndarray:

    U = np.eye(molecule.n_basis)

    # Don't apply linear map if "CARTHARMONICS" is used

    U_S = np.eye(1)

    U_P = np.eye(3)

    # Cartesian harmonics are ordered from x^n, ... y^n, ... z^n 

    # get hese lines up one by one

    U_D = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],                                 # d_xy
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],                                 # d_yz
        [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],                               # d_z^2
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],                                 # d_xz
        [np.sqrt(3)/2, 0.0, 0.0, -np.sqrt(3)/2, 0.0, 0.0],              # d_x^2-y^2
    ], dtype=float)
    
    U_F = np.array([
        [0.0, 3*np.sqrt(2)/4, 0.0, 0.0, 0.0, 0.0, -np.sqrt(10)/4, 0.0, 0.0, 0.0], # y(3x^2-y^2)
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # xyz
        [0.0, -np.sqrt(30)/20, 0.0, 0.0, 0.0, 0.0, -np.sqrt(6)/4, 0.0, np.sqrt(30)/5, 0.0], # y(4z^2-x^2-y^2)
        [0.0, 0.0, -3*np.sqrt(5)/10, 0.0, 0.0, 0.0, 0.0, -3*np.sqrt(5)/10, 0.0, 1.0], # z(2z^2-3x^2-3y^2)
        [-np.sqrt(6)/4, 0.0, 0.0, -np.sqrt(30)/20, 0.0, np.sqrt(30)/5, 0.0, 0.0, 0.0, 0.0], # x(4z^2-x^2-y^2)
        [0.0, 0.0, np.sqrt(3)/2, 0.0, 0.0, 0.0, 0.0, -np.sqrt(3)/2, 0.0, 0.0], # z(x^2-y^2)
        [np.sqrt(10)/4, 0.0, 0.0, -3*np.sqrt(2)/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # x(x^2-3y^2)
    ], dtype=float)

    #print(np.array([molecule.basis_functions[i].shell for i in range(molecule.n_basis)]))


    U_G = np.array([
            # m = -4: xy(x^2-y^2)
            [0.0, np.sqrt(35)/4, 0.0, 0.0, 0.0, 0.0, -np.sqrt(35)/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # m = -3: yz(3x^2-y^2)
            [0.0, 0.0, 0.0, 0.0, np.sqrt(14)/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.sqrt(14)/4, 0.0, 0.0, 0.0],
            # m = -2: xy(7z^2-r^2)
            [0.0, -np.sqrt(7)/4, 0.0, 0.0, 0.0, 0.0, -np.sqrt(7)/4, 0.0, np.sqrt(7)/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # m = -1: yz(7z^2-3r^2)
            [0.0, 0.0, 0.0, 0.0, -3*np.sqrt(14)/20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3*np.sqrt(14)/20, 0.0, np.sqrt(14)/5, 0.0],
            # m = 0: 35z^4 - 30z^2r^2 + 3r^4
            [3/8, 0.0, 0.0, 3/4, 0.0, -3*np.sqrt(5)/8, 0.0, 0.0, 0.0, 0.0, 3/8, 0.0, -3*np.sqrt(5)/8, 0.0, 1.0],
            # m = 1: xz(7z^2-3r^2)
            [0.0, 0.0, -3*np.sqrt(14)/20, 0.0, 0.0, 0.0, 0.0, -3*np.sqrt(14)/20, 0.0, np.sqrt(14)/5, 0.0, 0.0, 0.0, 0.0, 0.0],
            # m = 2: (x^2-y^2)(7z^2-r^2)
            [-np.sqrt(7)/4, 0.0, 0.0, 0.0, 0.0, np.sqrt(7)/2, 0.0, 0.0, 0.0, 0.0, np.sqrt(7)/4, 0.0, -np.sqrt(7)/2, 0.0, 0.0],
            # m = 3: xz(x^2-3y^2)
            [0.0, 0.0, np.sqrt(14)/4, 0.0, 0.0, 0.0, 0.0, -np.sqrt(14)/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # m = 4: x^4 - 6x^2y^2 + y^4
            [np.sqrt(35)/8, 0.0, 0.0, -3*np.sqrt(3)/4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(35)/8, 0.0, 0.0, 0.0, 0.0]
        ], dtype=float)


    # Links angular momentum to linear map matrix block

    block_map = {0: U_S, 1: U_P, 2: U_D, 3: U_F, 4: U_G, 5: None, 6: None}
    
    i = 0

    # Iteratively builds block diagonal transformation matrix

    while i < molecule.n_basis:

        # Angular momentum for shell

        L = sum(molecule.basis_functions[i].shell)

        # How many iterations to jump forwards depends on angular momentum (1 for s, 3 for p, 6 for d, 10 for f, etc.)

        n_cart = (L + 1) * (L + 2) // 2

        U = block_diag(U, block_map[L]) if i != 0 else np.eye(1)

        i += n_cart 


    return U










def apply_electric_field(D: ndarray, electric_field: ndarray) -> ndarray:

    """
    
    Determines the one-electron integrals within an electric field, the contribution to the Hamiltonian.

    Args:
        D (array): Dipole integrals
        electric_field (array): Electric field
    
    Returns:
        F (array): Electric field integrals
    
    """

    F = np.einsum("i,ijk->jk", electric_field, D, optimize=True)

    return F










def apply_electric_field_gradient(Q: ndarray, electric_field_gradient: ndarray) -> ndarray:

    """
    
    Determines the one-electron integrals within an electric field, the contribution to the Hamiltonian.

    Args:
        D (array): Dipole integrals
        electric_field_gradient (array): Electric field gradient
    
    Returns:
        F (array): Electric field integrals
    
    """

    # There are only two independent components of the quadrupole tensor for diatomics

    Q = np.array([Q[0], np.zeros_like(Q[0]), Q[1]])

    F = np.einsum("i,ijk->jk", electric_field_gradient, Q, optimize=True)

    return F










def calculate_nuclear_repulsion_energy(charges: ndarray, coordinates: ndarray, calculation: Calculation, silent: bool = False) -> float:
    
    """

    Calculates nuclear repulsion energy.

    Args:
        charges (array): Nuclear charges
        coordinates (array): Atomic coordinates
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        V_NN (float): Nuclear-nuclear repulsion energy

    """
    
    log(" Calculating nuclear repulsion energy...  ", calculation, 1, end="", silent=silent)

    # Does not rely on molecule being aligned on z axis

    V_NN = np.prod(charges) / np.linalg.norm(coordinates[1] - coordinates[0])
    
    log(f"[Done]\n\n Nuclear repulsion energy: {V_NN:.10f}\n", calculation, 1, silent=silent)

    return V_NN
    









def calculate_Fock_transformation_matrix(S: ndarray, calculation: Calculation, silent: bool = False) -> tuple[ndarray, float, ndarray]:

    """

    Diagonalises the overlap matrix to find its square root, then inverts this as X = S^-1/2.

    Args:
        S (array): Overlap matrix in AO basis
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        X (array): Fock transformation matrix
        smallest_S_eigenvalue (float): Smallest overlap matrix eigenvalue.
        S_inverse (array): Inverse overlap matrix
        
    """

    log(" Constructing Fock transformation matrix...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Symmetrise the overlap matrix

    S = symmetrise(S)

    # Diagonalises overlap matrix

    S_vals, S_vecs = np.linalg.eigh(S)

    # This error only happens in really weird situations

    if min(S_vals) < 0:

        error("A negative overlap matrix eigenvalue was found!")

    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    # Finds the smalest eigenvalue of the overlap matrix to check for linear dependencies

    smallest_S_eigenvalue = np.min(S_vals)

    # Inverse square root of overlap matrix is Fock transformation matrix

    X = np.linalg.inv(S_sqrt)

    # Forms inverse density matrix

    S_inverse = np.linalg.inv(S)

    log("[Done]", calculation, 1, silent=silent)

    return X, smallest_S_eigenvalue, S_inverse










def print_SCF_energy(final_energy: float, reference: str, method: Method, calculation: Calculation, silent: bool) -> None:

    """
    
    Prints the converged SCF energy.

    Args:
        final_energy (float): Final SCF energy
        reference (str): Reference state
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
    
    """
    
    space = " " * max(0, 8 - len(method.name))

    if reference == "RHF" and not calculation.DFT_calculation: 
        
        log("\n Restricted Hartree-Fock energy:   " + f"{final_energy:16.10f}", calculation, 1, silent=silent)
    
    elif reference == "UHF" and not calculation.DFT_calculation: 
        
        log("\n Unrestricted Hartree-Fock energy: " + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    elif reference == "RHF":
        
        log(f"\n Restricted {method.name} energy: {space}      " + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    elif reference == "UHF":
            
            log(f"\n Unrestricted {method.name} energy: {space}    " + f"{final_energy:16.10f}", calculation, 1, silent=silent)


    return
        









def check_overlap_eigenvalues(smallest_S_eigenvalue: float, calculation: Calculation, silent: bool = False) -> None:

    """

    Checks the smallest eigenvalue of the overlap matrix against a threshold, raising an error if it is too small.

    Args:
        smallest_S_eigenvalue (float): Smallest eigenvalue of the overlap matrix
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    """

    log(f"\n Smallest overlap matrix eigenvalue is {smallest_S_eigenvalue:.8f}, threshold is {calculation.S_eigenvalue_threshold:.8f}.", calculation, 2, silent=silent)

    if smallest_S_eigenvalue < calculation.S_eigenvalue_threshold:

        error("An overlap matrix eigenvalue is too small! Change the basis set or decrease the threshold with STHRESH.")
    
    elif smallest_S_eigenvalue < 10 * calculation.S_eigenvalue_threshold:

        warning(f"Smallest overlap matrix eigenvalue is close to the threshold, at {smallest_S_eigenvalue:.8f}! \n", space=1)

    return










def calculate_fractional_coordination_number(atoms: list, bond_length: float) -> float:

    """

    Calculates the fractional coordination number for D3 dispersion correction.

    Args:
        atoms (list): List of atom objects
        bond_length (float): Bond length in bohr
    
    Returns:
        coordination_number (float): Fractional coordination number
    
    """

    k_1 = 16
    k_2 = 4 / 3

    # This equation is from Grimme2010

    exponential_term = -k_1 * (k_2 * (atoms[0].vdw_radius + atoms[1].vdw_radius) / bond_length - 1)

    coordination_number = 1 / (1 + np.exp(exponential_term))

    return coordination_number










def calculate_D3_dispersion_energy(molecule: Molecule, bond_length: float) -> float:
    
    """

    Calculates the semi-empirical D3 dispersion energy.

    Args:
        molecule (list): List of atom objects
        bond_length (float): Bond length in bohr
    
    Returns:
        E_D3 (float): D3 dispersion energy
   
    """

    def damping_function(alpha_n, s_n, R):

        return 1 / (1 + 6 * (bond_length / (s_n * R)) ** - alpha_n)

    s_6, s_8 = 0.0, 0.0
    C_6, C_8 = 0.0, 0.0
    alpha_6, alpha_8 = 0.0, 0.0

    R = 0.0

    E_D3_S6 = s_6 * C_6 / bond_length ** 6 * damping_function(alpha_6, s_6, R)
    E_D3_S8 = s_8 * C_8 / bond_length ** 8 * damping_function(alpha_8, s_8, R)

    E_D3 = E_D3_S6 + E_D3_S8

    raise(NotImplementedError)

    return E_D3










def calculate_D2_dispersion_energy(molecule: Molecule, calculation: Calculation, silent: bool) -> float:

    """

    Calculates the D2 semi-empirical dispersion energy.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        silent (bool): Should anything be printed

    Returns:
        E_D2 (float): D2 semi-empirical dispersion energy

    """

    atoms = molecule.atoms

    # If a D2 parameterised exchange-correlation functional is used, take that value, otherwise 1.2 matches ORCA's Hartree-Fock implementation

    S6 = calculation.functional.D2_S6 if calculation.DFT_calculation else 1.2

    log(f" Calculating semi-empirical dispersion energy with S6 value of {S6:.3f}...  ", calculation, 1, end="", silent=silent)
    
    # This parameter was chosen to match the implementation of Hartree-Fock D2 in ORCA

    damping_factor = 20
    
    C6 = np.sqrt(atoms[0].C6 * atoms[1].C6)
    vdw_sum = atoms[0].vdw_radius + atoms[1].vdw_radius

    f_damp = 1 / (1 + np.exp(-1 * damping_factor * (molecule.bond_length / vdw_sum - 1)))
    
    # Uses conventional dispersion energy expression, with damping factor to account for short bond lengths

    E_D2 = -1 * S6 * C6 / molecule.bond_length ** 6 * f_damp
    
    log(f"[Done]\n\n Dispersion energy (D2): {E_D2:.10f}\n", calculation, 1, silent=silent)

    return E_D2
        









def run_post_SCF_energy_calculation(molecule: Molecule, integrals: Integrals, SCF_output: Output, grid_container: tuple, calculation: Calculation, X: ndarray, E_D2: float, V_NN: float, silent: bool, terse: bool) -> tuple:

    """
    
    Runs the post-SCF parts of an energy calculation.

    Args:
        molecule (Molecule): Molecule object
        integrals (Integrals): Molecular integrals
        SCF_output (Output): Output object
        grid_container (tuple): Grid information for DFT
        calculation (Calculation): Calculation
        X (array): Fock transformation matrix
        E_D2 (float): D2 dispersion energy
        V_NN (float): Nuclear repulsion energy
        silent (bool): Cancel logging
        terse (bool): Cancel post-SCF output

    Returns:
        E (float): Total molecular energy
        P (array): Total molecular density matrix
    
    """

    reference = calculation.reference
    method = calculation.method
    do_DFT = calculation.DFT_calculation

    bfs_on_grid, weights, _ = grid_container

    molecular_orbitals = SCF_output.molecular_orbitals
    P = SCF_output.P
    P_alpha = SCF_output.P_alpha
    P_beta = SCF_output.P_beta
    final_energy = SCF_output.energy

    E_CC, E_CC_perturbative = 0, 0
    E_CIS, E_transition = 0, 0

    natural_orbitals = None
    SCF_output.D = integrals.D
    SCF_output.Q = integrals.Q


    if reference == "UHF": 
        
        reference_type = "UKS" if do_DFT else "UHF"

        # Calculates UHF spin contamination and prints to the console

        props.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, reference_type, silent=silent)

        # Calculates the natural orbitals if requested

        if calculation.natural_orbitals: 
                
            _, natural_orbitals = mp.calculate_natural_orbitals(P, X, calculation, silent=silent)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1, silent=silent)


    # Prints the individual components of the total SCF energy

    props.print_energy_components(SCF_output, V_NN, calculation, silent=silent)

    if do_DFT: 
        
        dft.integrate_final_density(SCF_output.alpha_density, SCF_output.beta_density, SCF_output.density, weights, calculation, silent)


    # If a Moller-Plesset calculation is requested, calculates the energy and density matrices

    if method.perturbative_method or calculation.MPC_prop != 0: 
            
        E_MP2, E_MP3, E_MP4, P, P_alpha, P_beta, _, natural_orbitals = mp.run_perturbation_theory_calculation(method, molecule, SCF_output, integrals, calculation, V_NN, silent=silent)
        
        props.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, "MP2", silent=silent)


    # If a coupled-cluster calculation is requested, calculates the energy and density matrices

    elif method.method_base == "CC":

        E_CC, E_CC_perturbative, (P, P_alpha, P_beta), _, natural_orbitals = cc.begin_coupled_cluster_calculation(method, molecule, SCF_output, integrals, X, calculation, silent)
        
        props.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, "Coupled cluster", silent=silent)


    if method.correlated_method:

        calculation.correlation_time = time.perf_counter()

        log(f"\n Time taken for correlated calculation:  {calculation.correlation_time - calculation.SCF_time:.2f} seconds", calculation, 3, silent=silent)


    # Prints post SCF information, as long as its not an optimisation that hasn't finished yet

    if not terse and not silent:
        
        props.calculate_molecular_properties(molecule, calculation, P, integrals.S, SCF_output, P_alpha, P_beta)
    

    if method.excited_state_method:

        log("\n\n Beginning excited state calculation...", calculation, 1, silent=silent)

        if molecule.n_virt <= 0: 
            
            error("Excited state calculation requested on system with no virtual orbitals!")

        # Calculates the CIS excited states energy and density

        E_CIS, E_transition, P, P_alpha, P_beta, P_transition, P_transition_alpha, P_transition_beta = ci.run_CIS(integrals.ERI_AO, molecule.n_occ, molecule.n_virt, molecule.n_SO, calculation, SCF_output, molecule, silent=silent)
        
        calculation.excited_state_time = time.perf_counter()

        log(f"\n Time taken for excited state calculation:  {calculation.excited_state_time - calculation.SCF_time:.2f} seconds", calculation, 3, silent=silent)

        if calculation.additional_print: 
           
           # Optionally uses CIS density for dipole moment and population analysis

           props.calculate_molecular_properties(molecule, calculation, P, integrals.S, SCF_output, P_alpha, P_beta)

    else:
        
        P_transition = P_transition_alpha = P_transition_beta = None


    # Prints Hartree-Fock or Kohn-Sham energy

    print_SCF_energy(final_energy, reference, method, calculation, silent)


    # Adds up and prints MP2 energies

    if method.method_base == "MP2" or calculation.MPC_prop != 0: 
        
        space = " " * max(0, 8 - len(method.name))

        # If a double-hybrid functional is being used, multiply by the correlation proportion

        E_MP2 *= calculation.MPC_prop if do_DFT else 1

        final_energy += E_MP2

        if do_DFT:
            
            log(f" Double-hybrid correlation energy: " + f"{E_MP2:16.10f}\n", calculation, 1, silent=silent)

        else:
            
            log(f" Correlation energy from {method.name}: {space}" + f"{E_MP2:16.10f}\n", calculation, 1, silent=silent)


    # Adds up and prints MP3 energies

    elif method.method_base == "MP3":
        
        final_energy += E_MP2 + E_MP3

        if method.name == "SCS-MP3":

            log(f" Correlation energy from SCS-MP2:  " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from SCS-MP3:  " + f"{E_MP3:16.10f}\n", calculation, 1, silent=silent)

        else:

            log(f" Correlation energy from MP2:      " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from MP3:      " + f"{E_MP3:16.10f}\n", calculation, 1, silent=silent)

        log(f" Total correlation energy:         " + f"{E_MP2 + E_MP3:16.10f}\n", calculation, 3, silent=silent)

    # Adds up and prints MP4 energies

    elif method.method_base == "MP4":
        
        final_energy += E_MP2 + E_MP3 + E_MP4

        log(f" Correlation energy from MP2:      " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
        log(f" Correlation energy from MP3:      " + f"{E_MP3:16.10f}", calculation, 1, silent=silent)

        if method.name in ["MP4", "MP4[SDTQ]"]:

            log(f" Correlation energy from MP4:      " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        elif method.name == "MP4[SDQ]":

            log(f" Correlation energy from MP4(SDQ): " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        elif method.name == "MP4[DQ]":

            log(f" Correlation energy from MP4(DQ):  " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        log(f" Total correlation energy:         " + f"{E_MP2 + E_MP3 + E_MP4:16.10f}\n", calculation, 3, silent=silent)

    # Adds up and prints coupled cluster energies

    elif method.method_base == "CC":

        method.name = method.name.replace("[", "(").replace("]", ")")

        final_energy += E_CC + E_CC_perturbative
        
        space = " " * max(0, 8 - len(method.name))

        if "(" in method.name:

            log(f" Correlation energy from {method.name.split("(")[0]}:{space}    {E_CC:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from {method.name}: {space}{E_CC_perturbative:16.10f}\n", calculation, 1, silent=silent)
            log(f" Total correlation energy: {space}       {E_CC + E_CC_perturbative:16.10f}\n", calculation, 3, silent=silent)

        else:
            
            log(f" Correlation energy from {method.name}:{space} " + f"{E_CC:16.10f}\n", calculation, 1, silent=silent)

        # Important to return this to baseline for multi-energy calculations

        method.name = method.name.replace("(", "[").replace(")", "]")



    # Prints CIS energy of state of interest

    elif method.excited_state_method:

        final_energy = E_CIS

        method.name = method.name.replace("[", "(").replace("]", ")")
        space = " " * max(0, 8 - len(method.name))

        log(f"\n Excitation energy is the energy difference to excited state {calculation.root}.", calculation, 1, silent=silent)
        
        log(f"\n Excitation energy from {method.name}:  {space}" + f"{E_transition:16.10f}", calculation, 1, silent=silent)
    
    
    # This is the total final energy

    log(" Final single point energy:        " + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    # Adds on D2 energy, and prints this as dispersion-corrected final energy

    if calculation.D2:
    
        final_energy += E_D2

        log("\n Semi-empirical dispersion energy: " + f"{E_D2:16.10f}", calculation, 1, silent=silent)
        log(" Dispersion-corrected final energy:" + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    # If plotting has been requested, send the density and orbital information to the plotting module

    if not silent and calculation.plot_something:

        out.show_two_dimensional_plot(calculation, molecule, P, P_alpha, P_beta, P_transition_alpha, P_transition_beta, P_transition, molecular_orbitals, natural_orbitals)


    return final_energy, P










def calculate_charge_change_energy(reference_energy: float, charged_energy: float, reference_molecule: Molecule, charged_molecule: Molecule, calculation: Calculation) -> float:
 
    """
    
    Calculates and prints the vertical or adiabatic ionisation potential or electron affinity.

    Args:
        reference_energy (float): Final energy of original system
        charged_energy (float): Final energy of charged system
        reference_molecule (Molecule): Final molecule of original system
        charged_molecule (Molecule): Final molecule of charged system
        calculation (Calculation): Calculation object
    
    Returns:
        energy_change (float): Either ionisation potential or electron affinity
    
    """

    charge_difference = charged_molecule.charge - reference_molecule.charge

    # The convention for electron affinity is the other way around from ionisation potential

    energy_change = charged_energy - reference_energy if charge_difference > 0 else reference_energy - charged_energy

    # Is the molecule allowed to relax or not

    prefix = "Vertical" if calculation.vertical or calculation.monatomic else "Adiabatic"

    if charge_difference > 0:

        property_name = "Ionisation Potential"

        action_line = f"  Ionisation from charge {format_charge(reference_molecule.charge)} to {format_charge(charged_molecule.charge)}..."
        

    # There will always be a charge difference of some kind

    else:

        property_name = "Electron Affinity"

        action_line = f"  Electron attachment from charge {format_charge(reference_molecule.charge)} to {format_charge(charged_molecule.charge)}..."
        

    log_spacer(calculation, start="\n")

    log(f"{property_name:^55}", calculation)

    log_spacer(calculation)

    log(action_line, calculation)

    log(f"\n  Energy of reference system:      {reference_energy:16.10f}", calculation)
    log(f"  Energy of charged system:        {charged_energy:16.10f}", calculation, end="\n\n")

    if not calculation.monatomic and not calculation.vertical:

        log(f"  Bond length of reference system:     {bohr_to_angstrom(reference_molecule.bond_length):12.5f}", calculation)
        log(f"  Bond length of charged system:       {bohr_to_angstrom(charged_molecule.bond_length):12.5f}", calculation, end="\n\n") 

    label = f"  {prefix} {property_name.lower()}:"
    log(f"{label:<35}{energy_change:16.10f}", calculation)

    log_spacer(calculation)


    return energy_change










def print_bond_dissociation_energy_information(first_atom_energy: float, second_atom_energy: float, optimised_energy: float, zero_point_energy: float, optimised_molecule: Molecule, calculation: Calculation) -> None:

    """
    
    Calculates the bond dissociation energy from input energies, and prints it.

    Args:
        first_atom_energy (float): Final energy of first atom
        second_atom_energy (float): Final energy of second atom
        optimised_energy (float): Final energy of optimised molecule
        zero_point_energy (float): Zero-point energy of optimised molecule
        optimised_molecule (Molecule): Molecule at equilibrium geometry
        calculation (Calculation): Calculation object    
    
    """

    # Simply calculate the bond dissociation energy, with and without zero-point correction

    corrected_diatomic_energy = optimised_energy + zero_point_energy

    bond_dissociation_energy = first_atom_energy + second_atom_energy - optimised_energy
    bond_dissociation_energy_corrected = first_atom_energy + second_atom_energy - corrected_diatomic_energy

    log_spacer(calculation, start="\n")
    log(f"             Bond Disscociation Energy", calculation)
    log_spacer(calculation)
    
    if calculation.no_counterpoise_correction:
        
        log(f"  Atomic energies are not counterpoise corrected...\n", calculation)
    
    else:

        log(f"  Atomic energies are counterpoise corrected...\n", calculation)

    first_space = " " * (5 - len(optimised_molecule.atoms[0].symbol_formatted))
    second_space = " " * (5 - len(optimised_molecule.atoms[1].symbol_formatted))

    log(f"  Energy of {optimised_molecule.atoms[0].symbol_formatted} atom:            {first_space}{first_atom_energy:16.10f}", calculation)
    
    # Only print out second atom information if the molecule is heteronuclear

    if optimised_molecule.heteronuclear:
        
        log(f"  Energy of {optimised_molecule.atoms[1].symbol_formatted} atom:            {second_space}{second_atom_energy:16.10f}", calculation)
    
    log(f"\n  Molecular energy:                {optimised_energy:16.10f}", calculation)
    
    if calculation.do_ZPE_correction:

        log(f"  Zero-point energy:               {zero_point_energy:16.10f}", calculation)
        log(f"\n  Corrected molecular energy:      {corrected_diatomic_energy:16.10f}", calculation)

    log(f"\n  Bond dissociation energy:        {bond_dissociation_energy:16.10f}", calculation)
    
    if calculation.do_ZPE_correction:

        log(f"  Corrected dissociation energy:   {bond_dissociation_energy_corrected:16.10f}", calculation)

    log_spacer(calculation)

    return