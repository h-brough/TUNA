
from tuna_integrals import tuna_integral as ints
from tuna_molecule import Molecule, Atom
from tuna_util import *
import numpy as np
from numpy import ndarray
import tuna_dft as dft
import sys, time
import tuna_ci as ci
import tuna_postscf as postscf
import tuna_mp as mp
import tuna_cc as cc
import tuna_out as out


"""

This is the TUNA module for various low level calculations, written first for version 0.10.0.

Here live various fairly random functions, that are used within an energy calculation but are not needed in the high level tuna_energy module.

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

    if len(molecule.atomic_symbols) == 2: log(f"  Bond length: {bohr_to_angstrom(molecule.bond_length):.5f} ", calculation, 1, silent=silent)

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

    log(" Enforcing density matrix idempotency...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Forces the trace of the guess density to be correct

    P_guess_alpha = dft.clean_density_matrix(P_guess_alpha, S, n_alpha)
    P_guess_beta = dft.clean_density_matrix(P_guess_beta, S, n_beta)
    
    P_guess = P_guess_alpha + P_guess_beta

    log("[Done]\n", calculation, 1, silent=silent)

    return P_guess, P_guess_alpha, P_guess_beta










def calculate_extrapolated_energy(small_basis: str, E_SCF_small: float, E_SCF_large: float, E_corr_small: float, E_corr_large: float, calculation: Calculation, silent: bool) -> float:

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
    
    Returns:
        E_extrapolated (float): Extrapolated energy
    
    """

    double_zeta_bases = ["CC-PVDZ", "AUG-CC-PVDZ", "PC-1", "DEF2-SVP", "DEF2-SVPD", "ANO-PVDZ", "AUG-ANO-PVDZ"]

    # Values from ORCA manual or Neese2010

    alpha_values = {

        "CC-PVDZ" : 4.42, "CC-PVTZ" : 5.46,
        "AUG-CC-PVDZ" : 4.30, "AUG-CC-PVTZ" : 5.79,
        "PC-1" : 7.02, "PC-2" : 9.78,
        "DEF2-SVP" : 10.39, "DEF2-TZVPP" : 7.88,
        "DEF2-SVPD" : 10.39, "DEF2-TZVPPD" : 7.88,
        "ANO-PVDZ" : 5.41, "ANO-PVTZ" : 4.48,
        "AUG-ANO-PVDZ" : 5.12,  "AUG-ANO-PVTZ" : 5.00

    }

    beta_values = {

        "CC-PVDZ" : 2.46, "CC-PVTZ" : 3.05,
        "AUG-CC-PVDZ" : 2.51, "AUG-CC-PVTZ" : 3.05,
        "PC-1": 2.01, "PC-2": 4.09,
        "DEF2-SVP" : 2.40, "DEF2-TZVPP" : 2.97,
        "DEF2-SVPD" : 2.40, "DEF2-TZVPPD" : 2.97,
        "ANO-PVDZ" : 2.43, "ANO-PVTZ" : 2.97,
        "AUG-ANO-PVDZ" : 2.41, "AUG-ANO-PVTZ" : 2.52
    }

    alpha = alpha_values.get(small_basis)
    beta = beta_values.get(small_basis)

    exponent_small = 2 if small_basis in double_zeta_bases else 3
    exponent_large = 3 if small_basis in double_zeta_bases else 4

    # Same SCF extrapolation as used in ORCA

    E_SCF_extrapolated = E_SCF_small + (E_SCF_large - E_SCF_small) / (1 - np.exp(alpha * (np.sqrt(exponent_small) - np.sqrt(exponent_large))))

    # Same correlation energy extrapolation as used in ORCA

    E_corr_extrapolated = (exponent_small ** beta * E_corr_small - exponent_large ** beta  * E_corr_large) / (exponent_small ** beta - exponent_large ** beta)

    E_extrapolated = E_SCF_extrapolated + E_corr_extrapolated

    log_spacer(calculation, silent=silent, start="\n")
    log(f"              Basis Set Extrapolation", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    if small_basis in double_zeta_bases:

        log(f"  Double-zeta SCF energy:          {E_SCF_small:16.10f}", calculation, 1, silent=silent)
        log(f"  Triple-zeta SCF energy:          {E_SCF_large:16.10f}", calculation, 1, silent=silent)

    else:

        log(f"  Triple-zeta SCF energy:          {E_SCF_small:16.10f}", calculation, 1, silent=silent)
        log(f"  Quadruple-zeta SCF energy:       {E_SCF_large:16.10f}", calculation, 1, silent=silent)

    if calculation.method in correlated_methods:
        
        if small_basis in double_zeta_bases:

            log(f"\n  Double-zeta correlation energy:  {E_corr_small:16.10f}", calculation, 1, silent=silent)
            log(f"  Triple-zeta correlation energy:  {E_corr_large:16.10f}", calculation, 1, silent=silent)

        else:

            log(f"\n  Triple-zeta correlation energy:  {E_corr_small:16.10f}", calculation, 1, silent=silent)
            log(f"  Quadruple-zeta correlation energy: {E_corr_large:14.10f}", calculation, 1, silent=silent)

    log(f"\n  Extrapolated SCF energy:         {E_SCF_extrapolated:16.10f}", calculation, 1, silent=silent)

    if calculation.method in correlated_methods:
        
        log(f"  Extrapolated correlation energy: {E_corr_extrapolated:16.10f}", calculation, 1, silent=silent)

    log(f"  Extrapolated total energy:       {E_extrapolated:16.10f}", calculation, 1, silent=silent)

    log_spacer(calculation, silent=silent)

    return E_extrapolated










def print_reference_type(method: str, calculation: Calculation, silent: bool) -> None:

    """
    
    Prints whether the calculation is an (un)restricted HF or KS calculation.

    Args:
        method (str): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Whether to suppress output

    """

    reference_type = "Kohn-Sham" if method in DFT_methods else "Hartree-Fock"

    if calculation.reference == "RHF": 
        
        log(f" Beginning restricted {reference_type} calculation...  \n", calculation, 1, silent=silent)

    else: 
        
        log(f" Beginning unrestricted {reference_type} calculation...  \n", calculation, 1, silent=silent)

    return










def calculate_one_electron_integrals(atoms: list[Atom], n_basis: int, basis_functions: list, centre_of_mass: float) -> tuple[ndarray, ndarray, ndarray, ndarray]:

    """"
    
    Calculates one-electron integrals.

    Args:
        atoms (list): List of atoms
        n_basis (int): Number of basis functions
        basis_functions (array): Basis functions
        centre_of_mass (float): Z-coordinate of centre of mass

    Returns:
        S (array): Overlap matrix in AO basis
        T (array): Kinetic energy matrix in AO basis
        V_NE (array): Nuclear-electron matrix in AO basis
        D (array): Dipole integrals in AO basis

    """

    # Initialises the matrices

    S = np.zeros((n_basis, n_basis)) 
    V_NE = np.zeros((n_basis, n_basis)) 
    T = np.zeros((n_basis, n_basis)) 
    D = np.zeros((3, n_basis, n_basis)) 

    dipole_origin = np.array([0, 0, centre_of_mass])


    for i in range(n_basis):

        for j in range(i + 1):
            
            # Forms the overlap and kinetic matrices

            S[i, j] = S[j, i] = ints.calculate_overlap_integral(basis_functions[i], basis_functions[j])
            T[i, j] = T[j, i] = ints.calculate_kinetic_integral(basis_functions[i], basis_functions[j])

            # Forms the x, y and z components of the dipole moment matrix

            D[0, i, j] = D[0, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "x")
            D[1, i, j] = D[1, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "y")
            D[2, i, j] = D[2, j, i] = ints.calculate_dipole_integral(basis_functions[i], basis_functions[j], dipole_origin, "z")

            for atom in atoms:

                # Adds to the nuclear-electron attraction matrix

                V_NE[i, j] += -atom.charge * ints.calculate_nuclear_electron_integral(basis_functions[i], basis_functions[j], atom.origin)

            V_NE[j, i] = V_NE[i, j]


    return S, T, V_NE, D










def calculate_two_electron_integrals(n_basis: int, basis_functions: list) -> ndarray:

    """"
    
    Calculates two-electron integrals.

    Args:
        n_basis (int): Number of basis functions
        basis_functions (list): Basis functions
    
    Returns:
        ERI_AO (array): Electron repulsion integrals in AO basis
        
    """

    ERI_AO = np.zeros((n_basis, n_basis, n_basis, n_basis))  

    # Calculates electron repulsion integrals - diatomic parity skips over known zero values if molecule is aligned on the z axis

    ERI_AO = ints.calculate_electron_repulsion_integrals(n_basis, ERI_AO, basis_functions)

    ERI_AO = np.asarray(ERI_AO)

    return ERI_AO










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

    log(" Calculating one-electron integrals...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    S, T, V_NE, D = calculate_one_electron_integrals(molecule.atoms, n_basis, molecule.basis_functions, molecule.centre_of_mass)

    log("[Done]", calculation, 1, silent=silent)

    # Makes sure the two-electron integrals can fit in memory, and calculate them

    log(" Calculating two-electron integrals...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    try:

        if not is_molecule_aligned_on_z_axis(molecule): 
            
            error("Molecule is incorrectly aligned! Unable to calculate two-electron integrals.")

        ERI_AO = calculate_two_electron_integrals(n_basis, molecule.basis_functions)

    except MemoryError:

        error("Not enough memory to build two-electron integrals array! Uh oh!")
    
    log("[Done]", calculation, 1, silent=silent)

    # Measure the time taken to calculate the integrals, print if requested

    calculation.integrals_time = time.perf_counter()

    log(f"\n Time taken for integrals:  {calculation.integrals_time - calculation.start_time:.2f} seconds", calculation, 3, silent=silent)

    # Packages up the one- and two-electron integrals

    integrals = Integrals(S, T, V_NE, D, ERI_AO)

    return integrals













def apply_electric_field(D: ndarray, electric_field: ndarray) -> ndarray:

    """
    
    Determines the one-electron integrals within an electric field, the contribution to the Hamiltonian.

    Args:
        D (array): Dipole integrals
        electric-field (array): Electric field
    
    Returns:
        Q (array): Electric field integrals
    
    """

    Q = np.einsum("i,ijk->jk", electric_field, D, optimize=True)

    return Q










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

    log("\n Constructing Fock transformation matrix...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Symmetrise the overlap matrix

    S = symmetrise(S)

    # Diagonalises overlap matrix

    S_vals, S_vecs = np.linalg.eigh(S)
    S_sqrt = S_vecs * np.sqrt(S_vals) @ S_vecs.T
    
    # Finds the smalest eigenvalue of the overlap matrix to check for linear dependencies

    smallest_S_eigenvalue = np.min(S_vals)

    # Inverse square root of overlap matrix is Fock transformation matrix

    X = np.linalg.inv(S_sqrt)

    # Forms inverse density matrix

    S_inverse = np.linalg.inv(S)

    log("[Done]", calculation, 1, silent=silent)

    return X, smallest_S_eigenvalue, S_inverse










def print_SCF_energy(final_energy: float, reference: str, method: str, calculation: Calculation, silent: bool) -> None:

    """
    
    Prints the converged SCF energy.

    Args:
        final_energy (float): Final SCF energy
        reference (str): Reference state
        method (str): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
    
    """
    
    space = " " * max(0, 8 - len(method))

    if reference == "RHF" and not calculation.DFT_calculation: 
        
        log("\n Restricted Hartree-Fock energy:   " + f"{final_energy:16.10f}", calculation, 1, silent=silent)
    
    elif reference == "UHF" and not calculation.DFT_calculation: 
        
        log("\n Unrestricted Hartree-Fock energy: " + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    elif reference == "RHF":
        
        log(f"\n Restricted {method} energy: {space}      " + f"{final_energy:16.10f}", calculation, 1, silent=silent)

    elif reference == "UHF":
            
            log(f"\n Unrestricted {method} energy: {space}    " + f"{final_energy:16.10f}", calculation, 1, silent=silent)


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
        









def run_post_SCF_energy_calculation(molecule: Molecule, integrals: Integrals, SCF_output: Output, weights: ndarray, calculation: Calculation, X: ndarray, E_D2: float, V_NN: float, silent: bool, terse: bool) -> tuple:

    """
    
    Runs the post-SCF parts of an energy calculation.

    Args:
        molecule (Molecule): Molecule object
        integrals (Integrals): Molecular integrals
        SCF_output (Output): Output object
        weights (array): Grid weights for DFT
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

    molecular_orbitals = SCF_output.molecular_orbitals
    P = SCF_output.P
    P_alpha = SCF_output.P_alpha
    P_beta = SCF_output.P_beta
    final_energy = SCF_output.energy

    E_CC, E_CC_perturbative = 0, 0
    E_CIS, E_transition = 0, 0

    natural_orbitals = None
    SCF_output.D = integrals.D

    if reference == "UHF": 
        
        reference_type = "UKS" if do_DFT else "UHF"

        # Calculates UHF spin contamination and prints to the console

        postscf.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, reference_type, silent=silent)

        # Calculates the natural orbitals if requested

        if calculation.natural_orbitals: 
                
            _, natural_orbitals = mp.calculate_natural_orbitals(P, X, calculation, silent=silent)

            log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", calculation, 1, silent=silent)


    # Prints the individual components of the total SCF energy

    postscf.print_energy_components(SCF_output, V_NN, calculation, silent=silent)

    if do_DFT: 
        
        dft.integrate_final_density(SCF_output.alpha_density, SCF_output.beta_density, SCF_output.density, weights, calculation, silent)


    # If a Moller-Plesset calculation is requested, calculates the energy and density matrices

    if method in perturbative_methods or calculation.MPC_prop != 0: 
            
        E_MP2, E_MP3, E_MP4, P, P_alpha, P_beta, _, natural_orbitals = mp.calculate_Moller_Plesset(method, molecule, SCF_output, integrals, calculation, X, V_NN, silent=silent)
        
        postscf.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, "MP2", silent=silent)


    # If a coupled-cluster calculation is requested, calculates the energy and density matrices

    elif method in coupled_cluster_methods:

        E_CC, E_CC_perturbative, (P, P_alpha, P_beta), _, natural_orbitals = cc.begin_coupled_cluster_calculation(method, molecule, SCF_output, integrals, X, calculation, silent)
        
        postscf.calculate_spin_contamination(P_alpha, P_beta, molecule.n_alpha, molecule.n_beta, integrals.S, calculation, "Coupled cluster", silent=silent)


    if method in correlated_methods:

        calculation.correlation_time = time.perf_counter()

        log(f"\n Time taken for correlated calculation:  {calculation.correlation_time - calculation.SCF_time:.2f} seconds", calculation, 3, silent=silent)


    # Prints post SCF information, as long as its not an optimisation that hasn't finished yet

    if not terse and not silent:
        
        postscf.post_SCF_output(molecule, calculation, P, integrals.S, molecule.partition_ranges, SCF_output, P_alpha, P_beta)
    

    if method in excited_state_methods:

        log("\n\n Beginning excited state calculation...", calculation, 1, silent=silent)

        if molecule.n_virt <= 0: 
            
            error("Excited state calculation requested on system with no virtual orbitals!")

        # Calculates the CIS excited states energy and density

        E_CIS, E_transition, P, P_alpha, P_beta, P_transition, P_transition_alpha, P_transition_beta = ci.run_CIS(integrals.ERI_AO, molecule.n_occ, molecule.n_virt, molecule.n_SO, calculation, SCF_output, molecule, silent=silent)
        
        calculation.excited_state_time = time.perf_counter()

        log(f"\n Time taken for excited state calculation:  {calculation.excited_state_time - calculation.SCF_time:.2f} seconds", calculation, 3, silent=silent)

        if calculation.additional_print: 
           
           # Optionally uses CIS density for dipole moment and population analysis

           postscf.post_SCF_output(molecule, calculation, P, integrals.S, molecule.partition_ranges, SCF_output, P_alpha, P_beta)

    else:
        
        P_transition = P_transition_alpha = P_transition_beta = None


    # Prints Hartree-Fock or Kohn-Sham energy

    print_SCF_energy(final_energy, reference, method, calculation, silent)


    # Adds up and prints MP2 energies

    if method in second_order_perturbative_methods or calculation.MPC_prop != 0: 
        
        space = " " * max(0, 8 - len(method))

        # If a double-hybrid functional is being used, multiply by the correlation proportion

        E_MP2 *= calculation.MPC_prop if do_DFT else 1

        final_energy += E_MP2

        if do_DFT:
            
            log(f" Double-hybrid correlation energy: " + f"{E_MP2:16.10f}\n", calculation, 1, silent=silent)

        else:
            
            log(f" Correlation energy from {method}: {space}" + f"{E_MP2:16.10f}\n", calculation, 1, silent=silent)


    # Adds up and prints MP3 energies

    elif method in third_order_perturbative_methods:
        
        final_energy += E_MP2 + E_MP3

        if method == "SCS-MP3":

            log(f" Correlation energy from SCS-MP2:  " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from SCS-MP3:  " + f"{E_MP3:16.10f}\n", calculation, 1, silent=silent)

        else:

            log(f" Correlation energy from MP2:      " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from MP3:      " + f"{E_MP3:16.10f}\n", calculation, 1, silent=silent)

        log(f" Total correlation energy:         " + f"{E_MP2 + E_MP3:16.10f}\n", calculation, 3, silent=silent)

    # Adds up and prints MP4 energies

    elif method in fourth_order_perturbative_methods:
        
        final_energy += E_MP2 + E_MP3 + E_MP4

        log(f" Correlation energy from MP2:      " + f"{E_MP2:16.10f}", calculation, 1, silent=silent)
        log(f" Correlation energy from MP3:      " + f"{E_MP3:16.10f}", calculation, 1, silent=silent)

        if method == "MP4" or method == "MP4[SDTQ]":

            log(f" Correlation energy from MP4:      " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        elif method == "MP4[SDQ]":

            log(f" Correlation energy from MP4(SDQ): " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        elif method == "MP4[DQ]":

            log(f" Correlation energy from MP4(DQ):  " + f"{E_MP4:16.10f}\n", calculation, 1, silent=silent)

        log(f" Total correlation energy:         " + f"{E_MP2 + E_MP3 + E_MP4:16.10f}\n", calculation, 3, silent=silent)

    # Adds up and prints coupled cluster energies

    elif method in coupled_cluster_methods:

        final_energy += E_CC + E_CC_perturbative

        if "CCSD[T]" in method:

            log(f" Correlation energy from CCSD:     " + f"{E_CC:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from CCSD(T):  " + f"{E_CC_perturbative:16.10f}\n", calculation, 1, silent=silent)
            log(f" Total correlation energy:         " + f"{E_CC + E_CC_perturbative:16.10f}\n", calculation, 3, silent=silent)
        
        elif "QCISD[T]" in method:

            log(f" Correlation energy from QCISD:    " + f"{E_CC:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from QCISD(T): " + f"{E_CC_perturbative:16.10f}\n", calculation, 1, silent=silent)
            log(f" Total correlation energy:         " + f"{E_CC + E_CC_perturbative:16.10f}\n", calculation, 3, silent=silent)

        elif "CCSDT[Q]" in method:

            log(f" Correlation energy from CCSDT:    " + f"{E_CC:16.10f}", calculation, 1, silent=silent)
            log(f" Correlation energy from CCSDT(Q): " + f"{E_CC_perturbative:16.10f}\n", calculation, 1, silent=silent)
            log(f" Total correlation energy:         " + f"{E_CC + E_CC_perturbative:16.10f}\n", calculation, 3, silent=silent)

        else:
            
            space = " " * max(0, 8 - len(method))

            log(f" Correlation energy from {method}:{space} " + f"{E_CC:16.10f}\n", calculation, 1, silent=silent)


    # Prints CIS energy of state of interest

    elif method in excited_state_methods:

        final_energy = E_CIS

        method = method.replace("[", "(").replace("]", ")")
        space = " " * max(0, 8 - len(method))

        log(f"\n Excitation energy is the energy difference to excited state {calculation.root}.", calculation, 1, silent=silent)
        
        log(f"\n Excitation energy from {method}:  {space}" + f"{E_transition:16.10f}", calculation, 1, silent=silent)
    
    
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