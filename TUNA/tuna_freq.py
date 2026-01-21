from tuna_util import *
import tuna_energy as energ
import tuna_out as out
import tuna_opt as opt
import tuna_thermo as thermo
import tuna_postscf as postscf
import scipy as sp
import numpy as np


"""

This is the TUNA module for calculating harmonic and anharmonic vibrational frequencies, written first for version 0.10.0 of TUNA.

The module contains:

1. Some uti

"""




def calculate_anharmonicity_constant(transition_matrix, harmonic_frequency) -> np.float64:

    """

    Calculates the anharmonicity constant, chi, from the anharmonic and harmonic transition frequencies.

    Args:
        transition_matrix (array): Matrix of transition energies in hartree
        harmonic_frequency (float): Harmonic frequency in hartree
    
    Returns:
        chi (float): Anharmonicity constant

    """

    # Assumes the 0 -> 1 and 1 -> 2 transitions are fully converged
    chi = (transition_matrix[0][1] - transition_matrix[1][2]) / (2 * harmonic_frequency)

    return chi







def calculate_transition_intensity(frequency_matrix, dipole_matrix) -> np.ndarray:
    
    """

    Calculates transition intensities from frequency and dipole matrices.

    Args:
        frequency_matrix (array): Matrix of transition frequencies in per cm
        dipole_matrix (array): Matrix of dipole moment integrals in atomic units
    
    Returns:
        intensity_matrix (array): Matrix of transition intensities in km per mol
    
    """

    # This equation comes from Neugebauer2002
    prefactor = constants.elementary_charge_in_coulombs ** 2 / constants.electron_mass_in_kilograms * constants.avogadro / (6000 * constants.permittivity_in_farad_per_metre * constants.c_in_metres_per_second ** 2)

    # Converts frequency matrix from per cm to hartree
    frequency_hartree = frequency_matrix / constants.per_cm_in_hartree

    # Intensities depend on the square of the dipole matrix elements
    intensity_matrix = prefactor * dipole_matrix ** 2 * frequency_hartree
    
    return intensity_matrix








def calculate_dipole_matrix(vibrational_wavefunctions, dipole_moments_interpolated) -> np.ndarray:

    """

    Calculates the dipole matrix elements between vibrational states.

    Args:
        vibrational_wavefunctions (array): Array of vibrational wavefunctions, shape (N_grid, N_states)
        dipole_moments_interpolated (array): Interpolated dipole moments along the vibrational coordinate

    Returns:
        dipole_matrix (array): Dipole matrix elements between vibrational states

    """

    # No dx term here as its already included due to continuum normalisation of vibrational_wavefunctions
    dipole_matrix = np.einsum("ni,n,nj->ij", vibrational_wavefunctions, dipole_moments_interpolated, vibrational_wavefunctions, optimize=True) 

    return dipole_matrix








def diagonalise_hamiltonian_tridiagonal(main_diag, off_diag) -> tuple:

    """

    Diagonalises a tridiagonal Hamiltonian matrix.

    Args:
        main_diag (array): Main diagonal of the Hamiltonian
        off_diag (array): Off-diagonal of the Hamiltonian
    
    Returns:
        vibrational_energy_levels (array): Vibrational energy levels (eigenvalues)
        vibrational_wavefunctions (array): Vibrational wavefunctions (eigenvectors)

    """

    # This method is much faster and more memory efficient for tridiagonal matrices than forming the full matrix and diagonalising it
    vibrational_energy_levels, vibrational_wavefunctions = sp.linalg.eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, 5))

    return vibrational_energy_levels, vibrational_wavefunctions








def calculate_transition_matrix(vibrational_energy_levels) -> np.ndarray:

    """

    Calculates the transition matrix between vibrational energy levels.

    Args:
        vibrational_energy_levels (array): Vibrational energy levels

    Returns:
        transition_matrix (array): Transition matrix between vibrational energy levels

    """

    # This allows easy indexing for the transitions ([0][1] is the transition from level 0 to level 1, etc.)
    transition_matrix = np.abs(vibrational_energy_levels[:, np.newaxis] - vibrational_energy_levels[np.newaxis, :])

    return transition_matrix








def construct_nuclear_hamiltonian(x_interpolated, V_interpolated, reduced_mass) -> tuple:

    """

    Constructs the nuclear Hamiltonian matrix for the potential energy surface.

    Args:
        x_interpolated (array): Interpolated nuclear coordinate
        V_interpolated (array): Interpolated potential energy surface
        reduced_mass (float): Molecular reduced mass

    Returns:
        main_diag (array): Main diagonal part of tridiagonal Hamiltonian
        off_diag (array): Off diagonal part of tridiagonal Hamiltonian

    """

    # Differential assuming the coordinate array is very fine
    dx = x_interpolated[1] - x_interpolated[0]

    # Kinetic term
    T = 1 / (reduced_mass * dx ** 2) 

    # The potential energy surface affects the diagonal term only, the off-diagonal includes coupling between grid points in the kinetic term
    main_diag = T + V_interpolated
    off_diag = np.full(len(V_interpolated) - 1, -T / 2)

    return main_diag, off_diag








def interpolate_potential_energy(V_raw, x_raw, n_grid_points) -> tuple:

    """

    Uses cubic splines to interpolate the potential energy surface.

    Args:
        V_raw (array): Potential energy surface before interpolation
        x_raw (array): Nuclear coordinate array before interpolation
        n_grid_points (int): Number of interpolation grid points

    Returns:
        x (array): Interpolated nuclear coordinate array
        V (array): Interpolated potential energy surface

    """

    # Builds a linearly spaced x-axis between the computed end points, to the desired grid density
    x = np.linspace(x_raw.min(), x_raw.max(), int(n_grid_points))

    # Cubic interpolation of the potential energy surface
    interpolation_function = sp.interpolate.interp1d(x_raw, V_raw, kind="cubic")
    V = interpolation_function(x)

    return x, V









def print_absorption_spectrum(calculation, transition_matrix, frequency_matrix, wavelength_matrix, intensity_matrix) -> None:

    """

    Prints the absorption spectrum for anharmonic vibrational spectroscopy.

    Args:
        calculation (Calculation): Calculation object
        transition_matrix (array): Matrix of transition energies
        frequency_matrix (array): Matrix of transition frequencies in per cm
        wavelength_matrix (array): Matrix of wavelengths in nm
        intensity_matrix (array): Matrix of intensities in km per mol

    """

    log_big_spacer(calculation, 1, start="\n")
    log("                                        Anharmonic Absorption Spectrum", calculation, 1, colour="white")
    log_big_spacer(calculation, 1)
    log("  Transition         Energy          Frequency (per cm)       Wavelength (nm)     Intensity (km per mol)", calculation, 1)
    log_big_spacer(calculation, 1)

    # Only transitions up to the third energy level are shown
    for i in range(3):
        for j in range(i + 1, 4):
                
                log(f"    {i} -> {j}    {transition_matrix[i][j]:16.10f}    {frequency_matrix[i][j]:16.2f}       {wavelength_matrix[i][j]:16.2f}       {intensity_matrix[i][j]:16.2f}", calculation, 1)

    log_big_spacer(calculation, 1)

    return









def solve_numerical_schrodinger_equation(reduced_mass, calculation, atomic_symbols, optimised_coordinates, harmonic_frequency_per_cm) -> None:

    # This value is more than enough for accuracy and extrapolation is not going to be the rate limiting step
    extrapolation_grid_density = 1000


    st = angstrom_to_bohr(0.05)
    extent = angstrom_to_bohr(0.5)
    transition_per_cm = 0
    fund_old = 1
    i = 0

    log("\n Setting up anharmonic frequency calculation...\n", calculation, 1)

    log(" Calculating initial potential energy surface around minimum... ", calculation, 1, end=""); sys.stdout.flush()

    calculation.scan_step = bohr_to_angstrom(st)
    calculation.scan_number = int(extent / st) + 1

    coordinates = optimised_coordinates.copy()
    coordinates[1][2] -= extent / 2

    x_vals, V_vals, dipole_moments = energ.scan_coordinate(calculation, atomic_symbols, coordinates, silent=True)

    log("[Done]\n", calculation, 1)

    log_big_spacer(calculation, 1)
    log("                                          Anharmonic Frequency", calculation, 1, colour="white")
    log_big_spacer(calculation, 1)
    log("  Step       Fundamental Freq. (per cm)         Chi        Harmonic Freq. (per cm)     Bond Length Range", calculation, 1)
    log_big_spacer(calculation, 1)
    
    while np.abs(transition_per_cm - fund_old) > 0.01:

        fund_old = transition_per_cm

        extent = max(x_vals) - min(x_vals)

        coordinates_right = coordinates.copy()
        coordinates_right[1][2] = np.max(x_vals)

        coordinates_left = coordinates.copy()
        coordinates_left[1][2] = np.min(x_vals)
        
        calculation.scan_number = int(0.3 / st) + 1
        new_x_vals_right, new_V_vals_right, new_dipole_moments_right = energ.scan_coordinate(calculation, atomic_symbols, coordinates_right, silent=True)
        new_x_vals_left, new_V_vals_left, new_dipole_moments_left = energ.scan_coordinate(calculation, atomic_symbols, coordinates_left, silent=True, reverse=True)

        x_vals = np.concatenate((np.array(new_x_vals_left[1:][::-1]), np.array(x_vals), np.array(new_x_vals_right[1:])))
        V_vals = np.concatenate((np.array(new_V_vals_left[1:][::-1]), np.array(V_vals), np.array(new_V_vals_right[1:])))
        dipole_moments = np.concatenate((np.array(new_dipole_moments_left[1:][::-1]), np.array(dipole_moments), np.array(new_dipole_moments_right[1:])))

        x, V = interpolate_potential_energy(V_vals, x_vals, extrapolation_grid_density * extent)
        _, dipole_moments_interpolated = interpolate_potential_energy(dipole_moments, x_vals, extrapolation_grid_density * extent)

        main_diag, off_diag = construct_nuclear_hamiltonian(x, V, reduced_mass)

        vibrational_energy_levels, vibrational_wavefunctions = diagonalise_hamiltonian_tridiagonal(main_diag, off_diag)
        transition_matrix = calculate_transition_matrix(vibrational_energy_levels)


        transition_per_cm = transition_matrix[0][1] * constants.per_cm_in_hartree

        chi = calculate_anharmonicity_constant(transition_matrix, harmonic_frequency_per_cm / constants.per_cm_in_hartree)

        print(f"    {i + 1}               {transition_per_cm:8.2f}                 {chi:8.5f}             {harmonic_frequency_per_cm:8.2f}             {bohr_to_angstrom(min(x_vals)):.5f} - {bohr_to_angstrom(max(x_vals)):.5f}")

        i += 1
    

    log_big_spacer(calculation, 1)

    frequency_matrix = transition_matrix * constants.per_cm_in_hartree
    wavelength_matrix = 10000000 / np.where(frequency_matrix != 0, frequency_matrix, 1)

    log(f"\n Final fundamental frequency (per cm):  {frequency_matrix[0][1]:6.2f}", calculation, 1)
    log(f" Final anharmonicity constant:  {chi:7.5f}", calculation, 1)

    log(f"\n Zero-point energy:   {vibrational_energy_levels[0] - min(V_vals):13.10f}", calculation, 1)
    log(f" Equilibrium energy:  {vibrational_energy_levels[0]:13.10f}", calculation, 1)

    dipole_matrix = calculate_dipole_matrix(vibrational_wavefunctions, dipole_moments_interpolated)

    intensity_matrix = calculate_transition_intensity(frequency_matrix, dipole_matrix)



    print_absorption_spectrum(calculation, transition_matrix, frequency_matrix, wavelength_matrix, intensity_matrix)

    if calculation.plot_vibrational_wavefunctions:

        out.plot_vibrational_wavefunctions(x, vibrational_energy_levels, vibrational_wavefunctions, x_vals, V_vals)










def calculate_frequency(calculation, atomic_symbols=None, coordinates=None, molecule=None, energy=None) -> tuple:

    """

    Calculates harmonic frequency of a molecule.

    Args:   
        calculation (Calculation): Calculation object
        atomic_symbols (list, optional): List of atomic symbols
        coordinates (array, optional): Atomic coordinates
        molecule (Molecule, optional): Molecule object
        energy (float, optional): Total molecular energy

    Returns:
        k (float): Force constant
        reduced_mass (float): Reduced mass
        frequency_per_cm (float): Frequency in per cm

    """

    # If "FREQ" keyword has been used, calculates the energy using the supplied atoms and coordinates, otherwise uses the supplied molecule and energy
    if calculation.calculation_type in ["FREQ"]:
          
        if calculation.extrapolate:

            _, molecule, energy, _ = energ.extrapolate_energy(calculation, atomic_symbols, coordinates)

        else:
        
            _, molecule, energy, _ = energ.calculate_energy(calculation, atomic_symbols, coordinates)

    # Unpacks useful molecular quantities
    point_group = molecule.point_group
    bond_length = molecule.bond_length
    atomic_symbols = molecule.atomic_symbols
    coordinates = molecule.coordinates
    masses = molecule.masses

    # Unpacks useful calculation quantities from user-defined parameters
    temperature = calculation.temperature
    pressure = calculation.pressure  


    log("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log(" Beginning harmonic frequency calculation...", calculation, 1, colour="white")
    log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    
    log(f"\n Hessian will be calculated at a bond length of {bohr_to_angstrom(bond_length):.5f} angstroms.", calculation, 1)
    
    # Spring stiffness is calculated as the Hessian, through numerical second derivatives
    k, SCF_output_forward, P_forward, SCF_output_backward, P_backward = opt.calculate_Hessian(coordinates, calculation, atomic_symbols)

    # Reduced mass calculated in order to calculate frequency of harmonic oscillator
    reduced_mass = postscf.calculate_reduced_mass(masses)


    # Checks if an imaginary mode is present, and if so, appends an "i" and sets the vibrational entropy, internal energy and zero-point energy to zero
    if k > 0:
    
        frequency_hartree = np.sqrt(k / reduced_mass)
        i = ""

        ZPE = frequency_hartree / 2
        
    else:   
    
        frequency_hartree = np.sqrt(-k / reduced_mass)
        i = " i"

        ZPE = 0
        
        warning("Imaginary frequency calculated! Zero-point energy and vibrational thermochemical parameters set to zero!\n")

    # Converts frequency into human units from atomic units
    frequency_per_cm = frequency_hartree * constants.per_cm_in_hartree

    dipole_derivative = opt.calculate_dipole_derivative(coordinates, molecule, SCF_output_forward, SCF_output_backward, P_forward, P_backward)

    # Adding vibrational overlap contribution to match ORCA results (frequencies cancel for harmonic oscillator)
    vibrational_overlap = 1 / np.sqrt(2 * frequency_hartree)
    dipole_derivative *= vibrational_overlap

    dipole_derivative_squared = dipole_derivative ** 2

    transition_intensity_km_per_mol = calculate_transition_intensity(frequency_per_cm, dipole_derivative)

    if calculation.custom_mass_1 is not None or calculation.custom_mass_2 is not None:
       
        log(f"\n Using atomic mass of {(masses[0] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[0].capitalize()}, {(masses[1] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[1].capitalize()}.", calculation, 1)
    
    else:
       
        log(" Using masses of most abundant isotopes.", calculation, 1)
    
    log(" Dipole moment derivative already includes vibrational overlap.\n", calculation, 1)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("           Harmonic Frequency                         Transition Intensity", calculation, 1, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log(f"  Force constant: {k:.5f}                    Dipole moment derivative: {dipole_derivative:.5f}", calculation, 1)
    log(f"  Reduced mass: {reduced_mass:7.2f}                      Squared derivative: {dipole_derivative_squared:.5f}", calculation, 1)
    log(f"\n  Frequency (per cm): {frequency_per_cm:.2f}{ i}                Intensity (km per mol): {transition_intensity_km_per_mol:.2f}", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    # Prints thermochemical information unless terse keyword is used 
    log(f"\n Temperature used is {temperature:.2f} K, pressure used is {(pressure)} Pa.", calculation, 2)
    log(" Entropies multiplied by temperature to give units of energy.", calculation, 2)
    log(f" Using symmetry number derived from {point_group} point group for rotational entropy.", calculation, 2)

    # Calculates rotational constant for thermochemical calculations
    rotational_constant_per_cm, _ = postscf.calculate_rotational_constant(masses, coordinates)

    # Calculates and prints thermochemical corrections
    thermo.calculate_thermochemical_corrections(calculation, frequency_per_cm, point_group, rotational_constant_per_cm, masses, temperature, pressure, energy, ZPE)

    return k, reduced_mass, frequency_per_cm