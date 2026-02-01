from tuna_util import log, log_big_spacer, error, warning, constants, angstrom_to_bohr, bohr_to_angstrom, log_spacer
import tuna_energy as energ
import tuna_out as out
import tuna_opt as opt
import tuna_thermo as thermo
from scipy import linalg, interpolate
import numpy as np
import sys


"""

This is the TUNA module for calculating harmonic and anharmonic vibrational frequencies, written first for version 0.10.0 of TUNA.

Harmonic frequencies are calculated from the numerical Hessian at a particular bond length, and intensities are determined through numerical dipole moment
derivatives. Anharmonic frequencies are solved by numerical solution of the nuclear Schrodinger equation on a grid. The coordinates on either side of the 
optimised bond length are scanned, and the nuclear Hamiltonian diagonalised iteratively, until a large enough section of the potential energy surface has
been sampled such that the fundamental transition frequency is converged to within a hundreth of a per cm. Then, the absorption spectrum is printed.

The module contains:

1. Some utility functions that are shared between harmonic and anharmonic calculations (e.g. calculate_transition_intensity)
2. Useful functions for the anharmonic frequency calculations (e.g. diagonalise_hamiltonian_tridiagonal, frequency_is_converged, calculate_dipole_matrix)
3. The main function for the anharmonic frequency calculation, solve_nuclear_schrodinger_equation
4. The main function for the harmonic frequency calculation, calculate_harmonic_frequency

"""





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








def check_sign_of_hessian(hessian, reduced_mass) -> tuple:

    """

    Checks if the Hessian is positive or negative, and processes the frequency and zero-point energy accordingly.

    Args:
        hessian (float): Hessian matrix
        reduced_mass (float): Reduced mass in au
    
    Returns:
        frequency_hartree (float): Harmonic frequency
        zero_point_energy (float): Harmonic zero-point energy

    """

    # Checks if an imaginary mode is present, and if so zero-point energy is set to zero
    if hessian > 0:
    
        frequency_hartree = np.sqrt(hessian / reduced_mass)

        zero_point_energy = frequency_hartree / 2
        
    else:   
    
        frequency_hartree = np.sqrt(-hessian / reduced_mass)

        zero_point_energy = 0
        
        warning("Imaginary frequency calculated! Zero-point energy and vibrational thermochemical parameters set to zero!\n")

    return frequency_hartree, zero_point_energy








def calculate_anharmonicity_constant(transition_matrix, harmonic_frequency) -> float:

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








def interpolate_and_build_hamiltonian(x_values, V_values, reduced_mass, extent, extrapolation_grid_density, dipole_moments) -> tuple:

    """

    Interpolates the potential energy and dipole moment surface, solves the eigenvalue problem for the nuclear Hamiltonian.

    Args:
        x_values (array): Bond length values
        V_values (array): Raw potential energy values
        reduced_mass (float): Reduced mass
        extent (float): Extent of scan progress
        extrapolation_grid_density (int): How many extrapolation points per computed grid point
        dipole_moments (array): Dipole moments along scan coordinate

    Returns:
        vibrational_energy_levels (array): Nuclear Hamiltonian eigenvalues
        vibrational_wavefunctions (array): Nuclear Hamiltonian eigenvectors
        dipole_moments_interpolated (array): Interpolated dipole moments
        x (array): Interpolated bond lengths

    """

    n_interpolation_points = extrapolation_grid_density * extent

    # Interpolates the potential energies and dipole moments using the grid density multiplied by the extent, to give a consistent number of interpolation points
    x, V = interpolate_potential_energy(V_values, x_values, n_interpolation_points)
    _, dipole_moments_interpolated = interpolate_potential_energy(dipole_moments, x_values, n_interpolation_points)

    # Builds the main and off diagonal parts of the nuclear Hamiltonian
    main_diag, off_diag = construct_nuclear_hamiltonian(x, V, reduced_mass)

    # Diagonalises the nuclear Hamiltonian
    vibrational_energy_levels, vibrational_wavefunctions = diagonalise_hamiltonian_tridiagonal(main_diag, off_diag)

    return vibrational_energy_levels, vibrational_wavefunctions, dipole_moments_interpolated, x








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










def interpolate_potential_energy(F_raw, x_raw, n_grid_points) -> tuple:

    """

    Uses cubic splines to interpolate a function of coordinates.

    Args:
        F_raw (array): Function of nuclear coordinates to interpolate
        x_raw (array): Nuclear coordinate array before interpolation
        n_grid_points (int): Number of interpolation grid points

    Returns:
        x (array): Interpolated nuclear coordinate array
        F (array): Interpolated function of coordinates

    """

    # Builds a linearly spaced x-axis between the computed end points, to the desired grid density
    x = np.linspace(x_raw.min(), x_raw.max(), int(n_grid_points))

    # Cubic interpolation of the potential energy surface
    interpolation_function = interpolate.interp1d(x_raw, F_raw, kind="cubic")
    F = interpolation_function(x)

    return x, F








def diagonalise_hamiltonian_tridiagonal(main_diag, off_diag, n_states=6) -> tuple:

    """

    Diagonalises a tridiagonal Hamiltonian matrix.

    Args:
        main_diag (array): Main diagonal of the Hamiltonian
        off_diag (array): Off-diagonal of the Hamiltonian
        n_states (int, optional): How many vibrational states to be calculated
    
    Returns:
        vibrational_energy_levels (array): Vibrational energy levels (eigenvalues)
        vibrational_wavefunctions (array): Vibrational wavefunctions (eigenvectors)

    """

    # This method is much faster and more memory efficient for tridiagonal matrices than forming the full matrix and diagonalising it
    vibrational_energy_levels, vibrational_wavefunctions = linalg.eigh_tridiagonal(main_diag, off_diag, select="i", select_range=(0, n_states - 1))

    return vibrational_energy_levels, vibrational_wavefunctions









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









def frequency_is_converged(frequency, frequency_old) -> bool:

    """

    Checks the convergence of the fundamental anharmonic frequency, to 0.01 per cm.

    Args:
        frequency (float): Frequency from current step
        frequency_old (float): Frequency from previous step

    Returns:
        frequency_is_converged (bool): Is the frequency converged

    """

    if np.abs(frequency - frequency_old) < 0.01:

        return True
    
    return False









def process_anharmonic_output(calculation, x_values, V_values, vibrational_wavefunctions, vibrational_energy_levels, transition_matrix, chi, dipole_moments_interpolated, x, molecule) -> None:

    """

    Processes a converged anharmonic frequency calculation and prints output.

    Args:
        calculation (Calculation): Calculation object
        x_values (array): Raw bond length values
        V_values (array): Raw potential energy values
        vibrational_wavefunctions (array): Vibrational wavefunctions
        vibrational_energy_levels (array): Nuclear Hamiltonian eigenvalues
        transition_matrix (array): Matrix of transition energies between states
        chi (float): Anharmonicity constant
        dipole_moments_interpolated (array): Interpolated dipole moments
        x (array): Interpolated bond lengths
        molecule (Molecule): Molecule object

    """

    zero_point_energy = vibrational_energy_levels[0] - min(V_values)

    # Uses unit conversions to get the transition matrix in per cm and in nm
    frequency_matrix = transition_matrix * constants.per_cm_in_hartree
    wavelength_matrix = 10000000 / np.where(frequency_matrix != 0, frequency_matrix, 1)

    log(f"\n Final fundamental frequency (per cm):  {frequency_matrix[0][1]:6.2f}", calculation, 1)
    log(f" Final anharmonicity constant:  {chi:7.5f}", calculation, 1)

    log(f"\n Zero-point energy:   {zero_point_energy:13.10f}", calculation, 1)
    log(f" Equilibrium energy:  {vibrational_energy_levels[0]:13.10f}", calculation, 1)
    
    # Calculates the matrix of transition dipole moments between nuclear vibrational states
    dipole_matrix = calculate_dipole_matrix(vibrational_wavefunctions, dipole_moments_interpolated)

    # Converts the dipole transition moments into intensities of transitions
    intensity_matrix = calculate_transition_intensity(frequency_matrix, dipole_matrix)

    # Prints the anharmonic absorption spectrum
    print_absorption_spectrum(calculation, transition_matrix, frequency_matrix, wavelength_matrix, intensity_matrix)

    if calculation.additional_print:

        thermo.calculate_thermochemical_corrections(molecule, calculation, transition_matrix[0][1], vibrational_energy_levels[0], zero_point_energy)

    # Plots the vibrational wavefunctions and nuclear potential energy, if requested by "PLOTVIB" keyword
    if calculation.plot_vibrational_wavefunctions:

        out.plot_vibrational_wavefunctions(x, vibrational_energy_levels, vibrational_wavefunctions, x_values, V_values)

    return










def solve_nuclear_schrodinger_equation(calculation, atomic_symbols, harmonic_frequency_per_cm, molecule) -> np.ndarray:

    """

    Solves the nuclear Schrodinger equation on a grid increasing the coordinate domain iteratively, until convergence is reached.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        harmonic_frequency_per_cm (array): Harmonic frequency in per cm
        molecule (Molecule): Molecule object

    Returns:
        vibrational_energy_levels (array): Eigenvalues from nuclear Hamiltonian

    """

    # This value is more than enough for accuracy and extrapolation is not going to be the rate limiting step
    extrapolation_grid_density = 1000

    # Chooses a 0.05 angstrom step length if none has been chosen
    calculation.scan_step = 0.05 if calculation.scan_step is None else calculation.scan_step
    
    # The extent is the total distance (in angstroms) of the first scan around the minimum - half backwards, half forwards
    extent = 0.5

    # Initialises fundamental transition frequency for loop
    transition_per_cm = 0

    log_spacer(calculation, 1, start="\n", space="")
    log(" Beginning anharmonic frequency calculation...", calculation, 1, colour="white")
    log_spacer(calculation, 1, space="")

    log(f"\n Using a scan step length of {calculation.scan_step} angstroms.\n", calculation, 1)
    log(f" Using atomic mass of {(molecule.masses[0] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[0].capitalize()}, {(molecule.masses[1] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[1].capitalize()}.", calculation, 3)

    log(" Calculating initial potential energy surface around minimum... ", calculation, 1, end=""); sys.stdout.flush()

    # Determines how many scan steps are necessary, based on extent and step length
    calculation.scan_number = int(extent / calculation.scan_step) + 1

    # Starts from optimised bond lengths, moves back by half the "extent"
    coordinates, coordinates_right, coordinates_left = molecule.coordinates.copy(), molecule.coordinates.copy(), molecule.coordinates.copy()
    coordinates[1][2] -= angstrom_to_bohr(extent) / 2

    # Does the first scan over the minimum; gets the bond lengths, energies and dipole moments
    x_values, V_values, dipole_moments = energ.scan_coordinate(calculation, atomic_symbols, coordinates, silent=True)

    log("[Done]\n", calculation, 1)

    # Determines how many scan steps are necessary for the extensions - the division by three is arbitrary
    calculation.scan_number = int(extent / calculation.scan_step / 3) + 1

    log_big_spacer(calculation, 1)
    log("                                          Anharmonic Frequency", calculation, 1, colour="white")
    log_big_spacer(calculation, 1)
    log("  Step       Fundamental Freq. (per cm)         Chi        Harmonic Freq. (per cm)     Bond Length Range", calculation, 1)
    log_big_spacer(calculation, 1)
    
    for iteration in range(10):

        transition_per_cm_old = transition_per_cm

        # Updates the total extent based on the scanned distance so far, in bohr
        extent = max(x_values) - min(x_values)

        # Updates the left and rightmost coordinates, based on the scan endpoints so far
        coordinates_right[1][2] = np.max(x_values)
        coordinates_left[1][2] = np.min(x_values)
        
        # Performs the forward and backwards scans
        new_x_values_right, new_V_values_right, new_dipole_moments_right = energ.scan_coordinate(calculation, atomic_symbols, coordinates_right, silent=True)
        new_x_values_left, new_V_values_left, new_dipole_moments_left = energ.scan_coordinate(calculation, atomic_symbols, coordinates_left, silent=True, reverse=True)

        # Updates the bond length values and energies by concatenating the results from the left and right scans
        x_values = np.concatenate((np.array(new_x_values_left[1:][::-1]), np.array(x_values), np.array(new_x_values_right[1:])))
        V_values = np.concatenate((np.array(new_V_values_left[1:][::-1]), np.array(V_values), np.array(new_V_values_right[1:])))

        # Updates the dipole moments by concatenating the left and right scan results
        dipole_moments = np.concatenate((np.array(new_dipole_moments_left[1:][::-1]), np.array(dipole_moments), np.array(new_dipole_moments_right[1:])))
        
        # Interpolates the energy and dipole moments, and solves the eigenvalue equation 
        vibrational_energy_levels, vibrational_wavefunctions, dipole_moments_interpolated, x = interpolate_and_build_hamiltonian(x_values, V_values, molecule.reduced_mass, extent, extrapolation_grid_density, dipole_moments)

        transition_matrix = calculate_transition_matrix(vibrational_energy_levels)
        transition_per_cm = transition_matrix[0][1] * constants.per_cm_in_hartree

        # Calculates the anharmonicity constant
        chi = calculate_anharmonicity_constant(transition_matrix, harmonic_frequency_per_cm / constants.per_cm_in_hartree)

        log(f"    {iteration + 1}               {transition_per_cm:8.2f}                 {chi:8.5f}             {harmonic_frequency_per_cm:8.2f}             {bohr_to_angstrom(min(x_values)):.5f} - {bohr_to_angstrom(max(x_values)):.5f}", calculation, 1)

        if frequency_is_converged(transition_per_cm, transition_per_cm_old):
            
            log_big_spacer(calculation, 1)

            process_anharmonic_output(calculation, x_values, V_values, vibrational_wavefunctions, vibrational_energy_levels, transition_matrix, chi, dipole_moments_interpolated, x, molecule)

            return vibrational_energy_levels

    error("Anharmonic frequency calculation did not converge!")
    








def calculate_harmonic_frequency(calculation, atomic_symbols=None, coordinates=None, molecule=None, energy=None) -> tuple:

    """

    Calculates the harmonic frequency of a molecule.

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
    if calculation.calculation_type == "FREQ":
    
        _, molecule, energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates)

    # Unpacks useful molecular quantities
    point_group = molecule.point_group
    bond_length = molecule.bond_length
    atomic_symbols = molecule.atomic_symbols
    coordinates = molecule.coordinates
    masses = molecule.masses
    reduced_mass = molecule.reduced_mass
    rotational_constant_per_cm = molecule.rotational_constant_per_cm

    # Unpacks useful calculation quantities from user-defined parameters
    temperature = calculation.temperature
    pressure = calculation.pressure  

    log_spacer(calculation, 1, start="\n", space="")
    log(" Beginning harmonic frequency calculation...", calculation, 1, colour="white")
    log_spacer(calculation, 1, space="")
    
    log(f"\n Hessian will be calculated at a bond length of {bohr_to_angstrom(bond_length):.5f} angstroms.", calculation, 1)
    
    # Spring stiffness is calculated as the Hessian, through numerical second derivatives
    hessian, SCF_output_forward, P_forward, SCF_output_backward, P_backward = opt.calculate_hessian(coordinates, calculation, atomic_symbols, energy)

    # Checks if the Hessian is negative
    frequency_hartree, zero_point_energy = check_sign_of_hessian(hessian, reduced_mass)

    imaginary_unit = " i" if zero_point_energy == 0 else ""

    # Converts frequency into human units from atomic units
    frequency_per_cm = frequency_hartree * constants.per_cm_in_hartree

    # Calculates the dipole derivative, maintaining gauge invariance
    dipole_derivative = opt.calculate_dipole_derivative(coordinates, molecule, SCF_output_forward, SCF_output_backward, P_forward, P_backward)

    # Adding vibrational overlap contribution to match ORCA results (frequencies cancel for harmonic oscillator)
    dipole_derivative /= np.sqrt(2 * frequency_hartree)
    dipole_derivative_squared = dipole_derivative ** 2

    # Converts the dipole derivative into a transition intensity
    transition_intensity_km_per_mol = calculate_transition_intensity(frequency_per_cm, dipole_derivative)
       
    log(f" Using atomic mass of {(masses[0] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[0].capitalize()}, {(masses[1] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[1].capitalize()}.", calculation, 3)
    
    log(" Dipole moment derivative already includes vibrational overlap.\n", calculation, 1)

    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log("           Harmonic Frequency                         Transition Intensity", calculation, 1, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)
    log(f"  Force constant: {hessian:.5f}                    Dipole moment derivative: {dipole_derivative:.5f}", calculation, 1)
    log(f"  Reduced mass: {reduced_mass:7.2f}                      Squared derivative: {dipole_derivative_squared:.5f}", calculation, 1)
    log(f"\n  Frequency (per cm): {frequency_per_cm:.2f}{imaginary_unit}                Intensity (km per mol): {transition_intensity_km_per_mol:.2f}", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)


    # Calculates and prints thermochemical corrections
    thermo.calculate_thermochemical_corrections(molecule, calculation, frequency_hartree, energy, zero_point_energy)

    return hessian, reduced_mass, frequency_per_cm