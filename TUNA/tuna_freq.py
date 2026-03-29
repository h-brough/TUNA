from tuna_util import log, log_big_spacer, error, warning, constants, angstrom_to_bohr, bohr_to_angstrom, log_spacer, calculate_first_derivative, Output, calculate_fourth_derivative, calculate_third_derivative
import tuna_energy as energ
from tuna_calc import Calculation
import tuna_out as out
import tuna_props as props
import tuna_opt as opt
import tuna_thermo as thermo
from scipy import linalg, interpolate
import numpy as np
from numpy import ndarray
from tuna_molecule import Molecule
import sys


"""

This is the TUNA module for calculating harmonic and anharmonic vibrational frequencies, written first for version 0.10.0 of TUNA.

Harmonic frequencies are calculated from the numerical Hessian at a particular bond length, and intensities are determined through numerical dipole moment
derivatives. Anharmonic frequencies are solved by numerical solution of the nuclear Schrodinger equation on a grid. The coordinates on either side of the 
optimised bond length are scanned, and the nuclear Hamiltonian diagonalised iteratively, until a large enough section of the potential energy surface has
been sampled such that the fundamental transition frequency is converged to within a hundreth of a per cm. Then, the absorption spectrum is printed.

Updated in version 0.10.1 to include VPT2 approximate anharmonic frequencies.

The module contains:

1. Some utility functions that are shared between harmonic and anharmonic calculations (e.g. calculate_transition_intensity)
2. Useful functions for the anharmonic frequency calculations (e.g. diagonalise_hamiltonian_tridiagonal, frequency_is_converged, calculate_dipole_matrix)
3. The main function for the anharmonic frequency calculation, calculate_anharmonic_frequency
4. The main function for the harmonic frequency calculation, calculate_harmonic_frequency

"""



def calculate_transition_intensity(frequency_matrix: ndarray, dipole_matrix: ndarray) -> ndarray:
    
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










def check_sign_of_hessian(hessian: ndarray, reduced_mass: float) -> tuple:

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










def calculate_anharmonicity_constant(transition_matrix: ndarray, harmonic_frequency: float) -> float:

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










def calculate_dipole_derivative(coordinates: ndarray, molecule: Molecule, SCF_output_forward: Output, SCF_output_backward: Output, P_forward: ndarray, P_backward: ndarray):

    """

    Calculates the dipole derivative in normal coordinates.

    This is the numerical geometric derivative of the analytical dipole moment.

    Args:   
        coordinates (array): Atomic coordinates
        molecule (Molecule): Molecule object
        SCF_output_forward (Output): SCF output from prodded forward coordinates
        P_forward (array): Density matrix from prodded forward coordinates
        SCF_output_backward (Output): SCF output from prodded backward coordinates
        P_backward (array): Density matrix from prodded backward coordinates

    Returns:
        dipole_derivative (float): Dipole derivative in normal coordinates

    """

    # Forward and backward coordinates are symmetrical by the mass weighting, to prevent influence of moving electric field origin in dipole moment calculations

    prodding_coords = np.array([[0.0, 0.0, - molecule.masses[1] * constants.SECOND_GEOM_DERIVATIVE_PROD], [0.0, 0.0, molecule.masses[0] * constants.SECOND_GEOM_DERIVATIVE_PROD]]) / molecule.total_mass
    
    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords
    
    # Calculates forward and backward dipole moments, using dipole integrals calculated from centre of mass

    dipole_moment_forward, _, _ = props.calculate_analytical_dipole_moment(molecule.centre_of_mass, molecule.charges, forward_coords, P_forward, SCF_output_forward.integrals.D)
    dipole_moment_backward, _, _ = props.calculate_analytical_dipole_moment(molecule.centre_of_mass, molecule.charges, backward_coords, P_backward, SCF_output_backward.integrals.D)

    # Calculates dipole derivative by central differences method

    dipole_derivative = calculate_first_derivative(dipole_moment_backward, dipole_moment_forward, constants.SECOND_GEOM_DERIVATIVE_PROD)
    
    # Converts to normal coordinates by mass weighting

    dipole_derivative /= np.sqrt(molecule.reduced_mass)


    return dipole_derivative










def calculate_dipole_matrix(vibrational_wavefunctions: ndarray, dipole_moments_interpolated: ndarray) -> ndarray:

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










def calculate_transition_matrix(vibrational_energy_levels: ndarray) -> ndarray:

    """

    Calculates the transition matrix between vibrational energy levels.

    Args:
        vibrational_energy_levels (array): Vibrational energy levels

    Returns:
        transition_matrix (array): Transition matrix between vibrational energy levels

    """

    # This allows easy indexing for the transitions ([0][1] is the transition from level 0 to level 1, etc.)

    transition_matrix = np.abs(vibrational_energy_levels[:, None] - vibrational_energy_levels[None, :])

    return transition_matrix










def interpolate_and_build_hamiltonian(x_values: ndarray, V_values: ndarray, reduced_mass: float, SCAN_EXTENT: float, EXTRAPOLATION_GRID_DENSITY: float, dipole_moments: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:

    """

    Interpolates the potential energy and dipole moment surface, solves the eigenvalue problem for the nuclear Hamiltonian.

    Args:
        x_values (array): Bond length values
        V_values (array): Raw potential energy values
        reduced_mass (float): Reduced mass
        SCAN_EXTENT (float): Extent of scan progress
        EXTRAPOLATION_GRID_DENSITY (int): How many extrapolation points per computed grid point
        dipole_moments (array): Dipole moments along scan coordinate

    Returns:
        vibrational_energy_levels (array): Nuclear Hamiltonian eigenvalues
        vibrational_wavefunctions (array): Nuclear Hamiltonian eigenvectors
        dipole_moments_interpolated (array): Interpolated dipole moments
        x (array): Interpolated bond lengths
        V (array): Interpolated potential energies

    """

    n_grid_points = int(EXTRAPOLATION_GRID_DENSITY * SCAN_EXTENT)

    # Interpolates the potential energies and dipole moments using the grid density multiplied by the extent, to give a consistent number of interpolation points

    x, V = interpolate_potential_energy(V_values, x_values, n_grid_points)
    _, dipole_moments_interpolated = interpolate_potential_energy(dipole_moments, x_values, n_grid_points)

    # Builds the main and off diagonal parts of the nuclear Hamiltonian

    main_diag, off_diag = construct_nuclear_hamiltonian(x, V, reduced_mass)

    # Diagonalises the nuclear Hamiltonian

    vibrational_energy_levels, vibrational_wavefunctions = diagonalise_hamiltonian_tridiagonal(main_diag, off_diag)

    return vibrational_energy_levels, vibrational_wavefunctions, dipole_moments_interpolated, x, V










def construct_nuclear_hamiltonian(x_interpolated: ndarray, V_interpolated: ndarray, reduced_mass: float) -> tuple[ndarray, ndarray]:

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










def interpolate_potential_energy(F_raw: ndarray, x_raw: ndarray, n_grid_points: int) -> tuple[ndarray, ndarray]:

    """

    Uses cubic splines to interpolate a function of coordinates.

    Args:
        F_raw (array): Function of nuclear coordinates to interpolate
        x_raw (array): Nuclear coordinate array before interpolation
        n_grid_points (int): Number of interpolation grid points

    Returns:
        x (array): Interpolated nuclear coordinate array
        function_interpolated (array): Interpolated function of coordinates

    """

    # Builds a linearly spaced x-axis between the computed end points, to the desired grid density

    x = np.linspace(x_raw.min(), x_raw.max(), n_grid_points)

    # Cubic interpolation of the potential energy surface

    interpolation_function = interpolate.interp1d(x_raw, F_raw, kind="cubic")
    function_interpolated = interpolation_function(x)

    return x, function_interpolated










def diagonalise_hamiltonian_tridiagonal(main_diag: ndarray, off_diag: ndarray, n_states=6) -> tuple[ndarray, ndarray]:

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










def print_absorption_spectrum(calculation: Calculation, transition_matrix: ndarray, frequency_matrix: ndarray, wavelength_matrix: ndarray, intensity_matrix: ndarray) -> None:

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










def frequency_is_converged(frequency: float, frequency_old: float, calculation: Calculation) -> bool:

    """

    Checks the convergence of the fundamental anharmonic frequency, to 0.01 per cm by default.

    Args:
        frequency (float): Frequency from current step in per cm
        frequency_old (float): Frequency from previous step in per cm
        calculation (Calculation): Calculation object

    Returns:
        frequency_is_converged (bool): Is the frequency converged

    """

    if np.abs(frequency - frequency_old) < calculation.anharm_convergence:

        return True
    
    return False










def process_anharmonic_output(calculation: Calculation, vibrational_wavefunctions: ndarray, vibrational_energy_levels: ndarray, transition_matrix: ndarray, chi: float, dipole_moments_interpolated: ndarray, x: ndarray, V: ndarray, molecule: Molecule) -> None:

    """

    Processes a converged anharmonic frequency calculation and prints output.

    Args:
        calculation (Calculation): Calculation object
        vibrational_wavefunctions (array): Vibrational wavefunctions
        vibrational_energy_levels (array): Nuclear Hamiltonian eigenvalues
        transition_matrix (array): Matrix of transition energies between states
        chi (float): Anharmonicity constant
        dipole_moments_interpolated (array): Interpolated dipole moments
        x (array): Interpolated bond lengths
        V (array): Interpolated potential energies
        molecule (Molecule): Molecule object

    """

    zero_point_energy = vibrational_energy_levels[0] - min(V)

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

        out.plot_vibrational_wavefunctions(calculation, bohr_to_angstrom(x), V, vibrational_energy_levels, vibrational_wavefunctions)

    return










def calculate_anharmonic_frequency(calculation: Calculation, atomic_symbols: list[str], harmonic_frequency_per_cm: float, molecule: Molecule) -> ndarray:

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

    EXTRAPOLATION_GRID_DENSITY = 1000

    # The extent is the total distance (in angstroms) of the first scan around the minimum - half backwards, half forwards

    SCAN_EXTENT = 0.35

    # Chooses a 0.05 angstrom step length if none has been chosen

    calculation.step = 0.05 if calculation.step is None else calculation.step
    
    # Initialises fundamental transition frequency for loop

    transition_per_cm = 0

    log_spacer(calculation, 1, start="\n", space="")
    log(" Beginning anharmonic frequency calculation...", calculation, 1, colour="white")
    log_spacer(calculation, 1, space="")

    log(f"\n Using a scan step length of {calculation.step} angstroms.\n", calculation, 1)
    log(f" Using atomic mass of {(molecule.masses[0] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[0].capitalize()}, {(molecule.masses[1] / constants.atomic_mass_unit_in_electron_mass):.6f} amu for {atomic_symbols[1].capitalize()}.", calculation, 3)

    log(" Calculating initial potential energy surface around minimum...  ", calculation, 1, end=""); sys.stdout.flush()

    # Determines how many scan steps are necessary, based on INITIAL_SCAN_EXTENT and step length

    calculation.number_of_steps = int(SCAN_EXTENT / calculation.step) + 1

    # Starts from optimised bond lengths, moves back by half the "INITIAL_SCAN_EXTENT"

    coordinates, coordinates_right, coordinates_left = molecule.coordinates.copy(), molecule.coordinates.copy(), molecule.coordinates.copy()
    coordinates[1][2] -= angstrom_to_bohr(SCAN_EXTENT) / 2

    # Does the first scan over the minimum; gets the bond lengths, energies and dipole moments

    x_values, V_values, dipole_moments = energ.scan_coordinate(calculation, atomic_symbols, coordinates, silent=True)

    log("[Done]\n", calculation, 1)

    # Determines how many scan steps are necessary for the extensions - the division by three is arbitrary

    calculation.number_of_steps = int(SCAN_EXTENT / calculation.step / 3) + 1

    log_big_spacer(calculation, 1)
    log("                                          Anharmonic Frequency", calculation, 1, colour="white")
    log_big_spacer(calculation, 1)
    log("  Step       Fundamental Freq. (per cm)         Chi        Harmonic Freq. (per cm)     Bond Length Range", calculation, 1)
    log_big_spacer(calculation, 1)
    
    for iteration in range(30):

        transition_per_cm_old = transition_per_cm

        # Updates the total extent based on the scanned distance so far, in bohr

        SCAN_EXTENT = max(x_values) - min(x_values)

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

        vibrational_energy_levels, vibrational_wavefunctions, dipole_moments_interpolated, x, V = interpolate_and_build_hamiltonian(x_values, V_values, molecule.reduced_mass, SCAN_EXTENT, EXTRAPOLATION_GRID_DENSITY, dipole_moments)

        transition_matrix = calculate_transition_matrix(vibrational_energy_levels)
        transition_per_cm = transition_matrix[0][1] * constants.per_cm_in_hartree

        # Calculates the anharmonicity constant

        chi = calculate_anharmonicity_constant(transition_matrix, harmonic_frequency_per_cm / constants.per_cm_in_hartree)

        log(f"    {iteration + 1}               {transition_per_cm:8.2f}                 {chi:8.5f}             {harmonic_frequency_per_cm:8.2f}             {bohr_to_angstrom(min(x_values)):.5f} - {bohr_to_angstrom(max(x_values)):.5f}", calculation, 1)

        if frequency_is_converged(transition_per_cm, transition_per_cm_old, calculation):
            
            log_big_spacer(calculation, 1)

            process_anharmonic_output(calculation, vibrational_wavefunctions, vibrational_energy_levels, transition_matrix, chi, dipole_moments_interpolated, x, V, molecule)

            return vibrational_energy_levels

    error("Anharmonic frequency calculation did not converge!")
    









def calculate_harmonic_frequency(calculation: Calculation, atomic_symbols: list[str] = None, coordinates: ndarray = None, molecule: Molecule = None, energy: float = None) -> tuple[ndarray, float, float]:

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
        zero_point_energy (float): Zero point energy in hartree

    """

    # If "FREQ" keyword has been used, calculates the energy using the supplied atoms and coordinates, otherwise uses the supplied molecule and energy

    if calculation.calculation_type == "FREQ":
    
        _, molecule, energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates)


    if calculation.perturbative_anharmonic:

        # If VPT2 is requested, we need the second and third derivative prods to be identical - this only marginally reduces second derivative quality

        constants.SECOND_GEOM_DERIVATIVE_PROD = constants.THIRD_GEOM_DERIVATIVE_PROD
    

    bond_length = molecule.bond_length
    atomic_symbols = molecule.atomic_symbols
    coordinates = molecule.coordinates
    masses = molecule.masses
    reduced_mass = molecule.reduced_mass

    log_spacer(calculation, 1, start="\n", space="")
    log(" Beginning harmonic frequency calculation...", calculation, 1, colour="white")
    log_spacer(calculation, 1, space="")
    
    log(f"\n Hessian will be calculated at a bond length of {bohr_to_angstrom(bond_length):.5f} angstroms.", calculation, 1)
    
    # Spring stiffness is calculated as the Hessian, through numerical second derivatives

    hessian, SCF_output_forward, P_forward, SCF_output_backward, P_backward, displaced_energies = opt.calculate_hessian(coordinates, calculation, atomic_symbols, energy)

    # Checks if the Hessian is negative

    frequency_hartree, zero_point_energy = check_sign_of_hessian(hessian, reduced_mass)

    imaginary_unit = "i" if zero_point_energy == 0 else " "

    # Converts frequency into human units from atomic units

    frequency_per_cm = frequency_hartree * constants.per_cm_in_hartree

    # Calculates the dipole derivative, maintaining gauge invariance

    dipole_derivative = calculate_dipole_derivative(coordinates, molecule, SCF_output_forward, SCF_output_backward, P_forward, P_backward)

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
    log(f"  Force constant:           {hessian:10.5f}       Dipole moment derivative:  {dipole_derivative:10.5f}", calculation, 1)
    log(f"  Reduced mass:           {reduced_mass:12.5f}       Squared derivative:        {dipole_derivative_squared:10.5f}", calculation, 1)
    log(f"\n  Frequency (per cm):         {imaginary_unit}{frequency_per_cm:7.2f}       Intensity (km per mol):       {transition_intensity_km_per_mol:7.2f}", calculation, 1)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1)

    # Checks if the "VPT2" keyword has been used

    if calculation.perturbative_anharmonic:

        frequency_hartree, zero_point_energy = calculate_VPT2_frequency(frequency_hartree, energy, calculation, atomic_symbols, coordinates, molecule, displaced_energies)
    
    # Calculates and prints thermochemical corrections

    thermo.calculate_thermochemical_corrections(molecule, calculation, frequency_hartree, energy, zero_point_energy)


    return hessian, reduced_mass, frequency_per_cm, zero_point_energy










def calculate_VPT2_frequency(frequency_hartree: float, energy: float, calculation: Calculation, atomic_symbols: list, coordinates: ndarray, molecule: Molecule, displaced_energies: tuple) -> tuple:
 
    """
    
    Calculates the second-order vibrational perturbation theory fundamental frequency.

    Args:
        frequency_hartree (float): Harmonic frequency in hartree
        energy (float): Molecular energy
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        molecule (Molecule): Molecule object
        displaced_energies (tuple): Energies from harmonic frequency calculation
    
    Returns:
        fundamental_frequency (float): Perturbative fundamental vibrational frequency
        zero_point_energy (float): Anharmonic perturbative Zero-point energy
    
    """

    log("\n Initialising second-order vibrational perturbation theory..   \n", calculation)

    log_spacer(calculation)
    log("              VPT2 Frequency Correction", calculation)
    log_spacer(calculation)
    
    log(f"  Using finite difference of {constants.THIRD_GEOM_DERIVATIVE_PROD} a.u.   \n", calculation)

    prodding_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, constants.THIRD_GEOM_DERIVATIVE_PROD]])

    super_far_backward_coords = coordinates - 4 * prodding_coords
    very_far_backward_coords = coordinates - 3 * prodding_coords
    very_far_forward_coords = coordinates + 3 * prodding_coords
    super_far_forward_coords = coordinates + 4 * prodding_coords
    
    # Can't use the previously calculated energies if a different derivative step was used

    if constants.THIRD_GEOM_DERIVATIVE_PROD != constants.SECOND_GEOM_DERIVATIVE_PROD:

        error("Mismatch in numerical derivatives for (an)harmonic frequency calculations!")

    energy_far_backward, energy_backward, energy_forward, energy_far_forward = displaced_energies

    log("  Calculating displaced energy 1 of 4...     ", calculation, end=""); sys.stdout.flush()

    _, _, energy_super_far_backward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, super_far_backward_coords, silent=True)
    
    log("[Done]", calculation)  

    log("  Calculating displaced energy 2 of 4...     ", calculation, end=""); sys.stdout.flush()

    _, _, energy_very_far_backward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, very_far_backward_coords, silent=True)

    log("[Done]", calculation)  

    log("  Calculating displaced energy 3 of 4...     ", calculation, end=""); sys.stdout.flush()

    _, _, energy_very_far_forward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, very_far_forward_coords, silent=True)
    
    log("[Done]", calculation)  

    log("  Calculating displaced energy 4 of 4...     ", calculation, end=""); sys.stdout.flush()

    _, _, energy_super_far_forward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, super_far_forward_coords, silent=True)
    
    log("[Done]", calculation)  

    # Calculates the third and fourth derivatives with the harmonic energies, and the four additional energies

    d3E_dR3 = calculate_third_derivative(energy_super_far_backward, energy_very_far_backward, energy_far_backward, energy_backward, energy_forward, energy_far_forward, energy_very_far_forward, energy_super_far_forward, constants.THIRD_GEOM_DERIVATIVE_PROD)
    
    d4E_dR4 = calculate_fourth_derivative(energy_super_far_backward, energy_very_far_backward, energy_far_backward, energy_backward, energy, energy_forward, energy_far_forward, energy_very_far_forward, energy_super_far_forward, constants.THIRD_GEOM_DERIVATIVE_PROD)

    # Distinct terms involving either the third or fourth derivative

    third_derivative_term = -1 * d3E_dR3 ** 2 / (molecule.reduced_mass ** 3 * frequency_hartree ** 4)
    fourth_derivative_term = d4E_dR4 / (molecule.reduced_mass ** 2 * frequency_hartree ** 2)

    # Calculates the anharmonicity parameter

    anharmonicity = (5 / 48) * third_derivative_term + (1 / 16) * fourth_derivative_term
    
    chi = -anharmonicity / frequency_hartree

    fundamental_frequency = frequency_hartree + 2 * anharmonicity
    first_overtone = 2 * frequency_hartree + 6 * anharmonicity


    # The new, anharmonic perturbative zero-point energy

    zero_point_energy = (1 / 2) * frequency_hartree + (1 / 32) * fourth_derivative_term + (11 / 288) * third_derivative_term

    equilibrium_energy = energy + zero_point_energy
    
    log(f"\n  Anharmonicity constant:                {chi:10.5f}", calculation)   

    log(f"\n  Zero-point energy:               {zero_point_energy:16.10f}", calculation)   
    log(f"  Equilibrium energy:              {equilibrium_energy:16.10f}", calculation)   

    log(f"\n  Fundamental frequency (per cm):        {fundamental_frequency * constants.per_cm_in_hartree:10.2f}", calculation)   
    log(f"  First overtone (per cm):               {first_overtone * constants.per_cm_in_hartree:10.2f}", calculation)   
    
    log_spacer(calculation)

    return fundamental_frequency, zero_point_energy