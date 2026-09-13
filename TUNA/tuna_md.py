import TUNA.tuna_opt as opt
import TUNA.tuna_energy as energ
import numpy as np
from numpy import ndarray
from TUNA.tuna_util import *
import TUNA.tuna_out as out
from TUNA.tuna_molecule import Molecule
from TUNA.tuna_calc import Calculation


"""

This is the TUNA module for ab initio molecular dynamics, written first for version 0.5.0 and rewritten for version 0.10.0.

The implementation of molecular dynamics here is Born-Oppenheimer molecular dynamics - where the nuclei are treated classically for the VelocityVerlet integration step,
which uses Newton's second law, but the forces are calculated quantum mechanically. These forces can be calculated by numerical differentiation with any implemented
electronic structure method. The molecule is free to rotate in three-dimensional space. As having a constant temperature would explode a diatomic molecule, all MD calculations
run in the NVE ensemble, although a starting temperature can be defined to give the molecule random initial velocities, satisfying the Maxwell-Boltzmann distribution. The
net linear momentum is removed in these cases, to prevent the molecule flying off, although it is allowed to rotate (because this is fun to look at).

The module contains:

1. Mathematical functions called within the MD loop (eg. calculate_accelerations, calculate_kinetic_energy, etc.).
2. Helper functions called within the MD loop (eg. calculate_nuclear_energy_components, print_molecular_dynamics_energy_components, etc.).
3. The main function to run molecular dynamics simulations, run_molecular_dynamics_simulation, which is called from the main tuna module.

"""





def calculate_accelerations(forces: ndarray, masses: ndarray) -> ndarray:

    """

    Calculates the acceleration vectors via Newton's second law.

    Args:
        forces (array): Force vector for both atoms
        masses (array): Masses array for both atoms

    Returns:
        accelerations (array): Acceleration vector for both atoms

    """

    inv_masses = 1 / masses

    accelerations = np.einsum("ij,i->ij", forces, inv_masses, optimize = True)

    return accelerations










def calculate_kinetic_energy(masses: ndarray, velocities: ndarray) -> float:

    """

    Calculates the classical nuclear kinetic energy.

    Args:
        masses (array): Mass array
        velocities (array): Velocity vectors for both atoms

    Returns:
        kinetic_energy (array): Classical nuclear kinetic energy

    """

    kinetic_energy = (1 / 2) * np.einsum("i,ij->", masses, velocities ** 2, optimize = True)

    return kinetic_energy










def calculate_temperature(masses: ndarray, velocities: ndarray, degrees_of_freedom: int) -> float:

    """

    Calculates the temperature from the kinetic energy.

    Args:
        masses (array): Mass array
        velocities (array): Velocity vectors for both atoms
        degrees_of_freedom (int): Number of degrees of freedom

    Returns:
        temperature (float): Temperature in kelvin

    """

    temperature = 2 * calculate_kinetic_energy(masses, velocities) / (degrees_of_freedom * constants.k)

    return temperature










def calculate_initial_velocities(masses: ndarray, requested_temperature: float, degrees_of_freedom: int) -> ndarray:

    """

    Calculates the initial velocities in line with the Maxwell-Boltzmann distribution.

    Args:
        masses (array): Mass array
        temperature (float): Temperature in kelvin
        degrees_of_freedom (int): Number of degrees of freedom

    Returns:
        initial_velocities (array): Randomly generated initial velocity vectors

    """

    # Calculates initial velocities to match Maxwell-Boltzmann distribution

    initial_velocities = np.einsum("i,ij->ij", np.sqrt(constants.k * requested_temperature / masses), np.random.normal(0, 1, (2, 3)), optimize = True)

    if requested_temperature > 0:

        # Removes net linear momentum

        linear_momentum = np.einsum("i,ij->j", masses, initial_velocities, optimize = True)
        initial_velocities -= linear_momentum / np.sum(masses)

        # Calculates new temperature from kinetic energies after linear momentum has been removed

        temperature = calculate_temperature(masses, initial_velocities, degrees_of_freedom)

        # Rescales velocities to match requested temperature

        initial_velocities *= np.sqrt(requested_temperature / temperature)

    return initial_velocities










def calculate_forces(coordinates: ndarray, calculation: Calculation, atomic_symbols: list) -> ndarray:

    """

    Calculates the 3D force vectors for both atoms.

    Args:
        coordinates (array): Atomic coordinates in 3D
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols

    Returns:
        forces (array): Force vectors for both atoms in 3D bas

    """

    # Align for the QM gradient calculation to ensure SCF convergence

    aligned_coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, calculate_bond_length(coordinates)]])
    
    # Calculate scalar gradient using aligned coordinates

    force = opt.calculate_gradient(aligned_coordinates, calculation, atomic_symbols, silent=True)
    
    # Project scalar force onto the TRUE 3D difference vector

    difference_vector = coordinates[1] - coordinates[0]

    difference_vector *= -force / calculate_bond_length(coordinates)
    
    forces = np.array([difference_vector, -1 * difference_vector])

    return forces










def print_molecular_dynamics_energy_components(time: float, iteration: int, masses: ndarray, velocities: ndarray, starting_energy: float, degrees_of_freedom: int, electronic_energy: float, calculation: Calculation, molecule: Molecule) -> None:

    """

    Prints information about the current iteration of a molecular dynamics simulation.

    Args:
        time (float): Time in femtoseconds
        iteration (int): Molecular dynamics iteration
        masses (array): Masses for both atoms
        velocities (array): Velocity vectors for both atoms
        starting_energy (float): Energy at beginning of MD simulation
        degrees_of_freedom (int): Number of degrees of freedom
        electronic_energy (float): Total electronic energy
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object

    """

    # Potential energy of the nuclei is the total electronic energy, kinetic energy is calculated classically

    kinetic_energy = calculate_kinetic_energy(masses, velocities)

    total_energy = kinetic_energy + electronic_energy

    temperature = calculate_temperature(masses, velocities, degrees_of_freedom)

    # Unphysical change in total energy over course of simulation (lower for lower timestep)

    drift = total_energy - starting_energy

    log(f" {(iteration + 1):4.0f}    {time:5.2f}     {bohr_to_angstrom(molecule.bond_length):.4f}    {temperature:10.2f}     {electronic_energy:12.6f}   {kinetic_energy:12.6f}     {total_energy:12.6f}   {drift:12.6f}", calculation, 1)


    return










def run_molecular_dynamics_simulation(calculation: Calculation, atomic_symbols: list, coordinates: ndarray) -> None:

    """

    Runs a Born-Oppenheimer molecular dynamics simulation of a given diatomic molecule.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates for both atoms

    """

    time = 0.0

    # Linear molecules lose one rotational degree of freedom

    degrees_of_freedom = 3

    # Convert to atomic units from femtoseconds for integration

    timestep_fs = calculation.step if calculation.step is not None else 0.1

    timestep_au = timestep_fs / constants.atomic_time_in_femtoseconds

    log(f"\nBeginning TUNA molecular dynamics calculation with {calculation.number_of_steps} steps in the NVE ensemble...\n", calculation, 1)
    log(f"Using timestep of {timestep_fs:.3f} femtoseconds and initial temperature of {calculation.temperature:.2f} K.", calculation, 1)

    # Prints trajectory to XYZ file by default, unless "NOTRAJ" keyword used

    if calculation.trajectory:

        log(f"Printing trajectory data to \"{calculation.trajectory_path}\".", calculation, 1)

        # Clears and recreates output file

        open(calculation.trajectory_path, "w").close()

    log_big_spacer(calculation, start = "\n")
    log("                                  Ab Initio Molecular Dynamics Simulation", calculation, 1, colour = "white")
    log_big_spacer(calculation)
    log("  Step    Time    Distance    Temperature    Pot. Energy     Kin. Energy        Energy          Drift", calculation, 1)
    log_big_spacer(calculation)

    # Remains silent to prevent too much printing, just prints to table

    SCF_output, molecule, electronic_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent = True)

    # Calculates inverse mass array for acceleration calculation

    masses = molecule.masses

    velocities = calculate_initial_velocities(masses, calculation.temperature, degrees_of_freedom)

    # Calculates forces without rotation, so uses identity matrix as rotation matrix

    forces = calculate_forces(coordinates, calculation, atomic_symbols)

    accelerations = calculate_accelerations(forces, masses)

    # Total energy of molecule is nuclear potential energy (electronic total energy) and classically calculated kinetic energy

    initial_energy = electronic_energy + calculate_kinetic_energy(masses, velocities)

    # Calculates various energy components and MD quantities, then prints these

    print_molecular_dynamics_energy_components(0, 0, masses, velocities, initial_energy, degrees_of_freedom, electronic_energy, calculation, molecule)

    P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None

    # Iterates over MD steps, up to the number of steps specified, in MD simulation

    for iteration in range(1, calculation.number_of_steps):

        # Velocity Verlet algorithm with finite timestep, accelerations are recalculated halfway through to allow simultaneous calculation of velocities

        coordinates += velocities * timestep_au + (1 / 2) * accelerations * timestep_au ** 2

        # Optional (default) reading in of orbitals from previous MD step - turn off with "NOMOREAD"

        if calculation.MO_read:

            P_guess = SCF_output.P
            P_guess_alpha = SCF_output.P_alpha
            P_guess_beta = SCF_output.P_beta
            E_guess = SCF_output.energy

        # Rotate the difference vector so it lies along the z axis only

        aligned_coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, calculate_bond_length(coordinates)]])

        # Additional print makes a big mess - prints all energy calculations to console

        SCF_output, molecule, electronic_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, aligned_coordinates, P_guess=P_guess, E_guess=E_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, silent=not(calculation.additional_print))

        forces = calculate_forces(coordinates, calculation, atomic_symbols)

        accelerations_new = calculate_accelerations(forces, masses)

        velocities += (1 / 2) * timestep_au * (accelerations + accelerations_new)

        # Updates accelerations and increments timestep

        accelerations = accelerations_new
        time += timestep_fs

        # Prints out the energy components for the current iteration

        print_molecular_dynamics_energy_components(time, iteration, masses, velocities, initial_energy, degrees_of_freedom, electronic_energy, calculation, molecule)

        # By default prints trajectory to file, can be viewed with visualisation programs - turn this off with "NOTRAJ"

        if calculation.trajectory:

            out.save_trajectory_to_file(molecule, electronic_energy, coordinates, calculation.trajectory_path)


    log_big_spacer(calculation)

    return
