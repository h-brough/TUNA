import tuna_opt as opt
import tuna_energy as energ
import numpy as np
from numpy import ndarray
from tuna_util import *
import tuna_out as out
from tuna_molecule import Molecule


"""

This is the TUNA module for ab initio molecular dynamics, written first for version 0.5.0 and rewritten for version 0.10.0.

The implementation of molecular dynamics here is Born-Oppenheimer molecular dynamics - where the nuclei are treated classically for the VelocityVerlet integration step, 
which uses Newton's second law, but the forces are calculated quantum mechanically. These forces can be calculated by numerical differentiation with any implemented
electronic structure method. The molecule is free to rotate in three-dimensional space. To allow this, the molecule is rotated onto the z-axis for the energy evaluations
and then back-transformed to its original coordinates to continue its trajectory. As having a constant temperature would explode a diatomic molecule, all MD calculations
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

    accelerations = np.einsum("ij,i->ij", forces, inv_masses, optimize=True)

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

    kinetic_energy = (1 / 2) * np.einsum("i,ij->", masses, velocities ** 2, optimize=True)

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

    initial_velocities = np.einsum("i,ij->ij", np.sqrt(constants.k * requested_temperature / masses), np.random.normal(0, 1, (2, 3)), optimize=True)

    if requested_temperature > 0:

        # Removes net linear momentum

        linear_momentum = np.einsum("i,ij->j", masses, initial_velocities, optimize=True)
        initial_velocities -= linear_momentum / np.sum(masses)

        # Calculates new temperature from kinetic energies after linear momentum has been removed

        temperature = calculate_temperature(masses, initial_velocities, degrees_of_freedom)

        # Rescales velocities to match requested temperature

        initial_velocities *= np.sqrt(requested_temperature / temperature)

    return initial_velocities










def calculate_forces(coordinates: ndarray, calculation: Calculation, atomic_symbols: list, rotation_matrix: ndarray) -> ndarray:

    """

    Calculates the 3D force vectors for both atoms.

    Args:
        coordinates (array): Atomic coordinates in 3D
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        rotation_matrix (array): Rotation matrix to place molecule along z axis

    Returns:
        forces (array): Force vectors for both atoms in 3D

    """

    force = opt.calculate_gradient(coordinates, calculation, atomic_symbols, silent=True)

    force_array_1D = np.array([0.0, 0.0, force])

    # Uses rotation matrix to bring forces back to original coordinate system

    force_array_3D = force_array_1D @ rotation_matrix

    # Applies equal and opposite to other atom

    forces = np.array([force_array_3D, -1 * force_array_3D])

    return forces










def rotate_coordinates_to_z_axis(difference_vector: ndarray) -> tuple[ndarray, ndarray]:

    """

    Calculates axis of rotation and rotates difference vector using Rodrigues' formula.

    Args:   
        difference_vector (array): Difference vector between atoms

    Returns:
        difference_vector_rotated (array) : Rotated difference vector on z axis
        rotation_matrix (array) : Rotation matrix

    """

    normalised_vector = difference_vector / np.linalg.norm(difference_vector)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Calculate the axis of rotation by the cross product

    rotation_axis = np.cross(normalised_vector, z_axis)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-10:

        # If the axis is too small, the vector is almost aligned with the z-axis

        rotation_matrix = np.eye(3)

    else:

        # Normalize the rotation axis

        rotation_axis /= axis_norm
        
        # Calculate the angle of rotation by the dot product

        cos_theta = np.dot(normalised_vector, z_axis)
        sin_theta = axis_norm
        
        # Rodrigues' rotation formula

        K = np.array([[0.0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0.0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0.0]])
        
        rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    
    # Rotate the difference vector to align it with the z-axis

    difference_vector_rotated = np.dot(rotation_matrix, difference_vector)
    
    return difference_vector_rotated, rotation_matrix










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
    
    degrees_of_freedom = 5

    # Convert to atomic units from femtoseconds for integration

    timestep_fs = calculation.timestep

    timestep_au = timestep_fs / constants.atomic_time_in_femtoseconds

    log(f"\nBeginning TUNA molecular dynamics calculation with {calculation.MD_number_of_steps} steps in the NVE ensemble...\n", calculation, 1)
    log(f"Using timestep of {timestep_fs:.3f} femtoseconds and initial temperature of {calculation.temperature:.2f} K.", calculation, 1)

    # Prints trajectory to XYZ file by default, unless "NOTRAJ" keyword used

    if calculation.trajectory: 

        log(f"Printing trajectory data to \"{calculation.trajectory_path}\".", calculation, 1)

        # Clears and recreates output file

        open(calculation.trajectory_path, "w").close()

    log_big_spacer(calculation, start="\n")
    log("                                  Ab Initio Molecular Dynamics Simulation", calculation, 1, colour="white")
    log_big_spacer(calculation)
    log("  Step    Time    Distance    Temperature    Pot. Energy     Kin. Energy        Energy          Drift", calculation, 1)
    log_big_spacer(calculation)

    # Remains silent to prevent too much printing, just prints to table

    SCF_output, molecule, electronic_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, silent=True)

    # Calculates inverse mass array for acceleration calculation
    masses = molecule.masses

    # Calculates forces without rotation, so uses identity matrix as rotation matrix

    forces = calculate_forces(coordinates, calculation, atomic_symbols, np.eye(3))

    accelerations = calculate_accelerations(forces, masses) 

    velocities = calculate_initial_velocities(masses, calculation.temperature, degrees_of_freedom)

    # Total energy of molecule is nuclear potential energy (electronic total energy) and classically calculated kinetic energy

    initial_energy = electronic_energy + calculate_kinetic_energy(masses, velocities)

    # Calculates various energy components and MD quantities, then prints these

    print_molecular_dynamics_energy_components(0, 1, masses, velocities, initial_energy, degrees_of_freedom, electronic_energy, calculation, molecule)

    P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None

    # Iterates over MD steps, up to the number of steps specified, in MD simulation

    for iteration in range(1, calculation.MD_number_of_steps):

        # Velocity Verlet algorithm with finite timestep, accelerations are recalculated halfway through to allow simultaneous calculation of velocities

        coordinates += velocities * timestep_au + (1 / 2) * accelerations * timestep_au ** 2

        # Optional (default) reading in of orbitals from previous MD step - turn off with "NOMOREAD"

        if calculation.MO_read: 
            
            P_guess = SCF_output.P
            P_guess_alpha = SCF_output.P_alpha
            P_guess_beta = SCF_output.P_beta
            E_guess = SCF_output.energy

        # Defines a 3D vector of the differences between atomic positions to rotate to the z axis

        difference_vector = np.array([coordinates[0][0] - coordinates[1][0], coordinates[0][1] - coordinates[1][1], coordinates[0][2] - coordinates[1][2]])

        # Rotate the difference vector so it lies along the z axis only

        difference_vector_rotated, rotation_matrix = rotate_coordinates_to_z_axis(difference_vector)
        aligned_coordinates = np.array([[0.0, 0.0, 0.0], -1 * difference_vector_rotated])

        # Additional print makes a big mess - prints all energy calculations to console

        SCF_output, molecule, electronic_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, aligned_coordinates, P_guess=P_guess, E_guess=E_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, silent=not(calculation.additional_print))

        forces = calculate_forces(aligned_coordinates, calculation, atomic_symbols, rotation_matrix)

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


