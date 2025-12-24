import tuna_optfreq as optfreq
import tuna_energy as energ
import numpy as np
from tuna_util import *


def calculate_accelerations(forces, inv_masses): 

    """

    Calculates the acceleration vectors.

    Args:
        forces (array): Force vector for both atoms
        inv_masses (array): Inverted masses array for both atoms

    Returns:
        accelerations (array): Acceleration vector for both atoms

    """

    accelerations = np.einsum("ij,i->ij", forces, inv_masses, optimize=True)

    return accelerations







def calculate_kinetic_energy(masses, velocities): 
    
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






def calculate_temperature(masses, velocities, degrees_of_freedom):

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





def calculate_initial_velocities(masses, requested_temperature, degrees_of_freedom):

    """

    Calculates the initial velocities in line with the Maxwell-Boltzmann distribution.

    Args:
        masses (array): Mass array
        temperature (float): Temperature in kelvin
        degrees_of_freedom (int): Number of degrees of freedom

    Returns:
        initial_velocities (array): Randomly generated initial velocity vectors

    """

    # Calculates to match Maxwell-Boltzmann distribution
    initial_velocities = np.einsum("i,ij->ij", np.sqrt(constants.k * requested_temperature / masses), np.random.normal(0,1,(2,3)), optimize=True)

    if requested_temperature > 0:

        # Removes net linear momentum
        momenta = np.einsum("i,ij->j", masses, initial_velocities, optimize=True)
        initial_velocities -= momenta / np.sum(masses)

        # Calculates new temperature from kinetic energies after linear momentum has been removed
        temperature = calculate_temperature(masses, initial_velocities, degrees_of_freedom)

        # Rescales velocities
        initial_velocities *= np.sqrt(requested_temperature / temperature)

    return initial_velocities







def calculate_forces(coordinates, calculation, atoms, rotation_matrix):

    """

    Calculates the 3D force vectors for both molecules.

    Args:
        coordinates (array): Atomic coordinates in 3D
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        rotation_matrix (array): Rotation matrix to place molecule along z axis

    Returns:
        forces (array): Force vectors for both atoms in 3D

    """

    force = optfreq.calculate_gradient(coordinates, calculation, atoms, silent=True)

    force_array_1D = [0.0, 0.0, force]

    # Uses rotation matrix to bring forces back to original coordinate system
    force_array_3D = force_array_1D @ rotation_matrix

    # Applies equal and opposite to other atom
    forces = np.array([force_array_3D, -1 * force_array_3D])


    return forces










def rotate_coordinates_to_z_axis(difference_vector):

    """

    Calculates axis of rotation and rotates difference vector using Rodrigues' formula.

    Args:   
        difference_vector (array): Difference vector

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
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
        
        rotation_matrix = np.eye(3, dtype=np.float64) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    
    
    # Rotate the difference vector to align it with the z-axis
    difference_vector_rotated = np.dot(rotation_matrix, difference_vector)
    
    return difference_vector_rotated, rotation_matrix










def calculate_MD_components(molecule, masses, velocities, starting_energy, degrees_of_freedom, electronic_energy):

    """

    Calculates various energies and other quantities for use in an MD simulation.

    Args:
        molecule (Molecule): Molecule object
        masses (array): Masses for both atoms
        velocities (array): Velocity vectors for both atoms
        starting_energy (float): Energy at beginning of MD simulation
        degrees_of_freedom (int): Number of degrees of freedom
        electronic_energy (float): Total electronic energy

    Returns:
        potential_energy (float): Total electronic energy
        kinetic_energy (float): Classical nuclear kinetic energy
        total_energy (float): Sum of nuclear kinetic and total electronic energy
        temperature (float): Temperature of molecule
        bond_length (float): Bond length in angstroms
        drift (float): Difference between energy and initial energy

    """

    # Potential energy of the nuclei is the total electronic energy, kinetic energy is calculated classically
    potential_energy = electronic_energy
    kinetic_energy = calculate_kinetic_energy(masses, velocities)

    total_energy = kinetic_energy + potential_energy

    temperature = calculate_temperature(masses, velocities, degrees_of_freedom)
    bond_length = bohr_to_angstrom(molecule.bond_length)

    # Unphysical change in total energy over course of simulation (lower for lower timestep)
    drift = total_energy - starting_energy 

    return potential_energy, kinetic_energy, total_energy, temperature, bond_length, drift







def run_MD(calculation, atoms, coordinates):

    """

    Runs an ab initio molecular dynamics simulation of a given molecule.

    Args:
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        coordinates (array): Atomic coordinates for both atoms

    Returns:
        None: Nothing is returned

    """

    time = 0
    trajectory_path = calculation.trajectory_path

    # Linear molecules lose one rotational degree of freedom
    degrees_of_freedom = 5

    # Unpacks useful quantities from calculation object
    n_steps = calculation.MD_number_of_steps
    timestep = calculation.timestep
    initial_temperature = calculation.temperature

    # Convert to atomic units from femtoseconds for integration
    timestep_au = timestep / constants.atomic_time_in_femtoseconds

    log(f"\nBeginning TUNA molecular dynamics calculation with {n_steps} steps in the NVE ensemble...\n", calculation, 1)
    log(f"Using timestep of {timestep:.3f} femtoseconds and initial temperature of {initial_temperature:.2f} K.", calculation, 1)

    # Prints trajectory to XYZ file by default, unless NOTRAJ keyword used
    if calculation.trajectory: 

        log(f"Printing trajectory data to \"{trajectory_path}\".", calculation, 1)

        # Clears and recreates output file
        open(trajectory_path, "w").close()

    log_big_spacer(calculation, start="\n")
    log("                                  Ab Initio Molecular Dynamics Simulation", calculation, 1, colour="white")
    log_big_spacer(calculation)
    log("  Step    Time    Distance    Temperature    Pot. Energy     Kin. Energy        Energy          Drift", calculation, 1)
    log_big_spacer(calculation)

    # Remains silent to prevent too much printing, just prints to table
    if calculation.extrapolate:

        SCF_output, molecule, electronic_energy, _ = energ.extrapolate_energy(calculation, atoms, coordinates, silent=True)

    else:
        
        SCF_output, molecule, electronic_energy, _ = energ.calculate_energy(calculation, atoms, coordinates, silent=True)

    # Calculates inverse mass array for acceleration calculation
    masses = molecule.masses
    inv_masses = 1 / masses

    # Calculates forces without rotation, so uses identity matrix as rotation matrix
    forces = calculate_forces(coordinates, calculation, atoms, np.eye(3))
    accelerations = calculate_accelerations(forces, inv_masses) 
    velocities = calculate_initial_velocities(masses, initial_temperature, degrees_of_freedom)

    # Total energy of molecule is nuclear potential energy (electronic total energy) and classically calculated kinetic energy
    initial_energy = electronic_energy + calculate_kinetic_energy(masses, velocities)

    # Calculates various energy components and MD quantities, then prints these
    potential_energy, kinetic_energy, total_energy, temperature, bond_length, drift = calculate_MD_components(molecule, masses, velocities, initial_energy, degrees_of_freedom, electronic_energy)
    log(f"  {1:3.0f}    {time:5.2f}     {bond_length:.4f}    {temperature:10.2f}     {potential_energy:12.6f}   {kinetic_energy:12.6f}     {total_energy:12.6f}   {drift:12.6f}", calculation, 1)

    # Iterates over MD steps, up to the number of steps specified, in MD simulation
    for i in range(1, n_steps):

        # Velocity Verlet algorithm with finite timestep, accelerations are recalculated halfway through to allow simultaneous calculation of velocities
        coordinates += velocities * timestep_au + 0.5 * accelerations * timestep_au ** 2

        # Optional (default) reading in of orbitals from previous MD step
        if calculation.MO_read: 
            
            P_guess = SCF_output.P
            P_guess_alpha = SCF_output.P_alpha
            P_guess_beta = SCF_output.P_beta
            E_guess = SCF_output.energy

        else: 
            
            P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None

        # Defines a 3D vector of the differences between atomic positions to rotate to the z axis
        difference_vector = np.array([coordinates[0][0] - coordinates[1][0], 
                                      coordinates[0][1] - coordinates[1][1], 
                                      coordinates[0][2] - coordinates[1][2]])

        # Rotate the difference vector so it lies along the z axis only
        difference_vector_rotated, rotation_matrix = rotate_coordinates_to_z_axis(difference_vector)
        aligned_coordinates = np.array([[0.0, 0.0, 0.0], -1 * difference_vector_rotated])

        # Additional print makes a big mess - prints all energy calculations to console
        if calculation.extrapolate:

            SCF_output, molecule, electronic_energy, _ = energ.extrapolate_energy(calculation, atoms, aligned_coordinates, P_guess=P_guess, E_guess=E_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, silent=not(calculation.additional_print))

        else:
            
            SCF_output, molecule, electronic_energy, _ = energ.calculate_energy(calculation, atoms, aligned_coordinates, P_guess=P_guess, E_guess=E_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, silent=not(calculation.additional_print))

        forces = calculate_forces(aligned_coordinates, calculation, atoms, rotation_matrix)

        accelerations_new = calculate_accelerations(forces, inv_masses) 
        velocities += (1 / 2) * timestep_au * (accelerations + accelerations_new) 

        # Updates accelerations and increments timestep
        accelerations = accelerations_new
        time += timestep

        # Cycle begins again as new energy components are printed
        potential_energy, kinetic_energy, total_energy, temperature, bond_length, drift = calculate_MD_components(molecule, masses, velocities, initial_energy, degrees_of_freedom, electronic_energy)
        log(f" {(i + 1):4.0f}    {time:5.2f}     {bond_length:.4f}    {temperature:10.2f}     {potential_energy:12.6f}   {kinetic_energy:12.6f}     {total_energy:12.6f}   {drift:12.6f}", calculation, 1)
        
        # By default prints trajectory to file, can be viewed with Jmol
        if calculation.trajectory: 
            
            import tuna_out as out

            out.print_trajectory(molecule, potential_energy, coordinates, trajectory_path)
        


    log_big_spacer(calculation)








