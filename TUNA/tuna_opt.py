from tuna_util import *
import numpy as np
from numpy import ndarray
import tuna_energy as energ
import sys
from termcolor import colored
import tuna_props as props
import tuna_out as out
import tuna_freq as freq
import tuna_kernel as kern
from tuna_calc import Calculation


"""

This is the TUNA module for geometry optimisation, written first for version 0.3.0 and rewritten for version 0.10.0.

For a one-dimensional system, there is an exact "best" way to converge the geometry, using the approximate Hessian (which is not a matrix, but a number). This 
module implements this optimisation method with numerical derivatives of the total molecular energy to find the force. Several options are available, including
calculating an "exact" Hessian instead of the approximate one - which becomes more worthwhile further from the optimisation endpoint. The optimisation can 
optionally be printed to an ".xyz" output file with the "TRAJ" keyword.

Updated in version 0.10.1 to include the bond dissociation energy calculation type.

The module contains:

1. Functions to call tuna_energy to calculate numerical derivatives for the gradient and hessian (calculate_gradient, calculate_hessian).
2. Useful functions for the optimisation (e.g. optimisation_is_converged, print_optimisation_convergence_information).
3. The main function for the geometry optimisation calculation, optimise_geometry.
4. The main function to calculate optimisations for electron affinity and ionisation energy calculations, calculate_charged_state_energies.

"""




def calculate_gradient(coordinates: ndarray, calculation: Calculation, atomic_symbols: list, silent: bool = False) -> float:

    """

    Calculates the numerical derivative of the molecular energy with respect to bond length.

    Args:   
        coordinates (array): Atomic coordinates
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        silent (bool, optional): Should anything be printed

    Returns:
        gradient (float): Derivative of energy wrt. bond length

    """

    prodding_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, constants.FIRST_GEOM_DERIVATIVE_PROD]])  

    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords

    log(" Calculating energy on displaced geometry 1 of 2...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    _, _, energy_forward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)

    log(" Calculating energy on displaced geometry 2 of 2...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    _, _, energy_backward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, backward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)

    # Calculates numerical first derivative

    gradient = calculate_first_derivative(energy_backward, energy_forward, constants.FIRST_GEOM_DERIVATIVE_PROD)


    return gradient
    









def calculate_hessian(coordinates: ndarray, calculation: Calculation, atomic_symbols: list, energy: float, silent: bool = False) -> tuple:

    """

    Calculates the hessian, the second derivative of molecular energy with respect to bond length.

    Also provides wavefunction information for seminumerical dipole moment derivatives.

    Args:   
        coordinates (array): Atomic coordinates
        calculation (Calculation): Calculation object
        atomic_symbols (list): Atomic symbol list
        silent (bool, optional): Cancel logging

    Returns:
        hessian (float): Second derivative of energy wrt. bond length
        SCF_output_forward (Output): SCF output object from forward prodded coordinates
        P_forward (array): Density matrix in AO basis from forward prodded coordinates
        SCF_output_backward (Output): SCF output object from backward prodded coordinates
        P_backward (array): Density matrix in AO basis from backward prodded coordinates

    """

    prodding_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, constants.SECOND_GEOM_DERIVATIVE_PROD]])  

    far_forward_coords = coordinates + 2 * prodding_coords
    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords
    far_backward_coords = coordinates - 2 * prodding_coords  

    log("\n Calculating energy on displaced geometry 1 of 4...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    _, _, energy_far_forward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, far_forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 2 of 4...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    SCF_output_forward, _, energy_forward, P_forward = energ.evaluate_molecular_energy(calculation, atomic_symbols, forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 3 of 4...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    SCF_output_backward, _, energy_backward, P_backward = energ.evaluate_molecular_energy(calculation, atomic_symbols, backward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 4 of 4...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    _, _, energy_far_backward, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, far_backward_coords, silent=True)

    log("[Done]\n", calculation, 1, silent=silent)   

    # Calculates numerical second derivative

    hessian = calculate_second_derivative(energy_far_backward, energy_backward, energy, energy_forward, energy_far_forward, constants.SECOND_GEOM_DERIVATIVE_PROD)
    
    displaced_energies = energy_far_backward, energy_backward, energy_forward, energy_far_forward

    return hessian, SCF_output_forward, P_forward, SCF_output_backward, P_backward, displaced_energies










def calculate_approximate_hessian(delta_bond_length: float, delta_grad: float) -> float: 

    """

    Calculates the approximate hessian.

    Args:   
        delta_bond_length (float): Change in bond length
        delta_grad (float): Change in gradient

    Returns:
        hessian (float): Approximate second derivative of energy wrt. bond length

    """

    hessian = delta_grad / delta_bond_length

    return hessian










def optimisation_is_converged(iteration: int, gradient: float, step: float, calculation: Calculation) -> bool:
    
    """
    
    Checks if the optimisation is converged.

    Args:
        iteration (int): Current optimisation iteration
        gradient (float): Gradient of electronic energy
        step (float): Change in nuclear positions in bohr
        calculation (Calculation): Calculation object
    
    Returns:
        converged (bool): Is the optimisation converged
    
    """

    converged_grad = abs(gradient) < calculation.geom_conv.get("gradient")

    converged_step = abs(step) < calculation.geom_conv.get("step")

    # Require convergence in both force and step size

    converged = converged_grad and converged_step
    
    if converged:

        log_spacer(calculation, start="\n",space="")
        log(colored(f"      Optimisation converged in {iteration} iterations!","white"), calculation, 1)
        log_spacer(calculation,space="")


    return converged










def update_hessian(calculation: Calculation, coordinates: ndarray, atomic_symbols: list, energy: float, bond_length: float, old_bond_length: float, gradient: float, old_gradient: float) -> float:

    """
    
    Calculates an updated Hessian during a geometry optimisation.

    Args:
        calculation (Calculation): Calculatio object
        coordinates (array): Atomic coordinates
        atomic_symbols (list): List of atomic symbols
        energy (float): Molecular energy
        bond_length (float): Bond length
        old_bond_length (float): Bond length of last iteration
        gradient (float): Gradient
        old_gradient (float): Gradient of last iteration

    Returns:
        hessian (float): Updated Hessian

    """

    hessian = calculation.default_hessian

    if calculation.calc_hess: 

        log("\n Beginning calculation of exact hessian...    ", calculation, 1)

        candidate_hessian, _, _, _, _, _ = calculate_hessian(coordinates, calculation, atomic_symbols, energy, silent=False)

    else: 

        # Calculates approximate hessian if "CALCHESS" keyword not used

        candidate_hessian = calculate_approximate_hessian(bond_length - old_bond_length, gradient - old_gradient)


    # Checks if region is convex or concave, if in the correct region for opt to min/max, sets the hessian to the second derivative
    
    if calculation.opt_max and candidate_hessian < -0.01:

        hessian = -candidate_hessian

    elif candidate_hessian > 0.01:
        
        hessian = candidate_hessian


    return hessian










def print_optimisation_convergence_information(gradient: float, step: float, calculation: Calculation) -> None:

    """
    
    Prints the progress of the geometry optimisation.

    Args:
        gradient (float): Electronic energy gradient
        step (float): Step size in bohr
        calculation (Calculation): Calculation object
    
    """

    gradient_convergence_criteria = calculation.geom_conv.get("gradient")
    step_convergence_criteria = calculation.geom_conv.get("step")

    # Checks for convergence of individual components

    is_gradient_converged_str = convert_boolean_to_string(np.abs(gradient) < gradient_convergence_criteria)
    is_step_converged_str = convert_boolean_to_string(np.abs(step) < step_convergence_criteria)

    log_spacer(calculation, start="\n")

    log("   Factor        Value       Criteria    Converged?", calculation, 1)

    log_spacer(calculation)

    log(f"  Gradient   {gradient:11.8f}   {gradient_convergence_criteria:11.8f}      {is_gradient_converged_str} ", calculation, 1)
    log(f"    Step     {step:11.8f}   {step_convergence_criteria:11.8f}      {is_step_converged_str} ", calculation, 1)

    log_spacer(calculation)

    return 










def optimise_geometry(calculation: Calculation, atomic_symbols: list, coordinates: ndarray, multiple_iterations: bool = True) -> tuple | None:
    
    """

    Optimises the geometry of the molecule to a stationary point on the potential energy surface.

    Args:   
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates
        multiple_iterations (bool, optional): Do multiple geometry iterations

    Returns:
        molecule (Molecule): Optimised molecule object
        bond_length (float) : Optimised bond length

    """

    geom_conv_criteria = calculation.geom_conv
    max_geom_iter = calculation.geom_max_iter

    log("\nInitialising geometry optimisation...\n", calculation, 1)

    # If "TRAJ" keyword is used, prints trajectory to file - opening and closing it here clears it

    if calculation.trajectory: 
        
        log(f"Printing trajectory data to \"{calculation.trajectory_path}\"\n", calculation, 1)

        open(calculation.trajectory_path, "w").close()

    # Prints type of Hessian used for geometry update

    hessian_type = "exact" if calculation.calc_hess else "approximate"

    log(f"Using {hessian_type} hessian in convex region, hessian of {calculation.default_hessian:.3f} outside.\n", calculation, 1)

    log(f"Convergence criteria for gradient is {geom_conv_criteria.get("gradient"):.8f}, step convergence is {geom_conv_criteria.get("step"):.8f} angstroms.", calculation, 1)
    log(f"Geometry iterations will not exceed {max_geom_iter}, maximum step is {calculation.max_step} angstroms.", calculation, 1)


    P_guess, P_guess_alpha, P_guess_beta, E_guess = None, None, None, None


    for iteration in range(1, max_geom_iter + 1):
        
        # A "FORCE" calculation only runs a single iteration

        if iteration > 1 and not multiple_iterations: 
            
            break
        
        # Calculates bond length from current coordinates

        bond_length = calculate_bond_length(coordinates)

        log_big_spacer(calculation, start="\n",space="")
        log(f"Beginning energy and gradient iteration {iteration} with bond length of {bohr_to_angstrom(bond_length):5f} angstroms...", calculation, 1)
        log_big_spacer(calculation, space="")

        # Prevents running the post-SCF calculations on an energy evaluation

        terse = not calculation.additional_print

        # Evaluates the energy and density

        SCF_output, molecule, energy, P = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=terse)

        # By default, reads in the density from the last step as a guess - can be turned off with "NOMOREAD"

        if calculation.MO_read:

            P_guess = SCF_output.P
            P_guess_alpha = SCF_output.P_alpha
            P_guess_beta = SCF_output.P_beta

            E_guess = SCF_output.energy

        # Calculates gradient at each point

        log("\n Beginning numerical gradient calculation...  \n", calculation, 1)

        gradient = calculate_gradient(coordinates, calculation, atomic_symbols, silent=False)

        # Updates bond length

        bond_length = molecule.bond_length
     
        # Calculates either the exact or approximate Hessian - avoid the first iteration as there's no old bond length

        hessian = update_hessian(calculation, coordinates, atomic_symbols, energy, bond_length, old_bond_length, gradient, old_gradient) if iteration > 1 else calculation.default_hessian

        # Calculates step to be taken using Wikipedia equation for Newton's method

        step = gradient / hessian

        # Prints convergence information of various criteria for optimisation

        print_optimisation_convergence_information(gradient, step, calculation)

        # Prints trajectory to file if "TRAJ" keyword has been used

        if calculation.trajectory: 

            out.save_trajectory_to_file(molecule, energy, coordinates, calculation.trajectory_path)

        # If optimisation is converged, begin post SCF output and print to console, then finish calculation

        if optimisation_is_converged(iteration, gradient, step, calculation): 

            props.calculate_molecular_properties(molecule, calculation, P, SCF_output.S, SCF_output, SCF_output.P_alpha, SCF_output.P_beta)

            log(f"\n Optimisation converged in {iteration} iterations to bond length of {bohr_to_angstrom(bond_length):.5f} angstroms!", calculation, 1)
            log(f"\n Final single point energy: {energy:.10f}", calculation, 1)

            return molecule, energy

        else:
            
            if np.abs(step) > calculation.max_step:

                step = np.sign(step) * calculation.max_step

                warning("Calculated step is outside of trust radius, taking maximum step instead.")

            # Checks direction in which step should be taken, depending on whether "OPTMAX" keyword has been used
            
            direction = -1 if calculation.opt_max else 1

            # Builds new coordinates - a minus sign here for direction as we are minimising

            coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, coordinates[1][2] - direction * step]])

            if coordinates[1][2] < 0.01: 
                
                error("Optimisation generated negative bond length! Decrease maximum step!")

            # Updates "old" quantities to be used for comparison to check convergence

            old_bond_length = bond_length
            old_gradient = gradient
     
    if multiple_iterations:

        error(F"Geometry optimisation did not converge in {max_geom_iter} iterations! Increase the maximum or give up!")

    return










def calculate_charged_state_energies(calculation: Calculation, atomic_symbols: list[str], coordinates: ndarray, charge_delta: int) -> tuple:

    """
    
    Calculates the reference-state and charged-state energies needed for ionisation energy or electron affinity calculations.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): Atomic symbols
        coordinates (array): Atomic coordinates
        charge_delta (int): +1 for ionisation energy, -1 for electron affinity

    Returns:
        reference_energy (float): Energy of the original charge state
        charged_energy (float): Energy of the final charge state
        reference_molecule (float): Molecule of the original charge state
        charged_molecule (Molecule): Molecule of the final charge state

    """

    # Optimise the molecule unless it's an atom, or "VERTICAL" is used

    if calculation.vertical or calculation.monatomic:
        
        log_spacer(calculation, start="\n", space="")
        log("Calculating energy of original system...", calculation)
        log_spacer(calculation, space="")

        method = calculation.method

        reference_SCF_output, reference_molecule, reference_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates)

        calculation.charge += charge_delta * calculation.n_electrons_for_ip_or_ea
        
        log_spacer(calculation, start="\n", space="")
        log("Calculating energy of charged system...", calculation)
        log_spacer(calculation, space="")

        # Resets the method - it may have been overridden during the first evaluation

        calculation.method = method

        _, charged_molecule, charged_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, coordinates, integrals=reference_SCF_output.integrals)

    else:

        # These functions will calculate the adiabatic electron affinity or ionisation energy

        log_spacer(calculation, start="\n", space="")
        log("Optimising energy of original molecule...", calculation)
        log_spacer(calculation, space="")

        method = calculation.method

        reference_molecule, reference_energy = optimise_geometry(calculation, atomic_symbols, coordinates)

        calculation.charge += charge_delta * calculation.n_electrons_for_ip_or_ea

        log_spacer(calculation, start="\n", space="")
        log("Optimising energy of charged molecule...", calculation)
        log_spacer(calculation, space="")

        # Resets the method - it may have been overridden during the first evaluation

        calculation.method = method

        charged_molecule, charged_energy = optimise_geometry(calculation, atomic_symbols, reference_molecule.coordinates)


    return reference_energy, charged_energy,  reference_molecule, charged_molecule










def calculate_bond_dissociation_energy(calculation: Calculation, atomic_symbols: list, coordinates: ndarray) -> None:

    """
    
    Calculates the counterpoise corrected, optionally zero-point energy corrected, bond dissociation energy.

    Args:
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates in bohr
    
    """

    # First, optimise the molecular geometry

    optimised_molecule, optimised_energy = optimise_geometry(calculation, atomic_symbols, coordinates)

    zero_point_energy = 0.0

    # If the "ZPE" keyword is used, also calculate the harmonic zero-point energy

    if calculation.do_ZPE_correction:

        _, _, _, zero_point_energy = freq.calculate_harmonic_frequency(calculation, molecule=optimised_molecule, energy=optimised_energy)

    log_spacer(calculation, start="\n", space="", end="~~~")
    log("Calculating energy on atoms", calculation) if calculation.no_counterpoise_correction else log("Calculating counterpoise-corrected atomic energies...", calculation)
    log_spacer(calculation, space="", end="~~~")

    # Define the new coordinates, with a ghost atom in the equilibrium geometry position by default for counterpoise correction

    if calculation.no_counterpoise_correction:
        
        atomic_coordinates = np.array([[0.0, 0.0, 0.0]]) 

    else:
        
        atomic_coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, optimised_molecule.bond_length]])
    
    # Update the calculation object, turning on core Hamiltonian guess rather than SCF/SAD guess which doesn't work with ghost atoms

    calculation.monatomic, calculation.diatomic, calculation.core_guess = True, False, True

    # Cache the original set of atomic symbols

    original_symbols = atomic_symbols

    # Build new atomic symbols, which may be counterpoise corrected with ghost atoms

    atomic_symbols = [original_symbols[0]] if calculation.no_counterpoise_correction else [original_symbols[0], "X" + original_symbols[1]]

    # Evaluate the energy on the isolated atom (with or without ghost basis functions)

    _, _, first_atom_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, atomic_coordinates)

    # If it's a heteronuclear molecule, evaluate the energy on the second isolated atom

    if optimised_molecule.heteronuclear:

        atomic_symbols = [original_symbols[1]] if calculation.no_counterpoise_correction else [original_symbols[1], "X" + original_symbols[0]]
        
        _, _, second_atom_energy, _ = energ.evaluate_molecular_energy(calculation, atomic_symbols, atomic_coordinates)

    else:

        second_atom_energy = first_atom_energy

    # Prints out the bond dissociation energy information

    kern.print_bond_dissociation_energy_information(first_atom_energy, second_atom_energy, optimised_energy, zero_point_energy, optimised_molecule, calculation)

    return