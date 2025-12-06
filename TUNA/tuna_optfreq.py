from tuna_util import *
import numpy as np
import tuna_energy as energ
import sys
from termcolor import colored
import tuna_postscf as postscf
import tuna_thermo as thermo


def calculate_gradient(coordinates, calculation, atoms, silent=False):

    """

    Calculates the derivative of the molecular energy with respect to bond length.

    Args:   
        coordinates (array): Atomic coordinates
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        silent (bool, optional): Should anything be printed

    Returns:
        gradient (float): Derivative of energy wrt. bond length

    """

    prod = 0.0001

    prodding_coords = np.array([[0,0,0], [0,0, prod]])  

    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords

    log(" Calculating energy on displaced geometry 1 of 2...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    if calculation.extrapolate:

        _, _, energy_forward, _ = energ.extrapolate_energy(calculation, atoms, forward_coords, silent=True)

    else:
        
        _, _, energy_forward, _ = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)

    log(" Calculating energy on displaced geometry 2 of 2...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    if calculation.extrapolate:

        _, _, energy_backward, _ = energ.extrapolate_energy(calculation, atoms, backward_coords, silent=True)

    else:
        
        _, _, energy_backward, _ = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)


    gradient = (energy_forward - energy_backward) / (2 * prod)

    return gradient
    




def calculate_approximate_Hessian(delta_coords, delta_grad): 

    """

    Calculates the approximate Hessian.

    Args:   
        delta_coords (float): Change in bond length
        delta_grad (float): Change in gradient

    Returns:
        Hessian (float): Approximate second derivative of energy wrt. bond length

    """

    Hessian = delta_grad / delta_coords

    return Hessian







def calculate_Hessian(coordinates, calculation, atoms, silent=False):

    """

    Calculates the Hessian, the second derivative of molecular energy with respect to bond length.

    Args:   
        coordinates (array): Atomic coordinates
        calculation (Calculation): Calculation object
        atoms (list): Atomic symbol list
        silent (bool, optional): Should anything be printed

    Returns:
        Hessian (float): Second derivative of energy wrt. bond length
        SCF_output_forward (Output): SCF output object from forward prodded coordinates
        P_forward (array): Density matrix in AO basis from forward prodded coordinates
        SCF_output_backward (Output): SCF output object from backward prodded coordinates
        P_backward (array): Density matrix in AO basis from backward prodded coordinates

    """

    prod = 0.0001

    prodding_coords = np.array([[0,0,0], [0,0, prod]])  

    far_forward_coords = coordinates + 2 * prodding_coords
    forward_coords = coordinates + prodding_coords
    backward_coords = coordinates - prodding_coords
    far_backward_coords = coordinates - 2 * prodding_coords  

    log("\n Calculating energy on displaced geometry 1 of 5...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.extrapolate:

        _, _, energy_far_forward, _ = energ.extrapolate_energy(calculation, atoms, far_forward_coords, silent=True)

    else:
        
        _, _, energy_far_forward, _ = energ.calculate_energy(calculation, atoms, far_forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 2 of 5...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.extrapolate:

        SCF_output_forward, _, energy_forward, P_forward = energ.extrapolate_energy(calculation, atoms, forward_coords, silent=True)

    else:
        
        SCF_output_forward, _, energy_forward, P_forward = energ.calculate_energy(calculation, atoms, forward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 3 of 5...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.extrapolate:

        _, _, energy, _ = energ.extrapolate_energy(calculation, atoms, coordinates, silent=True)

    else:
        
        _, _, energy, _ = energ.calculate_energy(calculation, atoms, coordinates, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 4 of 5...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.extrapolate:

        SCF_output_backward, _, energy_backward, P_backward = energ.extrapolate_energy(calculation, atoms, backward_coords, silent=True)

    else:
        
        SCF_output_backward, _, energy_backward, P_backward = energ.calculate_energy(calculation, atoms, backward_coords, silent=True)

    log("[Done]", calculation, 1, silent=silent)   

    log(" Calculating energy on displaced geometry 5 of 5...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    if calculation.extrapolate:

        _, _, energy_far_backward, _ = energ.extrapolate_energy(calculation, atoms, far_backward_coords, silent=True)

    else:
        
        _, _, energy_far_backward, _ = energ.calculate_energy(calculation, atoms, far_backward_coords, silent=True)

    log("[Done]\n", calculation, 1, silent=silent)   

    # Equation from Wikipedia page on numerical second derivative methods, fairly noise-resistant formula
    Hessian = (-energy_far_forward + 16 * energy_forward - 30 * energy + 16 * energy_backward -  energy_far_backward) / (12 * prod ** 2)

    return Hessian, SCF_output_forward, P_forward, SCF_output_backward, P_backward







def calculate_dipole_derivative(coordinates, molecule, SCF_output_forward, SCF_output_backward, P_forward, P_backward):

    """

    Calculates the dipole derivative in normal coordinates.

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

    prod = 0.0001

    masses = molecule.masses
    charges = molecule.charges

    # Forward and backward coordinates are symmetrical by the mass weighting, to prevent influence of centre of mass in dipole moment calculations
    forward_coords = coordinates + np.array([[0,0,0 - masses[1] * prod], [0,0,0 + masses[0] * prod]]) / np.sum(masses)
    backward_coords = coordinates - np.array([[0,0,0 - masses[1] * prod], [0,0,0 + masses[0] * prod]]) / np.sum(masses)

    # Calculates centre of mass for unperturbed molecule
    centre_of_mass = postscf.calculate_centre_of_mass(masses, coordinates)

    # Calculates forward and backward dipole moments, using dipole integrals calculated from centre of mass
    dipole_moment_forward = postscf.calculate_dipole_moment(centre_of_mass, charges, forward_coords, P_forward, SCF_output_forward.D)[0]
    dipole_moment_backward = postscf. calculate_dipole_moment(centre_of_mass, charges, backward_coords, P_backward, SCF_output_backward.D)[0]

    # Calculates dipole derivative by central differences method
    dipole_derivative = (dipole_moment_forward - dipole_moment_backward) / (2 * prod)

    # Converts to normal coordinates by mass weighting
    dipole_derivative /= np.sqrt(postscf.calculate_reduced_mass(masses))

    return dipole_derivative








def optimise_geometry(calculation, atoms, coordinates, multiple_iterations=True):
    
    """

    Optimises the geometry of the molecule to a stationary point on the potential energy surface.

    Args:   
        calculation (Calculation): Calculation object
        atoms (list): List of atomic symbols
        coordinates (array): Atomic coordinates

    Returns:
        molecule (Molecule): Optimised molecule object
        bond_length (float) : Optimised bond length

    """

    maximum_step = angstrom_to_bohr(calculation.max_step)
    default_Hessian = calculation.default_Hessian
    geom_conv_criteria = calculation.geom_conv
    max_geom_iter = calculation.geom_max_iter
    opt_max = calculation.opt_max
    calc_hess = calculation.calc_hess

    trajectory_path = calculation.trajectory_path


    log("\nInitialising geometry optimisation...\n", calculation, 1)

    #If TRAJ keyword is used, prints trajectory to file
    if calculation.trajectory: 
        
        log(f"Printing trajectory data to \"{trajectory_path}\"\n", calculation, 1)
        
        with open(trajectory_path, 'w'): pass


    if calc_hess: log(f"Using exact Hessian in convex region, Hessian of {default_Hessian:.3f} outside.\n", calculation, 1)
    else: log(f"Using approximate Hessian in convex region, Hessian of {default_Hessian:.3f} outside.\n", calculation, 1)

    log(f"Convergence criteria for gradient is {geom_conv_criteria.get("gradient"):.8f}, step convergence is {geom_conv_criteria.get("step"):.8f} angstroms.", calculation, 1)
    log(f"Geometry iterations will not exceed {max_geom_iter}, maximum step is {bohr_to_angstrom(maximum_step)} angstroms.", calculation, 1)


    P_guess = None
    E_guess = None
    P_guess_alpha = None
    P_guess_beta = None


    for iteration in range(1, max_geom_iter + 1):
        
        if iteration > 1 and not multiple_iterations: break

        bond_length = np.linalg.norm(coordinates[1] - coordinates[0])

        log_big_spacer(calculation, start="\n",space="")
        log(f"Beginning energy and gradient iteration {iteration} with bond length of {bohr_to_angstrom(bond_length):.5f} angstroms...", calculation, 1)
        log_big_spacer(calculation, space="")

        terse = not calculation.additional_print

        if calculation.extrapolate:

            SCF_output, molecule, energy, P = energ.extrapolate_energy(calculation, atoms, coordinates, P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=terse)

        else:
        
            SCF_output, molecule, energy, P = energ.calculate_energy(calculation, atoms, coordinates, P_guess, P_guess_alpha=P_guess_alpha, P_guess_beta=P_guess_beta, E_guess=E_guess, terse=terse)


        if calculation.MO_read:

            P_guess =  SCF_output.P
            P_guess_alpha = SCF_output.P_alpha
            P_guess_beta = SCF_output.P_beta

            E_guess = SCF_output.energy

        # Calculates gradient at each point
        log("\n Beginning numerical gradient calculation...  \n", calculation, 1)
        gradient = calculate_gradient(coordinates, calculation, atoms, silent=False)

        bond_length = molecule.bond_length
        Hessian = default_Hessian

     
        if iteration > 1:

            if calc_hess: 

                log("\n Beginning calculation of exact Hessian...    ", calculation, 1); sys.stdout.flush()

                hess, _, _, _, _ = calculate_Hessian(coordinates, calculation, atoms, silent=False)

            else: 

                # Calculates approximate Hessian if CALCHESS keyword not used
                hess = calculate_approximate_Hessian(bond_length - old_bond_length, gradient - old_gradient)


            # Checks if region is convex or concave, if in the correct region for opt to min/max, sets the Hessian to the second derivative
            if opt_max:

                if hess < 0.01: Hessian = -hess

            elif hess > 0.01: Hessian = hess


        # Calculates step to be taken using Wikipedia equation for Newton's method
        inverse_Hessian = 1 / Hessian           
        step = inverse_Hessian * gradient
        

        def bool_to_word(bool):

            if bool: return "Yes"
            else: return "No" 


        # Checks for convergence of various criteria for optimisation

        converged_grad = True if np.abs(gradient) < geom_conv_criteria.get("gradient") else False
        converged_step = True if np.abs(step) < geom_conv_criteria.get("step") else False

        
        log_spacer(calculation, start="\n")
        log("   Factor        Value       Criteria    Converged?", calculation, 1)
        log_spacer(calculation)
        log(f"  Gradient   {gradient:11.8f}   {geom_conv_criteria.get("gradient"):11.8f}      {bool_to_word(converged_grad)} ", calculation, 1)
        log(f"    Step     {step:11.8f}   {geom_conv_criteria.get("step"):11.8f}      {bool_to_word(converged_step)} ", calculation, 1)
        log_spacer(calculation)

        # Prints trajectory to file if TRAJ keyword has been used
        if calculation.trajectory: 
            import tuna_out as out

            out.print_trajectory(molecule, energy, coordinates, trajectory_path)

        # If optimisation is converged, begin post SCF output and print to console, then finish calculation
        if converged_grad and converged_step: 

            log_spacer(calculation, start="\n",space="")
            log(colored(f"      Optimisation converged in {iteration} iterations!","white"), calculation, 1)
            log_spacer(calculation,space="")

            postscf.post_SCF_output(molecule, calculation, SCF_output.epsilons, SCF_output.molecular_orbitals, P, SCF_output.S, molecule.partition_ranges, SCF_output.D, SCF_output.P_alpha, SCF_output.P_beta, SCF_output.epsilons_alpha, SCF_output.epsilons_beta, SCF_output.molecular_orbitals_alpha, SCF_output.molecular_orbitals_beta)
          
            log(f"\n Optimisation converged in {iteration} iterations to bond length of {bohr_to_angstrom(bond_length):.5f} angstroms!", calculation, 1)
            log(f"\n Final single point energy: {energy:.10f}", calculation, 1)

            return molecule, energy

        else:
            
            if step > maximum_step: 

                step = maximum_step

                warning("Calculated step is outside of trust radius, taking maximum step instead.")

            elif step < -maximum_step:

                step = -maximum_step

                warning("Calculated step is outside of trust radius, taking maximum step instead.")

            # Checks direction in which step should be taken, depending on whether OPTMAX keyword has been used
            direction = -1 if opt_max else 1

            # Builds new coordinates
            coordinates = np.array([[0, 0, 0], [0, 0, coordinates[1][2] - direction * step]])

            if coordinates[1][2] <= 0: error("Optimisation generated negative bond length! Decrease maximum step!")

            # Updates "old" quantities to be used for comparison to check convergence
            old_bond_length = bond_length
            old_gradient = gradient
     
    if multiple_iterations:

        error(F"Geometry optimisation did not converge in {max_geom_iter} iterations! Increase the maximum or give up!")







def calculate_frequency(calculation, atomic_symbols=None, coordinates=None, molecule=None, energy=None):

    """

    Calculates harmonic frequency of a molecule.

    Args:   
        calculation (Calculation): Calculation object
        atomic_symbols (list, optional): List of atomic symbols
        coordinates (array, optional): Atomic coordinates
        molecule (Molecule): Molecule object
        energy (float): Total molecular energy

    Returns:
        None: Nothing is returned

    """

    # If "FREQ" keyword has been used, calculates the energy using the supplied atoms and coordinates, otherwise uses the supplied molecule and energy
    if calculation.calculation_type == "FREQ":
          
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
    k, SCF_output_forward, P_forward, SCF_output_backward, P_backward = calculate_Hessian(coordinates, calculation, atomic_symbols)

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
        vibrational_entropy = 0
        vibrational_internal_energy = 0
        
        warning("Imaginary frequency calculated! Zero-point energy and vibrational thermochemical parameters set to zero!\n")

    # Converts frequency into human units from atomic units
    frequency_per_cm = frequency_hartree * constants.per_cm_in_hartree

    dipole_derivative = calculate_dipole_derivative(coordinates, molecule, SCF_output_forward, SCF_output_backward, P_forward, P_backward)

    # Adding vibrational overlap contribution to match ORCA results
    vibrational_overlap = 1 / np.sqrt(2 * frequency_hartree)
    dipole_derivative *= vibrational_overlap

    dipole_derivative_squared = dipole_derivative ** 2

    # This equation comes from Neugebauer 2002, except for factor of 2 and frequency, which are to counteract including this vibrational term previously
    dipole_derivative_squared_C_squared_per_kg = dipole_derivative_squared * 2 * frequency_hartree * (constants.elementary_charge_in_coulombs) ** 2 / constants.electron_mass_in_kilograms
    transition_intensity_km_per_mol = dipole_derivative_squared_C_squared_per_kg * constants.avogadro / (12 * constants.permittivity_in_farad_per_metre * constants.c_in_metres_per_second ** 2 * 1000)

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

    # Uses thermochemistry module to calculate various thermochemical properties
    U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy = thermo.calculate_internal_energy(energy, ZPE, temperature, frequency_per_cm)
    H = thermo.calculate_enthalpy(U, temperature)
    S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy = thermo.calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_cm * 100, masses, pressure)
    G = thermo.calculate_free_energy(H, temperature, S)

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("                                   Thermochemistry", calculation, 2, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  Electronic energy:   {energy:16.10f}     Electronic entropy:   {temperature*electronic_entropy:16.10f}", calculation, 2)
    log(f"\n  Translational energy:{translational_internal_energy:16.10f}     Translational entropy:{temperature*translational_entropy:16.10f}", calculation, 2)
    log(f"  Rotational energy:   {rotational_internal_energy:16.10f}     Rotational entropy:   {temperature*rotational_entropy:16.10f}", calculation, 2)
    log(f"  Vibrational energy:  {vibrational_internal_energy:16.10f}     Vibrational entropy:  {temperature*vibrational_entropy:16.10f}  ", calculation, 2)
    log(f"  Zero-point energy:   {ZPE:16.10f}", calculation, 2)
    log(f"\n  Internal energy:     {U:16.10f}", calculation, 2)
    log(f"  Enthalpy:            {H:16.10f}     Entropy:              {temperature*S:16.10f}", calculation, 2)
    log(f"\n  Gibbs free energy:   {G:16.10f}     Non-electronic energy:{energy - G:16.10f}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)

    