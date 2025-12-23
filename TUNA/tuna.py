#!/usr/bin/env python3

import __init__ as init
import sys
version_number = init.__version__
from termcolor import colored

if len(sys.argv) == 2 and sys.argv[1] in ["-version", "--version"]:

    print(f"TUNA {version_number}")
    sys.exit(0)


# Prints big fish logo
print(colored("\n      _______ _    _ _   _                     ___           \n     |__   __| |  | | \\ | |   /\\            __/__/__  _      \n","white", force_color = True) + colored(" ~~~~~~","light_grey", force_color = True)+colored(" | |  | |  | |  \\| |  /  \\","white", force_color = True)+colored(" ~~~~~~~~","light_grey", force_color = True)+colored(" / .      \\/ ) ","white", force_color = True)+colored("~~~~\n ~~~~~~","light_grey")+colored(" | |  | |  | | . ` | / /\\ \\","white", force_color = True)+colored(" ~~~~~~","light_grey", force_color = True)+colored(" (     ))    (","white", force_color = True)+colored(" ~~~~~\n ~~~~~~","light_grey", force_color = True)+colored(" | |  | |__| | |\\  |/ ____ \\ ","white", force_color = True)+colored("~~~~~~","light_grey", force_color = True)+colored(" \\___  ___/\\_) ","white", force_color = True)+colored("~~~~","light_grey", force_color = True)+colored("\n        |_|   \\____/|_| \\_/_/    \\_\\          \\\\_\\           ", "white", force_color = True))

print(colored(f"\n\nWelcome to version {version_number} of TUNA!\n", "light_grey", force_color=True))
print(colored("Importing required libraries...  ", "light_grey", force_color=True),end="")


import numpy as np; sys.stdout.flush()
import time
from tuna_util import *
import tuna_energy as energ
import tuna_optfreq as optfreq
import tuna_md as md

print(colored("[Done]\n", "light_grey", force_color=True))

start_time = time.perf_counter()


def parse_input():

    """

    Parses the input line in the console and returns extracted quantities.

    Returns:
        calculation_type (string): Type of calculation
        method (string): Electronic structure method
        basis (string): Basis set
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Array of atomic coordinates
        params (list): User-specified parameters

    """

    # Allowed options for the input line
    atom_options = atomic_properties.keys() 
    ghost_options = [f"X{key}" for key in atomic_properties.keys()]
    calculation_options = calculation_types.keys()
    method_options = method_types.keys()

    # Puts input line into standardised format, capital letters and separated by spaces for each argument
    input_line = " ".join(sys.argv[1:]).upper().strip()

    try: 
        
        # Separates input line into sections separated by a colon, extracts relevant information from those sections
        sections = input_line.split(":")

        calculation_type = sections[0].strip()
        geometry_section = sections[1].strip()
        method, basis = sections[2].strip().split()

        if len(sections) == 4: params = sections[3].strip().split()  
        else: params = []

        for param in params: param = param.strip()   

    except: error("Input line formatted incorrectly! Read the manual for help.")

    # Creates a list of atoms, either one or two long
    atomic_symbols = [atom.strip() for atom in geometry_section.split(" ")[0:2] if atom.strip()]
    
    try:
        
        # Extracts bond length from geometry section if it exists, sets coordinates to be [0, bond length]
        coordinates_1D = [0] + [float(bond_length.strip()) for bond_length in geometry_section.split(" ")[2:] if bond_length.strip()]
    
    except ValueError: error("Could not parse bond length!")
    
    # Checks if requested calculation, method, basis, etc. are in the allowed options, then gives relevant error message if not 
    if calculation_type not in calculation_options: error(f"Calculation type \"{calculation_type}\" is not supported.")
    if method not in method_options: error(f"Calculation method \"{method}\" is not supported.")
    if basis not in basis_types.keys(): error(f"Basis set \"{basis}\" is not supported.")
    if not all(atom in atom_options or atom in ghost_options for atom in atomic_symbols): error("One or more atom types not recognised! Check the manual for available atoms.")
    if len(atomic_symbols) != len(coordinates_1D): error("Two atoms requested without a bond length!")

    # Rejects requests for tiny bond lengths, such as two atoms on top of each other
    if len(coordinates_1D) == 2 and coordinates_1D[1] < 0.05: error(f"Bond length ({coordinates_1D[1]} angstroms) is too small! Minimum bond length is 0.05 angstroms.")

    # Converts 1D coordinate array in angstroms to 3D array ion bohr
    coordinates = one_dimension_to_three(angstrom_to_bohr(np.array(coordinates_1D)))

    return calculation_type, method, basis, atomic_symbols, coordinates, params




def run_calculation(calculation_type, calculation, atomic_symbols, coordinates):

    """

    Runs the calculation, handing off to the relevant modules requested.

    Args:
        calculation_type (string): Calculation type
        calculation (Calculation): Calculation object
        atomic_symbols (list): List of atomic symbols
        coordinates (array): Atomic coordinates

    """

    ghost_atom_present = any("X" in atom for atom in atomic_symbols)

    # Single point energy
    if calculation_type == "SPE": 
        
        if calculation.extrapolate:

            energ.extrapolate_energy(calculation, atomic_symbols, coordinates)

        else:
            
            energ.calculate_energy(calculation, atomic_symbols, coordinates)


    # Coordinate scan
    elif calculation_type == "SCAN":

        if calculation.scan_step:
            if calculation.scan_number: 
                
                energ.scan_coordinate(calculation, atomic_symbols, coordinates)
                
            else: error(f"Coordinate scan requested but no number of steps given by keyword \"NUM\"!")
        else: error(f"Coordinate scan requested but no step size given by keyword \"STEP\"!")
        

    # Geometry optimisation
    elif calculation_type == "OPT" or calculation_type == "FORCE":
        
        if not len(atomic_symbols) == 1 and not ghost_atom_present: 
            
            multiple_iterations = False if calculation_type == "FORCE" else True

            optfreq.optimise_geometry(calculation, atomic_symbols, coordinates, multiple_iterations=multiple_iterations)
        
        else: error("Geometry optimisation requested for single atom!")
        

    # Harmonic frequency
    elif calculation_type == "FREQ":

        if not len(atomic_symbols) == 1 and not ghost_atom_present: 
            
            optfreq.calculate_frequency(calculation, atomic_symbols=atomic_symbols, coordinates=coordinates)
        
        else: error("Harmonic frequency requested for single atom!")
        
        
    # Geometry optimisation and harmonic frequency
    elif calculation_type == "OPTFREQ":

        if not len(atomic_symbols) == 1 and not ghost_atom_present: 
            
            optimised_molecule, optimised_energy = optfreq.optimise_geometry(calculation, atomic_symbols, coordinates)
            optfreq.calculate_frequency(calculation, molecule=optimised_molecule, energy=optimised_energy)

        else: error("Geometry optimisation requested for single atom!")
        
        
    # Ab initio molecular dynamics
    elif calculation_type == "MD":

        # Turns on printing the trajectory only if NOTRAJ parameter has not been used
        if not calculation.no_trajectory: calculation.trajectory = True

        if not len(atomic_symbols) == 1 and not ghost_atom_present: 
            
            md.run_MD(calculation, atomic_symbols, coordinates)
        
        else: error("Molecular dynamics simulation requested for single atom!")
        
        



        

def main(): 

    """

    Sets off TUNA calculation by parsing input line, building calculation object and handing off to relevant modules.

    """

    # Reads input line, makes sure it's okay and extracts the desired parameters
    calculation_type, method, basis, atomic_symbols, coordinates, params = parse_input()
    
    print(colored(f"{calculation_types.get(calculation_type)} calculation in {basis_types.get(basis)} basis set requested.", "light_grey", force_color=True))
    print(colored(f"Electronic structure method is {method_types.get(method)}.\n", "light_grey", force_color=True))

    # Builds calculation object which holds onto all the fundamental and derived parameters, passed through most functions in TUNA
    calculation = Calculation(calculation_type, method, start_time, params, basis, atomic_symbols)

    # If a decontracted basis has been requested, this is printed to the console
    contraction = "fully decontracted" if calculation.decontract else "partially contracted"
    print(colored(f"Setting up calculation using {contraction} basis set.", "light_grey", force_color=True))

    print(colored(f"\nDistances in angstroms and times in femtoseconds. Everything else in atomic units.", "light_grey", force_color=True))

    # Sets off the desired calculation with the requested parameters
    run_calculation(calculation_type, calculation, atomic_symbols, coordinates)

    # Finishes the calculation, printing the time taken
    finish_calculation(calculation)





if __name__ == "__main__": 

    try:

        while True:

            main()

    except KeyboardInterrupt: 
        
        error("The TUNA calculation has been interrupted by the user. Goodbye!")
    

