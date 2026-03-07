import numpy as np
from tuna_util import *


"""

This is the TUNA module for calculating thermochemical corrections to the molecular energy, written first for version 0.4.0 and rewritten for version 0.10.0.

The module contains:

1. Functions to calculate contributions to internal energy
2. Functions to calculate contributions to entropy
3. Functions to calculate contributions to enthalpy and free energy
4. The main routine to print all thermochemical information in a table

"""



# Boltzmann constant, speed of light and Planck constant all in atomic units

k = constants.k
c = constants.c
h = constants.h




def calculate_translational_internal_energy(temperature: float) -> float: 

    """

    Calculates translational contribution to internal energy.

    Args:
        temperature (float): Temperature in kelvin

    Returns:
        translational_internal_energy (float): Translational internal energy

    """

    translational_internal_energy = (3 / 2) * k * temperature

    return translational_internal_energy








def calculate_rotational_internal_energy(temperature: float) -> float: 
    
    """

    Calculates rotational contribution to internal energy.

    Args:
        temperature (float): Temperature in kelvin

    Returns:
       rotational_internal_energy (float): Rotational internal energy

    """
    
    rotational_internal_energy = k * temperature

    return rotational_internal_energy








def calculate_vibrational_internal_energy(vibrational_frequency: float, temperature: float) -> float: 
    
    """

    Calculates vibrational contribution to internal energy.

    Args:   
        vibrational_frequency (float): Frequency in hartree
        temperature (float): Temperature in kelvin

    Returns:
        vibrational_internal_energy (float): Vibrational internal energy

    """
     
    vibrational_temperature = calculate_vibrational_temperature(vibrational_frequency)
    
    # Makes sure an error message isn't printed when dividing by a very small number
    with np.errstate(divide="ignore"):
        
        vibrational_internal_energy = k * vibrational_temperature / (np.exp(vibrational_temperature / temperature) - 1)


    return vibrational_internal_energy







def calculate_internal_energy(energy: float, zero_point_energy: float, temperature: float, vibrational_frequency: float) -> tuple[float, float, float, float]:

    """

    Calculates total internal energy.

    Args:   
        energy (float): Molecular energy
        zero_point_energy (float): Zero-point energy
        temperature (float): Temperature in kelvin
        vibrational_frequency (float): Frequency in hartree

    Returns:
        internal_energy (float): Internal energy
        translational_internal_energy (float): Translational internal energy
        rotational_internal_energy (float): Rotational internal energy
        vibrational_internal_energy (float): Vibrational internal energy
    
    """

    translational_internal_energy = calculate_translational_internal_energy(temperature)
    rotational_internal_energy = calculate_rotational_internal_energy(temperature)
    vibrational_internal_energy = calculate_vibrational_internal_energy(vibrational_frequency, temperature)

    # Adds together all contributions to internal energy
    internal_energy = energy + zero_point_energy + translational_internal_energy + rotational_internal_energy + vibrational_internal_energy

    return internal_energy, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy







def calculate_translational_entropy(temperature: float, pressure: float, mass: float) -> float:

    """

    Calculates translational contribution to entropy.

    Args:   
        temperature (float): Temperature in kelvin
        pressure (float): Pressure in atomic units
        mass (float): Total molecular mass in atomic units

    Returns:
        translational_entropy (float): Translational entropy

    """

    pressure_atomic_units = pressure / constants.pascal_in_atomic_units

    translational_entropy = k * (5 / 2 + np.log(np.sqrt(((h * mass * k * temperature) / h ** 2)) ** 3 * (k * temperature / pressure_atomic_units)))

    return translational_entropy








def calculate_rotational_entropy(point_group: str, temperature: float, rotational_constant_per_m: float) -> float:
    
    """

    Calculates rotational contribution to entropy.

    Args:   
        point_group (string): Molecular point group
        temperature (float): Temperature in kelvin
        rotational_constant_per_m (float): Rotational constant in per m

    Returns:
        rotational_entropy (float): Rotational entropy

    """

    rotational_constant_per_bohr = bohr_to_angstrom(rotational_constant_per_m) * 1e-10

    symmetry_number = 2 if point_group == "Dinfh" else 1

    rotational_entropy = k * (1 + np.log(k * temperature / (symmetry_number * rotational_constant_per_bohr * h * c)))

    return rotational_entropy








def calculate_vibrational_entropy(vibrational_frequency: float, temperature: float) -> float:

    """

    Calculates vibrational contribution to entropy.

    Args:   
        vibrational_frequency (float): Frequency in hartree
        temperature (float): Temperature in kelvin

    Returns:
        vibrational_entropy (float): Vibrational entropy

    """

    vibrational_temperature = calculate_vibrational_temperature(vibrational_frequency)

    vibrational_entropy = k * (vibrational_temperature / (temperature * (np.exp(vibrational_temperature / temperature) - 1)) - np.log(1 - np.exp(-vibrational_temperature / temperature)))

    return vibrational_entropy








def calculate_electronic_entropy(multiplicity: int) -> float: 
    
    """

    Calculates electronic contribution to entropy. Assumes molecule is only in the ground state.

    Args:   
        multiplicity (int) : Multiplicity

    Returns:
        electronic_entropy (float) : Electronic contribution to entropy

    """
    
    electronic_entropy = k * np.log(multiplicity)

    return electronic_entropy







def calculate_entropy(temperature: float, vibrational_frequency: float, point_group: str, rotational_constant_per_m: float, masses: np.ndarray, pressure: float, multiplicity: int) -> tuple[float, float, float, float, float]:
    
    """

    Calculates total entropy.

    Args:   
        temperature (float): Temperature in kelvin
        vibrational_frequency (float): Frequency in hartree
        point_group (string): Molecular point group
        rotational_constant_per_m (float): Rotational constant in per m
        masses (array): Array of masses in atomic units
        pressure (float): Pressure in atomic units
        multiplicity (int): Molecular multiplicity

    Returns:
        S (float) : Total entropy
        translational_entropy (float) : Total entropy
        translational_entropy (float) : Translational entropy
        rotational_entropy (float) : Rotational entropy
        vibrational_entropy (float) : Vibrational entropy
        electronic_entropy (float) : Electronic entropy

    """

    total_mass = np.sum(masses)

    translational_entropy = calculate_translational_entropy(temperature, pressure, total_mass)
    rotational_entropy = calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m)
    vibrational_entropy = calculate_vibrational_entropy(vibrational_frequency, temperature)
    electronic_entropy = calculate_electronic_entropy(multiplicity)

    # Total entropy is just the sum of all the contributions
    S = translational_entropy + rotational_entropy + vibrational_entropy + electronic_entropy

    return S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy







def calculate_vibrational_temperature(vibrational_frequency: float) -> float:

    """

    Calculates vibrational temperature.

    Args:   
        vibrational_frequency (float): Frequency in hartree

    Returns:
        vibrational_temperature (float) : Vibrational temperature in kelvin

    """

    vibrational_temperature = vibrational_frequency / k

    return vibrational_temperature







def calculate_enthalpy(internal_energy: float, temperature: float) -> float: 
    
    """

    Calculates enthalpy.

    Args:   
        internal_energy (float): Internal energy
        temperature (float): Temperature in kelvin

    Returns:
        H (float) : Enthalpy

    """

    H = internal_energy + k * temperature

    return H






def calculate_free_energy(H: float, temperature: float, S: float) -> float:
        
    """

    Calculates Gibbs free energy.

    Args:   
        H (float): Enthalpy in hartree
        temperature (float): Temperature in kelvin
        S (float): Entropy in hartree per kelvin

    Returns:
       G (float) : Gibbs free energy

    """

    G = H - temperature * S
    
    return G







def calculate_thermochemical_corrections(molecule: any, calculation: any, vibrational_frequency: float, energy: float, zero_point_energy: float) -> float:

    """

    Calculates and prints the thermochemical corrections to energy.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        vibrational_frequency (float): Vibrational frequency in hartree
        energy (float): Molecular energy in hartree
        zero_point_energy (float): Zero-point vibrational energy in hartree

    Returns:
        G (float): Gibbs free energy in hartree

    """

    # Extracts useful quantities
    point_group = molecule.point_group
    rotational_constant_per_cm = molecule.rotational_constant_per_cm
    masses = molecule.masses
    multiplicity = molecule.multiplicity

    temperature = calculation.temperature
    pressure = calculation.pressure

    # Prints thermochemical information unless terse keyword is used 
    log(f"\n Temperature used is {temperature:.2f} K, pressure used is {(pressure)} Pa.", calculation, 2)
    log(" Entropies multiplied by temperature to give units of energy.", calculation, 2)
    log(f" Using symmetry number derived from {point_group} point group for rotational entropy.", calculation, 2)

    internal_energy, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy = calculate_internal_energy(energy, zero_point_energy, temperature, vibrational_frequency)

    H = calculate_enthalpy(internal_energy, temperature)

    S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy = calculate_entropy(temperature, vibrational_frequency, point_group, rotational_constant_per_cm * 100, masses, pressure, multiplicity)
    
    G = calculate_free_energy(H, temperature, S)

    # All values are printed in units of hartree, for consistency

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log("                                   Thermochemistry", calculation, 2, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)
    log(f"  Electronic energy:   {energy:16.10f}     Electronic entropy:   {temperature * electronic_entropy:16.10f}", calculation, 2)
    log(f"\n  Translational energy:{translational_internal_energy:16.10f}     Translational entropy:{temperature * translational_entropy:16.10f}", calculation, 2)
    log(f"  Rotational energy:   {rotational_internal_energy:16.10f}     Rotational entropy:   {temperature * rotational_entropy:16.10f}", calculation, 2)
    log(f"  Vibrational energy:  {vibrational_internal_energy:16.10f}     Vibrational entropy:  {temperature * vibrational_entropy:16.10f}  ", calculation, 2)
    log(f"  Zero-point energy:   {zero_point_energy:16.10f}", calculation, 2)
    log(f"\n  Internal energy:     {internal_energy:16.10f}", calculation, 2)
    log(f"  Enthalpy:            {H:16.10f}     Entropy:              {temperature * S:16.10f}", calculation, 2)
    log(f"\n  Gibbs free energy:   {G:16.10f}     Non-electronic energy:{energy - G:16.10f}", calculation, 2)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 2)

    return G