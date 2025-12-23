import numpy as np
from tuna_util import *

k = constants.k
c = constants.c
h = constants.h


def calculate_translational_internal_energy(temperature): 

    """

    Calculates translational contribution to internal energy.

    Args:
        temperature (float): Temperature in kelvin

    Returns:
        (3 / 2) * k * temperature (float): Translational internal energy

    """

    return (3 / 2) * k * temperature






def calculate_rotational_internal_energy(temperature): 
    
    """

    Calculates rotational contribution to internal energy.

    Args:
        temperature (float): Temperature in kelvin

    Returns:
        k * temperature (float): Rotational internal energy

    """
    
    return k * temperature





def calculate_vibrational_internal_energy(frequency_per_cm, temperature): 
    
    """

    Calculates vibrational contribution to internal energy.

    Args:   
        frequency_per_cm (float): Frequency in per cm
        temperature (float): Temperature in kelvin

    Returns:
        vibrational_internal_energy (float): Vibrational internal energy

    """
     
    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)
    
    # Makes sure an error message isn't printed when dividing by a very small number
    with np.errstate(divide='ignore'):
        
        vibrational_internal_energy = k * vibrational_temperature / (np.exp(vibrational_temperature / temperature) - 1)


    return vibrational_internal_energy







def calculate_internal_energy(E, E_ZPE, temperature, frequency_per_cm):

    """

    Calculates total internal energy.

    Args:   
        E (float): Molecular energy
        E_ZPE (float): Zero-point energy
        temperature (float): Temperature in kelvin
        frequency_per_cm (float): Frequency in per cm

    Returns:
        U (float): Internal energy
        translational_internal_energy (float): Translational internal energy
        rotational_internal_energy (float): Rotational internal energy
        vibrational_internal_energy (float): Vibrational internal energy
    
    """

    translational_internal_energy = calculate_translational_internal_energy(temperature)
    rotational_internal_energy = calculate_rotational_internal_energy(temperature)
    vibrational_internal_energy = calculate_vibrational_internal_energy(frequency_per_cm, temperature)

    # Adds together all contributions to internal energy
    U = E + E_ZPE + translational_internal_energy + rotational_internal_energy + vibrational_internal_energy

    return U, translational_internal_energy, rotational_internal_energy, vibrational_internal_energy







def calculate_translational_entropy(temperature, pressure, mass):

    """

    Calculates translational contribution to entropy.

    Args:   
        temperature (float): Temperature in kelvin
        pressure (float): Pressure in atomic units
        mass (float): Mass in atomic units

    Returns:
        translational_entropy (float): Translational entropy

    """

    pressure_atomic_units = pressure / constants.pascal_in_atomic_units

    translational_entropy = k * (5 / 2 + np.log(np.sqrt(((h * mass * k * temperature) / (h ** 2))) ** 3 * (k * temperature / pressure_atomic_units)))

    return translational_entropy







def calculate_rotational_entropy(point_group, temperature, rotational_constant_per_m):
    
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

    if point_group == "Dinfh": symmetry_number = 2
    elif point_group == "Cinfv": symmetry_number = 1

    rotational_entropy = k * (1 + np.log(k * temperature / (symmetry_number * rotational_constant_per_bohr * h * c)))

    return rotational_entropy







def calculate_vibrational_entropy(frequency_per_cm, temperature):

    """

    Calculates vibrational contribution to entropy.

    Args:   
        frequency_per_cm (string): Frequency in per cm
        temperature (float): Temperature in kelvin

    Returns:
        vibrational_entropy (float): Vibrational entropy

    """

    vibrational_temperature = calculate_vibrational_temperature(frequency_per_cm)

    vibrational_entropy = k * (vibrational_temperature / (temperature * (np.exp(vibrational_temperature / temperature) - 1)) - np.log(1 - np.exp(-vibrational_temperature / temperature)))

    return vibrational_entropy






def calculate_electronic_entropy(): 
    
    """

    Calculates electronic contribution to entropy.

    Args:   
        None : This function does not require arguments

    Returns:
        0 : Assumes molecule is only in the ground state

    """
    
    return 0






def calculate_entropy(temperature, frequency_per_cm, point_group, rotational_constant_per_m, masses, pressure):

    """

    Calculates total entropy.

    Args:   
        temperature (float): Temperature in kelvin
        frequency_per_cm (float): Frequency in per cm
        point_group (string): Molecular point group
        rotational_constant_per_m (float): Rotational constant in per m
        masses (array): Array of masses in atomic units
        pressure (float): Pressure in atomic units

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
    vibrational_entropy = calculate_vibrational_entropy(frequency_per_cm, temperature)
    electronic_entropy = calculate_electronic_entropy()

    # Total entropy is just the sum of all the contributions
    S = translational_entropy + rotational_entropy + vibrational_entropy + electronic_entropy

    return S, translational_entropy, rotational_entropy, vibrational_entropy, electronic_entropy






def calculate_vibrational_temperature(frequency_per_cm):

    """

    Calculates vibrational temperature.

    Args:   
        frequency_per_cm (float): Frequency in per cm

    Returns:
        vibrational_temperature (float) : Vibrational temperature in kelvin

    """

    frequency_per_bohr = bohr_to_angstrom(frequency_per_cm) * 1e-8

    vibrational_temperature = h * frequency_per_bohr * c / k

    return vibrational_temperature






def calculate_enthalpy(U, temperature): 
    
    """

    Calculates enthalpy.

    Args:   
        U (float): Internal energy
        temperature (float): Temperature in kelvin

    Returns:
        U + k * temperature (float) : Enthalpy

    """

    return U + k * temperature





def calculate_free_energy(H, temperature, S):
        
    """

    Calculates Gibbs free energy.

    Args:   
        H (float): Enthalpy
        temperature (float): Temperature in kelvin
        S (float): Entropy

    Returns:
        H - temperature * S (float) : Gibbs free energy

    """
    
    return H - temperature * S