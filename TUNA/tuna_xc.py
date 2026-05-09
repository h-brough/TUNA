import numpy as np
from numpy import ndarray
from sympy import zeta
from tuna_calc import Calculation
from tuna_util import constants


"""

This is the TUNA module for exchange-correlation functionals, written first for version 0.11.0.

The formulae for the various density functional approximations in TUNA are found here - a lower level module than tuna_dft, which calculates 
the exchange and correlation matrices and sets up the grid. Note that throughout this module, we use ** (1 / 2) for square rooting, and 
np.cbrt() for cube rooting - this inconsistency is on purpose, asthese options seem to be significantly faster and more accurate than their 
counterparts. We also cube via x * x * x rather than x ** 3 for the same reason, although higher powers do not benefit as much. Optimising all 
of these low level operations makes quite a big difference to the speed; I observed about a factor of two increase for DFT functionals generally 
by optimising the cubing and cube rooting.

The module contains:

1. Some small utility functions (calculate_zeta, clean, etc.)
2. The complicated, horrible, explicit functional expressions for all the exchange and correlation functionals
3. Some dictionaries for all the implemented exchange and correlation functionals

"""



def clean(function_on_grid: ndarray, floor: float = constants.density_floor) -> ndarray:

    """
    
    Cleans a function on the integration grid.

    Args:
        function_on_grid (array): Function evaluated on integration grid
        floor (array): Minimum accepted value of function
    
    Returns:
        cleaned_function_on_grid (array): Cleaned function on integration grid
        
    """

    # Makes sure there are no zero or negative values in the electron density
    
    cleaned_function_on_grid = np.maximum(function_on_grid, floor)


    return cleaned_function_on_grid










def calculate_seitz_radius(density: ndarray) -> tuple[ndarray, ndarray]:

    """
    
    Calculates the Seitz radius.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        r_s (array): Seitz radius
        inv_density (array): Inverse density
    
    """

    # Cube rooting via numpy is faster and more precise than in pure Python

    inv_density = 1 / density

    r_s = np.cbrt(3 / (4 * np.pi) * inv_density)

    return r_s, inv_density










def calculate_zeta(alpha_density: ndarray, beta_density: ndarray) -> ndarray:

    """
    
    Calculates the local spin polarisation, zeta.

    Args:
        alpha_density (array): Alpha density
        beta_density (array): Alpha density
    
    Returns:
        zeta (array): Local spin polarisation
    
    """

    zeta = (alpha_density - beta_density) / (alpha_density + beta_density)

    # Makes sure there is no no numerical weirdness

    zeta = np.clip(zeta, -1, 1)

    return zeta










def calculate_f_zeta(zeta: ndarray) -> ndarray:

    """
    
    Calculates the spin polarisation function used in eg. VWN5 correlation in interpolation.

    Args:
        zeta (array): Spin polarisation
    
    Returns:
        f_zeta (array): Spin polarisation function
    
    """

    f_zeta = (np.cbrt(1 + zeta) ** 4 + np.cbrt(1 - zeta) ** 4 - 2) / (np.cbrt(2) ** 4 - 2)

    return f_zeta










def calculate_f_prime_zeta(zeta: ndarray) -> ndarray:

    """
    
    Calculates the derivative of spin polarisation function used in eg. VWN5 correlation in interpolation.

    Args:
        zeta (array): Spin polarisation
    
    Returns:
        f_prime_zeta (array): Derivative of spin polarisation function
    
    """

    f_prime_zeta = (np.cbrt(1 + zeta) - np.cbrt(1 - zeta)) / (np.cbrt(2) ** 4 - 2) * 4 / 3

    return f_prime_zeta










def calculate_Fermi_wavevector(density: ndarray = None, cbrt_density: ndarray = None) -> ndarray:

    """
    
    Calculates the Fermi wavevector from the electron density.

    Args:
        density (array, optional): Electron density on grid
        cbrt_density (array, optional): Cube root electron density on grid

    Returns:
        k_F (array): Fermi wavevector
    
    """

    k_F = np.cbrt(3 * np.pi ** 2)

    # Uses cube root density if this is already calculated

    k_F *= np.cbrt(density) if density is not None else cbrt_density

    return k_F










def calculate_Slater_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the Slater exchange energy density and derivative with respect to the density.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        e_X (array): Slater exchange energy density per particle
    
    """

    # Modifiable with "XA" keyword

    alpha = calculation.X_alpha

    # Calculates the derivative

    df_dn = - (3 / 2 * alpha) * np.cbrt(3 / np.pi * density)

    # Energy density is proportional to the derivative for Slater exchange

    e_X = 3 / 4 * df_dn

    return df_dn, None, None, e_X










def calculate_PBE_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the PBE exchange energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): PBE exchange energy density per particle
    
    """

    # Parameters for PBE - mu may be rounded differently

    kappa, mu = 0.804, 0.21952

    if calculation.functional.x_functional == "REVPBE":

        # This is the only difference between "revised" and regular PBE

        kappa = 1.245

    # Reduced density gradient

    s_squared = sigma / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)
    
    denom = 1 / (1 + mu / kappa * s_squared)

    # Exchange enhancement function

    F_X = 1 + kappa - kappa * denom

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Generalised gradient approximation exchange

    e_X = e_X_LDA * F_X

    # Derivative of the denominator in the exchange enhancement function

    denom_derivative = mu * s_squared * denom * denom

    # Derivatives with respect to sigma and the density

    df_ds = density * e_X_LDA * denom_derivative / sigma
    df_dn = (4 / 3) * e_X_LDA * (F_X - 2 * denom_derivative) 

    return df_dn, df_ds, None, e_X










def calculate_RPBE_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the RPBE ("modified" PBE) exchange energy density and derivative with respect to the density and square gradient.

    Implemented from 10.1103/PhysRevB.59.7413.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): RPBE exchange energy density per particle
    
    """

    # Parameters for RPBE

    kappa, mu = 0.804, 0.21952

    # Reduced density gradient

    s_squared = sigma / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)
    
    # Exchange enhancement function

    exponent_term = np.exp(-mu * s_squared / kappa)

    F_X = 1 + kappa * (1 - exponent_term)

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Generalised gradient approximation exchange

    e_X = e_X_LDA * F_X

    # Derivative of the enhancement function

    enhancement_derivative = mu * exponent_term * s_squared 

    # Derivatives with respect to sigma and the density

    df_ds = density * e_X_LDA * enhancement_derivative / sigma
    df_dn = (4 / 3) * e_X_LDA * (F_X - 2 * enhancement_derivative)

    return df_dn, df_ds, None, e_X










def calculate_B88_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the Becke 1988 exchange energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): Becke 1988 exchange energy density per particle
    
    """

    # This is the only adjustable parameter for B88 exchange

    beta = 0.0042 
    C = 2 / np.cbrt(4)

    # Calculates the local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density / 2, sigma, tau, calculation)

    # This is the form of the reduced density gradient, x, for Becke 1988 exchange

    cube_root_density = np.cbrt(density / 2) 
    x = (sigma / 4) ** (1 / 2) / cube_root_density ** 4

    x_squared = x * x
    A = np.arcsinh(x)

    D = 1 + 6 * beta * x * A
    D_squared = D * D
    dD_dx = 6 * beta * (A + x / (1 + x_squared) ** (1 / 2))

    # Exchange energy density per particle

    e_X = C * e_X_LDA - beta * cube_root_density * x_squared / D

    # Derivative with respect to the density

    df_dn = (e_X + C * e_X_LDA / 3 + beta * cube_root_density * (7 * x_squared * D - 4 * x_squared * x * dD_dx) / (3 * D_squared)) 

    # Derivative with respect to the square density gradient, sigma

    df_ds = -beta * density * cube_root_density * (x_squared * D - (1 / 2) * x_squared * x * dD_dx) / (sigma * D_squared)
    

    return df_dn, df_ds, None, e_X










def calculate_PW91_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the Perdew-Wang 1991 exchange energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): Perdew-Wang 1991 exchange energy density per particle
    
    """

    # These are the adjustable parameters for PW91 exchange

    a, b, c, d, e, f = 0.19645, 7.7956, 0.2743, 0.1508, 100, 0.004

    # The local density exchange

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # The reduced density gradient

    denom = 1 / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)
    s_squared = sigma * denom
    s = s_squared ** (1 / 2)
    
    # A useful repeatedly used value

    u = b * s
    A = np.arcsinh(u)
    E = np.exp(-e * s_squared)

    x = 1 + a * s * A
    numerator = x + (c - d * E) * s_squared
    denominator = x + f * s_squared * s_squared

    # The exchange enhancement function

    F_X = numerator / denominator

    # The exchange energy density per particle

    e_X = e_X_LDA * F_X
    f_LDA = density * e_X_LDA

    dx_ds = (1 / 2) * a * (A / s + b / (1 + u * u) ** (1 / 2))
    dA_ds = c + d * E * (e * s_squared - 1)
    dF_ds = ((dx_ds + dA_ds) * denominator - numerator * (dx_ds + 2 * f * s_squared)) / (denominator * denominator)

    # The derivatives with respect to the sigma and the density

    df_ds = f_LDA * dF_ds * denom
    df_dn = df_dn_LDA * F_X - f_LDA * dF_ds * (8 / 3) * s_squared / density

    return df_dn, df_ds, None, e_X










def calculate_mPW91_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
  
    """
    
    Calculates the modified Perdew-Wang 1991 exchange energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): Modified Perdew-Wang 1991 exchange energy density per particle
    
    """
  
    # These are the parameters for mPW exchange

    beta = 5 / np.cbrt(36 * np.pi) ** 5
    b, c, d, eps = 0.00426, 1.6455, 3.72, 1e-6

    # Local density exchange energy

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # The factor of two here maintains the spin-scaling relationship for exchange

    cbrt_density = np.cbrt(density / 2) 

    # Reduced density gradient

    x = sigma ** (1 / 2) / (density * cbrt_density)
    x_squared = x * x
    x_power_d = x ** d

    G = np.exp(-c * x_squared)
    A = np.arcsinh(x)

    K = e_X_LDA / cbrt_density

    # Calculates the function for exchange enhancement over LDA

    N = b * x_squared - (b - beta) * x_squared * G - eps * x_power_d
    D = 1 + 6 * b * x * A - eps * x_power_d / K
    F_X = N / D

    x_power_d_over_x = x_power_d / x

    dN_dx = 2 * b * x - 2 * (b - beta) * x * G * (1 - c * x_squared) - eps * d * x_power_d_over_x
    dD_dx = 6 * b * (A + x / (1 + x_squared) ** (1 / 2)) - eps * d * x_power_d_over_x / K

    # Derivative of enhancement over exchange function

    dF_dx = (dN_dx - F_X * dD_dx) / D

    # Exchange energy per particle for mPW exchange

    e_X = e_X_LDA - F_X * cbrt_density

    # Derivative with respect to density

    df_dn = df_dn_LDA - (4 / 3) * cbrt_density * (F_X - x * dF_dx)

    # Derivative with respect to sigma

    df_ds = -dF_dx / (2 * x * density * cbrt_density)

    return df_dn, df_ds, None, e_X










def calculate_TPSS_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the TPSS exchange energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        df_dt (array): Derivative of f = n * e_X with respect to tau
        e_X (array): TPSS exchange energy density per particle
    
    """

    # Parameters that define TPSS - same kappa and mu as PBE

    b, c, e, kappa, mu = 0.40, 1.59096, 1.537, 0.804, 0.21951

    # This is the local density exchange

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Reduced density gradients for TPSS

    p = sigma / (den_p := 4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)

    z = sigma / (8 * density * tau)

    # This is the "iso-orbital indicator", which interpolates between the von Weiszacker tau and the kinetic energy density of the uniform electron gas
    
    alpha = (5 * p / 3) * (1 / z - 1)

    # Equation straight from TPSS paper

    q_tilde = (9 / 20) * (alpha - 1) / (1 + b * alpha * (alpha - 1)) ** (1 / 2) + 2 * p / 3

    sqrt_e = e ** (1 / 2)
    z_squared = z ** 2
    t1 = 1 + z_squared
    A = 10 / 81 + c * z_squared / (t1 * t1)
    S = ((1 / 2) * ((3 / 5 * z) ** 2 + p * p)) ** (1 / 2)

    # Horrible equations from TPSS paper

    num = A * p + (146 / 2025) * q_tilde * q_tilde - (73 / 405) * q_tilde * S + (10 / 81) ** 2 / kappa * p ** 2 + 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z_squared + e * mu * p * p * p
    t = 1 + sqrt_e * p
    den = t * t
    x = num / den

    # Function for enhancement over local exchange

    F_X = 1 + kappa - kappa ** 2 / (kappa + x)

    # Exchange energy density per particle for TPSS
    
    e_X = e_X_LDA * F_X

    # Factors used for derivatives

    dp = np.stack([-(8 / 3) * p / density, 1 / den_p, np.zeros_like(p)])
    dz = np.stack([-z / density, 1 / (8 * density * tau), -z / tau])

    inv_z = 1 / z   
    
    # Derivative of the iso-orbital indicator

    dalpha = (5 / 3) * ((inv_z - 1) * dp - p * (inv_z * inv_z) * dz)

    g = 1 + b * alpha * (alpha - 1)
    sqrt_g = g ** (1 / 2)
    dh_dalpha = 1 / sqrt_g - (alpha - 1) * b * (2 * alpha - 1) / (2 * g * sqrt_g)
    dq = (9 / 20) * dh_dalpha * dalpha + (2 / 3) * dp

    dA_dz = c * 2 * z * (1 - z_squared) / (t1 * t1 * t1)
    dS = ((3 / 5) ** 2 * z * dz + p * dp) / (2 * S)
    
    # These equations are not necessarily the most efficient way of computing the derivatives

    dnum = (A * dp + p * dA_dz * dz) + 2 * (146 / 2025) * q_tilde * dq - (73 / 405) * (dq * S + q_tilde * dS) + 2 * (10 / 81) ** 2 / kappa * p * dp + 2 * 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z * dz + 3 * e * mu * p * p * dp
    dx = (dnum * den - num * (2 * sqrt_e * t * dp)) / (den * den)
    dF = (kappa / (kappa + x)) * (kappa / (kappa + x)) * dx

    f_LDA = density * e_X_LDA

    # Derivatives with respect to density, sigma and tau

    df_dn = df_dn_LDA * F_X + f_LDA * dF[0]
    df_ds = f_LDA * dF[1]
    df_dt = f_LDA * dF[2]


    return df_dn, df_ds, df_dt, e_X










def calculate_revTPSS_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the revised TPSS exchange energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        df_dt (array): Derivative of f = n * e_X with respect to tau
        e_X (array): Revised TPSS exchange energy density per particle
    
    """

    # Parameters that define revTPSS

    b, c, e, kappa, mu = 0.40, 2.35204, 2.1677, 0.804, 0.14

    # This is the local density exchange

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Reduced density gradients for TPSS

    p = sigma / (den_p := 4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)

    z = sigma / (8 * density * tau)

    # This is the "iso-orbital indicator", which interpolates between the von Weiszacker tau and the kinetic energy density of the uniform electron gas
    
    alpha = (5 * p / 3) * (1 / z - 1)

    # Equation straight from TPSS paper

    q_tilde = (9 / 20) * (alpha - 1) / (1 + b * alpha * (alpha - 1)) ** (1 / 2) + 2 * p / 3

    sqrt_e = e ** (1 / 2)
    z_squared = z * z
    z_cubed = z_squared * z
    t1 = 1 + z_squared
    A = 10 / 81 + c * z_cubed / (t1 * t1)
    S = ((1 / 2) * ((3 / 5 * z) ** 2 + p * p)) ** (1 / 2)

    # Horrible equations from TPSS paper

    num = A * p + (146 / 2025) * q_tilde * q_tilde - (73 / 405) * q_tilde * S + (10 / 81) ** 2 / kappa * p ** 2 + 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z_squared + e * mu * p * p * p
    t = 1 + sqrt_e * p
    den = t * t
    x = num / den

    # Function for enhancement over local exchange

    F_X = 1 + kappa - kappa ** 2 / (kappa + x)

    # Exchange energy density per particle for TPSS
    
    e_X = e_X_LDA * F_X

    # Factors used for derivatives

    dp = np.stack([-(8 / 3) * p / density, 1 / den_p, np.zeros_like(p)])
    dz = np.stack([-z / density, 1 / (8 * density * tau), -z / tau])

    inv_z = 1 / z   
    
    # Derivative of the iso-orbital indicator

    dalpha = (5 / 3) * ((inv_z - 1) * dp - p * (inv_z * inv_z) * dz)

    g = 1 + b * alpha * (alpha - 1)
    sqrt_g = g ** (1 / 2)
    dh_dalpha = 1 / sqrt_g - (alpha - 1) * b * (2 * alpha - 1) / (2 * g * sqrt_g)
    dq = (9 / 20) * dh_dalpha * dalpha + (2 / 3) * dp
    
    dA_dz = c * z_squared * (3 - z_squared) / (t1 * t1 * t1)
    dS = ((3 / 5) ** 2 * z * dz + p * dp) / (2 * S)
    
    # These equations are not necessarily the most efficient way of computing the derivatives

    dnum = (A * dp + p * dA_dz * dz) + 2 * (146 / 2025) * q_tilde * dq - (73 / 405) * (dq * S + q_tilde * dS) + 2 * (10 / 81) ** 2 / kappa * p * dp + 2 * 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z * dz + 3 * e * mu * p * p * dp
    dx = (dnum * den - num * (2 * sqrt_e * t * dp)) / (den * den)
    dF = (kappa / (kappa + x)) * (kappa / (kappa + x)) * dx

    f_LDA = density * e_X_LDA

    # Derivatives with respect to density, sigma and tau

    df_dn = df_dn_LDA * F_X + f_LDA * dF[0]
    df_ds = f_LDA * dF[1]
    df_dt = f_LDA * dF[2]


    return df_dn, df_ds, df_dt, e_X










def calculate_SCAN_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the SCAN exchange energy density and derivative with respect to the density, square gradient and kinetic energy density.

    A reasonably efficient implementation of equations in the supplementary information of 10.1103/PhysRevLett.115.036402.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        df_dt (array): Derivative of f = n * e_X with respect to tau
        e_X (array): SCAN exchange energy density per particle
    
    """

    a_1 = 4.9479
    c_1 = 0.667
    c_2 = 0.8
    k_0 = 0.174
    k_1 = 0.065
    mu = 10 / 81
    d_x = 1.24

    b_2 = (5913 / 405000) ** (1 / 2)
    b_1 = (511 / 13500) / (2 * b_2)
    b_3 = 0.5
    b_4 = mu ** 2 / k_1 - 1606 / 18225 - b_1 ** 2

    cbrt_density = np.cbrt(density)
    inv_density = clean(1 / density)

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)
    p = sigma * dp_ds
    p_fourth_root = np.sqrt(np.sqrt(p))

    # The Weiszacker and uniform electron gas kinetic energy densities
    
    dtau_w_ds = inv_density / 8

    tau_w = sigma * dtau_w_ds
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    tau_minus_tau_w = tau - tau_w

    # The regularised version of the iso-orbital indicator

    alpha_denominator = tau_u
    inv_alpha_denominator = 1 / alpha_denominator
    inv_alpha_denominator_squared = inv_alpha_denominator * inv_alpha_denominator

    alpha = tau_minus_tau_w * inv_alpha_denominator

    one_minus_alpha = 1 - alpha
    one_minus_alpha_squared = one_minus_alpha * one_minus_alpha
    inv_one_minus_alpha = 1 / one_minus_alpha
    inv_one_minus_alpha_squared = inv_one_minus_alpha * inv_one_minus_alpha
    
    y_p = (b_4 / mu) * p
    x_term_1 = 1 + y_p * np.exp(-y_p)
    x_term_2 = b_1 * p + b_2 * one_minus_alpha * np.exp(-b_3 * one_minus_alpha_squared)

    x = mu * p * x_term_1 + x_term_2 * x_term_2

    h_0 = 1 + k_0
    h_1 = 1 + k_1 - k_1 / (1 + x / k_1)
    h_0_minus_h_1 = h_0 - h_1

    # A not very smooth switching function

    f_x = np.zeros_like(density)

    small_alpha_exponent_term = np.exp(np.clip(-c_1 * alpha * inv_one_minus_alpha, None, constants.exponent_ceiling))
    large_alpha_exponent_term = -d_x * np.exp(np.clip(c_2 * inv_one_minus_alpha, None, constants.exponent_ceiling))

    f_x = np.where(alpha < 1, small_alpha_exponent_term, f_x)
    f_x = np.where(alpha > 1, large_alpha_exponent_term, f_x)

    g_exponent_term = np.exp(-a_1 / p_fourth_root)

    g_x = 1 - g_exponent_term

    # The exchange enhancement factor

    F_X = (h_1 + f_x * h_0_minus_h_1) * g_x

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # The exchange energy density

    e_X = e_X_LDA * F_X
    f_LDA = e_X_LDA * density

    # Derivative terms begin here
    
    dtau_w_dn = -tau_w * inv_density
    dtau_u_dn = (5 / 3) * tau_u * inv_density
    
    # Derivatives of the iso-orbital indicator
    
    d_alpha_dn = (-dtau_w_dn * alpha_denominator - tau_minus_tau_w * dtau_u_dn) * inv_alpha_denominator_squared

    d_alpha_ds = -inv_alpha_denominator * dtau_w_ds 
    d_alpha_dt = inv_alpha_denominator

    dp_dn = -(8 / 3) * p * inv_density
    dh_1_dx = 1 / ((1 + x / k_1) * (1 + x / k_1))
    dg_dp = -g_exponent_term * (a_1 / 4) * p_fourth_root ** -5  
    dx_dp = mu + b_4 * p * np.exp(-y_p) * (2 - y_p) + 2 * x_term_2 * b_1

    dx_dalpha = -2 * x_term_2 * b_2 * np.exp(-b_3 * one_minus_alpha_squared) * (1 - 2 * b_3 * one_minus_alpha_squared)

    # Derivatives of the h_1 function

    dh_1_dp = dh_1_dx * dx_dp
    dh_1_dalpha_prime = dh_1_dx * dx_dalpha
    
    dh_1_ds = dh_1_dp * dp_ds + dh_1_dalpha_prime * d_alpha_ds
    dh_1_dn = dh_1_dp * dp_dn + dh_1_dalpha_prime * d_alpha_dn
    dh_1_dt = dh_1_dalpha_prime * d_alpha_dt

    # Derivatives of the switching function
    
    df_x_d_alpha_prime = np.zeros_like(density)

    df_x_d_alpha_prime = np.where(alpha < 1, -c_1 * inv_one_minus_alpha_squared * small_alpha_exponent_term, df_x_d_alpha_prime)
    df_x_d_alpha_prime = np.where(alpha > 1, c_2 * inv_one_minus_alpha_squared * large_alpha_exponent_term, df_x_d_alpha_prime)

    # Derivatives of the exchange enhancement function

    dF_dn = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_dn + g_x * (dh_1_dn + df_x_d_alpha_prime * d_alpha_dn * h_0_minus_h_1 - f_x * dh_1_dn)
    dF_ds = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_ds + g_x * (dh_1_ds + df_x_d_alpha_prime * d_alpha_ds * h_0_minus_h_1 - f_x * dh_1_ds)
    dF_dt = g_x * (dh_1_dt + df_x_d_alpha_prime * d_alpha_dt * h_0_minus_h_1 - f_x * dh_1_dt)

    # Final derivatives with respect to density, sigma and tau

    df_dn = f_LDA * dF_dn + df_dn_LDA * F_X 
    df_ds = f_LDA * dF_ds
    df_dt = f_LDA * dF_dt

    return df_dn, df_ds, df_dt, e_X










def calculate_rSCAN_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the rSCAN exchange energy density and derivative with respect to the density, square gradient and kinetic energy density.

    A reasonably efficient implementation of equations in 10.1063/1.5094646.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        df_dt (array): Derivative of f = n * e_X with respect to tau
        e_X (array): rSCAN exchange energy density per particle
    
    """

    eta = 0.0001
    alpha_r = 0.001

    a_1 = 4.9479
    c_1 = 0.667
    c_2 = 0.8
    k_0 = 0.174
    k_1 = 0.065
    mu = 10 / 81
    d_x = 1.24

    b_2 = (5913 / 405000) ** (1 / 2)
    b_1 = (511 / 13500) / (2 * b_2)
    b_3 = 0.5
    b_4 = mu ** 2 / k_1 - 1606 / 18225 - b_1 ** 2

    c_x = [1, -0.667, -0.4445555, -0.663086601049, 1.451297044490, -0.887998041597, 0.234528941479, -0.023185843322]

    cbrt_density = np.cbrt(density)
    inv_density = clean(1 / density)

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)
    p = sigma * dp_ds
    p_fourth_root = np.sqrt(np.sqrt(p))

    # The Weiszacker and uniform electron gas kinetic energy densities
    
    dtau_w_ds = inv_density / 8

    tau_w = sigma * dtau_w_ds
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    tau_minus_tau_w = tau - tau_w

    # The regularised version of the iso-orbital indicator

    alpha_denominator = tau_u + eta
    inv_alpha_denominator = 1 / alpha_denominator
    inv_alpha_denominator_squared = inv_alpha_denominator * inv_alpha_denominator

    alpha = tau_minus_tau_w * inv_alpha_denominator
    alpha_squared = alpha * alpha
    inv_alpha_prime_denominator = 1 / (alpha_squared + alpha_r)

    alpha_prime = alpha_squared * alpha * inv_alpha_prime_denominator

    one_minus_alpha_prime = 1 - alpha_prime
    one_minus_alpha_prime_squared = one_minus_alpha_prime * one_minus_alpha_prime
    inv_one_minus_alpha_prime = 1 / one_minus_alpha_prime
    inv_one_minus_alpha_prime_squared = inv_one_minus_alpha_prime * inv_one_minus_alpha_prime
    
    y_p = (b_4 / mu) * p
    x_term_1 = 1 + y_p * np.exp(-y_p)
    x_term_2 = b_1 * p + b_2 * one_minus_alpha_prime * np.exp(-b_3 * one_minus_alpha_prime_squared)

    x = mu * p * x_term_1 + x_term_2 * x_term_2

    h_0 = 1 + k_0
    h_1 = 1 + k_1 - k_1 / (1 + x / k_1)
    h_0_minus_h_1 = h_0 - h_1

    # Smoother switching function

    f_x = ((((((c_x[7] * alpha_prime + c_x[6]) * alpha_prime + c_x[5]) * alpha_prime + c_x[4]) * alpha_prime + c_x[3]) * alpha_prime + c_x[2]) * alpha_prime + c_x[1]) * alpha_prime + c_x[0]

    # The original limits still apply from SCAN

    small_alpha_exponent_term = np.exp(np.clip(-c_1 * alpha_prime * inv_one_minus_alpha_prime, None, constants.exponent_ceiling))
    large_alpha_exponent_term = -d_x * np.exp(np.clip(c_2 * inv_one_minus_alpha_prime, None, constants.exponent_ceiling))

    f_x = np.where(alpha_prime < 0, small_alpha_exponent_term, f_x)
    f_x = np.where(alpha_prime > 2.5, large_alpha_exponent_term, f_x)

    g_exponent_term = np.exp(-a_1 / p_fourth_root)

    g_x = 1 - g_exponent_term

    # The exchange enhancement factor

    F_X = (h_1 + f_x * h_0_minus_h_1) * g_x

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # The exchange energy density

    e_X = e_X_LDA * F_X
    f_LDA = e_X_LDA * density

    # Derivative terms begin here
    
    dtau_w_dn = -tau_w * inv_density
    dtau_u_dn = (5 / 3) * tau_u * inv_density
    
    # Derivatives of the iso-orbital indicator
    
    d_alpha_prime_d_alpha = alpha_squared * (alpha_squared + 3 * alpha_r) * inv_alpha_prime_denominator * inv_alpha_prime_denominator
    d_alpha_prime_d_tau_w = -d_alpha_prime_d_alpha * inv_alpha_denominator
    d_alpha_dn = (-dtau_w_dn * alpha_denominator - tau_minus_tau_w * dtau_u_dn) * inv_alpha_denominator_squared

    d_alpha_prime_dn = d_alpha_prime_d_alpha * d_alpha_dn    
    d_alpha_prime_ds = d_alpha_prime_d_tau_w * dtau_w_ds
    d_alpha_prime_dt = d_alpha_prime_d_alpha * inv_alpha_denominator

    dp_dn = -(8 / 3) * p * inv_density
    dh_1_dx = 1 / ((1 + x / k_1) * (1 + x / k_1))
    dg_dp = -g_exponent_term * (a_1 / 4) * p_fourth_root ** -5  
    dx_dp = mu + b_4 * p * np.exp(-y_p) * (2 - y_p) + 2 * x_term_2 * b_1

    dx_dalpha_prime = -2 * x_term_2 * b_2 * np.exp(-b_3 * one_minus_alpha_prime_squared) * (1 - 2 * b_3 * one_minus_alpha_prime_squared)

    # Derivatives of the h_1 function

    dh_1_dp = dh_1_dx * dx_dp
    dh_1_dalpha_prime = dh_1_dx * dx_dalpha_prime
    
    dh_1_ds = dh_1_dp * dp_ds + dh_1_dalpha_prime * d_alpha_prime_ds
    dh_1_dn = dh_1_dp * dp_dn + dh_1_dalpha_prime * d_alpha_prime_dn
    dh_1_dt = dh_1_dalpha_prime * d_alpha_prime_dt

    # Derivatives of the switching function, including of the degree-seven polynomial
    
    df_x_d_alpha_prime = ((((((7 * c_x[7] * alpha_prime + 6 * c_x[6]) * alpha_prime + 5 * c_x[5]) * alpha_prime + 4 * c_x[4]) * alpha_prime + 3 * c_x[3]) * alpha_prime + 2 * c_x[2]) * alpha_prime + c_x[1])

    df_x_d_alpha_prime = np.where(alpha_prime < 0, -c_1 * inv_one_minus_alpha_prime_squared * small_alpha_exponent_term, df_x_d_alpha_prime)
    df_x_d_alpha_prime = np.where(alpha_prime > 2.5, c_2 * inv_one_minus_alpha_prime_squared * large_alpha_exponent_term, df_x_d_alpha_prime)

    # Derivatives of the exchange enhancement function

    dF_dn = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_dn + g_x * (dh_1_dn + df_x_d_alpha_prime * d_alpha_prime_dn * h_0_minus_h_1 - f_x * dh_1_dn)
    dF_ds = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_ds + g_x * (dh_1_ds + df_x_d_alpha_prime * d_alpha_prime_ds * h_0_minus_h_1 - f_x * dh_1_ds)
    dF_dt = g_x * (dh_1_dt + df_x_d_alpha_prime * d_alpha_prime_dt * h_0_minus_h_1 - f_x * dh_1_dt)

    # Final derivatives with respect to density, sigma and tau

    df_dn = f_LDA * dF_dn + df_dn_LDA * F_X 
    df_ds = f_LDA * dF_ds
    df_dt = f_LDA * dF_dt

    return df_dn, df_ds, df_dt, e_X










def calculate_r2SCAN_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the r2SCAN exchange energy density and derivative with respect to the density, square gradient and kinetic energy density.

    A reasonably efficient implementation of equations in supporting information of 10.1021/acs.jpclett.0c02405.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        df_dt (array): Derivative of f = n * e_X with respect to tau
        e_X (array): r2SCAN exchange energy density per particle
    
    """

    eta = 0.001

    a_1 = 4.9479
    c_1 = 0.667
    c_2 = 0.8
    k_0 = 0.174
    k_1 = 0.065
    mu = 10 / 81
    d = 0.361
    d_x = 1.24

    c_x = [1, -0.667, -0.4445555, -0.663086601049, 1.451297044490, -0.887998041597, 0.234528941479, -0.023185843322]

    C_eta = 20 / 27 + eta * 5 / 3
    C_2 = np.sum(c_x[1:] * np.linspace(1, 7, 7) * k_0)

    cbrt_density = np.cbrt(density)
    inv_density = clean(1 / density)

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)
    p = sigma * dp_ds
    p_squared = p * p
    p_fourth_root = np.sqrt(np.sqrt(p))

    # The Weiszacker and uniform electron gas kinetic energy densities
    
    dtau_w_ds = inv_density / 8

    tau_w = sigma * dtau_w_ds
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    tau_minus_tau_w = tau - tau_w

    # This equation is very different from the original SCAN

    x_exponent_term = np.exp(-p_squared / d ** 4)

    x = (C_eta * C_2 * x_exponent_term + mu) * p

    h_0 = 1 + k_0
    h_1 = 1 + k_1 - k_1 / (1 + x / k_1)
    h_0_minus_h_1 = h_0 - h_1

    # The regularised version of the iso-orbital indicator

    alpha_bar_denominator = tau_u + eta * tau_w
    inv_alpha_bar_denominator = 1 / alpha_bar_denominator
    inv_alpha_bar_denominator_squared = inv_alpha_bar_denominator * inv_alpha_bar_denominator
    alpha_bar = tau_minus_tau_w * inv_alpha_bar_denominator
    inv_one_minus_alpha_bar = 1 / (1 - alpha_bar)
    inv_one_minus_alpha_bar_squared = inv_one_minus_alpha_bar * inv_one_minus_alpha_bar

    # Smoother switching function

    f_x = ((((((c_x[7] * alpha_bar + c_x[6]) * alpha_bar + c_x[5]) * alpha_bar + c_x[4]) * alpha_bar + c_x[3]) * alpha_bar + c_x[2]) * alpha_bar + c_x[1]) * alpha_bar + c_x[0]

    # The original limits still apply from SCAN

    small_alpha_exponent_term = np.exp(np.clip(-c_1 * alpha_bar * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))
    large_alpha_exponent_term = -d_x * np.exp(np.clip(c_2 * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))

    f_x = np.where(alpha_bar < 0, small_alpha_exponent_term, f_x)
    f_x = np.where(alpha_bar > 2.5, large_alpha_exponent_term, f_x)

    g_exponent_term = np.exp(-a_1 / p_fourth_root)

    g_x = 1 - g_exponent_term

    # The exchange enhancement factor

    F_X = (h_1 + f_x * h_0_minus_h_1) * g_x

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # The exchange energy density

    e_X = e_X_LDA * F_X
    f_LDA = e_X_LDA * density

    # Derivative terms begin here
    
    dtau_w_dn = -tau_w * inv_density
    dtau_u_dn = (5 / 3) * tau_u * inv_density
    
    # Derivatives of the iso-orbital indicator

    d_alpha_bar_d_tau_w = -(tau_u + eta * tau) * inv_alpha_bar_denominator_squared

    d_alpha_bar_dn = (-dtau_w_dn * alpha_bar_denominator - tau_minus_tau_w * (dtau_u_dn + eta * dtau_w_dn)) * inv_alpha_bar_denominator_squared
    d_alpha_bar_ds = d_alpha_bar_d_tau_w * dtau_w_ds
    d_alpha_bar_dt = 1 / (tau_u + eta * tau_w)

    dp_dn = -(8 / 3) * p * inv_density
    dh_1_dx = 1 / ((1 + x / k_1) * (1 + x / k_1))
    dx_dp = mu + C_eta * C_2 * x_exponent_term * (1 - 2 * p_squared / d ** 4)
    dg_dp = -g_exponent_term * (a_1 / 4) * p_fourth_root ** -5

    # Derivatives of the h_1 function

    dh_1_dp = dh_1_dx * dx_dp
    dh_1_ds = dh_1_dp * dp_ds
    dh_1_dn = dh_1_dp * dp_dn

    # Derivatives of the switching function, including of the degree-seven polynomial
    
    df_x_d_alpha_bar = ((((((7 * c_x[7] * alpha_bar + 6 * c_x[6]) * alpha_bar + 5 * c_x[5]) * alpha_bar + 4 * c_x[4]) * alpha_bar + 3 * c_x[3]) * alpha_bar + 2 * c_x[2]) * alpha_bar + c_x[1])

    df_x_d_alpha_bar = np.where(alpha_bar < 0, -c_1 * inv_one_minus_alpha_bar_squared * small_alpha_exponent_term, df_x_d_alpha_bar)
    df_x_d_alpha_bar = np.where(alpha_bar > 2.5, c_2 * inv_one_minus_alpha_bar_squared * large_alpha_exponent_term, df_x_d_alpha_bar)

    # Derivatives of the exchange enhancement function

    dF_dn = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_dn + g_x * (dh_1_dn + df_x_d_alpha_bar * d_alpha_bar_dn * h_0_minus_h_1 - f_x * dh_1_dn)
    dF_ds = (h_1 + f_x * h_0_minus_h_1) * dg_dp * dp_ds + g_x * (dh_1_ds + df_x_d_alpha_bar * d_alpha_bar_ds * h_0_minus_h_1 - f_x * dh_1_ds)
    dF_dt = df_x_d_alpha_bar * d_alpha_bar_dt * h_0_minus_h_1 * g_x

    # Final derivatives with respect to density, sigma and tau

    df_dn = f_LDA * dF_dn + df_dn_LDA * F_X 
    df_ds = f_LDA * dF_ds
    df_dt = f_LDA * dF_dt

    return df_dn, df_ds, df_dt, e_X










def calculate_B97_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the B97 exchange energy density and derivative with respect to the density and square gradient.

    An efficient implementation of the equations in 10.1063/1.475007.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): B97 exchange energy density per particle
    
    """

    # The parameters can be for Becke's hybrid (first case) or Grimme's dispersion-corrected GGA (second case) - there is a discrepancy between the ORCA and LibXC implementations

    c_x = [0.8094, 0.5073, 0.7481] if calculation.method.name == "B97" else [1.08662, -0.52127, 3.25429]

    gamma = 0.004

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)
    
    # The cube root four makes the equations in Becke's paper match the TUNA treatment of exchange spin scaling

    s_squared = np.cbrt(4) * sigma / (np.cbrt(density) ** 8) 

    x = gamma * s_squared / (1 + gamma * s_squared)

    F_X = c_x[0] + (c_x[1] + c_x[2] * x) * x

    # Energy density per particle

    e_X = e_X_LDA * F_X

    # Derivative of the exchange enhancement factor

    dF_dx = c_x[1] + 2 * c_x[2] * x
    
    # This is the numerator or x divided by the squared denominator
    
    x_term = x * (1 - x)

    # Chain rule derivatives

    dx_dn = -8 / (3 * density)

    df_dx_times_x_term = density * e_X_LDA * dF_dx * x_term

    df_dn = df_dx_times_x_term * dx_dn + df_dn_LDA * F_X
    df_ds = df_dx_times_x_term / sigma

    return df_dn, df_ds, None, e_X










def calculate_B3_exchange(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the B3LYP exchange energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_X with respect to density
        df_ds (array): Derivative of f = n * e_X with respect to sigma
        e_X (array): B3LYP exchange energy density per particle
    
    """

    # Calculates the local density exchange energy and derivative

    df_dn_LDA, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)
    
    # Calculates the Becke 1988 GGA exchange energy density and derivatives

    df_dn_B88, df_ds_B88, _, e_X_B88 = calculate_B88_exchange(density, sigma, tau, calculation)

    # The factors here are chosen such that when combined with the multiplicative factors for Hartree-Fock exchange proportion, the B3LYP coefficients are used
    
    df_dn = 0.9 * df_dn_B88 + 0.1 * df_dn_LDA
    df_ds = 0.9 * df_ds_B88

    e_X = 0.9 * e_X_B88 + 0.1 * e_X_LDA 

    return df_dn, df_ds, None, e_X










def calculate_restricted_VWN3_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted VWN-III correlation energy density and derivative with respect to the density.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        e_C (array): VWN-III correlation energy density per particle
    
    """

    # Parameters for a restricted (paramagnetic) reference

    df_dn, e_C, _ = calculate_VWN_potential(density, -0.409286, 13.0720, 42.7198, 0.0310907)

    return df_dn, None, None, e_C










def calculate_unrestricted_VWN3_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
   
    """
    
    Calculates the unrestricted VWN-III correlation energy density and derivative with respect to the density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        e_C (array): Unrestricted VWN-III correlation energy density per particle
    
    """

    # Parameters for a restricted (paramagnetic) and fully polarised (ferromagnetic) reference  
     
    _, e_C_0, de0_dr = calculate_VWN_potential(density, -0.409286, 13.0720, 42.7198, 0.0310907)
    _, e_C_1, de1_dr = calculate_VWN_potential(density, -0.743294, 20.1231, 101.578, 0.01554535)

    # Calculates the local spin polarisation

    zeta = calculate_zeta(alpha_density, beta_density)

    # Spin polarisation function and derivatives

    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)

    # The energy density per particle

    e_C = e_C_0 + (e_C_1 - e_C_0) * f_zeta

    # Calculates the Seitz radius

    r_s, _ = calculate_seitz_radius(density)

    de_dr = de0_dr + (de1_dr - de0_dr) * f_zeta
    de_dzeta = (e_C_1 - e_C_0) * f_prime_zeta

    # Calculates derivatives with respect to the alpha and beta densities

    df_dn_alpha = e_C - r_s / 3 * de_dr - (zeta - 1) * de_dzeta
    df_dn_beta = e_C - r_s / 3 * de_dr - (zeta + 1) * de_dzeta

    return df_dn_alpha, df_dn_beta, None, None, None, None, None, e_C










def calculate_restricted_VWN5_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted VWN-V correlation energy density and derivative with respect to the density.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        e_C (array): VWN-V correlation energy density per particle
    
    """

    # Parameters for a restricted (paramagnetic) reference

    df_dn, e_C, _ = calculate_VWN_potential(density, -0.10498, 3.72744, 12.9352, 0.0310907)

    return df_dn, None, None, e_C










def calculate_unrestricted_VWN5_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
   
    """
    
    Calculates the unrestricted VWN-V correlation energy density and derivative with respect to the density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        e_C (array): Unrestricted VWN-V correlation energy density per particle
    
    """

    # Parameters for a restricted (paramagnetic), fully polarised (ferromagnetic) reference, and RPA fit (alpha)

    _, e_C_0, de0_dr = calculate_VWN_potential(density, -0.10498, 3.72744, 12.9352, 0.0310907)
    _, e_C_1, de1_dr = calculate_VWN_potential(density, -0.32500, 7.06042, 18.0578, 0.01554535)
    _, minus_alpha, dalpha_dr = calculate_VWN_potential(density, -0.0047584, 1.13107, 13.0045, 1 / (6 * np.pi ** 2))
    
    # Parameters for an unrestricted reference by interpolation between limits

    df_dn_alpha, df_dn_beta, e_C = calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, -dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr)
    
    return df_dn_alpha, df_dn_beta, None, None, None, None, None, e_C










def calculate_restricted_PW_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted PW92 correlation energy density and derivative with respect to the density.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        e_C (array): PW92 correlation energy density per particle
    
    """

    # Note - this is "modified" PW from LibXC has more significant figures than original paper

    df_dn, e_C, _ = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    return df_dn, None, None, e_C










def calculate_unrestricted_PW_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
       
    """
    
    Calculates the unrestricted PW92 correlation energy density and derivative with respect to the density.

    This is "modified" PW from LibXC and has more significant figures than the original paper.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        e_C (array): Unrestricted PW92 correlation energy density per particle
    
    """

    # Parameters for a restricted (paramagnetic), fully polarised (ferromagnetic) reference, and RPA fit (alpha)

    _, e_C_0, de0_dr = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    _, e_C_1, de1_dr = calculate_PW_potential(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    _, minus_alpha, dalpha_dr = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    # Parameters for an unrestricted reference by interpolation between limits

    df_dn_alpha, df_dn_beta, e_C = calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, -dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr)


    return df_dn_alpha, df_dn_beta, None, None, None, None, None, e_C










def calculate_PW_potential(density: ndarray, A: float, alpha_1: float, beta_1: float, beta_2: float, beta_3: float, beta_4: float, P: float) -> tuple:

    """
    
    Calculates PW92 local density correlation potential.

    Args:
        density (array): Electron density on grid
        A (float): Coefficient for PW92
        alpha_1 (float): Coefficient for PW92
        beta_1 (float): Coefficient for PW92
        beta_2 (float): Coefficient for PW92
        beta_3 (float): Coefficient for PW92
        beta_4 (float): Coefficient for PW92
        P (float): Exponent for PW92
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density, n
        e_C (array): Energy density per particle for PW92
        de_C_dr (array): Derivative of energy density with respect to Seitz radius

    """

    # Calculates the Seitz radius for the density

    r_s, _ = calculate_seitz_radius(density)

    # Intermediate quantities for PW92 LDA correlation

    Q_0 = -2 * A * (1 + alpha_1 * r_s)
    Q_1 = 2 * A * (beta_1 * r_s ** (1 / 2) + beta_2 * r_s + beta_3 * r_s ** (3 / 2) + beta_4 * r_s ** (P + 1))
    Q_1_prime = A * (beta_1 * r_s ** (-1 / 2) + 2 * beta_2 + 3 * beta_3 * r_s ** (1 / 2) + 2 * (P + 1) * beta_4 * r_s ** P)

    # Numpy's log_1_plus function is more numerically stable than log(1 + 1/Q_1)

    log_term = np.log1p(1 / Q_1)

    # Energy density per particle for PW92

    e_C = Q_0 * log_term

    # Derivative of energy density with respect to Seitz radius

    de_C_dr = -2 * A * alpha_1 * log_term - Q_0 * Q_1_prime / (Q_1 * Q_1 + Q_1)

    # Derivative with respect to density

    df_dn = e_C - r_s / 3 * de_C_dr

    return df_dn, e_C, de_C_dr










def calculate_VWN_potential(density: ndarray, x_0: float, b: float, c: float, A: float) -> tuple:

    """
    
    Calculates VWN local density correlation potential.

    Args:
        density (array): Electron density on grid
        x_0 (float): Coefficient for VWN correlation
        b (float): Coefficient for VWN correlation
        c (float): Coefficient for VWN correlation
        A (float): Coefficient for VWN correlation

    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density, n
        e_C (array): Energy density per particle for VWN
        de_C_dr (array): Derivative of energy density with respect to Seitz radius

    """

    # Useful intermediate constants for VWN

    Q = (4 * c - b ** 2) ** (1 / 2)
    X_0 = x_0 ** 2 + b * x_0 + c
    c_1 = -b * x_0 / X_0
    c_2 = 2 * b * (c - x_0 ** 2) / (Q * X_0)

    # Seitz radius as a function of density

    r_s, _ = calculate_seitz_radius(density)
    x = r_s ** (1 / 2)
    x_minus_x_0 = x - x_0

    # Useful intermediate quantities

    X = r_s + b * x + c

    log_term_1 = np.log(r_s / X) 
    log_term_2 = np.log(x_minus_x_0 * x_minus_x_0 / X)
    atan_term = np.arctan(Q / (2 * x + b))

    # Derivative of various quantities

    combo = (2 / x + 2 * c_1 / x_minus_x_0 - (2 * x + b) * (1 + c_1) / X - (1 / 2) * c_2 * Q / X)

    # Energy density per particle for VWN

    e_C = A * (log_term_1 + c_1 * log_term_2 + c_2 * atan_term)

    # Derivative of energy density with respect to Seitz radius

    de_C_dr = (A / 2) * combo / x

    # Derivative with respect to density

    df_dn = e_C - r_s / 3 * de_C_dr

    return df_dn, e_C, de_C_dr










def calculate_VWN5_spin_interpolation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, minus_alpha: ndarray, dalpha_dr: ndarray, e_C_0: ndarray, de0_dr: ndarray, e_C_1: ndarray, de1_dr: ndarray) -> tuple:

    """

    Calculates the derivatives and energy density for unrestricted VWN-V.

    Args:
        alpha_density (array): Alpha spin electron density on grid
        beta_density (array): Beta spin electron density on grid
        density (array): Electron density on grid
        minus_alpha (array): Negative RPA fit energy density
        dalpha_dr (array): Derivative of alpha with respect to Seitz radius
        e_C_0 (array): Energy density for paramagnetic system
        de0_dr (array): Derivative of energy density for paramagnetic system with respect to Seitz radius
        e_C_1 (array): Energy density for ferromagnetic system
        de1_dr (array): Derivative of energy density for ferromagnetic system with respect to Seitz radius

    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        e_C (array): Energy density for unrestricted VWN correlation

    """

    # Calculates local spin polarisation

    zeta = calculate_zeta(alpha_density, beta_density)

    alpha = -1 * minus_alpha

    # Spin interpolation quantities

    zeta_4 = zeta ** 4
    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)
    f_prime_prime_at_zero = 8 / (9 * (np.cbrt(2) ** 4 - 2))

    # Energy density per particle

    e_C = e_C_0 + alpha * f_zeta / f_prime_prime_at_zero * (1 - zeta_4) + (e_C_1 - e_C_0) * f_zeta * zeta_4

    # Local Seitz radius as a function of density

    r_s, _ = calculate_seitz_radius(density)

    # Derivative of energy density with respect to Seitz radius

    de_dr = de0_dr * (1 - f_zeta * zeta_4) + de1_dr * f_zeta * zeta_4 + dalpha_dr * f_zeta * (1 - zeta_4) / f_prime_prime_at_zero

    # Derivative of energy density with respect to spin polarisation

    de_dzeta = 4 * zeta * zeta * zeta * f_zeta * (e_C_1 - e_C_0 - alpha / f_prime_prime_at_zero) + f_prime_zeta * (zeta_4 * (e_C_1 - e_C_0) + (1 - zeta_4) * alpha / f_prime_prime_at_zero)

    # Derivatives of f with respect to alpha and beta densities

    df_dn_alpha = e_C - r_s / 3 * de_dr - (zeta - 1) * de_dzeta 
    df_dn_beta = e_C - r_s / 3 * de_dr - (zeta + 1) * de_dzeta 


    return df_dn_alpha, df_dn_beta, e_C










def calculate_restricted_PBE_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
   
    """
    
    Calculates the restricted PBE correlation energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        e_C (array): Restricted PBE exchange energy density per particle
    
    """

    # Calculates the local density correlation

    df_dn_LDA, _, _, e_C_LDA = calculate_restricted_PW_correlation(density, None, None, None)

    de_C_LDA_dn = (df_dn_LDA - e_C_LDA) / density

    # Key parameters for defining PBE - this value of beta is not exact, but chosen to match the ORCA implementation

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.066725

    # If we are in this function due to revised TPSS, use a different definition of beta

    if calculation.functional.c_functional == "REVTPSS":

        r_s, _ = calculate_seitz_radius(density)

        beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)


    k_F = calculate_Fermi_wavevector(density=density)

    # Form of reduced density gradient used in PBE correlation

    t_squared = sigma * np.pi / (16 * k_F * density * density)
    t_fourth = t_squared * t_squared

    exp_factor = np.exp(-e_C_LDA / gamma)
    A = beta / (gamma * (exp_factor - 1))

    k = 1 + A * t_squared
    D = k + A * A * t_fourth

    X = (beta / gamma) * t_squared * k / D
    
    # The GGA correction to the LDA correlation energy density

    H = gamma * np.log1p(X)

    # The energy density per partcile for RPBE

    e_C = e_C_LDA + H

    dA_dn = (A * A / beta) * exp_factor * de_C_LDA_dn

    pref = 1 / ((1 + X) * D * D)
    common = beta * (1 + 2 * A * t_squared)

    # Derivatives with respect to the density, n, and sigma, s

    dH_dn = pref * (common * -(7 / 3) * t_squared / density - beta * A * t_fourth * t_squared * (A * t_squared + 2) * dA_dn)
    dH_ds = pref * common * t_squared / sigma

    df_dn = e_C + density * (de_C_LDA_dn + dH_dn)
    df_ds = density * dH_ds

    return df_dn, df_ds, None, e_C










def calculate_unrestricted_PBE_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted PBE correlation energy density and derivative with respect to the density and square gradient.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        e_C (array): Unrestricted PBE correlation energy density per particle
    
    """

    # The PBE correlation functional only depends on the full sigma, not the spin channels. This is cleaned at the square of the density floor

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor = constants.sigma_floor)

    # Key parameters for PBE - gamma is exact and beta is given to match the ORCA implementation

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.066725

    # If we are in this function due to revised TPSS, use a different definition of beta

    if calculation.functional.c_functional == "REVTPSS":

        r_s, _ = calculate_seitz_radius(density)

        beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)

    B = beta / gamma

    # Local density correlation energy density and derivatives

    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)

    # Spin polarisation

    zeta = calculate_zeta(alpha_density, beta_density)

    # Spin polarisation dependent quantities - it is crucial that the cleans here are inside the square root! Otherwise PBE and TPSS break for one-electron systems
    
    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (1 / 2) * (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus)
    phi_prime = (1 / cbrt_plus - 1 / cbrt_minus) / 3

    # Repeatedly used grid quantities

    k_F = calculate_Fermi_wavevector(density=density)

    density_squared = density * density
    phi_squared = phi * phi
    phi_cubed = phi * phi_squared

    # Key reduced density gradient for PBE

    T = sigma * np.pi / (16 * phi_squared * k_F * density_squared)

    Q = np.exp(-e_C_LDA / (gamma * phi_cubed))
    A = B / (Q - 1)
    A_squared = A * A
    t_squared = T * T

    D = 1 + A * T + A_squared * t_squared
    N = B * T * (1 + A * T)
    X = N / D

    D_squared = D * D

    # Correction to the LDA correlation energy density, log1p is more stable

    H = gamma * phi_cubed * np.log1p(X)     
    e_C = e_C_LDA + H

    # Inverse and square inverse density

    inv_n = 1 / density
    inv_n_squared = inv_n * inv_n

    # Derivatives of spin poalrisation with respect to densities

    dphi_dn_alpha = phi_prime * (2 * beta_density * inv_n_squared)
    dphi_dn_beta = -phi_prime * (2 * alpha_density * inv_n_squared)

    # Derivatives of reduced density gradient

    dT_dn = -7 / 3 * T * inv_n
    dT_dphi = -2 * T / phi

    dT_dn_alpha = dT_dn + dT_dphi * dphi_dn_alpha
    dT_dn_beta  = dT_dn + dT_dphi * dphi_dn_beta

    T_over_sigma = np.pi / (16 * phi_squared * k_F * density_squared)

    # Derivatives of energy density with respect to densities

    de_C_LDA_dn_alpha = (df_dn_alpha_LDA - e_C_LDA) * inv_n
    de_C_LDA_dn_beta  = (df_dn_beta_LDA  - e_C_LDA) * inv_n

    phi_fourth = phi_squared * phi_squared
    dA_dE   = (A_squared / beta) * Q / phi_cubed
    dA_dphi = -3 * e_C_LDA * Q * A_squared / (beta * phi_fourth)

    # Derivatives of A with respect to densities

    dA_dn_alpha = dA_dE * de_C_LDA_dn_alpha + dA_dphi * dphi_dn_alpha
    dA_dn_beta = dA_dE * de_C_LDA_dn_beta + dA_dphi * dphi_dn_beta

    dD_dT = A + 2 * A_squared * T
    dX_dT = (B * (1 + 2 * A * T) * D - N * dD_dT) / D_squared

    dD_dA = T + 2 * A * t_squared
    dX_dA = (B * t_squared * D - N * dD_dA) / D_squared

    # Log1p here is more stable than log(1 + X)

    C1 = 3 * gamma * phi_squared * np.log1p(X)
    C2 = gamma * phi_cubed / (1 + X)

    # Derivatives of GGA correction to LDA correlation

    dH_dn_alpha = C1 * dphi_dn_alpha + C2 * (dX_dT * dT_dn_alpha + dX_dA * dA_dn_alpha)
    dH_dn_beta = C1 * dphi_dn_beta + C2 * (dX_dT * dT_dn_beta + dX_dA * dA_dn_beta)

    # Derivatives with respect to densities

    df_dn_alpha = e_C + density * (de_C_LDA_dn_alpha + dH_dn_alpha)
    df_dn_beta = e_C + density * (de_C_LDA_dn_beta + dH_dn_beta)

    # Derivatives with respect to sigma

    df_ds = density * C2 * dX_dT * T_over_sigma

    df_ds_aa, df_ds_bb, df_ds_ab = df_ds, df_ds, 2 * df_ds


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C










def calculate_restricted_LYP_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted LYP correlation energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        e_C (array): Restricted PBE exchange energy density per particle
    
    """

    # These constants define LYP correlation

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    # Repeatedly useful quantities

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = 1 + d * inv_cbrt_density
    C2 = 6 / 10 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8

    # Definition of w and delta directly from LYP paper

    w = inv_cbrt_density ** 11 * np.exp(-c * inv_cbrt_density) / X
    delta = inv_cbrt_density * (c + d / X)

    # This is often used

    minus_a_b_w = -a * b * w * density
    
    # More stable to form this quantity than w_prime directly

    w_prime_over_w = -(1 / 3) * inv_cbrt_density ** 4 * (11 * cbrt_density - c - d / X)

    # Directly from LYP paper for delta derivative

    delta_prime = (1 / 3) * (d * d * inv_cbrt_density ** 5 / (X * X) - delta * inv_density)

    # Derivative with respect to sigma

    df_ds = minus_a_b_w * density * (-7 * delta - 3) / 72

    # Derivative with respect to density

    df_dn = -a / X + minus_a_b_w * sigma * (-1 / 12 - 7 * delta / 36 + density * (-7 * delta_prime / 72 + w_prime_over_w * (-1 / 24 - 7 * delta / 72)))
    df_dn += density * (- a * d / (3 * X * X * cbrt_density ** 4) - 7 * C2 * a * b * w / 3 - (1 / 2) * C2 * a * b * density * w_prime_over_w * w)
    
    # Correlation energy density per particle for LYP

    e_C = (1 / 2) * C2 * minus_a_b_w - minus_a_b_w * sigma * (7 * delta + 3) / 72 - a / X

    return df_dn, df_ds, None, e_C










def calculate_unrestricted_LYP_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted LYP correlation energy density and derivative with respect to the density and square gradient.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        e_C (array): Unrestricted LYP correlation energy density per particle
    
    """
    
    # Defining parameters for LYP

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    # Cube roots of density channels

    cbrt_alpha_density = np.cbrt(alpha_density)
    cbrt_beta_density = np.cbrt(beta_density)

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = (1 + d * inv_cbrt_density)

    # Main constant

    C = np.cbrt(2) ** 11 * 3 / 10 * np.cbrt(3 * np.pi ** 2) ** 2

    # Useful quantities for the density spin channels

    density_product = alpha_density * beta_density
    densities_power_sum = cbrt_alpha_density ** 8 + cbrt_beta_density ** 8

    # Straight from Pople paper

    w = inv_cbrt_density ** 11 * np.exp(-c * inv_cbrt_density) / X
    delta = inv_cbrt_density * (c + d / X)

    # Reused quantity

    minus_a_b_w = -a * b * w

    # Derivatives straight from Pople paper

    w_prime = -(1 / 3) * inv_cbrt_density ** 4 * w * (11 * cbrt_density - c - d / X)
    delta_prime = (1 / 3) * (d * d * inv_cbrt_density ** 5 / (X * X) - delta * inv_density)

    # This is more stable than forming w_prime first

    w_prime_over_w = -(1 / 3) * inv_cbrt_density ** 4 * (11 * cbrt_density - c - d / X)

    # Derivatives with respect to each density gradient spin channel

    df_ds_aa = minus_a_b_w * ((1 / 9) * density_product * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - beta_density * beta_density)
    df_ds_bb = minus_a_b_w * ((1 / 9) * density_product * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - alpha_density * alpha_density)
    df_ds_ab = minus_a_b_w * ((1 / 9) * density_product * (47 - 7 * delta) - (4 / 3) * density * density)

    # From Pople paper, energy density per particle

    e_C = inv_density * (density_product * (C * minus_a_b_w * densities_power_sum - 4 * a / X * inv_density) + df_ds_aa * sigma_aa + df_ds_bb * sigma_bb + df_ds_ab * sigma_ab)

    # Second derivatives with respect to density channel and gradient channel

    d2f_dn_a_ds_aa = w_prime_over_w * df_ds_aa + minus_a_b_w * ((1 / 9) * beta_density * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - (1 / 9) * density_product * (delta_prime * (3 + alpha_density * inv_density) + (delta - 11) * beta_density * inv_density * inv_density))
    d2f_dn_b_ds_bb = w_prime_over_w * df_ds_bb + minus_a_b_w * ((1 / 9) * alpha_density * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - (1 / 9) * density_product * (delta_prime * (3 + beta_density * inv_density) + (delta - 11) * alpha_density * inv_density * inv_density))

    d2f_dn_a_ds_ab = w_prime_over_w * df_ds_ab + minus_a_b_w * ((1 / 9) * beta_density * (47 - 7 * delta) - (7 / 9) * density_product * delta_prime - (8 / 3) * density)
    d2f_dn_b_ds_ab = w_prime_over_w * df_ds_ab + minus_a_b_w * ((1 / 9) * alpha_density * (47 - 7 * delta) - (7 / 9) * density_product * delta_prime - (8 / 3) * density)

    d2f_dn_a_ds_bb = w_prime_over_w * df_ds_bb + minus_a_b_w * ((1 / 9) * beta_density * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - (1 / 9) * density_product * ((3 + beta_density * inv_density) * delta_prime - (delta - 11) * beta_density * inv_density * inv_density) - 2 * alpha_density)
    d2f_dn_b_ds_aa = w_prime_over_w * df_ds_aa + minus_a_b_w * ((1 / 9) * alpha_density * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - (1 / 9) * density_product * ((3 + alpha_density * inv_density) * delta_prime - (delta - 11) * alpha_density * inv_density * inv_density) - 2 * beta_density)

    # Derivatives with respect to density channels expressed as dependent on second derivatives with respect to sigma channels

    df_dn_alpha = -4 * a / X * density_product * inv_density * ((1 / 3) * d * inv_cbrt_density ** 4 / X + 1 / alpha_density - inv_density) - C * a * b * (w_prime * density_product * densities_power_sum + w * beta_density * (11 / 3 * cbrt_alpha_density ** 8 + cbrt_beta_density ** 8)) + d2f_dn_a_ds_aa * sigma_aa + d2f_dn_a_ds_bb * sigma_bb + d2f_dn_a_ds_ab * sigma_ab                                                         
    df_dn_beta = -4 * a / X * density_product * inv_density * ((1 / 3) * d * inv_cbrt_density ** 4 / X + 1 / beta_density - inv_density) - C * a * b * (w_prime * density_product * densities_power_sum + w * alpha_density * (11 / 3 * cbrt_beta_density ** 8 + cbrt_alpha_density ** 8)) + d2f_dn_b_ds_bb * sigma_bb + d2f_dn_b_ds_aa * sigma_aa + d2f_dn_b_ds_ab * sigma_ab                                                         
       

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C










def calculate_restricted_P86_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted P86 correlation energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        e_C (array): Restricted P86 exchange energy density per particle
    
    """

    # Constants defining P86 correlation

    alpha, beta, gamma, delta, f_tilde = 0.023266, 0.000007389, 8.723, 0.472, 0.11

    # Calculates local Seitz radius

    r_s, inv_density = calculate_seitz_radius(density)
    r_s_squared = r_s * r_s
    cbrt_density = np.cbrt(density)

    # Numerator and denominator for P86 correlation

    N = 0.002568 + alpha * r_s + beta * r_s_squared
    D = 1 + gamma * r_s + delta * r_s_squared + 10000 * beta * r_s_squared * r_s

    C = 0.001667 + N / D
    C_inf = 0.004235

    # Definition of phi from P86 paper

    phi = 1.745 * f_tilde * C_inf / C * sigma ** (1 / 2) / cbrt_density ** (7 / 2)

    # Local density energy density and derivative

    df_LDA_dn, _, _, e_C_LDA = calculate_restricted_PW_correlation(density, None, None, None)

    # GGA correction to LDA energy density

    H = C * sigma * np.exp(-phi) / cbrt_density ** 7

    # Energy density per particle

    e_C = e_C_LDA + H

    # Derivative with respect to sigma

    df_ds = C * np.exp(-phi) / cbrt_density ** 4 * (1 - phi / 2)

    # Chain rule derivatives

    dN_dr = alpha + 2 * beta * r_s
    dD_dr = gamma + 2 * delta * r_s + 3e4 * beta * r_s_squared
    dC_dr = (dN_dr * D - N * dD_dr) / (D * D)
    dC_dn = dC_dr * -(1 / 3) * r_s * inv_density
    
    # Derivative of GGA correction with respect to density

    dH_dn = H * ((1 + phi) * (density * dC_dn / C) + (7 / 6) * (phi - 2))

    # Derivative with respect to density

    df_dn = df_LDA_dn + H + dH_dn

    return df_dn, df_ds, None, e_C










def calculate_unrestricted_P86_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the unrestricted P86 correlation energy density and derivative with respect to the density and square gradient.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        e_C (array): Unrestricted P86 correlation energy density per particle
    
    """

    # Constants defining P86 correlation

    alpha, beta, gamma, delta, f_tilde = 0.023266, 0.000007389, 8.723, 0.472, 0.11

    # Calculates the cleans the total sigma

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor = constants.sigma_floor)

    # Calculates the local Seitz radius

    r_s, inv_density = calculate_seitz_radius(density)
    zeta = calculate_zeta(alpha_density, beta_density)

    cbrt_density = np.cbrt(density)

    r_s_squared = r_s * r_s

    # Polynomial defined in P86 paper

    N = 0.002568 + alpha * r_s + beta * r_s_squared
    D = 1 + gamma * r_s + delta * r_s_squared + 1e4 * beta * r_s * r_s_squared

    C = 0.001667 + N / D
    C_inf = 0.004235

    # Phi from P86 paper
    phi = 1.745 * f_tilde * C_inf / C * sigma ** (1 / 2) / cbrt_density ** (7 / 2)

    # Spin polarisation functions

    p = np.cbrt(1 + zeta)
    m = np.cbrt(1 - zeta)
    S = p ** 5 + m ** 5
    d = (S / 2) ** (1 / 2)

    # Local density correlation and derivatives

    df_dn_alpha, df_dn_beta, _,_, _, _, _, e_C_LDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)

    # GGA correction to LDA energy density

    H = (C * sigma * np.exp(-phi) / cbrt_density ** 7) / d
    e_C = e_C_LDA + H

    # Derivative with respect to sigma
    
    df_ds = (C * np.exp(-phi) / cbrt_density ** 4 * (1 - phi / 2)) / d
    df_ds_aa, df_ds_bb, df_ds_ab = df_ds, df_ds, 2 * df_ds

    # Chain rule derivatives

    dN_dr = alpha + 2 * beta * r_s
    dD_dr = gamma + 2 * delta * r_s + 3e4 * beta * r_s_squared
    dC_dr = (dN_dr * D - N * dD_dr) / (D * D)
    dC_dn = dC_dr * (-(1 / 3) * r_s * inv_density)

    # Derivative of GGA correction with respect to density

    dH_dn = H * ((1 + phi) * (density * dC_dn / C) + (7 / 6) * (phi - 2))

    # Derivatives of spin polarisation functions

    dln_inv_d_dzeta = -(5 / 6) * (p * p - m * m) / S
    dzeta_dn_alpha = 2 * beta_density * inv_density * inv_density
    dzeta_dn_beta  = -2 * alpha_density * inv_density * inv_density

    # Derivatives with respect to spin density cahnnels

    df_dn_alpha = df_dn_alpha + H + dH_dn + (density * H) * dln_inv_d_dzeta * dzeta_dn_alpha
    df_dn_beta = df_dn_beta + H + dH_dn + (density * H) * dln_inv_d_dzeta * dzeta_dn_beta

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C










def calculate_restricted_PW91_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
   
    """
    
    Calculates the restricted Perdew-Wang 1991 correlation energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        e_C (array): Restricted PW91 exchange energy density per particle
    
    """

    # The local density correlation
    
    df_dn_LDA, _, _, e_C_LDA = calculate_restricted_PW_correlation(density, None, None, None)

    # Defining constants for PW91 correlation

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    # Various useful quantities

    beta = 16 * np.cbrt(3 / np.pi) * C_0
    r_s, inv_density = calculate_seitz_radius(density)

    k_F = calculate_Fermi_wavevector(density=density)

    k_s = (4 * k_F / np.pi) ** (1 / 2)

    # Reduced density gradient for PW91
    
    t = sigma ** (1 / 2) / (2 * density * k_s)

    # Just squaring for speed

    r_s_squared = r_s * r_s
    k_F_squared = k_F * k_F
    k_s_squared = k_s * k_s
    t_squared = t * t

    # Form of "C(r_s)" from PW91

    C_numerator = 0.002568 + 0.023266 * r_s + 7.389e-6 * r_s_squared
    C_denominator = 1 + 8.723 * r_s + 0.472 * r_s_squared + 7.389e-2 * r_s_squared * r_s

    C = -C_X + C_numerator / C_denominator

    # Definition of A in PW91

    A = 2 * alpha / beta / (np.exp(-2 * alpha * e_C_LDA / beta ** 2) - 1)
    B = (C - C_0 - 3 * C_X / 7)
    
    # Term that will be logged

    Y = 1 + 2 * alpha / beta * t_squared * ((1 + A * t_squared) / (1 + A * t_squared + A * A * t_squared * t_squared))

    # First correction term to LDA correlation

    H_0 = beta ** 2 / (2 * alpha) * np.log(Y)

    # Second correction term to LDA correlation

    H_1 = 16 * np.cbrt(3 / np.pi) * B * t_squared * np.exp(-100 * t_squared * k_s_squared / k_F_squared)

    # Energy density per particle of PW91 correlation

    e_C = e_C_LDA + H_0 + H_1

    de_C_LDA_dn = (df_dn_LDA - e_C_LDA) * inv_density

    # Derivatives of t squared

    dt_squared_dn = -7 / 3 * inv_density * t_squared     
    dt_squared_ds = 1 / (4 * k_s_squared) * inv_density * inv_density

    # Derivatives of C(r_s) function

    dCnum_dr = 0.023266 + 2 * 7.389e-6 * r_s
    dCden_dr = 8.723 + 2 * 0.472 * r_s + 3 * 7.389e-2 * r_s_squared

    dfrac_drs = (dCnum_dr * C_denominator - C_numerator * dCden_dr) / (C_denominator * C_denominator)

    dC_dn = dfrac_drs * -r_s / 3 * inv_density

    exp_u = np.exp(-2 * alpha * e_C_LDA / beta ** 2)
    denom_u = exp_u - 1

    # Derivative of A from PW91

    dA_dn = (4 * alpha * alpha / beta ** 3) * exp_u / (denom_u * denom_u) * de_C_LDA_dn

    p = A * t_squared
    D1 = 1 + p + p * p
    R = (1 + p) / D1
    dR_dp = -(p * (p + 2)) / (D1 * D1)

    # Derivative of log term

    Y = 1 + 2 * alpha / beta * t_squared * R
    dY_dn = 2 * alpha / beta * (dt_squared_dn * R + t_squared * dR_dp * (dA_dn * t_squared + A * dt_squared_dn))
    dY_ds = 2 * alpha / beta * (dt_squared_ds * R + t_squared * dR_dp * A * dt_squared_ds)

    # Derivative of first GGA correction

    dH_0_dn = (beta ** 2 / (2 * alpha)) * (dY_dn / Y)
    dH_0_ds = (beta ** 2 / (2 * alpha)) * (dY_ds / Y)

    pref = 16 * np.cbrt(3 / np.pi)

    Q = k_s_squared / k_F_squared
    E = np.exp(-100 * t_squared * Q)

    dE_dn = E * -100 * (dt_squared_dn * Q + t_squared * -Q / 3 * inv_density)
    dE_ds = E * -100 * dt_squared_ds * Q

    # Derivative of second GGA correction

    dH_1_dn = pref * (dC_dn * t_squared * E + B * dt_squared_dn * E + B * t_squared * dE_dn)
    dH_1_ds = pref * (B * dt_squared_ds * E + B * t_squared * dE_ds)

    # Derivative of correlation energy density

    deC_dn = de_C_LDA_dn + dH_0_dn + dH_1_dn
    deC_ds = dH_0_ds + dH_1_ds

    # Final derivatives with respect to density and sigma

    df_dn = e_C + density * deC_dn
    df_ds = density * deC_ds


    return df_dn, df_ds, None, e_C










def calculate_unrestricted_PW91_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted Perdew-Wang 1991 correlation energy density and derivative with respect to the density and square gradient.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        e_C (array): Unrestricted PW91 correlation energy density per particle
    
    """
    
    # The local density correlation

    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)
    
    # This functional only depends on the total square density gradient, not its spin components

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    # Various useful quantities

    beta = 16 * np.cbrt(3 / np.pi) * C_0
    r_s, inv_density = calculate_seitz_radius(density)

    k_F = calculate_Fermi_wavevector(density=density)

    k_s = (4 * k_F / np.pi) ** (1 / 2)

    # Spin interpolation functions

    zeta = calculate_zeta(alpha_density, beta_density)

    phi = (1 / 2) * (np.cbrt(1 + zeta) * np.cbrt(1 + zeta) + np.cbrt(1 - zeta) * np.cbrt(1 - zeta))
    phi_cubed = phi * phi * phi

    # Reduced density gradient for PW91

    t = sigma ** (1 / 2) / (2 * phi * k_s) * inv_density

    # Just squaring for speed

    r_s_squared = r_s * r_s
    k_F_squared = k_F * k_F
    k_s_squared = k_s * k_s
    t_squared = t * t

    # The "C(r_s)" function from PW91

    C_numerator = 0.002568 + 0.023266 * r_s + 7.389e-6 * r_s_squared
    C_denominator = 1 + 8.723 * r_s + 0.472 * r_s_squared + 7.389e-2 * r_s_squared * r_s
    C = -C_X + C_numerator / C_denominator

    # The A function from PW91

    A = 2 * alpha / beta / (np.exp(-2 * alpha * e_C_LDA / (phi_cubed * beta ** 2)) - 1)
    B = (C - C_0 - 3 * C_X / 7)

    # The term to be logged

    Y = 1 + 2 * alpha / beta * t_squared * ((1 + A * t_squared) / (1 + A * t_squared + A * A * t_squared * t_squared))

    # The first GGA correction to the LDA correlation

    H_0 = phi_cubed * beta ** 2 / (2 * alpha) * np.log(Y)

    # The second GGA correction to the LDA correlation

    H_1 = 16 * np.cbrt(3 / np.pi) * B * phi_cubed * t_squared * np.exp(-100 * phi_cubed * phi * t_squared * k_s_squared / k_F_squared)

    # The energy density per particle for PW91 correlation

    e_C = e_C_LDA + H_0 + H_1

    de_C_LDA_dn_alpha = (df_dn_alpha_LDA - e_C_LDA) * inv_density
    de_C_LDA_dn_beta = (df_dn_beta_LDA  - e_C_LDA) * inv_density

    # Spin-scaling factor derivatives

    dphi_dzeta = (1 / 3) * (1 / np.cbrt(1 + zeta) - 1 / np.cbrt(1 - zeta))

    dzeta_dn_alpha = 2 * beta_density * inv_density * inv_density
    dzeta_dn_beta = -2 * alpha_density * inv_density * inv_density

    dphi_dn_alpha = dphi_dzeta * dzeta_dn_alpha
    dphi_dn_beta = dphi_dzeta * dzeta_dn_beta

    # Spin interpolation dependent terms

    phi_squared = phi * phi
    phi_fourth = phi_cubed * phi

    dphi_cubed_dn_alpha = 3 * phi_squared * dphi_dn_alpha
    dphi_cubed_dn_beta = 3 * phi_squared * dphi_dn_beta

    dphi_fourth_dn_alpha = 4 * phi_cubed * dphi_dn_alpha
    dphi_fourth_dn_beta = 4 * phi_cubed * dphi_dn_beta

    # Derivatives of t^2

    dt_squared_dn = -(7 / (3 * density)) * t_squared
    dt_squared_dn_alpha = dt_squared_dn - (2 / phi) * t_squared * dphi_dn_alpha
    dt_squared_dn_beta = dt_squared_dn - (2 / phi) * t_squared * dphi_dn_beta

    dt_squared_ds = 1 / (4 * density * density * phi * phi * k_s_squared)

    # C(rs) derivative (rs depends only on total density)

    dCnum_dr = 0.023266 + 2 * 7.389e-6 * r_s
    dCden_dr = 8.723 + 2 * 0.472 * r_s + 3 * 7.389e-2 * r_s_squared

    dfrac_drs = (dCnum_dr * C_denominator - C_numerator * dCden_dr) / (C_denominator * C_denominator)

    dC_dn = dfrac_drs * -r_s / 3 * inv_density

    # Derivative of A 

    exp_u = np.exp(-2 * alpha * e_C_LDA / (phi_cubed * beta ** 2))
    denom_u = exp_u - 1

    fac_u = exp_u / (denom_u * denom_u)

    dA_dn_alpha = (4 * alpha ** 2 / (beta ** 3 * phi_cubed)) * fac_u * de_C_LDA_dn_alpha - (12 * alpha ** 2 * e_C_LDA / (beta ** 3 * phi_fourth)) * fac_u * dphi_dn_alpha
    dA_dn_beta  = (4 * alpha ** 2 / (beta ** 3 * phi_cubed)) * fac_u * de_C_LDA_dn_beta - (12 * alpha ** 2 * e_C_LDA / (beta ** 3 * phi_fourth)) * fac_u * dphi_dn_beta

    # Log-term derivative machinery (same notation as restricted)

    p = A * t_squared
    D1 = 1 + p + p * p
    R = (1 + p) / D1
    dR_dp = -(p * (p + 2)) / (D1 * D1)

    Y = 1 + 2 * alpha / beta * t_squared * R

    dY_dn_alpha = 2 * alpha / beta * (dt_squared_dn_alpha * R + t_squared * dR_dp * (dA_dn_alpha * t_squared + A * dt_squared_dn_alpha))
    dY_dn_beta  = 2 * alpha / beta * (dt_squared_dn_beta  * R + t_squared * dR_dp * (dA_dn_beta  * t_squared + A * dt_squared_dn_beta))

    dY_ds = 2 * alpha / beta * (dt_squared_ds * R + t_squared * dR_dp * A * dt_squared_ds)

    # Derivative of H_0

    logY = np.log(Y)

    dH_0_dn_alpha = beta ** 2 / (2 * alpha) * (dphi_cubed_dn_alpha * logY + phi_cubed * (dY_dn_alpha / Y))
    dH_0_dn_beta  = beta ** 2 / (2 * alpha) * (dphi_cubed_dn_beta  * logY + phi_cubed * (dY_dn_beta  / Y))

    dH_0_ds = phi_cubed * beta ** 2 / (2 * alpha) * (dY_ds / Y)

    # Derivative of H_1

    pref = 16 * np.cbrt(3 / np.pi)

    Q = k_s_squared / k_F_squared
    E = np.exp(-100 * phi_fourth * t_squared * Q)

    dQ_dn = -Q / 3 * inv_density

    dW_dn_alpha = dphi_fourth_dn_alpha * t_squared * Q + phi_fourth * dt_squared_dn_alpha * Q + phi_fourth * t_squared * dQ_dn
    dW_dn_beta = dphi_fourth_dn_beta  * t_squared * Q + phi_fourth * dt_squared_dn_beta  * Q + phi_fourth * t_squared * dQ_dn

    dE_dn_alpha = E * -100 * dW_dn_alpha
    dE_dn_beta = E * -100 * dW_dn_beta

    dE_ds = E * -100 * (phi_fourth * dt_squared_ds * Q)

    dH_1_dn_alpha = pref * (dC_dn * phi_cubed * t_squared * E + B * dphi_cubed_dn_alpha * t_squared * E + B * phi_cubed * dt_squared_dn_alpha * E + B * phi_cubed * t_squared * dE_dn_alpha)
    dH_1_dn_beta  = pref * (dC_dn * phi_cubed * t_squared * E + B * dphi_cubed_dn_beta  * t_squared * E + B * phi_cubed * dt_squared_dn_beta * E + B * phi_cubed * t_squared * dE_dn_beta)

    dH_1_ds = pref * (B * phi_cubed * dt_squared_ds * E + B * phi_cubed * t_squared * dE_ds)

    # Per-particle derivatives

    deC_dn_alpha = de_C_LDA_dn_alpha + dH_0_dn_alpha + dH_1_dn_alpha
    deC_dn_beta = de_C_LDA_dn_beta + dH_0_dn_beta + dH_1_dn_beta

    deC_ds = dH_0_ds + dH_1_ds

    # Final derivatives of f = n * e_C for PW91 correlation

    df_dn_alpha = e_C + density * deC_dn_alpha
    df_dn_beta = e_C + density * deC_dn_beta

    df_ds_aa = density * deC_ds
    df_ds_bb = density * deC_ds
    df_ds_ab = 2 * density * deC_ds


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C









    
def calculate_restricted_TPSS_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted TPSS correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        df_dt (array): Derivative of f = n * e_C with respect to tau
        e_C (array): Restricted TPSS exchange energy density per particle
    
    """

    # Defining constants for TPSS

    C, d = 0.53, 2.8

    # Repeatedly used quantities

    z = sigma / (8 * tau * density)

    z_squared = z * z
    z_cubed = z * z * z
    A = 1 + C * z_squared

    zeros = np.zeros_like(density)

    # Limits for unpolarised and fully polarised spin densities
    
    df_dn_PBE, df_ds_PBE, _, e_C_PBE = calculate_restricted_PBE_correlation(density, sigma, tau, calculation)
    df_dna_one, _, df_dsaa_one, _, _, _, _, e_C_PBE_one_spin = calculate_unrestricted_PBE_correlation(density / 2, zeros, density / 2, sigma / 4, zeros, zeros, None, None, calculation=calculation)

    # Picks out the largest PBE correlation

    e_C_tilde = np.maximum(e_C_PBE, e_C_PBE_one_spin)
    
    # Energy density for TPSS correlation

    e_C_rev = e_C_PBE * (1 + C * z_squared) - (1 + C) * z_squared * e_C_tilde
    e_C = e_C_rev * (1 + d * e_C_rev * z_cubed)

    # Derivative of PBE correlation with respect to density and sigma

    inv_n = 1 / density
    deC_PBE_dn = (df_dn_PBE - e_C_PBE) * inv_n
    deC_PBE_ds = df_ds_PBE * inv_n

    deC_one_dn = (df_dna_one - e_C_PBE_one_spin) * inv_n 
    deC_one_ds = (1 / 2) * df_dsaa_one * inv_n    

    # Derivative of largest PBE correlation with respect to density and sigma

    deC_tilde_dn = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_dn, deC_one_dn)
    deC_tilde_ds = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_ds, deC_one_ds)

    # Derivative of tau dependent terms with respect to density, sigma and tau
    
    dz_dn, dz_ds, dz_dt = -z * inv_n, 1 / (8 * tau * density), -z / tau
    dz2_dn, dz2_ds, dz2_dt = 2 * z * dz_dn, 2 * z * dz_ds, 2 * z * dz_dt
    dz3_dn, dz3_ds, dz3_dt = 3 * z_squared * dz_dn, 3 * z_squared * dz_ds, 3 * z_squared * dz_dt

    deC_rev_dn = A * deC_PBE_dn +  C * e_C_PBE * dz2_dn - (1 + C) * (e_C_tilde * dz2_dn + z_squared * deC_tilde_dn)
    deC_rev_ds = A * deC_PBE_ds +  C * e_C_PBE * dz2_ds - (1 + C) * (e_C_tilde * dz2_ds + z_squared * deC_tilde_ds)
    deC_rev_dt = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dt 

    # Commonly used prefactor

    prefactor = 1 + 2 * d * e_C_rev * z_cubed

    # Derivative of energy density with respect to density, sigma and tau

    de_dn = deC_rev_dn * prefactor + d * e_C_rev * e_C_rev * dz3_dn
    de_ds = deC_rev_ds * prefactor + d * e_C_rev * e_C_rev * dz3_ds
    de_dt = deC_rev_dt * prefactor + d * e_C_rev * e_C_rev * dz3_dt

    # Derivatives of f = n * e_C with respect to density, sigma and tau

    df_dn = e_C + density * de_dn
    df_ds = density * de_ds
    df_dt = density * de_dt

    return df_dn, df_ds, df_dt, e_C










def calculate_unrestricted_TPSS_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted TPSS correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted TPSS correlation energy density per particle
    
    """
    
    # Forms total density, cleaned total sigma and total tau

    density = alpha_density + beta_density
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    # Defining constant for TPSS

    d = 2.8 

    zeros = np.zeros_like(density)

    # PBE correlation (spin-polarised)

    df_dna_PBE, df_dnb_PBE, df_dsaa_PBE, df_dsbb_PBE, df_dsab_PBE, _, _, e_C_PBE = calculate_unrestricted_PBE_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None,  calculation=calculation)

    # one-spin limits for the max construction in TPSS

    df_dna_a0, _, df_dsaa_a0, _, _, _, _, e_C_a0 = calculate_unrestricted_PBE_correlation(alpha_density, zeros, alpha_density, sigma_aa, zeros, zeros, None, None,  calculation=calculation)
    _, df_dnb_0b, _, df_dsbb_0b, _, _, _, e_C_0b = calculate_unrestricted_PBE_correlation(zeros, beta_density, beta_density, zeros, sigma_bb, zeros, None, None,  calculation=calculation)

    inv_n = 1 / density

    # Derivatives from PBE with respect to density and sigma

    deC_PBE_dna = (df_dna_PBE - e_C_PBE) * inv_n
    deC_PBE_dnb = (df_dnb_PBE - e_C_PBE) * inv_n
    deC_PBE_dsaa = df_dsaa_PBE * inv_n
    deC_PBE_dsbb = df_dsbb_PBE * inv_n
    deC_PBE_dsab = df_dsab_PBE * inv_n

    inv_na = 1 / alpha_density
    inv_nb = 1 / beta_density

    # Derivatives of spin polarised extremes from PBE

    deC_a0_dna = (df_dna_a0 - e_C_a0) * inv_na
    deC_0b_dnb = (df_dnb_0b - e_C_0b) * inv_nb

    deC_a0_dsaa = df_dsaa_a0 * inv_na
    deC_0b_dsbb = df_dsbb_0b * inv_nb

    # Finds the maximum between the PBE and fully polarised limits

    condA = e_C_PBE >= e_C_a0
    condB = e_C_PBE >= e_C_0b
    
    e_C_tilde_alpha = np.where(condA, e_C_PBE, e_C_a0)
    e_C_tilde_beta = np.where(condB, e_C_PBE, e_C_0b)

    deC_tilde_alpha_dna = np.where(condA, deC_PBE_dna,  deC_a0_dna)
    deC_tilde_alpha_dnb = np.where(condA, deC_PBE_dnb,  0.0)
    deC_tilde_alpha_dsaa = np.where(condA, deC_PBE_dsaa, deC_a0_dsaa)
    deC_tilde_alpha_dsbb = np.where(condA, deC_PBE_dsbb, 0.0)
    deC_tilde_alpha_dsab = np.where(condA, deC_PBE_dsab, 0.0)

    deC_tilde_beta_dna = np.where(condB, deC_PBE_dna,  0.0)
    deC_tilde_beta_dnb = np.where(condB, deC_PBE_dnb,  deC_0b_dnb)
    deC_tilde_beta_dsaa = np.where(condB, deC_PBE_dsaa, 0.0)
    deC_tilde_beta_dsbb = np.where(condB, deC_PBE_dsbb, deC_0b_dsbb)
    deC_tilde_beta_dsab = np.where(condB, deC_PBE_dsab, 0.0)

    # weighted tilde: sum_sigma (n_sigma/n) * tilde_e_c^sigma

    numer_tilde = alpha_density * e_C_tilde_alpha + beta_density * e_C_tilde_beta
    e_C_tilde = numer_tilde * inv_n

    deC_tilde_dna = (e_C_tilde_alpha + alpha_density * deC_tilde_alpha_dna + beta_density  * deC_tilde_beta_dna - e_C_tilde) * inv_n
    deC_tilde_dnb = (e_C_tilde_beta + beta_density  * deC_tilde_beta_dnb + alpha_density * deC_tilde_alpha_dnb - e_C_tilde) * inv_n

    # Derivatives with respect to sigma spin channels

    deC_tilde_dsaa = (alpha_density * deC_tilde_alpha_dsaa + beta_density * deC_tilde_beta_dsaa) * inv_n
    deC_tilde_dsbb = (alpha_density * deC_tilde_alpha_dsbb + beta_density * deC_tilde_beta_dsbb) * inv_n
    deC_tilde_dsab = (alpha_density * deC_tilde_alpha_dsab + beta_density * deC_tilde_beta_dsab) * inv_n

    # z = tau_W / tau, used to make iso-orbital indicator

    z = sigma / (8 * tau * density)
    z_squared = z * z
    z_cubed = z_squared * z

    # Calculates spin polarisation and gradient of spin polarisation with respect to the density

    zeta = calculate_zeta(alpha_density, beta_density)
    zeta_squared = zeta * zeta
    inv_n2 = inv_n * inv_n

    dzeta_dna = 2 * beta_density * inv_n2
    dzeta_dnb = -2 * alpha_density * inv_n2

    # Key spin polarisation machinery

    one_minus = clean(1 - zeta, constants.sigma_floor)
    one_plus = 1 + zeta
    one_minus2 = one_minus * one_minus
    one_plus2  = one_plus * one_plus
    one_minus_z2 = 1 - zeta * zeta

    B = clean(one_minus2 * sigma_aa + one_plus2 * sigma_bb - 2 * one_minus_z2 * sigma_ab, floor=constants.sigma_floor)
    sqrtB = B ** (1 / 2)
    inv_sqrtB = 1 / sqrtB

    dB_dzeta = -2 * one_minus * sigma_aa + 2 * one_plus * sigma_bb + 4 * zeta * sigma_ab

    dB_dna = dB_dzeta * dzeta_dna
    dB_dnb = dB_dzeta * dzeta_dnb

    # Derivative of zeta with respect to the densities

    zeta_gradient = sqrtB * inv_n

    # Derivatives of gradient of zeta with respect to density spin channels

    dzeta_grad_dna = inv_sqrtB * dB_dna * inv_n / 2 - sqrtB * inv_n2
    dzeta_grad_dnb = inv_sqrtB * dB_dnb * inv_n / 2 - sqrtB * inv_n2

    dzeta_grad_dsaa = inv_sqrtB * one_minus2 * inv_n / 2
    dzeta_grad_dsbb = inv_sqrtB * one_plus2 * inv_n / 2
    dzeta_grad_dsab = -inv_sqrtB * one_minus_z2 * inv_n

    inv_den_xi = 1 / (2 * np.cbrt(3 * np.pi ** 2 * density))
    xi = zeta_gradient * inv_den_xi

    dxi_dna = inv_den_xi * dzeta_grad_dna - (1 / 3) * xi * inv_n
    dxi_dnb = inv_den_xi * dzeta_grad_dnb - (1 / 3) * xi * inv_n

    dxi_dsaa = inv_den_xi * dzeta_grad_dsaa
    dxi_dsbb = inv_den_xi * dzeta_grad_dsbb
    dxi_dsab = inv_den_xi * dzeta_grad_dsab

    # C(zeta, xi) from revised TPSS - this is one of two changes to the basic functional

    dC0_dzeta = 1.74 * zeta + 2 * zeta * zeta_squared + 13.56 * zeta * zeta_squared * zeta_squared
    dC0_dna = dC0_dzeta * dzeta_dna
    dC0_dnb = dC0_dzeta * dzeta_dnb

    inv_m43_plus = 1 / np.cbrt(one_plus) ** 4
    inv_m43_minus = 1 / np.cbrt(one_minus) ** 4
    s = inv_m43_plus + inv_m43_minus

    dinv_m43_plus_dzeta  = -(4 / 3) * inv_m43_plus / one_plus
    dinv_m43_minus_dzeta = (4 / 3) * inv_m43_minus / one_minus
    ds_dzeta = dinv_m43_plus_dzeta + dinv_m43_minus_dzeta

    A = xi * xi * s / 2

    dA_dna = xi * dxi_dna * s + (1 / 2) * xi * xi * ds_dzeta * dzeta_dna
    dA_dnb = xi * dxi_dnb * s + (1 / 2) * xi * xi * ds_dzeta * dzeta_dnb

    dA_dsaa = xi * dxi_dsaa * s
    dA_dsbb = xi * dxi_dsbb * s
    dA_dsab = xi * dxi_dsab * s

    inv_1pA = 1 / (1 + A)
    inv_1pA4 = inv_1pA ** 4

    # C from TPSS paper
    
    C_0 = 0.53 + 0.87 * zeta_squared + 0.50 * zeta_squared * zeta_squared + 2.26 * zeta_squared * zeta_squared * zeta_squared

    C = C_0 * inv_1pA4

    # Derivatives of C with respect to spin densities and sigma channels

    dC_dna = inv_1pA4 * (dC0_dna - 4 * C_0 * inv_1pA * dA_dna)
    dC_dnb = inv_1pA4 * (dC0_dnb - 4 * C_0 * inv_1pA * dA_dnb)

    dC_dsaa = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsaa)
    dC_dsbb = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsbb)
    dC_dsab = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsab)

    # z-derivatives with respect to the unrestricted variables

    dz_dna = -z * inv_n
    dz_dnb = -z * inv_n

    dz_dsaa = 1 / (8 * tau * density)
    dz_dsbb = dz_dsaa
    dz_dsab = 1 / (4 * tau * density)

    dz_dta = -z / tau
    dz_dtb = -z / tau

    dz2_dna = 2 * z * dz_dna
    dz2_dnb = 2 * z * dz_dnb
    dz2_dsaa = 2 * z * dz_dsaa
    dz2_dsbb = 2 * z * dz_dsbb
    dz2_dsab = 2 * z * dz_dsab
    dz2_dta = 2 * z * dz_dta
    dz2_dtb = 2 * z * dz_dtb

    dz3_dna = 3 * z_squared * dz_dna
    dz3_dnb = 3 * z_squared * dz_dnb
    dz3_dsaa = 3 * z_squared * dz_dsaa
    dz3_dsbb = 3 * z_squared * dz_dsbb
    dz3_dsab = 3 * z_squared * dz_dsab
    dz3_dta = 3 * z_squared * dz_dta
    dz3_dtb = 3 * z_squared * dz_dtb

    # revPKZB (TPSS correlation core)

    A_tpss = 1 + C * z_squared

    e_C_rev = e_C_PBE * A_tpss - (1 + C) * z_squared * e_C_tilde
    e_C = e_C_rev * (1 + d * e_C_rev * z_cubed)

    # de_C_rev / dx  (now includes dC/dx terms!)

    deC_rev_dna = (A_tpss * deC_PBE_dna + e_C_PBE * (z_squared * dC_dna + C * dz2_dna) - (1 + C) * (e_C_tilde * dz2_dna + z_squared * deC_tilde_dna) - z_squared * e_C_tilde * dC_dna)
    deC_rev_dnb = (A_tpss * deC_PBE_dnb + e_C_PBE * (z_squared * dC_dnb + C * dz2_dnb) - (1 + C) * (e_C_tilde * dz2_dnb + z_squared * deC_tilde_dnb) - z_squared * e_C_tilde * dC_dnb)

    deC_rev_dsaa = (A_tpss * deC_PBE_dsaa + e_C_PBE * (z_squared * dC_dsaa + C * dz2_dsaa) - (1 + C) * (e_C_tilde * dz2_dsaa + z_squared * deC_tilde_dsaa) - z_squared * e_C_tilde * dC_dsaa)
    deC_rev_dsbb = (A_tpss * deC_PBE_dsbb + e_C_PBE * (z_squared * dC_dsbb + C * dz2_dsbb) - (1 + C) * (e_C_tilde * dz2_dsbb + z_squared * deC_tilde_dsbb) - z_squared * e_C_tilde * dC_dsbb)
    deC_rev_dsab = (A_tpss * deC_PBE_dsab + e_C_PBE * (z_squared * dC_dsab + C * dz2_dsab) - (1 + C) * (e_C_tilde * dz2_dsab + z_squared * deC_tilde_dsab) - z_squared * e_C_tilde * dC_dsab)

    deC_rev_dta = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dta
    deC_rev_dtb = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dtb

    # final TPSS correlation: e_C = e_C_rev * (1 + d e_C_rev z^3)

    prefactor = 1 + 2 * d * e_C_rev * z_cubed

    deC_dna = deC_rev_dna * prefactor + d * e_C_rev ** 2 * dz3_dna
    deC_dnb = deC_rev_dnb * prefactor + d * e_C_rev ** 2 * dz3_dnb

    deC_dsaa = deC_rev_dsaa * prefactor + d * e_C_rev ** 2 * dz3_dsaa
    deC_dsbb = deC_rev_dsbb * prefactor + d * e_C_rev ** 2 * dz3_dsbb
    deC_dsab = deC_rev_dsab * prefactor + d * e_C_rev ** 2 * dz3_dsab

    deC_dta = deC_rev_dta * prefactor + d * e_C_rev ** 2 * dz3_dta
    deC_dtb = deC_rev_dtb * prefactor + d * e_C_rev ** 2 * dz3_dtb

    # convert to df/dx for f = n e_C 

    df_dn_alpha = e_C + density * deC_dna
    df_dn_beta  = e_C + density * deC_dnb

    # Derivatives with respect to sigma channels

    df_ds_aa = density * deC_dsaa
    df_ds_bb = density * deC_dsbb
    df_ds_ab = density * deC_dsab

    # Derivatives with respect to tau channels

    df_dt_alpha = density * deC_dta
    df_dt_beta  = density * deC_dtb

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C









    
def calculate_restricted_revTPSS_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """
    
    Calculates the restricted revised TPSS correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        df_dt (array): Derivative of f = n * e_C with respect to tau
        e_C (array): Restricted revised TPSS exchange energy density per particle
    
    """

    # Defining constants for revTPSS - the 0.53 here matches ORCA - but is wrong? Should be 0.59 per paper I think

    C, d = 0.53, 2.8

    # Repeatedly used quantities

    z = sigma / (8 * tau * density)

    z_squared = z * z
    z_cubed = z * z * z
    A = 1 + C * z_squared

    zeros = np.zeros_like(density)

    # Limits for unpolarised and fully polarised spin densities

    df_dn_PBE, df_ds_PBE, _, e_C_PBE = calculate_restricted_PBE_correlation(density, sigma, tau, calculation)
    df_dna_one, _, df_dsaa_one, _, _, _, _, e_C_PBE_one_spin = calculate_unrestricted_PBE_correlation(density / 2, zeros, density / 2, sigma / 4, zeros, zeros, None, None, calculation=calculation)

    # Picks out the largest PBE correlation

    e_C_tilde = np.maximum(e_C_PBE, e_C_PBE_one_spin)
    
    # Energy density for TPSS correlation

    e_C_rev = e_C_PBE * (1 + C * z_squared) - (1 + C) * z_squared * e_C_tilde
    e_C = e_C_rev * (1 + d * e_C_rev * z_cubed)

    # Derivative of PBE correlation with respect to density and sigma

    inv_n = 1 / density
    deC_PBE_dn = (df_dn_PBE - e_C_PBE) * inv_n
    deC_PBE_ds = df_ds_PBE * inv_n

    deC_one_dn = (df_dna_one - e_C_PBE_one_spin) * inv_n 
    deC_one_ds = (1 / 2) * df_dsaa_one * inv_n    

    # Derivative of largest PBE correlation with respect to density and sigma

    deC_tilde_dn = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_dn, deC_one_dn)
    deC_tilde_ds = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_ds, deC_one_ds)

    # Derivative of tau dependent terms with respect to density, sigma and tau
    
    dz_dn, dz_ds, dz_dt = -z * inv_n, 1 / (8 * tau * density), -z / tau
    dz2_dn, dz2_ds, dz2_dt = 2 * z * dz_dn, 2 * z * dz_ds, 2 * z * dz_dt
    dz3_dn, dz3_ds, dz3_dt = 3 * z_squared * dz_dn, 3 * z_squared * dz_ds, 3 * z_squared * dz_dt

    deC_rev_dn = A * deC_PBE_dn +  C * e_C_PBE * dz2_dn - (1 + C) * (e_C_tilde * dz2_dn + z_squared * deC_tilde_dn)
    deC_rev_ds = A * deC_PBE_ds +  C * e_C_PBE * dz2_ds - (1 + C) * (e_C_tilde * dz2_ds + z_squared * deC_tilde_ds)
    deC_rev_dt = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dt 

    # Commonly used prefactor

    prefactor = 1 + 2 * d * e_C_rev * z_cubed

    # Derivative of energy density with respect to density, sigma and tau

    de_dn = deC_rev_dn * prefactor + d * e_C_rev * e_C_rev * dz3_dn
    de_ds = deC_rev_ds * prefactor + d * e_C_rev * e_C_rev * dz3_ds
    de_dt = deC_rev_dt * prefactor + d * e_C_rev * e_C_rev * dz3_dt

    # Derivatives of f = n * e_C with respect to density, sigma and tau

    df_dn = e_C + density * de_dn
    df_ds = density * de_ds
    df_dt = density * de_dt

    return df_dn, df_ds, df_dt, e_C










def calculate_unrestricted_revTPSS_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted revised TPSS correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted TPSS correlation energy density per particle
    
    """
    
    # Forms total density, cleaned total sigma and total tau

    density = alpha_density + beta_density
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    # Defining constant for TPSS

    d = 2.8

    zeros = np.zeros_like(density)

    # PBE correlation (spin-polarised)

    df_dna_PBE, df_dnb_PBE, df_dsaa_PBE, df_dsbb_PBE, df_dsab_PBE, _, _, e_C_PBE = calculate_unrestricted_PBE_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, None)

    # one-spin limits for the max construction in TPSS

    df_dna_a0, _, df_dsaa_a0, _, _, _, _, e_C_a0 = calculate_unrestricted_PBE_correlation(alpha_density, zeros, alpha_density, sigma_aa, zeros, zeros, None, None, None)
    _, df_dnb_0b, _, df_dsbb_0b, _, _, _, e_C_0b = calculate_unrestricted_PBE_correlation(zeros, beta_density, beta_density, zeros, sigma_bb, zeros, None, None, None)

    inv_n = 1 / density

    # Derivatives from PBE with respect to density and sigma

    deC_PBE_dna = (df_dna_PBE - e_C_PBE) * inv_n
    deC_PBE_dnb = (df_dnb_PBE - e_C_PBE) * inv_n
    deC_PBE_dsaa = df_dsaa_PBE * inv_n
    deC_PBE_dsbb = df_dsbb_PBE * inv_n
    deC_PBE_dsab = df_dsab_PBE * inv_n

    inv_na = 1 / alpha_density
    inv_nb = 1 / beta_density

    # Derivatives of spin polarised extremes from PBE

    deC_a0_dna = (df_dna_a0 - e_C_a0) * inv_na
    deC_0b_dnb = (df_dnb_0b - e_C_0b) * inv_nb

    deC_a0_dsaa = df_dsaa_a0 * inv_na
    deC_0b_dsbb = df_dsbb_0b * inv_nb

    # Finds the maximum between the PBE and fully polarised limits

    condA = e_C_PBE >= e_C_a0
    condB = e_C_PBE >= e_C_0b
    
    e_C_tilde_alpha = np.where(condA, e_C_PBE, e_C_a0)
    e_C_tilde_beta = np.where(condB, e_C_PBE, e_C_0b)

    deC_tilde_alpha_dna = np.where(condA, deC_PBE_dna,  deC_a0_dna)
    deC_tilde_alpha_dnb = np.where(condA, deC_PBE_dnb,  0.0)
    deC_tilde_alpha_dsaa = np.where(condA, deC_PBE_dsaa, deC_a0_dsaa)
    deC_tilde_alpha_dsbb = np.where(condA, deC_PBE_dsbb, 0.0)
    deC_tilde_alpha_dsab = np.where(condA, deC_PBE_dsab, 0.0)

    deC_tilde_beta_dna = np.where(condB, deC_PBE_dna,  0.0)
    deC_tilde_beta_dnb = np.where(condB, deC_PBE_dnb,  deC_0b_dnb)
    deC_tilde_beta_dsaa = np.where(condB, deC_PBE_dsaa, 0.0)
    deC_tilde_beta_dsbb = np.where(condB, deC_PBE_dsbb, deC_0b_dsbb)
    deC_tilde_beta_dsab = np.where(condB, deC_PBE_dsab, 0.0)

    # weighted tilde: sum_sigma (n_sigma/n) * tilde_e_c^sigma

    numer_tilde = alpha_density * e_C_tilde_alpha + beta_density * e_C_tilde_beta
    e_C_tilde = numer_tilde * inv_n

    deC_tilde_dna = (e_C_tilde_alpha + alpha_density * deC_tilde_alpha_dna + beta_density  * deC_tilde_beta_dna - e_C_tilde) * inv_n
    deC_tilde_dnb = (e_C_tilde_beta + beta_density  * deC_tilde_beta_dnb + alpha_density * deC_tilde_alpha_dnb - e_C_tilde) * inv_n

    # Derivatives with respect to sigma spin channels

    deC_tilde_dsaa = (alpha_density * deC_tilde_alpha_dsaa + beta_density * deC_tilde_beta_dsaa) * inv_n
    deC_tilde_dsbb = (alpha_density * deC_tilde_alpha_dsbb + beta_density * deC_tilde_beta_dsbb) * inv_n
    deC_tilde_dsab = (alpha_density * deC_tilde_alpha_dsab + beta_density * deC_tilde_beta_dsab) * inv_n

    # z = tau_W / tau, used to make iso-orbital indicator

    z = sigma / (8 * tau * density)
    z_squared = z * z
    z_cubed = z_squared * z

    # Calculates spin polarisation and gradient of spin polarisation with respect to the density

    zeta = calculate_zeta(alpha_density, beta_density)
    zeta_squared = zeta * zeta
    inv_n2 = inv_n * inv_n

    dzeta_dna = 2 * beta_density * inv_n2
    dzeta_dnb = -2 * alpha_density * inv_n2

    # Key spin polarisation machinery

    one_minus = clean(1 - zeta, constants.sigma_floor)
    one_plus = 1 + zeta
    one_minus2 = one_minus * one_minus
    one_plus2  = one_plus * one_plus
    one_minus_z2 = 1 - zeta * zeta

    B = clean(one_minus2 * sigma_aa + one_plus2 * sigma_bb - 2 * one_minus_z2 * sigma_ab, floor=constants.sigma_floor)
    sqrtB = B ** (1 / 2)
    inv_sqrtB = 1 / sqrtB

    dB_dzeta = -2 * one_minus * sigma_aa + 2 * one_plus * sigma_bb + 4 * zeta * sigma_ab

    dB_dna = dB_dzeta * dzeta_dna
    dB_dnb = dB_dzeta * dzeta_dnb

    # Derivative of zeta with respect to the densities

    zeta_gradient = sqrtB * inv_n

    # Derivatives of gradient of zeta with respect to density spin channels

    dzeta_grad_dna = inv_sqrtB * dB_dna * inv_n / 2 - sqrtB * inv_n2
    dzeta_grad_dnb = inv_sqrtB * dB_dnb * inv_n / 2 - sqrtB * inv_n2

    dzeta_grad_dsaa = inv_sqrtB * one_minus2 * inv_n / 2
    dzeta_grad_dsbb = inv_sqrtB * one_plus2 * inv_n / 2
    dzeta_grad_dsab = -inv_sqrtB * one_minus_z2 * inv_n

    inv_den_xi = 1 / (2 * np.cbrt(3 * np.pi ** 2 * density))
    xi = zeta_gradient * inv_den_xi

    dxi_dna = inv_den_xi * dzeta_grad_dna - (1 / 3) * xi * inv_n
    dxi_dnb = inv_den_xi * dzeta_grad_dnb - (1 / 3) * xi * inv_n

    dxi_dsaa = inv_den_xi * dzeta_grad_dsaa
    dxi_dsbb = inv_den_xi * dzeta_grad_dsbb
    dxi_dsab = inv_den_xi * dzeta_grad_dsab

    # C(zeta,xi) from TPSS

    C_0 = 0.59 + 0.9269 * zeta_squared + 0.6225 * zeta_squared * zeta_squared + 2.1540 * zeta_squared * zeta_squared * zeta_squared

    dC0_dzeta = 1.74 * zeta + 2 * zeta * zeta_squared + 13.56 * zeta * zeta_squared * zeta_squared
    dC0_dna = dC0_dzeta * dzeta_dna
    dC0_dnb = dC0_dzeta * dzeta_dnb

    inv_m43_plus = 1 / np.cbrt(one_plus) ** 4
    inv_m43_minus = 1 / np.cbrt(one_minus) ** 4
    s = inv_m43_plus + inv_m43_minus

    dinv_m43_plus_dzeta  = -(4 / 3) * inv_m43_plus / one_plus
    dinv_m43_minus_dzeta = (4 / 3) * inv_m43_minus / one_minus
    ds_dzeta = dinv_m43_plus_dzeta + dinv_m43_minus_dzeta

    A = xi * xi * s / 2

    dA_dna = xi * dxi_dna * s + (1 / 2) * xi * xi * ds_dzeta * dzeta_dna
    dA_dnb = xi * dxi_dnb * s + (1 / 2) * xi * xi * ds_dzeta * dzeta_dnb

    dA_dsaa = xi * dxi_dsaa * s
    dA_dsbb = xi * dxi_dsbb * s
    dA_dsab = xi * dxi_dsab * s

    inv_1pA = 1 / (1 + A)
    inv_1pA4 = inv_1pA ** 4

    # C from TPSS paper

    C = C_0 * inv_1pA4

    # Derivatives of C with respect to spin densities and sigma channels

    dC_dna = inv_1pA4 * (dC0_dna - 4 * C_0 * inv_1pA * dA_dna)
    dC_dnb = inv_1pA4 * (dC0_dnb - 4 * C_0 * inv_1pA * dA_dnb)

    dC_dsaa = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsaa)
    dC_dsbb = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsbb)
    dC_dsab = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsab)

    # z-derivatives with respect to the unrestricted variables

    dz_dna = -z * inv_n
    dz_dnb = -z * inv_n

    dz_dsaa = 1 / (8 * tau * density)
    dz_dsbb = dz_dsaa
    dz_dsab = 1 / (4 * tau * density)

    dz_dta = -z / tau
    dz_dtb = -z / tau

    dz2_dna = 2 * z * dz_dna
    dz2_dnb = 2 * z * dz_dnb
    dz2_dsaa = 2 * z * dz_dsaa
    dz2_dsbb = 2 * z * dz_dsbb
    dz2_dsab = 2 * z * dz_dsab
    dz2_dta = 2 * z * dz_dta
    dz2_dtb = 2 * z * dz_dtb

    dz3_dna = 3 * z_squared * dz_dna
    dz3_dnb = 3 * z_squared * dz_dnb
    dz3_dsaa = 3 * z_squared * dz_dsaa
    dz3_dsbb = 3 * z_squared * dz_dsbb
    dz3_dsab = 3 * z_squared * dz_dsab
    dz3_dta = 3 * z_squared * dz_dta
    dz3_dtb = 3 * z_squared * dz_dtb

    # revPKZB (TPSS correlation core)

    A_tpss = 1 + C * z_squared

    e_C_rev = e_C_PBE * A_tpss - (1 + C) * z_squared * e_C_tilde
    e_C = e_C_rev * (1 + d * e_C_rev * z_cubed)

    # de_C_rev / dx  (now includes dC/dx terms!)

    deC_rev_dna = (A_tpss * deC_PBE_dna + e_C_PBE * (z_squared * dC_dna + C * dz2_dna) - (1 + C) * (e_C_tilde * dz2_dna + z_squared * deC_tilde_dna) - z_squared * e_C_tilde * dC_dna)
    deC_rev_dnb = (A_tpss * deC_PBE_dnb + e_C_PBE * (z_squared * dC_dnb + C * dz2_dnb) - (1 + C) * (e_C_tilde * dz2_dnb + z_squared * deC_tilde_dnb) - z_squared * e_C_tilde * dC_dnb)

    deC_rev_dsaa = (A_tpss * deC_PBE_dsaa + e_C_PBE * (z_squared * dC_dsaa + C * dz2_dsaa) - (1 + C) * (e_C_tilde * dz2_dsaa + z_squared * deC_tilde_dsaa) - z_squared * e_C_tilde * dC_dsaa)
    deC_rev_dsbb = (A_tpss * deC_PBE_dsbb + e_C_PBE * (z_squared * dC_dsbb + C * dz2_dsbb) - (1 + C) * (e_C_tilde * dz2_dsbb + z_squared * deC_tilde_dsbb) - z_squared * e_C_tilde * dC_dsbb)
    deC_rev_dsab = (A_tpss * deC_PBE_dsab + e_C_PBE * (z_squared * dC_dsab + C * dz2_dsab) - (1 + C) * (e_C_tilde * dz2_dsab + z_squared * deC_tilde_dsab) - z_squared * e_C_tilde * dC_dsab)

    deC_rev_dta = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dta
    deC_rev_dtb = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dtb

    # final TPSS correlation: e_C = e_C_rev * (1 + d e_C_rev z^3)

    prefactor = 1 + 2 * d * e_C_rev * z_cubed

    deC_dna = deC_rev_dna * prefactor + d * e_C_rev ** 2 * dz3_dna
    deC_dnb = deC_rev_dnb * prefactor + d * e_C_rev ** 2 * dz3_dnb

    deC_dsaa = deC_rev_dsaa * prefactor + d * e_C_rev ** 2 * dz3_dsaa
    deC_dsbb = deC_rev_dsbb * prefactor + d * e_C_rev ** 2 * dz3_dsbb
    deC_dsab = deC_rev_dsab * prefactor + d * e_C_rev ** 2 * dz3_dsab

    deC_dta = deC_rev_dta * prefactor + d * e_C_rev ** 2 * dz3_dta
    deC_dtb = deC_rev_dtb * prefactor + d * e_C_rev ** 2 * dz3_dtb

    # convert to df/dx for f = n e_C 

    df_dn_alpha = e_C + density * deC_dna
    df_dn_beta  = e_C + density * deC_dnb

    # Derivatives with respect to sigma channels

    df_ds_aa = density * deC_dsaa
    df_ds_bb = density * deC_dsbb
    df_ds_ab = density * deC_dsab

    # Derivatives with respect to tau channels

    df_dt_alpha = density * deC_dta
    df_dt_beta  = density * deC_dtb

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C





def calculate_restricted_SCAN_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
    
    
    b_1c = 0.0285764
    b_2c = 0.0889
    b_3c = 0.125541

    c_1c = 0.64
    c_2c = 1.5
    d_c = 0.7


    r_s, inv_density = calculate_seitz_radius(density)

    cbrt_density = clean(np.cbrt(density))

    s_squared = sigma / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)

    tau_w = sigma / (8 * density)
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    alpha = (tau - tau_w) / tau_u
    one_minus_alpha = 1 - alpha

    f_c = np.zeros_like(alpha) 

    if np.any(alpha < 1):

        f_c[alpha < 1] = np.exp(-c_1c * alpha[alpha < 1] / one_minus_alpha[alpha < 1])

    if np.any(alpha > 1):

        f_c[alpha > 1] = -d_c * np.exp(c_2c / one_minus_alpha[alpha > 1])


    e_C_LDA_0 = -b_1c / (1 + b_2c * r_s ** (1 / 2) + b_3c * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1c) - 1

    chi_inf_zeta_0 = 0.128026

    g_inf = 1 / (1 + 4 * chi_inf_zeta_0 * s_squared) ** (1 / 4)

    H_0 = b_1c * np.log(1 + w_0 * (1 - g_inf))

    gamma = 0.031091
    beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)
    
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA_1 = calculate_unrestricted_PW_correlation(density / 2, density / 2, density, None, None, None, None, None, None)

    w_1 = np.exp(-e_C_LDA_1 / (gamma)) - 1

    A = beta / (gamma * w_1)

    t_squared = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * s_squared / (r_s)

    g = 1 / (1 + 4 * A * t_squared) ** (1 / 4)

    H_1 = gamma * np.log(1 + w_1 * (1 - g))

    e_C_0 = e_C_LDA_0 + H_0
    e_C_1 = e_C_LDA_1 + H_1

    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    df_dn = np.zeros_like(e_C)
    df_ds = np.zeros_like(e_C)
    df_dt = np.zeros_like(e_C)

    return df_dn, df_ds, df_dt, e_C










def calculate_unrestricted_SCAN_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted SCAN correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted SCAN correlation energy density per particle
    
    """
    
    b_1c = 0.0285764
    b_2c = 0.0889
    b_3c = 0.125541

    c_1c = 0.64
    c_2c = 1.5
    d_c = 0.7

    # Forms total density, cleaned total sigma and total tau

    density = alpha_density + beta_density
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    zeta = calculate_zeta(alpha_density, beta_density)

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (1 / 2) * (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus)
    phi_squared = phi * phi
    phi_cubed = phi_squared * phi

    r_s, inv_density = calculate_seitz_radius(density)

    d_x = (np.cbrt(1 + zeta) ** 4 + np.cbrt(1 - zeta) ** 4) / 2

    cbrt_density = clean(np.cbrt(density))

    s_squared = sigma / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)

    d_s = (1 / 2) * (cbrt_plus ** 5 + cbrt_minus ** 5)

    tau_w = sigma / (8 * density)
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5 * d_s

    alpha = (tau - tau_w) / tau_u
    one_minus_alpha = 1 - alpha

    f_c = np.zeros_like(alpha) 

    if np.any(alpha < 1):

        f_c[alpha < 1] = np.exp(-c_1c * alpha[alpha < 1] / one_minus_alpha[alpha < 1])

    if np.any(alpha > 1):

        f_c[alpha > 1] = -d_c * np.exp(c_2c / one_minus_alpha[alpha > 1])

    G_c = (1 - 2.3631 * (d_x - 1)) * (1 - zeta ** 12)

    e_C_LDA_0 = -b_1c / (1 + b_2c * r_s ** (1 / 2) + b_3c * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1c) - 1

    f_0 = -0.9
    c_x = -3 / (4 * np.pi) * np.cbrt(9 * np.pi / 4) * d_x

    chi_inf = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * (0.066725 * (0.1 / 0.1778)) * phi / (c_x - f_0)

    chi_inf_zeta_0 = 0.128026

    g_inf = 1 / np.sqrt(np.sqrt((1 + 4 * chi_inf_zeta_0 * s_squared)))

    H_0 = b_1c * np.log(1 + w_0 * (1 - g_inf))

    gamma = 0.031091
    beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)
    
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA_1 = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)

    w_1 = np.exp(-e_C_LDA_1 / (gamma * phi_cubed)) - 1

    A = beta / (gamma * w_1)

    t_squared = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * s_squared / (phi * phi * r_s)

    g = 1 / np.sqrt(np.sqrt((1 + 4 * A * t_squared)))

    H_1 = gamma * phi_cubed * np.log(1 + w_1 * (1 - g))

    e_C_0 = (e_C_LDA_0 + H_0) * G_c
    e_C_1 = e_C_LDA_1 + H_1


    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    df_dn_alpha = np.zeros_like(e_C)
    df_dn_beta = np.zeros_like(e_C)
    df_ds_aa = np.zeros_like(e_C)
    df_ds_bb = np.zeros_like(e_C)
    df_ds_ab = np.zeros_like(e_C)
    df_dt_alpha = np.zeros_like(e_C)
    df_dt_beta = np.zeros_like(e_C)


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C










def calculate_restricted_rSCAN_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
    
    
    b_1c = 0.0285764
    b_2c = 0.0889
    b_3c = 0.125541

    c_1c = 0.64
    c_2c = 1.5
    d_c = 0.7


    r_s, inv_density = calculate_seitz_radius(density)

    cbrt_density = clean(np.cbrt(density))

    s_squared = sigma / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)

    tau_w = sigma / (8 * density)
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    alpha = (tau - tau_w) / tau_u
    one_minus_alpha = 1 - alpha

    f_c = np.zeros_like(alpha) 

    if np.any(alpha < 1):

        f_c[alpha < 1] = np.exp(-c_1c * alpha[alpha < 1] / one_minus_alpha[alpha < 1])

    if np.any(alpha > 1):

        f_c[alpha > 1] = -d_c * np.exp(c_2c / one_minus_alpha[alpha > 1])


    e_C_LDA_0 = -b_1c / (1 + b_2c * r_s ** (1 / 2) + b_3c * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1c) - 1

    chi_inf_zeta_0 = 0.128026

    g_inf = 1 / (1 + 4 * chi_inf_zeta_0 * s_squared) ** (1 / 4)

    H_0 = b_1c * np.log(1 + w_0 * (1 - g_inf))

    gamma = 0.031091
    beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)
    
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA_1 = calculate_unrestricted_PW_correlation(density / 2, density / 2, density, None, None, None, None, None, None)

    w_1 = np.exp(-e_C_LDA_1 / (gamma)) - 1

    A = beta / (gamma * w_1)

    t_squared = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * s_squared / (r_s)

    g = 1 / (1 + 4 * A * t_squared) ** (1 / 4)

    H_1 = gamma * np.log(1 + w_1 * (1 - g))

    e_C_0 = e_C_LDA_0 + H_0
    e_C_1 = e_C_LDA_1 + H_1

    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    df_dn = np.zeros_like(e_C)
    df_ds = np.zeros_like(e_C)
    df_dt = np.zeros_like(e_C)

    return df_dn, df_ds, df_dt, e_C










def calculate_unrestricted_rSCAN_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted rSCAN correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted rSCAN correlation energy density per particle
    
    """
    
    b_1c = 0.0285764
    b_2c = 0.0889
    b_3c = 0.125541

    c_1c = 0.64
    c_2c = 1.5
    d_c = 0.7

    # Forms total density, cleaned total sigma and total tau

    density = alpha_density + beta_density
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    zeta = calculate_zeta(alpha_density, beta_density)

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (1 / 2) * (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus)
    phi_squared = phi * phi
    phi_cubed = phi_squared * phi

    r_s, inv_density = calculate_seitz_radius(density)

    d_x = (np.cbrt(1 + zeta) ** 4 + np.cbrt(1 - zeta) ** 4) / 2

    cbrt_density = clean(np.cbrt(density))

    s_squared = sigma / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 8)

    d_s = (1 / 2) * (cbrt_plus ** 5 + cbrt_minus ** 5)

    tau_w = sigma / (8 * density)
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5 * d_s

    alpha = (tau - tau_w) / tau_u
    one_minus_alpha = 1 - alpha

    f_c = np.zeros_like(alpha) 

    if np.any(alpha < 1):

        f_c[alpha < 1] = np.exp(-c_1c * alpha[alpha < 1] / one_minus_alpha[alpha < 1])

    if np.any(alpha > 1):

        f_c[alpha > 1] = -d_c * np.exp(c_2c / one_minus_alpha[alpha > 1])

    G_c = (1 - 2.3631 * (d_x - 1)) * (1 - zeta ** 12)

    e_C_LDA_0 = -b_1c / (1 + b_2c * r_s ** (1 / 2) + b_3c * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1c) - 1

    f_0 = -0.9
    c_x = -3 / (4 * np.pi) * np.cbrt(9 * np.pi / 4) * d_x

    chi_inf = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * (0.066725 * (0.1 / 0.1778)) * phi / (c_x - f_0)

    chi_inf_zeta_0 = 0.128026

    g_inf = 1 / (1 + 4 * chi_inf_zeta_0 * s_squared) ** (1 / 4)

    H_0 = b_1c * np.log(1 + w_0 * (1 - g_inf))

    gamma = 0.031091
    beta = 0.066725 * (1 + 0.1 * r_s) / (1 + 0.1778 * r_s)
    
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA_1 = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)
    #_, e_C_LDA_1, de1_dr = calculate_PW_potential(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)

    w_1 = np.exp(-e_C_LDA_1 / (gamma * phi_cubed)) - 1

    A = beta / (gamma * w_1)

    t_squared = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * s_squared / (phi * phi * r_s)

    g = 1 / (1 + 4 * A * t_squared) ** (1 / 4)

    H_1 = gamma * phi_cubed * np.log(1 + w_1 * (1 - g))

    e_C_0 = (e_C_LDA_0 + H_0) * G_c
    e_C_1 = e_C_LDA_1 + H_1


    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    df_dn_alpha = np.zeros_like(e_C)
    df_dn_beta = np.zeros_like(e_C)
    df_ds_aa = np.zeros_like(e_C)
    df_ds_bb = np.zeros_like(e_C)
    df_ds_ab = np.zeros_like(e_C)
    df_dt_alpha = np.zeros_like(e_C)
    df_dt_beta = np.zeros_like(e_C)


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C










def calculate_restricted_r2SCAN_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the restricted r2SCAN correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.
    
    A reasonably efficient implementation of equations in supporting information of 10.1021/acs.jpclett.0c02405.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        df_dt (array): Derivative of f = n * e_C with respect to tau
        e_C (array): Restricted r2SCAN exchange energy density per particle
    
    """

    eta = 0.001

    b_1 = 0.0285764
    b_2 = 0.0889
    b_3 = 0.125541

    c_1 = 0.64
    c_2 = 1.5

    d_p = 0.361
    d_c = 0.7 

    gamma = 0.0310907

    # These are the coefficients for the smoother switching function, a degree seven polynomial

    c_c = [1, -0.64, -0.4352, -1.535685604549, 3.061560252175, -1.915710236206, 0.516884468372, -0.051848879792]

    delta_f_c = np.sum(c_c[1:] * np.linspace(1, 7, 7))

    r_s, inv_density = calculate_seitz_radius(density)
    sqrt_r_s = r_s ** (1 / 2)
    cbrt_density = np.cbrt(density)

    dtau_w_ds = inv_density / 8

    tau_w = sigma * dtau_w_ds
    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5

    tau_minus_tau_w = tau - tau_w

    # Useful iso-orbital indicator-dependent quantities are precalculated here

    alpha_bar_denominator = tau_u + eta * tau_w
    inv_alpha_bar_denominator = 1 / alpha_bar_denominator
    inv_alpha_bar_denominator_squared = inv_alpha_bar_denominator * inv_alpha_bar_denominator
    alpha_bar = tau_minus_tau_w * inv_alpha_bar_denominator
    
    one_minus_alpha_bar = 1 - alpha_bar
    inv_one_minus_alpha_bar = 1 / one_minus_alpha_bar

    # Polynomial switching function to smooth the transition

    f_c = ((((((c_c[7] * alpha_bar + c_c[6]) * alpha_bar + c_c[5]) * alpha_bar + c_c[4]) * alpha_bar + c_c[3]) * alpha_bar + c_c[2]) * alpha_bar + c_c[1]) * alpha_bar + c_c[0]

    # Exponential limits from original SCAN paper

    small_alpha_exponent_term = np.exp(np.clip(-c_1 * alpha_bar * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))
    large_alpha_exponent_term = -d_c * np.exp(np.clip(c_2 * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))

    f_c = np.where(alpha_bar < 0, small_alpha_exponent_term, f_c)
    f_c = np.where(alpha_bar > 2.5, large_alpha_exponent_term, f_c)

    df_dn_LSDA, _, _, e_C_LSDA = calculate_restricted_PW_correlation(density, sigma, tau, calculation)

    e_C_LDA_0 = -b_1 / (1 + b_2 * sqrt_r_s + b_3 * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1) - 1
    w_1 = np.exp(-e_C_LSDA / gamma) - 1

    beta_denominator = 1 + 0.1778 * r_s
    beta_numerator = 1 + 0.1 * r_s

    beta = 0.066725 * beta_numerator / beta_denominator
    chi_inf = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * 0.066725 / (1.778 * (0.9 - 3 * np.cbrt(3 / (16 * np.pi)) ** 2))
    
    e_C_LSDA_0 = e_C_LDA_0

    # Calculate r_s derivatives of the LSDA and LDA0 correlation energies

    derivative_denominator = 1 + b_2 * sqrt_r_s + b_3 * r_s
    derivative_numerator = 0.5 * b_2 / sqrt_r_s + b_3

    de_C_LDA_0_dr_s = b_1 * derivative_numerator / (derivative_denominator * derivative_denominator) 
    de_C_LSDA_0_dr_s = de_C_LDA_0_dr_s
    de_C_LSDA_dr_s = -(3 / r_s) * (df_dn_LSDA - e_C_LSDA)

    k_F = calculate_Fermi_wavevector(cbrt_density = cbrt_density)
    k_F_squared = k_F * k_F

    k_s = (4 * k_F / np.pi) ** (1 / 2)
    k_s_squared = k_s * k_s
    
    density_squared = density * density

    s_squared = sigma / (4 * density_squared * k_F_squared) 
    s_fourth = s_squared * s_squared
    g_inf = (1 + 4 * chi_inf * s_squared) ** (-1 / 4)
    
    t_squared = sigma / (4 * k_s_squared * density_squared)
    beta_over_gamma_w_1 = beta / (gamma * w_1)
    y = beta_over_gamma_w_1 * t_squared
    
    s_fourth_exponent_term = s_fourth / d_p ** 4
    exp_s = np.exp(-s_fourth_exponent_term)
    delta_y = delta_f_c / (27 * gamma * w_1) * s_squared * exp_s * (20 * r_s * (de_C_LSDA_0_dr_s - de_C_LSDA_dr_s) - 45 * eta * (e_C_LSDA_0 - e_C_LSDA))

    g = (1 + 4 * (y - delta_y)) ** (-1 / 4)
    
    # Using log1p is slightly more stable than log(1 + x) for small x 

    H_1 = gamma * np.log1p(w_1 * (1 - g))
    H_0 = b_1 * np.log1p(w_0 * (1 - g_inf))

    e_C_0 = (e_C_LDA_0 + H_0)
    e_C_1 = e_C_LSDA + H_1

    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    # Derivatives of switching function

    df_c_dalpha_bar_poly = (((((7 * c_c[7] * alpha_bar + 6 * c_c[6]) * alpha_bar + 5 * c_c[5]) * alpha_bar + 4 * c_c[4]) * alpha_bar + 3 * c_c[3]) * alpha_bar + 2 * c_c[2]) * alpha_bar + c_c[1]
    df_c_dalpha_bar_small = small_alpha_exponent_term * (-c_1) * inv_one_minus_alpha_bar ** 2
    df_c_dalpha_bar_large = large_alpha_exponent_term * c_2 * inv_one_minus_alpha_bar ** 2

    df_c_dalpha_bar = np.where(alpha_bar < 0, df_c_dalpha_bar_small, df_c_dalpha_bar_poly)
    df_c_dalpha_bar = np.where(alpha_bar > 2.5, df_c_dalpha_bar_large, df_c_dalpha_bar)

    # Derivtives of the iso-orbital indicator

    dalpha_bar_dt = inv_alpha_bar_denominator
    dalpha_bar_ds = -dtau_w_ds * (tau_u + eta * tau) * inv_alpha_bar_denominator_squared
    
    dnum_dn = tau_w * inv_density    
    ddenom_dn = (5 / 3) * tau_u * inv_density + eta * (-tau_w * inv_density)
    dalpha_bar_dn = (dnum_dn * alpha_bar_denominator - tau_minus_tau_w * ddenom_dn) * inv_alpha_bar_denominator_squared

    dr_s_dn = -r_s / (3 * density)

    de_C_LDA_0_dn = de_C_LDA_0_dr_s * dr_s_dn
    de_C_LSDA_dn = (df_dn_LSDA - e_C_LSDA) * inv_density

    w_0_plus_1 = w_0 + 1
    w_1_plus_1 = w_1 + 1

    dw_0_dn = w_0_plus_1 * (-1 / b_1) * de_C_LDA_0_dn
    dw_1_dn = w_1_plus_1 * (-1 / gamma) * de_C_LSDA_dn

    # Derivative of beta
    
    dbeta_dr_s = 0.066725 * (0.1 * beta_denominator - 0.1778 * beta_numerator) / (beta_denominator * beta_denominator)
    dbeta_dn = dbeta_dr_s * dr_s_dn

    ds_squared_ds = 1 / (4 * density_squared * k_F_squared)
    ds_squared_dn = -(8 / 3) * s_squared * inv_density
    dt_squared_ds = 1 / (4 * k_s_squared * density_squared)
    dt_squared_dn = -(7 / 3) * t_squared * inv_density

    dg_inf_ds_squared = -chi_inf * g_inf ** 5

    # Derivatives of y

    dy_ds = beta_over_gamma_w_1 * dt_squared_ds
    dy_dn = (dbeta_dn * t_squared + beta * dt_squared_dn) / (gamma * w_1) - y * dw_1_dn / w_1

    # Derivatives of delta y

    de_difference = de_C_LSDA_0_dr_s - de_C_LSDA_dr_s

    A_delta = delta_f_c / (27 * gamma * w_1)
    B_delta = 20 * r_s * de_difference - 45 * eta * (e_C_LSDA_0 - e_C_LSDA)

    ddelta_y_ds_squared = A_delta * B_delta * exp_s * (1 - 2 * s_fourth_exponent_term)
    ddelta_y_ds = ddelta_y_ds_squared * ds_squared_ds

    dA_delta_dn = -A_delta * dw_1_dn / w_1
    dexp_s_dn = exp_s * (-2 * s_squared / d_p ** 4) * ds_squared_dn

    d2e_C_LDA_0_dr_s2 = (b_1 * (-0.25 * b_2 * r_s ** (-1.5)) / (derivative_denominator * derivative_denominator) - 2 * de_C_LDA_0_dr_s * derivative_numerator / derivative_denominator)

    # Second derivatives of PW92 LDA correlation energy

    a1_pw = 0.21370
    b1_pw, b2_pw, b3_pw, b4_pw = 7.5957, 3.5876, 1.6382, 0.49294
    Q_pw = 2 * gamma * (b1_pw * sqrt_r_s + b2_pw * r_s + b3_pw * r_s * sqrt_r_s + b4_pw * r_s * r_s)
    dQ_pw = 2 * gamma * (0.5 * b1_pw / sqrt_r_s + b2_pw + 1.5 * b3_pw * sqrt_r_s + 2 * b4_pw * r_s)
    d2Q_pw = 2 * gamma * (-0.25 * b1_pw * r_s ** (-1.5) + 0.75 * b3_pw / sqrt_r_s + 2 * b4_pw)
    QQ1 = Q_pw * (Q_pw + 1)
    term_ab = 2 * 2 * gamma * a1_pw * dQ_pw / QQ1
    term_c = 2 * gamma * (1 + a1_pw * r_s) * (d2Q_pw * QQ1 - dQ_pw * dQ_pw * (2 * Q_pw + 1)) / (QQ1 * QQ1)
    d2e_C_LSDA_dr_s2 = term_ab + term_c

    dB_delta_dr_s = 20 * r_s * (d2e_C_LDA_0_dr_s2 - d2e_C_LSDA_dr_s2) + 20 * de_difference - 45 * eta * de_difference
    dB_delta_dn = dB_delta_dr_s * dr_s_dn

    ddelta_y_dn = dA_delta_dn * s_squared * exp_s * B_delta + A_delta * ds_squared_dn * exp_s * B_delta + A_delta * s_squared * dexp_s_dn * B_delta + A_delta * s_squared * exp_s * dB_delta_dn

    dg_du = -g ** 5
    du_g_ds = dy_ds - ddelta_y_ds
    du_g_dn = dy_dn - ddelta_y_dn

    # Derivatives of H_0 and H_1
    
    arg_H1 = 1 + w_1 * (1 - g)
    dg_dn = dg_du * du_g_dn
    dH1_dn = gamma * (dw_1_dn * (1 - g) - w_1 * dg_dn) / arg_H1
    dH1_ds = gamma * w_1 * (-dg_du * du_g_ds) / arg_H1

    arg_H0 = 1 + w_0 * (1 - g_inf)
    dg_inf_dn = dg_inf_ds_squared * ds_squared_dn
    dg_inf_ds = dg_inf_ds_squared * ds_squared_ds
    dH0_dn = b_1 * (dw_0_dn * (1 - g_inf) - w_0 * dg_inf_dn) / arg_H0
    dH0_ds = b_1 * w_0 * (-dg_inf_ds) / arg_H0

    # Derivatives of e_C_0 and e_C_1

    de_C_0_dn = de_C_LDA_0_dn + dH0_dn
    de_C_0_ds = dH0_ds
    de_C_1_dn = de_C_LSDA_dn + dH1_dn
    de_C_1_ds = dH1_ds

    e_C_0_minus_e_C_1 = e_C_0 - e_C_1

    df_c_dn = df_c_dalpha_bar * dalpha_bar_dn
    df_c_ds = df_c_dalpha_bar * dalpha_bar_ds
    df_c_dt = df_c_dalpha_bar * dalpha_bar_dt

    # Derivatives of the correlation energy density

    de_C_dn = de_C_1_dn + df_c_dn * e_C_0_minus_e_C_1 + f_c * (de_C_0_dn - de_C_1_dn)
    de_C_ds = de_C_1_ds + df_c_ds * e_C_0_minus_e_C_1 + f_c * (de_C_0_ds - de_C_1_ds)
    de_C_dt = df_c_dt * e_C_0_minus_e_C_1

    df_dn = e_C + density * de_C_dn
    df_ds = density * de_C_ds
    df_dt = density * de_C_dt

    return df_dn, df_ds, df_dt, e_C










def calculate_unrestricted_r2SCAN_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted r2SCAN correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted r2SCAN correlation energy density per particle
    
    """
    
    eta = 0.001

    b_1 = 0.0285764
    b_2 = 0.0889
    b_3 = 0.125541

    c_1 = 0.64
    c_2 = 1.5

    d_p = 0.361
    d_c = 0.7 

    gamma = 0.0310907
    
    # These are the coefficients for the smoother switching function, a degree seven polynomial

    c_c = [1, -0.64, -0.4352, -1.535685604549, 3.061560252175, -1.915710236206, 0.516884468372, -0.051848879792]

    delta_f_c = np.sum(c_c[1:] * np.linspace(1, 7, 7))

    density = alpha_density + beta_density
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor = constants.sigma_floor)
    tau = tau_alpha + tau_beta
    
    zeta = calculate_zeta(alpha_density, beta_density)

    # Quantities are pre-calculated here that are repeatedly used later

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    cbrt_plus_squared = cbrt_plus * cbrt_plus
    cbrt_minus_squared = cbrt_minus * cbrt_minus

    cbrt_plus_fourth_power = cbrt_plus_squared * cbrt_plus_squared
    cbrt_minus_fourth_power = cbrt_minus_squared * cbrt_minus_squared

    d_s = (cbrt_plus * cbrt_plus_fourth_power + cbrt_minus * cbrt_minus_fourth_power) / 2
    d_x = (cbrt_plus_squared * cbrt_plus_squared + cbrt_minus_squared * cbrt_minus_squared) / 2
    phi = (cbrt_plus_squared + cbrt_minus_squared) / 2

    r_s, inv_density = calculate_seitz_radius(density)
    cbrt_density = np.cbrt(density)

    sqrt_r_s = r_s ** (1 / 2)
    phi_squared = phi * phi
    phi_cubed = phi_squared * phi

    dtau_w_ds = inv_density / 8

    tau_w = sigma * dtau_w_ds

    # The d_s factor is not in the r2SCAN paper - but it is in the original SCAN paper, and needed to get the correct answer

    tau_u = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * cbrt_density ** 5 * d_s

    tau_minus_tau_w = tau - tau_w

    alpha_bar_denominator = tau_u + eta * tau_w
    inv_alpha_bar_denominator = 1 / alpha_bar_denominator
    alpha_bar = tau_minus_tau_w * inv_alpha_bar_denominator
    
    one_minus_alpha_bar = 1 - alpha_bar
    inv_one_minus_alpha_bar = 1 / one_minus_alpha_bar

    # Polynomial switching function

    f_c = ((((((c_c[7] * alpha_bar + c_c[6]) * alpha_bar + c_c[5]) * alpha_bar + c_c[4]) * alpha_bar + c_c[3]) * alpha_bar + c_c[2]) * alpha_bar + c_c[1]) * alpha_bar + c_c[0]

    # Limits from the original SCAN paper

    small_alpha_exponent_term = np.exp(np.clip(-c_1 * alpha_bar * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))
    large_alpha_exponent_term = -d_c * np.exp(np.clip(c_2 * inv_one_minus_alpha_bar, None, constants.exponent_ceiling))

    f_c = np.where(alpha_bar < 0, small_alpha_exponent_term, f_c)
    f_c = np.where(alpha_bar > 2.5, large_alpha_exponent_term, f_c)

    # Calculates the LSDA correlation energy and its derivatives

    df_dn_alpha_LSDA, df_dn_beta_LSDA, _, _, _, _, _, e_C_LSDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    e_C_LDA_0 = -b_1 / (1 + b_2 * r_s ** (1 / 2) + b_3 * r_s)

    w_0 = np.exp(-e_C_LDA_0 / b_1) - 1
    w_1 = np.exp(-e_C_LSDA / (gamma * phi_cubed)) - 1
    
    beta_denominator = 1 + 0.1778 * r_s

    beta = 0.066725 * (1 + 0.1 * r_s) / beta_denominator
    chi_inf = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * 0.066725 / (1.778 * (0.9 - 3 * np.cbrt(3 / (16 * np.pi)) ** 2))
    
    G_c = (1 - 2.3631 * (d_x - 1)) * (1 - zeta ** 12)
    e_C_LSDA_0 = e_C_LDA_0 * G_c

    # Calculates the r_s derivatives of the LSDA and LDA0 correlation energies

    de_C_LDA_0_dr_s = b_1 * (0.5 * b_2 * r_s ** (-0.5) + b_3) / (1 + b_2 * r_s ** 0.5 + b_3 * r_s) ** 2
    de_C_LSDA_0_dr_s = de_C_LDA_0_dr_s * G_c
    de_C_LSDA_dr_s = -(3 / r_s) * (0.5 * (1 + zeta) * df_dn_alpha_LSDA + 0.5 * (1 - zeta) * df_dn_beta_LSDA - e_C_LSDA)

    k_F = calculate_Fermi_wavevector(cbrt_density=cbrt_density)

    k_s = (4 * k_F / np.pi) ** (1 / 2)
    
    s_squared = sigma / (4 * density * density * k_F * k_F) 
    s_fourth = s_squared * s_squared
    g_inf = (1 + 4 * chi_inf * s_squared) ** (-1 / 4)
    k_F_squared = k_F * k_F
    k_s_squared = k_s * k_s
    density_squared = density * density
    inv_alpha_bar_denominator_squared = inv_alpha_bar_denominator * inv_alpha_bar_denominator

    derivative_denominator = 1 + b_2 * sqrt_r_s + b_3 * r_s
    derivative_numerator = 0.5 * b_2 / sqrt_r_s + b_3
    de_C_LDA_0_dr_s = b_1 * derivative_numerator / (derivative_denominator * derivative_denominator)

    s_fourth_exponent_term = s_fourth / d_p ** 4
    exp_s = np.exp(-s_fourth_exponent_term)
    de_difference = de_C_LSDA_0_dr_s - de_C_LSDA_dr_s
    e_C_difference = e_C_LSDA_0 - e_C_LSDA
    A_delta = delta_f_c / (27 * gamma * d_s * phi_cubed * w_1)
    B_delta = 20 * r_s * de_difference - 45 * eta * e_C_difference
    beta_over_gamma_w_1 = beta / (gamma * w_1)

    delta_y = A_delta * s_squared * exp_s * B_delta  # reuse precomputed terms
    
    # Dimensionally corrected t_squared
    t_squared = sigma / (4 * k_s * k_s * phi_squared * density_squared)
    y = beta * t_squared / (gamma * w_1)
    
    g = (1 + 4 * (y - delta_y)) ** (-1 / 4)
    
    H_1 = gamma * phi_cubed * np.log(1 + w_1 * (1 - g))
    H_0 = b_1 * np.log(1 + w_0 * (1 - g_inf))

    e_C_0 = (e_C_LDA_0 + H_0) * G_c
    e_C_1 = e_C_LSDA + H_1

    e_C = e_C_1 + f_c * (e_C_0 - e_C_1)

    dzeta_dn_alpha = (1 - zeta) * inv_density
    dzeta_dn_beta = -(1 + zeta) * inv_density

    # Zeta derivatives of spin-scaling functions

    inv_cbrt_plus = 1 / cbrt_plus
    inv_cbrt_minus = 1 / cbrt_minus

    dphi_dzeta = (1 / 3) * (inv_cbrt_plus - inv_cbrt_minus)
    dd_s_dzeta = (5 / 6) * (cbrt_plus_squared - cbrt_minus_squared)
    dd_x_dzeta = (2 / 3) * (cbrt_plus - cbrt_minus)

    dphi_cubed_dzeta = 3 * phi_squared * dphi_dzeta

    G_c_factor_1 = 1 - 2.3631 * (d_x - 1)
    G_c_factor_2 = 1 - zeta ** 12
    dG_c_dzeta = -2.3631 * dd_x_dzeta * G_c_factor_2 - 12 * zeta ** 11 * G_c_factor_1

    # Derivatives of switching function

    df_c_dalpha_bar_poly = (((((7 * c_c[7] * alpha_bar + 6 * c_c[6]) * alpha_bar + 5 * c_c[5]) * alpha_bar + 4 * c_c[4]) * alpha_bar + 3 * c_c[3]) * alpha_bar + 2 * c_c[2]) * alpha_bar + c_c[1]
    df_c_dalpha_bar_small = small_alpha_exponent_term * (-c_1) * inv_one_minus_alpha_bar ** 2
    df_c_dalpha_bar_large = large_alpha_exponent_term * c_2 * inv_one_minus_alpha_bar ** 2

    df_c_dalpha_bar = np.where(alpha_bar < 0, df_c_dalpha_bar_small, df_c_dalpha_bar_poly)
    df_c_dalpha_bar = np.where(alpha_bar > 2.5, df_c_dalpha_bar_large, df_c_dalpha_bar)

    # Derivatives of the iso-orbital indicator

    dtau_u_dzeta = tau_u * dd_s_dzeta / d_s

    dnum_dn = tau_w * inv_density
    ddenom_dn = (5 / 3) * tau_u * inv_density + eta * (-tau_w * inv_density)
    dalpha_bar_dn = (dnum_dn * alpha_bar_denominator - tau_minus_tau_w * ddenom_dn) * inv_alpha_bar_denominator_squared
    dalpha_bar_dzeta = -tau_minus_tau_w * dtau_u_dzeta * inv_alpha_bar_denominator_squared

    dalpha_bar_ds = -dtau_w_ds * (tau_u + eta * tau) * inv_alpha_bar_denominator_squared
    dalpha_bar_dt = inv_alpha_bar_denominator

    df_c_ds = df_c_dalpha_bar * dalpha_bar_ds
    df_c_dt = df_c_dalpha_bar * dalpha_bar_dt

    dr_s_dn = -r_s / (3 * density)

    # LSDA energy derivatives

    de_C_LSDA_dn_alpha = (df_dn_alpha_LSDA - e_C_LSDA) * inv_density
    de_C_LSDA_dn_beta = (df_dn_beta_LSDA - e_C_LSDA) * inv_density
    de_C_LSDA_dn = 0.5 * (1 + zeta) * de_C_LSDA_dn_alpha + 0.5 * (1 - zeta) * de_C_LSDA_dn_beta
    de_C_LSDA_dzeta = 0.5 * (df_dn_alpha_LSDA - df_dn_beta_LSDA)

    de_C_LDA_0_dn = de_C_LDA_0_dr_s * dr_s_dn

    # Derivatives of w_0 and w_1

    w_0_plus_1 = w_0 + 1
    w_1_plus_1 = w_1 + 1

    dw_0_dn = w_0_plus_1 * (-1 / b_1) * de_C_LDA_0_dn

    inv_gamma_phi_cubed = 1 / (gamma * phi_cubed)

    dw_1_dn = w_1_plus_1 * (-inv_gamma_phi_cubed) * de_C_LSDA_dn
    dw_1_dzeta = w_1_plus_1 * (-inv_gamma_phi_cubed) * (de_C_LSDA_dzeta - e_C_LSDA * dphi_cubed_dzeta / phi_cubed)

    # Derivative of beta

    dbeta_dr_s = 0.066725 * (0.1 * beta_denominator - 0.1778 * (1 + 0.1 * r_s)) / (beta_denominator * beta_denominator)
    dbeta_dn = dbeta_dr_s * dr_s_dn

    ds_squared_ds = 1 / (4 * density_squared * k_F_squared)
    ds_squared_dn = -(8 / 3) * s_squared * inv_density
    dt_squared_ds = 1 / (4 * k_s_squared * phi_squared * density_squared)
    dt_squared_dn = -(7 / 3) * t_squared * inv_density
    dt_squared_dzeta = -2 * t_squared * dphi_dzeta / phi

    dg_inf_ds_squared = -chi_inf * (1 + 4 * chi_inf * s_squared) ** (-5 / 4)

    # Derivatives of y

    dy_ds = beta_over_gamma_w_1 * dt_squared_ds
    dy_dn = (dbeta_dn * t_squared + beta * dt_squared_dn) / (gamma * w_1) - y * dw_1_dn / w_1
    dy_dzeta = beta * dt_squared_dzeta / (gamma * w_1) - y * dw_1_dzeta / w_1

    # Derivatives of delta y

    dA_delta_dn = -A_delta * dw_1_dn / w_1
    dA_delta_dzeta = -A_delta * (dd_s_dzeta / d_s + dphi_cubed_dzeta / phi_cubed + dw_1_dzeta / w_1)

    dexp_s_dn = exp_s * (-2 * s_squared / d_p ** 4) * ds_squared_dn

    ddelta_y_ds_squared = A_delta * B_delta * exp_s * (1 - 2 * s_fourth_exponent_term)
    ddelta_y_ds = ddelta_y_ds_squared * ds_squared_ds

    # PW92 three-channel second derivatives for dB_delta - very awkard and not pretty

    A_0, a1_0, b1_0, b2_0, b3_0, b4_0 = 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    A_1, a1_1, b1_1, b2_1, b3_1, b4_1 = 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517
    A_a, a1_a, b1_a, b2_a, b3_a, b4_a = 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671

    inv_sqrt_r_s = 1 / sqrt_r_s
    r_s_neg_three_halves = inv_sqrt_r_s * inv_sqrt_r_s * inv_sqrt_r_s

    pw92_dG = []
    pw92_d2G = []

    for A_i, a1_i, b1_i, b2_i, b3_i, b4_i in [

        (A_0, a1_0, b1_0, b2_0, b3_0, b4_0),
        (A_1, a1_1, b1_1, b2_1, b3_1, b4_1),
        (A_a, a1_a, b1_a, b2_a, b3_a, b4_a),

    ]:
        
        Q_i = 2 * A_i * (b1_i * sqrt_r_s + b2_i * r_s + b3_i * r_s * sqrt_r_s + b4_i * r_s * r_s)
        dQ_i = 2 * A_i * (0.5 * b1_i * inv_sqrt_r_s + b2_i + 1.5 * b3_i * sqrt_r_s + 2 * b4_i * r_s)
        d2Q_i = 2 * A_i * (-0.25 * b1_i * r_s_neg_three_halves + 0.75 * b3_i * inv_sqrt_r_s + 2 * b4_i)

        QQ1_i = Q_i * (Q_i + 1)

        pw92_dG.append(-2 * A_i * a1_i * np.log1p(1 / Q_i) + 2 * A_i * (1 + a1_i * r_s) * dQ_i / QQ1_i)
        pw92_d2G.append(4 * A_i * a1_i * dQ_i / QQ1_i + 2 * A_i * (1 + a1_i * r_s) * (d2Q_i * QQ1_i - dQ_i * dQ_i * (2 * Q_i + 1)) / (QQ1_i * QQ1_i))

    # Spin interpolation function and its derivative

    two_to_four_thirds_minus_2 = 2 ** (4 / 3) - 2
    f_zeta = (cbrt_plus_fourth_power + cbrt_minus_fourth_power - 2) / two_to_four_thirds_minus_2
    f_prime_zeta = (4 / 3) * (cbrt_plus - cbrt_minus) / two_to_four_thirds_minus_2
    inv_f_double_prime_0 = (9 / 8) * two_to_four_thirds_minus_2

    zeta_cubed = zeta * zeta * zeta
    zeta_fourth = zeta_cubed * zeta

    zeta_term_1 = f_prime_zeta * zeta_fourth + 4 * zeta_cubed * f_zeta
    zeta_term_2 = f_prime_zeta * (1 - zeta_fourth) - 4 * zeta_cubed * f_zeta

    d2e_C_LSDA_dr_s2 = (pw92_d2G[0] + (pw92_d2G[1] - pw92_d2G[0]) * f_zeta * zeta_fourth - pw92_d2G[2] * f_zeta * inv_f_double_prime_0 * (1 - zeta_fourth))

    d2e_C_LSDA_dr_s_dzeta = ((pw92_dG[1] - pw92_dG[0]) * zeta_term_1 - pw92_dG[2] * zeta_term_2 * inv_f_double_prime_0)
    
    # Second r_s derivative of e_C_LDA_0

    d2e_C_LDA_0_dr_s2 = (b_1 * (-0.25 * b_2 * r_s_neg_three_halves) / (derivative_denominator * derivative_denominator) - 2 * de_C_LDA_0_dr_s * derivative_numerator / derivative_denominator)

    # B_delta derivatives

    dB_delta_dr_s = 20 * de_difference + 20 * r_s * (d2e_C_LDA_0_dr_s2 * G_c - d2e_C_LSDA_dr_s2) - 45 * eta * de_difference
    dB_delta_dn = dB_delta_dr_s * dr_s_dn

    dB_delta_dzeta = 20 * r_s * (de_C_LDA_0_dr_s * dG_c_dzeta - d2e_C_LSDA_dr_s_dzeta) - 45 * eta * (e_C_LDA_0 * dG_c_dzeta - de_C_LSDA_dzeta)

    ddelta_y_dn =(dA_delta_dn * s_squared * exp_s * B_delta + A_delta * ds_squared_dn * exp_s * B_delta + A_delta * s_squared * dexp_s_dn * B_delta + A_delta * s_squared * exp_s * dB_delta_dn)

    ddelta_y_dzeta = dA_delta_dzeta * s_squared * exp_s * B_delta + A_delta * s_squared * exp_s * dB_delta_dzeta

    dg_du = -(1 + 4 * (y - delta_y)) ** (-5 / 4)
    du_g_ds = dy_ds - ddelta_y_ds
    du_g_dn = dy_dn - ddelta_y_dn
    du_g_dzeta = dy_dzeta - ddelta_y_dzeta

    dg_dn = dg_du * du_g_dn
    dg_dzeta = dg_du * du_g_dzeta
    dg_ds = dg_du * du_g_ds

    # Derivatives of H_1

    arg_H1 = 1 + w_1 * (1 - g)
    log_arg_H1 = H_1 / (gamma * phi_cubed)

    dH1_dn = gamma * phi_cubed * (dw_1_dn * (1 - g) - w_1 * dg_dn) / arg_H1
    dH1_dzeta = dphi_cubed_dzeta * gamma * log_arg_H1 + gamma * phi_cubed * (dw_1_dzeta * (1 - g) - w_1 * dg_dzeta) / arg_H1
    dH1_ds = gamma * phi_cubed * (-w_1 * dg_ds) / arg_H1

    # Derivatives of H_0

    arg_H0 = 1 + w_0 * (1 - g_inf)
    dg_inf_dn = dg_inf_ds_squared * ds_squared_dn
    dg_inf_ds = dg_inf_ds_squared * ds_squared_ds
    dH0_dn = b_1 * (dw_0_dn * (1 - g_inf) - w_0 * dg_inf_dn) / arg_H0
    dH0_ds = b_1 * w_0 * (-dg_inf_ds) / arg_H0

    # Derivatives of e_C_0 and e_C_1

    de_C_0_dn = (de_C_LDA_0_dn + dH0_dn) * G_c
    de_C_0_dzeta = (e_C_LDA_0 + H_0) * dG_c_dzeta
    de_C_0_ds = dH0_ds * G_c

    de_C_1_dn = de_C_LSDA_dn + dH1_dn
    de_C_1_dzeta = de_C_LSDA_dzeta + dH1_dzeta
    de_C_1_ds = dH1_ds

    e_C_0_minus_e_C_1 = e_C_0 - e_C_1

    # Derivatives of the correlation energy density

    de_C_dn = de_C_1_dn + df_c_dalpha_bar * dalpha_bar_dn * e_C_0_minus_e_C_1 + f_c * (de_C_0_dn - de_C_1_dn)
    de_C_dzeta = de_C_1_dzeta + df_c_dalpha_bar * dalpha_bar_dzeta * e_C_0_minus_e_C_1 + f_c * (de_C_0_dzeta - de_C_1_dzeta)
    de_C_ds = de_C_1_ds + df_c_ds * e_C_0_minus_e_C_1 + f_c * (de_C_0_ds - de_C_1_ds)
    de_C_dt = df_c_dt * e_C_0_minus_e_C_1

    de_C_dn_alpha = de_C_dn + de_C_dzeta * dzeta_dn_alpha
    de_C_dn_beta = de_C_dn + de_C_dzeta * dzeta_dn_beta

    df_dn_alpha = e_C + density * de_C_dn_alpha
    df_dn_beta = e_C + density * de_C_dn_beta

    df_ds_aa = density * de_C_ds
    df_ds_bb = df_ds_aa
    df_ds_ab = 2 * df_ds_aa

    df_dt_alpha = density * de_C_dt
    df_dt_beta = df_dt_alpha


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C










def calculate_restricted_B97_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the restricted B97 correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.
    
    A fairly efficient implementation of the equations in 10.1063/1.475007.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        df_dt (array): Derivative of f = n * e_C with respect to tau
        e_C (array): Restricted B97 correlation energy density per particle
    
    """

    # The parameters can be for Becke's hybrid (first case) or Grimme's dispersion-corrected GGA (second case)

    c_ab = [0.9454, 0.7471, -4.5961] if calculation.method.name == "B97" else [0.69041, 6.30270, -14.9712]
    c_ss = [0.1737, 2.3487, -2.4868] if calculation.method.name == "B97" else [0.22340, -1.56208, 1.94293]

    # For some reason the B97-D correlation here (not B97) disagrees with ORCA, which disagrees with LibXC - the differences are sub-microhartree

    gamma_ss = 0.2
    gamma_ab = 0.006

    # Frequently useful quantities

    zeros = np.zeros_like(density)

    cbrt_density = np.cbrt(density)
    inv_density = 1 / density

    s_squared = np.cbrt(4) * sigma / cbrt_density ** 8

    # Parameter which is used in the gradient expansion

    inv_ss_denominator_term = 1 / (1 + gamma_ss * s_squared)
    inv_ab_denominator_term = 1 / (1 + gamma_ab * s_squared)

    x_ss = gamma_ss * s_squared * inv_ss_denominator_term
    x_ab = gamma_ab * s_squared * inv_ab_denominator_term

    # Enhancement functions for the parallel and anti-parallel spin channels

    g_ss = c_ss[0] + (c_ss[1] + c_ss[2] * x_ss) * x_ss
    g_ab = c_ab[0] + (c_ab[1] + c_ab[2] * x_ab) * x_ab

    df_dn_LSDA, _, _, e_C_LSDA = calculate_restricted_PW_correlation(density, sigma, tau, calculation)
    df_dn_LSDA_ss, _, _, _, _, _, _, e_C_LSDA_ss = calculate_unrestricted_PW_correlation(density / 2, zeros, density / 2, None, None, None, None, None, calculation)

    # Final spin-restricted correlation energy density per particle

    g_diff = g_ss - g_ab

    e_C = g_diff * e_C_LSDA_ss + g_ab * e_C_LSDA

    # Fundamental derivatives, reused often

    d_s_squared_ds = np.cbrt(4) / cbrt_density ** 8
    d_s_squared_dn = -(8 / 3) * s_squared / density

    dg_ss_dx_ss = c_ss[1] + 2 * c_ss[2] * x_ss
    dg_ab_dx_ab = c_ab[1] + 2 * c_ab[2] * x_ab

    # Derivatives of x with respect to s_squared

    dx_ss_d_s_squared = gamma_ss * inv_ss_denominator_term * inv_ss_denominator_term
    dx_ab_d_s_squared = gamma_ab * inv_ab_denominator_term * inv_ab_denominator_term

    # Derivatives of the spin-scaling functions via the chain rule

    dg_ss_dn = dg_ss_dx_ss * dx_ss_d_s_squared * d_s_squared_dn
    dg_ab_dn = dg_ab_dx_ab * dx_ab_d_s_squared * d_s_squared_dn

    dg_ss_ds = dg_ss_dx_ss * dx_ss_d_s_squared * d_s_squared_ds
    dg_ab_ds = dg_ab_dx_ab * dx_ab_d_s_squared * d_s_squared_ds

    # Derivatives of the LSDA correlation energy density

    de_dn_LSDA_ss = (df_dn_LSDA_ss - e_C_LSDA_ss) * inv_density
    de_dn_LSDA = (df_dn_LSDA - e_C_LSDA) * inv_density
    
    dg_dn_diff = dg_ss_dn - dg_ab_dn
    dg_ds_diff = dg_ss_ds - dg_ab_ds

    # Final derivatives

    df_dn = e_C + density * (de_dn_LSDA_ss * g_diff + g_ab * de_dn_LSDA + e_C_LSDA_ss * dg_dn_diff + e_C_LSDA * dg_ab_dn)

    df_ds = density * (e_C_LSDA_ss * dg_ds_diff + e_C_LSDA * dg_ab_ds)

    return df_dn, df_ds, None, e_C










def calculate_unrestricted_B97_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted B97 correlation energy density and derivative with respect to the density, square gradient and kinetic energy density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        df_dt_alpha (array): Derivative of f = n * e_C with respect to tau alpha
        df_dt_beta (array): Derivative of f = n * e_C with respect to tau beta
        e_C (array): Unrestricted B97 correlation energy density per particle
    
    """
    
    # The parameters can be for Becke's hybrid (first case) or Grimme's dispersion-corrected GGA (second case)

    c_ab = [0.9454, 0.7471, -4.5961] if calculation.method.name == "B97" else [0.69041, 6.30270, -14.9712]
    c_ss = [0.1737, 2.3487, -2.4868] if calculation.method.name == "B97" else [0.22340, -1.56208, 1.94293]

    # For some reason the B97-D correlation here (not B97) disagrees with ORCA, which disagrees with LibXC - the differences are sub-microhartree

    gamma_ss = 0.2
    gamma_ab = 0.006

    zeros = np.zeros_like(density)

    inv_density = 1 / density

    cbrt_alpha_density = np.cbrt(alpha_density)
    cbrt_beta_density = np.cbrt(beta_density)

    s_squared_alpha = sigma_aa / cbrt_alpha_density ** 8
    s_squared_beta = sigma_bb /  cbrt_beta_density ** 8

    s_squared_average = (1 / 2) * (s_squared_alpha + s_squared_beta)

    x_alpha = gamma_ss * s_squared_alpha / (1 + gamma_ss * s_squared_alpha)
    x_beta = gamma_ss * s_squared_beta / (1 + gamma_ss * s_squared_beta)

    x_alpha_beta = gamma_ab * s_squared_average / (1 + gamma_ab * s_squared_average)

    g_alpha = c_ss[0] + (c_ss[1] + c_ss[2] * x_alpha) * x_alpha
    g_beta = c_ss[0] + (c_ss[1] + c_ss[2] * x_beta) * x_beta

    g_alpha_beta = c_ab[0] + (c_ab[1] + c_ab[2] * x_alpha_beta) * x_alpha_beta

    df_dn_alpha_LSDA, df_dn_beta_LSDA, _, _, _, _, _, e_C_LSDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)
    
    df_dn_alpha_LSDA_alpha, df_dn_beta_LSDA_alpha, _, _, _, _, _, e_C_LSDA_alpha = calculate_unrestricted_PW_correlation(alpha_density, zeros, alpha_density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)
    df_dn_alpha_LSDA_beta, df_dn_beta_LSDA_beta, _, _, _, _, _, e_C_LSDA_beta = calculate_unrestricted_PW_correlation(zeros, beta_density, beta_density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    e_C_LSDA_ab = e_C_LSDA * density - e_C_LSDA_alpha * alpha_density - e_C_LSDA_beta * beta_density

    e_C_per_volume = g_alpha * e_C_LSDA_alpha * alpha_density + g_beta * e_C_LSDA_beta * beta_density + g_alpha_beta * e_C_LSDA_ab 

    e_C = e_C_per_volume * inv_density

    dg_alpha_dx_alpha = c_ss[1] + 2 * c_ss[2] * x_alpha
    dg_beta_dx_beta = c_ss[1] + 2 * c_ss[2] * x_beta

    dg_alpha_dx_alpha_beta = c_ab[1] + 2 * c_ab[2] * x_alpha_beta

    ds_squared_alpha_ds_aa = s_squared_alpha / sigma_aa
    ds_squared_beta_ds_bb = s_squared_beta / sigma_bb

    dx_alpha_ds_squared_alpha = gamma_ss / ((1 + gamma_ss * s_squared_alpha) * (1 + gamma_ss * s_squared_alpha))
    dx_beta_ds_squared_beta = gamma_ss / ((1 + gamma_ss * s_squared_beta) * (1 + gamma_ss * s_squared_beta))
    dx_alpha_beta_ds_squared_average = gamma_ab / ((1 + gamma_ab * s_squared_average) * (1 + gamma_ab * s_squared_average))

    dx_alpha_ds_aa = dx_alpha_ds_squared_alpha * ds_squared_alpha_ds_aa
    dx_beta_ds_bb = dx_beta_ds_squared_beta * ds_squared_beta_ds_bb

    dx_alpha_beta_ds_aa = dx_alpha_beta_ds_squared_average * (1 / 2) * ds_squared_alpha_ds_aa
    dx_alpha_beta_ds_bb = dx_alpha_beta_ds_squared_average * (1 / 2) * ds_squared_beta_ds_bb

    dg_alpha_ds_aa = dg_alpha_dx_alpha * dx_alpha_ds_aa
    dg_beta_ds_bb = dg_beta_dx_beta * dx_beta_ds_bb

    dg_alpha_beta_ds_aa = dg_alpha_dx_alpha_beta * dx_alpha_beta_ds_aa
    dg_alpha_beta_ds_bb = dg_alpha_dx_alpha_beta * dx_alpha_beta_ds_bb

    de_C_per_volume_ds_aa = e_C_LSDA_alpha * alpha_density * dg_alpha_ds_aa + e_C_LSDA_ab * dg_alpha_beta_ds_aa
    de_C_per_volume_ds_bb = e_C_LSDA_beta * beta_density * dg_beta_ds_bb + e_C_LSDA_ab * dg_alpha_beta_ds_bb

    df_ds_aa = de_C_per_volume_ds_aa
    df_ds_bb = de_C_per_volume_ds_bb
    df_ds_ab = np.zeros_like(e_C)


    df_dn_alpha = np.zeros_like(e_C)
    df_dn_beta = np.zeros_like(e_C)



    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C










def calculate_restricted_3P_correlation(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:
   
    """
    
    Calculates the restricted three parameter correlation energy density and derivative with respect to the density and square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        calculation (Calculation): Calculation object
    
    Returns:
        df_dn (array): Derivative of f = n * e_C with respect to density
        df_ds (array): Derivative of f = n * e_C with respect to sigma
        e_C (array): Restricted three-parameter exchange energy density per particle
    
    """
   
    method = calculation.method.name

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V

    df_dn_LDA, _, _, e_C_LDA = calculate_restricted_VWN3_correlation(density, None, None, calculation) if "G" in method else calculate_restricted_VWN5_correlation(density, None, None, calculation)

    # Picks the GGA correlation depending on the method

    if "LYP" in method: correlation_functional = calculate_restricted_LYP_correlation
    if "PW" in method: correlation_functional = calculate_restricted_PW91_correlation
    if "P86" in method: correlation_functional = calculate_restricted_P86_correlation

    # Calculates the energy density and derivatives for the GGA part

    df_dn_GGA, df_ds_GGA, _, e_C_GGA = correlation_functional(density, sigma, None, calculation)

    # These parameters are the standard B3LYP coefficients for correlation

    df_dn = 0.81 * df_dn_GGA + 0.19 * df_dn_LDA
    df_ds = 0.81 * df_ds_GGA
    e_C = 0.81 * e_C_GGA + 0.19 * e_C_LDA

    return df_dn, df_ds, None, e_C









def calculate_unrestricted_3P_correlation(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:
    
    """
    
    Calculates the unrestricted three-parameter correlation energy density and derivative with respect to the density and square gradient.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        df_ds_aa (array): Derivative of f = n * e_C with respect to sigma alpha-alpha
        df_ds_bb (array): Derivative of f = n * e_C with respect to sigma beta-beta
        df_ds_ab (array): Derivative of f = n * e_C with respect to sigma alpha-beta
        e_C (array): Unrestricted three-parameter correlation energy density per particle
    
    """

    method = calculation.method.name

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V

    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_unrestricted_VWN3_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, calculation) if "G" in method else calculate_unrestricted_VWN5_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, calculation)
    
    # Picks the GGA correlation depending on the method

    if "LYP" in method: correlation_functional = calculate_unrestricted_LYP_correlation
    if "PW" in method: correlation_functional = calculate_unrestricted_PW91_correlation
    if "P86" in method: correlation_functional = calculate_unrestricted_P86_correlation

    # Calculates the energy density and derivatives for the GGA part

    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, _, _, e_C = correlation_functional(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)
    
    # These parameters are the standard B3LYP coefficients for correlation

    df_dn_alpha = 0.81 * df_dn_alpha + 0.19 * df_dn_alpha_LDA
    df_dn_beta = 0.81 * df_dn_beta + 0.19 * df_dn_beta_LDA
    
    df_ds_aa = 0.81 * df_ds_aa
    df_ds_bb = 0.81 * df_ds_bb
    df_ds_ab = 0.81 * df_ds_ab
    
    e_C = 0.81 * e_C + 0.19 * e_C_LDA

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C










exchange_functionals = {

    "S": calculate_Slater_exchange,
    "PBE": calculate_PBE_exchange,
    "RPBE": calculate_RPBE_exchange,
    "REVPBE": calculate_PBE_exchange,
    "B": calculate_B88_exchange,
    "B3": calculate_B3_exchange,
    "TPSS": calculate_TPSS_exchange,
    "REVTPSS": calculate_revTPSS_exchange,
    "SCAN": calculate_SCAN_exchange,
    "RSCAN": calculate_rSCAN_exchange,
    "R2SCAN": calculate_r2SCAN_exchange,
    "PW": calculate_PW91_exchange,
    "MPW": calculate_mPW91_exchange,
    "B97": calculate_B97_exchange,

}










correlation_functionals = {

    "VWN3": calculate_restricted_VWN3_correlation,
    "UVWN3": calculate_unrestricted_VWN3_correlation,
    "VWN5": calculate_restricted_VWN5_correlation,
    "UVWN5": calculate_unrestricted_VWN5_correlation,
    "PW": calculate_restricted_PW_correlation,
    "UPW": calculate_unrestricted_PW_correlation,
    "PW91": calculate_restricted_PW91_correlation,
    "UPW91": calculate_unrestricted_PW91_correlation,
    "P86": calculate_restricted_P86_correlation,
    "UP86": calculate_unrestricted_P86_correlation,
    "PBE": calculate_restricted_PBE_correlation,
    "UPBE": calculate_unrestricted_PBE_correlation,
    "LYP": calculate_restricted_LYP_correlation,
    "ULYP": calculate_unrestricted_LYP_correlation,    
    "3P": calculate_restricted_3P_correlation,
    "U3P": calculate_unrestricted_3P_correlation,
    "TPSS": calculate_restricted_TPSS_correlation,
    "UTPSS": calculate_unrestricted_TPSS_correlation,
    "REVTPSS": calculate_restricted_revTPSS_correlation,
    "UREVTPSS": calculate_unrestricted_revTPSS_correlation,
    "SCAN": calculate_restricted_SCAN_correlation,
    "USCAN": calculate_unrestricted_SCAN_correlation,
    "RSCAN": calculate_restricted_rSCAN_correlation,
    "URSCAN": calculate_unrestricted_rSCAN_correlation,
    "R2SCAN": calculate_restricted_r2SCAN_correlation,
    "UR2SCAN": calculate_unrestricted_r2SCAN_correlation,
    "B97": calculate_restricted_B97_correlation,
    "UB97": calculate_unrestricted_B97_correlation,

}