import numpy as np
from numpy import ndarray
from TUNA.tuna_calc import Calculation
from TUNA.tuna_util import constants
from TUNA.tuna_xc import clean, calculate_zeta, calculate_f_zeta, calculate_f_prime_zeta, calculate_seitz_radius, calculate_Fermi_wavevector
from TUNA.tuna_xc import calculate_Slater_exchange, calculate_VWN_potential, calculate_PW_potential
from TUNA.tuna_xc import calculate_restricted_PW_correlation, calculate_unrestricted_PW_correlation


"""

This is the TUNA module for exchange-correlation kernels, written first for version 0.12.0.

The kernels are the second derivatives of the exchange-correlation energy density wrt. the (spin) density and the square density gradients, which
TD-DFT contracts with the transition density. The functionals and potentials the kernels are built from are imported from tuna_xc, and the same
conventions are used here, so ** (1 / 2) for square rooting and np.cbrt() for cube rooting.

Each kernel comes in a singlet form for a restricted reference, a triplet form for the same reference and a spin-resolved form for an unrestricted
one. Exchange spin scales exactly, so for exchange a single pair of generic functions rescales the restricted kernel into the other two. Note that
PBE, PW91 and P86 correlation only see the total square gradient, so their triplet kernels are a single density block, and that the B88, mPW91 and
P86 kernels clean sigma at the square of the density floor.

The module contains:

1. The exchange kernels, including the generic triplet and unrestricted ones (calculate_B88_exchange_kernel, calculate_GGA_exchange_spin_kernel, etc.)
2. The correlation kernels, and their helper functions (calculate_restricted_VWN5_correlation_kernel, calculate_unrestricted_LYP_correlation_kernel, etc.)
3. Some dictionaries for all the implemented kernels

"""





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ E X C H A N G E    K E R N E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #





def calculate_Slater_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the Slater exchange kernel.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object
    
    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
    
    """

    # Modifiable with "XA" keyword

    alpha = calculation.X_alpha

    inv_cbrt_density = 1 / np.cbrt(density)

    # Calculates the kernel

    d2f_dn2 = - (alpha / 2) * np.cbrt(3 / np.pi) * inv_cbrt_density * inv_cbrt_density

    return d2f_dn2










def calculate_GGA_exchange_kernel_blocks(density: ndarray, t: ndarray, dt_ds: ndarray, F_X: ndarray, dF_dt: ndarray, d2F_dt2: ndarray, f_LDA: ndarray) -> tuple:

    """

    Calculates the three second derivatives of a GGA exchange energy from its enhancement factor.

    Args:
        density (array): Electron density on integration grid
        t (array): Reduced gradient variable of the functional
        dt_ds (array): Derivative of t with respect to sigma, which is independent of sigma
        F_X (array): Exchange enhancement factor
        dF_dt (array): First derivative of the enhancement factor with respect to t
        d2F_dt2 (array): Second derivative of the enhancement factor with respect to t
        f_LDA (array): Local density exchange energy per unit volume, n * e_X_LDA

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Second derivatives with respect to the density and sigma, for any t proportional to sigma / n ^ (8 / 3)

    d2f_dn2 = (4 / 9) * (f_LDA / (density * density)) * (F_X + 6 * t * dF_dt + 16 * t * t * d2F_dt2)

    d2f_dnds = -(4 / 3) * (f_LDA / density) * (dF_dt + 2 * t * d2F_dt2) * dt_ds

    d2f_ds2 = f_LDA * d2F_dt2 * dt_ds * dt_ds

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_arcsinh_over_argument(u: ndarray) -> ndarray:

    """

    Calculates arcsinh(u) / u, which is smooth at the origin but indeterminate there when evaluated directly.

    Args:
        u (array): Argument

    Returns:
        arcsinh_over_u (array): The ratio, taken to its limiting value of one at the origin

    """

    # Below this cutoff the series is cheaper and more accurate

    cutoff = 0.01

    u_safe = np.maximum(u, cutoff)

    arcsinh_over_u = np.where(u < cutoff, 1 - u * u / 6 + 3 * u ** 4 / 40, np.arcsinh(u_safe) / u_safe)

    return arcsinh_over_u










def calculate_PW91_arcsinh_curvature(u: ndarray) -> ndarray:

    """

    Calculates (1 / sqrt(1 + u ^ 2) - arcsinh(u) / u) / u ^ 2, which appears in the second gradient derivative of PW91 exchange.

    Args:
        u (array): Argument

    Returns:
        curvature (array): The combination above, accurate on both sides of the cutoff

    """

    # The two terms cancel at small gradient, so a series is used below this cutoff

    cutoff = 0.01

    u_squared = u * u

    u_safe = np.maximum(u, cutoff)
    u_safe_squared = u_safe * u_safe

    direct = (1 / (1 + u_safe_squared) ** (1 / 2) - np.arcsinh(u_safe) / u_safe) / u_safe_squared

    series = -1 / 3 + (3 / 10) * u_squared - (15 / 56) * u_squared * u_squared

    curvature = np.where(u < cutoff, series, direct)

    return curvature










def calculate_mPW91_arcsinh_curvature(x: ndarray) -> ndarray:

    """

    Calculates (1 / (1 + x ^ 2) ^ (3 / 2) - arcsinh(x) / x) / x ^ 2, which appears in the second gradient derivative of mPW91 exchange.

    Args:
        x (array): Argument

    Returns:
        curvature (array): The combination above, accurate on both sides of the cutoff

    """

    # Same cancellation and cutoff as for PW91

    cutoff = 0.01

    x_squared = x * x

    x_safe = np.maximum(x, cutoff)
    x_safe_squared = x_safe * x_safe

    root = (1 + x_safe_squared) ** (1 / 2)

    direct = (1 / (root * root * root) - np.arcsinh(x_safe) / x_safe) / x_safe_squared

    series = -4 / 3 + (9 / 5) * x_squared - (15 / 7) * x_squared * x_squared

    curvature = np.where(x < cutoff, series, direct)

    return curvature










def calculate_B88_exchange_channel_kernel(spin_density: ndarray, spin_sigma: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the second derivatives of the B88 exchange energy per unit volume for a single spin channel.

    Args:
        spin_density (array): Density of one spin channel on integration grid
        spin_sigma (array): Square density gradient of the same spin channel
        calculation (Calculation): Calculation object

    Returns:
        d2g_dp2 (array): Second derivative of g with respect to the spin density
        d2g_dpds (array): Mixed second derivative with respect to the spin density and its square gradient
        d2g_ds2 (array): Second derivative with respect to the square gradient

    """

    # This is the only adjustable parameter for B88 exchange

    beta = 0.0042

    cbrt_spin_density = np.cbrt(spin_density)

    # The Becke reduced gradient for this channel

    x = spin_sigma ** (1 / 2) / cbrt_spin_density ** 4

    x_squared = x * x
    root = (1 + x_squared) ** (1 / 2)
    A = np.arcsinh(x)

    # The B88 denominator and its derivatives with respect to x

    D = 1 + 6 * beta * x * A
    D_squared = D * D
    D_cubed = D_squared * D

    dD_dx = 6 * beta * (A + x / root)
    d2D_dx2 = 6 * beta * (2 + x_squared) / (root * root * root)

    # Derivative of the denominator divided by x, which is needed below

    dD_dx_over_x = 6 * beta * (A / x + 1 / root)

    # The gradient correction is -beta * p ^ (4 / 3) * G

    G = x_squared / D
    dG_dx = 2 * x / D - x_squared * dD_dx / D_squared
    d2G_dx2 = 2 / D - 4 * x * dD_dx / D_squared + 2 * x_squared * dD_dx * dD_dx / D_cubed - x_squared * d2D_dx2 / D_squared

    # This is (d2G_dx2 - dG_dx / x) / x ^ 2, with the 2 / D terms cancelled by hand

    G_tilde = -3 * dD_dx_over_x / D_squared + 2 * dD_dx * dD_dx / D_cubed - d2D_dx2 / D_squared

    inv_cbrt_spin_density_squared = 1 / (cbrt_spin_density * cbrt_spin_density)

    # Local density exchange kernel for this channel, scaled as in the B88 energy

    d2g_dp2_LDA = np.cbrt(2) * calculate_Slater_exchange_kernel(spin_density, None, None, calculation)

    # Second derivatives of g, where the inverse powers of sigma cancel

    d2g_dp2 = d2g_dp2_LDA - (4 * beta / 9) * inv_cbrt_spin_density_squared * (G - x * dG_dx + 4 * x_squared * d2G_dx2)

    d2g_dpds = (2 * beta / 3) * d2G_dx2 / cbrt_spin_density ** 7

    d2g_ds2 = -(beta / 4) * G_tilde / spin_density ** 4

    return d2g_dp2, d2g_dpds, d2g_ds2










def calculate_B88_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B88 exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Sigma is cleaned at the square of the density floor, otherwise this breaks at zero gradient

    sigma = clean(sigma, floor = constants.sigma_floor)

    # The closed-shell energy is f = 2 * g(n / 2, sigma / 4)

    d2g_dp2, d2g_dpds, d2g_ds2 = calculate_B88_exchange_channel_kernel(density / 2, sigma / 4, calculation)

    # Chain rule factors from halving the density and quartering sigma

    d2f_dn2 = d2g_dp2 / 2
    d2f_dnds = d2g_dpds / 4
    d2f_ds2 = d2g_ds2 / 8

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_B3_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B3 exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Calculates the local density exchange kernel

    d2f_dn2_LDA = calculate_Slater_exchange_kernel(density, sigma, tau, calculation)

    # Calculates the Becke 1988 GGA exchange kernel

    d2f_dn2_B88, d2f_dnds_B88, d2f_ds2_B88 = calculate_B88_exchange_kernel(density, sigma, tau, calculation)

    # The factors here are chosen such that when combined with the multiplicative factors for Hartree-Fock exchange proportion, the B3LYP coefficients are used

    d2f_dn2 = 0.9 * d2f_dn2_B88 + 0.1 * d2f_dn2_LDA

    d2f_dnds = 0.9 * d2f_dnds_B88

    d2f_ds2 = 0.9 * d2f_ds2_B88

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_PBE_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted PBE exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Parameters for PBE - mu may be rounded differently

    kappa, mu = 0.804, 0.21952

    if calculation.functional.x_functional == "REVPBE":

        # This is the only difference between "revised" and regular PBE

        kappa = 1.245

    # Square of the reduced density gradient, which is linear in sigma

    ds_squared_ds = 1 / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)

    s_squared = sigma * ds_squared_ds

    denom = 1 / (1 + mu / kappa * s_squared)

    # Exchange enhancement function and its derivatives with respect to s squared

    F_X = 1 + kappa - kappa * denom

    dF_dt = mu * denom * denom

    d2F_dt2 = -2 * mu * mu / kappa * denom * denom * denom

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    f_LDA = density * e_X_LDA

    # Second derivatives with respect to the density and sigma

    d2f_dn2 = (4 / 9) * (f_LDA / (density * density)) * (F_X + 6 * s_squared * dF_dt + 16 * s_squared * s_squared * d2F_dt2)

    d2f_dnds = -(4 / 3) * (f_LDA / density) * (dF_dt + 2 * s_squared * d2F_dt2) * ds_squared_ds

    d2f_ds2 = f_LDA * d2F_dt2 * ds_squared_ds * ds_squared_ds

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_RPBE_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted RPBE exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Parameters for RPBE

    kappa, mu = 0.804, 0.21952

    # Square of the reduced density gradient, which is linear in sigma

    ds_squared_ds = 1 / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)

    s_squared = sigma * ds_squared_ds

    exponent_term = np.exp(-mu * s_squared / kappa)

    # Exchange enhancement function and its derivatives with respect to s squared

    F_X = 1 + kappa * (1 - exponent_term)

    dF_dt = mu * exponent_term

    d2F_dt2 = -(mu * mu / kappa) * exponent_term

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density and sigma

    d2f_dn2, d2f_dnds, d2f_ds2 = calculate_GGA_exchange_kernel_blocks(density, s_squared, ds_squared_ds, F_X, dF_dt, d2F_dt2, density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_PW91_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted Perdew-Wang 1991 exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # These are the adjustable parameters for PW91 exchange

    a, b, c, d, e, f = 0.19645, 7.7956, 0.2743, 0.1508, 100, 0.004

    # Square of the reduced density gradient, which is linear in sigma

    ds_squared_ds = 1 / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)

    s_squared = sigma * ds_squared_ds

    # Scaled reduced gradient inside the arcsinh

    u = b * s_squared ** (1 / 2)

    root = (1 + u * u) ** (1 / 2)

    arcsinh_over_u = calculate_arcsinh_over_argument(u)

    exponential_term = np.exp(-e * s_squared)

    # Arcsinh term shared by the numerator and denominator, and its derivatives

    shared = 1 + (a / b) * u * np.arcsinh(u)

    dshared_dt = (a * b / 2) * (arcsinh_over_u + 1 / root)

    d2shared_dt2 = (a * b * b * b / 4) * (calculate_PW91_arcsinh_curvature(u) - 1 / (root * root * root))

    # Numerator and denominator of the enhancement factor

    N = shared + (c - d * exponential_term) * s_squared
    D = shared + f * s_squared * s_squared

    F_X = N / D

    dN_dt = dshared_dt + c - d * exponential_term + d * e * s_squared * exponential_term
    d2N_dt2 = d2shared_dt2 + d * e * exponential_term * (2 - e * s_squared)

    dD_dt = dshared_dt + 2 * f * s_squared
    d2D_dt2 = d2shared_dt2 + 2 * f

    dF_dt = (dN_dt - F_X * dD_dt) / D
    d2F_dt2 = (d2N_dt2 - 2 * dF_dt * dD_dt - F_X * d2D_dt2) / D

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density and sigma

    d2f_dn2, d2f_dnds, d2f_ds2 = calculate_GGA_exchange_kernel_blocks(density, s_squared, ds_squared_ds, F_X, dF_dt, d2F_dt2, density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_mPW91_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted modified Perdew-Wang 1991 exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Sigma is cleaned at the square of the density floor, otherwise this breaks at zero gradient

    sigma = clean(sigma, floor=constants.sigma_floor)

    # These are the parameters for mPW exchange

    beta = 5 / np.cbrt(36 * np.pi) ** 5
    b, c, d, eps = 0.00426, 1.6455, 3.72, 1e-6

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # This is a constant, as both parts scale with the cube root of the density

    K = e_X_LDA / np.cbrt(density / 2)

    # The factor of two here maintains the spin-scaling relationship for exchange

    dx_squared_ds = np.cbrt(2) ** 2 / np.cbrt(density) ** 8

    x_squared = sigma * dx_squared_ds
    x = x_squared ** (1 / 2)

    root = (1 + x_squared) ** (1 / 2)

    arcsinh_over_x = calculate_arcsinh_over_argument(x)

    G = np.exp(-c * x_squared)

    # The x ^ 3.72 term makes the second sigma derivative diverge weakly at zero gradient

    x_power_d_minus_two = x ** (d - 2)

    # Numerator and denominator of the mPW enhancement

    N = b * x_squared - (b - beta) * x_squared * G - eps * x_power_d_minus_two * x_squared
    D = 1 + 6 * b * x * np.arcsinh(x) - eps * x_power_d_minus_two * x_squared / K

    F = N / D

    # Derivatives with respect to x squared

    dN_dt = b - (b - beta) * G * (1 - c * x_squared) - (eps * d / 2) * x_power_d_minus_two
    dD_dt = 3 * b * (arcsinh_over_x + 1 / root) - (eps * d / (2 * K)) * x_power_d_minus_two

    dF_dt = (dN_dt - F * dD_dt) / D

    # Curvature terms, with the order one parts cancelled by hand

    dN_curvature = 4 * c * (b - beta) * G * x_squared * (2 - c * x_squared) - eps * d * (d - 2) * x_power_d_minus_two
    dD_curvature = 6 * b * calculate_mPW91_arcsinh_curvature(x) * x_squared - eps * d * (d - 2) * x_power_d_minus_two / K

    d2F_dt2 = (dN_curvature - 8 * x_squared * dF_dt * dD_dt - F * dD_curvature) / (4 * x_squared * D)

    # Exchange enhancement function, which is the local part minus the mPW enhancement

    F_X = 1 - F / K
    dF_X_dt = -dF_dt / K
    d2F_X_dt2 = -d2F_dt2 / K

    # Second derivatives with respect to the density and sigma

    d2f_dn2, d2f_dnds, d2f_ds2 = calculate_GGA_exchange_kernel_blocks(density, x_squared, dx_squared_ds, F_X, dF_X_dt, d2F_X_dt2, density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_B97_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97 exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # The parameters can be for Becke's hybrid (first case) or Grimme's dispersion-corrected GGA (second case)

    c_x = [0.8094, 0.5073, 0.7481] if calculation.method.name == "B97" else [1.08662, -0.52127, 3.25429]

    gamma = 0.004

    # The cube root four makes the equations in Becke's paper match the TUNA treatment of exchange spin scaling

    ds_squared_ds = np.cbrt(4) / np.cbrt(density) ** 8

    s_squared = sigma * ds_squared_ds

    # Finite range variable and its derivatives with respect to s squared

    denom = 1 / (1 + gamma * s_squared)

    x = gamma * s_squared * denom

    dx_dt = gamma * denom * denom
    d2x_dt2 = -2 * gamma * gamma * denom * denom * denom

    # Exchange enhancement function, a quadratic in x, and its derivatives

    F_X = c_x[0] + (c_x[1] + c_x[2] * x) * x

    dF_dx = c_x[1] + 2 * c_x[2] * x

    dF_dt = dF_dx * dx_dt

    d2F_dt2 = 2 * c_x[2] * dx_dt * dx_dt + dF_dx * d2x_dt2

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density and sigma

    d2f_dn2, d2f_dnds, d2f_ds2 = calculate_GGA_exchange_kernel_blocks(density, s_squared, ds_squared_ds, F_X, dF_dt, d2F_dt2, density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_exchange_kernel_blocks(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Looks up and evaluates the restricted exchange kernel for whichever exchange functional the calculation selects.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma

    """

    # Picks the exchange kernel depending on the functional

    blocks = exchange_kernels[calculation.functional.x_functional](density, sigma, tau, calculation)

    zeros = np.zeros_like(density)

    # Slater exchange has no gradient blocks, so these are zeros

    if not isinstance(blocks, tuple):

        return blocks, zeros, zeros

    d2f_dn2, d2f_dnds, d2f_ds2 = blocks

    # Any block that vanishes identically is also replaced by zeros

    d2f_dnds = zeros if d2f_dnds is None else d2f_dnds
    d2f_ds2 = zeros if d2f_ds2 is None else d2f_ds2

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_GGA_exchange_spin_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted exchange spin kernel for triplet excitations, for any GGA exchange functional.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_X with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_nm_sigma_nm (array): Second derivative with respect to grad(n) . grad(m)

    """

    # Exchange spin scales exactly, so the triplet kernel is a rescaled singlet kernel

    d2f_dn2, d2f_dnds, d2f_ds2 = calculate_restricted_exchange_kernel_blocks(density, sigma, tau, calculation)

    f_mm = d2f_dn2
    f_m_sigma_nm = 2 * d2f_dnds
    f_sigma_nm_sigma_nm = 4 * d2f_ds2

    return f_mm, f_m_sigma_nm, f_sigma_nm_sigma_nm










def calculate_unrestricted_GGA_exchange_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved exchange kernel for an unrestricted reference, for any GGA exchange functional.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_X_alpha_alpha (array): Second derivative of f = n * e_X with respect to the alpha density
        f_X_beta_beta (array): Second derivative with respect to the beta density
        f_X_alpha_sigma_aa (array): Mixed second derivative with respect to the alpha density and sigma alpha-alpha
        f_X_beta_sigma_bb (array): Mixed second derivative with respect to the beta density and sigma beta-beta
        f_X_sigma_aa_sigma_aa (array): Second derivative with respect to sigma alpha-alpha
        f_X_sigma_bb_sigma_bb (array): Second derivative with respect to sigma beta-beta

    """

    # By spin scaling, each channel is the restricted kernel at twice the spin density

    d2f_dn2_alpha, d2f_dnds_alpha, d2f_ds2_alpha = calculate_restricted_exchange_kernel_blocks(2 * alpha_density, 4 * sigma_aa, tau_alpha, calculation)

    d2f_dn2_beta, d2f_dnds_beta, d2f_ds2_beta = calculate_restricted_exchange_kernel_blocks(2 * beta_density, 4 * sigma_bb, tau_beta, calculation)

    # Chain rule factors from spin scaling - blocks mixing alpha and beta vanish

    f_X_alpha_alpha = 2 * d2f_dn2_alpha
    f_X_beta_beta = 2 * d2f_dn2_beta

    f_X_alpha_sigma_aa = 4 * d2f_dnds_alpha
    f_X_beta_sigma_bb = 4 * d2f_dnds_beta

    f_X_sigma_aa_sigma_aa = 8 * d2f_ds2_alpha
    f_X_sigma_bb_sigma_bb = 8 * d2f_ds2_beta

    return f_X_alpha_alpha, f_X_beta_beta, f_X_alpha_sigma_aa, f_X_beta_sigma_bb, f_X_sigma_aa_sigma_aa, f_X_sigma_bb_sigma_bb





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ C O R R E L A T I O N    K E R N E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #





def calculate_restricted_VWN3_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted VWN-III correlation kernel.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
    
    """

    # Parameters for a restricted (paramagnetic) reference, with first and second Seitz-radius derivatives

    _, de_C_dr, d2e_C_dr2 = calculate_VWN_correlation_derivatives(density, -0.409286, 13.0720, 42.7198, 0.0310907)

    # Seitz radius as a function of density

    r_s, _ = calculate_seitz_radius(density)

    # Second derivative of f = n * e_C with respect to density

    d2f_dn2 = (r_s ** 2 * d2e_C_dr2 - 2 * r_s * de_C_dr) / (9 * density)

    return d2f_dn2










def calculate_restricted_VWN3_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted VWN-III spin correlation kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient (unused for LDA)
        tau (array): Non-interacting kinetic energy density (unused for LDA)
        calculation (Calculation): Calculation object
    
    Returns:
        f_mm (array): Spin correlation kernel evaluated on the grid
    
    """

    # Parameters for a restricted (paramagnetic) and fully polarised (ferromagnetic) reference  

    _, e_C_0, _ = calculate_VWN_potential(density, -0.409286, 13.0720, 42.7198, 0.0310907)
    _, e_C_1, _ = calculate_VWN_potential(density, -0.743294, 20.1231, 101.578, 0.01554535)

    # Second zeta-derivative of the spin-scaling function at zero polarisation

    f_prime_prime_at_zero = 8 / (9 * (np.cbrt(2) ** 4 - 2))

    # The spin stiffness of VWN-III follows from the simple interpolation between the paramagnetic and ferromagnetic limits

    f_mm = f_prime_prime_at_zero * (e_C_1 - e_C_0) / density

    return f_mm










def calculate_unrestricted_VWN3_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved VWN-III correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on grid
        beta_density (array): Beta electron density on grid
        density (array): Total electron density on grid

    Returns:
        f_C_alpha_alpha (array): Alpha-alpha correlation kernel block
        f_C_alpha_beta (array): Alpha-beta correlation kernel block
        f_C_beta_beta (array): Beta-beta correlation kernel block

    """

    zeta = calculate_zeta(alpha_density, beta_density)

    r_s, _ = calculate_seitz_radius(density)

    # Paramagnetic and ferromagnetic VWN fits, with first and second Seitz-radius derivatives

    e_C_0, de0_dr, d2e0_dr2 = calculate_VWN_correlation_derivatives(density, -0.409286, 13.0720, 42.7198, 0.0310907)
    e_C_1, de1_dr, d2e1_dr2 = calculate_VWN_correlation_derivatives(density, -0.743294, 20.1231, 101.578, 0.01554535)

    # Spin-scaling function and its first two derivatives, with the second derivative floored at full polarisation

    denominator = np.cbrt(2) ** 4 - 2

    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)

    one_plus = np.maximum(1 + zeta, constants.density_floor)
    one_minus = np.maximum(1 - zeta, constants.density_floor)

    f_prime_prime_zeta = (4 / 9) * (np.cbrt(one_plus) ** (-2) + np.cbrt(one_minus) ** (-2)) / denominator

    # Derivatives of the energy density per particle with respect to the Seitz radius and spin polarisation, for the simple VWN-III interpolation

    de_dr = de0_dr * (1 - f_zeta) + de1_dr * f_zeta
    d2e_dr2 = d2e0_dr2 * (1 - f_zeta) + d2e1_dr2 * f_zeta
    de_dzeta = (e_C_1 - e_C_0) * f_prime_zeta
    d2e_dzeta2 = (e_C_1 - e_C_0) * f_prime_prime_zeta
    d2e_dr_dzeta = (de1_dr - de0_dr) * f_prime_zeta

    # Second derivatives of f = n e_C in the (density, zeta) variables

    f_nn = (r_s ** 2 * d2e_dr2 - 2 * r_s * de_dr) / (9 * density)
    f_nz = de_dzeta - (r_s / 3) * d2e_dr_dzeta
    f_zz = density * d2e_dzeta2
    f_z = density * de_dzeta

    # Partial derivatives of zeta with respect to the alpha (u) and beta (w) densities

    inv_density_squared = 1 / (density * density)

    zeta_u = (1 - zeta) / density
    zeta_w = - (1 + zeta) / density
    zeta_uu = - 2 * (1 - zeta) * inv_density_squared
    zeta_ww = 2 * (1 + zeta) * inv_density_squared
    zeta_uw = 2 * zeta * inv_density_squared

    # Assembles the spin-resolved correlation kernel via the chain rule

    f_C_alpha_alpha = f_nn + 2 * f_nz * zeta_u + f_zz * zeta_u ** 2 + f_z * zeta_uu
    f_C_beta_beta = f_nn + 2 * f_nz * zeta_w + f_zz * zeta_w ** 2 + f_z * zeta_ww
    f_C_alpha_beta = f_nn + f_nz * (zeta_u + zeta_w) + f_zz * zeta_u * zeta_w + f_z * zeta_uw

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta










def calculate_restricted_VWN5_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted VWN-V correlation kernel.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
    
    """

    x_0 = -0.10498
    b = 3.72744
    c = 12.9352
    A = 0.0310907

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
    X_squared = X * X

    # First derivative term 

    combo = (2 / x + 2 * c_1 / x_minus_x_0 - (2 * x + b) * (1 + c_1) / X - (1 / 2) * c_2 * Q / X)

    # Derivative of the "combo" term with respect to x

    dcombo_dx = (- 2 / r_s - 2 * c_1 / (x_minus_x_0 * x_minus_x_0) - 2 * (1 + c_1) / X + ((2 * x + b) * (2 * x + b)) * (1 + c_1) / X_squared + (1 / 2) * c_2 * Q * (2 * x + b) / X_squared)

    # Second derivative of f = n * e_C with respect to density

    d2f_dn2 = (x * A) / (36 * density) * (x * dcombo_dx - 5 * combo)

    return d2f_dn2










def calculate_restricted_VWN5_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted VWN-V spin correlation kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient (unused for LDA)
        tau (array): Non-interacting kinetic energy density (unused for LDA)
        calculation (Calculation): Calculation object
    
    Returns:
        f_mm (array): Spin correlation kernel evaluated on the grid
    
    """

    # Calculates the spin stiffness

    _, minus_alpha, _ = calculate_VWN_potential(density, -0.0047584, 1.13107, 13.0045, 1 / (6 * np.pi ** 2))

    # The spin stiffness alpha is the negative of minus_alpha
    
    f_mm = - minus_alpha / density

    return f_mm










def calculate_VWN_correlation_derivatives(density: ndarray, x_0: float, b: float, c: float, A: float) -> tuple:

    """

    Calculates a single VWN correlation fit and its first two derivatives with respect to the Seitz radius.

    Args:
        density (array): Electron density on grid
        x_0 (float): Coefficient for VWN correlation
        b (float): Coefficient for VWN correlation
        c (float): Coefficient for VWN correlation
        A (float): Coefficient for VWN correlation

    Returns:
        e_C (array): Energy density per particle
        de_C_dr (array): First derivative of the energy density with respect to the Seitz radius
        d2e_C_dr2 (array): Second derivative of the energy density with respect to the Seitz radius

    """

    # Useful intermediate constants for VWN

    Q = (4 * c - b ** 2) ** (1 / 2)
    X_0 = x_0 ** 2 + b * x_0 + c
    c_1 = -b * x_0 / X_0
    c_2 = 2 * b * (c - x_0 ** 2) / (Q * X_0)

    r_s, _ = calculate_seitz_radius(density)
    x = r_s ** (1 / 2)
    x_minus_x_0 = x - x_0
    X = r_s + b * x + c
    X_squared = X * X
    x_minus_x_0_squared = x_minus_x_0 * x_minus_x_0

    # Energy density per particle

    e_C = A * (np.log(r_s / X) + c_1 * np.log(x_minus_x_0_squared / X) + c_2 * np.arctan(Q / (2 * x + b)))

    # First derivative with respect to x, and its x-derivative

    combo = 2 / x + 2 * c_1 / x_minus_x_0 - (2 * x + b) * (1 + c_1) / X - (1 / 2) * c_2 * Q / X

    dcombo_dx = (- 2 / r_s - 2 * c_1 / x_minus_x_0_squared - 2 * (1 + c_1) / X + ((2 * x + b) ** 2) * (1 + c_1) / X_squared + (1 / 2) * c_2 * Q * (2 * x + b) / X_squared)

    # Converts to Seitz-radius derivatives

    de_C_dr = (A / 2) * combo / x

    d2e_C_dr2 = A * (x * dcombo_dx - combo) / (4 * x ** 3)

    return e_C, de_C_dr, d2e_C_dr2










def calculate_unrestricted_VWN5_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved VWN-V correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on grid
        beta_density (array): Beta electron density on grid
        density (array): Total electron density on grid

    Returns:
        f_C_alpha_alpha (array): Alpha-alpha correlation kernel block
        f_C_alpha_beta (array): Alpha-beta correlation kernel block
        f_C_beta_beta (array): Beta-beta correlation kernel block

    """

    # Paramagnetic, ferromagnetic and spin-stiffness VWN fits, with first and second Seitz-radius derivatives

    e_C_0, de0_dr, d2e0_dr2 = calculate_VWN_correlation_derivatives(density, -0.10498, 3.72744, 12.9352, 0.0310907)
    e_C_1, de1_dr, d2e1_dr2 = calculate_VWN_correlation_derivatives(density, -0.32500, 7.06042, 18.0578, 0.01554535)
    vwn_alpha, dvwn_alpha_dr, d2vwn_alpha_dr2 = calculate_VWN_correlation_derivatives(density, -0.0047584, 1.13107, 13.0045, 1 / (6 * np.pi ** 2))

    # The spin stiffness is the negative of the VWN fit with the RPA parameters

    alpha_C, dalpha_C_dr, d2alpha_C_dr2 = - vwn_alpha, - dvwn_alpha_dr, - d2vwn_alpha_dr2

    # Kernel for an unrestricted reference by interpolation between limits

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta = calculate_VWN5_spin_interpolation_kernel(alpha_density, beta_density, density, alpha_C, dalpha_C_dr, d2alpha_C_dr2, e_C_0, de0_dr, d2e0_dr2, e_C_1, de1_dr, d2e1_dr2)

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta










def calculate_VWN5_spin_interpolation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, alpha_C: ndarray, dalpha_C_dr: ndarray, d2alpha_C_dr2: ndarray, e_C_0: ndarray, de0_dr: ndarray, d2e0_dr2: ndarray, e_C_1: ndarray, de1_dr: ndarray, d2e1_dr2: ndarray) -> tuple:

    """

    Calculates the spin-resolved correlation kernel for the VWN-V spin interpolation between LDA correlation fits.

    Args:
        alpha_density (array): Alpha spin electron density on grid
        beta_density (array): Beta spin electron density on grid
        density (array): Electron density on grid
        alpha_C (array): Spin stiffness
        dalpha_C_dr (array): First derivative of the spin stiffness with respect to the Seitz radius
        d2alpha_C_dr2 (array): Second derivative of the spin stiffness with respect to the Seitz radius
        e_C_0 (array): Energy density for paramagnetic system
        de0_dr (array): First derivative of the paramagnetic energy density with respect to the Seitz radius
        d2e0_dr2 (array): Second derivative of the paramagnetic energy density with respect to the Seitz radius
        e_C_1 (array): Energy density for ferromagnetic system
        de1_dr (array): First derivative of the ferromagnetic energy density with respect to the Seitz radius
        d2e1_dr2 (array): Second derivative of the ferromagnetic energy density with respect to the Seitz radius

    Returns:
        f_C_alpha_alpha (array): Alpha-alpha correlation kernel block
        f_C_alpha_beta (array): Alpha-beta correlation kernel block
        f_C_beta_beta (array): Beta-beta correlation kernel block

    """

    zeta = calculate_zeta(alpha_density, beta_density)

    r_s, _ = calculate_seitz_radius(density)

    # Spin-scaling function and its first two derivatives; f and f' reuse the energy convention, while the second derivative is floored at full polarisation

    denominator = np.cbrt(2) ** 4 - 2
    f_prime_prime_at_zero = 8 / (9 * denominator)

    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)

    one_plus = np.maximum(1 + zeta, constants.density_floor)
    one_minus = np.maximum(1 - zeta, constants.density_floor)

    f_prime_prime_zeta = (4 / 9) * (np.cbrt(one_plus) ** (-2) + np.cbrt(one_minus) ** (-2)) / denominator

    zeta_2 = zeta * zeta
    zeta_3 = zeta_2 * zeta
    zeta_4 = zeta_3 * zeta

    # Coefficient functions of zeta multiplying the spin stiffness (g) and the ferromagnetic difference (h)

    h = f_zeta * zeta_4
    h_prime = f_prime_zeta * zeta_4 + 4 * f_zeta * zeta_3
    h_prime_prime = f_prime_prime_zeta * zeta_4 + 8 * f_prime_zeta * zeta_3 + 12 * f_zeta * zeta_2

    g = f_zeta * (1 - zeta_4) / f_prime_prime_at_zero
    g_prime = (f_prime_zeta * (1 - zeta_4) - 4 * f_zeta * zeta_3) / f_prime_prime_at_zero
    g_prime_prime = (f_prime_prime_zeta * (1 - zeta_4) - 8 * f_prime_zeta * zeta_3 - 12 * f_zeta * zeta_2) / f_prime_prime_at_zero

    # Derivatives of the energy density per particle with respect to the Seitz radius and spin polarisation

    de_dr = de0_dr * (1 - h) + de1_dr * h + dalpha_C_dr * g
    d2e_dr2 = d2e0_dr2 * (1 - h) + d2e1_dr2 * h + d2alpha_C_dr2 * g
    de_dzeta = (e_C_1 - e_C_0) * h_prime + alpha_C * g_prime
    d2e_dzeta2 = (e_C_1 - e_C_0) * h_prime_prime + alpha_C * g_prime_prime
    d2e_dr_dzeta = (de1_dr - de0_dr) * h_prime + dalpha_C_dr * g_prime

    # Second derivatives of f = n e_C in the (density, zeta) variables

    f_nn = (r_s ** 2 * d2e_dr2 - 2 * r_s * de_dr) / (9 * density)
    f_nz = de_dzeta - (r_s / 3) * d2e_dr_dzeta
    f_zz = density * d2e_dzeta2
    f_z = density * de_dzeta

    # Partial derivatives of zeta with respect to the alpha (u) and beta (w) densities

    zeta_u = (1 - zeta) / density
    zeta_w = - (1 + zeta) / density
    zeta_uu = - 2 * (1 - zeta) / density ** 2
    zeta_ww = 2 * (1 + zeta) / density ** 2
    zeta_uw = 2 * zeta / density ** 2

    # Assembles the spin-resolved correlation kernel via the chain rule

    f_C_alpha_alpha = f_nn + 2 * f_nz * zeta_u + f_zz * zeta_u ** 2 + f_z * zeta_uu
    f_C_beta_beta = f_nn + 2 * f_nz * zeta_w + f_zz * zeta_w ** 2 + f_z * zeta_ww
    f_C_alpha_beta = f_nn + f_nz * (zeta_u + zeta_w) + f_zz * zeta_u * zeta_w + f_z * zeta_uw

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta










def calculate_restricted_PW_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted PW92 correlation kernel.

    Args:
        density (array): Electron density on integration grid
    
    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
    
    """

    # Parameters for a restricted (paramagnetic) reference, with first and second Seitz-radius derivatives

    _, de_C_dr, d2e_C_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    # Seitz radius as a function of density

    r_s, _ = calculate_seitz_radius(density)

    # Second derivative of f = n * e_C with respect to density

    d2f_dn2 = (r_s ** 2 * d2e_C_dr2 - 2 * r_s * de_C_dr) / (9 * density)

    return d2f_dn2










def calculate_restricted_PW_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """
    
    Calculates the restricted PW92 spin correlation kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient (unused for LDA)
        tau (array): Non-interacting kinetic energy density (unused for LDA)
        calculation (Calculation): Calculation object
    
    Returns:
        f_mm (array): Spin correlation kernel evaluated on the grid
    
    """

    # Calculates the spin stiffness

    _, minus_alpha, _ = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    # The spin stiffness alpha is the negative of minus_alpha
    
    f_mm = - minus_alpha / density

    return f_mm










def calculate_PW_correlation_derivatives(density: ndarray, A: float, alpha_1: float, beta_1: float, beta_2: float, beta_3: float, beta_4: float, P: float) -> tuple:

    """

    Calculates a single PW92 correlation fit and its first two derivatives with respect to the Seitz radius.

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
        e_C (array): Energy density per particle
        de_C_dr (array): First derivative of the energy density with respect to the Seitz radius
        d2e_C_dr2 (array): Second derivative of the energy density with respect to the Seitz radius

    """

    # Calculates the Seitz radius for the density

    r_s, _ = calculate_seitz_radius(density)

    # Square rooting via ** (1 / 2) is faster than np.sqrt, and the root is reused for all fractional powers

    sqrt_r_s = r_s ** (1 / 2)

    # Intermediate quantities for PW92 LDA correlation, and their first and second Seitz-radius derivatives

    Q_0 = -2 * A * (1 + alpha_1 * r_s)
    Q_1 = 2 * A * (beta_1 * sqrt_r_s + beta_2 * r_s + beta_3 * r_s * sqrt_r_s + beta_4 * r_s ** (P + 1))
    Q_1_prime = A * (beta_1 / sqrt_r_s + 2 * beta_2 + 3 * beta_3 * sqrt_r_s + 2 * (P + 1) * beta_4 * r_s ** P)
    Q_1_prime_prime = A * (-beta_1 / (2 * r_s * sqrt_r_s) + (3 / 2) * beta_3 / sqrt_r_s + 2 * P * (P + 1) * beta_4 * r_s ** (P - 1))

    # Numpy's log_1_plus function is more numerically stable than log(1 + 1/Q_1)

    log_term = np.log1p(1 / Q_1)

    denominator = Q_1 * Q_1 + Q_1

    # Energy density per particle for PW92

    e_C = Q_0 * log_term

    # First and second derivatives of energy density with respect to Seitz radius

    de_C_dr = -2 * A * alpha_1 * log_term - Q_0 * Q_1_prime / denominator

    d2e_C_dr2 = 4 * A * alpha_1 * Q_1_prime / denominator + Q_0 * (Q_1_prime * Q_1_prime * (2 * Q_1 + 1) / denominator - Q_1_prime_prime) / denominator

    return e_C, de_C_dr, d2e_C_dr2










def calculate_unrestricted_PW_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved PW92 correlation kernel for an unrestricted reference.

    This is "modified" PW92 from LibXC and has more significant figures than the original paper.

    Args:
        alpha_density (array): Alpha electron density on grid
        beta_density (array): Beta electron density on grid
        density (array): Total electron density on grid

    Returns:
        f_C_alpha_alpha (array): Alpha-alpha correlation kernel block
        f_C_alpha_beta (array): Alpha-beta correlation kernel block
        f_C_beta_beta (array): Beta-beta correlation kernel block

    """

    # Paramagnetic, ferromagnetic and spin-stiffness PW92 fits, with first and second Seitz-radius derivatives

    e_C_0, de0_dr, d2e0_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    e_C_1, de1_dr, d2e1_dr2 = calculate_PW_correlation_derivatives(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    minus_alpha, dminus_alpha_dr, d2minus_alpha_dr2 = calculate_PW_correlation_derivatives(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    # The spin stiffness is the negative of the PW92 fit with the RPA parameters

    alpha_C, dalpha_C_dr, d2alpha_C_dr2 = - minus_alpha, - dminus_alpha_dr, - d2minus_alpha_dr2

    # Kernel for an unrestricted reference by interpolation between limits

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta = calculate_VWN5_spin_interpolation_kernel(alpha_density, beta_density, density, alpha_C, dalpha_C_dr, d2alpha_C_dr2, e_C_0, de0_dr, d2e0_dr2, e_C_1, de1_dr, d2e1_dr2)

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta










def calculate_PW_spin_correlation_derivatives(alpha_density: ndarray, beta_density: ndarray, density: ndarray) -> tuple:

    """

    Calculates the PW92 correlation energy per particle and its first two derivatives with respect to the density and spin polarisation.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid

    Returns:
        e_C (array): Correlation energy density per particle
        de_C_dn (array): First derivative with respect to density
        de_C_dzeta (array): First derivative with respect to spin polarisation
        d2e_C_dn2 (array): Second derivative with respect to density
        d2e_C_dndzeta (array): Mixed second derivative
        d2e_C_dzeta2 (array): Second derivative with respect to spin polarisation
        zeta (array): Local spin polarisation

    """

    # Paramagnetic, ferromagnetic and spin-stiffness PW92 fits, with first and second Seitz-radius derivatives

    e_C_0, de0_dr, d2e0_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    e_C_1, de1_dr, d2e1_dr2 = calculate_PW_correlation_derivatives(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    minus_alpha, dminus_alpha_dr, d2minus_alpha_dr2 = calculate_PW_correlation_derivatives(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    # The spin stiffness is the negative of the PW92 fit with the RPA parameters

    alpha_C, dalpha_C_dr, d2alpha_C_dr2 = -minus_alpha, -dminus_alpha_dr, -d2minus_alpha_dr2

    zeta = calculate_zeta(alpha_density, beta_density)

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    # Spin-scaling function and its first two derivatives, with the second derivative floored at full polarisation

    denominator = np.cbrt(2) ** 4 - 2
    f_prime_prime_at_zero = 8 / (9 * denominator)

    f_zeta = (cbrt_plus ** 4 + cbrt_minus ** 4 - 2) / denominator
    f_prime_zeta = (4 / 3) * (cbrt_plus - cbrt_minus) / denominator
    f_prime_prime_zeta = (4 / 9) * (1 / (cbrt_plus * cbrt_plus) + 1 / (cbrt_minus * cbrt_minus)) / denominator

    zeta_2 = zeta * zeta
    zeta_3 = zeta_2 * zeta
    zeta_4 = zeta_3 * zeta

    # Spin interpolation functions and their derivatives

    h = f_zeta * zeta_4
    h_prime = f_prime_zeta * zeta_4 + 4 * f_zeta * zeta_3
    h_prime_prime = f_prime_prime_zeta * zeta_4 + 8 * f_prime_zeta * zeta_3 + 12 * f_zeta * zeta_2

    g = f_zeta * (1 - zeta_4) / f_prime_prime_at_zero
    g_prime = (f_prime_zeta * (1 - zeta_4) - 4 * f_zeta * zeta_3) / f_prime_prime_at_zero
    g_prime_prime = (f_prime_prime_zeta * (1 - zeta_4) - 8 * f_prime_zeta * zeta_3 - 12 * f_zeta * zeta_2) / f_prime_prime_at_zero

    # Energy density per particle and its derivatives

    e_C = e_C_0 + alpha_C * g + (e_C_1 - e_C_0) * h

    de_C_dr = de0_dr + dalpha_C_dr * g + (de1_dr - de0_dr) * h
    d2e_C_dr2 = d2e0_dr2 + d2alpha_C_dr2 * g + (d2e1_dr2 - d2e0_dr2) * h

    de_C_dzeta = alpha_C * g_prime + (e_C_1 - e_C_0) * h_prime
    d2e_C_dzeta2 = alpha_C * g_prime_prime + (e_C_1 - e_C_0) * h_prime_prime
    d2e_C_drdzeta = dalpha_C_dr * g_prime + (de1_dr - de0_dr) * h_prime

    # Converts the Seitz-radius derivatives into density ones

    r_s, inv_density = calculate_seitz_radius(density)

    dr_s_dn = -r_s * inv_density / 3
    d2r_s_dn2 = (4 / 9) * r_s * inv_density * inv_density

    de_C_dn = de_C_dr * dr_s_dn
    d2e_C_dn2 = d2e_C_dr2 * dr_s_dn * dr_s_dn + de_C_dr * d2r_s_dn2
    d2e_C_dndzeta = d2e_C_drdzeta * dr_s_dn

    return e_C, de_C_dn, de_C_dzeta, d2e_C_dn2, d2e_C_dndzeta, d2e_C_dzeta2, zeta










def calculate_spin_polarisation_transformation(zeta: ndarray, density: ndarray, df_dzeta: ndarray, d2f_dn2: ndarray, d2f_dndzeta: ndarray, d2f_dzeta2: ndarray, d2f_dnds: ndarray, d2f_dzetads: ndarray, d2f_ds2: ndarray) -> tuple:

    """

    Transforms second derivatives from the density and spin polarisation variables onto the two spin densities.

    Args:
        zeta (array): Local spin polarisation
        density (array): Electron density on integration grid
        df_dzeta (array): First derivative of f with respect to spin polarisation
        d2f_dn2 (array): Second derivative of f with respect to density
        d2f_dndzeta (array): Mixed second derivative of f with respect to density and spin polarisation
        d2f_dzeta2 (array): Second derivative of f with respect to spin polarisation
        d2f_dnds (array): Mixed second derivative of f with respect to density and sigma
        d2f_dzetads (array): Mixed second derivative of f with respect to spin polarisation and sigma
        d2f_ds2 (array): Second derivative of f with respect to sigma

    Returns:
        f_C_alpha_alpha (array): Second derivative of f with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma (array): Mixed second derivative with respect to the alpha density and sigma
        f_C_beta_sigma (array): Mixed second derivative with respect to the beta density and sigma
        f_C_sigma_sigma (array): Second derivative with respect to sigma

    """

    # Partial derivatives of zeta with respect to the alpha (u) and beta (w) densities

    inv_density = 1 / density

    zeta_u = (1 - zeta) * inv_density
    zeta_w = -(1 + zeta) * inv_density
    zeta_uu = -2 * (1 - zeta) * inv_density * inv_density
    zeta_ww = 2 * (1 + zeta) * inv_density * inv_density
    zeta_uw = 2 * zeta * inv_density * inv_density

    # Chain rule onto the two spin densities

    f_C_alpha_alpha = d2f_dn2 + 2 * d2f_dndzeta * zeta_u + d2f_dzeta2 * zeta_u * zeta_u + df_dzeta * zeta_uu
    f_C_alpha_beta = d2f_dn2 + d2f_dndzeta * (zeta_u + zeta_w) + d2f_dzeta2 * zeta_u * zeta_w + df_dzeta * zeta_uw
    f_C_beta_beta = d2f_dn2 + 2 * d2f_dndzeta * zeta_w + d2f_dzeta2 * zeta_w * zeta_w + df_dzeta * zeta_ww

    f_C_alpha_sigma = d2f_dnds + d2f_dzetads * zeta_u
    f_C_beta_sigma = d2f_dnds + d2f_dzetads * zeta_w
    f_C_sigma_sigma = d2f_ds2

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma










def calculate_correlation_gradient_coefficient(density: ndarray, offset: float) -> tuple:

    """

    Calculates the C(r_s) rational function shared by PW91 and P86 correlation, with its first two density derivatives.

    Args:
        density (array): Electron density on integration grid
        offset (float): The additive constant, which differs between the two functionals

    Returns:
        C (array): The coefficient
        dC_dn (array): First derivative with respect to density
        d2C_dn2 (array): Second derivative with respect to density

    """

    r_s, inv_density = calculate_seitz_radius(density)

    # Numerator and denominator, and their Seitz-radius derivatives

    N = 0.002568 + 0.023266 * r_s + 7.389e-6 * r_s * r_s
    D = 1 + 8.723 * r_s + 0.472 * r_s * r_s + 7.389e-2 * r_s * r_s * r_s

    dN_dr = 0.023266 + 2 * 7.389e-6 * r_s
    d2N_dr2 = 2 * 7.389e-6

    dD_dr = 8.723 + 2 * 0.472 * r_s + 3 * 7.389e-2 * r_s * r_s
    d2D_dr2 = 2 * 0.472 + 6 * 7.389e-2 * r_s

    dC_dr = dN_dr / D - N * dD_dr / (D * D)
    d2C_dr2 = d2N_dr2 / D - 2 * dN_dr * dD_dr / (D * D) - N * d2D_dr2 / (D * D) + 2 * N * dD_dr * dD_dr / D ** 3

    # Converts the Seitz-radius derivatives into density ones

    dr_s_dn = -r_s * inv_density / 3
    d2r_s_dn2 = (4 / 9) * r_s * inv_density * inv_density

    C = offset + N / D
    dC_dn = dC_dr * dr_s_dn
    d2C_dn2 = d2C_dr2 * dr_s_dn * dr_s_dn + dC_dr * d2r_s_dn2

    return C, dC_dn, d2C_dn2










def calculate_LYP_same_spin_gradient_derivatives(density_x: ndarray, density_y: ndarray, density: ndarray, delta: ndarray, delta_prime: ndarray, delta_prime_prime: ndarray) -> tuple:

    """

    Calculates the factor multiplying a same-spin square gradient in LYP, apart from the -a * b * w prefactor, with its derivatives with respect to the spin densities.

    Args:
        density_x (array): Spin density appearing in the same-spin gradient
        density_y (array): Opposite spin density
        density (array): Total electron density on integration grid
        delta (array): Delta function from the LYP paper
        delta_prime (array): First derivative of delta with respect to the total density
        delta_prime_prime (array): Second derivative of delta with respect to the total density

    Returns:
        B (array): The coefficient itself
        dB_dx (array): First derivative with respect to density_x
        dB_dy (array): First derivative with respect to density_y
        d2B_dx2 (array): Second derivative with respect to density_x
        d2B_dxdy (array): Mixed second derivative
        d2B_dy2 (array): Second derivative with respect to density_y

    """

    # The factor is B = (1 / 9) * n_x * n_y * (1 - 3 * delta - (delta - 11) * n_x / n) - n_y ^ 2

    inv_density = 1 / density

    # This is often used

    G = delta - 11

    # First piece, with no spin ratio

    T1 = (1 / 9) * density_x * density_y * (1 - 3 * delta)

    dT1_dx = (1 / 9) * (density_y * (1 - 3 * delta) - 3 * density_x * density_y * delta_prime)
    dT1_dy = (1 / 9) * (density_x * (1 - 3 * delta) - 3 * density_x * density_y * delta_prime)

    d2T1_dx2 = (1 / 9) * (-6 * density_y * delta_prime - 3 * density_x * density_y * delta_prime_prime)
    d2T1_dy2 = (1 / 9) * (-6 * density_x * delta_prime - 3 * density_x * density_y * delta_prime_prime)
    d2T1_dxdy = (1 / 9) * ((1 - 3 * delta) - 3 * density * delta_prime - 3 * density_x * density_y * delta_prime_prime)

    # Second piece, which is -(1 / 9) * G * S

    S = density_x * density_x * density_y * inv_density

    dS_dx = density_x * density_y * (density_x + 2 * density_y) * inv_density * inv_density
    dS_dy = density_x * density_x * density_x * inv_density * inv_density

    d2S_dx2 = 2 * density_y * inv_density - 2 * density_x * density_y * (density_x + 2 * density_y) * inv_density * inv_density * inv_density
    d2S_dy2 = -2 * density_x * density_x * density_x * inv_density * inv_density * inv_density
    d2S_dxdy = density_x * (density_x + 4 * density_y) * inv_density * inv_density - 2 * density_x * density_y * (density_x + 2 * density_y) * inv_density * inv_density * inv_density

    T2 = -(1 / 9) * G * S

    dT2_dx = -(1 / 9) * (delta_prime * S + G * dS_dx)
    dT2_dy = -(1 / 9) * (delta_prime * S + G * dS_dy)

    d2T2_dx2 = -(1 / 9) * (delta_prime_prime * S + 2 * delta_prime * dS_dx + G * d2S_dx2)
    d2T2_dy2 = -(1 / 9) * (delta_prime_prime * S + 2 * delta_prime * dS_dy + G * d2S_dy2)
    d2T2_dxdy = -(1 / 9) * (delta_prime_prime * S + delta_prime * (dS_dx + dS_dy) + G * d2S_dxdy)

    # The -n_y ^ 2 term only contributes to the opposite spin derivatives

    B = T1 + T2 - density_y * density_y

    dB_dx = dT1_dx + dT2_dx
    dB_dy = dT1_dy + dT2_dy - 2 * density_y

    d2B_dx2 = d2T1_dx2 + d2T2_dx2
    d2B_dy2 = d2T1_dy2 + d2T2_dy2 - 2
    d2B_dxdy = d2T1_dxdy + d2T2_dxdy

    return B, dB_dx, dB_dy, d2B_dx2, d2B_dxdy, d2B_dy2










def calculate_restricted_LYP_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted LYP correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma, identically zero for LYP

    """

    # These constants define LYP correlation

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    # Repeatedly useful quantities

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = 1 + d * inv_cbrt_density

    # Bounded ratio, which keeps the high inverse powers of density finite

    Y = d * inv_cbrt_density / X

    # Factor shared by every non-local term, as w = n ^ (-11 / 3) * P

    P = np.exp(-c * inv_cbrt_density) / X

    # Definition of delta directly from LYP paper, and its derivatives

    delta = c * inv_cbrt_density + Y

    delta_prime = (inv_density / 3) * (Y * Y - delta)

    delta_prime_prime = (inv_density * inv_density / 3) * (delta - (5 / 3) * Y * Y + (2 / 3) * Y * Y * Y) - delta_prime * inv_density / 3

    # Second derivative of the local -a * n / X term

    d2f_dn2_local = -(2 * a / 9) * inv_density * (Y + Y * Y) / X

    # Second derivative of the Thomas-Fermi term, -a * b * C_F * n * P

    C_F = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2

    d2f_dn2_TF = -a * b * C_F * P * (delta * (1 + delta / 3) * inv_density / 3 + delta_prime / 3)

    # The gradient part of f is (a * b / 72) * sigma * G, with G = (7 * delta + 3) * Q

    Q = inv_cbrt_density ** 5 * P

    A = 7 * delta_prime + (7 * delta + 3) * (delta - 5) * inv_density / 3

    A_prime = 7 * delta_prime_prime + delta_prime * (14 * delta - 32) * inv_density / 3 - (7 * delta + 3) * (delta - 5) * inv_density * inv_density / 3

    # First and second density derivatives of G

    G_prime = Q * A

    G_prime_prime = Q * (A * (delta - 5) * inv_density / 3 + A_prime)

    # Second derivatives with respect to the density and sigma

    d2f_dn2 = d2f_dn2_local + d2f_dn2_TF + (a * b / 72) * sigma * G_prime_prime

    d2f_dnds = (a * b / 72) * G_prime

    # The energy is linear in sigma, so the second sigma derivative vanishes

    return d2f_dn2, d2f_dnds, None










def calculate_restricted_LYP_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted LYP spin correlation kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_mm (array): Derivative with respect to the square spin density gradient

    """

    # These constants define LYP correlation

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    # Repeatedly useful quantities

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = 1 + d * inv_cbrt_density

    # Straight from LYP paper, with the small exponential factor multiplied in first

    P = np.exp(-c * inv_cbrt_density) / X

    Q = inv_cbrt_density ** 5 * P

    delta = c * inv_cbrt_density + d * inv_cbrt_density / X

    C_F = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2

    # Thomas-Fermi, local and gradient contributions to the spin density block

    f_mm = -(22 / 9) * a * b * C_F * P * inv_density + 2 * a * inv_density / X - a * b * Q * inv_density * inv_density * sigma * (7 * delta - 39) / 36

    # Coupling of the spin density to the gradient of the total density

    f_m_sigma_nm = -a * b * Q * inv_density * (47 - delta) / 72

    # Coefficient of the square spin density gradient

    f_sigma_mm = a * b * Q / 36

    return f_mm, f_m_sigma_nm, f_sigma_mm










def calculate_unrestricted_LYP_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved LYP correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_C_alpha_alpha (array): Second derivative of f = n * e_C with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma_aa (array): Mixed second derivative with respect to the alpha density and sigma alpha-alpha
        f_C_alpha_sigma_bb (array): Mixed second derivative with respect to the alpha density and sigma beta-beta
        f_C_alpha_sigma_ab (array): Mixed second derivative with respect to the alpha density and sigma alpha-beta
        f_C_beta_sigma_aa (array): Mixed second derivative with respect to the beta density and sigma alpha-alpha
        f_C_beta_sigma_bb (array): Mixed second derivative with respect to the beta density and sigma beta-beta
        f_C_beta_sigma_ab (array): Mixed second derivative with respect to the beta density and sigma alpha-beta

    """

    # These constants define LYP correlation

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    C_F = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2
    C = np.cbrt(2) ** 11 * C_F

    # Repeatedly useful quantities

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = 1 + d * inv_cbrt_density

    # Bounded ratio, which keeps the high inverse powers of density finite

    Y = d * inv_cbrt_density / X

    P = np.exp(-c * inv_cbrt_density) / X

    # More stable to form Q first than w directly

    Q = inv_cbrt_density ** 5 * P

    w = Q * inv_density * inv_density

    minus_a_b_w = -a * b * w

    delta = c * inv_cbrt_density + Y

    delta_prime = (inv_density / 3) * (Y * Y - delta)

    delta_prime_prime = (inv_density * inv_density / 3) * (delta - (5 / 3) * Y * Y + (2 / 3) * Y * Y * Y) - delta_prime * inv_density / 3

    # Derivatives of the prefactor with respect to the total density

    log_derivative = (delta - 11) * inv_density / 3

    log_derivative_prime = delta_prime * inv_density / 3 - (delta - 11) * inv_density * inv_density / 3

    minus_a_b_w_prime = minus_a_b_w * log_derivative
    minus_a_b_w_prime_prime = minus_a_b_w * (log_derivative * log_derivative + log_derivative_prime)

    # Thomas-Fermi term, which is C * minus_a_b_w * R

    cbrt_alpha_density = np.cbrt(alpha_density)
    cbrt_beta_density = np.cbrt(beta_density)

    densities_power_sum = cbrt_alpha_density ** 8 + cbrt_beta_density ** 8

    R = cbrt_alpha_density ** 11 * beta_density + alpha_density * cbrt_beta_density ** 11

    dR_da = (11 / 3) * cbrt_alpha_density ** 8 * beta_density + cbrt_beta_density ** 11
    dR_db = cbrt_alpha_density ** 11 + (11 / 3) * alpha_density * cbrt_beta_density ** 8

    d2R_da2 = (88 / 9) * cbrt_alpha_density ** 5 * beta_density
    d2R_db2 = (88 / 9) * alpha_density * cbrt_beta_density ** 5
    d2R_dadb = (11 / 3) * densities_power_sum

    d2_TF_da2 = C * (minus_a_b_w_prime_prime * R + 2 * minus_a_b_w_prime * dR_da + minus_a_b_w * d2R_da2)
    d2_TF_db2 = C * (minus_a_b_w_prime_prime * R + 2 * minus_a_b_w_prime * dR_db + minus_a_b_w * d2R_db2)
    d2_TF_dadb = C * (minus_a_b_w_prime_prime * R + minus_a_b_w_prime * (dR_da + dR_db) + minus_a_b_w * d2R_dadb)

    # Local term, which is -4 * a * n_alpha * n_beta * H

    H = inv_density / X

    N = X - d * inv_cbrt_density / 3
    N_prime = -2 * d * inv_cbrt_density * inv_density / 9

    H_prime = -N * H * H
    H_prime_prime = -N_prime * H * H + 2 * N * N * H * H * H

    d2_local_da2 = -4 * a * (2 * beta_density * H_prime + alpha_density * beta_density * H_prime_prime)
    d2_local_db2 = -4 * a * (2 * alpha_density * H_prime + alpha_density * beta_density * H_prime_prime)
    d2_local_dadb = -4 * a * (H + density * H_prime + alpha_density * beta_density * H_prime_prime)

    # Coefficients of the three square gradients and their derivatives

    B_aa, dB_aa_da, dB_aa_db, d2B_aa_da2, d2B_aa_dadb, d2B_aa_db2 = calculate_LYP_same_spin_gradient_derivatives(alpha_density, beta_density, density, delta, delta_prime, delta_prime_prime)

    # The beta-beta coefficient has the spin densities swapped

    B_bb, dB_bb_db, dB_bb_da, d2B_bb_db2, d2B_bb_dadb, d2B_bb_da2 = calculate_LYP_same_spin_gradient_derivatives(beta_density, alpha_density, density, delta, delta_prime, delta_prime_prime)

    # The opposite spin coefficient is symmetric in the two spin densities

    B_ab = (1 / 9) * alpha_density * beta_density * (47 - 7 * delta) - (4 / 3) * density * density

    dB_ab_da = (1 / 9) * beta_density * (47 - 7 * delta) - (7 / 9) * alpha_density * beta_density * delta_prime - (8 / 3) * density
    dB_ab_db = (1 / 9) * alpha_density * (47 - 7 * delta) - (7 / 9) * alpha_density * beta_density * delta_prime - (8 / 3) * density

    d2B_ab_da2 = -(14 / 9) * beta_density * delta_prime - (7 / 9) * alpha_density * beta_density * delta_prime_prime - 8 / 3
    d2B_ab_db2 = -(14 / 9) * alpha_density * delta_prime - (7 / 9) * alpha_density * beta_density * delta_prime_prime - 8 / 3
    d2B_ab_dadb = (1 / 9) * (47 - 7 * delta) - (7 / 9) * density * delta_prime - (7 / 9) * alpha_density * beta_density * delta_prime_prime - 8 / 3

    # Each gradient coefficient of the energy is minus_a_b_w * B

    d2F_aa_da2 = minus_a_b_w_prime_prime * B_aa + 2 * minus_a_b_w_prime * dB_aa_da + minus_a_b_w * d2B_aa_da2
    d2F_aa_db2 = minus_a_b_w_prime_prime * B_aa + 2 * minus_a_b_w_prime * dB_aa_db + minus_a_b_w * d2B_aa_db2
    d2F_aa_dadb = minus_a_b_w_prime_prime * B_aa + minus_a_b_w_prime * (dB_aa_da + dB_aa_db) + minus_a_b_w * d2B_aa_dadb

    d2F_bb_da2 = minus_a_b_w_prime_prime * B_bb + 2 * minus_a_b_w_prime * dB_bb_da + minus_a_b_w * d2B_bb_da2
    d2F_bb_db2 = minus_a_b_w_prime_prime * B_bb + 2 * minus_a_b_w_prime * dB_bb_db + minus_a_b_w * d2B_bb_db2
    d2F_bb_dadb = minus_a_b_w_prime_prime * B_bb + minus_a_b_w_prime * (dB_bb_da + dB_bb_db) + minus_a_b_w * d2B_bb_dadb

    d2F_ab_da2 = minus_a_b_w_prime_prime * B_ab + 2 * minus_a_b_w_prime * dB_ab_da + minus_a_b_w * d2B_ab_da2
    d2F_ab_db2 = minus_a_b_w_prime_prime * B_ab + 2 * minus_a_b_w_prime * dB_ab_db + minus_a_b_w * d2B_ab_db2
    d2F_ab_dadb = minus_a_b_w_prime_prime * B_ab + minus_a_b_w_prime * (dB_ab_da + dB_ab_db) + minus_a_b_w * d2B_ab_dadb

    # Density blocks from the Thomas-Fermi, local and gradient terms

    f_C_alpha_alpha = d2_TF_da2 + d2_local_da2 + sigma_aa * d2F_aa_da2 + sigma_bb * d2F_bb_da2 + sigma_ab * d2F_ab_da2
    f_C_alpha_beta = d2_TF_dadb + d2_local_dadb + sigma_aa * d2F_aa_dadb + sigma_bb * d2F_bb_dadb + sigma_ab * d2F_ab_dadb
    f_C_beta_beta = d2_TF_db2 + d2_local_db2 + sigma_aa * d2F_aa_db2 + sigma_bb * d2F_bb_db2 + sigma_ab * d2F_ab_db2

    # Mixed density and square gradient blocks

    f_C_alpha_sigma_aa = minus_a_b_w_prime * B_aa + minus_a_b_w * dB_aa_da
    f_C_beta_sigma_aa = minus_a_b_w_prime * B_aa + minus_a_b_w * dB_aa_db

    f_C_alpha_sigma_bb = minus_a_b_w_prime * B_bb + minus_a_b_w * dB_bb_da
    f_C_beta_sigma_bb = minus_a_b_w_prime * B_bb + minus_a_b_w * dB_bb_db

    f_C_alpha_sigma_ab = minus_a_b_w_prime * B_ab + minus_a_b_w * dB_ab_da
    f_C_beta_sigma_ab = minus_a_b_w_prime * B_ab + minus_a_b_w * dB_ab_db

    # LYP is linear in the square gradients, so their second derivatives vanish

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma_aa, f_C_alpha_sigma_bb, f_C_alpha_sigma_ab, f_C_beta_sigma_aa, f_C_beta_sigma_bb, f_C_beta_sigma_ab










def calculate_restricted_PBE_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted PBE correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma

    """

    # Key parameters for defining PBE - this value of beta is not exact, but chosen to match the ORCA implementation

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.066725

    B = beta / gamma

    # Local density correlation and its Seitz-radius derivatives

    e_C_LDA, de_C_dr, d2e_C_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    r_s, inv_density = calculate_seitz_radius(density)

    dr_s_dn = -r_s * inv_density / 3
    d2r_s_dn2 = (4 / 9) * r_s * inv_density * inv_density

    de_C_dn = de_C_dr * dr_s_dn
    d2e_C_dn2 = d2e_C_dr2 * dr_s_dn * dr_s_dn + de_C_dr * d2r_s_dn2

    # Definition of A and its derivatives

    exp_factor = np.exp(-e_C_LDA / gamma)

    # Expm1 here is more stable than exp - 1

    A = B / np.expm1(-e_C_LDA / gamma)

    dA_de = A * A * exp_factor / beta
    d2A_de2 = A * A * exp_factor * (2 * A * exp_factor / beta - 1 / gamma) / beta

    dA_dn = dA_de * de_C_dn
    d2A_dn2 = dA_de * d2e_C_dn2 + d2A_de2 * de_C_dn * de_C_dn

    # Form of reduced density gradient used in PBE correlation

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * k_F * density * density)

    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dnds = -(7 / 3) * dT_ds * inv_density

    # Numerator and denominator of the logged term, and their derivatives

    N = B * T * (1 + A * T)
    D = 1 + A * T + A * A * T * T

    X = N / D

    dN_dT = B * (1 + 2 * A * T)
    dN_dA = B * T * T
    d2N_dT2 = 2 * A * B
    d2N_dTdA = 2 * B * T

    dD_dT = A + 2 * A * A * T
    dD_dA = T + 2 * A * T * T
    d2D_dT2 = 2 * A * A
    d2D_dTdA = 1 + 4 * A * T
    d2D_dA2 = 2 * T * T

    # Derivatives of X, from differentiating N = X * D

    dX_dT = (dN_dT - X * dD_dT) / D
    dX_dA = (dN_dA - X * dD_dA) / D

    d2X_dT2 = (d2N_dT2 - 2 * dX_dT * dD_dT - X * d2D_dT2) / D
    d2X_dTdA = (d2N_dTdA - dX_dT * dD_dA - dX_dA * dD_dT - X * d2D_dTdA) / D
    d2X_dA2 = (-2 * dX_dA * dD_dA - X * d2D_dA2) / D

    # The GGA correction to the LDA correlation energy density

    one_plus_X = 1 + X

    dH_dX = gamma / one_plus_X
    d2H_dX2 = -gamma / (one_plus_X * one_plus_X)

    dH_dT = dH_dX * dX_dT
    dH_dA = dH_dX * dX_dA

    d2H_dT2 = d2H_dX2 * dX_dT * dX_dT + dH_dX * d2X_dT2
    d2H_dTdA = d2H_dX2 * dX_dT * dX_dA + dH_dX * d2X_dTdA
    d2H_dA2 = d2H_dX2 * dX_dA * dX_dA + dH_dX * d2X_dA2

    # Derivatives of H with respect to the density and sigma

    dH_dn = dH_dT * dT_dn + dH_dA * dA_dn
    dH_ds = dH_dT * dT_ds

    d2H_dn2 = d2H_dT2 * dT_dn * dT_dn + 2 * d2H_dTdA * dT_dn * dA_dn + d2H_dA2 * dA_dn * dA_dn + dH_dT * d2T_dn2 + dH_dA * d2A_dn2
    d2H_dnds = (d2H_dT2 * dT_dn + d2H_dTdA * dA_dn) * dT_ds + dH_dT * d2T_dnds
    d2H_ds2 = d2H_dT2 * dT_ds * dT_ds

    # Second derivatives with respect to the density and sigma

    d2f_dn2 = 2 * (de_C_dn + dH_dn) + density * (d2e_C_dn2 + d2H_dn2)

    d2f_dnds = dH_ds + density * d2H_dnds

    d2f_ds2 = density * d2H_ds2

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_PBE_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """

    Calculates the restricted PBE spin correlation kernel for triplet excitations.

    PBE correlation only sees the total square gradient, so the triplet kernel is a single density block.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density

    """

    # Key parameters for defining PBE - this value of beta is not exact, but chosen to match the ORCA implementation

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.066725

    B = beta / gamma

    # Local density correlation and the PW92 spin stiffness

    _, e_C_LDA, _ = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    _, minus_alpha, _ = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    alpha_C = -minus_alpha

    # The same T, A and X as the singlet kernel

    exp_factor = np.exp(-e_C_LDA / gamma)

    # Expm1 here is more stable than exp - 1

    A = B / np.expm1(-e_C_LDA / gamma)

    dA_de = A * A * exp_factor / beta

    k_F = calculate_Fermi_wavevector(density=density)

    T = sigma * np.pi / (16 * k_F * density * density)

    N = B * T * (1 + A * T)
    D = 1 + A * T + A * A * T * T

    X = N / D

    dX_dT = (B * (1 + 2 * A * T) - X * (A + 2 * A * A * T)) / D
    dX_dA = (B * T * T - X * (T + 2 * A * T * T)) / D

    # Spin polarisation curvatures, using phi(0) = 1 and phi''(0) = -2 / 9

    d2T_dzeta2 = (4 / 9) * T

    d2W_dzeta2 = alpha_C + (2 / 3) * e_C_LDA

    d2H_dzeta2 = gamma * (-(2 / 3) * np.log1p(X) + (dX_dT * d2T_dzeta2 + dX_dA * dA_de * d2W_dzeta2) / (1 + X))

    # Converts the spin polarisation curvature into the spin density kernel

    f_mm = (alpha_C + d2H_dzeta2) / density

    return f_mm










def calculate_unrestricted_PBE_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved PBE correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_C_alpha_alpha (array): Second derivative of f = n * e_C with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma (array): Mixed second derivative with respect to the alpha density and the total square gradient
        f_C_beta_sigma (array): Mixed second derivative with respect to the beta density and the total square gradient
        f_C_sigma_sigma (array): Second derivative with respect to the total square gradient

    """

    # Key parameters for PBE - gamma is exact and beta is given to match the ORCA implementation

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.066725

    B = beta / gamma

    inv_density = 1 / density

    # The PBE correlation functional only depends on the full sigma, not the spin channels. This is cleaned at the square of the density floor

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)

    # Local density correlation energy density and derivatives

    e_C, de_C_dn, de_C_dzeta, d2e_C_dn2, d2e_C_dndzeta, d2e_C_dzeta2, zeta = calculate_PW_spin_correlation_derivatives(alpha_density, beta_density, density)

    # Spin polarisation dependent quantities

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus) / 2
    phi_prime = (1 / cbrt_plus - 1 / cbrt_minus) / 3
    phi_prime_prime = -(1 / 9) * (1 / cbrt_plus ** 4 + 1 / cbrt_minus ** 4)

    phi_cubed = phi * phi * phi
    dphi_cubed = 3 * phi * phi * phi_prime
    d2phi_cubed = 6 * phi * phi_prime * phi_prime + 3 * phi * phi * phi_prime_prime

    # W is the argument of the exponential inside A

    gamma_phi_cubed = gamma * phi_cubed

    W = e_C / gamma_phi_cubed
    dW_dn = de_C_dn / gamma_phi_cubed
    dW_dzeta = (de_C_dzeta / phi_cubed - e_C * dphi_cubed / (phi_cubed * phi_cubed)) / gamma
    d2W_dn2 = d2e_C_dn2 / gamma_phi_cubed
    d2W_dndzeta = (d2e_C_dndzeta / phi_cubed - de_C_dn * dphi_cubed / (phi_cubed * phi_cubed)) / gamma
    d2W_dzeta2 = (d2e_C_dzeta2 / phi_cubed - 2 * de_C_dzeta * dphi_cubed / (phi_cubed * phi_cubed) - e_C * d2phi_cubed / (phi_cubed * phi_cubed) + 2 * e_C * dphi_cubed * dphi_cubed / (phi_cubed * phi_cubed * phi_cubed)) / gamma

    # Expm1 here is more stable than exp - 1

    exp_factor = np.exp(-W)

    A = B / np.expm1(-W)

    dA_dW = A * A * exp_factor / B
    d2A_dW2 = A * A * exp_factor * (2 * A * exp_factor / B - 1) / B

    dA_dn = dA_dW * dW_dn
    dA_dzeta = dA_dW * dW_dzeta
    d2A_dn2 = dA_dW * d2W_dn2 + d2A_dW2 * dW_dn * dW_dn
    d2A_dndzeta = dA_dW * d2W_dndzeta + d2A_dW2 * dW_dn * dW_dzeta
    d2A_dzeta2 = dA_dW * d2W_dzeta2 + d2A_dW2 * dW_dzeta * dW_dzeta

    # Key reduced density gradient for PBE

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * phi * phi * k_F * density * density)

    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    dT_dzeta = -2 * T * phi_prime / phi
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dnds = -(7 / 3) * dT_ds * inv_density
    d2T_dndzeta = -2 * dT_dn * phi_prime / phi
    d2T_dzetads = -2 * dT_ds * phi_prime / phi
    d2T_dzeta2 = T * (6 * phi_prime * phi_prime / (phi * phi) - 2 * phi_prime_prime / phi)

    # Numerator and denominator of the logged term, and the derivatives of X

    N = B * T * (1 + A * T)
    D = 1 + A * T + A * A * T * T

    X = N / D

    dN_dT = B * (1 + 2 * A * T)
    dN_dA = B * T * T
    d2N_dT2 = 2 * A * B
    d2N_dTdA = 2 * B * T

    dD_dT = A + 2 * A * A * T
    dD_dA = T + 2 * A * T * T
    d2D_dT2 = 2 * A * A
    d2D_dTdA = 1 + 4 * A * T
    d2D_dA2 = 2 * T * T

    dX_dT = (dN_dT - X * dD_dT) / D
    dX_dA = (dN_dA - X * dD_dA) / D

    d2X_dT2 = (d2N_dT2 - 2 * dX_dT * dD_dT - X * d2D_dT2) / D
    d2X_dTdA = (d2N_dTdA - dX_dT * dD_dA - dX_dA * dD_dT - X * d2D_dTdA) / D
    d2X_dA2 = (-2 * dX_dA * dD_dA - X * d2D_dA2) / D

    # Chain rule derivatives

    dX_dn = dX_dT * dT_dn + dX_dA * dA_dn
    dX_dzeta = dX_dT * dT_dzeta + dX_dA * dA_dzeta
    dX_ds = dX_dT * dT_ds

    d2X_dn2 = d2X_dT2 * dT_dn * dT_dn + 2 * d2X_dTdA * dT_dn * dA_dn + d2X_dA2 * dA_dn * dA_dn + dX_dT * d2T_dn2 + dX_dA * d2A_dn2
    d2X_dndzeta = d2X_dT2 * dT_dn * dT_dzeta + d2X_dTdA * (dT_dn * dA_dzeta + dT_dzeta * dA_dn) + d2X_dA2 * dA_dn * dA_dzeta + dX_dT * d2T_dndzeta + dX_dA * d2A_dndzeta
    d2X_dzeta2 = d2X_dT2 * dT_dzeta * dT_dzeta + 2 * d2X_dTdA * dT_dzeta * dA_dzeta + d2X_dA2 * dA_dzeta * dA_dzeta + dX_dT * d2T_dzeta2 + dX_dA * d2A_dzeta2
    d2X_dnds = d2X_dT2 * dT_dn * dT_ds + d2X_dTdA * dT_ds * dA_dn + dX_dT * d2T_dnds
    d2X_dzetads = d2X_dT2 * dT_dzeta * dT_ds + d2X_dTdA * dT_ds * dA_dzeta + dX_dT * d2T_dzetads
    d2X_ds2 = d2X_dT2 * dT_ds * dT_ds

    # Derivatives of GGA correction to LDA correlation

    one_plus_X = 1 + X
    one_plus_X_squared = one_plus_X * one_plus_X

    L = np.log1p(X)

    dL_dn = dX_dn / one_plus_X
    dL_dzeta = dX_dzeta / one_plus_X
    dL_ds = dX_ds / one_plus_X

    d2L_dn2 = d2X_dn2 / one_plus_X - dX_dn * dX_dn / one_plus_X_squared
    d2L_dndzeta = d2X_dndzeta / one_plus_X - dX_dn * dX_dzeta / one_plus_X_squared
    d2L_dzeta2 = d2X_dzeta2 / one_plus_X - dX_dzeta * dX_dzeta / one_plus_X_squared
    d2L_dnds = d2X_dnds / one_plus_X - dX_dn * dX_ds / one_plus_X_squared
    d2L_dzetads = d2X_dzetads / one_plus_X - dX_dzeta * dX_ds / one_plus_X_squared
    d2L_ds2 = d2X_ds2 / one_plus_X - dX_ds * dX_ds / one_plus_X_squared

    dH_dn = gamma * phi_cubed * dL_dn
    dH_dzeta = gamma * (dphi_cubed * L + phi_cubed * dL_dzeta)
    dH_ds = gamma * phi_cubed * dL_ds

    d2H_dn2 = gamma * phi_cubed * d2L_dn2
    d2H_dndzeta = gamma * (dphi_cubed * dL_dn + phi_cubed * d2L_dndzeta)
    d2H_dzeta2 = gamma * (d2phi_cubed * L + 2 * dphi_cubed * dL_dzeta + phi_cubed * d2L_dzeta2)
    d2H_dnds = gamma * phi_cubed * d2L_dnds
    d2H_dzetads = gamma * (dphi_cubed * dL_ds + phi_cubed * d2L_dzetads)
    d2H_ds2 = gamma * phi_cubed * d2L_ds2

    # Derivatives with respect to density and spin polarisation

    df_dzeta = density * (de_C_dzeta + dH_dzeta)

    d2f_dn2 = 2 * (de_C_dn + dH_dn) + density * (d2e_C_dn2 + d2H_dn2)
    d2f_dndzeta = (de_C_dzeta + dH_dzeta) + density * (d2e_C_dndzeta + d2H_dndzeta)
    d2f_dzeta2 = density * (d2e_C_dzeta2 + d2H_dzeta2)
    d2f_dnds = dH_ds + density * d2H_dnds
    d2f_dzetads = density * d2H_dzetads
    d2f_ds2 = density * d2H_ds2

    # Transforms onto the two spin densities

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma = calculate_spin_polarisation_transformation(zeta, density, df_dzeta, d2f_dn2, d2f_dndzeta, d2f_dzeta2, d2f_dnds, d2f_dzetads, d2f_ds2)

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma










def calculate_PW91_X_derivatives(T: ndarray, A: ndarray, B_over_gamma: float) -> tuple:

    """

    Calculates the logged quantity of the first PW91 gradient correction and its first two derivatives in T and A.

    Args:
        T (array): Reduced density gradient
        A (array): The density-dependent quantity inside the correction
        B_over_gamma (float): Ratio of the two defining constants

    Returns:
        X (array): The logged quantity
        dX_dT (array): First derivative with respect to T
        dX_dA (array): First derivative with respect to A
        d2X_dT2 (array): Second derivative with respect to T
        d2X_dTdA (array): Mixed second derivative with respect to T and A
        d2X_dA2 (array): Second derivative with respect to A

    """

    # The same rational function as in PBE correlation

    N = B_over_gamma * T * (1 + A * T)
    D = 1 + A * T + A * A * T * T

    X = N / D

    dN_dT, dN_dA = B_over_gamma * (1 + 2 * A * T), B_over_gamma * T * T
    d2N_dT2, d2N_dTdA = 2 * A * B_over_gamma, 2 * B_over_gamma * T

    dD_dT, dD_dA = A + 2 * A * A * T, T + 2 * A * T * T
    d2D_dT2, d2D_dTdA, d2D_dA2 = 2 * A * A, 1 + 4 * A * T, 2 * T * T

    dX_dT = (dN_dT - X * dD_dT) / D
    dX_dA = (dN_dA - X * dD_dA) / D

    d2X_dT2 = (d2N_dT2 - 2 * dX_dT * dD_dT - X * d2D_dT2) / D
    d2X_dTdA = (d2N_dTdA - dX_dT * dD_dA - dX_dA * dD_dT - X * d2D_dTdA) / D
    d2X_dA2 = (-2 * dX_dA * dD_dA - X * d2D_dA2) / D

    return X, dX_dT, dX_dA, d2X_dT2, d2X_dTdA, d2X_dA2










def calculate_PW91_damped_derivatives(T: ndarray, Z: ndarray) -> tuple:

    """

    Calculates G = T exp(-Z T), the shape of the second PW91 gradient correction, with its first two derivatives in T and Z.

    Args:
        T (array): Reduced density gradient
        Z (array): Damping coefficient

    Returns:
        G (array): The damped term
        dG_dT (array): First derivative with respect to T
        dG_dZ (array): First derivative with respect to Z
        d2G_dT2 (array): Second derivative with respect to T
        d2G_dTdZ (array): Mixed second derivative with respect to T and Z
        d2G_dZ2 (array): Second derivative with respect to Z

    """

    # The damping factor

    E = np.exp(-Z * T)

    G = T * E
    dG_dT = E * (1 - Z * T)
    dG_dZ = -T * T * E

    d2G_dT2 = -Z * E * (2 - Z * T)
    d2G_dTdZ = -T * E * (2 - Z * T)
    d2G_dZ2 = T ** 3 * E

    return G, dG_dT, dG_dZ, d2G_dT2, d2G_dTdZ, d2G_dZ2










def calculate_restricted_PW91_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted PW91 correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma

    """

    # Defining constants for PW91 correlation

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    beta = 16 * np.cbrt(3 / np.pi) * C_0
    gamma = beta * beta / (2 * alpha)
    B_over_gamma = beta / gamma
    prefactor = 16 * np.cbrt(3 / np.pi)

    inv_density = 1 / density

    # Local density correlation energy per particle and derivatives

    e_C, de_C_dr, d2e_C_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    r_s, _ = calculate_seitz_radius(density)

    dr_s_dn = -r_s * inv_density / 3
    d2r_s_dn2 = (4 / 9) * r_s * inv_density * inv_density

    de_C_dn = de_C_dr * dr_s_dn
    d2e_C_dn2 = d2e_C_dr2 * dr_s_dn * dr_s_dn + de_C_dr * d2r_s_dn2

    # Definition of A in PW91, where expm1 is more stable than exp - 1

    exp_factor = np.exp(-e_C / gamma)

    A = B_over_gamma / np.expm1(-e_C / gamma)

    dA_de = A * A * exp_factor / (B_over_gamma * gamma)
    d2A_de2 = A * A * exp_factor * (2 * A * exp_factor / B_over_gamma - 1) / (B_over_gamma * gamma * gamma)

    dA_dn = dA_de * de_C_dn
    d2A_dn2 = dA_de * d2e_C_dn2 + d2A_de2 * de_C_dn * de_C_dn

    # Reduced density gradient for PW91

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * k_F * density * density)

    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dnds = -(7 / 3) * dT_ds * inv_density

    # First correction term to LDA correlation

    X, dX_dT, dX_dA, d2X_dT2, d2X_dTdA, d2X_dA2 = calculate_PW91_X_derivatives(T, A, B_over_gamma)

    one_plus_X = 1 + X
    one_plus_X_squared = one_plus_X * one_plus_X

    dX_dn = dX_dT * dT_dn + dX_dA * dA_dn
    dX_ds = dX_dT * dT_ds
    d2X_dn2 = d2X_dT2 * dT_dn * dT_dn + 2 * d2X_dTdA * dT_dn * dA_dn + d2X_dA2 * dA_dn * dA_dn + dX_dT * d2T_dn2 + dX_dA * d2A_dn2
    d2X_dnds = d2X_dT2 * dT_dn * dT_ds + d2X_dTdA * dT_ds * dA_dn + dX_dT * d2T_dnds
    d2X_ds2 = d2X_dT2 * dT_ds * dT_ds

    dH0_dn = gamma * dX_dn / one_plus_X
    dH0_ds = gamma * dX_ds / one_plus_X
    d2H0_dn2 = gamma * (d2X_dn2 / one_plus_X - dX_dn * dX_dn / one_plus_X_squared)
    d2H0_dnds = gamma * (d2X_dnds / one_plus_X - dX_dn * dX_ds / one_plus_X_squared)
    d2H0_ds2 = gamma * (d2X_ds2 / one_plus_X - dX_ds * dX_ds / one_plus_X_squared)

    # Second correction term to LDA correlation

    B, dB_dn, d2B_dn2 = calculate_correlation_gradient_coefficient(density, -C_X)

    B = B - C_0 - 3 * C_X / 7

    ratio = 4 / (np.pi * k_F)

    dratio_dn = -ratio * inv_density / 3
    d2ratio_dn2 = (4 / 9) * ratio * inv_density * inv_density

    Z = 100 * ratio
    dZ_dn = 100 * dratio_dn
    d2Z_dn2 = 100 * d2ratio_dn2

    G, dG_dT, dG_dZ, d2G_dT2, d2G_dTdZ, d2G_dZ2 = calculate_PW91_damped_derivatives(T, Z)

    dG_dn = dG_dT * dT_dn + dG_dZ * dZ_dn
    dG_ds = dG_dT * dT_ds
    d2G_dn2 = d2G_dT2 * dT_dn * dT_dn + 2 * d2G_dTdZ * dT_dn * dZ_dn + d2G_dZ2 * dZ_dn * dZ_dn + dG_dT * d2T_dn2 + dG_dZ * d2Z_dn2
    d2G_dnds = d2G_dT2 * dT_dn * dT_ds + d2G_dTdZ * dT_ds * dZ_dn + dG_dT * d2T_dnds
    d2G_ds2 = d2G_dT2 * dT_ds * dT_ds

    dH1_dn = prefactor * (dB_dn * G + B * dG_dn)
    dH1_ds = prefactor * B * dG_ds
    d2H1_dn2 = prefactor * (d2B_dn2 * G + 2 * dB_dn * dG_dn + B * d2G_dn2)
    d2H1_dnds = prefactor * (dB_dn * dG_ds + B * d2G_dnds)
    d2H1_ds2 = prefactor * B * d2G_ds2

    # Final derivatives with respect to density and sigma

    d2f_dn2 = 2 * (de_C_dn + dH0_dn + dH1_dn) + density * (d2e_C_dn2 + d2H0_dn2 + d2H1_dn2)
    d2f_dnds = (dH0_ds + dH1_ds) + density * (d2H0_dnds + d2H1_dnds)
    d2f_ds2 = density * (d2H0_ds2 + d2H1_ds2)

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_PW91_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """

    Calculates the restricted PW91 spin correlation kernel for triplet excitations.

    PW91 correlation only sees the total square gradient, so the triplet kernel is a single density block.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density

    """

    # Defining constants for PW91 correlation

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    beta = 16 * np.cbrt(3 / np.pi) * C_0
    gamma = beta * beta / (2 * alpha)
    B_over_gamma = beta / gamma
    prefactor = 16 * np.cbrt(3 / np.pi)

    # Local density correlation and the PW92 spin stiffness

    _, e_C, _ = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    _, minus_alpha, _ = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    alpha_C = -minus_alpha

    exp_factor = np.exp(-e_C / gamma)

    A = B_over_gamma / np.expm1(-e_C / gamma)

    dA_de = A * A * exp_factor / (B_over_gamma * gamma)

    k_F = calculate_Fermi_wavevector(density=density)

    T = sigma * np.pi / (16 * k_F * density * density)

    X, dX_dT, dX_dA, _, _, _ = calculate_PW91_X_derivatives(T, A, B_over_gamma)

    # Spin polarisation curvatures, using phi(0) = 1 and phi''(0) = -2 / 9

    d2T_dzeta2 = (4 / 9) * T
    d2W_dzeta2 = alpha_C + (2 / 3) * e_C

    d2H0_dzeta2 = gamma * (-(2 / 3) * np.log1p(X) + (dX_dT * d2T_dzeta2 + dX_dA * dA_de * d2W_dzeta2) / (1 + X))

    B, _, _ = calculate_correlation_gradient_coefficient(density, -C_X)

    B = B - C_0 - 3 * C_X / 7

    ratio = 4 / (np.pi * k_F)

    G, dG_dT, dG_dZ, _, _, _ = calculate_PW91_damped_derivatives(T, 100 * ratio)

    # Curvature of the second correction term

    d2H1_dzeta2 = prefactor * B * (-(2 / 3) * G + dG_dT * d2T_dzeta2 - (800 / 9) * ratio * dG_dZ)

    # Converts the spin polarisation curvature into the spin density kernel

    f_mm = (alpha_C + d2H0_dzeta2 + d2H1_dzeta2) / density

    return f_mm










def calculate_unrestricted_PW91_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved PW91 correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_C_alpha_alpha (array): Second derivative of f = n * e_C with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma (array): Mixed second derivative with respect to the alpha density and the total square gradient
        f_C_beta_sigma (array): Mixed second derivative with respect to the beta density and the total square gradient
        f_C_sigma_sigma (array): Second derivative with respect to the total square gradient

    """

    # Defining constants for PW91 correlation

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    beta = 16 * np.cbrt(3 / np.pi) * C_0
    gamma = beta * beta / (2 * alpha)
    B_over_gamma = beta / gamma
    prefactor = 16 * np.cbrt(3 / np.pi)

    inv_density = 1 / density

    # This functional only depends on the total square density gradient, not its spin components

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)

    # Local density correlation energy density and derivatives

    e_C, de_C_dn, de_C_dzeta, d2e_C_dn2, d2e_C_dndzeta, d2e_C_dzeta2, zeta = calculate_PW_spin_correlation_derivatives(alpha_density, beta_density, density)

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus) / 2
    phi_prime = (1 / cbrt_plus - 1 / cbrt_minus) / 3
    phi_prime_prime = -(1 / 9) * (1 / cbrt_plus ** 4 + 1 / cbrt_minus ** 4)

    phi_cubed = phi * phi * phi
    dphi_cubed = 3 * phi * phi * phi_prime
    d2phi_cubed = 6 * phi * phi_prime * phi_prime + 3 * phi * phi * phi_prime_prime

    phi_fourth = phi_cubed * phi
    dphi_fourth = 4 * phi_cubed * phi_prime
    d2phi_fourth = 12 * phi * phi * phi_prime * phi_prime + 4 * phi_cubed * phi_prime_prime

    gamma_phi_cubed = gamma * phi_cubed

    W = e_C / gamma_phi_cubed
    dW_dn = de_C_dn / gamma_phi_cubed
    dW_dzeta = (de_C_dzeta / phi_cubed - e_C * dphi_cubed / (phi_cubed * phi_cubed)) / gamma
    d2W_dn2 = d2e_C_dn2 / gamma_phi_cubed
    d2W_dndzeta = (d2e_C_dndzeta / phi_cubed - de_C_dn * dphi_cubed / (phi_cubed * phi_cubed)) / gamma
    d2W_dzeta2 = (d2e_C_dzeta2 / phi_cubed - 2 * de_C_dzeta * dphi_cubed / (phi_cubed * phi_cubed) - e_C * d2phi_cubed / (phi_cubed * phi_cubed) + 2 * e_C * dphi_cubed * dphi_cubed / (phi_cubed * phi_cubed * phi_cubed)) / gamma

    exp_factor = np.exp(-W)

    A = B_over_gamma / np.expm1(-W)

    dA_dW = A * A * exp_factor / B_over_gamma
    d2A_dW2 = A * A * exp_factor * (2 * A * exp_factor / B_over_gamma - 1) / B_over_gamma

    dA_dn, dA_dzeta = dA_dW * dW_dn, dA_dW * dW_dzeta
    d2A_dn2 = dA_dW * d2W_dn2 + d2A_dW2 * dW_dn * dW_dn
    d2A_dndzeta = dA_dW * d2W_dndzeta + d2A_dW2 * dW_dn * dW_dzeta
    d2A_dzeta2 = dA_dW * d2W_dzeta2 + d2A_dW2 * dW_dzeta * dW_dzeta

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * phi * phi * k_F * density * density)

    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    dT_dzeta = -2 * T * phi_prime / phi
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dnds = -(7 / 3) * dT_ds * inv_density
    d2T_dndzeta = -2 * dT_dn * phi_prime / phi
    d2T_dzetads = -2 * dT_ds * phi_prime / phi
    d2T_dzeta2 = T * (6 * phi_prime * phi_prime / (phi * phi) - 2 * phi_prime_prime / phi)

    # First correction term to LDA correlation

    X, dX_dT, dX_dA, d2X_dT2, d2X_dTdA, d2X_dA2 = calculate_PW91_X_derivatives(T, A, B_over_gamma)

    dX_dn = dX_dT * dT_dn + dX_dA * dA_dn
    dX_dzeta = dX_dT * dT_dzeta + dX_dA * dA_dzeta
    dX_ds = dX_dT * dT_ds
    d2X_dn2 = d2X_dT2 * dT_dn * dT_dn + 2 * d2X_dTdA * dT_dn * dA_dn + d2X_dA2 * dA_dn * dA_dn + dX_dT * d2T_dn2 + dX_dA * d2A_dn2
    d2X_dndzeta = d2X_dT2 * dT_dn * dT_dzeta + d2X_dTdA * (dT_dn * dA_dzeta + dT_dzeta * dA_dn) + d2X_dA2 * dA_dn * dA_dzeta + dX_dT * d2T_dndzeta + dX_dA * d2A_dndzeta
    d2X_dzeta2 = d2X_dT2 * dT_dzeta * dT_dzeta + 2 * d2X_dTdA * dT_dzeta * dA_dzeta + d2X_dA2 * dA_dzeta * dA_dzeta + dX_dT * d2T_dzeta2 + dX_dA * d2A_dzeta2
    d2X_dnds = d2X_dT2 * dT_dn * dT_ds + d2X_dTdA * dT_ds * dA_dn + dX_dT * d2T_dnds
    d2X_dzetads = d2X_dT2 * dT_dzeta * dT_ds + d2X_dTdA * dT_ds * dA_dzeta + dX_dT * d2T_dzetads
    d2X_ds2 = d2X_dT2 * dT_ds * dT_ds

    one_plus_X = 1 + X
    one_plus_X_squared = one_plus_X * one_plus_X

    L = np.log1p(X)
    dL_dn, dL_dzeta, dL_ds = dX_dn / one_plus_X, dX_dzeta / one_plus_X, dX_ds / one_plus_X
    d2L_dn2 = d2X_dn2 / one_plus_X - dX_dn * dX_dn / one_plus_X_squared
    d2L_dndzeta = d2X_dndzeta / one_plus_X - dX_dn * dX_dzeta / one_plus_X_squared
    d2L_dzeta2 = d2X_dzeta2 / one_plus_X - dX_dzeta * dX_dzeta / one_plus_X_squared
    d2L_dnds = d2X_dnds / one_plus_X - dX_dn * dX_ds / one_plus_X_squared
    d2L_dzetads = d2X_dzetads / one_plus_X - dX_dzeta * dX_ds / one_plus_X_squared
    d2L_ds2 = d2X_ds2 / one_plus_X - dX_ds * dX_ds / one_plus_X_squared

    dH0_dn = gamma * phi_cubed * dL_dn
    dH0_dzeta = gamma * (dphi_cubed * L + phi_cubed * dL_dzeta)
    dH0_ds = gamma * phi_cubed * dL_ds
    d2H0_dn2 = gamma * phi_cubed * d2L_dn2
    d2H0_dndzeta = gamma * (dphi_cubed * dL_dn + phi_cubed * d2L_dndzeta)
    d2H0_dzeta2 = gamma * (d2phi_cubed * L + 2 * dphi_cubed * dL_dzeta + phi_cubed * d2L_dzeta2)
    d2H0_dnds = gamma * phi_cubed * d2L_dnds
    d2H0_dzetads = gamma * (dphi_cubed * dL_ds + phi_cubed * d2L_dzetads)
    d2H0_ds2 = gamma * phi_cubed * d2L_ds2

    # Second correction term to LDA correlation

    B, dB_dn, d2B_dn2 = calculate_correlation_gradient_coefficient(density, -C_X)

    B = B - C_0 - 3 * C_X / 7

    ratio = 4 / (np.pi * k_F)
    dratio_dn = -ratio * inv_density / 3
    d2ratio_dn2 = (4 / 9) * ratio * inv_density * inv_density

    M = B * phi_cubed
    dM_dn, dM_dzeta = dB_dn * phi_cubed, B * dphi_cubed
    d2M_dn2, d2M_dndzeta, d2M_dzeta2 = d2B_dn2 * phi_cubed, dB_dn * dphi_cubed, B * d2phi_cubed

    Z = 100 * phi_fourth * ratio
    dZ_dn, dZ_dzeta = 100 * phi_fourth * dratio_dn, 100 * dphi_fourth * ratio
    d2Z_dn2, d2Z_dndzeta, d2Z_dzeta2 = 100 * phi_fourth * d2ratio_dn2, 100 * dphi_fourth * dratio_dn, 100 * d2phi_fourth * ratio

    G, dG_dT, dG_dZ, d2G_dT2, d2G_dTdZ, d2G_dZ2 = calculate_PW91_damped_derivatives(T, Z)

    dG_dn = dG_dT * dT_dn + dG_dZ * dZ_dn
    dG_dzeta = dG_dT * dT_dzeta + dG_dZ * dZ_dzeta
    dG_ds = dG_dT * dT_ds

    d2G_dn2 = d2G_dT2 * dT_dn * dT_dn + 2 * d2G_dTdZ * dT_dn * dZ_dn + d2G_dZ2 * dZ_dn * dZ_dn + dG_dT * d2T_dn2 + dG_dZ * d2Z_dn2
    d2G_dndzeta = d2G_dT2 * dT_dn * dT_dzeta + d2G_dTdZ * (dT_dn * dZ_dzeta + dT_dzeta * dZ_dn) + d2G_dZ2 * dZ_dn * dZ_dzeta + dG_dT * d2T_dndzeta + dG_dZ * d2Z_dndzeta
    d2G_dzeta2 = d2G_dT2 * dT_dzeta * dT_dzeta + 2 * d2G_dTdZ * dT_dzeta * dZ_dzeta + d2G_dZ2 * dZ_dzeta * dZ_dzeta + dG_dT * d2T_dzeta2 + dG_dZ * d2Z_dzeta2
    d2G_dnds = d2G_dT2 * dT_dn * dT_ds + d2G_dTdZ * dT_ds * dZ_dn + dG_dT * d2T_dnds
    d2G_dzetads = d2G_dT2 * dT_dzeta * dT_ds + d2G_dTdZ * dT_ds * dZ_dzeta + dG_dT * d2T_dzetads
    d2G_ds2 = d2G_dT2 * dT_ds * dT_ds

    dH1_dn = prefactor * (dM_dn * G + M * dG_dn)
    dH1_dzeta = prefactor * (dM_dzeta * G + M * dG_dzeta)
    dH1_ds = prefactor * M * dG_ds
    d2H1_dn2 = prefactor * (d2M_dn2 * G + 2 * dM_dn * dG_dn + M * d2G_dn2)
    d2H1_dndzeta = prefactor * (d2M_dndzeta * G + dM_dn * dG_dzeta + dM_dzeta * dG_dn + M * d2G_dndzeta)
    d2H1_dzeta2 = prefactor * (d2M_dzeta2 * G + 2 * dM_dzeta * dG_dzeta + M * d2G_dzeta2)
    d2H1_dnds = prefactor * (dM_dn * dG_ds + M * d2G_dnds)
    d2H1_dzetads = prefactor * (dM_dzeta * dG_ds + M * d2G_dzetads)
    d2H1_ds2 = prefactor * M * d2G_ds2

    df_dzeta = density * (de_C_dzeta + dH0_dzeta + dH1_dzeta)
    d2f_dn2 = 2 * (de_C_dn + dH0_dn + dH1_dn) + density * (d2e_C_dn2 + d2H0_dn2 + d2H1_dn2)
    d2f_dndzeta = (de_C_dzeta + dH0_dzeta + dH1_dzeta) + density * (d2e_C_dndzeta + d2H0_dndzeta + d2H1_dndzeta)
    d2f_dzeta2 = density * (d2e_C_dzeta2 + d2H0_dzeta2 + d2H1_dzeta2)
    d2f_dnds = (dH0_ds + dH1_ds) + density * (d2H0_dnds + d2H1_dnds)
    d2f_dzetads = density * (d2H0_dzetads + d2H1_dzetads)
    d2f_ds2 = density * (d2H0_ds2 + d2H1_ds2)

    # Transforms onto the two spin densities

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma = calculate_spin_polarisation_transformation(zeta, density, df_dzeta, d2f_dn2, d2f_dndzeta, d2f_dzeta2, d2f_dnds, d2f_dzetads, d2f_ds2)

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma










def calculate_P86_gradient_derivatives(C: ndarray, P: ndarray) -> tuple:

    """

    Calculates the P86 gradient correction H = C P exp(-kappa sqrt(P) / C) and its second derivatives in C and P.

    Args:
        C (array): The P86 coefficient
        P (array): The reduced gradient sigma / n ^ (7 / 3)

    Returns:
        H (array): The gradient correction
        H_C (array): First derivative with respect to C
        H_P (array): First derivative with respect to P
        H_CC (array): Second derivative with respect to C
        H_CP (array): Mixed second derivative with respect to C and P
        H_PP (array): Second derivative with respect to P

    """

    # Constants defining P86 correlation, combined into one

    kappa = 1.745 * 0.11 * 0.004235

    # Definition of phi from P86 paper

    phi = kappa * P ** (1 / 2) / C

    E = np.exp(-phi)

    # Gradient correction and its derivatives, where H_PP diverges as P ^ (-1 / 2) at small gradient

    H = C * P * E
    H_C = P * E * (1 + phi)
    H_P = C * E * (1 - phi / 2)

    H_CC = P * phi * phi * E / C
    H_CP = E * ((1 + phi) - phi * phi / 2)
    H_PP = -C * phi * E * (3 - phi) / (4 * P)

    return H, H_C, H_P, H_CC, H_CP, H_PP










def calculate_P86_spin_scaling(zeta: ndarray) -> tuple:

    """

    Calculates the reciprocal of the P86 spin scaling factor and its first two derivatives with respect to the spin polarisation.

    Args:
        zeta (array): Local spin polarisation

    Returns:
        K (array): Reciprocal of the spin scaling factor
        dK_dzeta (array): First derivative with respect to spin polarisation
        d2K_dzeta2 (array): Second derivative with respect to spin polarisation

    """

    # Spin scaling factor from P86 paper and its derivatives

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    S = cbrt_plus ** 5 + cbrt_minus ** 5
    dS = (5 / 3) * (cbrt_plus * cbrt_plus - cbrt_minus * cbrt_minus)
    d2S = (10 / 9) * (1 / cbrt_plus + 1 / cbrt_minus)

    d = (S / 2) ** (1 / 2)
    dd = dS / (4 * d)
    d2d = d2S / (4 * d) - dd * dd / d

    # Reciprocal of the spin scaling factor and its derivatives

    K = 1 / d
    dK_dzeta = -dd / (d * d)
    d2K_dzeta2 = -d2d / (d * d) + 2 * dd * dd / (d * d * d)

    return K, dK_dzeta, d2K_dzeta2










def calculate_restricted_P86_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted P86 correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma

    """

    # Sigma is cleaned at the square of the density floor, otherwise this breaks at zero gradient

    sigma = clean(sigma, floor=constants.sigma_floor)

    inv_density = 1 / density

    # Local density correlation energy per particle and derivatives

    e_C, de_C_dr, d2e_C_dr2 = calculate_PW_correlation_derivatives(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    r_s, _ = calculate_seitz_radius(density)

    dr_s_dn = -r_s * inv_density / 3
    d2r_s_dn2 = (4 / 9) * r_s * inv_density * inv_density

    de_C_dn = de_C_dr * dr_s_dn
    d2e_C_dn2 = d2e_C_dr2 * dr_s_dn * dr_s_dn + de_C_dr * d2r_s_dn2

    # Form of C(r_s) and the reduced gradient, with their derivatives

    C, dC_dn, d2C_dn2 = calculate_correlation_gradient_coefficient(density, 0.001667)

    dP_ds = 1 / np.cbrt(density) ** 7

    P = sigma * dP_ds

    dP_dn = -(7 / 3) * P * inv_density
    d2P_dn2 = (70 / 9) * P * inv_density * inv_density
    d2P_dnds = -(7 / 3) * dP_ds * inv_density

    # GGA correction to LDA energy density and its derivatives

    H, H_C, H_P, H_CC, H_CP, H_PP = calculate_P86_gradient_derivatives(C, P)

    dH_dn = H_C * dC_dn + H_P * dP_dn
    dH_ds = H_P * dP_ds
    d2H_dn2 = H_CC * dC_dn * dC_dn + 2 * H_CP * dC_dn * dP_dn + H_PP * dP_dn * dP_dn + H_C * d2C_dn2 + H_P * d2P_dn2
    d2H_dnds = (H_CP * dC_dn + H_PP * dP_dn) * dP_ds + H_P * d2P_dnds
    d2H_ds2 = H_PP * dP_ds * dP_ds

    # Second derivatives with respect to the density and sigma

    d2f_dn2 = 2 * (de_C_dn + dH_dn) + density * (d2e_C_dn2 + d2H_dn2)
    d2f_dnds = dH_ds + density * d2H_dnds
    d2f_ds2 = density * d2H_ds2

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_P86_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """

    Calculates the restricted P86 spin correlation kernel for triplet excitations.

    P86 correlation only sees the total square gradient, so the triplet kernel is a single density block.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density

    """

    # The PW92 spin stiffness

    _, minus_alpha, _ = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    C, _, _ = calculate_correlation_gradient_coefficient(density, 0.001667)

    P = sigma / np.cbrt(density) ** 7

    H, _, _, _, _, _ = calculate_P86_gradient_derivatives(C, P)

    # The gradient correction enters through the reciprocal spin scaling factor, with curvature -5 / 9

    f_mm = (-minus_alpha - (5 / 9) * H) / density

    return f_mm










def calculate_unrestricted_P86_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved P86 correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_C_alpha_alpha (array): Second derivative of f = n * e_C with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma (array): Mixed second derivative with respect to the alpha density and the total square gradient
        f_C_beta_sigma (array): Mixed second derivative with respect to the beta density and the total square gradient
        f_C_sigma_sigma (array): Second derivative with respect to the total square gradient

    """

    inv_density = 1 / density

    # This functional only depends on the total square density gradient, not its spin components

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)

    e_C, de_C_dn, de_C_dzeta, d2e_C_dn2, d2e_C_dndzeta, d2e_C_dzeta2, zeta = calculate_PW_spin_correlation_derivatives(alpha_density, beta_density, density)

    C, dC_dn, d2C_dn2 = calculate_correlation_gradient_coefficient(density, 0.001667)

    dP_ds = 1 / np.cbrt(density) ** 7

    P = sigma * dP_ds

    dP_dn = -(7 / 3) * P * inv_density
    d2P_dn2 = (70 / 9) * P * inv_density * inv_density
    d2P_dnds = -(7 / 3) * dP_ds * inv_density

    H, H_C, H_P, H_CC, H_CP, H_PP = calculate_P86_gradient_derivatives(C, P)

    dH_dn = H_C * dC_dn + H_P * dP_dn
    dH_ds = H_P * dP_ds
    d2H_dn2 = H_CC * dC_dn * dC_dn + 2 * H_CP * dC_dn * dP_dn + H_PP * dP_dn * dP_dn + H_C * d2C_dn2 + H_P * d2P_dn2
    d2H_dnds = (H_CP * dC_dn + H_PP * dP_dn) * dP_ds + H_P * d2P_dnds
    d2H_ds2 = H_PP * dP_ds * dP_ds

    # Spin polarisation functions

    K, dK_dzeta, d2K_dzeta2 = calculate_P86_spin_scaling(zeta)

    df_dzeta = density * (de_C_dzeta + H * dK_dzeta)
    d2f_dn2 = 2 * (de_C_dn + dH_dn * K) + density * (d2e_C_dn2 + d2H_dn2 * K)
    d2f_dndzeta = (de_C_dzeta + H * dK_dzeta) + density * (d2e_C_dndzeta + dH_dn * dK_dzeta)
    d2f_dzeta2 = density * (d2e_C_dzeta2 + H * d2K_dzeta2)
    d2f_dnds = dH_ds * K + density * d2H_dnds * K
    d2f_dzetads = density * dH_ds * dK_dzeta
    d2f_ds2 = density * d2H_ds2 * K

    # Transforms onto the two spin densities

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma = calculate_spin_polarisation_transformation(zeta, density, df_dzeta, d2f_dn2, d2f_dndzeta, d2f_dzeta2, d2f_dnds, d2f_dzetads, d2f_ds2)

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma










def calculate_B97_correlation_coefficients(calculation: Calculation) -> tuple:

    """

    Returns the same-spin and opposite-spin B97 correlation coefficients, chosen as for the energy.

    Args:
        calculation (Calculation): Calculation object

    Returns:
        c_ss (list): Same-spin coefficients
        c_ab (list): Opposite-spin coefficients

    """

    # The parameters can be for Becke's hybrid (first case) or Grimme's dispersion-corrected GGA (second case)

    c_ab = [0.9454, 0.7471, -4.5961] if calculation.method.name == "B97" else [0.69041, 6.30270, -14.9712]
    c_ss = [0.1737, 2.3487, -2.4868] if calculation.method.name == "B97" else [0.22340, -1.56208, 1.94293]

    return c_ss, c_ab










def calculate_B97_inhomogeneity_derivatives(c: list, gamma: float, t: ndarray) -> tuple:

    """

    Calculates a B97 inhomogeneity factor and its first two derivatives with respect to its reduced gradient.

    Args:
        c (list): The three coefficients of the factor
        gamma (float): The finite range parameter
        t (array): Reduced gradient variable

    Returns:
        g (array): The inhomogeneity factor
        dg_dt (array): First derivative with respect to t
        d2g_dt2 (array): Second derivative with respect to t

    """

    # Finite range variable and its derivatives

    denom = 1 / (1 + gamma * t)

    u = gamma * t * denom

    du_dt = gamma * denom * denom
    d2u_dt2 = -2 * gamma * gamma * denom * denom * denom

    # Inhomogeneity factor, a quadratic in u, and its derivatives

    dg_du = c[1] + 2 * c[2] * u

    g = c[0] + (c[1] + c[2] * u) * u
    dg_dt = dg_du * du_dt
    d2g_dt2 = 2 * c[2] * du_dt * du_dt + dg_du * d2u_dt2

    return g, dg_dt, d2g_dt2










def calculate_PW_ferromagnetic_derivatives(spin_density: ndarray) -> tuple:

    """

    Calculates the same-spin Stoll piece W = p times the ferromagnetic PW92 energy per particle, with its first two density derivatives.

    Args:
        spin_density (array): Density of one spin channel on integration grid

    Returns:
        W (array): The same-spin energy per unit volume
        dW_dn (array): First derivative with respect to the spin density
        d2W_dn2 (array): Second derivative with respect to the spin density

    """

    # The ferromagnetic PW92 fit and its Seitz-radius derivatives

    e_C, de_C_dr, d2e_C_dr2 = calculate_PW_correlation_derivatives(spin_density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)

    r_s, _ = calculate_seitz_radius(spin_density)

    # Energy density and its derivatives with respect to the spin density

    W = spin_density * e_C
    dW_dn = e_C - r_s / 3 * de_C_dr
    d2W_dn2 = (r_s * r_s * d2e_C_dr2 - 2 * r_s * de_C_dr) / (9 * spin_density)

    return W, dW_dn, d2W_dn2










def calculate_B97_channel_gradient_derivatives(spin_density: ndarray, spin_sigma: ndarray) -> tuple:

    """

    Calculates the reduced gradient t = sigma_ss / p ^ (8 / 3) of one spin channel, with its derivatives.

    Args:
        spin_density (array): Density of one spin channel on integration grid
        spin_sigma (array): Square density gradient of the same spin channel

    Returns:
        t (array): Reduced gradient of the channel
        dt_dn (array): Derivative with respect to the spin density
        dt_ds (array): Derivative with respect to the channel square gradient
        d2t_dn2 (array): Second derivative with respect to the spin density
        d2t_dnds (array): Mixed second derivative

    """

    # Reduced gradient, which is linear in the channel square gradient

    dt_ds = 1 / np.cbrt(spin_density) ** 8

    t = spin_sigma * dt_ds

    # Density derivatives of the reduced gradient

    dt_dn = -(8 / 3) * t / spin_density
    d2t_dn2 = (88 / 9) * t / (spin_density * spin_density)
    d2t_dnds = -(8 / 3) * dt_ds / spin_density

    return t, dt_dn, dt_ds, d2t_dn2, d2t_dnds










def calculate_restricted_B97_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97 correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma

    """

    # Same-spin and opposite-spin coefficients and finite range parameters

    c_ss, c_ab = calculate_B97_correlation_coefficients(calculation)

    gamma_ss = 0.2
    gamma_ab = 0.006

    # For a closed shell the functional is U * g_ss + V * g_ab

    spin_density = density / 2

    W, dW_dn, d2W_dn2 = calculate_PW_ferromagnetic_derivatives(spin_density)

    # Same-spin piece U, whose derivatives pick up factors from the halved density

    U = 2 * W
    dU_dn = dW_dn
    d2U_dn2 = d2W_dn2 / 2

    # Opposite-spin piece V, the local correlation minus the same-spin piece

    df_dn_LSDA, _, _, e_C_LSDA = calculate_restricted_PW_correlation(density, sigma, tau, calculation)

    d2f_dn2_LSDA = calculate_restricted_PW_correlation_kernel(density, sigma, tau, calculation)

    V = density * e_C_LSDA - U
    dV_dn = df_dn_LSDA - dU_dn
    d2V_dn2 = d2f_dn2_LSDA - d2U_dn2

    # The cube root four makes the channel reduced gradient match the total density convention

    dt_ds = np.cbrt(4) / np.cbrt(density) ** 8

    t = sigma * dt_ds

    dt_dn = -(8 / 3) * t / density
    d2t_dn2 = (88 / 9) * t / (density * density)
    d2t_dnds = -(8 / 3) * dt_ds / density

    g_ss, dg_ss_dt, d2g_ss_dt2 = calculate_B97_inhomogeneity_derivatives(c_ss, gamma_ss, t)
    g_ab, dg_ab_dt, d2g_ab_dt2 = calculate_B97_inhomogeneity_derivatives(c_ab, gamma_ab, t)

    # These two combinations appear repeatedly

    first = U * dg_ss_dt + V * dg_ab_dt
    second = U * d2g_ss_dt2 + V * d2g_ab_dt2

    # Second derivatives with respect to the density and sigma

    d2f_dn2 = d2U_dn2 * g_ss + d2V_dn2 * g_ab + 2 * (dU_dn * dg_ss_dt + dV_dn * dg_ab_dt) * dt_dn + second * dt_dn * dt_dn + first * d2t_dn2

    d2f_dnds = (dU_dn * dg_ss_dt + dV_dn * dg_ab_dt) * dt_ds + second * dt_dn * dt_ds + first * d2t_dnds

    d2f_ds2 = second * dt_ds * dt_ds

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_B97_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97 spin correlation kernel for triplet excitations.

    B97 sees the two channel square gradients separately, so unlike PBE the triplet kernel has gradient blocks.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_nm_sigma_nm (array): Second derivative with respect to grad(n) . grad(m)
        f_sigma_mm (array): Derivative with respect to the square spin density gradient

    """

    # Same-spin and opposite-spin coefficients and finite range parameters

    c_ss, c_ab = calculate_B97_correlation_coefficients(calculation)

    gamma_ss = 0.2
    gamma_ab = 0.006

    spin_density = density / 2

    W, dW_dn, d2W_dn2 = calculate_PW_ferromagnetic_derivatives(spin_density)

    t, dt_dn, dt_ds, d2t_dn2, d2t_dnds = calculate_B97_channel_gradient_derivatives(spin_density, sigma / 4)

    g_ss, dg_ss_dt, d2g_ss_dt2 = calculate_B97_inhomogeneity_derivatives(c_ss, gamma_ss, t)
    g_ab, dg_ab_dt, d2g_ab_dt2 = calculate_B97_inhomogeneity_derivatives(c_ab, gamma_ab, t)

    # Local correlation and its spin-resolved kernel blocks

    df_dn_LSDA, _, _, e_C_LSDA = calculate_restricted_PW_correlation(density, sigma, tau, calculation)

    f_C_alpha_alpha, f_C_alpha_beta, _ = calculate_unrestricted_PW_correlation_kernel(spin_density, spin_density, density, None, None, None, None, None, calculation)

    # Opposite-spin piece and its derivatives with respect to one spin density

    C = density * e_C_LSDA - 2 * W
    dC_dn = df_dn_LSDA - dW_dn
    d2C_dn2 = f_C_alpha_alpha - d2W_dn2
    d2C_dndn = f_C_alpha_beta

    # Same-spin piece of one channel and its derivatives

    P_nn = d2g_ss_dt2 * dt_dn * dt_dn * W + dg_ss_dt * d2t_dn2 * W + 2 * dg_ss_dt * dt_dn * dW_dn + g_ss * d2W_dn2

    P_nds = d2g_ss_dt2 * dt_dn * dt_ds * W + dg_ss_dt * d2t_dnds * W + dg_ss_dt * dt_ds * dW_dn

    # Terms even in the two channels cancel at the closed shell

    f_mm = (P_nn + dg_ab_dt * (d2t_dn2 / 2) * C + g_ab * (d2C_dn2 - d2C_dndn)) / 2

    f_m_sigma_nm = (P_nds + dg_ab_dt * (d2t_dnds / 2) * C) / 2

    f_sigma_nm_sigma_nm = d2g_ss_dt2 * dt_ds * dt_ds * W / 2

    f_sigma_mm = (dg_ss_dt * dt_ds * W + dg_ab_dt * (dt_ds / 2) * C) / 2

    return f_mm, f_m_sigma_nm, f_sigma_nm_sigma_nm, f_sigma_mm










def calculate_unrestricted_B97_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved B97 correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_C_alpha_alpha (array): Second derivative of f = n * e_C with respect to the alpha density
        f_C_alpha_beta (array): Mixed second derivative with respect to the alpha and beta densities
        f_C_beta_beta (array): Second derivative with respect to the beta density
        f_C_alpha_sigma_aa (array): Mixed second derivative with respect to the alpha density and sigma alpha-alpha
        f_C_alpha_sigma_bb (array): Mixed second derivative with respect to the alpha density and sigma beta-beta
        f_C_beta_sigma_aa (array): Mixed second derivative with respect to the beta density and sigma alpha-alpha
        f_C_beta_sigma_bb (array): Mixed second derivative with respect to the beta density and sigma beta-beta
        f_C_sigma_aa_sigma_aa (array): Second derivative with respect to sigma alpha-alpha
        f_C_sigma_aa_sigma_bb (array): Mixed second derivative with respect to the two square gradients
        f_C_sigma_bb_sigma_bb (array): Second derivative with respect to sigma beta-beta

    """

    # Same-spin and opposite-spin coefficients and finite range parameters

    c_ss, c_ab = calculate_B97_correlation_coefficients(calculation)

    gamma_ss = 0.2
    gamma_ab = 0.006

    # The same-spin pieces, one per channel

    W_a, dW_a, d2W_a = calculate_PW_ferromagnetic_derivatives(alpha_density)
    W_b, dW_b, d2W_b = calculate_PW_ferromagnetic_derivatives(beta_density)

    t_a, dt_a_dna, dt_a_ds, d2t_a_dna2, d2t_a_dnads = calculate_B97_channel_gradient_derivatives(alpha_density, sigma_aa)
    t_b, dt_b_dnb, dt_b_ds, d2t_b_dnb2, d2t_b_dnbds = calculate_B97_channel_gradient_derivatives(beta_density, sigma_bb)

    # The opposite-spin factor uses the mean reduced gradient, so its derivatives are halved

    u_na, u_nb = dt_a_dna / 2, dt_b_dnb / 2
    u_saa, u_sbb = dt_a_ds / 2, dt_b_ds / 2
    u_nana, u_nbnb = d2t_a_dna2 / 2, d2t_b_dnb2 / 2
    u_na_saa, u_nb_sbb = d2t_a_dnads / 2, d2t_b_dnbds / 2

    g_a, dg_a, d2g_a = calculate_B97_inhomogeneity_derivatives(c_ss, gamma_ss, t_a)
    g_b, dg_b, d2g_b = calculate_B97_inhomogeneity_derivatives(c_ss, gamma_ss, t_b)
    g_ab, dg_ab, d2g_ab = calculate_B97_inhomogeneity_derivatives(c_ab, gamma_ab, (t_a + t_b) / 2)

    # Local correlation, its potential and its spin-resolved kernel

    df_dn_alpha, df_dn_beta, _, _, _, _, _, e_C_LSDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    F_aa, F_ab, F_bb = calculate_unrestricted_PW_correlation_kernel(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    # Opposite-spin piece, which has no gradient dependence of its own

    C = density * e_C_LSDA - W_a - W_b
    dC_dna, dC_dnb = df_dn_alpha - dW_a, df_dn_beta - dW_b
    d2C_dna2, d2C_dnb2, d2C_dnadnb = F_aa - d2W_a, F_bb - d2W_b, F_ab

    # Same-spin contributions, each confined to its own channel

    P_nana = d2g_a * dt_a_dna * dt_a_dna * W_a + dg_a * d2t_a_dna2 * W_a + 2 * dg_a * dt_a_dna * dW_a + g_a * d2W_a
    P_na_saa = d2g_a * dt_a_dna * dt_a_ds * W_a + dg_a * d2t_a_dnads * W_a + dg_a * dt_a_ds * dW_a
    P_saa_saa = d2g_a * dt_a_ds * dt_a_ds * W_a

    Q_nbnb = d2g_b * dt_b_dnb * dt_b_dnb * W_b + dg_b * d2t_b_dnb2 * W_b + 2 * dg_b * dt_b_dnb * dW_b + g_b * d2W_b
    Q_nb_sbb = d2g_b * dt_b_dnb * dt_b_ds * W_b + dg_b * d2t_b_dnbds * W_b + dg_b * dt_b_ds * dW_b
    Q_sbb_sbb = d2g_b * dt_b_ds * dt_b_ds * W_b

    # Opposite-spin contributions, which reach every block

    R_nana = d2g_ab * u_na * u_na * C + dg_ab * u_nana * C + 2 * dg_ab * u_na * dC_dna + g_ab * d2C_dna2
    R_nanb = d2g_ab * u_na * u_nb * C + dg_ab * (u_na * dC_dnb + u_nb * dC_dna) + g_ab * d2C_dnadnb
    R_nbnb = d2g_ab * u_nb * u_nb * C + dg_ab * u_nbnb * C + 2 * dg_ab * u_nb * dC_dnb + g_ab * d2C_dnb2

    R_na_saa = d2g_ab * u_na * u_saa * C + dg_ab * u_na_saa * C + dg_ab * u_saa * dC_dna
    R_na_sbb = d2g_ab * u_na * u_sbb * C + dg_ab * u_sbb * dC_dna
    R_nb_saa = d2g_ab * u_nb * u_saa * C + dg_ab * u_saa * dC_dnb
    R_nb_sbb = d2g_ab * u_nb * u_sbb * C + dg_ab * u_nb_sbb * C + dg_ab * u_sbb * dC_dnb

    R_saa_saa = d2g_ab * u_saa * u_saa * C
    R_saa_sbb = d2g_ab * u_saa * u_sbb * C
    R_sbb_sbb = d2g_ab * u_sbb * u_sbb * C

    # Nothing depends on sigma_ab, so its blocks vanish and are not returned

    f_C_alpha_alpha = P_nana + R_nana
    f_C_alpha_beta = R_nanb
    f_C_beta_beta = Q_nbnb + R_nbnb

    f_C_alpha_sigma_aa = P_na_saa + R_na_saa
    f_C_alpha_sigma_bb = R_na_sbb
    f_C_beta_sigma_aa = R_nb_saa
    f_C_beta_sigma_bb = Q_nb_sbb + R_nb_sbb

    f_C_sigma_aa_sigma_aa = P_saa_saa + R_saa_saa
    f_C_sigma_aa_sigma_bb = R_saa_sbb
    f_C_sigma_bb_sigma_bb = Q_sbb_sbb + R_sbb_sbb

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma_aa, f_C_alpha_sigma_bb, f_C_beta_sigma_aa, f_C_beta_sigma_bb, f_C_sigma_aa_sigma_aa, f_C_sigma_aa_sigma_bb, f_C_sigma_bb_sigma_bb










def calculate_restricted_3P_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted three-parameter correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma

    """

    method = calculation.method.name

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V

    d2f_dn2_LDA = correlation_density_kernels["VWN3" if "G" in method else "VWN5"](density, sigma, tau, calculation)

    # Picks the GGA correlation kernel depending on the method

    if "LYP" in method: correlation_kernel = correlation_density_kernels["LYP"]
    if "PW" in method: correlation_kernel = correlation_density_kernels["PW91"]
    if "P86" in method: correlation_kernel = correlation_density_kernels["P86"]

    # Calculates the kernel for the GGA part

    d2f_dn2_GGA, d2f_dnds_GGA, d2f_ds2_GGA = correlation_kernel(density, sigma, tau, calculation)

    # These parameters are the standard B3LYP coefficients for correlation

    d2f_dn2 = 0.81 * d2f_dn2_GGA + 0.19 * d2f_dn2_LDA

    d2f_dnds = 0.81 * d2f_dnds_GGA

    d2f_ds2 = None if d2f_ds2_GGA is None else 0.81 * d2f_ds2_GGA

    return d2f_dn2, d2f_dnds, d2f_ds2










def calculate_restricted_3P_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted three-parameter spin correlation kernel for triplet excitations.

    For B3PW91 and B3P86 the two gradient blocks are zero, as PW91 and P86 correlation only see the total square gradient.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_mm (array): Derivative with respect to the square spin density gradient

    """

    method = calculation.method.name

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V

    f_mm_LDA = correlation_spin_kernels["VWN3" if "G" in method else "VWN5"](density, sigma, tau, calculation)

    # Picks the GGA correlation kernel depending on the method

    if "LYP" in method: correlation_kernel = correlation_spin_kernels["LYP"]
    if "PW" in method: correlation_kernel = correlation_spin_kernels["PW91"]
    if "P86" in method: correlation_kernel = correlation_spin_kernels["P86"]

    # Calculates the kernel for the GGA part

    blocks = correlation_kernel(density, sigma, tau, calculation)

    # PW91 and P86 have no gradient blocks, so these are zeros

    if not isinstance(blocks, tuple):

        zeros = np.zeros_like(density)

        blocks = blocks, zeros, zeros

    f_mm_GGA, f_m_sigma_nm_GGA, f_sigma_mm_GGA = blocks

    # These parameters are the standard B3LYP coefficients for correlation

    f_mm = 0.81 * f_mm_GGA + 0.19 * f_mm_LDA
    f_m_sigma_nm = 0.81 * f_m_sigma_nm_GGA
    f_sigma_mm = 0.81 * f_sigma_mm_GGA

    return f_mm, f_m_sigma_nm, f_sigma_mm










def calculate_unrestricted_3P_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved three-parameter correlation kernel for an unrestricted reference.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma_aa (array): Alpha-alpha square density gradient
        sigma_bb (array): Beta-beta square density gradient
        sigma_ab (array): Alpha-beta square density gradient
        tau_alpha (array): Alpha kinetic energy density
        tau_beta (array): Beta kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        blocks (tuple): Blocks of the unrestricted GGA correlation kernel in the same order, combined with the local density kernel

    """

    method = calculation.method.name

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V

    f_aa_LDA, f_ab_LDA, f_bb_LDA = unrestricted_correlation_kernels["VWN3" if "G" in method else "VWN5"](alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    # Picks the GGA correlation kernel depending on the method

    if "LYP" in method: correlation_kernel = unrestricted_correlation_kernels["LYP"]
    if "PW" in method: correlation_kernel = unrestricted_correlation_kernels["PW91"]
    if "P86" in method: correlation_kernel = unrestricted_correlation_kernels["P86"]

    # Calculates the kernel for the GGA part

    blocks_GGA = correlation_kernel(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    f_aa_GGA, f_ab_GGA, f_bb_GGA = blocks_GGA[0], blocks_GGA[1], blocks_GGA[2]

    # These parameters are the standard B3LYP coefficients for correlation

    f_aa = 0.81 * f_aa_GGA + 0.19 * f_aa_LDA
    f_ab = 0.81 * f_ab_GGA + 0.19 * f_ab_LDA
    f_bb = 0.81 * f_bb_GGA + 0.19 * f_bb_LDA

    # Gradient blocks from the GGA part, whose number depends on the functional

    gradient_blocks = [0.81 * block for block in blocks_GGA[3:]]

    blocks = tuple([f_aa, f_ab, f_bb] + gradient_blocks)

    return blocks





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ D I C T I O N A R I E S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #





exchange_kernels = {

    "S": calculate_Slater_exchange_kernel,
    "PBE": calculate_PBE_exchange_kernel,
    "RPBE": calculate_RPBE_exchange_kernel,
    "REVPBE": calculate_PBE_exchange_kernel,
    "B": calculate_B88_exchange_kernel,
    "B3": calculate_B3_exchange_kernel,
    "PW": calculate_PW91_exchange_kernel,
    "MPW": calculate_mPW91_exchange_kernel,
    "B97": calculate_B97_exchange_kernel,

}










correlation_density_kernels = {

    "VWN3": calculate_restricted_VWN3_correlation_kernel,
    "VWN5": calculate_restricted_VWN5_correlation_kernel,
    "PW": calculate_restricted_PW_correlation_kernel,
    "PW91": calculate_restricted_PW91_correlation_kernel,
    "P86": calculate_restricted_P86_correlation_kernel,
    "PBE": calculate_restricted_PBE_correlation_kernel,
    "LYP": calculate_restricted_LYP_correlation_kernel,
    "3P": calculate_restricted_3P_correlation_kernel,
    "B97": calculate_restricted_B97_correlation_kernel,

}










correlation_spin_kernels = {

    "VWN3": calculate_restricted_VWN3_spin_correlation_kernel,
    "VWN5": calculate_restricted_VWN5_spin_correlation_kernel,
    "PW": calculate_restricted_PW_spin_correlation_kernel,
    "PW91": calculate_restricted_PW91_spin_correlation_kernel,
    "P86": calculate_restricted_P86_spin_correlation_kernel,
    "PBE": calculate_restricted_PBE_spin_correlation_kernel,
    "LYP": calculate_restricted_LYP_spin_correlation_kernel,
    "3P": calculate_restricted_3P_spin_correlation_kernel,
    "B97": calculate_restricted_B97_spin_correlation_kernel,

}










unrestricted_correlation_kernels = {

    "VWN3": calculate_unrestricted_VWN3_correlation_kernel,
    "VWN5": calculate_unrestricted_VWN5_correlation_kernel,
    "PW": calculate_unrestricted_PW_correlation_kernel,
    "PW91": calculate_unrestricted_PW91_correlation_kernel,
    "P86": calculate_unrestricted_P86_correlation_kernel,
    "PBE": calculate_unrestricted_PBE_correlation_kernel,
    "LYP": calculate_unrestricted_LYP_correlation_kernel,
    "3P": calculate_unrestricted_3P_correlation_kernel,
    "B97": calculate_unrestricted_B97_correlation_kernel,

}
