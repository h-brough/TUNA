import numpy as np
from numpy import ndarray
from TUNA.tuna_calc import Calculation
from TUNA.tuna_util import constants
from TUNA.tuna_xc import clean, calculate_zeta, calculate_f_zeta, calculate_f_prime_zeta, calculate_seitz_radius, calculate_Fermi_wavevector
from TUNA.tuna_xc import calculate_Slater_exchange, calculate_VWN_potential, calculate_PW_potential
from TUNA.tuna_xc import calculate_restricted_PW_correlation, calculate_unrestricted_PW_correlation
from TUNA.tuna_xc import calculate_unrestricted_TPSS_correlation, calculate_unrestricted_revTPSS_correlation, calculate_unrestricted_B97M_correlation


"""

This is the TUNA module for exchange-correlation kernels, written first for version 0.12.0.

The kernels are the second derivatives of the exchange-correlation energy density wrt. the (spin) density, the square density gradients and, for
meta-GGAs, the kinetic energy densities, which TD-DFT contracts with the transition density. The functionals and potentials the kernels are built
from are imported from tuna_xc, and the same conventions are used here, so ** (1 / 2) for square rooting and np.cbrt() for cube rooting.

Each kernel comes in a singlet form for a restricted reference, a triplet form for the same reference and a spin-resolved form for an unrestricted
one. Exchange spin scales exactly, so for exchange a single pair of generic functions rescales the restricted kernel into the other two.

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

    sigma = clean(sigma, floor = constants.sigma_floor)

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










def calculate_meta_GGA_exchange_kernel_blocks(density: ndarray, p: ndarray, dp_ds: ndarray, a: ndarray, da: tuple, d2a: tuple, F_X: ndarray, dF: tuple, d2F: tuple, f_LDA: ndarray) -> tuple:

    """

    Calculates the six second derivatives of a meta-GGA exchange energy from its enhancement factor, which depends on the reduced density gradient p
    and one kinetic energy density variable a.

    Args:
        density (array): Electron density on integration grid
        p (array): Reduced density gradient, sigma / (4 * (3 pi^2)^(2/3) * n^(8/3))
        dp_ds (array): Derivative of p with respect to sigma, which is independent of sigma
        a (array): Kinetic energy density variable of the functional
        da (tuple): First derivatives of a with respect to density, sigma and tau
        d2a (tuple): Second derivatives of a, ordered nn, ns, ss, nt, st, tt
        F_X (array): Exchange enhancement factor
        dF (tuple): First derivatives of the enhancement factor with respect to p and a
        d2F (tuple): Second derivatives of the enhancement factor, ordered pp, pa, aa
        f_LDA (array): Local density exchange energy per unit volume, n * e_X_LDA

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_X with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_X with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_X with respect to tau

    """

    da_dn, da_ds, da_dt = da
    d2a_dn2, d2a_dnds, d2a_ds2, d2a_dndt, d2a_dsdt, d2a_dt2 = d2a

    dF_dp, dF_da = dF
    d2F_dp2, d2F_dpda, d2F_da2 = d2F

    inv_density = 1 / density

    # Derivatives of p, which is proportional to sigma / n^(8/3)

    dp_dn = -(8 / 3) * p * inv_density
    d2p_dn2 = (88 / 9) * p * inv_density * inv_density
    d2p_dnds = -(8 / 3) * dp_ds * inv_density

    # Chain rule for the enhancement factor, where p does not depend on tau

    dF_dn = dF_dp * dp_dn + dF_da * da_dn
    dF_ds = dF_dp * dp_ds + dF_da * da_ds
    dF_dt = dF_da * da_dt

    d2F_dn2 = d2F_dp2 * dp_dn * dp_dn + 2 * d2F_dpda * dp_dn * da_dn + d2F_da2 * da_dn * da_dn + dF_dp * d2p_dn2 + dF_da * d2a_dn2
    d2F_dnds = d2F_dp2 * dp_dn * dp_ds + d2F_dpda * (dp_dn * da_ds + dp_ds * da_dn) + d2F_da2 * da_dn * da_ds + dF_dp * d2p_dnds + dF_da * d2a_dnds
    d2F_ds2 = d2F_dp2 * dp_ds * dp_ds + 2 * d2F_dpda * dp_ds * da_ds + d2F_da2 * da_ds * da_ds + dF_da * d2a_ds2
    d2F_dndt = d2F_dpda * dp_dn * da_dt + d2F_da2 * da_dn * da_dt + dF_da * d2a_dndt
    d2F_dsdt = d2F_dpda * dp_ds * da_dt + d2F_da2 * da_ds * da_dt + dF_da * d2a_dsdt
    d2F_dt2 = d2F_da2 * da_dt * da_dt + dF_da * d2a_dt2

    # Local density exchange goes as n^(4/3)

    df_LDA_dn = (4 / 3) * f_LDA * inv_density
    d2f_LDA_dn2 = (4 / 9) * f_LDA * inv_density * inv_density

    # Second derivatives of f = f_LDA * F_X with respect to the density, sigma and tau

    d2f_dn2 = d2f_LDA_dn2 * F_X + 2 * df_LDA_dn * dF_dn + f_LDA * d2F_dn2
    d2f_dnds = df_LDA_dn * dF_ds + f_LDA * d2F_dnds
    d2f_ds2 = f_LDA * d2F_ds2
    d2f_dndt = df_LDA_dn * dF_dt + f_LDA * d2F_dndt
    d2f_dsdt = f_LDA * d2F_dsdt
    d2f_dt2 = f_LDA * d2F_dt2

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_iso_orbital_indicator_derivatives(density: ndarray, sigma: ndarray, tau: ndarray, eta_constant: float = 0, eta_weizsacker: float = 0) -> tuple:

    """

    Calculates the iso-orbital indicator alpha = (tau - tau_W) / (tau_U + eta_constant + eta_weizsacker * tau_W) and its derivatives. The two constants
    are zero for TPSS and SCAN, eta_constant regularises rSCAN and eta_weizsacker regularises r2SCAN.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        eta_constant (float, optional): Constant added to the denominator
        eta_weizsacker (float, optional): Multiple of the Weizsacker kinetic energy density added to the denominator

    Returns:
        alpha (array): Iso-orbital indicator
        da (tuple): First derivatives with respect to density, sigma and tau
        d2a (tuple): Second derivatives, ordered nn, ns, ss, nt, st, tt

    """

    inv_density = 1 / density

    # The Weizsacker and uniform electron gas kinetic energy densities

    tau_W = sigma * inv_density / 8
    tau_U = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 5

    # Numerator and its derivatives, where tau only enters linearly

    N = tau - tau_W

    dN_dn = tau_W * inv_density
    dN_ds = -inv_density / 8
    d2N_dn2 = -2 * tau_W * inv_density * inv_density
    d2N_dnds = inv_density * inv_density / 8

    # Denominator and its derivatives, which do not depend on tau

    D = tau_U + eta_constant + eta_weizsacker * tau_W

    dD_dn = (5 / 3) * tau_U * inv_density - eta_weizsacker * tau_W * inv_density
    dD_ds = eta_weizsacker * inv_density / 8
    d2D_dn2 = (10 / 9) * tau_U * inv_density * inv_density + 2 * eta_weizsacker * tau_W * inv_density * inv_density
    d2D_dnds = -eta_weizsacker * inv_density * inv_density / 8

    inv_D = 1 / D

    alpha = N * inv_D

    # Derivatives of alpha, from differentiating N = alpha * D

    da_dn = (dN_dn - alpha * dD_dn) * inv_D
    da_ds = (dN_ds - alpha * dD_ds) * inv_D
    da_dt = inv_D

    d2a_dn2 = (d2N_dn2 - 2 * da_dn * dD_dn - alpha * d2D_dn2) * inv_D
    d2a_dnds = (d2N_dnds - da_dn * dD_ds - da_ds * dD_dn - alpha * d2D_dnds) * inv_D
    d2a_ds2 = -2 * da_ds * dD_ds * inv_D
    d2a_dndt = -da_dt * dD_dn * inv_D
    d2a_dsdt = -da_dt * dD_ds * inv_D
    d2a_dt2 = np.zeros_like(alpha)

    return alpha, (da_dn, da_ds, da_dt), (d2a_dn2, d2a_dnds, d2a_ds2, d2a_dndt, d2a_dsdt, d2a_dt2)










def calculate_SCAN_switching_derivatives(alpha: ndarray, c_1: float, c_2: float, d: float, c_polynomial: list = None) -> tuple:

    """

    Calculates the switching function of SCAN, or of rSCAN and r2SCAN when the polynomial coefficients are given, with its first two derivatives.

    Args:
        alpha (array): Iso-orbital indicator, or its regularised form
        c_1 (float): Exponent parameter below the switching region
        c_2 (float): Exponent parameter above the switching region
        d (float): Prefactor above the switching region
        c_polynomial (list, optional): Coefficients of the degree seven polynomial used between zero and 2.5

    Returns:
        f (array): Switching function
        df (array): First derivative with respect to alpha
        d2f (array): Second derivative with respect to alpha

    """

    # SCAN switches at alpha = 1, while the regularised functionals use a polynomial between zero and 2.5

    lower, upper = (1, 1) if c_polynomial is None else (0, 2.5)

    # Each exponential form is evaluated at a harmless indicator wherever it is not used, which avoids dividing by zero

    alpha_small = np.where(alpha < lower, alpha, 0)
    alpha_large = np.where(alpha > upper, alpha, upper + 1)

    # Below the switching region, f = exp(u) with u = -c_1 * alpha / (1 - alpha)

    inv_one_minus_alpha_small = 1 / (1 - alpha_small)

    f_small = np.exp(-c_1 * alpha_small * inv_one_minus_alpha_small)
    du_small = -c_1 * inv_one_minus_alpha_small * inv_one_minus_alpha_small
    d2u_small = 2 * du_small * inv_one_minus_alpha_small

    df_small = f_small * du_small
    d2f_small = f_small * (du_small * du_small + d2u_small)

    # Above the switching region, f = -d * exp(v) with v = c_2 / (1 - alpha)

    inv_one_minus_alpha_large = 1 / (1 - alpha_large)

    f_large = -d * np.exp(c_2 * inv_one_minus_alpha_large)
    dv_large = c_2 * inv_one_minus_alpha_large * inv_one_minus_alpha_large
    d2v_large = 2 * dv_large * inv_one_minus_alpha_large

    df_large = f_large * dv_large
    d2f_large = f_large * (dv_large * dv_large + d2v_large)

    # The switching function is exactly zero at alpha = 1 for SCAN, otherwise the polynomial and its derivatives by Horner's method

    f_middle, df_middle, d2f_middle = 0, 0, 0

    if c_polynomial is not None:

        for coefficient in reversed(c_polynomial):

            d2f_middle = d2f_middle * alpha + 2 * df_middle
            df_middle = df_middle * alpha + f_middle
            f_middle = f_middle * alpha + coefficient

    f = np.where(alpha < lower, f_small, np.where(alpha > upper, f_large, f_middle))
    df = np.where(alpha < lower, df_small, np.where(alpha > upper, df_large, df_middle))
    d2f = np.where(alpha < lower, d2f_small, np.where(alpha > upper, d2f_large, d2f_middle))

    return f, df, d2f










def calculate_TPSS_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted TPSS or revised TPSS exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_X with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_X with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_X with respect to tau

    """

    # Parameters that define TPSS - same kappa and mu as PBE

    b, c, e, kappa, mu = 0.40, 1.59096, 1.537, 0.804, 0.21951

    revised = calculation.functional.x_functional == "REVTPSS"

    if revised:

        # Revised TPSS changes three parameters, and the power of z in A

        c, e, mu = 2.35204, 2.1677, 0.14

    # Reduced density gradient, which is linear in sigma

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)
    p = sigma * dp_ds

    # The enhancement factor is written in terms of p and the iso-orbital indicator

    alpha, da, d2a = calculate_iso_orbital_indicator_derivatives(density, sigma, tau)

    # As tau / tau_U = alpha + 5 p / 3, both z = tau_W / tau and t = tau_U / tau are simple functions of p and alpha

    inv_D = 1 / (5 * p + 3 * alpha)
    inv_D_squared = inv_D * inv_D
    inv_D_cubed = inv_D_squared * inv_D

    z = 5 * p * inv_D
    dz_dp = 15 * alpha * inv_D_squared
    dz_da = -15 * p * inv_D_squared
    d2z_dp2 = -150 * alpha * inv_D_cubed
    d2z_dpda = (75 * p - 45 * alpha) * inv_D_cubed
    d2z_da2 = 90 * p * inv_D_cubed

    t = 3 * inv_D
    dt_dp = -15 * inv_D_squared
    dt_da = -9 * inv_D_squared
    d2t_dp2 = 150 * inv_D_cubed
    d2t_dpda = 90 * inv_D_cubed
    d2t_da2 = 54 * inv_D_cubed

    # As 3 z / 5 = p * t, the paper's S = sqrt(((3 z / 5)^2 + p^2) / 2) is p * R, which avoids a singular square root as sigma vanishes

    R = ((1 + t * t) / 2) ** (1 / 2)
    dR_dt = t / (2 * R)
    d2R_dt2 = 1 / (4 * R * R * R)

    dR_dp = dR_dt * dt_dp
    dR_da = dR_dt * dt_da
    d2R_dp2 = d2R_dt2 * dt_dp * dt_dp + dR_dt * d2t_dp2
    d2R_dpda = d2R_dt2 * dt_dp * dt_da + dR_dt * d2t_dpda
    d2R_da2 = d2R_dt2 * dt_da * dt_da + dR_dt * d2t_da2

    S = p * R
    dS_dp = R + p * dR_dp
    dS_da = p * dR_da
    d2S_dp2 = 2 * dR_dp + p * d2R_dp2
    d2S_dpda = dR_da + p * d2R_dpda
    d2S_da2 = p * d2R_da2

    # The q_tilde variable from the TPSS paper, which is linear in p

    Q_b = 1 + b * alpha * (alpha - 1)
    dQ_b = b * (2 * alpha - 1)
    d2Q_b = 2 * b

    inv_sqrt_Q_b = 1 / Q_b ** (1 / 2)
    inv_Q_b = 1 / Q_b

    h = (alpha - 1) * inv_sqrt_Q_b
    dh = inv_sqrt_Q_b - (1 / 2) * (alpha - 1) * dQ_b * inv_Q_b * inv_sqrt_Q_b
    d2h = inv_sqrt_Q_b * inv_Q_b * (-dQ_b + (3 / 4) * (alpha - 1) * dQ_b * dQ_b * inv_Q_b - (1 / 2) * (alpha - 1) * d2Q_b)

    q = (9 / 20) * h + 2 * p / 3
    dq_dp = 2 / 3
    dq_da = (9 / 20) * dh
    d2q_da2 = (9 / 20) * d2h

    # A = 10 / 81 + c * z^k / (1 + z^2)^2, with k = 2 for TPSS and k = 3 for revised TPSS

    k = 3 if revised else 2

    inv_one_plus_z_squared = 1 / (1 + z * z)

    u = z ** k
    du = k * z ** (k - 1)
    d2u = k * (k - 1) * z ** (k - 2)

    v = inv_one_plus_z_squared * inv_one_plus_z_squared
    dv = -4 * z * v * inv_one_plus_z_squared
    d2v = -4 * v * inv_one_plus_z_squared + 24 * z * z * v * inv_one_plus_z_squared * inv_one_plus_z_squared

    A = 10 / 81 + c * u * v
    dA_dz = c * (du * v + u * dv)
    d2A_dz2 = c * (d2u * v + 2 * du * dv + u * d2v)

    dA_dp = dA_dz * dz_dp
    dA_da = dA_dz * dz_da
    d2A_dp2 = d2A_dz2 * dz_dp * dz_dp + dA_dz * d2z_dp2
    d2A_dpda = d2A_dz2 * dz_dp * dz_da + dA_dz * d2z_dpda
    d2A_da2 = d2A_dz2 * dz_da * dz_da + dA_dz * d2z_da2

    # Coefficients of the terms in the numerator of x

    sqrt_e = e ** (1 / 2)

    c_qq = 146 / 2025
    c_qS = 73 / 405
    c_pp = (10 / 81) ** 2 / kappa
    c_zz = 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2
    c_ppp = e * mu

    # Horrible equations from TPSS paper, and their derivatives with respect to p and alpha

    num = A * p + c_qq * q * q - c_qS * q * S + c_pp * p * p + c_zz * z * z + c_ppp * p * p * p

    dnum_dp = dA_dp * p + A + 2 * c_qq * q * dq_dp - c_qS * (dq_dp * S + q * dS_dp) + 2 * c_pp * p + 2 * c_zz * z * dz_dp + 3 * c_ppp * p * p
    dnum_da = dA_da * p + 2 * c_qq * q * dq_da - c_qS * (dq_da * S + q * dS_da) + 2 * c_zz * z * dz_da

    d2num_dp2 = d2A_dp2 * p + 2 * dA_dp + 2 * c_qq * dq_dp * dq_dp - c_qS * (2 * dq_dp * dS_dp + q * d2S_dp2) + 2 * c_pp + 2 * c_zz * (dz_dp * dz_dp + z * d2z_dp2) + 6 * c_ppp * p
    d2num_dpda = d2A_dpda * p + dA_da + 2 * c_qq * dq_dp * dq_da - c_qS * (dq_dp * dS_da + dq_da * dS_dp + q * d2S_dpda) + 2 * c_zz * (dz_dp * dz_da + z * d2z_dpda)
    d2num_da2 = d2A_da2 * p + 2 * c_qq * (dq_da * dq_da + q * d2q_da2) - c_qS * (d2q_da2 * S + 2 * dq_da * dS_da + q * d2S_da2) + 2 * c_zz * (dz_da * dz_da + z * d2z_da2)

    den = (1 + sqrt_e * p) ** 2
    dden_dp = 2 * sqrt_e * (1 + sqrt_e * p)
    d2den_dp2 = 2 * e

    # Derivatives of x, from differentiating num = x * den

    x = num / den

    dx_dp = (dnum_dp - x * dden_dp) / den
    dx_da = dnum_da / den
    d2x_dp2 = (d2num_dp2 - 2 * dx_dp * dden_dp - x * d2den_dp2) / den
    d2x_dpda = (d2num_dpda - dx_da * dden_dp) / den
    d2x_da2 = d2num_da2 / den

    # Function for enhancement over local exchange, and its derivatives

    inv_kappa_plus_x = 1 / (kappa + x)

    F_X = 1 + kappa - kappa * kappa * inv_kappa_plus_x

    dF_dx = kappa * kappa * inv_kappa_plus_x * inv_kappa_plus_x
    d2F_dx2 = -2 * dF_dx * inv_kappa_plus_x

    dF_dp = dF_dx * dx_dp
    dF_da = dF_dx * dx_da
    d2F_dp2 = d2F_dx2 * dx_dp * dx_dp + dF_dx * d2x_dp2
    d2F_dpda = d2F_dx2 * dx_dp * dx_da + dF_dx * d2x_dpda
    d2F_da2 = d2F_dx2 * dx_da * dx_da + dF_dx * d2x_da2

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density, sigma and tau

    d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2 = calculate_meta_GGA_exchange_kernel_blocks(density, p, dp_ds, alpha, da, d2a, F_X, (dF_dp, dF_da), (d2F_dp2, d2F_dpda, d2F_da2), density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_SCAN_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted SCAN, rSCAN or r2SCAN exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_X with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_X with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_X with respect to tau

    """

    functional = calculation.functional.x_functional

    # Parameters shared by the SCAN family

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

    # Coefficients of the smoother switching function of rSCAN and r2SCAN

    c_x = [1, -0.667, -0.4445555, -0.663086601049, 1.451297044490, -0.887998041597, 0.234528941479, -0.023185843322]

    zeros = np.zeros_like(density)

    # Reduced density gradient, which is linear in sigma

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)
    p = sigma * dp_ds

    # The iso-orbital indicator, or its regularised versions, with derivatives

    if functional == "RSCAN":

        alpha_tilde, da_tilde, d2a_tilde = calculate_iso_orbital_indicator_derivatives(density, sigma, tau, eta_constant=0.0001)

        # The regularised indicator is alpha_tilde^3 / (alpha_tilde^2 + 0.001), so the chain rule is applied to each derivative

        alpha_r = 0.001

        alpha_tilde_squared = alpha_tilde * alpha_tilde
        inv_denominator = 1 / (alpha_tilde_squared + alpha_r)

        alpha = alpha_tilde_squared * alpha_tilde * inv_denominator

        dalpha = alpha_tilde_squared * (alpha_tilde_squared + 3 * alpha_r) * inv_denominator * inv_denominator
        d2alpha = 2 * alpha_r * alpha_tilde * (3 * alpha_r - alpha_tilde_squared) * inv_denominator * inv_denominator * inv_denominator

        da = tuple(dalpha * derivative for derivative in da_tilde)
        d2a = tuple(d2alpha * da_tilde[i] * da_tilde[j] + dalpha * d2a_tilde[k] for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]))

    else:

        alpha, da, d2a = calculate_iso_orbital_indicator_derivatives(density, sigma, tau, eta_weizsacker=0.001 if functional == "R2SCAN" else 0)

    # The r2SCAN form of x only depends on p, and is very different from the original SCAN

    if functional == "R2SCAN":

        C_eta = 20 / 27 + 0.001 * 5 / 3
        C_2 = sum(i * c_x[i] * k_0 for i in range(1, 8))
        d = 0.361

        x_exponent_term = np.exp(-p * p / d ** 4)

        x = (C_eta * C_2 * x_exponent_term + mu) * p

        dx_dp = mu + C_eta * C_2 * x_exponent_term * (1 - 2 * p * p / d ** 4)
        dx_da = zeros
        d2x_dp2 = C_eta * C_2 * x_exponent_term * (-2 * p / d ** 4) * (3 - 2 * p * p / d ** 4)
        d2x_dpda = zeros
        d2x_da2 = zeros

    else:

        # First term of x, mu * p * (1 + y * exp(-y)) with y = b_4 * p / mu

        k_p = b_4 / mu

        p_exponent_term = np.exp(-k_p * p)

        x_1 = mu * p + mu * k_p * p * p * p_exponent_term
        dx_1_dp = mu + mu * k_p * (2 * p - k_p * p * p) * p_exponent_term
        d2x_1_dp2 = mu * k_p * (2 - 4 * k_p * p + k_p * k_p * p * p) * p_exponent_term

        # Second term of x, W^2 with W = b_1 * p + b_2 * (1 - alpha) * exp(-b_3 * (1 - alpha)^2)

        one_minus_alpha = 1 - alpha

        alpha_exponent_term = np.exp(-b_3 * one_minus_alpha * one_minus_alpha)

        W = b_1 * p + b_2 * one_minus_alpha * alpha_exponent_term
        dW_da = -b_2 * alpha_exponent_term * (1 - 2 * b_3 * one_minus_alpha * one_minus_alpha)
        d2W_da2 = -2 * b_2 * b_3 * one_minus_alpha * alpha_exponent_term * (3 - 2 * b_3 * one_minus_alpha * one_minus_alpha)

        x = x_1 + W * W

        dx_dp = dx_1_dp + 2 * W * b_1
        dx_da = 2 * W * dW_da
        d2x_dp2 = d2x_1_dp2 + 2 * b_1 * b_1
        d2x_dpda = 2 * b_1 * dW_da
        d2x_da2 = 2 * (dW_da * dW_da + W * d2W_da2)

    # Interpolation limits, h_1 as a function of x

    h_0 = 1 + k_0

    inv_k_1_plus_x = 1 / (k_1 + x)

    h_1 = 1 + k_1 - k_1 * k_1 * inv_k_1_plus_x

    dh_1_dx = k_1 * k_1 * inv_k_1_plus_x * inv_k_1_plus_x
    d2h_1_dx2 = -2 * dh_1_dx * inv_k_1_plus_x

    dh_1_dp = dh_1_dx * dx_dp
    dh_1_da = dh_1_dx * dx_da
    d2h_1_dp2 = d2h_1_dx2 * dx_dp * dx_dp + dh_1_dx * d2x_dp2
    d2h_1_dpda = d2h_1_dx2 * dx_dp * dx_da + dh_1_dx * d2x_dpda
    d2h_1_da2 = d2h_1_dx2 * dx_da * dx_da + dh_1_dx * d2x_da2

    # Switching function, which is the original one for SCAN and the smoother polynomial otherwise

    f_x, df_x, d2f_x = calculate_SCAN_switching_derivatives(alpha, c_1, c_2, d_x, None if functional == "SCAN" else c_x)

    # The g_x factor and its derivatives with respect to p

    g_exponent_term = np.exp(-a_1 / p ** (1 / 4))

    g_x = 1 - g_exponent_term
    dg_x_dp = -g_exponent_term * (a_1 / 4) * p ** (-5 / 4)
    d2g_x_dp2 = -g_exponent_term * ((a_1 * a_1 / 16) * p ** (-5 / 2) - (5 * a_1 / 16) * p ** (-9 / 4))

    # Interpolated factor, h_1 + f_x * (h_0 - h_1)

    H = h_1 * (1 - f_x) + h_0 * f_x

    dH_dp = dh_1_dp * (1 - f_x)
    dH_da = dh_1_da * (1 - f_x) + (h_0 - h_1) * df_x
    d2H_dp2 = d2h_1_dp2 * (1 - f_x)
    d2H_dpda = d2h_1_dpda * (1 - f_x) - dh_1_dp * df_x
    d2H_da2 = d2h_1_da2 * (1 - f_x) - 2 * dh_1_da * df_x + (h_0 - h_1) * d2f_x

    # The exchange enhancement factor and its derivatives

    F_X = H * g_x

    dF_dp = dH_dp * g_x + H * dg_x_dp
    dF_da = dH_da * g_x
    d2F_dp2 = d2H_dp2 * g_x + 2 * dH_dp * dg_x_dp + H * d2g_x_dp2
    d2F_dpda = d2H_dpda * g_x + dH_da * dg_x_dp
    d2F_da2 = d2H_da2 * g_x

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density, sigma and tau

    d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2 = calculate_meta_GGA_exchange_kernel_blocks(density, p, dp_ds, alpha, da, d2a, F_X, (dF_dp, dF_da), (d2F_dp2, d2F_dpda, d2F_da2), density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_B97M_exchange_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97M exchange kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_X with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_X with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_X with respect to tau

    """

    # The empirical parameters for Head-Gordon's meta-GGA

    c_x = [1, 0.416, 1.308, 3.07, 1.901]

    gamma = 0.004

    zeros = np.zeros_like(density)

    # Reduced density gradient, which is linear in sigma

    dp_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)
    p = sigma * dp_ds

    # The cube root four makes Becke's s^2 = cbrt(4) * sigma / n^(8/3) match the TUNA treatment of exchange spin scaling, which is a multiple of p

    gamma_p = gamma * np.cbrt(4) * 4 * np.cbrt(3 * np.pi ** 2) ** 2

    inv_one_plus_gamma_p = 1 / (1 + gamma_p * p)

    x = gamma_p * p * inv_one_plus_gamma_p
    dx_dp = gamma_p * inv_one_plus_gamma_p * inv_one_plus_gamma_p
    d2x_dp2 = -2 * gamma_p * dx_dp * inv_one_plus_gamma_p

    # Ratio of the uniform electron gas to the actual kinetic energy density, and its derivatives

    inv_density = 1 / density
    inv_tau = 1 / tau

    t = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 5 * inv_tau

    dt_dn = (5 / 3) * t * inv_density
    dt_dt = -t * inv_tau
    d2t_dn2 = (10 / 9) * t * inv_density * inv_density
    d2t_dndt = -(5 / 3) * t * inv_density * inv_tau
    d2t_dt2 = 2 * t * inv_tau * inv_tau

    # The finite range variable w = (t - 1) / (t + 1) is the kinetic energy density variable of the enhancement factor

    inv_t_plus_one = 1 / (t + 1)

    w = (t - 1) * inv_t_plus_one
    dw_dtt = 2 * inv_t_plus_one * inv_t_plus_one
    d2w_dtt2 = -2 * dw_dtt * inv_t_plus_one

    dw = (dw_dtt * dt_dn, zeros, dw_dtt * dt_dt)
    d2w = (d2w_dtt2 * dt_dn * dt_dn + dw_dtt * d2t_dn2, zeros, zeros, d2w_dtt2 * dt_dn * dt_dt + dw_dtt * d2t_dndt, zeros, d2w_dtt2 * dt_dt * dt_dt + dw_dtt * d2t_dt2)

    # Enhancement factor, c_0 + c_1 * w + (c_2 + c_3 * w + c_4 * x) * x, and its derivatives

    F_X = c_x[0] + c_x[1] * w + (c_x[2] + c_x[3] * w + c_x[4] * x) * x

    dF_dx = c_x[2] + c_x[3] * w + 2 * c_x[4] * x

    dF_dp = dF_dx * dx_dp
    dF_dw = c_x[1] + c_x[3] * x
    d2F_dp2 = 2 * c_x[4] * dx_dp * dx_dp + dF_dx * d2x_dp2
    d2F_dpdw = c_x[3] * dx_dp
    d2F_dw2 = zeros

    # Local density exchange

    _, _, _, e_X_LDA = calculate_Slater_exchange(density, sigma, tau, calculation)

    # Second derivatives with respect to the density, sigma and tau

    d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2 = calculate_meta_GGA_exchange_kernel_blocks(density, p, dp_ds, w, dw, d2w, F_X, (dF_dp, dF_dw), (d2F_dp2, d2F_dpdw, d2F_dw2), density * e_X_LDA)

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_restricted_meta_GGA_exchange_kernel_blocks(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Looks up and evaluates the restricted exchange kernel for whichever exchange functional the calculation selects, with all six meta-GGA blocks.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_X with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_X with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_X with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_X with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_X with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_X with respect to tau

    """

    # Picks the exchange kernel depending on the functional

    blocks = exchange_kernels[calculation.functional.x_functional](density, sigma, tau, calculation)

    zeros = np.zeros_like(density)

    # Slater exchange has no gradient or kinetic energy density blocks, so these are zeros

    if not isinstance(blocks, tuple):

        return blocks, zeros, zeros, zeros, zeros, zeros

    # Any block that vanishes identically is replaced by zeros, as are the kinetic energy density blocks that a GGA kernel does not return

    blocks = tuple(zeros if block is None else block for block in blocks)

    d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2 = blocks + (zeros,) * (6 - len(blocks))

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_meta_GGA_exchange_spin_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted exchange spin kernel for triplet excitations, for any meta-GGA exchange functional.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_X with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_nm_sigma_nm (array): Second derivative with respect to grad(n) . grad(m)
        f_m_tau_m (array): Mixed second derivative with respect to the spin density and the spin kinetic energy density, tau_alpha - tau_beta
        f_sigma_nm_tau_m (array): Mixed second derivative with respect to grad(n) . grad(m) and the spin kinetic energy density
        f_tau_m_tau_m (array): Second derivative with respect to the spin kinetic energy density

    """

    # Exchange spin scales exactly, so the triplet kernel is a rescaled singlet kernel

    d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2 = calculate_restricted_meta_GGA_exchange_kernel_blocks(density, sigma, tau, calculation)

    f_mm = d2f_dn2
    f_m_sigma_nm = 2 * d2f_dnds
    f_sigma_nm_sigma_nm = 4 * d2f_ds2

    # The spin kinetic energy density enters each channel as tau does, with no extra factors

    f_m_tau_m = d2f_dndt
    f_sigma_nm_tau_m = 2 * d2f_dsdt
    f_tau_m_tau_m = d2f_dt2

    return f_mm, f_m_sigma_nm, f_sigma_nm_sigma_nm, f_m_tau_m, f_sigma_nm_tau_m, f_tau_m_tau_m










def calculate_unrestricted_meta_GGA_exchange_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved exchange kernel for an unrestricted reference, for any meta-GGA exchange functional.

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
        f_X_alpha_tau_alpha (array): Mixed second derivative with respect to the alpha density and tau alpha
        f_X_beta_tau_beta (array): Mixed second derivative with respect to the beta density and tau beta
        f_X_sigma_aa_tau_alpha (array): Mixed second derivative with respect to sigma alpha-alpha and tau alpha
        f_X_sigma_bb_tau_beta (array): Mixed second derivative with respect to sigma beta-beta and tau beta
        f_X_tau_alpha_tau_alpha (array): Second derivative with respect to tau alpha
        f_X_tau_beta_tau_beta (array): Second derivative with respect to tau beta

    """

    # By spin scaling, each channel is the restricted kernel at twice the spin density and kinetic energy density, and four times the square gradient

    d2f_dn2_alpha, d2f_dnds_alpha, d2f_ds2_alpha, d2f_dndt_alpha, d2f_dsdt_alpha, d2f_dt2_alpha = calculate_restricted_meta_GGA_exchange_kernel_blocks(2 * alpha_density, 4 * sigma_aa, 2 * tau_alpha, calculation)

    d2f_dn2_beta, d2f_dnds_beta, d2f_ds2_beta, d2f_dndt_beta, d2f_dsdt_beta, d2f_dt2_beta = calculate_restricted_meta_GGA_exchange_kernel_blocks(2 * beta_density, 4 * sigma_bb, 2 * tau_beta, calculation)

    # Chain rule factors from spin scaling - blocks mixing alpha and beta vanish

    f_X_alpha_alpha = 2 * d2f_dn2_alpha
    f_X_beta_beta = 2 * d2f_dn2_beta

    f_X_alpha_sigma_aa = 4 * d2f_dnds_alpha
    f_X_beta_sigma_bb = 4 * d2f_dnds_beta

    f_X_sigma_aa_sigma_aa = 8 * d2f_ds2_alpha
    f_X_sigma_bb_sigma_bb = 8 * d2f_ds2_beta

    f_X_alpha_tau_alpha = 2 * d2f_dndt_alpha
    f_X_beta_tau_beta = 2 * d2f_dndt_beta

    f_X_sigma_aa_tau_alpha = 4 * d2f_dsdt_alpha
    f_X_sigma_bb_tau_beta = 4 * d2f_dsdt_beta

    f_X_tau_alpha_tau_alpha = 2 * d2f_dt2_alpha
    f_X_tau_beta_tau_beta = 2 * d2f_dt2_beta

    return f_X_alpha_alpha, f_X_beta_beta, f_X_alpha_sigma_aa, f_X_beta_sigma_bb, f_X_sigma_aa_sigma_aa, f_X_sigma_bb_sigma_bb, f_X_alpha_tau_alpha, f_X_beta_tau_beta, f_X_sigma_aa_tau_alpha, f_X_sigma_bb_tau_beta, f_X_tau_alpha_tau_alpha, f_X_tau_beta_tau_beta





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










def calculate_PW_correlation_third_derivative(density: ndarray, A: float, alpha_1: float, beta_1: float, beta_2: float, beta_3: float, beta_4: float, P: float) -> ndarray:

    """

    Calculates the third derivative of a single PW92 correlation fit with respect to the Seitz radius, which the r2SCAN kernel needs.

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
        d3e_C_dr3 (array): Third derivative of the energy density with respect to the Seitz radius

    """

    r_s, _ = calculate_seitz_radius(density)

    sqrt_r_s = r_s ** (1 / 2)

    # Intermediate quantities for PW92 LDA correlation, and their Seitz-radius derivatives

    Q_0 = -2 * A * (1 + alpha_1 * r_s)
    dQ_0 = -2 * A * alpha_1

    Q_1 = 2 * A * (beta_1 * sqrt_r_s + beta_2 * r_s + beta_3 * r_s * sqrt_r_s + beta_4 * r_s ** (P + 1))
    Q_1_prime = A * (beta_1 / sqrt_r_s + 2 * beta_2 + 3 * beta_3 * sqrt_r_s + 2 * (P + 1) * beta_4 * r_s ** P)
    Q_1_prime_prime = A * (-beta_1 / (2 * r_s * sqrt_r_s) + (3 / 2) * beta_3 / sqrt_r_s + 2 * P * (P + 1) * beta_4 * r_s ** (P - 1))
    Q_1_third = A * ((3 / 4) * beta_1 / (r_s * r_s * sqrt_r_s) - (3 / 4) * beta_3 / (r_s * sqrt_r_s) + 2 * P * (P + 1) * (P - 1) * beta_4 * r_s ** (P - 2))

    # The logged term L = log(1 + 1 / Q_1) has first derivative -Q_1' / D, with D = Q_1^2 + Q_1

    D = Q_1 * Q_1 + Q_1
    dD = Q_1_prime * (2 * Q_1 + 1)
    d2D = Q_1_prime_prime * (2 * Q_1 + 1) + 2 * Q_1_prime * Q_1_prime

    d2L = -Q_1_prime_prime / D + Q_1_prime * dD / (D * D)
    d3L = -Q_1_third / D + (2 * Q_1_prime_prime * dD + Q_1_prime * d2D) / (D * D) - 2 * Q_1_prime * dD * dD / (D * D * D)

    # The energy density is Q_0 * L, and Q_0 is linear in the Seitz radius

    d3e_C_dr3 = 3 * dQ_0 * d2L + Q_0 * d3L

    return d3e_C_dr3










def calculate_unrestricted_SCAN_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved SCAN, rSCAN or r2SCAN correlation kernel for an unrestricted reference. As for PBE, the blocks are with respect to
    the total square gradient, sigma = sigma_aa + sigma_bb + 2 * sigma_ab, and likewise the total kinetic energy density, tau = tau_alpha + tau_beta.

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
        f_C_alpha_sigma (array): Mixed second derivative with respect to the alpha density and sigma
        f_C_beta_sigma (array): Mixed second derivative with respect to the beta density and sigma
        f_C_sigma_sigma (array): Second derivative with respect to sigma
        f_C_alpha_tau (array): Mixed second derivative with respect to the alpha density and tau
        f_C_beta_tau (array): Mixed second derivative with respect to the beta density and tau
        f_C_sigma_tau (array): Mixed second derivative with respect to sigma and tau
        f_C_tau_tau (array): Second derivative with respect to tau

    """

    functional = calculation.functional.c_functional

    b_1c = 0.0285764
    b_2c = 0.0889
    b_3c = 0.125541

    c_1c = 0.64
    c_2c = 1.5
    d_c = 0.7

    # These are the coefficients for the smoother switching function of rSCAN and r2SCAN, a degree seven polynomial

    c_c = [1, -0.64, -0.4352, -1.535685604549, 3.061560252175, -1.915710236206, 0.516884468372, -0.051848879792]

    # The r2SCAN paper uses a more precise gamma, and derives the value of chi at infinity

    gamma = 0.031091
    chi_infinity = 0.128026

    if functional == "R2SCAN":

        gamma = 0.0310907
        chi_infinity = np.cbrt(3 * np.pi ** 2 / 16) ** 2 * 0.066725 / (1.778 * (0.9 - 3 * np.cbrt(3 / (16 * np.pi)) ** 2))

    inv_density = 1 / density
    zeros = np.zeros_like(density)

    # These functionals only depend on the total square gradient and kinetic energy density

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    # Local spin density correlation and its derivatives with respect to density and spin polarisation

    e_L, de_L_dn, de_L_dzeta, d2e_L_dn2, d2e_L_dndzeta, d2e_L_dzeta2, zeta = calculate_PW_spin_correlation_derivatives(alpha_density, beta_density, density)

    # Spin scaling functions and their derivatives with respect to spin polarisation

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus) / 2
    dphi = (1 / cbrt_plus - 1 / cbrt_minus) / 3
    d2phi = -(1 / 9) * (1 / cbrt_plus ** 4 + 1 / cbrt_minus ** 4)

    d_s = (cbrt_plus ** 5 + cbrt_minus ** 5) / 2
    dd_s = (5 / 6) * (cbrt_plus * cbrt_plus - cbrt_minus * cbrt_minus)
    d2d_s = (5 / 9) * (1 / cbrt_plus + 1 / cbrt_minus)

    d_x = (cbrt_plus ** 4 + cbrt_minus ** 4) / 2
    dd_x = (2 / 3) * (cbrt_plus - cbrt_minus)
    d2d_x = (2 / 9) * (1 / (cbrt_plus * cbrt_plus) + 1 / (cbrt_minus * cbrt_minus))

    phi_cubed = phi * phi * phi
    dphi_cubed = 3 * phi * phi * dphi
    d2phi_cubed = 6 * phi * dphi * dphi + 3 * phi * phi * d2phi

    G_c_1 = 1 - 2.3631 * (d_x - 1)
    G_c_2 = 1 - zeta ** 12

    G_c = G_c_1 * G_c_2
    dG_c = -2.3631 * dd_x * G_c_2 - 12 * zeta ** 11 * G_c_1
    d2G_c = -2.3631 * d2d_x * G_c_2 + 24 * 2.3631 * dd_x * zeta ** 11 - 132 * zeta ** 10 * G_c_1

    # Seitz radius and its density derivatives

    r_s, _ = calculate_seitz_radius(density)

    dr_dn = -r_s * inv_density / 3
    d2r_dn2 = (4 / 9) * r_s * inv_density * inv_density

    # W is the argument of the exponential in w_1 = exp(-W) - 1

    gamma_phi_cubed = gamma * phi_cubed

    W = e_L / gamma_phi_cubed
    dW_dn = de_L_dn / gamma_phi_cubed
    dW_dzeta = (de_L_dzeta - e_L * dphi_cubed / phi_cubed) / gamma_phi_cubed
    d2W_dn2 = d2e_L_dn2 / gamma_phi_cubed
    d2W_dndzeta = (d2e_L_dndzeta - de_L_dn * dphi_cubed / phi_cubed) / gamma_phi_cubed
    d2W_dzeta2 = (d2e_L_dzeta2 - 2 * de_L_dzeta * dphi_cubed / phi_cubed - e_L * d2phi_cubed / phi_cubed + 2 * e_L * dphi_cubed * dphi_cubed / (phi_cubed * phi_cubed)) / gamma_phi_cubed

    # Expm1 here is more stable than exp - 1

    exp_minus_W = np.exp(-W)

    w_1 = np.expm1(-W)
    dw1_dn = -exp_minus_W * dW_dn
    dw1_dzeta = -exp_minus_W * dW_dzeta
    d2w1_dn2 = exp_minus_W * (dW_dn * dW_dn - d2W_dn2)
    d2w1_dndzeta = exp_minus_W * (dW_dn * dW_dzeta - d2W_dndzeta)
    d2w1_dzeta2 = exp_minus_W * (dW_dzeta * dW_dzeta - d2W_dzeta2)

    # Density-dependent beta and its derivatives

    beta_denominator = 1 + 0.1778 * r_s

    beta = 0.066725 * (1 + 0.1 * r_s) / beta_denominator
    dbeta_dr = 0.066725 * (0.1 - 0.1778) / (beta_denominator * beta_denominator)
    d2beta_dr2 = -2 * 0.1778 * dbeta_dr / beta_denominator

    dbeta_dn = dbeta_dr * dr_dn
    d2beta_dn2 = d2beta_dr2 * dr_dn * dr_dn + dbeta_dr * d2r_dn2

    # Reduced density gradient t^2, which is linear in sigma

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * phi * phi * k_F * density * density)
    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    dT_dzeta = -2 * T * dphi / phi
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dndzeta = -2 * dT_dn * dphi / phi
    d2T_dzeta2 = T * (6 * dphi * dphi / (phi * phi) - 2 * d2phi / phi)
    d2T_dnds = -(7 / 3) * dT_ds * inv_density
    d2T_dzetads = -2 * dT_ds * dphi / phi

    # K = beta / (gamma w_1), from the derivatives of 1 / w_1

    inv_w_1 = 1 / w_1
    inv_w_1_squared = inv_w_1 * inv_w_1

    dinv_w1_dn = -dw1_dn * inv_w_1_squared
    dinv_w1_dzeta = -dw1_dzeta * inv_w_1_squared
    d2inv_w1_dn2 = (2 * dw1_dn * dw1_dn * inv_w_1 - d2w1_dn2) * inv_w_1_squared
    d2inv_w1_dndzeta = (2 * dw1_dn * dw1_dzeta * inv_w_1 - d2w1_dndzeta) * inv_w_1_squared
    d2inv_w1_dzeta2 = (2 * dw1_dzeta * dw1_dzeta * inv_w_1 - d2w1_dzeta2) * inv_w_1_squared

    K = beta * inv_w_1 / gamma
    dK_dn = (dbeta_dn * inv_w_1 + beta * dinv_w1_dn) / gamma
    dK_dzeta = beta * dinv_w1_dzeta / gamma
    d2K_dn2 = (d2beta_dn2 * inv_w_1 + 2 * dbeta_dn * dinv_w1_dn + beta * d2inv_w1_dn2) / gamma
    d2K_dndzeta = (dbeta_dn * dinv_w1_dzeta + beta * d2inv_w1_dndzeta) / gamma
    d2K_dzeta2 = beta * d2inv_w1_dzeta2 / gamma

    # y = K * t^2, with derivatives in density, zeta and sigma

    y = K * T

    dy_dn = dK_dn * T + K * dT_dn
    dy_dzeta = dK_dzeta * T + K * dT_dzeta
    dy_ds = K * dT_ds
    d2y_dn2 = d2K_dn2 * T + 2 * dK_dn * dT_dn + K * d2T_dn2
    d2y_dndzeta = d2K_dndzeta * T + dK_dn * dT_dzeta + dK_dzeta * dT_dn + K * d2T_dndzeta
    d2y_dzeta2 = d2K_dzeta2 * T + 2 * dK_dzeta * dT_dzeta + K * d2T_dzeta2
    d2y_dnds = dK_dn * dT_ds + K * d2T_dnds
    d2y_dzetads = dK_dzeta * dT_ds + K * d2T_dzetads
    d2y_ds2 = zeros

    # The zeta-independent LDA0 correlation and its Seitz radius derivatives

    sqrt_r_s = r_s ** (1 / 2)

    LDA_0_denominator = 1 + b_2c * sqrt_r_s + b_3c * r_s
    dLDA_0_denominator = b_2c / (2 * sqrt_r_s) + b_3c
    d2LDA_0_denominator = -b_2c / (4 * r_s * sqrt_r_s)

    e_0 = -b_1c / LDA_0_denominator
    de0_dr = b_1c * dLDA_0_denominator / LDA_0_denominator ** 2
    d2e0_dr2 = b_1c * (d2LDA_0_denominator / LDA_0_denominator ** 2 - 2 * dLDA_0_denominator ** 2 / LDA_0_denominator ** 3)

    de0_dn = de0_dr * dr_dn
    d2e0_dn2 = d2e0_dr2 * dr_dn * dr_dn + de0_dr * d2r_dn2

    # Reduced density gradient s^2, which is linear in sigma

    ds2_ds = 1 / (4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)
    s2 = sigma * ds2_ds

    ds2_dn = -(8 / 3) * s2 * inv_density
    d2s2_dn2 = (88 / 9) * s2 * inv_density * inv_density
    d2s2_dnds = -(8 / 3) * ds2_ds * inv_density

    if functional == "R2SCAN":

        # The r2SCAN correction to y needs the Seitz radius derivatives of both local correlation energies at fixed spin polarisation

        eta = 0.001
        d_p = 0.361

        delta_f_c = sum(i * c_c[i] for i in range(1, 8))

        # Paramagnetic, ferromagnetic and spin-stiffness PW92 fits with Seitz radius derivatives up to third order

        parameters_0 = (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
        parameters_1 = (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
        parameters_alpha = (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

        _, de_P0_dr, d2e_P0_dr2 = calculate_PW_correlation_derivatives(density, *parameters_0)
        _, de_P1_dr, d2e_P1_dr2 = calculate_PW_correlation_derivatives(density, *parameters_1)
        _, dm_dr, d2m_dr2 = calculate_PW_correlation_derivatives(density, *parameters_alpha)

        d3e_P0_dr3 = calculate_PW_correlation_third_derivative(density, *parameters_0)
        d3e_P1_dr3 = calculate_PW_correlation_third_derivative(density, *parameters_1)
        d3m_dr3 = calculate_PW_correlation_third_derivative(density, *parameters_alpha)

        # Spin interpolation functions of PW92 and their spin polarisation derivatives

        f_denominator = np.cbrt(2) ** 4 - 2
        f_prime_prime_at_zero = 8 / (9 * f_denominator)

        f_zeta = (cbrt_plus ** 4 + cbrt_minus ** 4 - 2) / f_denominator
        df_zeta = (4 / 3) * (cbrt_plus - cbrt_minus) / f_denominator
        d2f_zeta = (4 / 9) * (1 / (cbrt_plus * cbrt_plus) + 1 / (cbrt_minus * cbrt_minus)) / f_denominator

        zeta_2 = zeta * zeta
        zeta_3 = zeta_2 * zeta
        zeta_4 = zeta_3 * zeta

        h = f_zeta * zeta_4
        dh = df_zeta * zeta_4 + 4 * f_zeta * zeta_3
        d2h = d2f_zeta * zeta_4 + 8 * df_zeta * zeta_3 + 12 * f_zeta * zeta_2

        g_s = f_zeta * (1 - zeta_4) / f_prime_prime_at_zero
        dg_s = (df_zeta * (1 - zeta_4) - 4 * f_zeta * zeta_3) / f_prime_prime_at_zero
        d2g_s = (d2f_zeta * (1 - zeta_4) - 8 * df_zeta * zeta_3 - 12 * f_zeta * zeta_2) / f_prime_prime_at_zero

        # Seitz radius derivatives of e_L = e_0 - minus_alpha * g + (e_1 - e_0) * h at fixed zeta, and their zeta derivatives

        E_1 = de_P0_dr - dm_dr * g_s + (de_P1_dr - de_P0_dr) * h
        E_2 = d2e_P0_dr2 - d2m_dr2 * g_s + (d2e_P1_dr2 - d2e_P0_dr2) * h
        E_3 = d3e_P0_dr3 - d3m_dr3 * g_s + (d3e_P1_dr3 - d3e_P0_dr3) * h

        dE_1_dzeta = -dm_dr * dg_s + (de_P1_dr - de_P0_dr) * dh
        d2E_1_dzeta2 = -dm_dr * d2g_s + (de_P1_dr - de_P0_dr) * d2h
        dE_2_dzeta = -d2m_dr2 * dg_s + (d2e_P1_dr2 - d2e_P0_dr2) * dh

        # Third Seitz radius derivative of the LDA0 correlation

        d3LDA_0_denominator = (3 / 8) * b_2c / (r_s * r_s * sqrt_r_s)
        d3e0_dr3 = b_1c * (d3LDA_0_denominator / LDA_0_denominator ** 2 - 6 * dLDA_0_denominator * d2LDA_0_denominator / LDA_0_denominator ** 3 + 6 * dLDA_0_denominator ** 3 / LDA_0_denominator ** 4)

        # Differences between the local correlation of the LDA0 limit, G_c * e_0, and of the LSDA limit, and their Seitz radius derivatives

        B_0 = G_c * e_0 - e_L
        B_1 = G_c * de0_dr - E_1
        B_2 = G_c * d2e0_dr2 - E_2
        B_3 = G_c * d3e0_dr3 - E_3

        dB_0_dzeta = dG_c * e_0 - de_L_dzeta
        d2B_0_dzeta2 = d2G_c * e_0 - d2e_L_dzeta2
        dB_1_dzeta = dG_c * de0_dr - dE_1_dzeta
        d2B_1_dzeta2 = d2G_c * de0_dr - d2E_1_dzeta2
        dB_2_dzeta = dG_c * d2e0_dr2 - dE_2_dzeta

        # The bracket B = 20 r_s B_1 - 45 eta B_0 of the r2SCAN paper, with its derivatives in Seitz radius and zeta

        B = 20 * r_s * B_1 - 45 * eta * B_0
        dB_dr = 20 * B_1 + 20 * r_s * B_2 - 45 * eta * B_1
        d2B_dr2 = 40 * B_2 + 20 * r_s * B_3 - 45 * eta * B_2
        dB_dzeta = 20 * r_s * dB_1_dzeta - 45 * eta * dB_0_dzeta
        d2B_dzeta2 = 20 * r_s * d2B_1_dzeta2 - 45 * eta * d2B_0_dzeta2
        d2B_drdzeta = 20 * dB_1_dzeta + 20 * r_s * dB_2_dzeta - 45 * eta * dB_1_dzeta

        dB_dn = dB_dr * dr_dn
        d2B_dn2 = d2B_dr2 * dr_dn * dr_dn + dB_dr * d2r_dn2
        d2B_dndzeta = d2B_drdzeta * dr_dn

        # S = s^2 exp(-s^4 / d_p^4) and its derivatives

        exp_s = np.exp(-s2 * s2 / d_p ** 4)

        S = s2 * exp_s
        dS_ds2 = exp_s * (1 - 2 * s2 * s2 / d_p ** 4)
        d2S_ds22 = -(2 * s2 / d_p ** 4) * exp_s * (3 - 2 * s2 * s2 / d_p ** 4)

        dS_dn = dS_ds2 * ds2_dn
        dS_ds = dS_ds2 * ds2_ds
        d2S_dn2 = d2S_ds22 * ds2_dn * ds2_dn + dS_ds2 * d2s2_dn2
        d2S_dnds = d2S_ds22 * ds2_dn * ds2_ds + dS_ds2 * d2s2_dnds
        d2S_ds2 = d2S_ds22 * ds2_ds * ds2_ds

        # V = 1 / (d_s phi^3 w_1) from its logarithmic derivatives, as all three factors are positive

        V = 1 / (d_s * phi_cubed * w_1)

        l_n = -dw1_dn * inv_w_1
        l_zeta = -dd_s / d_s - 3 * dphi / phi - dw1_dzeta * inv_w_1
        l_nn = -(d2w1_dn2 * inv_w_1 - dw1_dn * dw1_dn * inv_w_1_squared)
        l_nzeta = -(d2w1_dndzeta * inv_w_1 - dw1_dn * dw1_dzeta * inv_w_1_squared)
        l_zetazeta = -(d2d_s / d_s - dd_s * dd_s / (d_s * d_s)) - 3 * (d2phi / phi - dphi * dphi / (phi * phi)) - (d2w1_dzeta2 * inv_w_1 - dw1_dzeta * dw1_dzeta * inv_w_1_squared)

        dV_dn = V * l_n
        dV_dzeta = V * l_zeta
        d2V_dn2 = V * (l_n * l_n + l_nn)
        d2V_dndzeta = V * (l_n * l_zeta + l_nzeta)
        d2V_dzeta2 = V * (l_zeta * l_zeta + l_zetazeta)

        # M = B V, which depends on density and zeta

        M = B * V
        dM_dn = dB_dn * V + B * dV_dn
        dM_dzeta = dB_dzeta * V + B * dV_dzeta
        d2M_dn2 = d2B_dn2 * V + 2 * dB_dn * dV_dn + B * d2V_dn2
        d2M_dndzeta = d2B_dndzeta * V + dB_dn * dV_dzeta + dB_dzeta * dV_dn + B * d2V_dndzeta
        d2M_dzeta2 = d2B_dzeta2 * V + 2 * dB_dzeta * dV_dzeta + B * d2V_dzeta2

        # The correction is delta_y = delta_f_c / (27 gamma) * S * M, which is subtracted from y

        K_delta = delta_f_c / (27 * gamma)

        y = y - K_delta * S * M

        dy_dn = dy_dn - K_delta * (dS_dn * M + S * dM_dn)
        dy_dzeta = dy_dzeta - K_delta * S * dM_dzeta
        dy_ds = dy_ds - K_delta * dS_ds * M
        d2y_dn2 = d2y_dn2 - K_delta * (d2S_dn2 * M + 2 * dS_dn * dM_dn + S * d2M_dn2)
        d2y_dndzeta = d2y_dndzeta - K_delta * (dS_dn * dM_dzeta + S * d2M_dndzeta)
        d2y_dzeta2 = d2y_dzeta2 - K_delta * S * d2M_dzeta2
        d2y_dnds = d2y_dnds - K_delta * (d2S_dnds * M + dS_ds * dM_dn)
        d2y_dzetads = d2y_dzetads - K_delta * dS_ds * dM_dzeta
        d2y_ds2 = d2y_ds2 - K_delta * d2S_ds2 * M

    # The function g of y and its derivatives

    g = (1 + 4 * y) ** (-1 / 4)
    dg_dy = -g ** 5
    d2g_dy2 = 5 * g ** 9

    dg_dn = dg_dy * dy_dn
    dg_dzeta = dg_dy * dy_dzeta
    dg_ds = dg_dy * dy_ds
    d2g_dn2 = d2g_dy2 * dy_dn * dy_dn + dg_dy * d2y_dn2
    d2g_dndzeta = d2g_dy2 * dy_dn * dy_dzeta + dg_dy * d2y_dndzeta
    d2g_dzeta2 = d2g_dy2 * dy_dzeta * dy_dzeta + dg_dy * d2y_dzeta2
    d2g_dnds = d2g_dy2 * dy_dn * dy_ds + dg_dy * d2y_dnds
    d2g_dzetads = d2g_dy2 * dy_dzeta * dy_ds + dg_dy * d2y_dzetads
    d2g_ds2 = d2g_dy2 * dy_ds * dy_ds + dg_dy * d2y_ds2

    # The logged quantity of the single-orbital limit, X_1 = w_1 (1 - g)

    X_1 = w_1 * (1 - g)

    dX1_dn = dw1_dn * (1 - g) - w_1 * dg_dn
    dX1_dzeta = dw1_dzeta * (1 - g) - w_1 * dg_dzeta
    dX1_ds = -w_1 * dg_ds
    d2X1_dn2 = d2w1_dn2 * (1 - g) - 2 * dw1_dn * dg_dn - w_1 * d2g_dn2
    d2X1_dndzeta = d2w1_dndzeta * (1 - g) - dw1_dn * dg_dzeta - dw1_dzeta * dg_dn - w_1 * d2g_dndzeta
    d2X1_dzeta2 = d2w1_dzeta2 * (1 - g) - 2 * dw1_dzeta * dg_dzeta - w_1 * d2g_dzeta2
    d2X1_dnds = -dw1_dn * dg_ds - w_1 * d2g_dnds
    d2X1_dzetads = -dw1_dzeta * dg_ds - w_1 * d2g_dzetads
    d2X1_ds2 = -w_1 * d2g_ds2

    # Derivatives of L_1 = log(1 + X_1)

    inv_one_plus_X1 = 1 / (1 + X_1)

    L_1 = np.log1p(X_1)

    dL1_dn = dX1_dn * inv_one_plus_X1
    dL1_dzeta = dX1_dzeta * inv_one_plus_X1
    dL1_ds = dX1_ds * inv_one_plus_X1
    d2L1_dn2 = d2X1_dn2 * inv_one_plus_X1 - dL1_dn * dL1_dn
    d2L1_dndzeta = d2X1_dndzeta * inv_one_plus_X1 - dL1_dn * dL1_dzeta
    d2L1_dzeta2 = d2X1_dzeta2 * inv_one_plus_X1 - dL1_dzeta * dL1_dzeta
    d2L1_dnds = d2X1_dnds * inv_one_plus_X1 - dL1_dn * dL1_ds
    d2L1_dzetads = d2X1_dzetads * inv_one_plus_X1 - dL1_dzeta * dL1_ds
    d2L1_ds2 = d2X1_ds2 * inv_one_plus_X1 - dL1_ds * dL1_ds

    # Single-orbital limit, e_C_1 = e_L + gamma phi^3 L_1

    e_1 = e_L + gamma_phi_cubed * L_1

    de1_dn = de_L_dn + gamma_phi_cubed * dL1_dn
    de1_dzeta = de_L_dzeta + gamma * (dphi_cubed * L_1 + phi_cubed * dL1_dzeta)
    de1_ds = gamma_phi_cubed * dL1_ds
    d2e1_dn2 = d2e_L_dn2 + gamma_phi_cubed * d2L1_dn2
    d2e1_dndzeta = d2e_L_dndzeta + gamma * (dphi_cubed * dL1_dn + phi_cubed * d2L1_dndzeta)
    d2e1_dzeta2 = d2e_L_dzeta2 + gamma * (d2phi_cubed * L_1 + 2 * dphi_cubed * dL1_dzeta + phi_cubed * d2L1_dzeta2)
    d2e1_dnds = gamma_phi_cubed * d2L1_dnds
    d2e1_dzetads = gamma * (dphi_cubed * dL1_ds + phi_cubed * d2L1_dzetads)
    d2e1_ds2 = gamma_phi_cubed * d2L1_ds2

    # Slowly-varying limit, from w_0 = exp(-e_0 / b_1c) - 1 and g_infinity, which depend on density and sigma only

    w_0 = np.expm1(-e_0 / b_1c)
    dw0_dn = -(w_0 + 1) * de0_dn / b_1c
    d2w0_dn2 = (w_0 + 1) * (de0_dn * de0_dn / b_1c ** 2 - d2e0_dn2 / b_1c)

    g_infinity = (1 + 4 * chi_infinity * s2) ** (-1 / 4)
    dg_infinity = -chi_infinity * g_infinity ** 5
    d2g_infinity = 5 * chi_infinity ** 2 * g_infinity ** 9

    dginf_dn = dg_infinity * ds2_dn
    dginf_ds = dg_infinity * ds2_ds
    d2ginf_dn2 = d2g_infinity * ds2_dn * ds2_dn + dg_infinity * d2s2_dn2
    d2ginf_dnds = d2g_infinity * ds2_dn * ds2_ds + dg_infinity * d2s2_dnds
    d2ginf_ds2 = d2g_infinity * ds2_ds * ds2_ds

    X_0 = w_0 * (1 - g_infinity)

    dX0_dn = dw0_dn * (1 - g_infinity) - w_0 * dginf_dn
    dX0_ds = -w_0 * dginf_ds
    d2X0_dn2 = d2w0_dn2 * (1 - g_infinity) - 2 * dw0_dn * dginf_dn - w_0 * d2ginf_dn2
    d2X0_dnds = -dw0_dn * dginf_ds - w_0 * d2ginf_dnds
    d2X0_ds2 = -w_0 * d2ginf_ds2

    inv_one_plus_X0 = 1 / (1 + X_0)

    H_0 = b_1c * np.log1p(X_0)

    dH0_dn = b_1c * dX0_dn * inv_one_plus_X0
    dH0_ds = b_1c * dX0_ds * inv_one_plus_X0
    d2H0_dn2 = b_1c * (d2X0_dn2 * inv_one_plus_X0 - dX0_dn * dX0_dn * inv_one_plus_X0 * inv_one_plus_X0)
    d2H0_dnds = b_1c * (d2X0_dnds * inv_one_plus_X0 - dX0_dn * dX0_ds * inv_one_plus_X0 * inv_one_plus_X0)
    d2H0_ds2 = b_1c * (d2X0_ds2 * inv_one_plus_X0 - dX0_ds * dX0_ds * inv_one_plus_X0 * inv_one_plus_X0)

    # The slowly-varying limit e_C_0 = (e_0 + H_0) G_c, where G_c only depends on zeta

    e_C_0 = (e_0 + H_0) * G_c

    dec0_dn = (de0_dn + dH0_dn) * G_c
    dec0_dzeta = (e_0 + H_0) * dG_c
    dec0_ds = dH0_ds * G_c
    d2ec0_dn2 = (d2e0_dn2 + d2H0_dn2) * G_c
    d2ec0_dndzeta = (de0_dn + dH0_dn) * dG_c
    d2ec0_dzeta2 = (e_0 + H_0) * d2G_c
    d2ec0_dnds = d2H0_dnds * G_c
    d2ec0_dzetads = dH0_ds * dG_c
    d2ec0_ds2 = d2H0_ds2 * G_c

    # Iso-orbital indicator N / D, where D is regularised for rSCAN and r2SCAN and tau only enters N

    eta_constant = 0.0001 if functional == "RSCAN" else 0
    eta_weizsacker = 0.001 if functional == "R2SCAN" else 0

    tau_W = sigma * inv_density / 8
    tau_U = (3 / 10) * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 5

    N = tau - tau_W

    dN_dn = tau_W * inv_density
    dN_ds = -inv_density / 8
    d2N_dn2 = -2 * tau_W * inv_density * inv_density
    d2N_dnds = inv_density * inv_density / 8

    D = (tau_U + eta_constant) * d_s + eta_weizsacker * tau_W

    dD_dn = (5 / 3) * tau_U * inv_density * d_s - eta_weizsacker * tau_W * inv_density
    dD_dzeta = (tau_U + eta_constant) * dd_s
    dD_ds = eta_weizsacker * inv_density / 8
    d2D_dn2 = (10 / 9) * tau_U * inv_density * inv_density * d_s + 2 * eta_weizsacker * tau_W * inv_density * inv_density
    d2D_dndzeta = (5 / 3) * tau_U * inv_density * dd_s
    d2D_dzeta2 = (tau_U + eta_constant) * d2d_s
    d2D_dnds = -eta_weizsacker * inv_density * inv_density / 8

    inv_D = 1 / D

    alpha = N * inv_D

    # Derivatives of alpha, from differentiating N = alpha * D

    da_dn = (dN_dn - alpha * dD_dn) * inv_D
    da_dzeta = -alpha * dD_dzeta * inv_D
    da_ds = (dN_ds - alpha * dD_ds) * inv_D
    da_dt = inv_D

    d2a_dn2 = (d2N_dn2 - 2 * da_dn * dD_dn - alpha * d2D_dn2) * inv_D
    d2a_dndzeta = -(da_dn * dD_dzeta + da_dzeta * dD_dn + alpha * d2D_dndzeta) * inv_D
    d2a_dzeta2 = -(2 * da_dzeta * dD_dzeta + alpha * d2D_dzeta2) * inv_D
    d2a_dnds = (d2N_dnds - da_dn * dD_ds - da_ds * dD_dn - alpha * d2D_dnds) * inv_D
    d2a_dzetads = -(da_dzeta * dD_ds + da_ds * dD_dzeta) * inv_D
    d2a_ds2 = -2 * da_ds * dD_ds * inv_D
    d2a_dndt = -da_dt * dD_dn * inv_D
    d2a_dzetadt = -da_dt * dD_dzeta * inv_D
    d2a_dsdt = -da_dt * dD_ds * inv_D
    d2a_dt2 = zeros

    if functional == "RSCAN":

        # The rSCAN indicator is alpha^3 / (alpha^2 + 0.001), so the second derivatives are updated before the first

        alpha_r = 0.001

        alpha_squared = alpha * alpha
        inv_denominator = 1 / (alpha_squared + alpha_r)

        dmap = alpha_squared * (alpha_squared + 3 * alpha_r) * inv_denominator * inv_denominator
        d2map = 2 * alpha_r * alpha * (3 * alpha_r - alpha_squared) * inv_denominator * inv_denominator * inv_denominator

        alpha = alpha_squared * alpha * inv_denominator

        d2a_dn2 = d2map * da_dn * da_dn + dmap * d2a_dn2
        d2a_dndzeta = d2map * da_dn * da_dzeta + dmap * d2a_dndzeta
        d2a_dzeta2 = d2map * da_dzeta * da_dzeta + dmap * d2a_dzeta2
        d2a_dnds = d2map * da_dn * da_ds + dmap * d2a_dnds
        d2a_dzetads = d2map * da_dzeta * da_ds + dmap * d2a_dzetads
        d2a_ds2 = d2map * da_ds * da_ds + dmap * d2a_ds2
        d2a_dndt = d2map * da_dn * da_dt + dmap * d2a_dndt
        d2a_dzetadt = d2map * da_dzeta * da_dt + dmap * d2a_dzetadt
        d2a_dsdt = d2map * da_ds * da_dt + dmap * d2a_dsdt
        d2a_dt2 = d2map * da_dt * da_dt

        da_dn, da_dzeta, da_ds, da_dt = dmap * da_dn, dmap * da_dzeta, dmap * da_ds, dmap * da_dt

    # Switching function of alpha, and its derivatives

    f_c, df_c, d2f_c = calculate_SCAN_switching_derivatives(alpha, c_1c, c_2c, d_c, None if functional == "SCAN" else c_c)

    dfc_dn = df_c * da_dn
    dfc_dzeta = df_c * da_dzeta
    dfc_ds = df_c * da_ds
    dfc_dt = df_c * da_dt

    d2fc_dn2 = d2f_c * da_dn * da_dn + df_c * d2a_dn2
    d2fc_dndzeta = d2f_c * da_dn * da_dzeta + df_c * d2a_dndzeta
    d2fc_dzeta2 = d2f_c * da_dzeta * da_dzeta + df_c * d2a_dzeta2
    d2fc_dnds = d2f_c * da_dn * da_ds + df_c * d2a_dnds
    d2fc_dzetads = d2f_c * da_dzeta * da_ds + df_c * d2a_dzetads
    d2fc_ds2 = d2f_c * da_ds * da_ds + df_c * d2a_ds2
    d2fc_dndt = d2f_c * da_dn * da_dt + df_c * d2a_dndt
    d2fc_dzetadt = d2f_c * da_dzeta * da_dt + df_c * d2a_dzetadt
    d2fc_dsdt = d2f_c * da_ds * da_dt + df_c * d2a_dsdt
    d2fc_dt2 = d2f_c * da_dt * da_dt + df_c * d2a_dt2

    # Difference between the two limits, which does not depend on tau

    delta = e_C_0 - e_1

    ddelta_dn = dec0_dn - de1_dn
    ddelta_dzeta = dec0_dzeta - de1_dzeta
    ddelta_ds = dec0_ds - de1_ds
    d2delta_dn2 = d2ec0_dn2 - d2e1_dn2
    d2delta_dndzeta = d2ec0_dndzeta - d2e1_dndzeta
    d2delta_dzeta2 = d2ec0_dzeta2 - d2e1_dzeta2
    d2delta_dnds = d2ec0_dnds - d2e1_dnds
    d2delta_dzetads = d2ec0_dzetads - d2e1_dzetads
    d2delta_ds2 = d2ec0_ds2 - d2e1_ds2

    # Interpolated correlation energy density per particle, e_C = e_C_1 + f_c (e_C_0 - e_C_1)

    de_dn = de1_dn + dfc_dn * delta + f_c * ddelta_dn
    de_dzeta = de1_dzeta + dfc_dzeta * delta + f_c * ddelta_dzeta
    de_ds = de1_ds + dfc_ds * delta + f_c * ddelta_ds
    de_dt = dfc_dt * delta

    d2e_dn2 = d2e1_dn2 + d2fc_dn2 * delta + 2 * dfc_dn * ddelta_dn + f_c * d2delta_dn2
    d2e_dndzeta = d2e1_dndzeta + d2fc_dndzeta * delta + dfc_dn * ddelta_dzeta + dfc_dzeta * ddelta_dn + f_c * d2delta_dndzeta
    d2e_dzeta2 = d2e1_dzeta2 + d2fc_dzeta2 * delta + 2 * dfc_dzeta * ddelta_dzeta + f_c * d2delta_dzeta2
    d2e_dnds = d2e1_dnds + d2fc_dnds * delta + dfc_dn * ddelta_ds + dfc_ds * ddelta_dn + f_c * d2delta_dnds
    d2e_dzetads = d2e1_dzetads + d2fc_dzetads * delta + dfc_dzeta * ddelta_ds + dfc_ds * ddelta_dzeta + f_c * d2delta_dzetads
    d2e_ds2 = d2e1_ds2 + d2fc_ds2 * delta + 2 * dfc_ds * ddelta_ds + f_c * d2delta_ds2
    d2e_dndt = d2fc_dndt * delta + dfc_dt * ddelta_dn
    d2e_dzetadt = d2fc_dzetadt * delta + dfc_dt * ddelta_dzeta
    d2e_dsdt = d2fc_dsdt * delta + dfc_dt * ddelta_ds
    d2e_dt2 = d2fc_dt2 * delta

    # Derivatives of f = n * e_C with respect to density, zeta, sigma and tau

    df_dzeta = density * de_dzeta

    d2f_dn2 = 2 * de_dn + density * d2e_dn2
    d2f_dndzeta = de_dzeta + density * d2e_dndzeta
    d2f_dzeta2 = density * d2e_dzeta2
    d2f_dnds = de_ds + density * d2e_dnds
    d2f_dzetads = density * d2e_dzetads
    d2f_ds2 = density * d2e_ds2
    d2f_dndt = de_dt + density * d2e_dndt
    d2f_dzetadt = density * d2e_dzetadt
    d2f_dsdt = density * d2e_dsdt
    d2f_dt2 = density * d2e_dt2

    # Transforms onto the two spin densities, with the tau blocks transformed like the sigma ones

    f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma = calculate_spin_polarisation_transformation(zeta, density, df_dzeta, d2f_dn2, d2f_dndzeta, d2f_dzeta2, d2f_dnds, d2f_dzetads, d2f_ds2)

    f_C_alpha_tau = d2f_dndt + d2f_dzetadt * (1 - zeta) * inv_density
    f_C_beta_tau = d2f_dndt - d2f_dzetadt * (1 + zeta) * inv_density

    return f_C_alpha_alpha, f_C_alpha_beta, f_C_beta_beta, f_C_alpha_sigma, f_C_beta_sigma, f_C_sigma_sigma, f_C_alpha_tau, f_C_beta_tau, d2f_dsdt, d2f_dt2










def calculate_restricted_SCAN_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted SCAN, rSCAN or r2SCAN correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_C with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_C with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_C with respect to tau

    """

    # Spin-resolved blocks at the closed shell, where the total sigma and tau are those of the restricted reference

    f_aa, f_ab, f_bb, f_a_s, f_b_s, f_ss, f_a_t, f_b_t, f_st, f_tt = calculate_unrestricted_SCAN_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For singlet excitations both spin densities change by half the density change

    d2f_dn2 = (f_aa + 2 * f_ab + f_bb) / 4
    d2f_dnds = (f_a_s + f_b_s) / 2
    d2f_ds2 = f_ss
    d2f_dndt = (f_a_t + f_b_t) / 2
    d2f_dsdt = f_st
    d2f_dt2 = f_tt

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_restricted_SCAN_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> ndarray:

    """

    Calculates the restricted SCAN, rSCAN or r2SCAN correlation spin kernel for triplet excitations. These functionals only see the total square
    gradient and kinetic energy density, which do not change in a triplet excitation, so as for PBE only the spin density block is needed.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density

    """

    f_aa, f_ab, f_bb, _, _, _, _, _, _, _ = calculate_unrestricted_SCAN_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For triplet excitations the spin densities change by plus and minus half the spin density change

    f_mm = (f_aa - 2 * f_ab + f_bb) / 4

    return f_mm










def calculate_B97M_kinetic_ratio_derivatives(spin_density: ndarray, spin_tau: ndarray) -> tuple:

    """

    Calculates the ratio t = tau_U / tau of one spin channel, where tau_U = (3 / 10) * (6 pi^2)^(2/3) * p^(5/3), with its derivatives.

    Args:
        spin_density (array): Density of one spin channel on integration grid
        spin_tau (array): Kinetic energy density of the same spin channel

    Returns:
        t (array): Kinetic energy density ratio of the channel
        dt_dn (array): Derivative with respect to the spin density
        dt_dt (array): Derivative with respect to the channel kinetic energy density
        d2t_dn2 (array): Second derivative with respect to the spin density
        d2t_dndt (array): Mixed second derivative
        d2t_dt2 (array): Second derivative with respect to the channel kinetic energy density

    """

    inv_spin_density = 1 / spin_density
    inv_spin_tau = 1 / spin_tau

    t = (3 / 10) * np.cbrt(6 * np.pi ** 2) ** 2 * np.cbrt(spin_density) ** 5 * inv_spin_tau

    # The ratio goes as p^(5/3) / tau

    dt_dn = (5 / 3) * t * inv_spin_density
    dt_dt = -t * inv_spin_tau
    d2t_dn2 = (10 / 9) * t * inv_spin_density * inv_spin_density
    d2t_dndt = -(5 / 3) * t * inv_spin_density * inv_spin_tau
    d2t_dt2 = 2 * t * inv_spin_tau * inv_spin_tau

    return t, dt_dn, dt_dt, d2t_dn2, d2t_dndt, d2t_dt2










def calculate_B97M_same_spin_derivatives(spin_density: ndarray, spin_sigma: ndarray, spin_tau: ndarray) -> tuple:

    """

    Calculates the second derivatives of the same-spin B97M term g_ss(w, u) * W for one spin channel, where W is the spin density times the
    ferromagnetic PW92 energy per particle.

    Args:
        spin_density (array): Density of one spin channel on integration grid
        spin_sigma (array): Square density gradient of the same spin channel
        spin_tau (array): Kinetic energy density of the same spin channel

    Returns:
        P_nn (array): Second derivative with respect to the spin density
        P_ns (array): Mixed second derivative with respect to the spin density and square gradient
        P_nt (array): Mixed second derivative with respect to the spin density and kinetic energy density
        P_ss (array): Second derivative with respect to the square gradient
        P_st (array): Mixed second derivative with respect to the square gradient and kinetic energy density
        P_tt (array): Second derivative with respect to the kinetic energy density

    """

    # Same-spin coefficients and finite range parameter

    c_ss = [1, -5.668, -1.855, -20.497, -20.364]
    gamma_ss = 0.2

    # The same-spin PW92 piece, and the reduced gradient and kinetic energy density ratio of the channel

    W, dW_dn, d2W_dn2 = calculate_PW_ferromagnetic_derivatives(spin_density)

    t, dt_dn, dt_dt, d2t_dn2, d2t_dndt, d2t_dt2 = calculate_B97M_kinetic_ratio_derivatives(spin_density, spin_tau)
    s2, ds2_dn, ds2_ds, d2s2_dn2, d2s2_dnds = calculate_B97_channel_gradient_derivatives(spin_density, spin_sigma)

    # Finite range kinetic energy variable w = (t - 1) / (t + 1)

    inv_t_plus_one = 1 / (t + 1)

    w = (t - 1) * inv_t_plus_one
    dw_dt_ratio = 2 * inv_t_plus_one * inv_t_plus_one
    d2w_dt_ratio2 = -2 * dw_dt_ratio * inv_t_plus_one

    dw_dn = dw_dt_ratio * dt_dn
    dw_dt = dw_dt_ratio * dt_dt
    d2w_dn2 = d2w_dt_ratio2 * dt_dn * dt_dn + dw_dt_ratio * d2t_dn2
    d2w_dndt = d2w_dt_ratio2 * dt_dn * dt_dt + dw_dt_ratio * d2t_dndt
    d2w_dt2 = d2w_dt_ratio2 * dt_dt * dt_dt + dw_dt_ratio * d2t_dt2

    # Finite range gradient variable u = gamma s^2 / (1 + gamma s^2)

    inv_one_plus_gamma_s2 = 1 / (1 + gamma_ss * s2)

    u = gamma_ss * s2 * inv_one_plus_gamma_s2
    du_ds2 = gamma_ss * inv_one_plus_gamma_s2 * inv_one_plus_gamma_s2
    d2u_ds22 = -2 * gamma_ss * du_ds2 * inv_one_plus_gamma_s2

    du_dn = du_ds2 * ds2_dn
    du_ds = du_ds2 * ds2_ds
    d2u_dn2 = d2u_ds22 * ds2_dn * ds2_dn + du_ds2 * d2s2_dn2
    d2u_dnds = d2u_ds22 * ds2_dn * ds2_ds + du_ds2 * d2s2_dnds
    d2u_ds2 = d2u_ds22 * ds2_ds * ds2_ds

    # Same-spin factor g = c_0 + c_1 w + (c_2 + c_3 w^3 + c_4 w^4) u^2 and its derivatives in w and u

    polynomial = c_ss[2] + c_ss[3] * w ** 3 + c_ss[4] * w ** 4
    dpolynomial = 3 * c_ss[3] * w * w + 4 * c_ss[4] * w ** 3
    d2polynomial = 6 * c_ss[3] * w + 12 * c_ss[4] * w * w

    g = c_ss[0] + c_ss[1] * w + polynomial * u * u

    g_w = c_ss[1] + dpolynomial * u * u
    g_u = 2 * polynomial * u
    g_ww = d2polynomial * u * u
    g_wu = 2 * dpolynomial * u
    g_uu = 2 * polynomial

    # Chain rule onto the spin density, square gradient and kinetic energy density of the channel

    dg_dn = g_w * dw_dn + g_u * du_dn
    dg_ds = g_u * du_ds
    dg_dt = g_w * dw_dt

    d2g_dn2 = g_ww * dw_dn * dw_dn + 2 * g_wu * dw_dn * du_dn + g_uu * du_dn * du_dn + g_w * d2w_dn2 + g_u * d2u_dn2
    d2g_dnds = g_wu * dw_dn * du_ds + g_uu * du_dn * du_ds + g_u * d2u_dnds
    d2g_dndt = g_ww * dw_dn * dw_dt + g_wu * du_dn * dw_dt + g_w * d2w_dndt
    d2g_ds2 = g_uu * du_ds * du_ds + g_u * d2u_ds2
    d2g_dsdt = g_wu * du_ds * dw_dt
    d2g_dt2 = g_ww * dw_dt * dw_dt + g_w * d2w_dt2

    # The same-spin term is g * W, and W only depends on the spin density

    P_nn = d2g_dn2 * W + 2 * dg_dn * dW_dn + g * d2W_dn2
    P_ns = d2g_dnds * W + dg_ds * dW_dn
    P_nt = d2g_dndt * W + dg_dt * dW_dn
    P_ss = d2g_ds2 * W
    P_st = d2g_dsdt * W
    P_tt = d2g_dt2 * W

    return P_nn, P_ns, P_nt, P_ss, P_st, P_tt










def calculate_unrestricted_B97M_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved B97M correlation kernel for an unrestricted reference. The functional does not depend on sigma_ab.

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
        f_C_alpha_tau_alpha (array): Mixed second derivative with respect to the alpha density and tau alpha
        f_C_alpha_tau_beta (array): Mixed second derivative with respect to the alpha density and tau beta
        f_C_beta_tau_alpha (array): Mixed second derivative with respect to the beta density and tau alpha
        f_C_beta_tau_beta (array): Mixed second derivative with respect to the beta density and tau beta
        f_C_sigma_aa_tau_alpha (array): Mixed second derivative with respect to sigma alpha-alpha and tau alpha
        f_C_sigma_aa_tau_beta (array): Mixed second derivative with respect to sigma alpha-alpha and tau beta
        f_C_sigma_bb_tau_alpha (array): Mixed second derivative with respect to sigma beta-beta and tau alpha
        f_C_sigma_bb_tau_beta (array): Mixed second derivative with respect to sigma beta-beta and tau beta
        f_C_tau_alpha_tau_alpha (array): Second derivative with respect to tau alpha
        f_C_tau_alpha_tau_beta (array): Mixed second derivative with respect to tau alpha and tau beta
        f_C_tau_beta_tau_beta (array): Second derivative with respect to tau beta

    """

    # Opposite-spin coefficients and finite range parameter

    c_ab = [1, 2.535, 1.573, -6.427, -6.298]
    gamma_ab = 0.006

    zeros = np.zeros_like(density)

    # Same-spin terms, each confined to its own channel

    P_a_nn, P_a_ns, P_a_nt, P_a_ss, P_a_st, P_a_tt = calculate_B97M_same_spin_derivatives(alpha_density, sigma_aa, tau_alpha)
    P_b_nn, P_b_ns, P_b_nt, P_b_ss, P_b_st, P_b_tt = calculate_B97M_same_spin_derivatives(beta_density, sigma_bb, tau_beta)

    # Opposite-spin piece C = n * e_LSDA - W_a - W_b, which only depends on the two spin densities, as for B97

    W_a, dW_a, d2W_a = calculate_PW_ferromagnetic_derivatives(alpha_density)
    W_b, dW_b, d2W_b = calculate_PW_ferromagnetic_derivatives(beta_density)

    df_dn_alpha, df_dn_beta, _, _, _, _, _, e_C_LSDA = calculate_unrestricted_PW_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)
    F_aa, F_ab, F_bb = calculate_unrestricted_PW_correlation_kernel(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation)

    C = density * e_C_LSDA - W_a - W_b

    # The kinetic energy density ratios and reduced gradients of both channels

    t_a, dt_a_dn, dt_a_dt, d2t_a_dn2, d2t_a_dndt, d2t_a_dt2 = calculate_B97M_kinetic_ratio_derivatives(alpha_density, tau_alpha)
    t_b, dt_b_dn, dt_b_dt, d2t_b_dn2, d2t_b_dndt, d2t_b_dt2 = calculate_B97M_kinetic_ratio_derivatives(beta_density, tau_beta)

    s2_a, ds2_a_dn, ds2_a_ds, d2s2_a_dn2, d2s2_a_dnds = calculate_B97_channel_gradient_derivatives(alpha_density, sigma_aa)
    s2_b, ds2_b_dn, ds2_b_ds, d2s2_b_dn2, d2s2_b_dnds = calculate_B97_channel_gradient_derivatives(beta_density, sigma_bb)

    # The opposite-spin finite range variables use the means of the two channels

    inv_t_plus_one = 1 / ((t_a + t_b) / 2 + 1)

    w = ((t_a + t_b) / 2 - 1) * inv_t_plus_one
    dw = 2 * inv_t_plus_one * inv_t_plus_one
    d2w = -2 * dw * inv_t_plus_one

    inv_one_plus_gamma_s2 = 1 / (1 + gamma_ab * (s2_a + s2_b) / 2)

    u = gamma_ab * (s2_a + s2_b) / 2 * inv_one_plus_gamma_s2
    du = gamma_ab * inv_one_plus_gamma_s2 * inv_one_plus_gamma_s2
    d2u = -2 * gamma_ab * du * inv_one_plus_gamma_s2

    # Opposite-spin factor g = c_0 + c_1 w + c_2 u + c_3 w^3 u^2 + c_4 u^3 and its derivatives in w and u

    g = c_ab[0] + c_ab[1] * w + c_ab[2] * u + c_ab[3] * w ** 3 * u * u + c_ab[4] * u ** 3

    g_w = c_ab[1] + 3 * c_ab[3] * w * w * u * u
    g_u = c_ab[2] + 2 * c_ab[3] * w ** 3 * u + 3 * c_ab[4] * u * u
    g_ww = 6 * c_ab[3] * w * u * u
    g_wu = 6 * c_ab[3] * w * w * u
    g_uu = 2 * c_ab[3] * w ** 3 + 6 * c_ab[4] * u

    # Derivatives of the channel means, and of C, over the variables alpha, beta, sigma_aa, sigma_bb, tau_alpha, tau_beta - second derivatives only couple variables of the same channel

    dt_mean = [dt_a_dn / 2, dt_b_dn / 2, 0, 0, dt_a_dt / 2, dt_b_dt / 2]
    d2t_mean = {(0, 0): d2t_a_dn2 / 2, (0, 4): d2t_a_dndt / 2, (4, 4): d2t_a_dt2 / 2, (1, 1): d2t_b_dn2 / 2, (1, 5): d2t_b_dndt / 2, (5, 5): d2t_b_dt2 / 2}

    ds2_mean = [ds2_a_dn / 2, ds2_b_dn / 2, ds2_a_ds / 2, ds2_b_ds / 2, 0, 0]
    d2s2_mean = {(0, 0): d2s2_a_dn2 / 2, (0, 2): d2s2_a_dnds / 2, (1, 1): d2s2_b_dn2 / 2, (1, 3): d2s2_b_dnds / 2}

    dC = [df_dn_alpha - dW_a, df_dn_beta - dW_b, 0, 0, 0, 0]
    d2C = {(0, 0): F_aa - d2W_a, (0, 1): F_ab, (1, 1): F_bb - d2W_b}

    # Same-spin blocks placed among the six variables, the alpha channel being (0, 2, 4) and the beta channel (1, 3, 5)

    P = {(0, 0): P_a_nn, (0, 2): P_a_ns, (0, 4): P_a_nt, (2, 2): P_a_ss, (2, 4): P_a_st, (4, 4): P_a_tt, (1, 1): P_b_nn, (1, 3): P_b_ns, (1, 5): P_b_nt, (3, 3): P_b_ss, (3, 5): P_b_st, (5, 5): P_b_tt}

    # The first ten blocks are ordered as for B97, and the kinetic energy density blocks follow

    index_pairs = [(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 4), (4, 5), (5, 5)]

    blocks = []

    for i, j in index_pairs:

        # First and second derivatives of w and u through the channel means

        w_i, w_j = dw * dt_mean[i], dw * dt_mean[j]
        u_i, u_j = du * ds2_mean[i], du * ds2_mean[j]

        w_ij = d2w * dt_mean[i] * dt_mean[j] + dw * d2t_mean.get((i, j), 0)
        u_ij = d2u * ds2_mean[i] * ds2_mean[j] + du * d2s2_mean.get((i, j), 0)

        # Chain rule for the opposite-spin factor

        g_i = g_w * w_i + g_u * u_i
        g_j = g_w * w_j + g_u * u_j

        g_ij = g_ww * w_i * w_j + g_wu * (w_i * u_j + w_j * u_i) + g_uu * u_i * u_j + g_w * w_ij + g_u * u_ij

        # Product rule for the opposite-spin term g * C, plus the same-spin term of the channel

        blocks.append(g_ij * C + g_i * dC[j] + g_j * dC[i] + g * d2C.get((i, j), 0) + P.get((i, j), 0) + zeros)

    return tuple(blocks)










def calculate_restricted_B97M_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97M correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_C with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_C with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_C with respect to tau

    """

    # Spin-resolved blocks at the closed shell

    f_aa, f_ab, f_bb, f_a_saa, f_a_sbb, f_b_saa, f_b_sbb, f_saa_saa, f_saa_sbb, f_sbb_sbb, f_a_ta, f_a_tb, f_b_ta, f_b_tb, f_saa_ta, f_saa_tb, f_sbb_ta, f_sbb_tb, f_ta_ta, f_ta_tb, f_tb_tb = calculate_unrestricted_B97M_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For singlet excitations each channel gets half the density and kinetic energy density change and a quarter of the sigma change

    d2f_dn2 = (f_aa + 2 * f_ab + f_bb) / 4
    d2f_dnds = (f_a_saa + f_a_sbb + f_b_saa + f_b_sbb) / 8
    d2f_ds2 = (f_saa_saa + 2 * f_saa_sbb + f_sbb_sbb) / 16
    d2f_dndt = (f_a_ta + f_a_tb + f_b_ta + f_b_tb) / 4
    d2f_dsdt = (f_saa_ta + f_saa_tb + f_sbb_ta + f_sbb_tb) / 8
    d2f_dt2 = (f_ta_ta + 2 * f_ta_tb + f_tb_tb) / 4

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_restricted_B97M_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted B97M correlation spin kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_nm_sigma_nm (array): Second derivative with respect to grad(n) . grad(m)
        f_sigma_mm (array): First derivative with respect to grad(m) . grad(m)
        f_m_tau_m (array): Mixed second derivative with respect to the spin density and the spin kinetic energy density, tau_alpha - tau_beta
        f_sigma_nm_tau_m (array): Mixed second derivative with respect to grad(n) . grad(m) and the spin kinetic energy density
        f_tau_m_tau_m (array): Second derivative with respect to the spin kinetic energy density

    """

    # Spin-resolved blocks at the closed shell

    f_aa, f_ab, f_bb, f_a_saa, f_a_sbb, f_b_saa, f_b_sbb, f_saa_saa, f_saa_sbb, f_sbb_sbb, f_a_ta, f_a_tb, f_b_ta, f_b_tb, f_saa_ta, f_saa_tb, f_sbb_ta, f_sbb_tb, f_ta_ta, f_ta_tb, f_tb_tb = calculate_unrestricted_B97M_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For triplet excitations the two channels change in opposite directions, by half the spin density, grad(n) . grad(m) and tau_m changes

    f_mm = (f_aa - 2 * f_ab + f_bb) / 4
    f_m_sigma_nm = (f_a_saa - f_a_sbb - f_b_saa + f_b_sbb) / 4
    f_sigma_nm_sigma_nm = (f_saa_saa - 2 * f_saa_sbb + f_sbb_sbb) / 4
    f_m_tau_m = (f_a_ta - f_a_tb - f_b_ta + f_b_tb) / 4
    f_sigma_nm_tau_m = (f_saa_ta - f_saa_tb - f_sbb_ta + f_sbb_tb) / 4
    f_tau_m_tau_m = (f_ta_ta - 2 * f_ta_tb + f_tb_tb) / 4

    # As grad(m) . grad(m) is quadratic in the response, only its first derivative enters, and it adds a quarter to each same-spin square gradient

    _, _, df_ds_aa, df_ds_bb, _, _, _, _ = calculate_unrestricted_B97M_correlation(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    f_sigma_mm = (df_ds_aa + df_ds_bb) / 4

    return f_mm, f_m_sigma_nm, f_sigma_nm_sigma_nm, f_sigma_mm, f_m_tau_m, f_sigma_nm_tau_m, f_tau_m_tau_m










def calculate_TPSS_PBE_correlation_derivatives(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma: ndarray, revised: bool) -> tuple:

    """

    Calculates the PBE correlation energy per particle used inside TPSS, with its first and second derivatives with respect to the two spin densities
    and the total square gradient. Revised TPSS uses a density-dependent beta.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
        sigma (array): Total square density gradient
        revised (bool): Is this for revised TPSS

    Returns:
        e_C (array): PBE correlation energy per particle
        de_da (array): Derivative with respect to the alpha density
        de_db (array): Derivative with respect to the beta density
        de_ds (array): Derivative with respect to sigma
        d2e_da2 (array): Second derivative with respect to the alpha density
        d2e_dadb (array): Mixed second derivative with respect to the two spin densities
        d2e_db2 (array): Second derivative with respect to the beta density
        d2e_dads (array): Mixed second derivative with respect to the alpha density and sigma
        d2e_dbds (array): Mixed second derivative with respect to the beta density and sigma
        d2e_ds2 (array): Second derivative with respect to sigma

    """

    gamma = (1 - np.log(2)) / np.pi ** 2

    inv_density = 1 / density

    # Local spin density correlation and its derivatives with respect to density and spin polarisation

    e_L, de_L_dn, de_L_dzeta, d2e_L_dn2, d2e_L_dndzeta, d2e_L_dzeta2, zeta = calculate_PW_spin_correlation_derivatives(alpha_density, beta_density, density)

    # Spin scaling factor and its cube

    cbrt_plus = np.cbrt(clean(1 + zeta))
    cbrt_minus = np.cbrt(clean(1 - zeta))

    phi = (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus) / 2
    dphi = (1 / cbrt_plus - 1 / cbrt_minus) / 3
    d2phi = -(1 / 9) * (1 / cbrt_plus ** 4 + 1 / cbrt_minus ** 4)

    phi_cubed = phi * phi * phi
    dphi_cubed = 3 * phi * phi * dphi
    d2phi_cubed = 6 * phi * dphi * dphi + 3 * phi * phi * d2phi

    # The ratio B = beta / gamma, which only depends on density for revised TPSS

    if revised:

        r_s, _ = calculate_seitz_radius(density)

        dr_dn = -r_s * inv_density / 3
        d2r_dn2 = (4 / 9) * r_s * inv_density * inv_density

        beta_denominator = 1 + 0.1778 * r_s

        dbeta_dr = 0.066725 * (0.1 - 0.1778) / (beta_denominator * beta_denominator)
        d2beta_dr2 = -2 * 0.1778 * dbeta_dr / beta_denominator

        B = 0.066725 * (1 + 0.1 * r_s) / beta_denominator / gamma
        dB_dn = dbeta_dr * dr_dn / gamma
        d2B_dn2 = (d2beta_dr2 * dr_dn * dr_dn + dbeta_dr * d2r_dn2) / gamma

    else:

        B, dB_dn, d2B_dn2 = 0.066725 / gamma, 0, 0

    # W is the argument of the exponential inside A = B / (exp(-W) - 1)

    gamma_phi_cubed = gamma * phi_cubed

    W = e_L / gamma_phi_cubed
    dW_dn = de_L_dn / gamma_phi_cubed
    dW_dzeta = (de_L_dzeta - e_L * dphi_cubed / phi_cubed) / gamma_phi_cubed
    d2W_dn2 = d2e_L_dn2 / gamma_phi_cubed
    d2W_dndzeta = (d2e_L_dndzeta - de_L_dn * dphi_cubed / phi_cubed) / gamma_phi_cubed
    d2W_dzeta2 = (d2e_L_dzeta2 - 2 * de_L_dzeta * dphi_cubed / phi_cubed - e_L * d2phi_cubed / phi_cubed + 2 * e_L * dphi_cubed * dphi_cubed / (phi_cubed * phi_cubed)) / gamma_phi_cubed

    # Derivatives of 1 / w_1 = 1 / (exp(-W) - 1) with respect to W, where expm1 is more stable than exp - 1

    exp_minus_W = np.exp(-W)
    inv_w_1 = 1 / np.expm1(-W)

    dinv_dW = exp_minus_W * inv_w_1 * inv_w_1
    d2inv_dW2 = dinv_dW * (2 * exp_minus_W * inv_w_1 - 1)

    A = B * inv_w_1
    dA_dn = dB_dn * inv_w_1 + B * dinv_dW * dW_dn
    dA_dzeta = B * dinv_dW * dW_dzeta
    d2A_dn2 = d2B_dn2 * inv_w_1 + 2 * dB_dn * dinv_dW * dW_dn + B * (d2inv_dW2 * dW_dn * dW_dn + dinv_dW * d2W_dn2)
    d2A_dndzeta = dB_dn * dinv_dW * dW_dzeta + B * (d2inv_dW2 * dW_dn * dW_dzeta + dinv_dW * d2W_dndzeta)
    d2A_dzeta2 = B * (d2inv_dW2 * dW_dzeta * dW_dzeta + dinv_dW * d2W_dzeta2)

    # Reduced density gradient T = t^2, which is linear in sigma

    k_F = calculate_Fermi_wavevector(density=density)

    dT_ds = np.pi / (16 * phi * phi * k_F * density * density)
    T = sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_density
    dT_dzeta = -2 * T * dphi / phi
    d2T_dn2 = (70 / 9) * T * inv_density * inv_density
    d2T_dndzeta = -2 * dT_dn * dphi / phi
    d2T_dzeta2 = T * (6 * dphi * dphi / (phi * phi) - 2 * d2phi / phi)
    d2T_dnds = -(7 / 3) * dT_ds * inv_density
    d2T_dzetads = -2 * dT_ds * dphi / phi

    # The logged quantity is X = B * Y, where Y = T (1 + A T) / (1 + A T + A^2 T^2) is the PW91 rational function with unit prefactor

    Y, dY_dT, dY_dA, d2Y_dT2, d2Y_dTdA, d2Y_dA2 = calculate_PW91_X_derivatives(T, A, 1)

    dY_dn = dY_dT * dT_dn + dY_dA * dA_dn
    dY_dzeta = dY_dT * dT_dzeta + dY_dA * dA_dzeta
    dY_ds = dY_dT * dT_ds
    d2Y_dn2 = d2Y_dT2 * dT_dn * dT_dn + 2 * d2Y_dTdA * dT_dn * dA_dn + d2Y_dA2 * dA_dn * dA_dn + dY_dT * d2T_dn2 + dY_dA * d2A_dn2
    d2Y_dndzeta = d2Y_dT2 * dT_dn * dT_dzeta + d2Y_dTdA * (dT_dn * dA_dzeta + dT_dzeta * dA_dn) + d2Y_dA2 * dA_dn * dA_dzeta + dY_dT * d2T_dndzeta + dY_dA * d2A_dndzeta
    d2Y_dzeta2 = d2Y_dT2 * dT_dzeta * dT_dzeta + 2 * d2Y_dTdA * dT_dzeta * dA_dzeta + d2Y_dA2 * dA_dzeta * dA_dzeta + dY_dT * d2T_dzeta2 + dY_dA * d2A_dzeta2
    d2Y_dnds = d2Y_dT2 * dT_dn * dT_ds + d2Y_dTdA * dT_ds * dA_dn + dY_dT * d2T_dnds
    d2Y_dzetads = d2Y_dT2 * dT_dzeta * dT_ds + d2Y_dTdA * dT_ds * dA_dzeta + dY_dT * d2T_dzetads
    d2Y_ds2 = d2Y_dT2 * dT_ds * dT_ds

    X = B * Y

    dX_dn = dB_dn * Y + B * dY_dn
    dX_dzeta = B * dY_dzeta
    dX_ds = B * dY_ds
    d2X_dn2 = d2B_dn2 * Y + 2 * dB_dn * dY_dn + B * d2Y_dn2
    d2X_dndzeta = dB_dn * dY_dzeta + B * d2Y_dndzeta
    d2X_dzeta2 = B * d2Y_dzeta2
    d2X_dnds = dB_dn * dY_ds + B * d2Y_dnds
    d2X_dzetads = B * d2Y_dzetads
    d2X_ds2 = B * d2Y_ds2

    # Derivatives of L = log(1 + X)

    inv_one_plus_X = 1 / (1 + X)

    L = np.log1p(X)

    dL_dn = dX_dn * inv_one_plus_X
    dL_dzeta = dX_dzeta * inv_one_plus_X
    dL_ds = dX_ds * inv_one_plus_X
    d2L_dn2 = d2X_dn2 * inv_one_plus_X - dL_dn * dL_dn
    d2L_dndzeta = d2X_dndzeta * inv_one_plus_X - dL_dn * dL_dzeta
    d2L_dzeta2 = d2X_dzeta2 * inv_one_plus_X - dL_dzeta * dL_dzeta
    d2L_dnds = d2X_dnds * inv_one_plus_X - dL_dn * dL_ds
    d2L_dzetads = d2X_dzetads * inv_one_plus_X - dL_dzeta * dL_ds
    d2L_ds2 = d2X_ds2 * inv_one_plus_X - dL_ds * dL_ds

    # PBE correlation energy per particle, e_L + gamma phi^3 L, with derivatives in density, zeta and sigma

    e_C = e_L + gamma_phi_cubed * L

    de_dn = de_L_dn + gamma_phi_cubed * dL_dn
    de_dzeta = de_L_dzeta + gamma * (dphi_cubed * L + phi_cubed * dL_dzeta)
    de_ds = gamma_phi_cubed * dL_ds
    d2e_dn2 = d2e_L_dn2 + gamma_phi_cubed * d2L_dn2
    d2e_dndzeta = d2e_L_dndzeta + gamma * (dphi_cubed * dL_dn + phi_cubed * d2L_dndzeta)
    d2e_dzeta2 = d2e_L_dzeta2 + gamma * (d2phi_cubed * L + 2 * dphi_cubed * dL_dzeta + phi_cubed * d2L_dzeta2)
    d2e_dnds = gamma_phi_cubed * d2L_dnds
    d2e_dzetads = gamma * (dphi_cubed * dL_ds + phi_cubed * d2L_dzetads)
    d2e_ds2 = gamma_phi_cubed * d2L_ds2

    # Transforms onto the two spin densities

    d2e_da2, d2e_dadb, d2e_db2, d2e_dads, d2e_dbds, d2e_ds2 = calculate_spin_polarisation_transformation(zeta, density, de_dzeta, d2e_dn2, d2e_dndzeta, d2e_dzeta2, d2e_dnds, d2e_dzetads, d2e_ds2)

    de_da = de_dn + de_dzeta * (1 - zeta) * inv_density
    de_db = de_dn - de_dzeta * (1 + zeta) * inv_density

    return e_C, de_da, de_db, de_ds, d2e_da2, d2e_dadb, d2e_db2, d2e_dads, d2e_dbds, d2e_ds2










def calculate_polarised_PBE_correlation_derivatives(spin_density: ndarray, spin_sigma: ndarray, revised: bool) -> tuple:

    """

    Calculates the PBE correlation energy per particle of one fully polarised spin channel, as used by TPSS, with its first and second derivatives.

    Args:
        spin_density (array): Density of one spin channel on integration grid
        spin_sigma (array): Square density gradient of the same spin channel
        revised (bool): Is this for revised TPSS, with a density-dependent beta

    Returns:
        e_C (array): Fully polarised PBE correlation energy per particle
        de_dn (array): Derivative with respect to the spin density
        de_ds (array): Derivative with respect to the square gradient
        d2e_dn2 (array): Second derivative with respect to the spin density
        d2e_dnds (array): Mixed second derivative
        d2e_ds2 (array): Second derivative with respect to the square gradient

    """

    gamma = (1 - np.log(2)) / np.pi ** 2

    # For a fully polarised density phi = (2^(2/3) + 0) / 2 is constant

    phi = np.cbrt(2) ** 2 / 2
    gamma_phi_cubed = gamma * phi ** 3

    inv_spin_density = 1 / spin_density

    spin_sigma = clean(spin_sigma, floor=constants.sigma_floor)

    # Only the ferromagnetic PW92 fit survives, with its derivatives converted to the spin density

    e_1, de1_dr, d2e1_dr2 = calculate_PW_correlation_derivatives(spin_density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)

    r_s, _ = calculate_seitz_radius(spin_density)

    dr_dn = -r_s * inv_spin_density / 3
    d2r_dn2 = (4 / 9) * r_s * inv_spin_density * inv_spin_density

    de1_dn = de1_dr * dr_dn
    d2e1_dn2 = d2e1_dr2 * dr_dn * dr_dn + de1_dr * d2r_dn2

    # The ratio B = beta / gamma, at the Seitz radius of the spin density for revised TPSS

    if revised:

        beta_denominator = 1 + 0.1778 * r_s

        dbeta_dr = 0.066725 * (0.1 - 0.1778) / (beta_denominator * beta_denominator)
        d2beta_dr2 = -2 * 0.1778 * dbeta_dr / beta_denominator

        B = 0.066725 * (1 + 0.1 * r_s) / beta_denominator / gamma
        dB_dn = dbeta_dr * dr_dn / gamma
        d2B_dn2 = (d2beta_dr2 * dr_dn * dr_dn + dbeta_dr * d2r_dn2) / gamma

    else:

        B, dB_dn, d2B_dn2 = 0.066725 / gamma, 0, 0

    # A = B / (exp(-W) - 1) with W = e_1 / (gamma phi^3)

    W = e_1 / gamma_phi_cubed
    dW_dn = de1_dn / gamma_phi_cubed
    d2W_dn2 = d2e1_dn2 / gamma_phi_cubed

    exp_minus_W = np.exp(-W)
    inv_w_1 = 1 / np.expm1(-W)

    dinv_dW = exp_minus_W * inv_w_1 * inv_w_1
    d2inv_dW2 = dinv_dW * (2 * exp_minus_W * inv_w_1 - 1)

    A = B * inv_w_1
    dA_dn = dB_dn * inv_w_1 + B * dinv_dW * dW_dn
    d2A_dn2 = d2B_dn2 * inv_w_1 + 2 * dB_dn * dinv_dW * dW_dn + B * (d2inv_dW2 * dW_dn * dW_dn + dinv_dW * d2W_dn2)

    # Reduced density gradient T = t^2 of the channel

    k_F = calculate_Fermi_wavevector(density=spin_density)

    dT_ds = np.pi / (16 * phi * phi * k_F * spin_density * spin_density)
    T = spin_sigma * dT_ds

    dT_dn = -(7 / 3) * T * inv_spin_density
    d2T_dn2 = (70 / 9) * T * inv_spin_density * inv_spin_density
    d2T_dnds = -(7 / 3) * dT_ds * inv_spin_density

    # The logged quantity X = B * Y, with the PW91 rational function Y

    Y, dY_dT, dY_dA, d2Y_dT2, d2Y_dTdA, d2Y_dA2 = calculate_PW91_X_derivatives(T, A, 1)

    dY_dn = dY_dT * dT_dn + dY_dA * dA_dn
    dY_ds = dY_dT * dT_ds
    d2Y_dn2 = d2Y_dT2 * dT_dn * dT_dn + 2 * d2Y_dTdA * dT_dn * dA_dn + d2Y_dA2 * dA_dn * dA_dn + dY_dT * d2T_dn2 + dY_dA * d2A_dn2
    d2Y_dnds = d2Y_dT2 * dT_dn * dT_ds + d2Y_dTdA * dT_ds * dA_dn + dY_dT * d2T_dnds
    d2Y_ds2 = d2Y_dT2 * dT_ds * dT_ds

    X = B * Y

    dX_dn = dB_dn * Y + B * dY_dn
    dX_ds = B * dY_ds
    d2X_dn2 = d2B_dn2 * Y + 2 * dB_dn * dY_dn + B * d2Y_dn2
    d2X_dnds = dB_dn * dY_ds + B * d2Y_dnds
    d2X_ds2 = B * d2Y_ds2

    # Derivatives of L = log(1 + X), and the energy e_1 + gamma phi^3 L

    inv_one_plus_X = 1 / (1 + X)

    dL_dn = dX_dn * inv_one_plus_X
    dL_ds = dX_ds * inv_one_plus_X

    e_C = e_1 + gamma_phi_cubed * np.log1p(X)

    de_dn = de1_dn + gamma_phi_cubed * dL_dn
    de_ds = gamma_phi_cubed * dL_ds
    d2e_dn2 = d2e1_dn2 + gamma_phi_cubed * (d2X_dn2 * inv_one_plus_X - dL_dn * dL_dn)
    d2e_dnds = gamma_phi_cubed * (d2X_dnds * inv_one_plus_X - dL_dn * dL_ds)
    d2e_ds2 = gamma_phi_cubed * (d2X_ds2 * inv_one_plus_X - dL_ds * dL_ds)

    return e_C, de_dn, de_ds, d2e_dn2, d2e_dnds, d2e_ds2










def calculate_unrestricted_TPSS_correlation_kernel(alpha_density: ndarray, beta_density: ndarray, density: ndarray, sigma_aa: ndarray, sigma_bb: ndarray, sigma_ab: ndarray, tau_alpha: ndarray, tau_beta: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the spin-resolved TPSS or revised TPSS correlation kernel for an unrestricted reference. The functional only depends on the total
    kinetic energy density, tau, so its blocks are the same for tau alpha and tau beta.

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
        f_C_sigma_aa_sigma_aa (array): Second derivative with respect to sigma alpha-alpha
        f_C_sigma_aa_sigma_bb (array): Mixed second derivative with respect to sigma alpha-alpha and sigma beta-beta
        f_C_sigma_aa_sigma_ab (array): Mixed second derivative with respect to sigma alpha-alpha and sigma alpha-beta
        f_C_sigma_bb_sigma_bb (array): Second derivative with respect to sigma beta-beta
        f_C_sigma_bb_sigma_ab (array): Mixed second derivative with respect to sigma beta-beta and sigma alpha-beta
        f_C_sigma_ab_sigma_ab (array): Second derivative with respect to sigma alpha-beta
        f_C_alpha_tau (array): Mixed second derivative with respect to the alpha density and tau
        f_C_beta_tau (array): Mixed second derivative with respect to the beta density and tau
        f_C_sigma_aa_tau (array): Mixed second derivative with respect to sigma alpha-alpha and tau
        f_C_sigma_bb_tau (array): Mixed second derivative with respect to sigma beta-beta and tau
        f_C_sigma_ab_tau (array): Mixed second derivative with respect to sigma alpha-beta and tau
        f_C_tau_tau (array): Second derivative with respect to tau

    """

    revised = calculation.functional.c_functional == "REVTPSS"

    d = 2.8

    # Coefficients of C(zeta, 0) in powers of zeta squared, which revised TPSS changes

    c_0 = [0.53, 0.9269, 0.6225, 2.1540] if revised else [0.53, 0.87, 0.50, 2.26]

    zeros = np.zeros_like(density)

    inv_density = 1 / density
    inv_density_squared = inv_density * inv_density
    inv_density_cubed = inv_density_squared * inv_density

    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=constants.sigma_floor)
    tau = tau_alpha + tau_beta

    # Derivatives are stored over the variables alpha, beta, sigma_aa, sigma_bb, sigma_ab and tau, with symmetric tables of second derivatives

    def table(entries: dict) -> list:

        second = [[zeros] * 6 for _ in range(6)]

        for (i, j), value in entries.items():

            second[i][j] = second[j][i] = value

        return second

    # The total square gradient is sigma_aa + sigma_bb + 2 * sigma_ab

    sigma_weights = [0, 0, 1, 1, 2, 0]

    # PBE correlation at the actual spin densities, mapped from the total square gradient onto its three components

    e_P, dP_da, dP_db, dP_ds, d2P_da2, d2P_dadb, d2P_db2, d2P_dads, d2P_dbds, d2P_ds2 = calculate_TPSS_PBE_correlation_derivatives(alpha_density, beta_density, density, sigma, revised)

    dP = [dP_da, dP_db, dP_ds, dP_ds, 2 * dP_ds, zeros]

    d2P = table({(0, 0): d2P_da2, (0, 1): d2P_dadb, (1, 1): d2P_db2})

    for k in (2, 3, 4):

        d2P[0][k] = d2P[k][0] = sigma_weights[k] * d2P_dads
        d2P[1][k] = d2P[k][1] = sigma_weights[k] * d2P_dbds

        for l in (2, 3, 4):

            d2P[k][l] = sigma_weights[k] * sigma_weights[l] * d2P_ds2

    # Fully polarised PBE correlation of each channel

    e_A, dA_dn, dA_ds, d2A_dn2, d2A_dnds, d2A_ds2 = calculate_polarised_PBE_correlation_derivatives(alpha_density, sigma_aa, revised)
    e_B, dB_dn, dB_ds, d2B_dn2, d2B_dnds, d2B_ds2 = calculate_polarised_PBE_correlation_derivatives(beta_density, sigma_bb, revised)

    dA = [dA_dn, zeros, dA_ds, zeros, zeros, zeros]
    dB = [zeros, dB_dn, zeros, dB_ds, zeros, zeros]

    d2A = table({(0, 0): d2A_dn2, (0, 2): d2A_dnds, (2, 2): d2A_ds2})
    d2B = table({(1, 1): d2B_dn2, (1, 3): d2B_dnds, (3, 3): d2B_ds2})

    # The TPSS maximum is taken separately against each fully polarised channel, point by point along with its derivatives

    alpha_uses_PBE = e_P >= e_A
    beta_uses_PBE = e_P >= e_B

    e_tilde_A = np.where(alpha_uses_PBE, e_P, e_A)
    e_tilde_B = np.where(beta_uses_PBE, e_P, e_B)

    de_tilde_A = [np.where(alpha_uses_PBE, dP[i], dA[i]) for i in range(6)]
    de_tilde_B = [np.where(beta_uses_PBE, dP[i], dB[i]) for i in range(6)]

    d2e_tilde_A = [[np.where(alpha_uses_PBE, d2P[i][j], d2A[i][j]) for j in range(6)] for i in range(6)]
    d2e_tilde_B = [[np.where(beta_uses_PBE, d2P[i][j], d2B[i][j]) for j in range(6)] for i in range(6)]

    # The weighted e_tilde = (n_alpha e_tilde_A + n_beta e_tilde_B) / n, with weights x_A = n_alpha / n and x_B = 1 - x_A

    x_A = alpha_density * inv_density
    x_B = beta_density * inv_density

    dx_A = [beta_density * inv_density_squared, -alpha_density * inv_density_squared, zeros, zeros, zeros, zeros]
    d2x_A = table({(0, 0): -2 * beta_density * inv_density_cubed, (0, 1): (alpha_density - beta_density) * inv_density_cubed, (1, 1): 2 * alpha_density * inv_density_cubed})

    e_T = x_A * e_tilde_A + x_B * e_tilde_B

    dE = [dx_A[i] * (e_tilde_A - e_tilde_B) + x_A * de_tilde_A[i] + x_B * de_tilde_B[i] for i in range(6)]

    d2E = [[d2x_A[i][j] * (e_tilde_A - e_tilde_B) + dx_A[i] * (de_tilde_A[j] - de_tilde_B[j]) + dx_A[j] * (de_tilde_A[i] - de_tilde_B[i]) + x_A * d2e_tilde_A[i][j] + x_B * d2e_tilde_B[i][j] for j in range(6)] for i in range(6)]

    # The spin factor C(zeta, xi) = C_0(zeta) / (1 + A)^4. Here A = xi^2 s(zeta) / 2, which is P_q * (n_a^(-4/3) + n_b^(-4/3)) / (2^(7/3) (3 pi^2)^(2/3) n^(10/3)) with P_q = n_b^2 sigma_aa - 2 n_a n_b sigma_ab + n_a^2 sigma_bb

    kappa_A = 1 / (np.cbrt(2) ** 7 * np.cbrt(3 * np.pi ** 2) ** 2)

    P_q = beta_density * beta_density * sigma_aa - 2 * alpha_density * beta_density * sigma_ab + alpha_density * alpha_density * sigma_bb

    dP_q = [2 * alpha_density * sigma_bb - 2 * beta_density * sigma_ab, 2 * beta_density * sigma_aa - 2 * alpha_density * sigma_ab, beta_density * beta_density, alpha_density * alpha_density, -2 * alpha_density * beta_density, zeros]
    d2P_q = table({(0, 0): 2 * sigma_bb, (0, 1): -2 * sigma_ab, (1, 1): 2 * sigma_aa, (0, 3): 2 * alpha_density, (0, 4): -2 * beta_density, (1, 2): 2 * beta_density, (1, 4): -2 * alpha_density})

    # The density factor S_n = (n_a^(-4/3) + n_b^(-4/3)) n^(-10/3)

    inv_cbrt_alpha = 1 / np.cbrt(alpha_density)
    inv_cbrt_beta = 1 / np.cbrt(beta_density)

    S = inv_cbrt_alpha ** 4 + inv_cbrt_beta ** 4
    dS_da = -(4 / 3) * inv_cbrt_alpha ** 7
    dS_db = -(4 / 3) * inv_cbrt_beta ** 7
    d2S_da2 = (28 / 9) * inv_cbrt_alpha ** 10
    d2S_db2 = (28 / 9) * inv_cbrt_beta ** 10

    N = inv_density ** (10 / 3)
    dN = -(10 / 3) * N * inv_density
    d2N = (130 / 9) * N * inv_density_squared

    S_n = S * N

    dS_n = [dS_da * N + S * dN, dS_db * N + S * dN, zeros, zeros, zeros, zeros]
    d2S_n = table({(0, 0): d2S_da2 * N + 2 * dS_da * dN + S * d2N, (0, 1): (dS_da + dS_db) * dN + S * d2N, (1, 1): d2S_db2 * N + 2 * dS_db * dN + S * d2N})

    A = kappa_A * P_q * S_n

    dA_C = [kappa_A * (dP_q[i] * S_n + P_q * dS_n[i]) for i in range(6)]
    d2A_C = [[kappa_A * (d2P_q[i][j] * S_n + dP_q[i] * dS_n[j] + dP_q[j] * dS_n[i] + P_q * d2S_n[i][j]) for j in range(6)] for i in range(6)]

    # The factor (1 + A)^(-4)

    inv_one_plus_A = 1 / (1 + A)

    Y = inv_one_plus_A ** 4
    dY_dA = -4 * Y * inv_one_plus_A
    d2Y_dA2 = 20 * Y * inv_one_plus_A * inv_one_plus_A

    dY = [dY_dA * dA_C[i] for i in range(6)]
    d2Y = [[d2Y_dA2 * dA_C[i] * dA_C[j] + dY_dA * d2A_C[i][j] for j in range(6)] for i in range(6)]

    # The polynomial C_0(zeta) and the derivatives of zeta = (n_a - n_b) / n

    zeta = (alpha_density - beta_density) * inv_density
    zeta_squared = zeta * zeta

    C_0 = c_0[0] + (c_0[1] + (c_0[2] + c_0[3] * zeta_squared) * zeta_squared) * zeta_squared
    dC_0_dzeta = (2 * c_0[1] + (4 * c_0[2] + 6 * c_0[3] * zeta_squared) * zeta_squared) * zeta
    d2C_0_dzeta2 = 2 * c_0[1] + (12 * c_0[2] + 30 * c_0[3] * zeta_squared) * zeta_squared

    dzeta = [2 * beta_density * inv_density_squared, -2 * alpha_density * inv_density_squared, zeros, zeros, zeros, zeros]
    d2zeta = table({(0, 0): -4 * beta_density * inv_density_cubed, (0, 1): 2 * (alpha_density - beta_density) * inv_density_cubed, (1, 1): 4 * alpha_density * inv_density_cubed})

    dC_0 = [dC_0_dzeta * dzeta[i] for i in range(6)]
    d2C_0 = [[d2C_0_dzeta2 * dzeta[i] * dzeta[j] + dC_0_dzeta * d2zeta[i][j] for j in range(6)] for i in range(6)]

    # The spin factor C = C_0 (1 + A)^(-4)

    C = C_0 * Y

    dC = [dC_0[i] * Y + C_0 * dY[i] for i in range(6)]
    d2C = [[d2C_0[i][j] * Y + dC_0[i] * dY[j] + dC_0[j] * dY[i] + C_0 * d2Y[i][j] for j in range(6)] for i in range(6)]

    # The TPSS variable z = sigma / (8 n tau), which depends on the density, all three square gradients and tau

    inv_tau = 1 / tau

    z = sigma * inv_density * inv_tau / 8
    dz_dsigma = inv_density * inv_tau / 8

    dz = [-z * inv_density, -z * inv_density, dz_dsigma, dz_dsigma, 2 * dz_dsigma, -z * inv_tau]

    d2z = table({(0, 0): 2 * z * inv_density_squared, (0, 1): 2 * z * inv_density_squared, (1, 1): 2 * z * inv_density_squared, (0, 5): z * inv_density * inv_tau, (1, 5): z * inv_density * inv_tau, (5, 5): 2 * z * inv_tau * inv_tau})

    for k in (2, 3, 4):

        d2z[0][k] = d2z[k][0] = d2z[1][k] = d2z[k][1] = -sigma_weights[k] * dz_dsigma * inv_density
        d2z[k][5] = d2z[5][k] = -sigma_weights[k] * dz_dsigma * inv_tau

    # The TPSS energy per particle e_C = R (1 + d z^3 R), with R = e_P (1 + C z^2) - (1 + C) z^2 e_T, is a function of (e_P, e_T, C, z)

    z_squared = z * z
    z_cubed = z_squared * z

    difference = e_P - e_T

    R = e_P * (1 + C * z_squared) - (1 + C) * z_squared * e_T

    R_u = [1 + C * z_squared, -(1 + C) * z_squared, z_squared * difference, 2 * z * (C * difference - e_T)]

    R_uv = [[zeros, zeros, z_squared, 2 * z * C], [zeros, zeros, -z_squared, -2 * z * (1 + C)], [z_squared, -z_squared, zeros, 2 * z * difference], [2 * z * C, -2 * z * (1 + C), 2 * z * difference, 2 * (C * difference - e_T)]]

    # Derivatives of e_C with respect to R and directly with respect to z

    e_R = 1 + 2 * d * z_cubed * R
    e_RR = 2 * d * z_cubed
    e_Rz = 6 * d * z_squared * R
    e_z = 3 * d * z_squared * R * R
    e_zz = 6 * d * z * R * R

    # First and second derivatives of e_C with respect to the four intermediates, where only z has a direct dependence

    e_u = [e_R * R_u[u] + (e_z if u == 3 else 0) for u in range(4)]

    e_uv = [[e_RR * R_u[u] * R_u[v] + e_R * R_uv[u][v] + e_Rz * (R_u[u] * (v == 3) + R_u[v] * (u == 3)) + e_zz * (u == 3) * (v == 3) for v in range(4)] for u in range(4)]

    # Chain rule onto the six variables, then the product rule for f = n * e_C

    first = [dP, dE, dC, dz]
    second = [d2P, d2E, d2C, d2z]

    de_C = [sum(e_u[u] * first[u][i] for u in range(4)) for i in range(6)]

    density_weights = [1, 1, 0, 0, 0, 0]

    index_pairs = [(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    blocks = []

    for i, j in index_pairs:

        d2e_C = sum(e_uv[u][v] * first[u][i] * first[v][j] for u in range(4) for v in range(4)) + sum(e_u[u] * second[u][i][j] for u in range(4))

        blocks.append(density * d2e_C + density_weights[i] * de_C[j] + density_weights[j] * de_C[i] + zeros)

    return tuple(blocks)










def calculate_restricted_TPSS_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted TPSS or revised TPSS correlation kernel for singlet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        d2f_dn2 (array): Second derivative of f = n * e_C with respect to density
        d2f_dnds (array): Mixed second derivative of f = n * e_C with respect to density and sigma
        d2f_ds2 (array): Second derivative of f = n * e_C with respect to sigma
        d2f_dndt (array): Mixed second derivative of f = n * e_C with respect to density and tau
        d2f_dsdt (array): Mixed second derivative of f = n * e_C with respect to sigma and tau
        d2f_dt2 (array): Second derivative of f = n * e_C with respect to tau

    """

    # Spin-resolved blocks at the closed shell, where each of the three square gradients is a quarter of sigma

    f_aa, f_ab, f_bb, f_a_saa, f_a_sbb, f_a_sab, f_b_saa, f_b_sbb, f_b_sab, f_saa_saa, f_saa_sbb, f_saa_sab, f_sbb_sbb, f_sbb_sab, f_sab_sab, f_a_t, f_b_t, f_saa_t, f_sbb_t, f_sab_t, f_tt = calculate_unrestricted_TPSS_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For singlet excitations each spin density changes by half, and each square gradient by a quarter, of the total change

    d2f_dn2 = (f_aa + 2 * f_ab + f_bb) / 4
    d2f_dnds = (f_a_saa + f_a_sbb + f_a_sab + f_b_saa + f_b_sbb + f_b_sab) / 8
    d2f_ds2 = (f_saa_saa + f_sbb_sbb + f_sab_sab + 2 * (f_saa_sbb + f_saa_sab + f_sbb_sab)) / 16
    d2f_dndt = (f_a_t + f_b_t) / 2
    d2f_dsdt = (f_saa_t + f_sbb_t + f_sab_t) / 4
    d2f_dt2 = f_tt

    return d2f_dn2, d2f_dnds, d2f_ds2, d2f_dndt, d2f_dsdt, d2f_dt2










def calculate_restricted_TPSS_spin_correlation_kernel(density: ndarray, sigma: ndarray, tau: ndarray, calculation: Calculation) -> tuple:

    """

    Calculates the restricted TPSS or revised TPSS correlation spin kernel for triplet excitations.

    Args:
        density (array): Electron density on integration grid
        sigma (array): Square density gradient
        tau (array): Non-interacting kinetic energy density
        calculation (Calculation): Calculation object

    Returns:
        f_mm (array): Second derivative of f = n * e_C with respect to the spin density
        f_m_sigma_nm (array): Mixed second derivative with respect to the spin density and grad(n) . grad(m)
        f_sigma_nm_sigma_nm (array): Second derivative with respect to grad(n) . grad(m)
        f_sigma_mm (array): First derivative with respect to grad(m) . grad(m)

    """

    # Spin-resolved blocks at the closed shell

    f_aa, f_ab, f_bb, f_a_saa, f_a_sbb, _, f_b_saa, f_b_sbb, _, f_saa_saa, f_saa_sbb, _, f_sbb_sbb, _, _, _, _, _, _, _, _ = calculate_unrestricted_TPSS_correlation_kernel(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    # For triplet excitations the spin densities and same-spin square gradients change in opposite directions, while sigma_ab and tau do not change

    f_mm = (f_aa - 2 * f_ab + f_bb) / 4
    f_m_sigma_nm = (f_a_saa - f_a_sbb - f_b_saa + f_b_sbb) / 4
    f_sigma_nm_sigma_nm = (f_saa_saa - 2 * f_saa_sbb + f_sbb_sbb) / 4

    # As grad(m) . grad(m) is quadratic in the response, only its first derivative enters - it adds a quarter to sigma_aa and sigma_bb and removes a quarter from sigma_ab

    correlation_functional = calculate_unrestricted_revTPSS_correlation if calculation.functional.c_functional == "REVTPSS" else calculate_unrestricted_TPSS_correlation

    _, _, df_ds_aa, df_ds_bb, df_ds_ab, _, _, _ = correlation_functional(density / 2, density / 2, density, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, calculation)

    f_sigma_mm = (df_ds_aa + df_ds_bb - df_ds_ab) / 4

    return f_mm, f_m_sigma_nm, f_sigma_nm_sigma_nm, f_sigma_mm





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
    "TPSS": calculate_TPSS_exchange_kernel,
    "REVTPSS": calculate_TPSS_exchange_kernel,
    "SCAN": calculate_SCAN_exchange_kernel,
    "RSCAN": calculate_SCAN_exchange_kernel,
    "R2SCAN": calculate_SCAN_exchange_kernel,
    "B97M": calculate_B97M_exchange_kernel,

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
    "TPSS": calculate_restricted_TPSS_correlation_kernel,
    "REVTPSS": calculate_restricted_TPSS_correlation_kernel,
    "SCAN": calculate_restricted_SCAN_correlation_kernel,
    "RSCAN": calculate_restricted_SCAN_correlation_kernel,
    "R2SCAN": calculate_restricted_SCAN_correlation_kernel,
    "B97M": calculate_restricted_B97M_correlation_kernel,

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
    "TPSS": calculate_restricted_TPSS_spin_correlation_kernel,
    "REVTPSS": calculate_restricted_TPSS_spin_correlation_kernel,
    "SCAN": calculate_restricted_SCAN_spin_correlation_kernel,
    "RSCAN": calculate_restricted_SCAN_spin_correlation_kernel,
    "R2SCAN": calculate_restricted_SCAN_spin_correlation_kernel,
    "B97M": calculate_restricted_B97M_spin_correlation_kernel,

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
    "TPSS": calculate_unrestricted_TPSS_correlation_kernel,
    "REVTPSS": calculate_unrestricted_TPSS_correlation_kernel,
    "SCAN": calculate_unrestricted_SCAN_correlation_kernel,
    "RSCAN": calculate_unrestricted_SCAN_correlation_kernel,
    "R2SCAN": calculate_unrestricted_SCAN_correlation_kernel,
    "B97M": calculate_unrestricted_B97M_correlation_kernel,

}
