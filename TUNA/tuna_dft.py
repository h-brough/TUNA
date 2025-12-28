from tuna_util import log, warning, error, symmetrise
import numpy as np
import sys
from scipy.integrate import lebedev_rule


"""

This is the TUNA module for density functional theory (DFT), written first for version 0.9.0.

The DFT integration grids are constructed by Legendre-Gauss quadrature for the radial parts and Lebedev quadrature for the angular parts, they
have been optimised to some extent "by hand" but are not expected to be anywhere near as efficient as proper quantum chemistry codes. Exchange and 
correlation matrices can be calculated for LDA, GGA and meta-GGA functionals, and a variety are implemented. The routines for hybrid and double-hybrid 
functionals are found in the SCF module, not here.

The module contains:

1. Some small utility functions (calculate_zeta, clean, integrate_on_grid, etc.)
2. Functions to set up the integration grid (set_up_integration_grid, build_molecular_grid, etc.)
3. Functions to evaluate the density, gradient, and kinetic energy density on the grid (construct_density_on_grid, etc.)
4. Functions to evaluate the exchange and correlation matrices (calculate_V_X, calculate_V_C, etc.)
5. The complicated, horrible, explicit functional expressions for all the exchange and correlation functionals
6. Some dictionaries for all the implemented exchange and correlation functionals

Note that throughout this module, we use ** (1 / 2) for square rooting, and np.cbrt() for cube rooting - this inconsistency is on purpose, as
these options seem to be significantly faster and more accurate than their counterparts. We also cube via x * x * x rather than x ** 3 for the same
reason, although higher powers do not benefit as much. Optimising all of these low level operations makes quite a big difference to the speed; I
observed about a factor of two increase for DFT functionals generally by optimising the cubing and cube rooting.

"""


# These are constants for cleaning grid-based arrays - the sigma floor needs to be the square of the density floor
DENSITY_FLOOR = 1e-26
SIGMA_FLOOR = DENSITY_FLOOR ** 2



def calculate_seitz_radius(density):

    # Calculates the Seitz radius for a given density, cube rooting via numpy is faster and more accurate than in pure Python

    inv_density = 1 / density

    r_s = np.cbrt(3 / (4 * np.pi) * inv_density)

    return r_s, inv_density



def calculate_zeta(alpha_density, beta_density):

    # The local spin polarisation, zeta, depends on the alpha and beta densities

    zeta = (alpha_density - beta_density) / (alpha_density + beta_density)

    # Makes sure no numerical weirdness
    zeta = np.clip(zeta, -1, 1)

    return zeta



def calculate_f_zeta(zeta):

    # This is the spin polarisation function used in eg. VWN5 correlation in interpolation

    f_zeta = (np.cbrt(1 + zeta) ** 4 + np.cbrt(1 - zeta) ** 4 - 2) / (np.cbrt(2) ** 4 - 2)

    return f_zeta



def calculate_f_prime_zeta(zeta):

    # This is the derivative of spin polarisation function used in eg. VWN5 correlation in interpolation

    f_prime_zeta = (np.cbrt(1 + zeta) - np.cbrt(1 - zeta)) / (np.cbrt(2) ** 4 - 2) * 4 / 3

    return f_prime_zeta



def clean(function_on_grid, floor=DENSITY_FLOOR):

    # Makes sure there are no zero or negative values in the electron density
    
    function_on_grid = np.maximum(function_on_grid, floor)

    return function_on_grid



def clean_density_matrix(P, S, n_electrons):

    # Forces the trace of the density matrix to be correct

    P *= n_electrons / np.trace(P @ S) if n_electrons > 0 else 0

    return P



def integrate_on_grid(integrand, weights):

    # Uses quadrature to calculate the integral of a function expressed on a grid

    integral = np.sum(integrand * weights)

    return integral





def set_up_integration_grid(basis_functions, atoms, bond_length, n_electrons, P_guess_alpha, P_guess_beta, calculation, silent=False):

    """

    High level function that sets up the integration grid and prints out information about it.

    Args:
        basis_functions (list): List of basis function objects
        atoms (list): List of atoms
        bond_length (float): Bond length of the molecule
        n_electrons (int): Number of electrons
        P_guess (array): Density matrix in AO basis for guess
        calculation (Calculation): Calculation object#
        silent (bool, optional): Should anything be printed
    
    Returns:
        bfs_on_grid: Basis functions evaluated on grid, shape (n_basis, n_radial, n_angular)
        weights: Integration weights for grid points, shape (n_radial, n_angular)
        bf_gradients_on_grid: Basis function gradients evaluated on grid (3, n_basis, n_radial, n_angular)

    """


    log(f" Setting up DFT integration grid with \"{calculation.grid_conv["name"]}\" accuracy...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Reads the integration grid parameters from the requested convergence criteria
    extent_multiplier = calculation.grid_conv["extent_multiplier"]
    integral_accuracy = calculation.grid_conv["integral_accuracy"] if not calculation.integral_accuracy_requested else calculation.integral_accuracy

    # The extent of the grid away from the nucleus is given by this function, which is homemade. It can be directly modified via the extent_multiplier.
    extent = extent_multiplier * np.max([atoms[i].real_vdw_radius for i in range(len(atoms))]) / 6

    # Uses the integral accuracy to map to a particular Lebedev order
    n = int(integral_accuracy * 9)

    LEBEDEV_ORDERS = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])

    idx = np.abs(LEBEDEV_ORDERS - n).argmin()
    Lebedev_order = int(LEBEDEV_ORDERS[idx])

    # Uses the integral accuracy to map to a number of radial points, scaled by extent of the grid to keep the density the same for larger atoms
    n_radial = int(extent * integral_accuracy)

    # Builds the molecular grid - the atomic grids are the same for both atoms even if heteroatomic
    points, weights = build_molecular_grid(extent, n_radial, Lebedev_order, bond_length, atoms)

    log("[Done]", calculation, 1, silent=silent)

    # Calculates the total number of grid points, and the number per atom
    total_points = points.shape[1] * points.shape[2]
    points_per_atom = total_points // len(atoms)

    log(f"\n Integration grid has {n_radial} radial and {points.shape[2]} angular points, a Lebedev order of {Lebedev_order}.", calculation, 1, silent=silent)
    log(f" In total there are {total_points} grid points, {points_per_atom} per atom.", calculation, 1, silent=silent)

    log("\n Building guess density on grid...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Calculates the basis functions expressed on the integration grid
    bfs_on_grid = construct_basis_functions_on_grid(basis_functions, points)
    
    # If a (meta-)GGA calculation has been requested, determines the basis functions expressed on the integration grid
    bf_gradients_on_grid = construct_basis_function_gradients_on_grid(basis_functions, points) if calculation.functional.functional_class in ["GGA", "meta-GGA"] else None

    # Constructs the electron density, using the guess density matrix, on the grid
    alpha_density = construct_density_on_grid(P_guess_alpha, bfs_on_grid)
    beta_density = construct_density_on_grid(P_guess_beta, bfs_on_grid)

    density = alpha_density + beta_density

    log("[Done]", calculation, 1, silent=silent)

    # Integrates the grid to get the number of electrons
    n_alpha_DFT = integrate_on_grid(alpha_density, weights)
    n_beta_DFT = integrate_on_grid(beta_density, weights)

    n_electrons_DFT = integrate_on_grid(density, weights)

    log(f"\n Integral of the guess alpha density: {n_alpha_DFT:13.10f}", calculation, 1, silent=silent)
    log(f" Integral of the guess beta density:  {n_beta_DFT:13.10f}\n", calculation, 1, silent=silent)

    log(f" Integral of the guess total density: {n_electrons_DFT:13.10f}\n", calculation, 1, silent=silent)

    # Prints a warning of the density integral is a bit dodgy, and throws and error if its totally wrong
    if np.abs(n_electrons_DFT - n_electrons) > 0.00001:

        warning(" Integral of density is far from the number of electrons! Be careful with your results.")

        if np.abs(n_electrons_DFT - n_electrons) > 0.5:

            error("Integral for the density is completely wrong!")
    

    return bfs_on_grid, weights, bf_gradients_on_grid









def build_atomic_radial_and_angular_grid(radial_grid_cutoff, n_radial, lebedev_order, radial_power=3):

    """

    Sets up the radial and angular integration grid for an atom. The default radial power of 3 seems to work
    better than either 2 or 4 on average, but hasn't seriously been optimised.

    Args:
        radial_grid_cutoff (float): Extent of radial grid
        n_radial (int): Number of radial grid points
        lebedev_order (int): Order of Lebedev quadrature
        radial_power (float, optional): Tightness near nucleus of radial grid (higher -> tighter)

    Returns:
        atomic_points (array): Grid points for atom, shape (N_radial, N_angular)
        atomic_weights (array): Integration weights for each grid point, shape (N_radial, N_angular)

    """

    # This gives quadrature between -1 and 1
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_radial)

    # Radial quadrature is mapped between 0 and 1
    t = (t_nodes + 1) / 2  
    w_t = t_weights / 2

    # Multiplying by radial cutoff maps it to the radial grid cutoff, and the power term keeps points close to the nucleus
    r = radial_grid_cutoff * t ** radial_power
    dr_dt = radial_grid_cutoff * radial_power * t ** (radial_power - 1)
    weights_radial = w_t * dr_dt       
    
    # Uses Lebedev quadrature to get the directions and weights
    unit_sphere_directions, weights_angular = lebedev_rule(lebedev_order)

    # Builds the Cartesian grid out of the radial and angular grids
    atomic_points = np.einsum("m,in->imn", r, unit_sphere_directions, optimize=True)

    # Radial weights scaled by R^2 to account for surface of sphere getting larger away from nucleus
    atomic_weights = np.einsum("m,m,n->mn", weights_radial, r ** 2, weights_angular, optimize=True)

    return atomic_points, atomic_weights









def calculate_Becke_diatomic_weights(X, Y, Z, bond_length, atoms, steepness=4):

    """

    Calculates the relative weights for each atom in the diatomic, using Becke's method. The default steepness of 4
    is higher than Becke's recommended value of 3, but seems to perform better for diatomic molecules (although this has
    not been tested thoroughly).

    Args:
        X (array): Cartesian grid for x axis
        Y (array): Cartesian grid for y axis
        Z (array): Cartesian grid for z axis
        bond_length (float): Bond length of molecule
        atoms (list): List of atoms
        steepness (int, optional): How many times should "steepening" function be applied

    Returns:
        weights_A (array): Weights for first atom in diatomic
        weights_B (array): Weights for second atom in diatomic
    
    """

    # Distance to the atomic centres for each point on the Cartesian grid
    R_A = (X * X + Y * Y + Z * Z) ** (1 / 2)
    R_B = (X * X + Y * Y + (Z - bond_length) * (Z - bond_length)) ** (1 / 2)

    # This is -1 on atom A, 1 on atom B and 0 in the centre. It is an elliptical coordinate
    s = (R_A - R_B) / bond_length

    # The values of Van der Waals radius are taken as a proxy for atomic size, chi is the ratio between sizes
    chi = atoms[0].real_vdw_radius / atoms[1].real_vdw_radius

    # These equations are straight from Becke's paper on integration weights for heteroatomic systems
    u = (chi - 1) / (chi + 1)
    a = u / (u * u - 1)
    s = s + a * (1 - s * s)

    # The more this function is applied, the steeper the transition becomes and the more highly weighted points around the nuclei
    for i in range(steepness):

        s = (3 * s - s * s * s) / 2 
 
    # These weights are 1 on atom A, 0 on atom B and 0.5 when the elliptical coordinate is zero. Vice versa for B.
    weights_A = (1 - s) / 2
    weights_B = (1 + s) / 2


    return weights_A, weights_B








def build_molecular_grid(radial_grid_cutoff, n_radial, lebedev_order, bond_length, atoms):

    """

    Sets up the molecular grid for a diatomic molecule.

    Args:

        radial_grid_cutoff (float): Extent of radial grid
        n_radial (int): Number of radial grid points
        lebedev_order (int): Order of Lebedev quadrature
        bond_length (float): Bond length of diatomic
        atoms (list): List of atoms

    Returns:
        points (array): Molecular grid points
        weights (array): Weights for molecular grid points

    """

    # Builds the radial and angular grid for the first atom
    points_A, atomic_weights_A = build_atomic_radial_and_angular_grid(radial_grid_cutoff, n_radial, lebedev_order)

    # Extracts grid points into Cartesian components
    X_A, Y_A, Z_A = points_A

    # If its just an atomic calculation, return this grid
    if len(atoms) == 1 or len(atoms) == 2 and any(atom.ghost for atom in atoms):

        return points_A, atomic_weights_A
    
    # For a diatomic, the grid will be the same (we don't optimize atomic grids for atomic species)
    X_B, Y_B, Z_B = X_A, Y_A, Z_A + bond_length
    atomic_weights_B = atomic_weights_A

    # Builds the molecular Cartesian grid
    X = np.concatenate([X_A, X_B], axis=0)
    Y = np.concatenate([Y_A, Y_B], axis=0)
    Z = np.concatenate([Z_A, Z_B], axis=0)

    points = np.stack((X, Y, Z), axis=0)

    # Uses Becke atomic partitioning to get weights for each atom
    diatomic_weights_A, diatomic_weights_B = calculate_Becke_diatomic_weights(X, Y, Z, bond_length, atoms)

    n_points_atom_A = X_A.shape[0]

    # Combines the atomic weights with the molecular weights - both are between 0 and 1, so the combined weights are too
    weights_combined_A = atomic_weights_A * diatomic_weights_A[:n_points_atom_A]
    weights_combined_B = atomic_weights_B * diatomic_weights_B[n_points_atom_A:]

    # Forms total molecular weights array
    weights = np.concatenate([weights_combined_A, weights_combined_B], axis=0)

    return points, weights








def construct_basis_functions_on_grid(basis_functions, points):
    
    """
    
    Expresses the basis functions on the integration grid.

    Args:
        basis_functions (list): List of basis function objects
        points (array): Integration grid points
    
    Returns:
        bfs_on_grid (array): Basis functions expressed on integration grid, shape (basis_size, radial_points, angular_points)
    
    """

    # For DFT, points is three-dimensional but for plotting, it is two-dimensional (X, Z)
    X, Y, Z = points if len(points) == 3 else (points[0], np.full_like(points[0], 0), points[1])

    # These are the number of radial and angular points respectively for DFT, or number of Cartesian X and Z points for outputs
    _, N, M = points.shape

    # Initialises the array for basis functions on a grid
    bfs_on_grid = np.zeros((len(basis_functions), N, M))


    for i, bf in enumerate(basis_functions):

        # Defines the X, Y and Z coordinates with respect to the origin of the basis function
        X_relative = X - bf.origin[0]
        Y_relative = Y - bf.origin[1]
        Z_relative = Z - bf.origin[2]

        # These are the angular momentum exponents
        l, m, n = bf.shell

        # The shared distance squared from the basis function origin
        r_squared = X_relative * X_relative + Y_relative * Y_relative + Z_relative * Z_relative

        # The shared exponent term for the primitive Gaussians
        exponent_term = np.exp(-1 * np.einsum("i,jk->ijk", bf.exps, r_squared, optimize=True))

        # Basis functions are a product of the coefficient, norm, angular part and radial exponent part
        bfs_on_grid[i] = np.einsum("i,i,jk,jk,jk,ijk->jk", bf.coefs, bf.norm, X_relative ** l, Y_relative ** m, Z_relative ** n, exponent_term, optimize=True)


    return bfs_on_grid










def construct_basis_function_gradients_on_grid(basis_functions, points):
    
    """
    
    Expresses the analytic gradients of the basis functions on the integration grid.

    Args:
        basis_functions (list): List of basis function objects
        points (array): Integration grid points
    
    Returns:
        bf_gradients_on_grid (array): Basis function gradients on integration grid, shape (3, basis_size, radial_points, angular_points)
    
    """

    # For DFT, points is three-dimensional but for plotting, it is two-dimensional (X, Z)
    X, Y, Z = points if len(points) == 3 else (points[0], np.full_like(points[0], 0), points[1])

    # These are the number of radial and angular points respectively for DFT, or number of Cartesian X and Z points for outputs
    _, N, M = points.shape

    n_basis = len(basis_functions)

    # Initialises basis function gradient arrays
    bf_gradients_on_grid = np.zeros((n_basis, 3, N, M))

    for i, bf in enumerate(basis_functions):

        # Defines the X, Y and Z coordinates with respect to the origin of the basis function
        X_relative = X - bf.origin[0]
        Y_relative = Y - bf.origin[1]
        Z_relative = Z - bf.origin[2]

        # These are the angular momentum exponents
        l, m, n = bf.shell

        # The shared distance squared from the basis function origin
        r_squared = X_relative * X_relative + Y_relative * Y_relative + Z_relative * Z_relative

        # The shared exponent term for the primitive Gaussians
        exponent_term = np.exp(-1 * np.einsum("i,jk->ijk", bf.exps, r_squared, optimize=True))

        poly_x = X_relative ** l 
        poly_y = Y_relative ** m 
        poly_z = Z_relative ** n

        P_poly = poly_x * poly_y * poly_z    

        # Derivative polynomials for x, y and z
        dP_dx_poly = l * (X_relative ** (l - 1)) * poly_y * poly_z if l > 0 else np.zeros_like(P_poly)
        dP_dy_poly = m * poly_x * (Y_relative ** (m - 1)) * poly_z if m > 0 else np.zeros_like(P_poly)
        dP_dz_poly = n * poly_x * poly_y * (Z_relative ** (n - 1)) if n > 0 else np.zeros_like(P_poly)

        # Primitives for the x, y and z derivatives
        primitives_dx = exponent_term * (dP_dx_poly - 2 * bf.exps[:, None, None] * X_relative * P_poly)
        primitives_dy = exponent_term * (dP_dy_poly - 2 * bf.exps[:, None, None] * Y_relative * P_poly)
        primitives_dz = exponent_term * (dP_dz_poly - 2 * bf.exps[:, None, None] * Z_relative * P_poly)

        # Package the three gradient components together for final contraction
        primitives = np.array([primitives_dx, primitives_dy, primitives_dz])

        # Builds gradients on grid via primitives, norms and coefficients
        bf_gradients_on_grid[i] = np.einsum("i,i,aijk->ajk", bf.coefs, bf.norm, primitives, optimize=True)

    # Shuffles and unpacks the gradients into Cartesian components
    bf_gradients_on_grid = bf_gradients_on_grid.swapaxes(0, 1)

    return bf_gradients_on_grid









def construct_density_on_grid(P, bfs_on_grid, clean_density=True):
    
    """
    
    Constructs the electron density on the grid using the atomic orbitals, then cleans it up.

    Args:
        P (array): Density matrix in AO basis
        bfs_on_grid (array): Basis functions on integration grid
        clean_density (bool, optional): Should the density be cleaned
    
    Returns:
        density (array): Electron density on molecular grid
    
    """

    # Conventional expression for the electron density in terms of basis functions - using P encodes that only occupied orbitals are summed
    density = np.einsum("ij,ikl,jkl->kl", P, bfs_on_grid, bfs_on_grid, optimize=True)

    # This is on by default to get rid of very small, zero and negative values that break functionals
    if clean_density:
        
        density = clean(density)


    return density








def calculate_density_gradient(P, bfs_on_grid, bf_gradients_on_grid):

    """

    Calculates the density gradient, and its square, on the integration grid.

    Args:
        P (array): Density matrix in AO basis
        bfs_on_grid (array): Basis functions evaluated on grid
        bf_gradients_on_grid (array): Basis function gradients evaluated on grid
    
    Returns:
        sigma (array): Square density gradient, shape (n_radial, n_angular)
        density_gradient (array): Gradient of density on grid, shape (3, n_radial, n_angular)

    """

    # Constructs the density gradient on a grid analytically - the factor of two is due to the symmetry of the density matrix
    density_gradient = 2 * np.einsum("ij,ikl,ajkl->akl", P, bfs_on_grid, bf_gradients_on_grid, optimize=True)

    # Builds the square density gradient, used in GGA functionals
    sigma = np.einsum("akl->kl", density_gradient * density_gradient, optimize=True)

    # It is important that this is cleaned at the square of the floor that the density and kinetic energy density are cleaned
    # Quantities that rely on ratios between the density and its gradient will break if they're cleaned equally
    sigma = clean(sigma, floor=SIGMA_FLOOR)

    return sigma, density_gradient








def calculate_kinetic_energy_density(P, bf_gradients_on_grid):

    """

    Calculates the density gradient, and its square, on the integration grid.

    Args:
        P (array): Density matrix in AO basis
        bf_gradients_on_grid (array): Basis function gradients evaluated on grid
    
    Returns:
        tau (array): Non-interacting kinetic energy density, shape (n_radial, n_angular)

    """

    # Conventional expression, with factor of a half, for non-interacting kinetic energy density used in meta-GGA functionals
    tau = (1 / 2) * np.einsum("ij,aikl,ajkl->kl", P, bf_gradients_on_grid, bf_gradients_on_grid, optimize=True)
    
    # This needs to be cleaned with the same floor as the density, higher than sigma
    tau = clean(tau, floor=DENSITY_FLOOR)

    return tau








def calculate_V_X(weights, bfs_on_grid, df_dn, df_ds, df_dt, bf_gradients_on_grid, density_gradient):
    
    """
    
    Calculates the density functional theory exchange matrix. Separately integrates the LDA, GGA and meta-GGA contributions.

    Args:
        weights (array): Integration weights 
        bfs_on_grid (array): Basis functions evaluated on integration grid
        df_dn (array): Derivative of n * e_X with respect to the density
        df_ds (array): Derivative of n * e_X with respect to the square density gradient, sigma
        df_dt (array): Derivative of n * e_X with respect to the kinetic energy density, tau
        bf_gradients_on_grid (array): Gradient of basis functions evaluated on integration grid
        density_gradient (array): Gradient of density evaluated on integration grid      

    Returns:
        V_X (array): Symmetrised density functional theory exchange matrix in AO basis  
    
    """

    # Contribution to exchange matrix from LDA part of functional
    V_X = np.einsum("kl,mkl,nkl,kl->mn", df_dn, bfs_on_grid, bfs_on_grid, weights, optimize=True)

    if df_ds is not None:

        # Contribution to exchange matrix from GGA part of functional
        V_X += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize=True)

    if df_dt is not None:
        
        # Contribution to exchange matrix from meta-GGA part of functional

        V_X += (1 / 2) * np.einsum("kl,amkl,ankl,kl->mn", df_dt, bf_gradients_on_grid, bf_gradients_on_grid, weights, optimize=True)

    # Absolutely necessary to symmetrise as these are not symmetric by design
    V_X = symmetrise(V_X)

    return V_X








def calculate_V_C(weights, bfs_on_grid, df_dn, df_ds, df_dt, bf_gradients_on_grid, density_gradient, density_gradient_other_spin=None, df_ds_ab=None):
    
    """
    
    Calculates the density functional theory correlation matrix. Separately integrates the LDA, GGA and meta-GGA contributions.

    Args:
        weights (array): Integration weights 
        bfs_on_grid (array): Basis functions evaluated on integration grid
        df_dn (array): Derivative of n * e_C with respect to the density
        df_ds (array): Derivative of n * e_C with respect to the square density gradient, sigma
        df_dt (array): Derivative of n * e_C with respect to the kinetic energy density, tau
        bf_gradients_on_grid (array): Gradient of basis functions evaluated on integration grid
        density_gradient (array): Gradient of density evaluated on integration grid      
        density_gradient_other_spin (array, optional): Gradient of density of other spin evaluated on integration grid      
        df_ds_ab (array, optional): Derivative of n * e_C with respect to densgrad_alpha dot densgrad_beta 

    Returns:
        V_C (array): Symmetrised density functional theory correlation matrix in AO basis  
    
    """

    # Contribution to exchange matrix from LDA part of functional
    V_C = np.einsum("kl,mkl,nkl,kl->mn", df_dn, bfs_on_grid, bfs_on_grid, weights, optimize=True)

    if df_ds is not None:

        # Contribution to exchange matrix from GGA part of functional

        if df_ds_ab is not None:
        
            V_C += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize=True)
            V_C += 2 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds_ab, density_gradient_other_spin, bfs_on_grid, bf_gradients_on_grid, weights, optimize=True)

        else:

            V_C += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize=True)

    if df_dt is not None:
        
        # Contribution to exchange matrix from meta-GGA part of functional

        V_C += (1 / 2) * np.einsum("kl,amkl,ankl,kl->mn", df_dt, bf_gradients_on_grid, bf_gradients_on_grid, weights, optimize=True)

    # Absolutely necessary to symmetrise as these are not symmetric by design
    V_C = symmetrise(V_C)

    return V_C







def calculate_Slater_exchange(density, sigma, tau, calculation):

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







def calculate_PBE_exchange(density, sigma, tau, calculation):

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








def calculate_B88_exchange(density, sigma, tau, calculation):
    
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








def calculate_PW91_exchange(density, sigma, tau, calculation):

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








def calculate_mPW91_exchange(density, sigma, tau, calculation):
  
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







def calculate_TPSS_exchange(density, sigma, tau, calculation):

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







def calculate_B3_exchange(density, sigma, tau, calculation):

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







def calculate_RVWN3_correlation(density, sigma, tau, calculation):

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







def calculate_UVWN3_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
   
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







def calculate_RVWN5_correlation(density, sigma, tau, calculation):

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







def calculate_UVWN5_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
   
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







def calculate_RPWLDA_correlation(density, sigma, tau, calculation):

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








def calculate_UPWLDA_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
       
    """
    
    Calculates the unrestricted PW92 correlation energy density and derivative with respect to the density.

    Args:
        alpha_density (array): Alpha electron density on integration grid
        beta_density (array): Beta electron density on integration grid
        density (array): Electron density on integration grid
    
    Returns:
        df_dn_alpha (array): Derivative of f = n * e_C with respect to alpha density
        df_dn_beta (array): Derivative of f = n * e_C with respect to beta density
        e_C (array): Unrestricted PW92 correlation energy density per particle
    
    """

    # Note - this is "modified" PW from LibXC has more significant figures than original paper

    # Parameters for a restricted (paramagnetic), fully polarised (ferromagnetic) reference, and RPA fit (alpha)
    _, e_C_0, de0_dr = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    _, e_C_1, de1_dr = calculate_PW_potential(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    _, minus_alpha, dalpha_dr = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    # Parameters for an unrestricted reference by interpolation between limits
    df_dn_alpha, df_dn_beta, e_C = calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, -dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr)


    return df_dn_alpha, df_dn_beta, None, None, None, None, None, e_C







def calculate_PW_potential(density, A, alpha_1, beta_1, beta_2, beta_3, beta_4, P):

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







def calculate_VWN_potential(density, x_0, b, c, A):

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








def calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr):

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








def calculate_PBE_correlation(density, sigma, tau, calculation):
   
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
    df_dn_LDA, _, _, e_C_LDA = calculate_RPWLDA_correlation(density, None, None, None)

    de_C_LDA_dn = (df_dn_LDA - e_C_LDA) / density

    # Key parameters for defining PBE - this value of beta is more significant figures than the original paper
    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.06672455060314922

    k_F = np.cbrt(3 * np.pi ** 2 * density)

    # Form of reduced density gradient used in PBE correlation
    t_squared = sigma * np.pi / (16 * k_F * density * density)
    t_fourth = t_squared * t_squared

    exp_factor = np.exp(-e_C_LDA / gamma)
    A = beta / (gamma * (exp_factor - 1))

    k = 1 + A * t_squared
    D = k + A * A * t_fourth

    X = (beta / gamma) * t_squared * k / D
    
    # The GGA correction to the LDA correlation energy density
    H = gamma * np.log(1 + X)

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








def calculate_UPBE_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
    
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
    
    # The PBE correlation functional only depends on the full sigma, not the spin channels. This is cleaned at the square of the density floor.
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=SIGMA_FLOOR)

    # Key parameters for PBE - gamma is exact and beta has many more significant figures than in the original PBE paper
    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.06672455060314922
    B = beta / gamma

    # Local density correlation energy density and derivatives
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_UPWLDA_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)

    # Spin polarisation
    zeta = calculate_zeta(alpha_density, beta_density)

    # Spin polarisation dependent quantities
    cbrt_plus = clean(np.cbrt(1 + zeta))
    cbrt_minus = clean(np.cbrt(1 - zeta))

    phi = (1 / 2) * (cbrt_plus * cbrt_plus + cbrt_minus * cbrt_minus)
    phi_prime = (1 / cbrt_plus - 1 / cbrt_minus) / 3

    # Repeatedly used grid quantities
    k_F = np.cbrt(3 * np.pi ** 2 * density)
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








def calculate_RLYP_correlation(density, sigma, tau, calculation):

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







def calculate_ULYP_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
    
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








def calculate_RP86_correlation(density, sigma, tau, calculation):

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
    df_LDA_dn, _, _, e_C_LDA = calculate_RPWLDA_correlation(density, None, None, None)

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








def calculate_UP86_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):

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
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=SIGMA_FLOOR)

    # Calculates the local Seitz radius
    r_s, inv_density = calculate_seitz_radius(density)
    zeta = calculate_zeta(alpha_density, beta_density)

    r_s_squared = r_s * r_s

    # Polynomial defined in P86 paper
    N = 0.002568 + alpha * r_s + beta * r_s_squared
    D = 1 + gamma * r_s + delta * r_s_squared + 1e4 * beta * r_s * r_s_squared

    C = 0.001667 + N / D
    C_inf = 0.004235

    # Phi from P86 paper
    phi = 1.745 * f_tilde * C_inf / C * sigma ** (1 / 2) / np.cbrt(density) ** (7 / 2)

    # Spin polarisation functions
    p = np.cbrt(1 + zeta)
    m = np.cbrt(1 - zeta)
    S = p ** 5 + m ** 5
    d = (S / 2) ** (1 / 2)

    # Local density correlation and derivatives
    df_dn_alpha, df_dn_beta, _,_, _, _, _, e_C_LDA = calculate_UPWLDA_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)

    # GGA correction to LDA energy density
    H = (C * sigma * np.exp(-phi) / np.cbrt(density) ** 7) / d
    e_C = e_C_LDA + H

    # Derivative with respect to sigma
    df_ds = (C * np.exp(-phi) / np.cbrt(density) ** 4 * (1 - phi / 2)) / d
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









def calculate_RPW_correlation(density, sigma, tau, calculation):
   
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
    df_dn_LDA, _, _, e_C_LDA = calculate_RPWLDA_correlation(density, None, None, None)

    # Defining constants for PW91 correlation
    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    # Various useful quantities
    beta = 16 * np.cbrt(3 / np.pi) * C_0
    r_s, inv_density = calculate_seitz_radius(density)

    k_F = np.cbrt(3 * np.pi ** 2 * density)
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









def calculate_UPW_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
    
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
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_UPWLDA_correlation(alpha_density, beta_density, density, None, None, None, None, None, None)
    
    # This functional only depends on the total square density gradient, not its spin components
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab, floor=SIGMA_FLOOR)

    C_0, C_X, alpha = 0.004235, -0.001667212, 0.09

    # Various useful quantities
    beta = 16 * np.cbrt(3 / np.pi) * C_0
    r_s, inv_density = calculate_seitz_radius(density)

    k_F = np.cbrt(3 * np.pi ** 2 * density)
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









    
def calculate_RTPSS_correlation(density, sigma, tau, calculation):

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
    df_dn_PBE, df_ds_PBE, _, e_C_PBE = calculate_PBE_correlation(density, sigma, tau, calculation)
    df_dna_one, _, df_dsaa_one, _, _, _, _, e_C_PBE_one_spin = calculate_UPBE_correlation(density / 2, zeros, density / 2, sigma / 4, zeros, zeros, None, None, None)

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







def calculate_UTPSS_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
    
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
    sigma = clean(sigma_aa + sigma_bb + 2 * sigma_ab,floor=SIGMA_FLOOR)
    tau = tau_alpha + tau_beta

    # Defining constant for TPSS
    d = 2.8

    zeros = np.zeros_like(density)

    # PBE correlation (spin-polarised)
    df_dna_PBE, df_dnb_PBE, df_dsaa_PBE, df_dsbb_PBE, df_dsab_PBE, _, _, e_C_PBE = calculate_UPBE_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, None)

    # one-spin limits for the max construction in TPSS
    df_dna_a0, _, df_dsaa_a0, _, _, _, _, e_C_a0 = calculate_UPBE_correlation(alpha_density, zeros, alpha_density, sigma_aa, zeros, zeros, None, None, None)
    _, df_dnb_0b, _, df_dsbb_0b, _, _, _, e_C_0b = calculate_UPBE_correlation(zeros, beta_density, beta_density, zeros, sigma_bb, zeros, None, None, None)

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
    one_minus = clean(1 - zeta, SIGMA_FLOOR)
    one_plus = 1 + zeta
    one_minus2 = one_minus * one_minus
    one_plus2  = one_plus * one_plus
    one_minus_z2 = 1 - zeta * zeta

    B = clean(one_minus2 * sigma_aa + one_plus2 * sigma_bb - 2 * one_minus_z2 * sigma_ab, floor=SIGMA_FLOOR)
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
    C_0 = 0.53 + 0.87 * zeta_squared + 0.50 * zeta_squared * zeta_squared + 2.26 * zeta_squared * zeta_squared * zeta_squared
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









def calculate_R3P_correlation(density, sigma, tau, calculation):
   
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
   
    method = calculation.method

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V
    df_dn_LDA, _, _, e_C_LDA = calculate_RVWN3_correlation(density, None, None, calculation) if "G" in method else calculate_RVWN5_correlation(density, None, None, calculation)

    # Picks the GGA correlation depending on the method
    if "LYP" in method: correlation_functional = calculate_RLYP_correlation
    if "PW" in method: correlation_functional = calculate_RPW_correlation
    if "P86" in method: correlation_functional = calculate_RP86_correlation

    # Calculates the energy density and derivatives for the GGA part
    df_dn_GGA, df_ds_GGA, _, e_C_GGA = correlation_functional(density, sigma, None, calculation)

    # These parameters are the standard B3LYP coefficients for correlation
    df_dn = 0.81 * df_dn_GGA + 0.19 * df_dn_LDA
    df_ds = 0.81 * df_ds_GGA
    e_C = 0.81 * e_C_GGA + 0.19 * e_C_LDA

    return df_dn, df_ds, None, e_C









def calculate_U3P_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):
    
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

    method = calculation.method

    # If "/G" is used, uses the Gaussian parameterisation for B3LYP with VWN-III instead of the more commonly used VWN-V
    df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, _, _, e_C_LDA = calculate_UVWN3_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, calculation) if "G" in method else calculate_UVWN5_correlation(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, None, None, calculation)
    
    # Picks the GGA correlation depending on the method
    if "LYP" in method: correlation_functional = calculate_ULYP_correlation
    if "PW" in method: correlation_functional = calculate_UPW_correlation
    if "P86" in method: correlation_functional = calculate_UP86_correlation

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
    "B": calculate_B88_exchange,
    "B3": calculate_B3_exchange,
    "TPSS": calculate_TPSS_exchange,
    "PW": calculate_PW91_exchange,
    "MPW": calculate_mPW91_exchange,

}






correlation_functionals = {

    "VWN3": calculate_RVWN3_correlation,
    "UVWN3": calculate_UVWN3_correlation,
    "VWN5": calculate_RVWN5_correlation,
    "UVWN5": calculate_UVWN5_correlation,
    "PW": calculate_RPWLDA_correlation,
    "UPW": calculate_UPWLDA_correlation,
    "PW91": calculate_RPW_correlation,
    "UPW91": calculate_UPW_correlation,
    "P86": calculate_RP86_correlation,
    "UP86": calculate_UP86_correlation,
    "PBE": calculate_PBE_correlation,
    "UPBE": calculate_UPBE_correlation,
    "LYP": calculate_RLYP_correlation,
    "ULYP": calculate_ULYP_correlation,    
    "3P": calculate_R3P_correlation,
    "U3P": calculate_U3P_correlation,
    "TPSS": calculate_RTPSS_correlation,
    "UTPSS": calculate_UTPSS_correlation,

}

