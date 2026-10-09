from TUNA.tuna_util import log, warning, symmetrise, constants, timer, log_spacer, check
from TUNA.tuna_calc import Calculation
import TUNA.tuna_xc as xc
import TUNA.tuna_kernels as kernels
from TUNA.tuna_molecule import Molecule
import numpy as np
from numpy import ndarray
from scipy.integrate import lebedev_rule
from scipy.spatial.distance import cdist


"""

This is the TUNA module for density functional theory (DFT), written first for version 0.9.0.

The DFT integration grids are constructed by Legendre-Gauss quadrature for the radial parts and Lebedev quadrature for the angular parts, they
have been optimised to some extent "by hand" but are not expected to be anywhere near as efficient as proper quantum chemistry codes. Exchange and
correlation matrices can be calculated for LDA, GGA and meta-GGA functionals, and a variety are implemented. The routines for hybrid and double-hybrid
functionals are found in the SCF module, not here.

Updated in version 0.10.1 to allow expressing molecular orbitals on grid, and calculating the differential overlap integrals.
Updated in version 0.11.0 to rotate Cartesian basis functions expressed on a grid onto spherical harmonics, add VV10 energy and exchange-correlation kernel matrix.
Updated in version 0.11.1 to express molecular orbital gradients on the grid.
Updated in version 0.12.0 to handle (meta-)GGA exchange-correlation kernels.

The module contains:

1. Some small utility functions (clean_density_matrix, integrate_on_grid, etc.)
2. Functions to set up the integration grid (set_up_integration_grid, build_molecular_grid, etc.)
3. Functions to evaluate the density, gradient, and kinetic energy density on the grid (construct_density_on_grid, etc.)
4. Functions to evaluate the exchange and correlation matrices (calculate_V_X, calculate_V_C, etc.)
5. Functions to evaluate the exchange-correlation kernel matrices for TD-DFT (construct_transition_variables_on_grid, etc.)

"""



def clean_density_matrix(P: ndarray, S: ndarray, n_electrons: int) -> ndarray:

    # Forces the trace of the density matrix to be correct

    scale = n_electrons / np.trace(P @ S) if n_electrons > 0 else 0

    return P * scale










def integrate_on_grid(integrand: ndarray, weights: ndarray) -> float:

    # Uses quadrature to calculate the integral of a function expressed on a grid

    integral = np.sum(integrand * weights)

    return integral










def integrate_final_density(alpha_density: ndarray, beta_density: ndarray, density: ndarray, weights: ndarray, calculation: Calculation, silent: bool = False) -> None:

    # Calculates the final integrals of the alpha, beta and total density

    n_alpha_DFT = integrate_on_grid(alpha_density, weights)
    n_beta_DFT = integrate_on_grid(beta_density, weights)

    n_electrons_DFT = integrate_on_grid(density, weights)

    log(f"\n Integral of the alpha density:       {n_alpha_DFT:13.10f}", calculation, 1, silent = silent)
    log(f" Integral of the beta density:        {n_beta_DFT:13.10f}\n", calculation, 1, silent = silent)

    log(f" Integral of the total density:       {n_electrons_DFT:13.10f}", calculation, 1, silent = silent)

    return










def set_up_integration_grid(molecule: Molecule, P_guess_alpha: ndarray, P_guess_beta: ndarray, calculation: Calculation, silent: bool) -> tuple:

    """

    High level function that sets up the integration grid and prints out information about it.

    Args:
        molecule (Molecule): Molecule object
        P_guess_alpha (array): Alpha density matrix in AO basis for guess
        P_guess_beta (array): Beta density matrix in AO basis for guess
        calculation (Calculation): Calculation object
        silent (bool): Should anything be printed

    Returns:
        bfs_on_grid: Basis functions evaluated on grid, shape (n_basis, n_radial, n_angular)
        weights: Integration weights for grid points, shape (n_radial, n_angular)
        bf_gradients_on_grid: Basis function gradients evaluated on grid (3, n_basis, n_radial, n_angular)
        points: Integration grid points, shape (n_radial, n_angular)

    """

    timer("Integration grid setup", 0)

    log(f" Setting up DFT integration grid with \"{calculation.grid_conv["name"]}\" accuracy...  ", calculation, 1, end = "", silent = silent)

    # Reads the integration grid parameters from the requested convergence criteria

    extent_multiplier = calculation.grid_conv["extent_multiplier"]
    integral_accuracy = calculation.grid_conv["integral_accuracy"] if not calculation.integral_accuracy_requested else calculation.integral_accuracy

    # The extent of the grid away from the nucleus is given by this function, which is homemade. It can be directly modified via the extent_multiplier.

    extent = extent_multiplier * np.max([molecule.atoms[i].real_vdw_radius for i in range(molecule.n_atoms)]) / 6

    # Uses the integral accuracy to map to a particular Lebedev order

    n = int(integral_accuracy * 9)

    LEBEDEV_ORDERS = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])

    idx = np.abs(LEBEDEV_ORDERS - n).argmin()
    Lebedev_order = int(LEBEDEV_ORDERS[idx])

    # Uses the integral accuracy to map to a number of radial points, scaled by extent of the grid to keep the density the same for larger atoms

    n_radial = int(extent * integral_accuracy)

    # Builds the molecular grid - the atomic grids are the same for both atoms even if heteroatomic

    points, weights = build_molecular_grid(extent, n_radial, Lebedev_order, molecule.bond_length, molecule.atoms)

    log("[Done]", calculation, 1, silent = silent)

    # Calculates the total number of grid points, and the number per atom

    total_points = points.shape[1] * points.shape[2]
    points_per_atom = total_points // molecule.n_atoms

    log(f"\n Integration grid has {n_radial} radial and {points.shape[2]} angular points, a Lebedev order of {Lebedev_order}.", calculation, 1, silent = silent)
    log(f" In total there are {total_points} grid points, {points_per_atom} per atom.", calculation, 1, silent = silent)

    log("\n Building guess density on grid...  ", calculation, 1, end = "", silent = silent)

    # Calculates the basis functions expressed on the integration grid

    bfs_on_grid = construct_basis_functions_on_grid(molecule.cartesian_basis_functions, points, molecule.spherical_harmonic_transformation_matrix)

    # If a (meta-)GGA calculation has been requested, determines the basis functions expressed on the integration grid

    bf_gradients_on_grid = construct_basis_function_gradients_on_grid(molecule.cartesian_basis_functions, points, molecule.spherical_harmonic_transformation_matrix) if calculation.functional.functional_class in ["GGA", "meta-GGA"] or calculation.VV10 else None

    # Constructs the electron density, using the guess density matrix, on the grid

    alpha_density = construct_density_on_grid(P_guess_alpha, bfs_on_grid)
    beta_density = construct_density_on_grid(P_guess_beta, bfs_on_grid)

    density = alpha_density + beta_density

    log("[Done]", calculation, 1, silent = silent)

    # Integrates the grid to get the number of electrons

    n_alpha_DFT = integrate_on_grid(alpha_density, weights)
    n_beta_DFT = integrate_on_grid(beta_density, weights)

    n_electrons_DFT = integrate_on_grid(density, weights)

    log(f"\n Integral of the guess alpha density: {n_alpha_DFT:14.10f}", calculation, 1, silent = silent)
    log(f" Integral of the guess beta density:  {n_beta_DFT:14.10f}\n", calculation, 1, silent = silent)

    log(f" Integral of the guess total density: {n_electrons_DFT:14.10f}\n", calculation, 1, silent = silent)

    # Prints a warning of the density integral is a bit dodgy, and throws an error if its totally wrong

    if np.abs(n_electrons_DFT - molecule.n_electrons) > 0.0001:

        warning(" Integral of density is far from the number of electrons! Be careful with your results.")

        check(np.abs(n_electrons_DFT - molecule.n_electrons) < 0.5, "Integral for the density is completely wrong!")

    log(f" Using {100 * calculation.DFX_prop:.1f}% density functional exchange and {100 * calculation.HFX_prop:.1f}% Hartree-Fock exchange.", calculation, 2, silent = silent)
    log(f" Using {100 * calculation.DFC_prop:.1f}% density functional correlation and {100 * calculation.MPC_prop:.1f}% Moller-Plesset correlation.\n", calculation, 2, silent = silent)

    timer("Integration grid setup", 1)

    return bfs_on_grid, weights, bf_gradients_on_grid, points










def build_atomic_radial_and_angular_grid(radial_grid_cutoff: float, n_radial: int, lebedev_order: int, radial_power: int = 3) -> tuple:

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

    atomic_points = np.einsum("m,in->imn", r, unit_sphere_directions, optimize = True)

    # Radial weights scaled by R^2 to account for surface of sphere getting larger away from nucleus

    atomic_weights = np.einsum("m,m,n->mn", weights_radial, r ** 2, weights_angular, optimize = True)

    return atomic_points, atomic_weights










def calculate_Becke_diatomic_weights(X: ndarray, Y: ndarray, Z: ndarray, bond_length: float, atoms: list, steepness: int = 4) -> tuple:

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










def build_molecular_grid(radial_grid_cutoff: float, n_radial: int, lebedev_order: int, bond_length: float, atoms: list) -> tuple:

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

    X = np.concatenate([X_A, X_B], axis = 0)
    Y = np.concatenate([Y_A, Y_B], axis = 0)
    Z = np.concatenate([Z_A, Z_B], axis = 0)

    points = np.stack((X, Y, Z), axis = 0)

    # Uses Becke atomic partitioning to get weights for each atom

    diatomic_weights_A, diatomic_weights_B = calculate_Becke_diatomic_weights(X, Y, Z, bond_length, atoms)

    n_points_atom_A = X_A.shape[0]

    # Combines the atomic weights with the molecular weights - both are between 0 and 1, so the combined weights are too

    weights_combined_A = atomic_weights_A * diatomic_weights_A[:n_points_atom_A]
    weights_combined_B = atomic_weights_B * diatomic_weights_B[n_points_atom_A:]

    # Forms total molecular weights array

    weights = np.concatenate([weights_combined_A, weights_combined_B], axis = 0)

    return points, weights










def construct_molecular_orbitals_on_grid(bfs_on_grid: ndarray, molecular_orbitals: ndarray) -> ndarray:

    """

    Expresses the molecular orbitals on the integration grid.

    Args:
        bfs_on_grid (array): Basis functions evaluated on integration grid
        molecular_orbitals (array): Molecular orbitals

    Returns:
        mos_on_grid (array): Molecular orbitals evaluated on integration grid, shape (orbital, radial, angular)

    """

    mos_on_grid = np.einsum("mp,mra->pra", molecular_orbitals, bfs_on_grid, optimize = True)

    return mos_on_grid










def construct_molecular_orbital_gradients_on_grid(bf_gradients_on_grid: ndarray, molecular_orbitals: ndarray) -> ndarray:

    """

    Expresses the molecular orbital gradients on the integration grid.

    Args:
        bf_gradients_on_grid (array): Basis function gradients evaluated on integration grid
        molecular_orbitals (array): Molecular orbitals

    Returns:
        mo_gradients_on_grid (array): Molecular orbital gradients evaluated on integration grid, shape (cartesian, orbital, radial, angular)

    """

    mo_gradients_on_grid = np.einsum("mp,cmra->cpra", molecular_orbitals, bf_gradients_on_grid, optimize = True)

    return mo_gradients_on_grid










def calculate_differential_overlap_integrals(molecular_orbitals: ndarray, molecule: Molecule, bfs_on_grid: ndarray, weights: ndarray, calculation: Calculation) -> ndarray:

    """

    Calculates the differential overlap integrals, (ia|ia), between the occupied and virtual real orbitals.

    Args:
        molecular_orbitals (array): Molecular orbitals
        molecule (Molecule): Molecule object
        bfs_on_grid (array): Basis functions evaluated on integration grid
        weights (array) Weights for integration
        calculation (Calculation): Calculation object

    Returns:
        differential_overlap_integrals (array): Overlap of one-electron densities of occupied and molecular orbitals

    """

    n_occ = molecule.n_occ if calculation.reference == "UHF" else molecule.n_doubly_occ
    n_virt = molecular_orbitals.shape[1] - n_occ

    # The DOI integrals are only between occupied and virtual orbitals

    differential_overlap_integrals = np.zeros((n_occ, n_virt))

    # Builds the molecular orbitals on the integration grid

    molecular_orbitals_on_grid = construct_molecular_orbitals_on_grid(bfs_on_grid, molecular_orbitals)

    for i in range(n_occ):

        for a in range(n_virt):

            # Each term is squared to get the one-electron density, get rid of the phase

            occupied_density = np.abs(molecular_orbitals_on_grid[i, :, :]) * np.abs(molecular_orbitals_on_grid[i, :, :])

            virtual_density = np.abs(molecular_orbitals_on_grid[n_occ + a, :, :]) * np.abs(molecular_orbitals_on_grid[n_occ + a, :, :])

            # The integral is square rooted to restore the norm

            differential_overlap_integrals[i, a] = integrate_on_grid(occupied_density * virtual_density, weights) ** (1 / 2)


    return differential_overlap_integrals










def construct_basis_functions_on_grid(basis_functions: list, points: ndarray, spherical_harmonic_transformation_matrix: ndarray) -> ndarray:

    """

    Expresses the basis functions on the integration grid.

    Args:
        basis_functions (list): List of basis function objects
        points (array): Integration grid points
        spherical_harmonic_transformation_matrix (array): Spherical harmonic transformation matrix

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

        exponent_term = np.exp(-1 * np.einsum("i,jk->ijk", bf.exps, r_squared, optimize = True))

        # Basis functions are a product of the coefficient, norm, angular part and radial exponent part

        contracted = np.einsum("i,i,ijk->jk", bf.coefs, bf.norm, exponent_term)

        bfs_on_grid[i] = contracted * X_relative ** l * Y_relative ** m * Z_relative ** n

    # Transforms from Cartesian harmonics to spherical harmonics

    bfs_on_grid = np.einsum("pq,qjk->pjk", spherical_harmonic_transformation_matrix, bfs_on_grid, optimize = True)

    return bfs_on_grid










def construct_basis_function_gradients_on_grid(basis_functions: list, points: ndarray, spherical_harmonic_transformation_matrix: ndarray) -> ndarray:

    """

    Expresses the analytic gradients of the basis functions on the integration grid.

    Args:
        basis_functions (list): List of basis function objects
        points (array): Integration grid points
        spherical_harmonic_transformation_matrix (array): Spherical harmonic transformation matrix

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

        exponent_term = np.exp(-1 * np.einsum("i,jk->ijk", bf.exps, r_squared, optimize = True))

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

        bf_gradients_on_grid[i] = np.einsum("i,i,aijk->ajk", bf.coefs, bf.norm, primitives, optimize = True)

    # Transforms from Cartesian harmonics to spherical harmonics

    bf_gradients_on_grid = np.einsum("pq,qajk->apjk", spherical_harmonic_transformation_matrix, bf_gradients_on_grid, optimize = True)

    return bf_gradients_on_grid










def construct_density_on_grid(P: ndarray, bfs_on_grid: ndarray, clean_density: bool = True) -> ndarray:

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

    density = np.einsum("ij,ikl,jkl->kl", P, bfs_on_grid, bfs_on_grid, optimize = True)

    # This is on by default to get rid of very small, zero and negative values that break functionals

    if clean_density:

        density = xc.clean(density)

    return density










def calculate_density_gradient(P: ndarray, bfs_on_grid: ndarray, bf_gradients_on_grid: ndarray) -> tuple:

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

    density_gradient = 2 * np.einsum("ij,ikl,ajkl->akl", P, bfs_on_grid, bf_gradients_on_grid, optimize = True)

    # Builds the square density gradient, used in GGA functionals

    sigma = np.einsum("akl->kl", density_gradient * density_gradient, optimize = True)

    # It is important that this is cleaned at the *square* of the floor that the density and kinetic energy density are cleaned

    sigma = xc.clean(sigma, floor = constants.sigma_floor)

    return sigma, density_gradient










def calculate_kinetic_energy_density(P: ndarray, bf_gradients_on_grid: ndarray) -> ndarray:

    """

    Calculates the density gradient, and its square, on the integration grid.

    Args:
        P (array): Density matrix in AO basis
        bf_gradients_on_grid (array): Basis function gradients evaluated on grid

    Returns:
        tau (array): Non-interacting kinetic energy density, shape (n_radial, n_angular)

    """

    # Conventional expression, with factor of a half, for non-interacting kinetic energy density used in meta-GGA functionals

    tau = (1 / 2) * np.einsum("ij,aikl,ajkl->kl", P, bf_gradients_on_grid, bf_gradients_on_grid, optimize = True)

    # This needs to be cleaned with the same floor as the density, higher than sigma

    tau = xc.clean(tau, floor = constants.density_floor)

    return tau










def calculate_V_X(weights: ndarray, bfs_on_grid: ndarray, df_dn: ndarray, df_ds: ndarray, df_dt: ndarray, bf_gradients_on_grid: ndarray, density_gradient: ndarray) -> ndarray:

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

    V_X = np.einsum("kl,mkl,nkl,kl->mn", df_dn, bfs_on_grid, bfs_on_grid, weights, optimize = True)

    if df_ds is not None:

        # Contribution to exchange matrix from GGA part of functional

        V_X += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize = True)

    if df_dt is not None:

        # Contribution to exchange matrix from meta-GGA part of functional

        V_X += (1 / 2) * np.einsum("kl,amkl,ankl,kl->mn", df_dt, bf_gradients_on_grid, bf_gradients_on_grid, weights, optimize = True)

    # Absolutely necessary to symmetrise as these are not symmetric by design

    V_X = symmetrise(V_X)

    return V_X










def calculate_V_C(weights: ndarray, bfs_on_grid: ndarray, df_dn: ndarray, df_ds: ndarray, df_dt: ndarray, bf_gradients_on_grid: ndarray, density_gradient: ndarray, density_gradient_other_spin: ndarray = None, df_ds_ab: ndarray = None) -> ndarray:

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

    V_C = np.einsum("kl,mkl,nkl,kl->mn", df_dn, bfs_on_grid, bfs_on_grid, weights, optimize = True)

    if df_ds is not None:

        # Contribution to exchange matrix from GGA part of functional

        if df_ds_ab is not None:

            V_C += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize = True)
            V_C += 2 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds_ab, density_gradient_other_spin, bfs_on_grid, bf_gradients_on_grid, weights, optimize = True)

        else:

            V_C += 4 * np.einsum("kl,akl,mkl,ankl,kl->mn", df_ds, density_gradient, bfs_on_grid, bf_gradients_on_grid, weights, optimize = True)

    if df_dt is not None:

        # Contribution to exchange matrix from meta-GGA part of functional

        V_C += (1 / 2) * np.einsum("kl,amkl,ankl,kl->mn", df_dt, bf_gradients_on_grid, bf_gradients_on_grid, weights, optimize = True)

    # Absolutely necessary to symmetrise as these are not symmetric by design

    V_C = symmetrise(V_C)

    return V_C












def calculate_VV10_inner_integral(points: ndarray, omega: ndarray, kappa: ndarray, density: ndarray, chunk: int = 192) -> ndarray:

    """

    Calculates the inner integral by looping over blocks for the VV10 energy.

    Args:
        points (array): Integration grid points
        omega (array): Omega values
        kappa (array): Kappa values
        density (array): Electron density on grid
        chunk (int, optional): Chunk size

    Returns:
        inner_integral (array): Inner VV10 integral

    """

    inner_integral = np.zeros(points.shape[0])

    # The number of blocks over which we loop

    n_blocks = (points.shape[0] + chunk - 1) // chunk

    GJ = np.empty((chunk, chunk))
    SM = np.empty((chunk, chunk))

    # Unfortunately this can't be done as a NumPy operation as the arrays are four-dimensional, which is too big for grids

    for block in range(n_blocks):

        i0, i1 = block * chunk, min((block + 1) * chunk, points.shape[0])

        ci = i1 - i0

        for jc in range(block, n_blocks):

            j0, j1 = jc * chunk, min((jc + 1) * chunk, points.shape[0])

            cj = j1 - j0

            # Determines the distance between points

            d2 = cdist(points[i0:i1], points[j0:j1], "sqeuclidean")

            gj, sm = GJ[:ci, :cj], SM[:ci, :cj]

            np.multiply(d2, omega[j0:j1], out = gj)

            gj += kappa[j0:j1]

            d2 *= omega[i0:i1, None]
            d2 += kappa[i0:i1, None]

            # Kernel calculations

            np.add(d2, gj, out = sm)

            d2 *= gj
            d2 *= sm

            np.divide(-1.5, d2, out = d2)

            inner_integral[i0:i1] += d2 @ density[j0:j1]

            # Complete inner integration

            if block != jc:

                inner_integral[j0:j1] += density[i0:i1] @ d2

    return inner_integral










def calculate_VV10_energy(P: ndarray, grid_container: tuple, calculation: Calculation, silent: bool) -> float:

    """

    Calculates the non-local VV10 dispersion energy.

    Args:
        P (array): Density matrix in AO basis
        grid_container (tuple): Integration grid information
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        E_VV10 (float): Non-local dispersion energy

    """

    bfs, weights, bf_grads, points = grid_container

    # The VV10 "C" parameter is fixed, and the "b" parameter is dependent on the functional

    b = calculation.functional.VV10_b if calculation.functional is not None else 3.9

    C = calculation.functional.VV10_C if calculation.functional is not None else 0.0093

    timer("Non-local VV10 dispersion", 0)

    log_spacer(calculation, 1, silent)
    log("             Non-local Dispersion Energy", calculation, 1, silent)
    log_spacer(calculation, 1, silent)

    log(f"  Using a \"b\" value of {b} and \"c\" value of {C}.", calculation, 3, silent, end="\n\n")

    log("  Calculating VV10 dispersion energy...      ", calculation, 1, silent, end = "")

    density_full = construct_density_on_grid(P, bfs).ravel()
    sigma_full, _ = calculate_density_gradient(P, bfs, bf_grads)

    # We mask out tiny densities for speed and stability

    mask = density_full > 1e-10
    density = density_full[mask]
    weights = weights.ravel()[mask]
    sigma = sigma_full.ravel()[mask]
    points = np.ascontiguousarray(points.reshape(3, -1).T[mask])

    # We need these density-dependent quantities multiple times

    density_squared  = density * density

    weighted_density = density * weights

    # Some density-dependent parameters for VV10

    sigma_over_square_density = sigma / density_squared

    omega = (C * sigma_over_square_density * sigma_over_square_density + (4 / 3) * np.pi * density) ** (1 / 2)

    kappa = (3 / 2) * np.pi * b * (density / (9 * np.pi)) ** (1 / 6)

    # A scalar parameter for VV10

    beta = (1 / 32) * (3 / b ** 2) ** (3 / 4)

    # Calculates the VV10 inner integral

    inner_integral = calculate_VV10_inner_integral(points, omega, kappa, weighted_density)

    # Final integration is done with a matrix multiplication for BLAS3 speedups

    E_VV10 = weighted_density @ (beta + (1 / 2) * inner_integral) * calculation.functional.VV10_scaling

    log("[Done]", calculation, 1, silent)

    log(f"\n  Energy from VV10:                {E_VV10:16.10f}", calculation, 1, silent)

    log_spacer(calculation, 1, silent, end = "\n")

    timer("Non-local VV10 dispersion", 1)

    return E_VV10










def construct_orbital_pair_products(first_orbitals: ndarray, second_orbitals: ndarray) -> ndarray:

    """

    Builds orbital pair products on the grid, the change in the density from rotating each orbital of the first set into each of the second.

    Args:
        first_orbitals (array): Molecular orbitals on the grid
        second_orbitals (array): Molecular orbitals on the grid

    Returns:
        pair_products (array): Products of pairs of orbitals on the grid

    """

    pair_products = np.einsum("pmn,qmn->pqmn", first_orbitals, second_orbitals, optimize = True)

    return pair_products










def construct_orbital_pair_gradients(first_orbitals: ndarray, first_orbital_gradients: ndarray, second_orbitals: ndarray, second_orbital_gradients: ndarray, cartesian: int) -> ndarray:

    """

    Builds one Cartesian component of the gradients of orbital pair products on the grid.

    Args:
        first_orbitals (array): Molecular orbitals on the grid
        first_orbital_gradients (array): Molecular orbital gradients on the grid
        second_orbitals (array): Molecular orbitals on the grid
        second_orbital_gradients (array): Molecular orbital gradients on the grid
        cartesian (int): Either 0, 1, or 2

    Returns:
        pair_gradients (array): Gradients of pairs of orbitals on the grid, shape (orbital 1, orbital 2, radial, angular)

    """

    # Product rule, so each component picks up a term from each orbital of the pair

    pair_gradients = np.einsum("pmn,qmn->pqmn", first_orbital_gradients[cartesian], second_orbitals, optimize = True)
    pair_gradients += np.einsum("pmn,qmn->pqmn", first_orbitals, second_orbital_gradients[cartesian], optimize = True)

    return pair_gradients










def construct_transition_variables_on_grid(first_orbitals: ndarray, first_orbital_gradients: ndarray, second_orbitals: ndarray, second_orbital_gradients: ndarray, density_gradients: list, functional_class: str) -> tuple:

    """

    Builds the changes in the variables of the exchange-correlation functional caused by rotating each orbital of the first set into each of the
    second, which the kernel is contracted with. The density changes by the pair product, each square gradient by the dot product of a density
    gradient with the gradient of the pair product, and the kinetic energy density by half the dot product of the orbital gradients.

    Args:
        first_orbitals (array): Molecular orbitals on the grid
        first_orbital_gradients (array): Molecular orbital gradients on the grid
        second_orbitals (array): Molecular orbitals on the grid
        second_orbital_gradients (array): Molecular orbital gradients on the grid
        density_gradients (list): Density gradients the square gradient variables are built from, each scaled by the factor its square gradient changes by
        functional_class (str): Either "LDA", "GGA" or "meta-GGA"

    Returns:
        U (array): Changes in the density, square gradients and kinetic energy density, shape (variable, orbital 1, orbital 2, radial, angular)
        T_gradient (array): Gradient of the pair products, shape (cartesian, orbital 1, orbital 2, radial, angular), or None for an LDA

    """

    U = [construct_orbital_pair_products(first_orbitals, second_orbitals)]

    T_gradient = None

    if functional_class in ["GGA", "meta-GGA"]:

        T_gradient = np.array([construct_orbital_pair_gradients(first_orbitals, first_orbital_gradients, second_orbitals, second_orbital_gradients, cartesian) for cartesian in range(3)])

        for density_gradient in density_gradients:

            U.append(np.einsum("cmn,cpqmn->pqmn", density_gradient, T_gradient, optimize = True))

    # The kinetic energy density changes by half the dot product of the orbital gradients, as it carries a factor of a half

    if functional_class == "meta-GGA":

        U.append((1 / 2) * np.einsum("cpmn,cqmn->pqmn", first_orbital_gradients, second_orbital_gradients, optimize = True))

    U = np.array(U)

    return U, T_gradient










def contract_kernel_with_transition_variables(f_XC: ndarray, df_ds_XC: ndarray, U_1: ndarray, T_gradient_1: ndarray, U_2: ndarray, T_gradient_2: ndarray) -> ndarray:

    """

    Calculates the matrix elements of the exchange-correlation kernel between two sets of orbital rotations.

    The second derivatives of the functional are contracted with the linear changes in its variables. As the square gradient is quadratic in the
    density, the first derivative with respect to it also enters, contracted with the gradients of the two pair products.

    Args:
        f_XC (array): Second derivatives of the functional with respect to its variables, with integration weights folded in
        df_ds_XC (array): First derivative of the functional with respect to the square gradient, with integration weights folded in
        U_1 (array): Changes in the variables from the first set of orbital rotations
        T_gradient_1 (array): Gradients of the pair products of the first set of orbital rotations
        U_2 (array): Changes in the variables from the second set of orbital rotations
        T_gradient_2 (array): Gradients of the pair products of the second set of orbital rotations

    Returns:
        K_XC (array): Exchange-correlation kernel matrix, shape (orbital 1, orbital 2, orbital 1, orbital 2)

    """

    # This also accounts for the kinetic energy density and square gradients terms for (meta-)GGA functionals

    K_XC = np.einsum("upqmn,uvmn,vrsmn->pqrs", U_1, f_XC, U_2, optimize = True)

    if T_gradient_1 is not None:

        K_XC += np.einsum("cpqmn,crsmn,mn->pqrs", T_gradient_1, T_gradient_2, df_ds_XC, optimize = True)

    return K_XC










def contract_kernel_with_orbital_pairs(f_XC: ndarray, df_ds_XC: ndarray, U: ndarray, T_gradient: ndarray, orbitals: ndarray, orbital_gradients: ndarray, density_gradients: list, functional_class: str) -> ndarray:

    """

    Calculates the matrix elements of the exchange-correlation kernel between a set of orbital rotations and every pair of orbitals, for
    coupled-perturbed Kohn-Sham equations. The changes in the variables from the orbital pairs are built one at a time to limit memory use.

    Args:
        f_XC (array): Second derivatives of the functional with respect to its variables, with integration weights folded in
        df_ds_XC (array): First derivative of the functional with respect to the square gradient, with integration weights folded in
        U (array): Changes in the variables from the orbital rotations
        T_gradient (array): Gradients of the pair products of the orbital rotations
        orbitals (array): Molecular orbitals on the grid
        orbital_gradients (array): Molecular orbital gradients on the grid
        density_gradients (list): Density gradients the square gradient variables are built from, each scaled by the factor its square gradient changes by
        functional_class (str): Either "LDA", "GGA" or "meta-GGA"

    Returns:
        K_XC_full (array): Exchange-correlation kernel matrix, shape (orbital 1, orbital 2, orbital, orbital)

    """

    # Contracts the kernel with the orbital rotations first, leaving one array per variable to integrate against the orbital pairs

    A = np.einsum("upqmn,uvmn->vpqmn", U, f_XC, optimize = True)

    K_XC_full = np.einsum("pqmn,rsmn->pqrs", A[0], construct_orbital_pair_products(orbitals, orbitals), optimize = True)

    if functional_class in ["GGA", "meta-GGA"]:

        for variable, density_gradient in enumerate(density_gradients, start = 1):

            # The change in a square gradient from a pair has the same product rule structure as the pair gradient, with grad(n) . grad(phi) in place of grad(phi)

            orbital_density_gradients = np.einsum("cmn,cpmn->pmn", density_gradient, orbital_gradients, optimize = True)[None]

            K_XC_full += np.einsum("pqmn,rsmn->pqrs", A[variable], construct_orbital_pair_gradients(orbitals, orbital_density_gradients, orbitals, orbital_density_gradients, 0), optimize = True)

        for cartesian in range(3):

            K_XC_full += np.einsum("pqmn,rsmn->pqrs", T_gradient[cartesian] * df_ds_XC, construct_orbital_pair_gradients(orbitals, orbital_gradients, orbitals, orbital_gradients, cartesian), optimize = True)

    if functional_class == "meta-GGA":

        K_XC_full += np.einsum("pqmn,rsmn->pqrs", A[-1], (1 / 2) * np.einsum("cpmn,cqmn->pqmn", orbital_gradients, orbital_gradients, optimize = True), optimize = True)

    return K_XC_full










def calculate_restricted_exchange_correlation_kernel_matrices(o: slice, v: slice, P: ndarray, bfs_on_grid: ndarray, bf_gradients_on_grid: ndarray, molecular_orbitals: ndarray, calculation: Calculation, weights: ndarray, silent: bool, return_full_kernel: bool = False) -> tuple:

    """

    Calculates the matrix elements of the exchange-correlation kernel for a restricted reference.

    For singlet excitations both spin densities change together, so the kernel is the second derivative of the functional with respect to the
    total density, square gradient and kinetic energy density, contracted with the changes rho_ia, 2 grad(n)grad(rho_ia) and
    grad(phi_i)grad(phi_a). For triplets the spin densities change oppositely, so the second derivatives with respect to the spin density,
    grad(n)grad(m) and the spin kinetic energy density are used instead, contracted with rho_ia, grad(n)grad(rho_ia) and grad(phi_i)grad(phi_a).

    Args:
        o (slice): Occupied slice
        v (slice): Virtual slice
        P (array): Density matrix in AO basis
        bfs_on_grid (array): Basis functions on grid
        bf_gradients_on_grid (array): Basis function gradients on grid, or None for an LDA
        molecular_orbitals (array): Molecular orbitals
        calculation (Calculation): Calculation object
        weights (array): Integration weights
        silent (bool): Cancel logging
        return_full_kernel (bool, optional): Should the unsliced kernel also be returned, for coupled-perturbed Kohn-Sham equations

    Returns:
        K_XC_singlet (array): Singlet exchange-correlation kernel matrix
        K_XC_triplet (array): Triplet exchange-correlation kernel matrix
        K_XC_full (array): Unsliced singlet exchange-correlation kernel matrix, None unless return_full_kernel is used

    """

    functional_class = calculation.functional.functional_class

    # Builds molecular orbitals, and their gradients, on the integration grid

    log("\n Evaluating molecular orbitals on grid...    ", calculation, 1, silent, end = "")

    molecular_orbitals_on_grid = construct_molecular_orbitals_on_grid(bfs_on_grid, molecular_orbitals)

    # Constructs the density, square gradient and kinetic energy density on the grid, as in the SCF, including any frozen-core electrons

    density = construct_density_on_grid(P, bfs_on_grid)

    sigma, tau, density_gradient, molecular_orbital_gradients_on_grid = None, None, None, None

    if functional_class in ["GGA", "meta-GGA"]:

        sigma, density_gradient = calculate_density_gradient(P, bfs_on_grid, bf_gradients_on_grid)

        molecular_orbital_gradients_on_grid = construct_molecular_orbital_gradients_on_grid(bf_gradients_on_grid, molecular_orbitals)

        if functional_class == "meta-GGA":

            tau = calculate_kinetic_energy_density(P, bf_gradients_on_grid)

    log("[Done]", calculation, 1, silent)

    log(" Evaluating exchange-correlation kernel...   ", calculation, 1, silent, end = "")

    # Looks up the kernels and functional - exchange-only functionals have no correlation kernel, so all its blocks are None

    exchange_kernel = kernels.exchange_kernels[calculation.functional.x_functional]
    correlation_kernel = kernels.correlation_density_kernels.get(calculation.functional.c_functional)
    correlation_spin_kernel = kernels.correlation_spin_kernels.get(calculation.functional.c_functional)

    exchange_functional = xc.exchange_functionals[calculation.functional.x_functional]
    correlation_functional = xc.correlation_functionals.get(calculation.functional.c_functional)

    # Second derivatives of the exchange and correlation energies for singlet excitations, with respect to the density, sigma and tau, and the
    # first derivatives with respect to sigma, which enter as sigma is quadratic in the density

    f_X_singlet = kernels.arrange_restricted_kernel_blocks(exchange_kernel(density, sigma, tau, calculation), density)
    f_C_singlet = kernels.arrange_restricted_kernel_blocks(correlation_kernel(density, sigma, tau, calculation) if correlation_kernel is not None else (None,) * 6, density)

    _, df_ds_X_singlet, _, _ = exchange_functional(density, sigma, tau, calculation)
    _, df_ds_C_singlet, _, _ = correlation_functional(density, sigma, tau, calculation) if correlation_functional is not None else (None,) * 4

    # Second derivatives for triplet excitations, with respect to the spin density, grad(n) . grad(m) and the spin kinetic energy density, and the
    # first derivatives with respect to grad(m)grad(m), which the spin kernels return last

    *f_X_triplet, df_ds_X_triplet = kernels.calculate_exchange_spin_kernel(density, sigma, tau, calculation)
    *f_C_triplet, df_ds_C_triplet = correlation_spin_kernel(density, sigma, tau, calculation) if correlation_spin_kernel is not None else (None,) * 7

    f_X_triplet = kernels.arrange_restricted_kernel_blocks(f_X_triplet, density)
    f_C_triplet = kernels.arrange_restricted_kernel_blocks(f_C_triplet, density)

    df_ds_X_singlet, df_ds_C_singlet, df_ds_X_triplet, df_ds_C_triplet = (np.zeros_like(density) if derivative is None else derivative for derivative in (df_ds_X_singlet, df_ds_C_singlet, df_ds_X_triplet, df_ds_C_triplet))

    # Folds in the DFX and DFC proportions, and the integration weights. The sum over the spin of the second transition density gives a factor
    # of two, and the quadratic change in sigma a factor of four

    n_variables = {"LDA": 1, "GGA": 2, "meta-GGA": 3}[functional_class]

    f_XC_singlet = 2 * (f_X_singlet * calculation.DFX_prop + f_C_singlet * calculation.DFC_prop)[:n_variables, :n_variables] * weights
    f_XC_triplet = 2 * (f_X_triplet * calculation.DFX_prop + f_C_triplet * calculation.DFC_prop)[:n_variables, :n_variables] * weights

    df_ds_XC_singlet = 4 * (df_ds_X_singlet * calculation.DFX_prop + df_ds_C_singlet * calculation.DFC_prop) * weights
    df_ds_XC_triplet = 4 * (df_ds_X_triplet * calculation.DFX_prop + df_ds_C_triplet * calculation.DFC_prop) * weights

    # Singlets see the change in sigma, 2 grad(n)grad(rho_ia), and triplets the change in grad(n)grad(m), grad(n)grad(rho_ia). The same
    # transition variables are used for both, with the factor of two folded into the singlet kernel blocks

    sigma_scaling = np.array([1, 2, 1])[:n_variables]

    f_XC_singlet = f_XC_singlet * sigma_scaling[:, None, None, None] * sigma_scaling[None, :, None, None]

    log("[Done]", calculation, 1, silent)

    log(" Calculating matrix elements...              ", calculation, 1, silent, end = "")

    # Slice out occupied and virtual orbitals, and their gradients, on the grid

    occupied_orbitals, virtual_orbitals = molecular_orbitals_on_grid[o], molecular_orbitals_on_grid[v]

    occupied_gradients, virtual_gradients = (molecular_orbital_gradients_on_grid[:, o], molecular_orbital_gradients_on_grid[:, v]) if functional_class != "LDA" else (None, None)

    # Changes in the variables from each occupied-virtual rotation, and the gradients of the transition densities

    U, T_gradient = construct_transition_variables_on_grid(occupied_orbitals, occupied_gradients, virtual_orbitals, virtual_gradients, [density_gradient], functional_class)

    # Contracts the kernels with the changes in the variables

    K_XC_singlet = contract_kernel_with_transition_variables(f_XC_singlet, df_ds_XC_singlet, U, T_gradient, U, T_gradient)
    K_XC_triplet = contract_kernel_with_transition_variables(f_XC_triplet, df_ds_XC_triplet, U, T_gradient, U, T_gradient)

    # Optionally builds the singlet kernel matrix over the full orbital space, for relaxed double-hybrid densities

    K_XC_full = None

    if return_full_kernel:

        K_XC_full = contract_kernel_with_orbital_pairs(f_XC_singlet, df_ds_XC_singlet, U, T_gradient, molecular_orbitals_on_grid, molecular_orbital_gradients_on_grid, [density_gradient], functional_class)

    log("[Done]", calculation, 1, silent)

    return K_XC_singlet, K_XC_triplet, K_XC_full










def calculate_unrestricted_exchange_correlation_kernel_matrices(o: slice, v: slice, P_alpha: ndarray, P_beta: ndarray, bfs_on_grid: ndarray, bf_gradients_on_grid: ndarray, C_spin_block: ndarray, spin_labels: list, calculation: Calculation, weights: ndarray, silent: bool, return_full_kernel: bool = False) -> ndarray:

    """

    Calculates the matrix elements of the spin-resolved exchange-correlation kernel for an unrestricted reference.

    Rotating an occupied spin orbital of spin sigma into a virtual one changes the density of that spin by rho_ia, the same-spin square gradient
    by 2 grad(n_sigma) . grad(rho_ia), the opposite-spin square gradient by grad(n_sigma') . grad(rho_ia) and the kinetic energy density of that
    spin by grad(phi_i) . grad(phi_a). Only spin-conserving rotations are considered, so the spin-flip blocks of the kernel matrix are zero.

    Args:
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        bfs_on_grid (array): Basis functions on grid
        bf_gradients_on_grid (array): Basis function gradients on grid, or None for an LDA
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis
        spin_labels (list): Spin ("a" or "b") of each spin orbital, in ascending energy order
        calculation (Calculation): Calculation object
        weights (array): Integration weights
        silent (bool): Cancel logging
        return_full_kernel (bool, optional): Should the unsliced kernel also be returned, for coupled-perturbed Kohn-Sham equations

    Returns:
        K_XC (array): Spin orbital exchange-correlation kernel matrix
        K_XC_full (array): Unsliced exchange-correlation kernel matrix, only if return_full_kernel is used

    """

    functional_class = calculation.functional.functional_class

    # Builds the spin orbitals, and their gradients, on the integration grid

    log("\n Evaluating molecular orbitals on grid...    ", calculation, 1, silent, end = "")

    basis_functions_spin_blocked = np.concatenate([bfs_on_grid, bfs_on_grid], axis = 0)

    molecular_orbitals_on_grid = construct_molecular_orbitals_on_grid(basis_functions_spin_blocked, C_spin_block)

    # Constructs the alpha and beta densities, square gradients and kinetic energy densities on the grid, as in the SCF, including any frozen-core electrons

    alpha_density = construct_density_on_grid(P_alpha, bfs_on_grid)
    beta_density = construct_density_on_grid(P_beta, bfs_on_grid)

    density = alpha_density + beta_density

    sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, molecular_orbital_gradients_on_grid = None, None, None, None, None, None
    density_gradients = {"a": None, "b": None}

    if functional_class in ["GGA", "meta-GGA"]:

        sigma_aa, density_gradients["a"] = calculate_density_gradient(P_alpha, bfs_on_grid, bf_gradients_on_grid)
        sigma_bb, density_gradients["b"] = calculate_density_gradient(P_beta, bfs_on_grid, bf_gradients_on_grid)

        # This sigma is made here as the others are cleaned in calculate_density_gradient - do NOT clean this

        sigma_ab = np.einsum("akl,akl->kl", density_gradients["a"], density_gradients["b"], optimize = True)

        molecular_orbital_gradients_on_grid = construct_molecular_orbital_gradients_on_grid(np.concatenate([bf_gradients_on_grid, bf_gradients_on_grid], axis = 1), C_spin_block)

        if functional_class == "meta-GGA":

            tau_alpha = calculate_kinetic_energy_density(P_alpha, bf_gradients_on_grid)
            tau_beta = calculate_kinetic_energy_density(P_beta, bf_gradients_on_grid)

    log("[Done]", calculation, 1, silent)

    log(" Evaluating exchange-correlation kernel...   ", calculation, 1, silent, end = "")

    # Looks up the kernel and functional - exchange-only functionals have no correlation kernel, so all its blocks are None

    correlation_kernel = kernels.unrestricted_correlation_kernels.get(calculation.functional.c_functional)

    exchange_functional = xc.exchange_functionals[calculation.functional.x_functional]
    correlation_functional = xc.correlation_functionals.get("U" + str(calculation.functional.c_functional))

    # Second derivatives of the exchange and correlation energies with respect to the alpha and beta densities, the three square gradients and the
    # two kinetic energy densities, in the order (n_a, n_b, sigma_aa, sigma_bb, sigma_ab, tau_a, tau_b)

    f_X = kernels.arrange_unrestricted_kernel_blocks(kernels.calculate_unrestricted_exchange_kernel(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation), density)
    f_C = kernels.arrange_unrestricted_kernel_blocks(correlation_kernel(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation) if correlation_kernel is not None else (None,) * 28, density)

    # The first derivatives with respect to the three square gradients, which enter as these are quadratic in the densities. Exchange spin scales,
    # so its derivative with respect to a same-spin square gradient is twice the restricted derivative at twice the spin density

    sigma_aa_scaled, sigma_bb_scaled = (4 * sigma_aa, 4 * sigma_bb) if sigma_aa is not None else (None, None)
    tau_alpha_scaled, tau_beta_scaled = (2 * tau_alpha, 2 * tau_beta) if tau_alpha is not None else (None, None)

    _, df_ds_X_aa, _, _ = exchange_functional(2 * alpha_density, sigma_aa_scaled, tau_alpha_scaled, calculation)
    _, df_ds_X_bb, _, _ = exchange_functional(2 * beta_density, sigma_bb_scaled, tau_beta_scaled, calculation)

    _, _, df_ds_C_aa, df_ds_C_bb, df_ds_C_ab, _, _, _ = correlation_functional(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation) if correlation_functional is not None else (None,) * 8

    df_ds_X_aa, df_ds_X_bb, df_ds_C_aa, df_ds_C_bb, df_ds_C_ab = (np.zeros_like(density) if derivative is None else derivative for derivative in (df_ds_X_aa, df_ds_X_bb, df_ds_C_aa, df_ds_C_bb, df_ds_C_ab))

    # Folds in the DFX and DFC proportions, and the integration weights. The same-spin square gradients are quadratic in one density, so their
    # quadratic change is 2 grad(rho_ia) . grad(rho_jb), while the opposite-spin square gradient changes by grad(rho_ia) . grad(rho_jb)

    f_XC = (f_X * calculation.DFX_prop + f_C * calculation.DFC_prop) * weights

    df_ds_XC = {

        "aa": 2 * (2 * df_ds_X_aa * calculation.DFX_prop + df_ds_C_aa * calculation.DFC_prop) * weights,
        "ab": df_ds_C_ab * calculation.DFC_prop * weights,
        "bb": 2 * (2 * df_ds_X_bb * calculation.DFX_prop + df_ds_C_bb * calculation.DFC_prop) * weights

    }

    log("[Done]", calculation, 1, silent)

    log(" Calculating matrix elements...              ", calculation, 1, silent, end = "")

    spin_labels = np.array(spin_labels)

    # Indices of the spin orbitals of each spin in the full, occupied and virtual spaces, and the variables each spin sees in the order (n, same-spin sigma, opposite-spin sigma, tau)

    spin_orbitals = {"a": np.where(spin_labels == "a")[0], "b": np.where(spin_labels == "b")[0]}

    occupied = {spin: np.where(spin_labels[o] == spin)[0] for spin in "ab"}
    virtual = {spin: np.where(spin_labels[v] == spin)[0] for spin in "ab"}

    variables = {"a": [0, 2, 4, 5], "b": [1, 3, 4, 6]}

    # The number of variables a spin sees, and the scaled density gradients the same- and opposite-spin square gradient changes are built from

    n_variables = {"LDA": 1, "GGA": 3, "meta-GGA": 4}[functional_class]

    spin_density_gradients = {spin: [2 * density_gradients[spin], density_gradients[other]] if functional_class != "LDA" else [] for spin, other in (("a", "b"), ("b", "a"))}

    # Changes in the variables from the spin-conserving rotations of each spin, and the gradients of the transition densities

    U, T_gradient = {}, {}

    for spin in "ab":

        occupied_orbitals = molecular_orbitals_on_grid[o][occupied[spin]]
        virtual_orbitals = molecular_orbitals_on_grid[v][virtual[spin]]

        occupied_gradients, virtual_gradients = (molecular_orbital_gradients_on_grid[:, o][:, occupied[spin]], molecular_orbital_gradients_on_grid[:, v][:, virtual[spin]]) if functional_class != "LDA" else (None, None)

        U[spin], T_gradient[spin] = construct_transition_variables_on_grid(occupied_orbitals, occupied_gradients, virtual_orbitals, virtual_gradients, spin_density_gradients[spin], functional_class)

    # Contracts each spin block of the kernel with the appropriate changes in the variables, where the two spins see different variables

    n_occ, n_virt = len(spin_labels[o]), len(spin_labels[v])

    K_XC = np.zeros((n_occ, n_virt, n_occ, n_virt))

    for spin_1 in "ab":

        for spin_2 in "ab":

            f_XC_block = f_XC[np.ix_(variables[spin_1][:n_variables], variables[spin_2][:n_variables])]

            K_XC[np.ix_(occupied[spin_1], virtual[spin_1], occupied[spin_2], virtual[spin_2])] = contract_kernel_with_transition_variables(f_XC_block, df_ds_XC["".join(sorted(spin_1 + spin_2))], U[spin_1], T_gradient[spin_1], U[spin_2], T_gradient[spin_2])

    # Optionally builds the kernel matrix over the full spin orbital space, for coupled-perturbed Kohn-Sham equations, one spin block at a time to limit memory use

    if return_full_kernel:

        n_spin_orbitals = len(spin_labels)

        K_XC_full = np.zeros((n_occ, n_virt, n_spin_orbitals, n_spin_orbitals))

        for spin_1 in "ab":

            for spin_2 in "ab":

                f_XC_block = f_XC[np.ix_(variables[spin_1][:n_variables], variables[spin_2][:n_variables])]

                orbitals = molecular_orbitals_on_grid[spin_orbitals[spin_2]]
                orbital_gradients = molecular_orbital_gradients_on_grid[:, spin_orbitals[spin_2]] if functional_class != "LDA" else None

                K_XC_full[np.ix_(occupied[spin_1], virtual[spin_1], spin_orbitals[spin_2], spin_orbitals[spin_2])] = contract_kernel_with_orbital_pairs(f_XC_block, df_ds_XC["".join(sorted(spin_1 + spin_2))], U[spin_1], T_gradient[spin_1], orbitals, orbital_gradients, spin_density_gradients[spin_2], functional_class)

    log("[Done]", calculation, 1, silent)

    if return_full_kernel:

        return K_XC, K_XC_full

    return K_XC
