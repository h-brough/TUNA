import numpy as np
from tuna_util import *
from scipy.integrate import lebedev_rule

# Note we used np.cbrt for cube roots as this makes a large difference to speed. Similarly, cubing via x * x * x instead of x ** 3 seems to be much faster
# This is onlt true for powers of 2 or 3, abvoe this the ** becomes faster




def set_up_integration_grid(basis_functions, atoms, bond_length, n_electrons, P_guess, calculation, silent=False):

    log(f" Setting up DFT integration grid with \"{calculation.grid_conv["name"]}\" accuracy...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    extent_multiplier = calculation.grid_conv["extent_multiplier"]
    integral_accuracy = calculation.grid_conv["integral_accuracy"] if not calculation.integral_accuracy_requested else calculation.integral_accuracy

    extent = extent_multiplier * np.max([atoms[i].real_vdw_radius for i in range(len(atoms))]) / 6

    n = int(integral_accuracy * 9)

    LEBEDEV_ORDERS = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])

    idx = np.abs(LEBEDEV_ORDERS - n).argmin()
    Lebedev_order = int(LEBEDEV_ORDERS[idx])

    n_radial = int(extent * integral_accuracy)


    points, weights = build_molecular_grid(extent, n_radial, Lebedev_order, bond_length, atoms)

    log("[Done]", calculation, 1, silent=silent)

    total_points = points.shape[1] * points.shape[2]
    points_per_atom = total_points // len(atoms)

    log(f"\n Integration grid has {n_radial} radial and {points.shape[2]} angular points, a Lebedev order of {Lebedev_order}.", calculation, 1, silent=silent)
    log(f" In total there are {total_points} grid points, {points_per_atom} per atom.", calculation, 1, silent=silent)

    log("\n Building guess density on grid...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()


    bfs_on_grid = construct_basis_functions_on_grid(basis_functions, points)

    bf_gradients_on_grid = construct_basis_function_gradients_on_grid(basis_functions, points) if calculation.functional.functional_class in ["GGA", "meta-GGA"] else None


    density = construct_density_on_grid(P_guess, bfs_on_grid)

    log("[Done]", calculation, 1, silent=silent)

    n_electrons_DFT = integrate_on_grid(density, weights)

    log(f"\n Integral of the guess density: {n_electrons_DFT:13.10f}\n", calculation, 1, silent=silent)



    if np.abs(n_electrons_DFT - n_electrons) > 0.00001:

        warning(" Integral of density is far from the number of electrons! Be careful with your results.")

        if np.abs(n_electrons_DFT - n_electrons) > 0.5:

            error("Integral for the density is completely wrong!")
    


    return bfs_on_grid, weights, points, bf_gradients_on_grid





def build_radial_and_angular_grid(radial_grid_cutoff, n_radial, lebedev_order, radial_power=3):


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
    x_sph, y_sph, z_sph = unit_sphere_directions    

    # Builds the Cartesian grid out of the radial and angular grids
    R = r[:, None]               

    atomic_points = np.stack((R * x_sph[None, :], R * y_sph[None, :], R * z_sph[None, :]), axis=0)
    
    # Radial weights scaled by R^2 to account for surface of sphere getting larger away from nucleus
    atomic_weights = weights_radial[:, None] * (R ** 2) * weights_angular[None, :]

    return atomic_points, atomic_weights







def calculate_Becke_diatomic_weights(X, Y, Z, bond_length, atoms, steepness=4):

    # Distance to the atomic centres for each point on the Cartesian grid
    R_A = (X ** 2 + Y ** 2 + Z ** 2) ** (1 / 2)
    R_B = (X ** 2 + Y ** 2 + (Z - bond_length) ** 2) ** (1 / 2)

    # This is -1 on atom A, 1 on atom B and 0 in the centre. It is an elliptical coordinate
    s = (R_A - R_B) / bond_length

    chi = atoms[0].real_vdw_radius / atoms[1].real_vdw_radius

    u = (chi - 1) / (chi + 1)
    a = u / (u ** 2 - 1)
    s = s + a * (1 - s ** 2)

    # The more p(x) is applied, the steeper the transition becomes and the more highly weighted points around the nuclei
    for i in range(steepness):

        s = (3 * s - s ** 3) / 2 
 
    # These weights are 1 on atom A, 0 on atom B and 0.5 when elliptical coordinate is zero. Vice versa for B.
    weights_A = (1 - s) / 2
    weights_B = (1 + s) / 2


    return weights_A, weights_B






def build_molecular_grid(radial_grid_cutoff, n_radial, Lebedev_order, bond_length, atoms):

    # Builds the radial and angular grid for the first atom
    points_A, atomic_weights_A = build_radial_and_angular_grid(radial_grid_cutoff, n_radial, Lebedev_order)

    X_A, Y_A, Z_A = points_A

    # If its just an atomic calculation, return this grid
    if len(atoms) == 1:

        return points_A, atomic_weights_A
    
    # For a diatomic, the grid will be the same, but the Z grid shifted by the bond length
    X_B, Y_B, Z_B = X_A, Y_A, Z_A + bond_length
    atomic_weights_B = atomic_weights_A

    # Builds the molecular Cartesian grid
    X = np.concatenate([X_A, X_B], axis=0)
    Y = np.concatenate([Y_A, Y_B], axis=0)
    Z = np.concatenate([Z_A, Z_B], axis=0)

    points = np.stack((X, Y, Z), axis=0)

    # Uses Becke atomic partitioning to get weights for each atom
    diatomic_weights_A, diatomic_weights_B = calculate_Becke_diatomic_weights(X, Y, Z, bond_length, atoms)

    N = X_A.shape[0]

    # Combines the atomic weights with the molecular weights
    weights_combined_A = atomic_weights_A * diatomic_weights_A[:N]
    weights_combined_B = atomic_weights_B * diatomic_weights_B[N:]

    # Forms total molecular weights array
    weights = np.concatenate([weights_combined_A, weights_combined_B], axis=0)

    return points, weights







def construct_basis_functions_on_grid(basis_functions, points):
    
    # This is for two-dimensional output plots
    if len(points) == 2: 
            
        X, Z = points

        Y = np.full_like(X, 0)

    # This is for three-dimensional DFT grids
    else:

        X, Y, Z = points


    _, N, M = points.shape

    basis_functions_on_grid = np.zeros((len(basis_functions), N, M))


    for i, bf in enumerate(basis_functions):

        X_relative = X - bf.origin[0]
        Y_relative = Y - bf.origin[1]
        Z_relative = Z - bf.origin[2]

        l, m, n = bf.shell
        r_squared = X_relative ** 2 + Y_relative ** 2 + Z_relative ** 2

        primitives = bf.norm[:, None, None] * (X_relative ** l) * (Y_relative ** m) * (Z_relative ** n) * np.exp(-bf.exps[:, None, None] * r_squared)

        basis_functions_on_grid[i] = np.einsum("ijk,i->jk", primitives, bf.coefs, optimize=True)


    return basis_functions_on_grid





def construct_basis_function_gradients_on_grid(basis_functions, points):

    if len(points) == 2:

        X, Z = points
        Y = np.full_like(X, 0)

    else:

        X, Y, Z = points

    _, N, M = points.shape

    n_basis = len(basis_functions)

    dphi_dx = np.zeros((n_basis, N, M))
    dphi_dy = np.zeros((n_basis, N, M))
    dphi_dz = np.zeros((n_basis, N, M))

    for i, bf in enumerate(basis_functions):

        X_relative = X - bf.origin[0]
        Y_relative = Y - bf.origin[1]
        Z_relative = Z - bf.origin[2]

        l, m, n = bf.shell
        r_squared = X_relative ** 2 + Y_relative ** 2 + Z_relative ** 2

        poly_x = X_relative ** l if l > 0 else np.ones_like(X_relative)
        poly_y = Y_relative ** m if m > 0 else np.ones_like(Y_relative)
        poly_z = Z_relative ** n if n > 0 else np.ones_like(Z_relative)

        P_poly = poly_x * poly_y * poly_z    

        common = bf.norm[:, None, None] * np.exp(-bf.exps[:, None, None] * r_squared)

        dP_dx_poly = l * (X_relative ** (l - 1)) * poly_y * poly_z if l > 0 else np.zeros_like(P_poly)
        dP_dy_poly = m * poly_x * (Y_relative ** (m - 1)) * poly_z if m > 0 else np.zeros_like(P_poly)
        dP_dz_poly = n * poly_x * poly_y * (Z_relative ** (n - 1)) if n > 0 else np.zeros_like(P_poly)

        dphi_dx_primitives = common * (dP_dx_poly - 2 * bf.exps[:, None, None] * X_relative * P_poly)
        dphi_dy_primitives = common * (dP_dy_poly - 2 * bf.exps[:, None, None] * Y_relative * P_poly)
        dphi_dz_primitives = common * (dP_dz_poly - 2 * bf.exps[:, None, None] * Z_relative * P_poly)

        dphi_dx[i] = np.einsum("ijk,i->jk", dphi_dx_primitives, bf.coefs, optimize=True)
        dphi_dy[i] = np.einsum("ijk,i->jk", dphi_dy_primitives, bf.coefs, optimize=True)
        dphi_dz[i] = np.einsum("ijk,i->jk", dphi_dz_primitives, bf.coefs, optimize=True)

    return dphi_dx, dphi_dy, dphi_dz







def calculate_density_gradient(P, bfs_on_grid, bf_gradients_on_grid):

    density_gradient = 2 * np.einsum("ij,ikl,ajkl->akl", P, bfs_on_grid, bf_gradients_on_grid, optimize=True)

    sigma = np.einsum("akl->kl", density_gradient ** 2, optimize=True)

    # It is important that this is cleaned at the square of the floor that the density and kinetic energy density are cleaned
    sigma = clean(sigma, 1e-52)

    return sigma, density_gradient






def calculate_kinetic_energy_density(P, bf_gradients_on_grid):

    tau = (1 / 2) * np.einsum("ij,aikl,ajkl->kl", P, bf_gradients_on_grid, bf_gradients_on_grid, optimize=True)

    tau = clean(tau)

    return tau




def integrate_on_grid(integrand, weights):

    """
    
    Uses quadrature to integrate something on a grid.

    Args:
        integrand (array): Integrand expressed on a grid
        weights (array): Weights for quadrature

    Returns:
        integral (float): Value of summed quadrature expression
    
    """

    integral = np.sum(integrand * weights)

    return integral






def construct_density_on_grid(P, atomic_orbitals, clean_density=True):
    
    """
    
    Constructs the electron density on the grid using the atomic orbitals, then cleans it up.

    Args:
        P (array): One-particle reduced density matrix
        atomic_orbitals (array): Atomic orbitals on molecular grid
    
    Returns:
        density (array): Electron density on molecular grid
    
    """


    density = np.einsum("ij,ikl,jkl->kl", P, atomic_orbitals, atomic_orbitals, optimize=True)

    if clean_density:
        
        density = clean(density)

    return density





def clean(function_on_grid, floor=1e-26):

    # Makes sure there are no zero or negative values in the electron density. 
    # Increasing the minimum messes up the B88 energy at the 6th decimal place
    
    function_on_grid = np.maximum(function_on_grid, floor)

    return function_on_grid




def clean_density_matrix(P, S, n_electrons):

    # Forces the trace of the density matrix to be correct

    P *= n_electrons / np.trace(P @ S) if n_electrons > 0 else 0

    return P





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





def calculate_seitz_radius(density):

    r_s = np.cbrt(3 / (4 * np.pi * density))

    return r_s



def calculate_zeta(alpha_density, beta_density):

    zeta = (alpha_density - beta_density) / (alpha_density + beta_density)

    return zeta


def calculate_f_zeta(zeta):

    f_zeta = (np.cbrt(1 + zeta) ** 4 + np.cbrt(1 - zeta) ** 4 - 2) / (np.cbrt(2) ** 4 - 2)

    return f_zeta



def calculate_f_prime_zeta(zeta):

    f_prime_zeta = (np.cbrt(1 + zeta) - np.cbrt(1 - zeta)) / (np.cbrt(2) ** 4 - 2) * 4 / 3

    return f_prime_zeta






def calculate_Slater_potential(density, calculation):

    alpha = calculation.X_alpha

    v_X = - (3 / 2 * alpha) * np.cbrt(3 / np.pi * density)

    e_X = 3 / 4 * v_X

    return v_X, e_X




def calculate_PW_potential(density, A, alpha_1, beta_1, beta_2, beta_3, beta_4, P):

    r_s = calculate_seitz_radius(density)

    Q_0 = -2 * A * (1 + alpha_1 * r_s)
    Q_1 = 2 * A * (beta_1 * r_s ** (1 / 2) + beta_2 * r_s + beta_3 * r_s ** (3 / 2) + beta_4 * r_s ** (P + 1))
    Q_1_prime = A * (beta_1 * r_s ** (-1 / 2) + 2 * beta_2 + 3 * beta_3 * r_s ** (1 / 2) + 2 * (P + 1) * beta_4 * r_s ** P)

    log_term = np.log(1 + 1 / Q_1)

    e_C = Q_0 * log_term

    de_dr = -2 * A * alpha_1 * log_term - Q_0 * Q_1_prime / (Q_1 ** 2 + Q_1)

    v_C = e_C - r_s / 3 * de_dr

    return v_C, e_C, de_dr







def calculate_VWN_potential(density, x_0, b, c, A):

    Q = (4 * c - b ** 2) ** (1 / 2)
    X_0 = x_0 ** 2 + b * x_0 + c
    c_1 = -b * x_0 / X_0
    c_2 = 2 * b * (c - x_0 ** 2) / (Q * X_0)

    r_s = calculate_seitz_radius(density)
    x = r_s ** (1 / 2)

    X = r_s + b * x + c

    log_term_1 = np.log(r_s / X) 
    log_term_2 = np.log((x - x_0) ** 2 / X)
    atan_term = np.arctan(Q / (2 * x + b))

    combo = (2 / x + 2 * c_1 / (x - x_0) - (2 * x + b) * (1 + c_1) / X - (1 / 2) * c_2 * Q / X)

    e_C = A * (log_term_1 + c_1 * log_term_2 + c_2 * atan_term)

    de_dr = (A / 2) * combo / x

    v_C = e_C - r_s / 3 * de_dr

    return v_C, e_C, de_dr





def calculate_VWN3_spin_interpolation(alpha_density, beta_density, density, e_C_0, de0_dr, e_C_1, de1_dr):

    zeta = calculate_zeta(alpha_density, beta_density)

    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)

    e_C = e_C_0 + (e_C_1 - e_C_0) * f_zeta

    r_s = calculate_seitz_radius(density)

    de_dr = de0_dr + (de1_dr - de0_dr) * f_zeta

    de_dzeta = (e_C_1 - e_C_0) * f_prime_zeta

    v_C_alpha = e_C - r_s / 3 * de_dr - (zeta - 1) * de_dzeta
    v_C_beta = e_C - r_s / 3 * de_dr - (zeta + 1) * de_dzeta


    return v_C_alpha, v_C_beta, e_C





def calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr):

    zeta = calculate_zeta(alpha_density, beta_density)

    alpha = -1 * minus_alpha

    zeta_4 = zeta ** 4
    f_zeta = calculate_f_zeta(zeta)
    f_prime_zeta = calculate_f_prime_zeta(zeta)
    f_prime_prime_at_zero = 8 / (9 * (np.cbrt(2) ** 4 - 2))

    e_C = e_C_0 + alpha * f_zeta / f_prime_prime_at_zero * (1 - zeta_4) + (e_C_1 - e_C_0) * f_zeta * zeta_4
    
    r_s = calculate_seitz_radius(density)

    de_dr = de0_dr * (1 - f_zeta * zeta_4) + de1_dr * f_zeta * zeta_4 + dalpha_dr * f_zeta * (1 - zeta_4) / f_prime_prime_at_zero

    de_dzeta = 4 * zeta ** 3 * f_zeta * (e_C_1 - e_C_0 - alpha / f_prime_prime_at_zero) + f_prime_zeta * (zeta_4 * (e_C_1 - e_C_0) + (1 - zeta_4) * alpha / f_prime_prime_at_zero)


    v_C_alpha = e_C - r_s / 3 * de_dr - (zeta - 1) * de_dzeta 
    v_C_beta = e_C - r_s / 3 * de_dr - (zeta + 1) * de_dzeta 


    return v_C_alpha, v_C_beta, e_C







def calculate_restricted_Slater_exchange_potential(density, calculation, sigma, tau):

    df_dn, e_X = calculate_Slater_potential(density, calculation)

    return df_dn, None, None, e_X


def calculate_unrestricted_Slater_exchange_potential(density, calculation, sigma, tau):

    df_dn, e_X = calculate_Slater_potential(2 * density, calculation)

    return df_dn, None, None, e_X


def calculate_restricted_PBE_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_PBE_exchange_potential(density, sigma, calculation)

    return df_dn, df_ds, None, e_X


def calculate_unrestricted_PBE_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_PBE_exchange_potential(2 * density, 4 * sigma, calculation)

    return df_dn, df_ds * 2, None, e_X


def calculate_restricted_B88_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_B88_exchange_potential(density / 2, sigma / 4, calculation)

    return df_dn, df_ds / 2, None, e_X


def calculate_unrestricted_B88_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_B88_exchange_potential(density, sigma, calculation)

    return df_dn, df_ds, None, e_X

def calculate_restricted_PW91_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_PW91_exchange_potential(density, sigma, calculation)

    return df_dn, df_ds, None, e_X


def calculate_unrestricted_PW91_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_PW91_exchange_potential(2 * density, 4 * sigma, calculation)

    return df_dn, df_ds * 2, None, e_X

def calculate_restricted_mPW91_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_mPW91_exchange_potential(density / 2, sigma / 4, calculation)

    return df_dn, df_ds / 2, None, e_X


def calculate_unrestricted_mPW91_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_X = calculate_mPW91_exchange_potential(density, sigma, calculation)

    return df_dn, df_ds, None, e_X



def calculate_restricted_B3_exchange_potential(density, calculation, sigma, tau):
    
    df_dn_LDA, e_X_LDA = calculate_Slater_potential(density, calculation)
    
    df_dn_B88, df_ds_B88, _, e_X_B88 = calculate_restricted_B88_exchange_potential(density, calculation, sigma, tau)

    df_dn = 0.9 * df_dn_B88 + 0.1 * df_dn_LDA
    df_ds = 0.9 * df_ds_B88
    e_X = 0.9 * e_X_B88 + 0.1 * e_X_LDA 

    return df_dn, df_ds, None, e_X


def calculate_unrestricted_B3_exchange_potential(density, calculation, sigma, tau):

    df_dn_LDA, e_X_LDA = calculate_Slater_potential(2 * density, calculation)
    df_dn_B88, df_ds_B88, _, e_X_B88 = calculate_unrestricted_B88_exchange_potential(density, calculation, sigma, tau)

    df_dn = 0.9 * df_dn_B88 + 0.1 * df_dn_LDA
    df_ds = 0.9 * df_ds_B88
    e_X = 0.9 * e_X_B88 + 0.1 * e_X_LDA 
    
    return df_dn, df_ds, None, e_X



def calculate_restricted_TPSS_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, df_dt, e_X = calculate_TPSS_exchange_potential(density, sigma, tau, calculation)

    return df_dn, df_ds, df_dt, e_X



def calculate_unrestricted_TPSS_exchange_potential(density, calculation, sigma, tau):

    df_dn, df_ds, df_dt, e_X = calculate_TPSS_exchange_potential(2 * density, 4 * sigma, 2 * tau, calculation)

    return df_dn, df_ds * 2, df_dt, e_X


def calculate_restricted_VWN3_correlation_potential(density, calculation, sigma, tau):

    df_dn, e_C, _ = calculate_VWN_potential(density, -0.409286, 13.0720, 42.7198, 0.0310907)

    return df_dn, None, None, e_C


def calculate_unrestricted_VWN3_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, calculation):

    _, e_C_0, de0_dr = calculate_VWN_potential(density, -0.409286, 13.0720, 42.7198, 0.0310907)
    _, e_C_1, de1_dr = calculate_VWN_potential(density, -0.743294, 20.1231, 101.578, 0.01554535)

    v_C_alpha, v_C_beta, e_C = calculate_VWN3_spin_interpolation(alpha_density, beta_density, density, e_C_0, de0_dr, e_C_1, de1_dr)

    return v_C_alpha, v_C_beta, None, None, None, e_C


def calculate_restricted_VWN5_correlation_potential(density, calculation, sigma, tau):

    df_dn, e_C, _ = calculate_VWN_potential(density, -0.10498, 3.72744, 12.9352, 0.0310907)

    return df_dn, None, None, e_C


def calculate_unrestricted_VWN5_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, calculation):

    _, e_C_0, de0_dr = calculate_VWN_potential(density, -0.10498, 3.72744, 12.9352, 0.0310907)
    _, e_C_1, de1_dr = calculate_VWN_potential(density, -0.32500, 7.06042, 18.0578, 0.01554535)
    _, minus_alpha, dalpha_dr = calculate_VWN_potential(density, -0.0047584, 1.13107, 13.0045, 1 / (6 * np.pi ** 2))

    v_C_alpha, v_C_beta, e_C = calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, -dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr)
    
    return v_C_alpha, v_C_beta, None, None, None, e_C


def calculate_restricted_PW_correlation_potential(density, calculation, sigma, tau):

    # Note - from this is "modified" PW from LibXC has more significant figures than original paper
    df_dn, e_C, _ = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    return df_dn, None, None, e_C


def calculate_unrestricted_PW_correlation_potential(alpha_density, beta_density, density, sigma, calculation):
    

    _, e_C_0, de0_dr = calculate_PW_potential(density, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    _, e_C_1, de1_dr = calculate_PW_potential(density, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    _, minus_alpha, dalpha_dr = calculate_PW_potential(density, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    df_dn_alpha, df_dn_beta, e_C = calculate_VWN5_spin_interpolation(alpha_density, beta_density, density, minus_alpha, -dalpha_dr, e_C_0, de0_dr, e_C_1, de1_dr)


    return df_dn_alpha, df_dn_beta, None, None, None, e_C



def calculate_restricted_PBE_correlation_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_C = calculate_PBE_correlation_potential(density, sigma, calculation)

    return df_dn, df_ds, None, e_C



def calculate_unrestricted_PBE_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):

    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C = calculate_UPBE_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab)

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C



def calculate_restricted_LYP_correlation_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_C = calculate_LYP_correlation_potential(density, sigma)

    return df_dn, df_ds, None, e_C


def calculate_unrestricted_LYP_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):

    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C = calculate_ULYP_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab)

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C


def calculate_restricted_P86_correlation_potential(density, calculation, sigma, tau):

    df_dn, df_ds, e_C = calculate_P86_correlation_potential(density, sigma)

    return df_dn, df_ds, None, e_C


def calculate_unrestricted_P86_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):

    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C = calculate_UP86_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta)

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, None, None, e_C


def calculate_restricted_TPSS_correlation_potential(density, calculation, sigma, tau):

    df_dn, df_ds, df_dt, e_C = calculate_TPSS_correlation_potential(density, sigma, tau, calculation)

    return df_dn, df_ds, df_dt, e_C


def calculate_unrestricted_TPSS_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta, calculation):

    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C = calculate_UTPSS_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta)
    
    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C



def calculate_restricted_3LYP_correlation_potential(density, calculation, sigma, tau):

    if "G" in calculation.method:

        df_dn_LDA, _, _, e_C_LDA = calculate_restricted_VWN3_correlation_potential(density, calculation, sigma, tau)

    else:

        df_dn_LDA, _, _, e_C_LDA = calculate_restricted_VWN5_correlation_potential(density, calculation, sigma, tau)

    df_dn_LYP, df_ds_LYP, e_C_LYP = calculate_LYP_correlation_potential(density, sigma)

    df_dn = 0.81 * df_dn_LYP + 0.19 * df_dn_LDA
    df_ds = 0.81 * df_ds_LYP
    e_C = 0.81 * e_C_LYP + 0.19 * e_C_LDA

    return df_dn, df_ds, None, e_C




def calculate_unrestricted_3LYP_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, calculation):

    if "G" in calculation.method:

        df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, e_C_LDA = calculate_unrestricted_VWN3_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, calculation)

    else:

        df_dn_alpha_LDA, df_dn_beta_LDA, _, _, _, e_C_LDA = calculate_unrestricted_VWN5_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, calculation)


    df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C = calculate_ULYP_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab)
    
    df_dn_alpha = 0.81 * df_dn_alpha + 0.19 * df_dn_alpha_LDA
    df_dn_beta = 0.81 * df_dn_beta + 0.19 * df_dn_beta_LDA
    
    df_ds_aa = 0.81 * df_ds_aa
    df_ds_bb = 0.81 * df_ds_bb
    df_ds_ab = 0.81 * df_ds_ab
    
    e_C = 0.81 * e_C + 0.19 * e_C_LDA


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C



def calculate_PBE_exchange_potential(density, sigma, calculation):

    s_squared = sigma / (np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8)
    
    kappa = 0.804
    mu = 0.21952

    denom = 1 / (1 + mu / kappa * s_squared)

    F_X = 1 + kappa - kappa * denom

    e_X_LDA = calculate_restricted_Slater_exchange_potential(density, calculation, sigma, None)[3]

    e_X = e_X_LDA * F_X

    denom_derivative = s_squared * denom ** 2

    df_ds = density * e_X_LDA * mu * denom_derivative / sigma
    df_dn = 4 * e_X_LDA / 3 * (F_X - 2 * mu * denom_derivative) 

    return df_dn, df_ds, e_X





def calculate_PBE_correlation_potential(density, sigma, calculation):

    v_C_LDA, _, _, e_C_LDA = calculate_restricted_PW_correlation_potential(density, calculation, sigma, None)

    de_C_LDA_dn = (v_C_LDA - e_C_LDA) / density

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.06672455060314922

    k_F = np.cbrt(3 * np.pi ** 2 * density)

    t_squared = sigma * np.pi / (16 * k_F * density ** 2)

    exp_factor = np.exp(-e_C_LDA / gamma)
    A = beta / (gamma * (exp_factor - 1))

    k = 1 + A * t_squared
    D = k + A ** 2 * t_squared ** 2

    X = (beta / gamma) * t_squared * k / D

    H = gamma * np.log(1 + X)
    e_C = e_C_LDA + H

    dA_dn = (A ** 2 / beta) * exp_factor * de_C_LDA_dn

    pref = 1 / ((1 + X) * D ** 2)
    common = beta * (1 + 2 * A * t_squared)
    dH_dn = pref * (common * -(7 / 3) * t_squared / density - beta * A * t_squared ** 3 * (A * t_squared + 2) * dA_dn)
    dH_ds = pref * common * t_squared / sigma

    df_dn = e_C + density * (de_C_LDA_dn + dH_dn)
    df_ds = density * dH_ds

    return df_dn, df_ds, e_C








def calculate_UPBE_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab):

    sigma = sigma_aa + sigma_bb + 2 * sigma_ab

    gamma = (1 - np.log(2)) / np.pi ** 2
    beta = 0.06672455060314922
    B = beta / gamma

    v_C_LDA_alpha, v_C_LDA_beta, _, _, _, e_C_LDA = calculate_unrestricted_PW_correlation_potential(alpha_density, beta_density, density, None, None)

    zeta = calculate_zeta(alpha_density, beta_density)


    one_plus_zeta  = 1 + zeta
    one_minus_zeta = clean(1 - zeta)

    cbrt_plus  = clean(np.cbrt(one_plus_zeta))
    cbrt_minus = clean(np.cbrt(one_minus_zeta))

    phi = (1 / 2) * (cbrt_plus ** 2 + cbrt_minus ** 2)
    phi_prime = (1 / cbrt_plus - 1 / cbrt_minus) / 3

    k_F = np.cbrt(3 * np.pi ** 2 * density)
    n2 = density ** 2
    phi2 = phi ** 2
    phi3 = phi ** 3

    T = sigma * np.pi / 16 / (phi2 * k_F * n2)

    Q = np.exp(-e_C_LDA / (gamma * phi3))
    A = B / (Q - 1.0)
    A2 = A ** 2
    T2 = T ** 2

    D = 1 + A * T + A2 * T2
    N = B * T * (1 + A * T)
    X = N / D

    H = gamma * phi3 * np.log(1 + X)     
    e_C = e_C_LDA + H

    inv_n = 1 / density
    inv_n2 = inv_n ** 2

    dphi_dn_alpha = phi_prime * (2 * beta_density * inv_n2)
    dphi_dn_beta = -phi_prime * (2 * alpha_density * inv_n2)

    dT_dn = -7 / 3 * T * inv_n
    dT_dphi = -2 * T / phi

    dT_dn_alpha = dT_dn + dT_dphi * dphi_dn_alpha
    dT_dn_beta  = dT_dn + dT_dphi * dphi_dn_beta

    T_over_sigma = np.pi / 16 / (phi2 * k_F * n2)
    dT_dsigma = T_over_sigma

    de_C_LDA_dn_alpha = (v_C_LDA_alpha - e_C_LDA) * inv_n
    de_C_LDA_dn_beta  = (v_C_LDA_beta  - e_C_LDA) * inv_n

    phi4 = phi2 ** 2
    dA_dE   = (A2 / beta) * Q / phi3
    dA_dphi = -3 * e_C_LDA * Q * A2 / (beta * phi4)

    dA_dn_alpha = dA_dE * de_C_LDA_dn_alpha + dA_dphi * dphi_dn_alpha
    dA_dn_beta = dA_dE * de_C_LDA_dn_beta + dA_dphi * dphi_dn_beta

    dD_dT = A + 2 * A2 * T
    dX_dT = (B * (1 + 2 * A * T) * D - N * dD_dT) / D ** 2

    dD_dA = T + 2 * A * T2
    dX_dA = (B * T2 * D - N * dD_dA) / D ** 2

    C1 = 3 * gamma * phi2 * np.log(1 + X)
    C2 = gamma * phi3 / (1 + X)

    dH_dn_alpha = C1 * dphi_dn_alpha + C2 * (dX_dT * dT_dn_alpha + dX_dA * dA_dn_alpha)
    dH_dn_beta = C1 * dphi_dn_beta  + C2 * (dX_dT * dT_dn_beta  + dX_dA * dA_dn_beta)

    df_dn_alpha = e_C + density * (de_C_LDA_dn_alpha + dH_dn_alpha)
    df_dn_beta = e_C + density * (de_C_LDA_dn_beta + dH_dn_beta)

    df_dsigma = density * C2 * dX_dT * dT_dsigma

    df_dsigma_aa, df_dsigma_bb, df_dsigma_ab = df_dsigma, df_dsigma, 2 * df_dsigma


    return df_dn_alpha, df_dn_beta, df_dsigma_aa, df_dsigma_bb, df_dsigma_ab, e_C





def calculate_B88_exchange_potential(density, sigma, calculation):

    e_X_LDA = calculate_Slater_potential(density, calculation)[1]

    beta = 0.0042 
    C = 2 / np.cbrt(4)

    cube_root_density = np.cbrt(density) 
    x = sigma ** (1 / 2) / cube_root_density ** 4

    A = np.arcsinh(x)

    D = 1 + 6 * beta * x * A
    Dprime = 6 * beta * (A + x / (1 + x ** 2) ** (1 / 2))

    e_X = C * e_X_LDA - beta * cube_root_density * x ** 2 / D

    df_dn = (e_X + C * e_X_LDA / 3 + beta * cube_root_density * (7 * x ** 2 * D - 4 * x ** 3 * Dprime) / (3 * D ** 2)) 

    df_ds = -beta * density * cube_root_density * (x ** 2 * D - 0.5 * x ** 3 * Dprime) / (sigma * D ** 2)
    
    return df_dn, df_ds, e_X




def calculate_PW91_exchange_potential(density, sigma, calculation):

    df_dn_LDA, e_X_LDA = calculate_Slater_potential(density, calculation)

    denom = np.cbrt(576 * np.pi ** 4) * np.cbrt(density) ** 8
    s_squared = sigma / denom
    s = np.sqrt(s_squared)

    u = 7.7956 * s
    asinh_u = np.arcsinh(u)
    E = np.exp(-100 * s_squared)

    x = 1 + 0.19645 * s * asinh_u
    num = x + (0.2743 - 0.1508 * E) * s_squared
    den = x + 0.004 * s_squared * s_squared
    F_x = num / den

    e_X = e_X_LDA * F_x
    f_LDA = density * e_X_LDA

    dx_ds2 = 0.19645 * 0.5 * (asinh_u / s + 7.7956 / np.sqrt(1 + u * u))
    dA_ds2 = 0.2743 - 0.1508 * E + 15.08 * s_squared * E 
    dF_ds2 = ((dx_ds2 + dA_ds2) * den - num * (dx_ds2 + 0.008 * s_squared)) / (den * den)

    df_ds = f_LDA * dF_ds2 / denom
    df_dn = df_dn_LDA * F_x - f_LDA * dF_ds2 * (8 / 3) * s_squared / density

    return df_dn, df_ds, e_X





def calculate_mPW91_exchange_potential(density, sigma, calculation):

    beta = 5.0 * (36.0 * np.pi) ** (-5.0 / 3.0)
    b, c, d = 0.00426, 1.6455, 3.72
    eps = 1e-6
    cbrt2 = np.cbrt(2.0)

    df_dn_LDA, e_X_LDA = calculate_Slater_potential(density, calculation)

    n = density
    n13 = np.cbrt(n)          # n^(1/3)
    n43 = n * n13             # n^(4/3)

    sqrt_sigma = np.sqrt(sigma)
    x = sqrt_sigma / n43
    x2 = x * x
    xd = x ** d

    G = np.exp(-c * x2)
    asinhx = np.arcsinh(x)

    # K = 2^(1/3) * e_X_LDA / n^(1/3); constant in n for Slater exchange
    K = (e_X_LDA * cbrt2) / n13

    N = b * x2 - (b - beta) * x2 * G - eps * xd
    D = 1.0 + 6.0 * b * x * asinhx - eps * xd / K
    invD = 1.0 / D
    F = N * invD

    # x^(d-1) = xd/x, with   limit x->0 (0) for d>1
    xdm1 = xd/x

    dNdx = 2.0 * b * x - 2.0 * (b - beta) * x * G * (1.0 - c * x2) - eps * d * xdm1
    dDdx = 6.0 * b * (asinhx + x / np.sqrt(1.0 + x2)) - eps * d * xdm1 / K

    dFdx = (dNdx - F * dDdx) * invD  # (dNdx - F*dDdx)/D

    e_X = e_X_LDA * cbrt2 - F * n13

    df_dn = cbrt2 * df_dn_LDA - (4.0 / 3.0) * n13 * (F - x * dFdx)


    df_ds = -(0.5 / n43) * dFdx/x

    return df_dn, df_ds, e_X






def calculate_LYP_correlation_potential(density, sigma):
  
    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)
    inv_cbrt_density = 1 / cbrt_density

    X = 1 + d * inv_cbrt_density

    C2 = 6 / 10 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8

    w = inv_cbrt_density ** 11 * np.exp(-c * inv_cbrt_density) / X
    delta = inv_cbrt_density * (c + d / X)

    minus_a_b_w = -a * b * w * density
    
    w_prime_over_w = -(1 / 3) * inv_cbrt_density ** 4 * (11 * cbrt_density - c - d / X)
    delta_prime = (1 / 3) * (d ** 2 * inv_cbrt_density ** 5 / X ** 2 - delta * inv_density)

    df_ds = minus_a_b_w * density * (-7 * delta - 3) / 72

    df_dn = -a / X + minus_a_b_w * sigma * (-1 / 12 - 7 * delta / 36 + density * (-7 * delta_prime / 72 + w_prime_over_w * (-1 / 24 - 7 * delta / 72)))
    df_dn += density * (- a * d / (3 * X ** 2 * cbrt_density ** 4) - 7 * C2 * a * b * w / 3 - (1 / 2) * C2 * a * b * density * w_prime_over_w * w)
    
    e_C = (1 / 2) * C2 * minus_a_b_w - minus_a_b_w * sigma * (7 * delta + 3) / 72 - a / X
    

    return df_dn, df_ds, e_C







def calculate_ULYP_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab):

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349

    cbrt_alpha_density = np.cbrt(alpha_density)
    cbrt_beta_density = np.cbrt(beta_density)

    inv_density = 1 / density
    cbrt_density = np.cbrt(density)

    inv_cbrt_density = 1 / cbrt_density
    X = (1 + d * inv_cbrt_density)

    C = np.cbrt(2) ** 11 * 3 / 10 * np.cbrt(3 * np.pi ** 2) ** 2

    density_product = alpha_density * beta_density
    densities_power_sum = cbrt_alpha_density ** 8 + cbrt_beta_density ** 8

    w = inv_cbrt_density ** 11 * np.exp(-c * inv_cbrt_density) / X
    delta = inv_cbrt_density * (c + d / X)

    minus_a_b_w = -a * b * w

    w_prime = -(1 / 3) * inv_cbrt_density ** 4 * w * (11 * cbrt_density - c - d / X)
    delta_prime = (1 / 3) * (d ** 2 * inv_cbrt_density ** 5 / X ** 2 - delta * inv_density)

    w_prime_over_w = -(1 / 3) * inv_cbrt_density ** 4 * (11 * cbrt_density - c - d / X)

    df_ds_aa = minus_a_b_w * ((1 / 9) * density_product * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - beta_density ** 2)
    df_ds_bb = minus_a_b_w * ((1 / 9) * density_product * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - alpha_density ** 2)
    df_ds_ab = minus_a_b_w * ((1 / 9) * density_product * (47 - 7 * delta) - (4 / 3) * density ** 2)

    f_C = density_product * (C * minus_a_b_w * densities_power_sum - 4 * a / X * inv_density) + df_ds_aa * sigma_aa + df_ds_bb * sigma_bb + df_ds_ab * sigma_ab

    d2f_dn_a_ds_aa = w_prime_over_w * df_ds_aa + minus_a_b_w * ((1 / 9) * beta_density * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - (1 / 9) * density_product * (delta_prime * (3 + alpha_density * inv_density) + (delta - 11) * beta_density * inv_density ** 2))
    d2f_dn_b_ds_bb = w_prime_over_w * df_ds_bb + minus_a_b_w * ((1 / 9) * alpha_density * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - (1 / 9) * density_product * (delta_prime * (3 + beta_density * inv_density) + (delta - 11) * alpha_density * inv_density ** 2))

    d2f_dn_a_ds_ab = w_prime_over_w * df_ds_ab + minus_a_b_w * ((1 / 9) * beta_density * (47 - 7 * delta) - (7 / 9) * density_product * delta_prime - (8 / 3) * density)
    d2f_dn_b_ds_ab = w_prime_over_w * df_ds_ab + minus_a_b_w * ((1 / 9) * alpha_density * (47 - 7 * delta) - (7 / 9) * density_product * delta_prime - (8 / 3) * density)

    d2f_dn_a_ds_bb = w_prime_over_w * df_ds_bb + minus_a_b_w * ((1 / 9) * beta_density * (1 - 3 * delta - (delta - 11) * beta_density * inv_density) - (1 / 9) * density_product * ((3 + beta_density * inv_density) * delta_prime - (delta - 11) * beta_density * inv_density ** 2) - 2 * alpha_density)
    d2f_dn_b_ds_aa = w_prime_over_w * df_ds_aa + minus_a_b_w * ((1 / 9) * alpha_density * (1 - 3 * delta - (delta - 11) * alpha_density * inv_density) - (1 / 9) * density_product * ((3 + alpha_density * inv_density) * delta_prime - (delta - 11) * alpha_density * inv_density ** 2) - 2 * beta_density)

    df_dn_alpha = -4 * a / X * density_product * inv_density * ((1 / 3) * d * inv_cbrt_density ** 4 / X + 1 / alpha_density - inv_density) - C * a * b * (w_prime * density_product * densities_power_sum + w * beta_density * (11 / 3 * cbrt_alpha_density ** 8 + cbrt_beta_density ** 8)) + d2f_dn_a_ds_aa * sigma_aa + d2f_dn_a_ds_bb * sigma_bb + d2f_dn_a_ds_ab * sigma_ab                                                         
    df_dn_beta = -4 * a / X * density_product * inv_density * ((1 / 3) * d * inv_cbrt_density ** 4 / X + 1 / beta_density - inv_density) - C * a * b * (w_prime * density_product * densities_power_sum + w * alpha_density * (11 / 3 * cbrt_beta_density ** 8 + cbrt_alpha_density ** 8)) + d2f_dn_b_ds_bb * sigma_bb + d2f_dn_b_ds_aa * sigma_aa + d2f_dn_b_ds_ab * sigma_ab                                                         
       
    e_C = f_C * inv_density


    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C







exchange_functionals = {

    "S": calculate_restricted_Slater_exchange_potential,
    "US": calculate_unrestricted_Slater_exchange_potential,
    "PBE": calculate_restricted_PBE_exchange_potential,
    "UPBE": calculate_unrestricted_PBE_exchange_potential,
    "B": calculate_restricted_B88_exchange_potential,
    "UB": calculate_unrestricted_B88_exchange_potential,
    "B3": calculate_restricted_B3_exchange_potential,
    "UB3": calculate_unrestricted_B3_exchange_potential,
    "TPSS": calculate_restricted_TPSS_exchange_potential,
    "UTPSS": calculate_unrestricted_TPSS_exchange_potential,
    "PW": calculate_restricted_PW91_exchange_potential,
    "UPW": calculate_unrestricted_PW91_exchange_potential,
    "MPW": calculate_restricted_mPW91_exchange_potential,
    "UMPW": calculate_unrestricted_mPW91_exchange_potential,
}




correlation_functionals = {

    "VWN3": calculate_restricted_VWN3_correlation_potential,
    "UVWN3": calculate_unrestricted_VWN3_correlation_potential,
    "VWN5": calculate_restricted_VWN5_correlation_potential,
    "UVWN5": calculate_unrestricted_VWN5_correlation_potential,
    "PW": calculate_restricted_PW_correlation_potential,
    "UPW": calculate_unrestricted_PW_correlation_potential,
    "P86": calculate_restricted_P86_correlation_potential,
    "UP86": calculate_unrestricted_P86_correlation_potential,
    "PBE": calculate_restricted_PBE_correlation_potential,
    "UPBE": calculate_unrestricted_PBE_correlation_potential,
    "LYP": calculate_restricted_LYP_correlation_potential,
    "ULYP": calculate_unrestricted_LYP_correlation_potential,    
    "3LYP": calculate_restricted_3LYP_correlation_potential,
    "U3LYP": calculate_unrestricted_3LYP_correlation_potential,
    "TPSS": calculate_restricted_TPSS_correlation_potential,
    "UTPSS": calculate_unrestricted_TPSS_correlation_potential,
}




def calculate_P86_correlation_potential(density, sigma):

    alpha, beta, gamma, delta, f_tilde = 0.023266, 0.000007389, 8.723, 0.472, 0.11

    r_s = calculate_seitz_radius(density)

    N = 0.002568 + alpha * r_s + beta * r_s ** 2
    D = 1 + gamma * r_s + delta * r_s ** 2 + 1e4 * beta * r_s ** 3

    C = 0.001667 + N / D
    C_inf = 0.004235

    phi = 1.745 * f_tilde * C_inf / C * sigma ** (1 / 2) / density ** (7 / 6)

    df_LDA_dn, _, _, e_C_LDA = calculate_restricted_PW_correlation_potential(density, None, sigma, None)

    H = C * sigma * np.exp(-phi) / np.cbrt(density) ** 7

    e_C = e_C_LDA + H

    df_ds = C * np.exp(-phi) / np.cbrt(density) ** 4 * (1 - phi / 2)

    dN_dr = alpha + 2 * beta * r_s
    dD_dr = gamma + 2 * delta * r_s + 3e4 * beta * r_s ** 2
    dC_dr = (dN_dr * D - N * dD_dr) / D ** 2
    dC_dn = dC_dr * (-(1 / 3) * r_s / density)

    dH_dn = H * ((1 + phi) * (density * dC_dn / C) + (7 / 6) * (phi - 2))

    df_dn = df_LDA_dn + H + dH_dn

    return df_dn, df_ds, e_C



def calculate_UP86_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta):

    alpha, beta, gamma, delta, f_tilde = 0.023266, 0.000007389, 8.723, 0.472, 0.11

    sigma = sigma_aa + sigma_bb + 2 * sigma_ab

    r_s = calculate_seitz_radius(density)
    zeta = calculate_zeta(alpha_density, beta_density)

    N = 0.002568 + alpha * r_s + beta * r_s ** 2
    D = 1 + gamma * r_s + delta * r_s ** 2 + 1e4 * beta * r_s ** 3

    C = 0.001667 + N / D
    C_inf = 0.004235

    phi = 1.745 * f_tilde * C_inf / C * sigma ** (1 / 2) / density ** (7 / 6)

    p = np.cbrt(1 + zeta)
    m = np.cbrt(1 - zeta)
    S = p**5 + m**5
    d = np.sqrt(S/2)
    df_dn_alpha, df_dn_beta, _,_, _, e_C_LDA =  calculate_unrestricted_PW_correlation_potential(alpha_density, beta_density, density, sigma, None)

    H = (C * sigma * np.exp(-phi) / np.cbrt(density) ** 7) / d
    e_C = e_C_LDA + H

    df_dsigma = (C * np.exp(-phi) / np.cbrt(density) ** 4 * (1 - phi / 2)) / d
    df_ds_aa, df_ds_bb, df_ds_ab = df_dsigma, df_dsigma, 2 * df_dsigma

    dN_dr = alpha + 2 * beta * r_s
    dD_dr = gamma + 2 * delta * r_s + 3e4 * beta * r_s ** 2
    dC_dr = (dN_dr * D - N * dD_dr) / D ** 2
    dC_dn = dC_dr * (-(1 / 3) * r_s / density)

    dH_dn = H * ((1 + phi) * (density * dC_dn / C) + (7 / 6) * (phi - 2))

    dln_inv_d_dzeta = -(5 / 6) * (p**2 - m**2) / S
    dzeta_dn_alpha = 2 * beta_density / density ** 2
    dzeta_dn_beta  = -2 * alpha_density / density ** 2

    df_dn_alpha = df_dn_alpha + H + dH_dn + (density * H) * dln_inv_d_dzeta * dzeta_dn_alpha
    df_dn_beta  = df_dn_beta  + H + dH_dn + (density * H) * dln_inv_d_dzeta * dzeta_dn_beta

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, e_C




def calculate_TPSS_exchange_potential(density, sigma, tau, calculation):

    b, c, e, kappa, mu = 0.40, 1.59096, 1.537, 0.804, 0.21951

    df_dn_LDA, e_X_LDA = calculate_Slater_potential(density, calculation)

    p = sigma / (den_p := 4 * np.cbrt(3 * np.pi ** 2) ** 2 * np.cbrt(density) ** 8)

    z = sigma / (8 * density * tau)

    alpha = (5 * p / 3) * (1 / z - 1)

    q_tilde = (9 / 20) * (alpha - 1) / (1 + b * alpha * (alpha - 1)) ** (1 / 2) + 2 * p / 3

    sqrt_e = e ** (1 / 2)
    z2 = z ** 2
    t1 = 1 + z2
    A = 10 / 81 + c * z2 / (t1 ** 2)
    S = np.sqrt(0.5 * ((3 / 5 * z) ** 2 + p ** 2))

    num = A * p + (146 / 2025) * q_tilde ** 2 - (73 / 405) * q_tilde * S + (10 / 81) ** 2 / kappa * p ** 2 + 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z2 + e * mu * p ** 3
    t = 1 + sqrt_e * p
    den = t ** 2
    x = num / den

    F_x = 1 + kappa - kappa ** 2 / (kappa + x)

    e_X = e_X_LDA * F_x

    dp = np.stack([-(8 / 3) * p / (density), 1 / (den_p), np.zeros_like(p)])
    dz = np.stack([-z / (density), 1 / (8 * density * tau), -z / (tau)])

    inv_z = 1 / z
    dalpha = (5 / 3) * ((inv_z - 1) * dp - p * (inv_z ** 2) * dz)

    g = 1 + b * alpha * (alpha - 1)
    sqrt_g = g ** (1 / 2)
    dh_dalpha = 1 / sqrt_g - (alpha - 1) * b * (2 * alpha - 1) / (2 * g * sqrt_g)
    dq = (9 / 20) * dh_dalpha * dalpha + (2 / 3) * dp

    dA_dz = c * 2 * z * (1 - z2) / (t1 ** 3)
    dS = ((3 / 5) ** 2 * z * dz + p * dp) / (2 * S)

    dnum = (A * dp + p * dA_dz * dz) + 2 * (146 / 2025) * q_tilde * dq - (73 / 405) * (dq * S + q_tilde * dS) + 2 * (10 / 81) ** 2 / kappa * p * dp + 2 * 2 * sqrt_e * (10 / 81) * (3 / 5) ** 2 * z * dz + 3 * e * mu * p ** 2 * dp
    dx = (dnum * den - num * (2 * sqrt_e * t * dp)) / den ** 2
    dF = (kappa / (kappa + x)) ** 2 * dx

    f_LDA = density * e_X_LDA
    df_dn = df_dn_LDA * F_x + f_LDA * dF[0]
    df_ds = f_LDA * dF[1]
    df_dt = f_LDA * dF[2]


    return df_dn, df_ds, df_dt, e_X




def calculate_UTPSS_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab, tau_alpha, tau_beta):

    density = alpha_density + beta_density
    sigma = sigma_aa + sigma_bb + 2 * sigma_ab
    tau = tau_alpha + tau_beta

    C_d, d = None, 2.8  # keep d separate; C is the TPSS C(zeta,xi), computed below

    zeros = np.zeros_like(density)

    # PBE correlation (spin-polarized)
    df_dna_PBE, df_dnb_PBE, df_dsaa_PBE, df_dsbb_PBE, df_dsab_PBE, e_C_PBE = \
        calculate_UPBE_correlation_potential(alpha_density, beta_density, density, sigma_aa, sigma_bb, sigma_ab)

    # one-spin limits for the max construction in TPSS
    df_dna_a0, _, df_dsaa_a0, _, _, e_C_a0 = \
        calculate_UPBE_correlation_potential(alpha_density, zeros, alpha_density, sigma_aa, zeros, zeros)

    _, df_dnb_0b, _, df_dsbb_0b, _, e_C_0b = \
        calculate_UPBE_correlation_potential(zeros, beta_density, beta_density, zeros, sigma_bb, zeros)

    inv_n = 1 / density

    # de/dx for PBE (convert from df/dx using f = n e, as in your restricted code)
    deC_PBE_dna = (df_dna_PBE - e_C_PBE) * inv_n
    deC_PBE_dnb = (df_dnb_PBE - e_C_PBE) * inv_n
    deC_PBE_dsaa = df_dsaa_PBE * inv_n
    deC_PBE_dsbb = df_dsbb_PBE * inv_n
    deC_PBE_dsab = df_dsab_PBE * inv_n

    inv_na = 1 / alpha_density
    inv_nb = 1 / beta_density

    deC_a0_dna = (df_dna_a0 - e_C_a0) * inv_na
    deC_a0_dsaa = df_dsaa_a0 * inv_na

    deC_0b_dnb = (df_dnb_0b - e_C_0b) * inv_nb
    deC_0b_dsbb = df_dsbb_0b * inv_nb

    # \tilde{e}_c^alpha, \tilde{e}_c^beta (piecewise via max), plus their derivatives
    condA = e_C_PBE >= e_C_a0
    condB = e_C_PBE >= e_C_0b

    e_C_tilde_alpha = np.where(condA, e_C_PBE, e_C_a0)
    e_C_tilde_beta  = np.where(condB, e_C_PBE, e_C_0b)

    deC_tilde_alpha_dna  = np.where(condA, deC_PBE_dna,  deC_a0_dna)
    deC_tilde_alpha_dnb  = np.where(condA, deC_PBE_dnb,  0.0)
    deC_tilde_alpha_dsaa = np.where(condA, deC_PBE_dsaa, deC_a0_dsaa)
    deC_tilde_alpha_dsbb = np.where(condA, deC_PBE_dsbb, 0.0)
    deC_tilde_alpha_dsab = np.where(condA, deC_PBE_dsab, 0.0)

    deC_tilde_beta_dna  = np.where(condB, deC_PBE_dna,  0.0)
    deC_tilde_beta_dnb  = np.where(condB, deC_PBE_dnb,  deC_0b_dnb)
    deC_tilde_beta_dsaa = np.where(condB, deC_PBE_dsaa, 0.0)
    deC_tilde_beta_dsbb = np.where(condB, deC_PBE_dsbb, deC_0b_dsbb)
    deC_tilde_beta_dsab = np.where(condB, deC_PBE_dsab, 0.0)

    # weighted tilde: sum_sigma (n_sigma/n) * tilde_e_c^sigma
    numer_tilde = alpha_density * e_C_tilde_alpha + beta_density * e_C_tilde_beta
    e_C_tilde = numer_tilde * inv_n

    deC_tilde_dna = (e_C_tilde_alpha
                     + alpha_density * deC_tilde_alpha_dna
                     + beta_density  * deC_tilde_beta_dna
                     - e_C_tilde) * inv_n

    deC_tilde_dnb = (e_C_tilde_beta
                     + beta_density  * deC_tilde_beta_dnb
                     + alpha_density * deC_tilde_alpha_dnb
                     - e_C_tilde) * inv_n

    deC_tilde_dsaa = (alpha_density * deC_tilde_alpha_dsaa + beta_density * deC_tilde_beta_dsaa) * inv_n
    deC_tilde_dsbb = (alpha_density * deC_tilde_alpha_dsbb + beta_density * deC_tilde_beta_dsbb) * inv_n
    deC_tilde_dsab = (alpha_density * deC_tilde_alpha_dsab + beta_density * deC_tilde_beta_dsab) * inv_n

    # z = tau_W / tau = sigma / (8 n tau)
    z = sigma / (8 * tau * density)
    z_squared = z * z
    z_cubed = z_squared * z

    # zeta and |grad zeta| in sigma-language (TPSS uses xi = |∇zeta| / (2 (3π^2 n)^(1/3)))
    zeta = calculate_zeta(alpha_density, beta_density)
    inv_n2 = inv_n * inv_n

    dzeta_dna = 2 * beta_density * inv_n2
    dzeta_dnb = -2 * alpha_density * inv_n2

    one_minus = 1 - zeta
    one_plus  = 1 + zeta
    one_minus2 = one_minus * one_minus
    one_plus2  = one_plus * one_plus
    one_minus_z2 = 1 - zeta * zeta

    B = clean(one_minus2 * sigma_aa + one_plus2 * sigma_bb - 2 * one_minus_z2 * sigma_ab, floor=1e-52)
    sqrtB = np.sqrt(B)
    inv_sqrtB = 1 / sqrtB

    dB_dzeta = -2 * one_minus * sigma_aa + 2 * one_plus * sigma_bb + 4 * zeta * sigma_ab

    dB_dna = dB_dzeta * dzeta_dna
    dB_dnb = dB_dzeta * dzeta_dnb

    dB_dsaa = one_minus2
    dB_dsbb = one_plus2
    dB_dsab = -2 * one_minus_z2

    zeta_gradient = sqrtB * inv_n

    dzeta_grad_dna = (0.5 * inv_sqrtB * dB_dna) * inv_n - sqrtB * inv_n2
    dzeta_grad_dnb = (0.5 * inv_sqrtB * dB_dnb) * inv_n - sqrtB * inv_n2

    dzeta_grad_dsaa = (0.5 * inv_sqrtB * dB_dsaa) * inv_n
    dzeta_grad_dsbb = (0.5 * inv_sqrtB * dB_dsbb) * inv_n
    dzeta_grad_dsab = (0.5 * inv_sqrtB * dB_dsab) * inv_n

    inv_den_xi = 1 / (2 * np.cbrt(3 * np.pi ** 2 * density))
    xi = zeta_gradient * inv_den_xi

    dxi_dna = inv_den_xi * dzeta_grad_dna - (1 / 3) * xi * inv_n
    dxi_dnb = inv_den_xi * dzeta_grad_dnb - (1 / 3) * xi * inv_n

    dxi_dsaa = inv_den_xi * dzeta_grad_dsaa
    dxi_dsbb = inv_den_xi * dzeta_grad_dsbb
    dxi_dsab = inv_den_xi * dzeta_grad_dsab

    # C(zeta,xi) from TPSS
    C_0 = 0.53 + 0.87 * zeta ** 2 + 0.50 * zeta ** 4 + 2.26 * zeta ** 6
    dC0_dzeta = 1.74 * zeta + 2.0 * zeta ** 3 + 13.56 * zeta ** 5
    dC0_dna = dC0_dzeta * dzeta_dna
    dC0_dnb = dC0_dzeta * dzeta_dnb

    inv_m43_plus = 1 / (np.cbrt(one_plus) ** 4)
    inv_m43_minus = 1 / (np.cbrt(one_minus) ** 4)
    s = inv_m43_plus + inv_m43_minus

    dinv_m43_plus_dzeta  = -(4 / 3) * inv_m43_plus / one_plus
    dinv_m43_minus_dzeta = (4 / 3) * inv_m43_minus / one_minus
    ds_dzeta = dinv_m43_plus_dzeta + dinv_m43_minus_dzeta

    A = 0.5 * xi * xi * s

    dA_dna = xi * dxi_dna * s + 0.5 * xi * xi * ds_dzeta * dzeta_dna
    dA_dnb = xi * dxi_dnb * s + 0.5 * xi * xi * ds_dzeta * dzeta_dnb

    dA_dsaa = xi * dxi_dsaa * s
    dA_dsbb = xi * dxi_dsbb * s
    dA_dsab = xi * dxi_dsab * s

    inv_1pA = 1 / (1 + A)
    inv_1pA4 = inv_1pA ** 4

    C = C_0 * inv_1pA4

    dC_dna = inv_1pA4 * (dC0_dna - 4 * C_0 * inv_1pA * dA_dna)
    dC_dnb = inv_1pA4 * (dC0_dnb - 4 * C_0 * inv_1pA * dA_dnb)

    dC_dsaa = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsaa)
    dC_dsbb = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsbb)
    dC_dsab = inv_1pA4 * (-4 * C_0 * inv_1pA * dA_dsab)

    # z-derivatives wrt the *unrestricted* variables
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
    deC_rev_dna = (A_tpss * deC_PBE_dna
                   + e_C_PBE * (z_squared * dC_dna + C * dz2_dna)
                   - (1 + C) * (e_C_tilde * dz2_dna + z_squared * deC_tilde_dna)
                   - z_squared * e_C_tilde * dC_dna)

    deC_rev_dnb = (A_tpss * deC_PBE_dnb
                   + e_C_PBE * (z_squared * dC_dnb + C * dz2_dnb)
                   - (1 + C) * (e_C_tilde * dz2_dnb + z_squared * deC_tilde_dnb)
                   - z_squared * e_C_tilde * dC_dnb)

    deC_rev_dsaa = (A_tpss * deC_PBE_dsaa
                    + e_C_PBE * (z_squared * dC_dsaa + C * dz2_dsaa)
                    - (1 + C) * (e_C_tilde * dz2_dsaa + z_squared * deC_tilde_dsaa)
                    - z_squared * e_C_tilde * dC_dsaa)

    deC_rev_dsbb = (A_tpss * deC_PBE_dsbb
                    + e_C_PBE * (z_squared * dC_dsbb + C * dz2_dsbb)
                    - (1 + C) * (e_C_tilde * dz2_dsbb + z_squared * deC_tilde_dsbb)
                    - z_squared * e_C_tilde * dC_dsbb)

    deC_rev_dsab = (A_tpss * deC_PBE_dsab
                    + e_C_PBE * (z_squared * dC_dsab + C * dz2_dsab)
                    - (1 + C) * (e_C_tilde * dz2_dsab + z_squared * deC_tilde_dsab)
                    - z_squared * e_C_tilde * dC_dsab)

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

    # convert to df/dx for f = n e_C (same pattern as your restricted function)
    df_dn_alpha = e_C + density * deC_dna
    df_dn_beta  = e_C + density * deC_dnb

    df_ds_aa = density * deC_dsaa
    df_ds_bb = density * deC_dsbb
    df_ds_ab = density * deC_dsab

    df_dt_alpha = density * deC_dta
    df_dt_beta  = density * deC_dtb

    return df_dn_alpha, df_dn_beta, df_ds_aa, df_ds_bb, df_ds_ab, df_dt_alpha, df_dt_beta, e_C





    
def calculate_TPSS_correlation_potential(density, sigma, tau, calculation):

    C, d = 0.53, 2.8

    z = sigma / (8 * tau * density)

    z_squared = z * z
    z_cubed = z * z * z
    A = 1 + C * z_squared

    zeros = np.zeros_like(density)

    df_dn_PBE, df_ds_PBE, e_C_PBE = calculate_PBE_correlation_potential(density, sigma, calculation)
    df_dna_one, _, df_dsaa_one, _, _, e_C_PBE_one_spin = calculate_UPBE_correlation_potential(density / 2, zeros, density / 2, sigma / 4, zeros, zeros)

    e_C_tilde = np.maximum(e_C_PBE, e_C_PBE_one_spin)

    e_C_rev = e_C_PBE * (1 + C * z_squared) - (1 + C) * z_squared * e_C_tilde
    e_C = e_C_rev * (1 + d * e_C_rev * z_cubed)

    inv_n = 1 / density
    deC_PBE_dn = (df_dn_PBE - e_C_PBE) * inv_n
    deC_PBE_ds = df_ds_PBE * inv_n

    deC_one_dn = (df_dna_one - e_C_PBE_one_spin) * inv_n 
    deC_one_ds = (1 / 2) * df_dsaa_one * inv_n           

    deC_tilde_dn = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_dn, deC_one_dn)
    deC_tilde_ds = np.where(e_C_PBE >= e_C_PBE_one_spin, deC_PBE_ds, deC_one_ds)

    dz_dn, dz_ds, dz_dt = -z * inv_n, 1 / (8 * tau * density), -z / tau
    dz2_dn, dz2_ds, dz2_dt = 2 * z * dz_dn, 2 * z * dz_ds, 2 * z * dz_dt
    dz3_dn, dz3_ds, dz3_dt = 3 * z_squared * dz_dn, 3 * z_squared * dz_ds, 3 * z_squared * dz_dt

    deC_rev_dn = A * deC_PBE_dn +  C * e_C_PBE * dz2_dn - (1 + C) * (e_C_tilde * dz2_dn + z_squared * deC_tilde_dn)
    deC_rev_ds = A * deC_PBE_ds +  C * e_C_PBE * dz2_ds - (1 + C) * (e_C_tilde * dz2_ds + z_squared * deC_tilde_ds)
    deC_rev_dt = (C * e_C_PBE - (1 + C) * e_C_tilde) * dz2_dt 

    prefactor = 1 + 2 * d * e_C_rev * z_cubed

    deC_dn = deC_rev_dn * prefactor + d * e_C_rev * e_C_rev * dz3_dn
    deC_ds = deC_rev_ds * prefactor + d * e_C_rev * e_C_rev * dz3_ds
    deC_dt = deC_rev_dt * prefactor + d * e_C_rev * e_C_rev * dz3_dt

    df_dn = e_C + density * deC_dn
    df_ds = density * deC_ds
    df_dt = density * deC_dt

    return df_dn, df_ds, df_dt, e_C





