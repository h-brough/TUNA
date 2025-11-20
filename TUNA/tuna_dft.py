import numpy as np
from tuna_util import *
from scipy.integrate import lebedev_rule





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

    log(f"\n Integration grid has {n_radial} radial and {points.shape[2]} angular points, a Lebedev order of {Lebedev_order}.", calculation, 1, silent=silent)
    log(f" In total there are {n_radial * points.shape[1]} grid points, {int(n_radial * points.shape[1] / 2)} per atom.", calculation, 1, silent=silent)

    log("\n Building guess density on grid...   ", calculation, 1, end="", silent=silent); sys.stdout.flush()


    atomic_orbitals = construct_basis_functions_on_grid_new(basis_functions, points)

    density = construct_density_on_grid(P_guess, atomic_orbitals)

    log("[Done]", calculation, 1, silent=silent)

    n_electrons_DFT = integrate_on_grid(density, weights)

    log(f"\n Integral of the guess density: {n_electrons_DFT:13.10f}\n", calculation, 1, silent=silent)



    if np.abs(n_electrons_DFT - n_electrons) > 0.00001:

        warning(" Integral of density is far from the number of electrons! Be careful with your results.")

        if np.abs(n_electrons_DFT - n_electrons) > 0.5:

            error("Integral for the density is completely wrong!")
    


    return atomic_orbitals, weights





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
    R_A = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R_B = np.sqrt(X ** 2 + Y ** 2 + (Z - bond_length) ** 2)

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







def construct_basis_functions_on_grid_new(basis_functions, points):
    
    if len(points) == 2: 
            
        X, Z = points

        Y = np.full_like(X, 0)

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






def construct_density_on_grid(P, atomic_orbitals):
    
    """
    
    Constructs the electron density on the grid using the atomic orbitals, then cleans it up.

    Args:
        P (array): One-particle reduced density matrix
        atomic_orbitals (array): Atomic orbitals on molecular grid
    
    Returns:
        density (array): Electron density on molecular grid
    
    """


    density = np.einsum("ij,ikl,jkl->kl", P, atomic_orbitals, atomic_orbitals, optimize=True)

    density = clean_density(density)

    return density






def clean_density(density):

    # Makes sure there are no zero or negative values in the electron density
    
    density = np.maximum(density, 1e-16)

    return density




def clean_density_matrix(P, S, n_electrons):

    # Forces the trace of the density matrix to be correct
    
    P *= n_electrons / np.trace(P @ S)

    return P








def calculate_overlap_matrix(atomic_orbitals, weights):

    S = np.zeros((len(atomic_orbitals), len(atomic_orbitals)))

    for i in range(len(atomic_orbitals)):
        for j in range(i, len(atomic_orbitals)):

            S[i,j] = S[j,i] = integrate_on_grid(atomic_orbitals[i] * atomic_orbitals[j], weights)


    return S







def calculate_exchange_energy(density, weights, exchange_functional, calculation):

    e_X = exchange_functional(density, calculation)

    E_X = integrate_on_grid(e_X * density, weights)

    return E_X, e_X




def calculate_correlation_energy(density, weights, correlation_functional, calculation):

    e_C = correlation_functional(density, calculation)

    E_C = integrate_on_grid(e_C * density, weights)

    return E_C





def calculate_Slater_energy_density(density, calculation):

    # Slater exchange differs by LDA exchange by a factor of 3/2 alpha

    alpha = calculation.X_alpha


    e_X = (3 / 2 * alpha) * -3 / 4 * np.cbrt(3 / np.pi * density)

    return e_X





def calculate_Slater_potential(density, calculation):

    alpha = calculation.X_alpha

    v_X = - (3 / 2 * alpha) * np.cbrt(3 / np.pi * density)

    return v_X




def calculate_unrestricted_Slater_energy_density(alpha_density, beta_density, calculation):

    # Slater exchange differs by LDA exchange by a factor of 3/2 alpha

    alpha = calculation.X_alpha

    e_X_alpha = (3 / 2 * alpha) * -3 / 4 * (3 / np.pi) ** (1 / 3) * alpha_density ** (1 / 3) 
    e_X_beta = (3 / 2 * alpha) * -3 / 4 * (3 / np.pi) ** (1 / 3) * beta_density ** (1 / 3) 

    return e_X_alpha, e_X_beta



def calculate_unrestricted_Slater_potential(alpha_density, beta_density, calculation):

    # Slater exchange differs by LDA exchange by a factor of 3/2 alpha

    alpha = calculation.X_alpha

    V_X_alpha = - (3 / 2 * alpha) * (3 / np.pi) ** (1 / 3) * alpha_density ** (1 / 3) 
    V_X_beta = - (3 / 2 * alpha) * (3 / np.pi) ** (1 / 3) * beta_density ** (1 / 3) 

    return V_X_alpha, V_X_beta










def calculate_VWN_energy_density(density, x_0, b, c):


    A = 0.0621814 / 2

    r_s = np.cbrt(3 / (4 * np.pi * density))

    x = np.sqrt(r_s)
    Q = np.sqrt(4 * c - b ** 2)

    def X(x):

        return x ** 2 + b * x + c


    X_term = X(x)
    X_0_term = X(x_0)

    log_term_1 = np.log(x ** 2 / X_term)
    log_term_2 = np.log((x - x_0) ** 2 / X_term)

    atan_term = np.arctan(Q / (2 * x + b))

    c_1 = - b * x_0 / X_0_term
    c_2 = 2 * b * (c - x_0 ** 2) / (Q * X_0_term)

    e_C = A * (log_term_1 + c_1 * log_term_2 + c_2 * atan_term)



    return e_C





def spin_polarisation_function(zeta):

    f_zeta = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)

    return f_zeta





def phi(zeta):

    phi_zeta = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2

    return phi_zeta






def calculate_PBE_exchange(density, calculation, density_gradient):

    e_X_Slater = calculate_Slater_energy_density(density, calculation)


    r = np.cbrt(3 / (4 * np.pi * density))
    k_F = np.cbrt(3 * np.pi ** 2 * density)

    k_s = np.sqrt(4 * k_F / np.pi)

    beta = 0.066725

    mu = beta * np.pi ** 2 / 3
    kappa = 0.804

    F_X = 1 + kappa - kappa / (1 + mu * s ** 2 / kappa)

    e_X = e_X_Slater * F_X

    return e_X







def calculate_VWN3_energy_density(density, calculation):

    x_0 = -0.409286 
    b = 13.0720
    c = 42.7198

    return calculate_VWN_energy_density(density, x_0, b, c)



def calculate_VWN5_energy_density(density, calculation):

    x_0 = -0.10498
    b = 3.72744
    c = 12.9352

    return calculate_VWN_energy_density(density, x_0, b, c)




def calculate_RPW_energy_density(density, calculation):

    e_C = calculate_PW_correlation_density(density, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    
    return e_C




def calculate_PW_correlation_density(density, A, alpha_1, beta_1, beta_2, beta_3, beta_4, P):

    r_s = np.cbrt(3 / (4 * np.pi * density))

    B = beta_1 * np.sqrt(r_s) + beta_2 * r_s + beta_3 * np.sqrt(r_s) ** 3 + beta_4 * r_s ** (P + 1)

    K = 1 / (2 * A * B)

    e_C = -2 * A * (1 + alpha_1 * r_s) * np.log(1 + K)

    return e_C




def calculate_PW_potential(density, A, alpha_1, beta_1, beta_2, beta_3, beta_4, P):

    r_s = np.cbrt(3 / (4 * np.pi * density))

    B = beta_1 * np.sqrt(r_s) + beta_2 * r_s + beta_3 * np.sqrt(r_s) ** 3 + beta_4 * r_s ** (P + 1)

    K = 1 / (2 * A * B)

    e_C = -2 * A * (1 + alpha_1 * r_s) * np.log(1 + K)

    dB_dr = (1 / 2) * beta_1 * 1 / np.sqrt(r_s) + beta_2 + (3 / 2) * beta_3 * np.sqrt(r_s) + (P + 1) * beta_4 * r_s ** P

    de_dr = -2 * A * alpha_1 * np.log(1 + K) + (1 + alpha_1 * r_s) * dB_dr / (B ** 2 * (1 + K))

    v_C = e_C - r_s / 3 * de_dr

    return v_C






def calculate_VWN3_correlation_potential(density, calculation):

    v_C = calculate_numerical_functional_derivative(density, calculate_VWN3_energy_density, calculation)

    return v_C


def calculate_VWN5_correlation_potential(density, calculation):

    v_C = calculate_numerical_functional_derivative(density, calculate_VWN5_energy_density, calculation)

    return v_C


def calculate_PW_correlation_potential(density, calculation):

    v_C = calculate_PW_potential(density, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    return v_C


def calculate_Slater_exchange_potential(density, calculation):

    v_X = calculate_Slater_potential(density, calculation)

    return v_X





def calculate_XC_potential(density, functional, calculation):

    v_XC = functional(density, calculation)

    return v_XC






def calculate_numerical_functional_derivative(density, e_XC_functional, calculation):

    dp = 0.0001 * density

    v_XC = e_XC_functional(density, calculation) + density * (e_XC_functional(density + dp, calculation) - e_XC_functional(density - dp, calculation)) / (2 * dp)

    return v_XC






def calculate_K_matrix(v_XC, atomic_orbitals, weights):

    K = np.einsum('ikl,kl,jkl->ij', atomic_orbitals, v_XC * weights, atomic_orbitals, optimize=True)

    return K








def calculate_zeta(alpha_density, beta_density):

    zeta = (alpha_density - beta_density) / (alpha_density + beta_density)

    return zeta






exchange_functionals = {

    "S": calculate_Slater_energy_density,

}




exchange_potentials = {

    "S": calculate_Slater_exchange_potential,

}



correlation_functionals = {

    "VWN3": calculate_VWN3_energy_density,
    "VWN5": calculate_VWN5_energy_density,
    "PW": calculate_RPW_energy_density,

}


correlation_potentials = {

    "VWN3": calculate_VWN3_correlation_potential,
    "VWN5": calculate_VWN5_correlation_potential,
    "PW": calculate_PW_correlation_potential,

}







def calculate_spin_polarisation(P_alpha, P_beta, atomic_orbitals, weights, P):
    
    density_alpha = construct_density_on_grid(P_alpha, atomic_orbitals)
    density_beta = construct_density_on_grid(P_beta, atomic_orbitals)

    density = density_alpha + density_beta

    zeta = (density_alpha - density_beta) / density

    density = construct_density_on_grid(P, atomic_orbitals)

    e_C_0 = calculate_PW_correlation_density(density, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)

    e_C_1 = calculate_PW_correlation_density(density, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)

    alpha = -calculate_PW_correlation_density(density, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

    f_zeta = spin_polarisation_function(zeta)

    e_C = e_C_0 + alpha * f_zeta / 1.709921 * (1 - zeta ** 4) + (e_C_1 - e_C_0) * f_zeta * zeta ** 4


    E_C = integrate_on_grid(e_C * density, weights)


    return E_C

