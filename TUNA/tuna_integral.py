import numpy as np
from scipy.special import erf

def special_function(x): 
    
    """

    Calculatess boys function.

    Args:
        x (array or float): Generic array

    Returns:
        boys_function (array): Boys function evaluated on array or float

    """

    boys_function = np.where(x == 0, 1, (0.5 * (np.pi / (x + 1e-17)) ** 0.5) * erf((x + 1e-17) ** 0.5))

    return boys_function


def evaluate_integrals(orbitals, charges, atomic_coords, centre_of_mass, two_electron_ints=True):

    """

    Evaluates all one and two-electron integrals.

    Args:
        orbitals (array): Atomic orbitals
        charges (array): Nuclear charges
        atomic_coords (array): Coordinates for nuclei
        centre_of_mass (float): Centre of mass for dipole calculation
        two_electron_ints (bool, optional): Checks whether to calculate two-electron integrals

    Returns:
        S (array): Overlap integral matrix
        T (array): Kinetic energy integral matrix
        V_NE (array): Nuclear-electron integral matrix
        D (array): Dipole integral matrix
        ERI_AO (array): Electron repulsion integral matrix

    """

    n_basis = len(orbitals)

    # Initialises the integral arrays
    S = np.zeros([n_basis, n_basis])
    T = np.zeros([n_basis, n_basis])
    V_NE = np.zeros([n_basis, n_basis])
    D = np.zeros([n_basis, n_basis])
    ERI_AO = np.zeros([n_basis, n_basis, n_basis, n_basis])

    for i in range(n_basis):
        for j in range(i, n_basis):  

            alphas_m = np.array([pg.alpha for pg in orbitals[i]])
            alphas_n = np.array([pg.alpha for pg in orbitals[j]])

            coeffs_m = np.array([pg.coeff for pg in orbitals[i]])
            coeffs_n = np.array([pg.coeff for pg in orbitals[j]])

            R_m = np.array([pg.coordinates for pg in orbitals[i]])
            R_n = np.array([pg.coordinates for pg in orbitals[j]])

            sum_mn = alphas_m[:, np.newaxis] + alphas_n
            product_mn = alphas_m[:, np.newaxis] * alphas_n
            coeffproduct_mn = coeffs_m[:, np.newaxis] * coeffs_n
            R_mn = np.linalg.norm(R_m[:, np.newaxis] - R_n, axis=2)

            R_m_com = R_m - np.array([0, 0, centre_of_mass])
            R_n_com = R_n - np.array([0, 0, centre_of_mass])
            
            alpha_m_R_m = np.einsum("i, ij->ij", alphas_m, R_m, optimize=True)
            alpha_n_R_n = np.einsum("i, ij->ij", alphas_n, R_n, optimize=True)

            alpha_m_R_m_com = np.einsum("i, ij->ij", alphas_m, R_m_com, optimize=True)
            alpha_n_R_n_com = np.einsum("i, ij->ij", alphas_n, R_n_com, optimize=True)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)

            Rk = np.einsum("ijk,ij->ij",(alpha_m_R_m[:, np.newaxis] + alpha_n_R_n), 1 / sum_mn, optimize=True)
            Rk_dipole = np.einsum("ijk,ij->ij", alpha_m_R_m_com[:, np.newaxis] + alpha_n_R_n_com, 1 / sum_mn, optimize=True)

            # Adds onto the integral arrays using tensor contraction
            S[i,j] = np.einsum("mn->", OM_mn, optimize=True)
            T[i,j] = np.einsum("mn,mn,mn->", OM_mn, (product_mn / sum_mn), (3 - (2 * product_mn * R_mn**2) / sum_mn), optimize=True)
            D[i,j] += np.einsum("mn,mn->", OM_mn, Rk_dipole, optimize=True)

            for atom in range(len(charges)):

                dfunc_to_atom_mn = (Rk - atomic_coords[atom][2]) ** 2

                V_NE[i,j] += -charges[atom] * np.einsum("mn,mn,mn->", OM_mn, special_function(sum_mn * dfunc_to_atom_mn), 2 * np.sqrt(sum_mn / np.pi), optimize=True)
            
            # Uses symmetry to speed up calculations
            S[j,i] = S[i,j]
            D[j,i] = D[i,j]
            T[j,i] = T[i,j]
            V_NE[j,i] = V_NE[i,j] 
                
            if two_electron_ints:

                for k in range(i, n_basis):
                    for l in range(k, n_basis): 

                        
                        alphas_o = np.array([pg.alpha for pg in orbitals[k]])
                        alphas_p = np.array([pg.alpha for pg in orbitals[l]])

                        coeffs_o = np.array([pg.coeff for pg in orbitals[k]])
                        coeffs_p = np.array([pg.coeff for pg in orbitals[l]])

                        R_o = np.array([pg.coordinates for pg in orbitals[k]])
                        R_p = np.array([pg.coordinates for pg in orbitals[l]])

                        sum_op = alphas_o[:, np.newaxis] + alphas_p
                        product_op = alphas_o[:, np.newaxis] * alphas_p
                        coeffproduct_op = coeffs_o[:, np.newaxis] * coeffs_p
                        R_op = np.linalg.norm(R_o[:, np.newaxis] - R_p, axis=2)

                        alpha_o_R_o = np.einsum("i, ij->ij", alphas_o, R_o, optimize=True)
                        alpha_p_R_p = np.einsum("i, ij->ij", alphas_p, R_p, optimize=True)

                        Rl = np.einsum("ijk,ij->ij",(alpha_o_R_o[:, np.newaxis] + alpha_p_R_p), 1 / sum_op, optimize=True)

                        OM_op = coeffproduct_op * (4 * product_op / sum_op ** 2) ** (3 / 4) * np.exp(-(product_op / sum_op) * R_op ** 2)
                        
                        prod_over_sum = np.einsum("mn,op,mnop->mnop",sum_mn, sum_op, 1 / (sum_mn[:, :, np.newaxis, np.newaxis] + sum_op[np.newaxis, np.newaxis, :, :]), optimize=True)

                        RkRl = (Rk[:, :, np.newaxis, np.newaxis] - Rl[np.newaxis, np.newaxis, :, :]) ** 2

                        input_function = np.einsum("ijkl,ijkl->ijkl", prod_over_sum, RkRl, optimize=True)

                        ERI_AO[i,j,k,l] = 2 / np.sqrt(np.pi) * np.einsum("mnop,mnop,mn,op->", np.sqrt(prod_over_sum), special_function(input_function), OM_mn, OM_op, optimize=True)
                    
                        # Uses permutational symmetry to save computational time

                        ERI_AO[j, i, l, k] = ERI_AO[i,j,k,l]
                        ERI_AO[j, i, k, l] = ERI_AO[i,j,k,l]
                        ERI_AO[i, j, l, k] = ERI_AO[i,j,k,l]
                        ERI_AO[k, l, i, j] = ERI_AO[i,j,k,l]
                        ERI_AO[l, k, j, i] = ERI_AO[i,j,k,l]
                        ERI_AO[l, k, i, j] = ERI_AO[i,j,k,l]
                        ERI_AO[k, l, j, i] = ERI_AO[i,j,k,l]


    return S, T, V_NE, D, ERI_AO





def evaluate_dipole_integrals(orbitals, centre_of_mass):

    """

    Evaluates dipole integrals.

    Args:
        orbitals (array): Atomic orbitals
        centre_of_mass (float): Centre of mass for dipole calculation

    Returns:
        D (array): Dipole intergal matrix

    """

    n_basis = len(orbitals)
    D = np.zeros([n_basis, n_basis])


    for i in range(n_basis):
        for j in range(i, n_basis):  

            alphas_m = np.array([pg.alpha for pg in orbitals[i]])
            alphas_n = np.array([pg.alpha for pg in orbitals[j]])

            coeffs_m = np.array([pg.coeff for pg in orbitals[i]])
            coeffs_n = np.array([pg.coeff for pg in orbitals[j]])

            R_m = np.array([pg.coordinates for pg in orbitals[i]])
            R_n = np.array([pg.coordinates for pg in orbitals[j]])

            sum_mn = alphas_m[:, np.newaxis] + alphas_n
            product_mn = alphas_m[:, np.newaxis] * alphas_n
            coeffproduct_mn = coeffs_m[:, np.newaxis] * coeffs_n
            R_mn = np.linalg.norm(R_m[:, np.newaxis] - R_n, axis=2)

            R_m_com = R_m - np.array([0, 0, centre_of_mass])
            R_n_com = R_n - np.array([0, 0, centre_of_mass])

            alpha_m_R_m_com = np.einsum("i, ij->ij", alphas_m, R_m_com, optimize=True)
            alpha_n_R_n_com = np.einsum("i, ij->ij", alphas_n, R_n_com, optimize=True)
                    
            OM_mn = coeffproduct_mn * (4 * product_mn / sum_mn ** 2) ** (3 / 4) * np.exp(-(product_mn / sum_mn) * R_mn ** 2)

            Rk_dipole = np.einsum("ijk,ij->ij", alpha_m_R_m_com[:, np.newaxis] + alpha_n_R_n_com, 1 / sum_mn, optimize=True)

            D[i,j] += np.einsum("mn,mn->", OM_mn, Rk_dipole, optimize=True)

    return D