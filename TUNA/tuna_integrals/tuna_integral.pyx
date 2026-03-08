# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
import numpy as np
from numpy import ndarray
cimport numpy as np
from libc.math cimport exp, pow, sqrt
from libc.stdlib cimport malloc, free
from scipy.special.cython_special cimport hyp1f1


cdef double PI = 3.141592653589793238462643383279


cdef class Basis:

    """

    Defines basis functions from primitive Gaussians.

    """

    cdef:
        double *origin      # Coordinates of the basis function
        long   *shell       # Angular momentum of the basis function
        long    num_exps    # Number of primitive Gaussians
        double *exps        # Exponents of primitive Gaussians
        double *coefs       # Coefficients of primitive Gaussians
        double *norm        # Normalisation constants of primitive Gaussians

    property origin:

        def __get__(self):

            cdef double[::1] view = <double[:3]> self.origin

            return np.asarray(view)

    property shell:

        def __get__(self):

            cdef long[::1] view = <long[:3]> self.shell

            return np.asarray(view)

    property num_exps:

        def __get__(self):

            cdef long view = <long> self.num_exps

            return view

    property exps:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.exps

            return np.asarray(view)

    property coefs:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.coefs

            return np.asarray(view)

    property norm:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.norm

            return np.asarray(view)



    def __cinit__(self, origin, shell, num_exps, exps, coefs):

        """
        
        Initialises Basis class to interface with Python.
        
        """

        self.origin = <double*>malloc(3 * sizeof(double))
        self.shell  = <long*>malloc(3 * sizeof(long))
        self.num_exps = num_exps
        self.exps = <double*>malloc(num_exps * sizeof(double))
        self.coefs = <double*>malloc(num_exps * sizeof(double))
        self.norm = <double*>malloc(num_exps * sizeof(double))

        for i in range(3):

            self.origin[i] = origin[i]
            self.shell[i] = shell[i]

        for i in range(num_exps):

            self.exps[i] = exps[i]
            self.coefs[i] = coefs[i]
            self.norm[i] = 0.0

        self.normalize()



    def normalize(self):

        """
        Normalises the primitives, then normalises the contracted functions.

        """

        l = self.shell[0]
        m = self.shell[1]
        n = self.shell[2]

        L = l + m + n

        for i in range(self.num_exps):

            self.norm[i] = sqrt(pow(2, 2 * L + 1.5) * pow(self.exps[i], L + 1.5) / double_fact(2 * l - 1) / double_fact(2 * m - 1) / double_fact(2 * n - 1) / pow(PI, 1.5))

        # Normalises the primitive Gaussians

        prefactor = pow(PI, 1.5) * double_fact(2 * l - 1) * double_fact(2 * m - 1) * double_fact(2 * n - 1) / pow(2.0, L)

        N = 0.0

        for i in range(self.num_exps):

            for j in range(self.num_exps):

                N += self.norm[i] * self.norm[j] * self.coefs[i] * self.coefs[j] / pow(self.exps[i] + self.exps[j], L + 1.5)
                
        # Normalises the contracted basis functions

        N = 1 / sqrt(prefactor * N)

        for i in range(self.num_exps):

            self.coefs[i] *= N


    def __dealloc__(self):

        if self.origin != NULL: free(self.origin)
        if self.shell != NULL: free(self.shell)
        if self.exps != NULL: free(self.exps)
        if self.coefs != NULL: free(self.coefs)
        if self.norm != NULL: free(self.norm)








cdef inline double double_fact(int n):

    """
    
    Calculates a dobule factorial.

    Args:
        n (int): Integer input
    
    Returns:
        result (float): double_factled integer
    
    """

    cdef double result = 1.0

    if n <= 0:

        return 1.0

    while n > 1:

        result *= n
        n -= 2

    return result










cpdef double calculate_dipole_integral(object bf_1, object bf_2, dipole_origin, str direction):

    """
    
    Calculates a dipole integral between basis functions, <1| r - dipole_origin |2>.
    
    Args:
        bf_1 (Basis): First basis function
        bf_2 (Basis): Second basis function
        dipole_origin (array): Coordinates of electric field gauge origin
        direction (str): Either "x", "y" or "z"

    Returns:
        integral (float): Dipole integral between contracted Gaussians
        
    """

    cdef double integral = 0.0

    for ia, ca in enumerate(bf_1.coefs):

        for ib, cb in enumerate(bf_2.coefs):

            # Applies coefficients and norms to integrals between primitive Gaussians

            integral += bf_1.norm[ia] * bf_2.norm[ib] * ca * cb * dipole(bf_1.exps[ia], bf_1.shell, bf_1.origin, bf_2.exps[ib], bf_2.shell, bf_2.origin, dipole_origin, direction)


    return integral









def dipole(exponent_1: float, angmom_1: ndarray, centre_1: ndarray, exponent_2: float, angmom_2: ndarray, centre_2: ndarray, dipole_origin: ndarray, direction: str) -> float:
   
    """
    
    Calculates a Cartesian component of a dipole integral between primitive Gaussians, <1| r - dipole_origin |2>.
    
    Args:
        exponent_1 (float): Gaussian exponent on first centre
        angmom_1 (ndarray): Angular momenta of primitive Gaussian of first centre
        centre_1 (array): Coordinates of first centre
        exponent_2 (float): Gaussian exponent on second centre
        angmom_2 (ndarray): Angular momenta of primitive Gaussian of second centre
        centre_2 (array): Coordinates of second centre
        dipole_origin (array): Electric field origin
        direction (str): Either "x", "y" or "z"
    
    Returns:
        integral (float): Dipole integral between primitive Gaussians

    """

    l_1, m_1, n_1 = angmom_1
    l_2, m_2, n_2 = angmom_2

    R_12 = centre_1 - centre_2

    exponent_sum = exponent_1 + exponent_2
    prefactor = pow(PI / exponent_sum, 1.5)

    P = (exponent_1 * centre_1 + exponent_2 * centre_2) / exponent_sum - dipole_origin

    # Calculates the overlap integrals between primitive Gaussians
    
    Sx = hermite_coeff(l_1, l_2, 0, R_12[0], exponent_1, exponent_2)
    Sy = hermite_coeff(m_1, m_2, 0, R_12[1], exponent_1, exponent_2)
    Sz = hermite_coeff(n_1, n_2, 0, R_12[2], exponent_1, exponent_2)

    # Only calculates one component of the dipole integrals

    if direction.lower() == "x":

        Dx = hermite_coeff(l_1, l_2, 1, R_12[0], exponent_1, exponent_2) + P[0] * Sx

        integral = prefactor * Dx * Sy * Sz

    elif direction.lower() == "y":

        Dy = hermite_coeff(m_1, m_2, 1, R_12[1], exponent_1, exponent_2) + P[1] * Sy

        integral = prefactor * Sx * Dy * Sz

    elif direction.lower() == "z":

        Dz = hermite_coeff(n_1, n_2, 1, R_12[2], exponent_1, exponent_2) + P[2] * Sz

        integral = prefactor * Sx * Sy * Dz


    return integral









cpdef double calculate_nuclear_electron_integral(object bf_1, object bf_2, double[:] nucleus):

    """
    
    Calculates a nuclear-electron integral between basis functions, <1| 1/(r-R_N) |2>.
    
    Args:
        bf_1 (Basis): First basis function
        bf_2 (Basis): Second basis function
        nucleus (array): Coordinates of nucleus

    Returns:
        integral (float): Nuclear-electron integral between contracted Gaussians
        
    """

    cdef double integral = 0.0

    for ia, ca in enumerate(bf_1.coefs):

        for ib, cb in enumerate(bf_2.coefs):

            # Applies coefficients and norms to integrals between primitive Gaussians

            integral += bf_1.norm[ia] * bf_2.norm[ib] * ca * cb * nuclear_attraction(bf_1.exps[ia], bf_1.shell, bf_1.origin, bf_2.exps[ib], bf_2.shell, bf_2.origin, nucleus)


    return integral










def nuclear_attraction(exponent_1: float, angmom_1: ndarray, centre_1: ndarray, exponent_2: float, angmom_2: ndarray, centre_2: ndarray, nucleus: ndarray) -> float:

    """
    
    Calculates a nuclear-electron integral between primitive Gaussians, <1| 1/(r-R_N) |2>.
    
    Args:
        exponent_1 (float): Gaussian exponent on first centre
        angmom_1 (ndarray): Angular momenta of primitive Gaussian of first centre
        centre_1 (array): Coordinates of first centre
        exponent_2 (float): Gaussian exponent on second centre
        angmom_2 (ndarray): Angular momenta of primitive Gaussian of second centre
        centre_2 (array): Coordinates of second centre
        nucleus (array): Coordinates of nucleus
    
    Returns:
        integral (float): Nuclear-electron integral between primitive Gaussian

    """

    l_1, m_1, n_1 = angmom_1
    l_2, m_2, n_2 = angmom_2

    cdef double exponent_sum = exponent_1 + exponent_2
    cdef double PCz = (exponent_1 * centre_1[2] + exponent_2 * centre_2[2]) / exponent_sum - nucleus[2]
    cdef double Rz_12 = centre_1[2] - centre_2[2]

    cdef int Vmax = n_1 + n_2
    cdef int Nmax = l_1 + l_2 + m_1 + m_2 + n_1 + n_2
    cdef int stride = Nmax + 1

    cdef double *boys_tab = <double*>malloc((Nmax + 1) * sizeof(double))
    cdef double *pow_tab = <double*>malloc((Nmax + 1) * sizeof(double))
    cdef double *Rz = <double*>malloc((Vmax + 1) * (Nmax + 1) * sizeof(double))

    cdef double integral = 0.0
    cdef double Ex, Ey, Ez
    cdef int t, u, v, nxy

    fill_boys_table(Nmax, exponent_sum * PCz * PCz, boys_tab)
    fill_pow_table(Nmax, exponent_sum, pow_tab)
    fill_Rz_linear_table(Vmax, Nmax, PCz, boys_tab, pow_tab, Rz)


    for t in range(0, l_1 + l_2 + 1, 2):

        Ex = hermite_coeff(l_1, l_2, t, 0.0, exponent_1, exponent_2) * odd_double_double_fact_from_even(t)

        for u in range(0, m_1 + m_2 + 1, 2):

            Ey = hermite_coeff(m_1, m_2, u, 0.0, exponent_1, exponent_2) * odd_double_double_fact_from_even(u)

            for v in range(n_1 + n_2 + 1):

                Ez = hermite_coeff(n_1, n_2, v, Rz_12, exponent_1, exponent_2)

                integral += Ex * Ey * Ez * Rz[v * stride + (t + u) // 2]


    free(boys_tab)
    free(pow_tab)
    free(Rz)

    integral *= 2.0 * PI / exponent_sum

    return integral









cpdef double calculate_kinetic_integral(object bf_1, object bf_2):

    """
    
    Calculates a kinetic integral between basis functions, <1| d^2/dx^2 + d^2/dy^2 + d^2/dz^2 |2>.
    
    Args:
        bf_1 (Basis): First basis function
        bf_2 (Basis): Second basis function

    Returns:
        integral (float): Kinetic integral between contracted Gaussians
        
    """

    cdef double integral = 0.0

    for ia, ca in enumerate(bf_1.coefs):

        for ib, cb in enumerate(bf_2.coefs):

            # Applies coefficients and norms to integrals between primitive Gaussians

            integral += bf_1.norm[ia] * bf_2.norm[ib] * ca * cb * kinetic(bf_1.exps[ia], bf_1.shell, bf_1.origin, bf_2.exps[ib], bf_2.shell, bf_2.origin)


    return integral









def kinetic(exponent_1: float, angmom_1: ndarray, centre_1: ndarray, exponent_2: float, angmom_2: ndarray, centre_2: ndarray) -> float:

    """
    
    Calculates a kinetic integral between primitive Gaussians, -1/2 * <1| d^2/dx^2 + d^2/dy^2 + d^2/dz^2 |2>.
    
    Args:
        exponent_1 (float): Gaussian exponent on first centre
        angmom_1 (ndarray): Angular momenta of primitive Gaussian of first centre
        centre_1 (array): Coordinates of first centre
        exponent_2 (float): Gaussian exponent on second centre
        angmom_2 (ndarray): Angular momenta of primitive Gaussian of second centre
        centre_2 (array): Coordinates of second centre
    
    Returns:
        integral (float): Kinetic integral between primitive Gaussian

    """

    l_1, m_1, n_1 = angmom_1
    l_2, m_2, n_2 = angmom_2

    R_12 = centre_1 - centre_2

    A = (2 * angmom_2 + 1) * exponent_2
    B = -2 * exponent_2 * exponent_2
    C = -0.5 * angmom_2 * (angmom_2 - 1)

    # Overlap integrals for each Cartesian component

    Sx = hermite_coeff(l_1, l_2, 0, R_12[0], exponent_1, exponent_2)
    Sy = hermite_coeff(m_1, m_2, 0, R_12[1], exponent_1, exponent_2)
    Sz = hermite_coeff(n_1, n_2, 0, R_12[2], exponent_1, exponent_2)

    # For each Cartesian component, calculate the second derivative then multiply by overlaps of other two components

    Tx = A[0] * Sx + B * hermite_coeff(l_1, l_2 + 2, 0, R_12[0], exponent_1, exponent_2) + C[0] * hermite_coeff(l_1, l_2 - 2, 0, R_12[0], exponent_1, exponent_2)
    Ty = A[1] * Sy + B * hermite_coeff(m_1, m_2 + 2, 0, R_12[1], exponent_1, exponent_2) + C[1] * hermite_coeff(m_1, m_2 - 2, 0, R_12[1], exponent_1, exponent_2)
    Tz = A[2] * Sz + B * hermite_coeff(n_1, n_2 + 2, 0, R_12[2], exponent_1, exponent_2) + C[2] * hermite_coeff(n_1, n_2 - 2, 0, R_12[2], exponent_1, exponent_2)

    # Combines the three Cartesian components into the integral

    integral = (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz) * pow(PI / (exponent_1 + exponent_2), 1.5)

    return integral







    
cpdef double calculate_overlap_integral(object bf_1, object bf_2):

    """
    
    Calculates an overlap integral between basis functions, <1|2>.
    
    Args:
        bf_1 (Basis): First basis function
        bf_2 (Basis): Second basis function

    Returns:
        integral (float): Overlap integral between contracted Gaussians
        
    """

    cdef double integral = 0.0

    for ia, ca in enumerate(bf_1.coefs):

        for ib, cb in enumerate(bf_2.coefs):
            
            # Applies coefficients and norms to integrals between primitive Gaussians

            integral += bf_1.norm[ia] * bf_2.norm[ib] * ca * cb * overlap(bf_1.exps[ia], bf_1.shell, bf_1.origin, bf_2.exps[ib], bf_2.shell, bf_2.origin)


    return integral








def overlap(exponent_1: float, angmom_1: ndarray, centre_1: ndarray, exponent_2: float, angmom_2: ndarray, centre_2: ndarray) -> float:

    """
    
    Calculates an overlap integral between primitive Gaussians, <1|2>.
    
    Args:
        exponent_1 (float): Gaussian exponent on first centre
        angmom_1 (ndarray): Angular momenta of primitive Gaussian of first centre
        centre_1 (array): Coordinates of first centre
        exponent_2 (float): Gaussian exponent on second centre
        angmom_2 (ndarray): Angular momenta of primitive Gaussian of second centre
        centre_2 (array): Coordinates of second centre
    
    Returns:
        integral (float): Overlap integral between primitive Gaussian

    """

    l_1, m_1, n_1 = angmom_1
    l_2, m_2, n_2 = angmom_2

    # Calculates the Cartesian components of the overlap integral

    Sx = hermite_coeff(l_1, l_2, 0, centre_1[0] - centre_2[0], exponent_1, exponent_2) 
    Sy = hermite_coeff(m_1, m_2, 0, centre_1[1] - centre_2[1], exponent_1, exponent_2) 
    Sz = hermite_coeff(n_1, n_2, 0, centre_1[2] - centre_2[2], exponent_1, exponent_2)
 
    integral = Sx * Sy * Sz * pow(PI / (exponent_1 + exponent_2), 1.5)

    return integral









cpdef double[:, :, :, :] calculate_electron_repulsion_integrals(long n_basis, double[:, :, :, :] ERI_AO, list bfs):

    """
    
    Calculates the electron repulsion integrals array between basis functions, <12(r)| 1/(r-r') |34(r')>.
    
    Args:
        n_basis (int): Number of basis functions
        ERI_AO (array): Electron repulsion integrals zeroed array
        bfs (list): List of basis functions

    Returns:
        ERI_AO (array): Electron repulsion integrals
        
    """

    cdef:
        long i, j, k, l, l_stop
        double integral
        Basis bf_1, bf_2, bf_3, bf_4

    for i in range(n_basis):

        bf_1 = <Basis>bfs[i]

        for j in range(i + 1):

            bf_2 = <Basis>bfs[j]

            for k in range(i + 1):

                bf_3 = <Basis>bfs[k]

                # Enforces (k,l) <= (i,j) in pair-index ordering

                l_stop = j + 1 if k == i else k + 1

                for l in range(l_stop):

                    bf_4 = <Basis>bfs[l]

                    # Checks if the sum of angular momenta is odd, in which case, for a diatomic, the integral is zero

                    if (
                        ((bf_1.shell[0] + bf_2.shell[0] + bf_3.shell[0] + bf_4.shell[0]) & 1) or
                        ((bf_1.shell[1] + bf_2.shell[1] + bf_3.shell[1] + bf_4.shell[1]) & 1)
                    ):

                        integral = 0.0

                    else:

                        integral = calculate_electron_repulsion_integral(bf_1, bf_2, bf_3, bf_4)

                    # Enforces the eightfold symmetry of two-electron integrals, saving lots of time

                    ERI_AO[i, j, k, l] = integral
                    ERI_AO[k, l, i, j] = integral
                    ERI_AO[j, i, l, k] = integral
                    ERI_AO[l, k, j, i] = integral
                    ERI_AO[j, i, k, l] = integral
                    ERI_AO[l, k, i, j] = integral
                    ERI_AO[i, j, l, k] = integral
                    ERI_AO[k, l, j, i] = integral


    return ERI_AO









cpdef double calculate_electron_repulsion_integral(Basis bf_1, Basis bf_2, Basis bf_3, Basis bf_4):

    """
    
    Calculates an electron repulsion integral between basis functions, <12(r)| 1/(r-r') |34(r')>.
    
    Args:
        bf_1 (Basis): First basis function
        bf_2 (Basis): Second basis function
        bf_3 (Basis): Third basis function
        bf_4 (Basis): Fourth basis function

    Returns:
        integral (float): Electron repulsion integral between contracted Gaussians
        
    """

    cdef double integral = 0.0
    cdef double primitive_value
    cdef long i, j, k, l
    cdef double contraction_prefactor

    for i in range(bf_1.num_exps):

        for j in range(bf_2.num_exps):

            for k in range(bf_3.num_exps):

                for l in range(bf_4.num_exps):

                    # Applies coefficients and norms to integrals between primitive Gaussians

                    contraction_prefactor = bf_1.norm[i] * bf_2.norm[j] * bf_3.norm[k] * bf_4.norm[l] * bf_1.coefs[i] * bf_2.coefs[j] * bf_3.coefs[k] * bf_4.coefs[l]
        
                    primitive_value = electron_repulsion(bf_1.exps[i], bf_1.shell, bf_1.origin, bf_2.exps[j], bf_2.shell, bf_2.origin, bf_3.exps[k], bf_3.shell, bf_3.origin, bf_4.exps[l], bf_4.shell, bf_4.origin)

                    integral += contraction_prefactor * primitive_value


    return integral










cdef inline double hermite_coeff(int l_1, int l_2, int t, double R_12, double exponent_1, double exponent_2):

    """
    
    Calculates the Hermite coefficient by recursion.
    
    Args:
        l_1 (int): Angular momentum on first Gaussian
        l_2 (int): Angular momentum on second Gaussian
        t (int): Hermite index
        R_12 (float): Distance between primitive Gaussian centres
        exponent_1 (float): Exponent on first primitive Gaussian
        exponent_2 (float): Exponent on second primitive Gaussian

    Returns:
        result (float): Hermite coefficient
        
    """

    cdef double exponent_sum = exponent_1 + exponent_2
    cdef double u = exponent_1 * exponent_2 / exponent_sum
    cdef double prefactor = 1.0 / (2.0 * exponent_sum)
    cdef double result = 0.0

    # Checks for an invalid Hermite coefficient

    if t < 0 or t > (l_1 + l_2):

        return 0.0

    # Calculates the Hermite coefficient for (s|s) integrals

    elif l_1 == 0 and l_2 == 0 and t == 0:

        return exp(-u * R_12 * R_12)

    # Build up angular momentum on Gaussian 1, for (X|s)

    elif l_2 == 0:

        result = prefactor * hermite_coeff(l_1 - 1, l_2, t - 1, R_12, exponent_1, exponent_2)
        result += - (u * R_12 / exponent_1) * hermite_coeff(l_1 - 1, l_2, t, R_12, exponent_1, exponent_2)
        result += (t + 1) * hermite_coeff(l_1 - 1, l_2, t + 1, R_12, exponent_1, exponent_2)
    
    # Build up angular momentum on Gaussian 2

    else:

        result = prefactor * hermite_coeff(l_1, l_2 - 1, t - 1, R_12, exponent_1, exponent_2)
        result += (u * R_12 / exponent_2) * hermite_coeff(l_1, l_2 - 1, t, R_12, exponent_1, exponent_2)
        result += (t + 1) * hermite_coeff(l_1, l_2 - 1, t + 1, R_12, exponent_1, exponent_2)


    return result








cdef inline double boys(int m, double T):

    """
    
    Calculates a Boys function.
    
    Args:
        m (int): Boys function order
        T (float): Boys function argument

    Returns:
        result (float): Boys function value
        
    """

    return hyp1f1(m + 0.5, m + 1.5, -T) / (2.0 * m + 1.0)









cdef inline double odd_double_double_fact_from_even(int n_even):

    """
    
    Calculates (n_even - 1)!! for an even integer.
    
    Args:
        n_even (int): Even integer input
    
    Returns:
        odd_double_double_fact_from_even (float): Odd double factorial
        
    """

    return double_fact(n_even - 1)










cdef inline void fill_boys_table(int M, double T, double* boys_table):

    """
    
    Fills a table of Boys functions from order 0 to M.
    
    Args:
        M (int): Maximum Boys function order
        T (float): Boys function argument
        boys_table (double*): Output table of Boys function values
        
    """

    cdef int m
    cdef double exponential_term
    cdef double two_T

    if T == 0.0:

        for m in range(M + 1):

            boys_table[m] = 1.0 / (2.0 * m + 1.0)

        return

    boys_table[M] = boys(M, T)

    exponential_term = exp(-T)
    two_T = 2.0 * T

    for m in range(M, 0, -1):

        boys_table[m - 1] = (two_T * boys_table[m] + exponential_term) / (2.0 * m - 1.0)









cdef inline void fill_pow_table(int M, double scale, double* pow_table):

    """
    
    Fills a table of powers of -2 * scale.
    
    Args:
        M (int): Maximum power
        scale (float): Scale factor in the recurrence
        pow_table (double*): Output table of powers
        
    """

    cdef int n
    cdef double factor = -2.0 * scale

    pow_table[0] = 1.0

    for n in range(1, M + 1):

        pow_table[n] = pow_table[n - 1] * factor









cdef inline void fill_Rz_linear_table(int Vmax, int Nmax, double PCz, double* boys_table, double* pow_table, double* Rz_table):

    """
    
    Fills the linear Coulomb Hermite recursion table along the z axis.
    
    Args:
        Vmax (int): Maximum z angular momentum index
        Nmax (int): Maximum Boys function order
        PCz (float): Distance between Gaussian product centres along z
        boys_table (double*): Table of Boys function values
        pow_table (double*): Table of powers of -2 * scale
        Rz_table (double*): Output Coulomb Hermite table
    
        
    """

    cdef int v, n
    cdef int stride = Nmax + 1
    cdef int row_offset
    cdef int prev_row_1_offset
    cdef int prev_row_2_offset

    for n in range(Nmax + 1):

        Rz_table[n] = pow_table[n] * boys_table[n]

    for v in range(1, Vmax + 1):

        row_offset = v * stride
        prev_row_1_offset = (v - 1) * stride
        prev_row_2_offset = (v - 2) * stride

        for n in range(Nmax - v, -1, -1):

            Rz_table[row_offset + n] = PCz * Rz_table[prev_row_1_offset + n + 1]

            if v > 1:

                Rz_table[row_offset + n] += (v - 1) * Rz_table[prev_row_2_offset + n + 1]









cdef inline void fill_hermite_table(int l_1, int l_2, double R_12, double exponent_1, double exponent_2, double* hermite_table, bint use_parity):

    """
    
    Fills a Hermite coefficient table for a Cartesian direction.
    
    Args:
        l_1 (int): Angular momentum on first Gaussian
        l_2 (int): Angular momentum on second Gaussian
        R_12 (float): Distance between primitive Gaussian centres
        exponent_1 (float): Exponent on first primitive Gaussian
        exponent_2 (float): Exponent on second primitive Gaussian
        hermite_table (double*): Output table of Hermite coefficients
        use_parity (bool): Whether to zero coefficients forbidden by linear parity
        
    """

    cdef int t
    cdef int n_terms = l_1 + l_2 + 1
    cdef int t_start

    if use_parity:

        t_start = (l_1 + l_2) & 1

        for t in range(n_terms):

            hermite_table[t] = 0.0

        for t in range(t_start, n_terms, 2):

            hermite_table[t] = hermite_coeff(l_1, l_2, t, R_12, exponent_1, exponent_2)

    else:

        for t in range(n_terms):

            hermite_table[t] = hermite_coeff(l_1, l_2, t, R_12, exponent_1, exponent_2)










cdef double electron_repulsion(double exponent_1, long *angmom_1, double *centre_1, double exponent_2, long *angmom_2, double *centre_2,
                               double exponent_3, long *angmom_3, double *centre_3, double exponent_4, long *angmom_4, double *centre_4):

    """
    
    Calculates an electron repulsion integral between primitive Gaussians, <12(r)| 1/(r-r') |34(r')>.
    
    Args:
        exponent_1 (float): Gaussian exponent on first centre
        angmom_1 (array): Angular momenta of primitive Gaussian of first centre
        centre_1 (array): Coordinates of first centre
        exponent_2 (float): Gaussian exponent on second centre
        angmom_2 (array): Angular momenta of primitive Gaussian of second centre
        centre_2 (array): Coordinates of second centre
        exponent_3 (float): Gaussian exponent on third centre
        angmom_3 (array): Angular momenta of primitive Gaussian of third centre
        centre_3 (array): Coordinates of third centre
        exponent_4 (float): Gaussian exponent on fourth centre
        angmom_4 (array): Angular momenta of primitive Gaussian of fourth centre
        centre_4 (array): Coordinates of fourth centre
    
    Returns:
        integral (float): Electron repulsion integral between primitive Gaussians
        
    """

    cdef:
        long l_1 = angmom_1[0]
        long m_1 = angmom_1[1]
        long n_1 = angmom_1[2]
        long l_2 = angmom_2[0]
        long m_2 = angmom_2[1]
        long n_2 = angmom_2[2]
        long l_3 = angmom_3[0]
        long m_3 = angmom_3[1]
        long n_3 = angmom_3[2]
        long l_4 = angmom_4[0]
        long m_4 = angmom_4[1]
        long n_4 = angmom_4[2]

        double exponent_sum_12 = exponent_1 + exponent_2
        double exponent_sum_34 = exponent_3 + exponent_4
        double reduced_exponent = exponent_sum_12 * exponent_sum_34 / (exponent_sum_12 + exponent_sum_34)

        double Rz_12 = centre_1[2] - centre_2[2]
        double Rz_34 = centre_3[2] - centre_4[2]

        double Pz = (exponent_1 * centre_1[2] + exponent_2 * centre_2[2]) / exponent_sum_12
        double Qz = (exponent_3 * centre_3[2] + exponent_4 * centre_4[2]) / exponent_sum_34
        double PQz = Pz - Qz

        int n_hermite_x_12 = <int>(l_1 + l_2 + 1)
        int n_hermite_y_12 = <int>(m_1 + m_2 + 1)
        int n_hermite_z_12 = <int>(n_1 + n_2 + 1)
        int n_hermite_x_34 = <int>(l_3 + l_4 + 1)
        int n_hermite_y_34 = <int>(m_3 + m_4 + 1)
        int n_hermite_z_34 = <int>(n_3 + n_4 + 1)

        int Vmax = <int>(n_1 + n_2 + n_3 + n_4)
        int Nmax = <int>((l_1 + m_1 + n_1) + (l_2 + m_2 + n_2) + (l_3 + m_3 + n_3) + (l_4 + m_4 + n_4))
        int stride = Nmax + 1

        int t, u, v, tau, nu, phi
        int t_start = <int>((l_1 + l_2) & 1)
        int u_start = <int>((m_1 + m_2) & 1)
        int tau_start = <int>((l_3 + l_4) & 1)
        int nu_start = <int>((m_3 + m_4) & 1)
        int n_xy

        double integral = 0.0
        double prefactor
        double coefficient_x_12, coefficient_y_12, coefficient_z_12
        double coefficient_x_34, coefficient_y_34, coefficient_z_34
        double x_factor, xy_factor
        double sign

        double *hermite_x_12 = NULL
        double *hermite_y_12 = NULL
        double *hermite_z_12 = NULL
        double *hermite_x_34 = NULL
        double *hermite_y_34 = NULL
        double *hermite_z_34 = NULL

        double *boys_table = NULL
        double *pow_table = NULL
        double *Rz_table = NULL

    # If the sum of angular momenta is odd, by parity the integral is zero

    if ((l_1 + l_2 + l_3 + l_4) & 1) or ((m_1 + m_2 + m_3 + m_4) & 1):

        return 0.0


    hermite_x_12 = <double*>malloc(n_hermite_x_12 * sizeof(double))
    hermite_y_12 = <double*>malloc(n_hermite_y_12 * sizeof(double))
    hermite_z_12 = <double*>malloc(n_hermite_z_12 * sizeof(double))
    hermite_x_34 = <double*>malloc(n_hermite_x_34 * sizeof(double))
    hermite_y_34 = <double*>malloc(n_hermite_y_34 * sizeof(double))
    hermite_z_34 = <double*>malloc(n_hermite_z_34 * sizeof(double))

    boys_table = <double*>malloc((Nmax + 1) * sizeof(double))
    pow_table = <double*>malloc((Nmax + 1) * sizeof(double))
    Rz_table = <double*>malloc((Vmax + 1) * (Nmax + 1) * sizeof(double))

    if (hermite_x_12 == NULL or hermite_y_12 == NULL or hermite_z_12 == NULL or
        hermite_x_34 == NULL or hermite_y_34 == NULL or hermite_z_34 == NULL or
        boys_table == NULL or pow_table == NULL or Rz_table == NULL):

        free(hermite_x_12)
        free(hermite_y_12)
        free(hermite_z_12)
        free(hermite_x_34)
        free(hermite_y_34)
        free(hermite_z_34)
        free(boys_table)
        free(pow_table)
        free(Rz_table)

        return 0.0

    fill_hermite_table(l_1, l_2, 0.0, exponent_1, exponent_2, hermite_x_12, True)
    fill_hermite_table(m_1, m_2, 0.0, exponent_1, exponent_2, hermite_y_12, True)
    fill_hermite_table(n_1, n_2, Rz_12, exponent_1, exponent_2, hermite_z_12, False)

    fill_hermite_table(l_3, l_4, 0.0, exponent_3, exponent_4, hermite_x_34, True)
    fill_hermite_table(m_3, m_4, 0.0, exponent_3, exponent_4, hermite_y_34, True)
    fill_hermite_table(n_3, n_4, Rz_34, exponent_3, exponent_4, hermite_z_34, False)

    fill_boys_table(Nmax, reduced_exponent * PQz * PQz, boys_table)
    fill_pow_table(Nmax, reduced_exponent, pow_table)
    fill_Rz_linear_table(Vmax, Nmax, PQz, boys_table, pow_table, Rz_table)

    for t in range(t_start, n_hermite_x_12, 2):

        coefficient_x_12 = hermite_x_12[t]

        for tau in range(tau_start, n_hermite_x_34, 2):

            coefficient_x_34 = hermite_x_34[tau]
            x_factor = coefficient_x_12 * coefficient_x_34 * odd_double_double_fact_from_even(t + tau)

            for u in range(u_start, n_hermite_y_12, 2):

                coefficient_y_12 = hermite_y_12[u]

                for nu in range(nu_start, n_hermite_y_34, 2):

                    coefficient_y_34 = hermite_y_34[nu]
                    xy_factor = x_factor * coefficient_y_12 * coefficient_y_34 * odd_double_double_fact_from_even(u + nu)
                    n_xy = ((t + tau) >> 1) + ((u + nu) >> 1)

                    for v in range(n_hermite_z_12):

                        coefficient_z_12 = hermite_z_12[v]

                        if coefficient_z_12 == 0.0:

                            continue

                        for phi in range(n_hermite_z_34):

                            coefficient_z_34 = hermite_z_34[phi]

                            if coefficient_z_34 == 0.0:

                                continue

                            if ((tau + nu + phi) & 1):

                                sign = -1.0

                            else:

                                sign = 1.0

                            integral += xy_factor * coefficient_z_12 * coefficient_z_34 * sign * Rz_table[(v + phi) * stride + n_xy]


    prefactor = 2.0 * pow(PI, 2.5) / (exponent_sum_12 * exponent_sum_34 * sqrt(exponent_sum_12 + exponent_sum_34))
    integral *= prefactor

    free(hermite_x_12)
    free(hermite_y_12)
    free(hermite_z_12)
    free(hermite_x_34)
    free(hermite_y_34)
    free(hermite_z_34)
    free(boys_table)
    free(pow_table)
    free(Rz_table)

    return integral