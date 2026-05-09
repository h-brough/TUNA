# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport cython
from cython.parallel cimport prange
import numpy as np
from numpy import ndarray
cimport numpy as np
from libc.math cimport exp, pow, sqrt
from libc.stdlib cimport malloc, free
from scipy.special.cython_special cimport hyp1f1


cdef double PI = 3.141592653589793238462643383279
cdef double PI32 = 5.5683279968317078452848179821188357



ctypedef struct BasisRaw:

    double* origin

    int l
    int m
    int n

    long num_exps

    double* exps
    double* coefs
    double* norm




ctypedef struct PrimitivePairERI:

    double coefficient
    double exponent_sum
    double product_centre_z
    double centre_distance_z

    double hermite_x[20]
    double hermite_y[20]
    double hermite_z[20]




ctypedef struct AOPairERI:

    long i
    long j

    int nx
    int ny
    int nz

    int lx_sum
    int ly_sum
    int lz_sum

    int t_start
    int u_start

    int n_primitive_pairs

    PrimitivePairERI* primitive_pairs










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



    def __cinit__(self, origin: ndarray, shell: ndarray, num_exps: int, exps: ndarray, coefs: ndarray):

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

    # Deallocate memory for class parameters

    def __dealloc__(self):

        if self.origin != NULL: 
        
            free(self.origin)
        
        if self.shell != NULL: 
            
            free(self.shell)
        
        if self.exps != NULL: 
        
            free(self.exps)
        
        if self.coefs != NULL: 
            
            free(self.coefs)
        
        if self.norm != NULL: 
        
            free(self.norm)










cdef inline double double_fact(int n) noexcept nogil:

    """
    
    Calculates a double factorial.

    Args:
        n (int): Integer input
    
    Returns:
        result (float): Double factorialed integer
    
    """

    cdef double result = 1.0

    if n <= 0:

        return 1.0

    while n > 1:

        result *= n

        n = n - 2

    return result










cpdef tuple calculate_one_electron_integrals(long n_basis, list basis_functions, long n_atoms, list atoms, double[:] dipole_origin, int num_threads):

    """
    
    Calculates the one-electron integral matrices in Cartesian harmonics.
    
    Args:
        n_basis (int): Number of Cartesian basis functions
        basis_functions (list): List of Cartesian basis functions
        n_atoms (int): Number of atoms
        atoms (list): List of atom objects
        dipole_origin (array): Dipole and quadrupole electric origin
        num_threads (int): Number of OpenMP threads for parallelisation

    Returns:
        S_cart (array): Filled overlap matrix in AO basis
        T_cart (array): Filled kinetic matrix in AO basis
        V_cart (array): Filled nuclear matrix in AO basis
        D_cart (array): Filled dipole matrices in AO basis
        Q_cart (array): Filled quadrupole matrices in AO basis
    
    """

    S_cart = np.empty((n_basis, n_basis))
    T_cart = np.empty((n_basis, n_basis))
    V_cart = np.empty((n_basis, n_basis))
    D_cart = np.empty((3, n_basis, n_basis))
    Q_cart = np.empty((3, n_basis, n_basis))

    cdef:

        double[:, :] S = S_cart
        double[:, :] T = T_cart
        double[:, :] V = V_cart
        double[:, :, :] D = D_cart
        double[:, :, :] Q = Q_cart

        long i, j, a
        double s_ij, t_ij, v_ij
        double d_ij_x, d_ij_y, d_ij_z
        double q_ij_xx, q_ij_yy, q_ij_zz

        double origin[3]

        Basis bf
        object atom

        BasisRaw* bfs = <BasisRaw*>malloc(n_basis * sizeof(BasisRaw))
        double* atom_coords = <double*>malloc(3 * n_atoms * sizeof(double))
        double* atom_charges = <double*>malloc(n_atoms * sizeof(double))

    try:
        
        origin[0] = <double>dipole_origin[0]
        origin[1] = <double>dipole_origin[1]
        origin[2] = <double>dipole_origin[2]

        # Pure Python loop to map onto BasisRaw objects

        for i in range(n_basis):

            bf = <Basis>basis_functions[i]

            bfs[i].origin = bf.origin

            bfs[i].l = <int>bf.shell[0]
            bfs[i].m = <int>bf.shell[1]
            bfs[i].n = <int>bf.shell[2]

            bfs[i].num_exps = bf.num_exps
            bfs[i].exps = bf.exps
            bfs[i].coefs = bf.coefs
            bfs[i].norm = bf.norm

        # Pure Python loop to copy atom coordinates into raw C memory

        for a in range(n_atoms):

            atom = atoms[a]

            atom_coords[3 * a + 0] = <double>atom.origin[0]
            atom_coords[3 * a + 1] = <double>atom.origin[1]
            atom_coords[3 * a + 2] = <double>atom.origin[2]

            atom_charges[a] = <double>atom.charge

        # Loops over unique basis function pairs, OpenMP parallel, guided dynamic scheduling to spread workload efficiently

        for i in prange(n_basis, schedule = "guided", nogil = True, num_threads = num_threads):

            for j in range(i + 1):

                # Calculates all the one-electron integrals for this basis function pair
                
                s_ij = 0.0
                t_ij = 0.0

                d_ij_x = 0.0
                d_ij_y = 0.0
                d_ij_z = 0.0

                q_ij_xx = 0.0
                q_ij_yy = 0.0
                q_ij_zz = 0.0

                calculate_contracted_local_integrals(&bfs[i], &bfs[j], origin, &s_ij, &t_ij, &d_ij_x, &d_ij_y, &d_ij_z, &q_ij_xx, &q_ij_yy, &q_ij_zz)

                # The nuclear-electron integrals require a loop over atomic centres

                v_ij = 0.0

                for a in range(n_atoms):

                    v_ij = v_ij - calculate_contracted_nuclear_integral(&bfs[i], &bfs[j], &atom_coords[3 * a]) * atom_charges[a]

                # Assigns the integrals to the corresponding matrices

                S[i, j] = s_ij
                T[i, j] = t_ij
                V[i, j] = v_ij

                D[0, i, j] = d_ij_x
                D[1, i, j] = d_ij_y
                D[2, i, j] = d_ij_z

                Q[0, i, j] = q_ij_xx
                Q[1, i, j] = q_ij_yy
                Q[2, i, j] = q_ij_zz

                # Avoids assignment for the same index, as it's already been done

                if i != j:

                    S[j, i] = s_ij
                    T[j, i] = t_ij
                    V[j, i] = v_ij

                    D[0, j, i] = d_ij_x
                    D[1, j, i] = d_ij_y
                    D[2, j, i] = d_ij_z

                    Q[0, j, i] = q_ij_xx
                    Q[1, j, i] = q_ij_yy
                    Q[2, j, i] = q_ij_zz

    finally:

        # No matter what, deallocate the memory for basis functions

        free(bfs)
        free(atom_coords)
        free(atom_charges)

    return S_cart, T_cart, V_cart, D_cart, Q_cart










cdef inline void calculate_contracted_local_integrals(BasisRaw* bf_1, BasisRaw* bf_2, double* origin, double* s_ij, double* t_ij, double* d_ij_x, double* d_ij_y, double* d_ij_z, double* q_ij_xx, double* q_ij_yy, double* q_ij_zz) noexcept nogil:
    
    """
    
    Calculates a local one-electron integrals between a pair of contracted Gaussian functions.

    Args:
        bf_1 (Basis, in): First cartesian contracted basis function
        bf_2 (Basis, in): Second cartesian contracted basis function
        origin (array, in): Dipole and quadrupole origin
        s_ij (float, out): Overlap integral for ij pair
        t_ij (float, out): Kinetic integral for ij pair
        d_ij_x (float, out): Dipole x integral for ij pair
        d_ij_y (float, out): Dipole y integral for ij pair
        d_ij_z (float, out): Dipole z integral for ij pair
        q_ij_xx (float, out): Quadrupole xx integral for ij pair
        q_ij_yy (float, out): Quadrupole yy integral for ij pair
        q_ij_zz (float, out): Quadrupole zz integral for ij pair
    
    """

    cdef:

        long i, j

        int l_1 = bf_1.l
        int m_1 = bf_1.m
        int n_1 = bf_1.n

        int l_2 = bf_2.l
        int m_2 = bf_2.m
        int n_2 = bf_2.n

        long n_prim_1 = bf_1.num_exps
        long n_prim_2 = bf_2.num_exps

        double* exps_1 = bf_1.exps
        double* coefs_1 = bf_1.coefs
        double* norms_1 = bf_1.norm

        double* exps_2 = bf_2.exps
        double* coefs_2 = bf_2.coefs
        double* norms_2 = bf_2.norm

        double dx = bf_1.origin[0] - bf_2.origin[0]
        double dy = bf_1.origin[1] - bf_2.origin[1]
        double dz = bf_1.origin[2] - bf_2.origin[2]

        double exponent_sum
        double exponent_1, exponent_2
        double prefactor_1, prefactor_2, primitive_prefactor

        double Ex_1, Ey_1, Ez_1
        double Ex_2, Ey_2, Ez_2

        double Sx, Sy, Sz
        double Tx, Ty, Tz
        double Ax, Ay, Az
        double Bx, By, Bz
        double Px, Py, Pz
        double Dx, Dy, Dz
        double Qx, Qy, Qz

        double s_integral = 0.0
        double t_integral = 0.0

        double d_integral_x = 0.0
        double d_integral_y = 0.0
        double d_integral_z = 0.0

        double q_integral_xx = 0.0
        double q_integral_yy = 0.0
        double q_integral_zz = 0.0

    # Loops over primitives

    for i in range(n_prim_1):

        exponent_1 = exps_1[i]

        prefactor_1 = norms_1[i] * coefs_1[i]

        for j in range(n_prim_2):

            exponent_2 = exps_2[j]

            prefactor_2 = norms_2[j] * coefs_2[j]

            exponent_sum = exponent_1 + exponent_2

            # Calculates the common prefactor, square root is quicker than the "pow" function

            primitive_prefactor = prefactor_1 * prefactor_2 * PI32 / (exponent_sum * sqrt(exponent_sum))

            # Calculates the Hermite coefficients needed for overlap

            Sx = hermite_coeff(l_1, l_2, 0, dx, exponent_1, exponent_2)
            Sy = hermite_coeff(m_1, m_2, 0, dy, exponent_1, exponent_2)
            Sz = hermite_coeff(n_1, n_2, 0, dz, exponent_1, exponent_2)

            # Calculates the Hermite coefficients needed for dipoles and quadrupoles

            Ex_1 = hermite_coeff(l_1, l_2, 1, dx, exponent_1, exponent_2)
            Ey_1 = hermite_coeff(m_1, m_2, 1, dy, exponent_1, exponent_2)
            Ez_1 = hermite_coeff(n_1, n_2, 1, dz, exponent_1, exponent_2)

            Ex_2 = hermite_coeff(l_1, l_2, 2, dx, exponent_1, exponent_2)
            Ey_2 = hermite_coeff(m_1, m_2, 2, dy, exponent_1, exponent_2)
            Ez_2 = hermite_coeff(n_1, n_2, 2, dz, exponent_1, exponent_2)

            # Calculates the kinetic integral

            Ax = (2 * l_2 + 1) * exponent_2
            Ay = (2 * m_2 + 1) * exponent_2
            Az = (2 * n_2 + 1) * exponent_2

            Bx = -0.5 * l_2 * (l_2 - 1)
            By = -0.5 * m_2 * (m_2 - 1)
            Bz = -0.5 * n_2 * (n_2 - 1)

            Tx = Ax * Sx - 2.0 * exponent_2 * exponent_2 * hermite_coeff(l_1, l_2 + 2, 0, dx, exponent_1, exponent_2) + Bx * hermite_coeff(l_1, l_2 - 2, 0, dx, exponent_1, exponent_2)
            Ty = Ay * Sy - 2.0 * exponent_2 * exponent_2 * hermite_coeff(m_1, m_2 + 2, 0, dy, exponent_1, exponent_2) + By * hermite_coeff(m_1, m_2 - 2, 0, dy, exponent_1, exponent_2)
            Tz = Az * Sz - 2.0 * exponent_2 * exponent_2 * hermite_coeff(n_1, n_2 + 2, 0, dz, exponent_1, exponent_2) + Bz * hermite_coeff(n_1, n_2 - 2, 0, dz, exponent_1, exponent_2)

            # Calculates the product centre relative to the electric origin

            Px = (exponent_1 * bf_1.origin[0] + exponent_2 * bf_2.origin[0]) / exponent_sum - origin[0]
            Py = (exponent_1 * bf_1.origin[1] + exponent_2 * bf_2.origin[1]) / exponent_sum - origin[1]
            Pz = (exponent_1 * bf_1.origin[2] + exponent_2 * bf_2.origin[2]) / exponent_sum - origin[2]

            # Calculates the dipole integrals

            Dx = Ex_1 + Px * Sx
            Dy = Ey_1 + Py * Sy
            Dz = Ez_1 + Pz * Sz

            # Calculates the diagonal quadrupole integrals

            Qx = 2.0 * Ex_2 + 2.0 * Px * Ex_1 + (Px * Px + 1.0 / (2.0 * exponent_sum)) * Sx
            Qy = 2.0 * Ey_2 + 2.0 * Py * Ey_1 + (Py * Py + 1.0 / (2.0 * exponent_sum)) * Sy
            Qz = 2.0 * Ez_2 + 2.0 * Pz * Ez_1 + (Pz * Pz + 1.0 / (2.0 * exponent_sum)) * Sz
            
            # Adds up the various one-electron integrals

            s_integral += primitive_prefactor * Sx * Sy * Sz

            t_integral += primitive_prefactor * (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz)

            d_integral_x += primitive_prefactor * Dx * Sy * Sz
            d_integral_y += primitive_prefactor * Sx * Dy * Sz
            d_integral_z += primitive_prefactor * Sx * Sy * Dz

            q_integral_xx += primitive_prefactor * Qx * Sy * Sz
            q_integral_yy += primitive_prefactor * Sx * Qy * Sz
            q_integral_zz += primitive_prefactor * Sx * Sy * Qz

    # Assigns the integrals to the output arrays

    s_ij[0] = s_integral
    t_ij[0] = t_integral

    d_ij_x[0] = d_integral_x
    d_ij_y[0] = d_integral_y
    d_ij_z[0] = d_integral_z

    q_ij_xx[0] = q_integral_xx
    q_ij_yy[0] = q_integral_yy
    q_ij_zz[0] = q_integral_zz

    return










cpdef double[:, :] calculate_cross_basis_overlap_matrix(long n_basis_1, long n_basis_2, list basis_functions_1, list basis_functions_2, int num_threads):

    """
    
    Calculates the overlap matrix between two different basis sets.

    Args:
        n_basis_1 (int): Number of basis functions in first basis set
        n_basis_2 (int): Number of basis functions in second basis set
        bfs_1 (list): List of basis functions in first basis set
        bfs_2 (list): List of basis functions in second basis set
        num_threads (int): Number of OpenMP threads
    
    Returns:
        S_cross (array): Cross basis overlap matrix
    
    """

    S_cross = np.empty((n_basis_1, n_basis_2))

    cdef:

        double[:, :] S = S_cross

        long i, j, k, l
        double s_ij

        int l_1, l_2
        int m_1, m_2
        int n_1, n_2

        double dx, dy, dz

        double exponent_sum
        double exponent_1, exponent_2
        double prefactor_1, prefactor_2, primitive_prefactor

        double Sx, Sy, Sz

        Basis bf

        BasisRaw* bfs_1 = <BasisRaw*>malloc(n_basis_1 * sizeof(BasisRaw))
        BasisRaw* bfs_2 = <BasisRaw*>malloc(n_basis_2 * sizeof(BasisRaw))

    try:
        
        # Pure Python loop to map onto BasisRaw objects

        for i in range(n_basis_1):

            bf = <Basis>basis_functions_1[i]

            bfs_1[i].origin = bf.origin

            bfs_1[i].l = <int>bf.shell[0]
            bfs_1[i].m = <int>bf.shell[1]
            bfs_1[i].n = <int>bf.shell[2]

            bfs_1[i].num_exps = bf.num_exps
            bfs_1[i].exps = bf.exps
            bfs_1[i].coefs = bf.coefs

            bfs_1[i].norm = bf.norm
        
        for j in range(n_basis_2):

            bf = <Basis>basis_functions_2[j]

            bfs_2[j].origin = bf.origin

            bfs_2[j].l = <int>bf.shell[0]
            bfs_2[j].m = <int>bf.shell[1]
            bfs_2[j].n = <int>bf.shell[2]

            bfs_2[j].num_exps = bf.num_exps
            bfs_2[j].exps = bf.exps
            bfs_2[j].coefs = bf.coefs
            bfs_2[j].norm = bf.norm
        
        # Loops over all basis function pairs, OpenMP parallel, static scheduling as we are looping over all pairs

        for i in prange(n_basis_1, schedule = "static", nogil = True, num_threads = num_threads):

            # Assign the anggular momenta for the first basis function here

            l_1 = bfs_1[i].l
            m_1 = bfs_1[i].m
            n_1 = bfs_1[i].n

            for j in range(n_basis_2):

                # The difference in position of the basis functions

                dx = bfs_1[i].origin[0] - bfs_2[j].origin[0]
                dy = bfs_1[i].origin[1] - bfs_2[j].origin[1]
                dz = bfs_1[i].origin[2] - bfs_2[j].origin[2]

                l_2 = bfs_2[j].l
                m_2 = bfs_2[j].m
                n_2 = bfs_2[j].n

                s_ij = 0.0

                # Loop over primitives in each basis function

                for k in range(bfs_1[i].num_exps):

                    exponent_1 = bfs_1[i].exps[k]
                    
                    prefactor_1 = bfs_1[i].norm[k] * bfs_1[i].coefs[k]

                    for l in range(bfs_2[j].num_exps):

                        exponent_2 = bfs_2[j].exps[l]

                        prefactor_2 = bfs_2[j].norm[l] * bfs_2[j].coefs[l]

                        exponent_sum = exponent_1 + exponent_2

                        # Calculates the common prefactor, square root is quicker than the "pow" function

                        primitive_prefactor = prefactor_1 * prefactor_2 * PI32 / (exponent_sum * sqrt(exponent_sum))

                        # Calculates the Hermite coefficients

                        Sx = hermite_coeff(l_1, l_2, 0, dx, exponent_1, exponent_2)
                        Sy = hermite_coeff(m_1, m_2, 0, dy, exponent_1, exponent_2)
                        Sz = hermite_coeff(n_1, n_2, 0, dz, exponent_1, exponent_2)

                        # Build up the overlap matrix element before assigning when all primitives are looped over

                        s_ij = s_ij + primitive_prefactor * Sx * Sy * Sz

                S[i, j] = s_ij

    finally:

        # No matter what, deallocate the memory for basis functions

        free(bfs_1)
        free(bfs_2)

    return S_cross










cdef inline double calculate_contracted_nuclear_integral(BasisRaw* bf_1, BasisRaw* bf_2, double* atom_coord) noexcept nogil:

    """
    
    Calculates a nuclear-electron integral between contracted Gaussians, <1| Z/r |2>.

    This is only correct for atoms aligned on the z axis!

    Args:
        bf_1 (Basis): First cartesian contracted basis function
        bf_2 (Basis): Second cartesian contracted basis function
        atom_coord (array): Atomic coordinates
    
    Returns:
        integral (float): Integral between basis functions
    
    """

    cdef:

        long i, j
        int t, u, v

        int l_1 = bf_1.l
        int m_1 = bf_1.m
        int n_1 = bf_1.n

        int l_2 = bf_2.l
        int m_2 = bf_2.m
        int n_2 = bf_2.n

        long n_prim_1 = bf_1.num_exps
        long n_prim_2 = bf_2.num_exps

        double* exps_1 = bf_1.exps
        double* coefs_1 = bf_1.coefs
        double* norms_1 = bf_1.norm

        double* exps_2 = bf_2.exps
        double* coefs_2 = bf_2.coefs
        double* norms_2 = bf_2.norm

        double z_1 = bf_1.origin[2]
        double z_2 = bf_2.origin[2]
        double z_nucleus = atom_coord[2]
        double Rz_12 = z_1 - z_2

        int Vmax = n_1 + n_2
        int Nmax = l_1 + l_2 + m_1 + m_2 + n_1 + n_2
        int stride = Nmax + 1

        double* boys_tab = <double*>malloc((Nmax + 1) * sizeof(double))
        double* pow_tab = <double*>malloc((Nmax + 1) * sizeof(double))
        double* Rz = <double*>malloc((Vmax + 1) * (Nmax + 1) * sizeof(double))

        double exponent_1, exponent_2, exponent_sum
        double prefactor_1, prefactor_2
        double PCz
        double Ex, Ey, Ez
        double primitive_integral
        double integral = 0.0

    # Loops over primitive Gaussians in the contraction

    for i in range(n_prim_1):

        exponent_1 = exps_1[i]

        prefactor_1 = norms_1[i] * coefs_1[i]

        for j in range(n_prim_2):

            exponent_2 = exps_2[j]

            prefactor_2 = norms_2[j] * coefs_2[j]

            exponent_sum = exponent_1 + exponent_2

            PCz = (exponent_1 * z_1 + exponent_2 * z_2) / exponent_sum - z_nucleus

            # Prepares the hash tables for efficient lookups

            fill_boys_table(Nmax, exponent_sum * PCz * PCz, boys_tab)

            fill_pow_table(Nmax, exponent_sum, pow_tab)
            
            fill_Rz_linear_table(Vmax, Nmax, PCz, boys_tab, pow_tab, Rz)

            primitive_integral = 0.0

            for t in range(0, l_1 + l_2 + 1, 2):

                Ex = hermite_coeff(l_1, l_2, t, 0.0, exponent_1, exponent_2) * odd_double_double_fact_from_even(t)

                for u in range(0, m_1 + m_2 + 1, 2):

                    Ey = hermite_coeff(m_1, m_2, u, 0.0, exponent_1, exponent_2) * odd_double_double_fact_from_even(u)

                    for v in range(n_1 + n_2 + 1):

                        Ez = hermite_coeff(n_1, n_2, v, Rz_12, exponent_1, exponent_2)

                        primitive_integral += Ex * Ey * Ez * Rz[v * stride + (t + u) // 2]

            # Builds up the total contribution to the contracted integral

            integral += prefactor_1 * prefactor_2 * primitive_integral * 2.0 * PI / exponent_sum

    free(boys_tab)
    free(pow_tab)
    free(Rz)

    return integral






















cdef inline double odd_double_fact_even_argument_fast(int n_even) noexcept nogil:

    """

    Returns (n_even - 1)!! for even n_even values encountered in the x/y
    contractions. The lookup avoids repeated tiny recursive/loop calls in the
    primitive ERI hot path.

    """

    if n_even <= 0:
        return 1.0
    elif n_even == 2:
        return 1.0
    elif n_even == 4:
        return 3.0
    elif n_even == 6:
        return 15.0
    elif n_even == 8:
        return 105.0
    elif n_even == 10:
        return 945.0
    elif n_even == 12:
        return 10395.0
    elif n_even == 14:
        return 135135.0
    elif n_even == 16:
        return 2027025.0
    elif n_even == 18:
        return 34459425.0
    elif n_even == 20:
        return 654729075.0
    else:
        return odd_double_double_fact_from_even(n_even)













cdef inline void fill_hermite_table_iter_eri(int l_1, int l_2, double R_12, double exponent_1, double exponent_2, double* hermite_table, bint use_parity):

    """

    Iterative Hermite coefficient table used by ERIs only. This replaces many
    repeated recursive hermite_coeff calls in the primitive-quartet hot path.

    """

    cdef:
        int i, j, t, idx
        int n_l2 = l_2 + 1
        int n_terms = l_1 + l_2 + 1
        int stride = n_terms + 1
        int total = (l_1 + 1) * (l_2 + 1) * stride
        int base_idx, prev_idx
        int t_start
        double exponent_sum = exponent_1 + exponent_2
        double reduced_exponent = exponent_1 * exponent_2 / exponent_sum
        double prefactor = 1.0 / (2.0 * exponent_sum)
        double shift_1 = -reduced_exponent * R_12 / exponent_1
        double shift_2 = reduced_exponent * R_12 / exponent_2
        double E[8400]

    for idx in range(total):
        E[idx] = 0.0

    E[0] = exp(-reduced_exponent * R_12 * R_12)

    for i in range(l_1 + 1):

        for j in range(l_2 + 1):

            if i == 0 and j == 0:
                continue

            base_idx = (i * n_l2 + j) * stride

            if j == 0:

                prev_idx = ((i - 1) * n_l2 + j) * stride

                for t in range(i + j + 1):

                    E[base_idx + t] = shift_1 * E[prev_idx + t] + (t + 1) * E[prev_idx + t + 1]

                    if t > 0:
                        E[base_idx + t] += prefactor * E[prev_idx + t - 1]

            else:

                prev_idx = (i * n_l2 + j - 1) * stride

                for t in range(i + j + 1):

                    E[base_idx + t] = shift_2 * E[prev_idx + t] + (t + 1) * E[prev_idx + t + 1]

                    if t > 0:
                        E[base_idx + t] += prefactor * E[prev_idx + t - 1]

    base_idx = (l_1 * n_l2 + l_2) * stride

    if use_parity:

        for t in range(n_terms):
            hermite_table[t] = 0.0

        t_start = (l_1 + l_2) & 1

        for t in range(t_start, n_terms, 2):
            hermite_table[t] = E[base_idx + t]

    else:

        for t in range(n_terms):
            hermite_table[t] = E[base_idx + t]













cdef inline void build_primitive_pair_eri(Basis bf_1, Basis bf_2, long i, long j, PrimitivePairERI* pair):

    """

    Builds all primitive-pair quantities that depend only on two basis
    functions. These are reused for every partner AO pair in the ERI build.

    """

    cdef:
        double exponent_1 = bf_1.exps[i]
        double exponent_2 = bf_2.exps[j]
        double exponent_sum = exponent_1 + exponent_2
        int l_1 = <int>bf_1.shell[0]
        int m_1 = <int>bf_1.shell[1]
        int n_1 = <int>bf_1.shell[2]
        int l_2 = <int>bf_2.shell[0]
        int m_2 = <int>bf_2.shell[1]
        int n_2 = <int>bf_2.shell[2]

    pair.coefficient = bf_1.norm[i] * bf_2.norm[j] * bf_1.coefs[i] * bf_2.coefs[j]
    pair.exponent_sum = exponent_sum
    pair.product_centre_z = (exponent_1 * bf_1.origin[2] + exponent_2 * bf_2.origin[2]) / exponent_sum
    pair.centre_distance_z = bf_1.origin[2] - bf_2.origin[2]

    fill_hermite_table_iter_eri(l_1, l_2, 0.0, exponent_1, exponent_2, pair.hermite_x, True)
    fill_hermite_table_iter_eri(m_1, m_2, 0.0, exponent_1, exponent_2, pair.hermite_y, True)
    fill_hermite_table_iter_eri(n_1, n_2, pair.centre_distance_z, exponent_1, exponent_2, pair.hermite_z, False)













cdef inline void build_ao_pair_eri(AOPairERI* pair, long i, long j, Basis bf_1, Basis bf_2):

    """

    Builds an AO-pair cache for all primitive pairs in the contracted pair.

    """

    cdef:
        long a, b
        int primitive_pair_index = 0

    pair.i = i
    pair.j = j

    pair.lx_sum = <int>(bf_1.shell[0] + bf_2.shell[0])
    pair.ly_sum = <int>(bf_1.shell[1] + bf_2.shell[1])
    pair.lz_sum = <int>(bf_1.shell[2] + bf_2.shell[2])

    pair.nx = pair.lx_sum + 1
    pair.ny = pair.ly_sum + 1
    pair.nz = pair.lz_sum + 1

    pair.t_start = pair.lx_sum & 1
    pair.u_start = pair.ly_sum & 1

    pair.n_primitive_pairs = <int>(bf_1.num_exps * bf_2.num_exps)
    pair.primitive_pairs = <PrimitivePairERI*>malloc(pair.n_primitive_pairs * sizeof(PrimitivePairERI))

    if pair.primitive_pairs == NULL:
        raise MemoryError()

    for a in range(bf_1.num_exps):

        for b in range(bf_2.num_exps):

            build_primitive_pair_eri(bf_1, bf_2, a, b, &pair.primitive_pairs[primitive_pair_index])
            primitive_pair_index += 1













cdef inline double primitive_pair_eri(AOPairERI* pair_12, PrimitivePairERI* primitive_pair_12,
                                     AOPairERI* pair_34, PrimitivePairERI* primitive_pair_34) noexcept nogil:

    """

    Primitive-pair contraction for a diatomic on the z axis.

    """

    cdef:
        double exponent_sum_12 = primitive_pair_12.exponent_sum
        double exponent_sum_34 = primitive_pair_34.exponent_sum
        double exponent_sum_total = exponent_sum_12 + exponent_sum_34
        double reduced_exponent = exponent_sum_12 * exponent_sum_34 / exponent_sum_total
        double PQz = primitive_pair_12.product_centre_z - primitive_pair_34.product_centre_z

        int Vmax = pair_12.lz_sum + pair_34.lz_sum
        int Nmax = pair_12.lx_sum + pair_12.ly_sum + pair_12.lz_sum + pair_34.lx_sum + pair_34.ly_sum + pair_34.lz_sum
        int stride = Nmax + 1

        int t, u, v, tau, nu, phi
        int n_xy

        double integral = 0.0
        double prefactor
        double coefficient_x_12, coefficient_y_12, coefficient_z_12
        double coefficient_x_34, coefficient_y_34, coefficient_z_34
        double x_factor, xy_factor
        double sign
        double boys_table[64]
        double pow_table[64]
        double Rz_table[1024]

    fill_boys_table(Nmax, reduced_exponent * PQz * PQz, boys_table)
    fill_pow_table(Nmax, reduced_exponent, pow_table)
    fill_Rz_linear_table(Vmax, Nmax, PQz, boys_table, pow_table, Rz_table)

    for t in range(pair_12.t_start, pair_12.nx, 2):

        coefficient_x_12 = primitive_pair_12.hermite_x[t]

        for tau in range(pair_34.t_start, pair_34.nx, 2):

            coefficient_x_34 = primitive_pair_34.hermite_x[tau]
            x_factor = coefficient_x_12 * coefficient_x_34 * odd_double_fact_even_argument_fast(t + tau)

            for u in range(pair_12.u_start, pair_12.ny, 2):

                coefficient_y_12 = primitive_pair_12.hermite_y[u]

                for nu in range(pair_34.u_start, pair_34.ny, 2):

                    coefficient_y_34 = primitive_pair_34.hermite_y[nu]
                    xy_factor = x_factor * coefficient_y_12 * coefficient_y_34 * odd_double_fact_even_argument_fast(u + nu)
                    n_xy = ((t + tau) >> 1) + ((u + nu) >> 1)

                    for v in range(pair_12.nz):

                        coefficient_z_12 = primitive_pair_12.hermite_z[v]

                        if coefficient_z_12 == 0.0:
                            continue

                        for phi in range(pair_34.nz):

                            coefficient_z_34 = primitive_pair_34.hermite_z[phi]

                            if coefficient_z_34 == 0.0:
                                continue

                            if ((tau + nu + phi) & 1):
                                sign = -1.0
                            else:
                                sign = 1.0

                            integral += xy_factor * coefficient_z_12 * coefficient_z_34 * sign * Rz_table[(v + phi) * stride + n_xy]

    prefactor = 34.986836655249725 / (exponent_sum_12 * exponent_sum_34 * sqrt(exponent_sum_total))

    return primitive_pair_12.coefficient * primitive_pair_34.coefficient * prefactor * integral













cdef inline double calculate_electron_repulsion_integral_cached(AOPairERI* pair_12, AOPairERI* pair_34) noexcept nogil:

    """

    Contracted ERI from two cached AO pairs.

    """

    cdef:
        int p, q
        double integral = 0.0

    for p in range(pair_12.n_primitive_pairs):

        for q in range(pair_34.n_primitive_pairs):

            integral += primitive_pair_eri(pair_12, &pair_12.primitive_pairs[p], pair_34, &pair_34.primitive_pairs[q])

    return integral













cpdef double[:, :, :, :] calculate_electron_repulsion_integrals(long n_basis, double[:, :, :, :] ERI_AO, list bfs, int num_threads):

    """

    Calculates the electron repulsion integrals array between basis functions, <12(r)| 1/(r-r') |34(r')>.

    This version uses AO-pair primitive caches. For a contracted AO pair (i,j),
    the primitive-pair Hermite data are built once, then reused against every
    partner AO pair.

    """

    cdef:
        long i, j, k, l
        long pair_index_12, pair_index_34, pair_count, pair_build_index
        double integral
        AOPairERI* ao_pairs = NULL
        Basis bf_1, bf_2
        bint initialized = False

    pair_count = n_basis * (n_basis + 1) // 2
    ao_pairs = <AOPairERI*>malloc(pair_count * sizeof(AOPairERI))

    if ao_pairs == NULL:
        raise MemoryError()

    for pair_build_index in range(pair_count):
        ao_pairs[pair_build_index].primitive_pairs = NULL

    try:

        pair_build_index = 0

        for i in range(n_basis):

            bf_1 = <Basis>bfs[i]

            for j in range(i + 1):

                bf_2 = <Basis>bfs[j]
                build_ao_pair_eri(&ao_pairs[pair_build_index], i, j, bf_1, bf_2)
                pair_build_index += 1

        initialized = True

        with nogil:

            for pair_index_12 in prange(pair_count, schedule="dynamic", num_threads = num_threads):

                i = ao_pairs[pair_index_12].i
                j = ao_pairs[pair_index_12].j

                for pair_index_34 in range(pair_index_12 + 1):

                    k = ao_pairs[pair_index_34].i
                    l = ao_pairs[pair_index_34].j

                    if (((ao_pairs[pair_index_12].lx_sum + ao_pairs[pair_index_34].lx_sum) & 1) or
                        ((ao_pairs[pair_index_12].ly_sum + ao_pairs[pair_index_34].ly_sum) & 1)):

                        integral = 0.0

                    else:

                        integral = calculate_electron_repulsion_integral_cached(&ao_pairs[pair_index_12], &ao_pairs[pair_index_34])

                    # Enforces the eightfold symmetry of two-electron integrals.

                    ERI_AO[i, j, k, l] = integral
                    ERI_AO[k, l, i, j] = integral
                    ERI_AO[j, i, l, k] = integral
                    ERI_AO[l, k, j, i] = integral
                    ERI_AO[j, i, k, l] = integral
                    ERI_AO[l, k, i, j] = integral
                    ERI_AO[i, j, l, k] = integral
                    ERI_AO[k, l, j, i] = integral

    finally:

        if ao_pairs != NULL:

            for pair_build_index in range(pair_count):

                if ao_pairs[pair_build_index].primitive_pairs != NULL:
                    free(ao_pairs[pair_build_index].primitive_pairs)

            free(ao_pairs)

    return ERI_AO




















cpdef double calculate_electron_repulsion_integral(Basis bf_1, Basis bf_2, Basis bf_3, Basis bf_4):

    """

    Calculates an electron repulsion integral between basis functions, <12(r)| 1/(r-r') |34(r')>.

    """

    cdef:
        AOPairERI pair_12
        AOPairERI pair_34
        double integral

    pair_12.primitive_pairs = NULL
    pair_34.primitive_pairs = NULL

    try:

        build_ao_pair_eri(&pair_12, 0, 0, bf_1, bf_2)
        build_ao_pair_eri(&pair_34, 0, 0, bf_3, bf_4)

        if (((pair_12.lx_sum + pair_34.lx_sum) & 1) or
            ((pair_12.ly_sum + pair_34.ly_sum) & 1)):

            integral = 0.0

        else:

            integral = calculate_electron_repulsion_integral_cached(&pair_12, &pair_34)

    finally:

        if pair_12.primitive_pairs != NULL:
            free(pair_12.primitive_pairs)

        if pair_34.primitive_pairs != NULL:
            free(pair_34.primitive_pairs)

    return integral













cdef inline double hermite_coeff(int l_1, int l_2, int t, double R_12, double exponent_1, double exponent_2) noexcept nogil:

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








cdef inline double boys(int m, double T) noexcept nogil:

    """
    
    Calculates a Boys function.
    
    Args:
        m (int): Boys function order
        T (float): Boys function argument

    Returns:
        result (float): Boys function value
        
    """

    return hyp1f1(m + 0.5, m + 1.5, -T) / (2.0 * m + 1.0)









cdef inline double odd_double_double_fact_from_even(int n_even) noexcept nogil:

    """
    
    Calculates (n_even - 1)!! for an even integer.
    
    Args:
        n_even (int): Even integer input
    
    Returns:
        odd_double_double_fact_from_even (float): Odd double factorial
        
    """

    return double_fact(n_even - 1)










cdef inline void fill_boys_table(int M, double T, double* boys_table) noexcept nogil:

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









cdef inline void fill_pow_table(int M, double scale, double* pow_table) noexcept nogil:

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









cdef inline void fill_Rz_linear_table(int Vmax, int Nmax, double PCz, double* boys_table, double* pow_table, double* Rz_table) noexcept nogil:

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

    return