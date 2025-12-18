# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from __future__ import division
cimport cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from libc.stdlib cimport malloc, free
from scipy.special.cython_special cimport hyp1f1 
from scipy.special import factorial2
from cpython.exc cimport PyErr_CheckSignals

cdef double pi = 3.141592653589793238462643383279


cdef class Basis:

    """
    
    Defines basis functions from primitive Gaussians.
    
    """

    cdef:

        double *origin
        long    *shell
        long    num_exps
        double *exps 
        double *coefs
        double *norm

    # Coordinates of basis function
    property origin:

        def __get__(self):

            cdef double[::1] view = <double[:3]> self.origin

            return np.asarray(view)

    # Subshells
    property shell:

        def __get__(self):

            cdef long[::1] view = <long[:3]> self.shell

            return np.asarray(view)
    
    # Number of exponents
    property num_exps:

        def __get__(self):

            cdef long view = <long> self.num_exps

            return view

    # Array of exponents
    property exps:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.exps

            return np.asarray(view)

    # Array of coefficients
    property coefs:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.coefs

            return np.asarray(view)

    # Norm of primitive Gaussian
    property norm:

        def __get__(self):

            cdef double[::1] view = <double[:self.num_exps]> self.norm

            return np.asarray(view)


    def __cinit__(self, origin, shell, num_exps, exps, coefs):

        self.origin = <double*>malloc(3 * sizeof(double))
        self.shell  = <long*>malloc(3 * sizeof(long))
        self.num_exps = num_exps
        self.exps = <double*>malloc(num_exps * sizeof(double))
        self.coefs = <double*>malloc(num_exps * sizeof(double))
        self.norm = <double*>malloc(num_exps * sizeof(double))

        # Iterates through x, y and z axes
        for i in range(3):

            self.origin[i] = origin[i]
            self.shell[i] = shell[i]

        # Iterates through number of primitive Gaussians
        for i in range(num_exps):

            self.exps[i] = exps[i]
            self.coefs[i] = coefs[i]
            self.norm[i] = 0.0 

        # Normalises basis set
        self.normalize()


    def __dealloc__(self):

        if self.origin != NULL: free(self.origin)
        if self.shell != NULL: free(self.shell)
        if self.exps != NULL: free(self.exps)
        if self.coefs != NULL: free(self.coefs)
        if self.norm != NULL: free(self.norm)



    def normalize(self):

        """
        
        Normalises the primitives, then normalises the contracted functions.
        
        """

        l = self.shell[0]
        m = self.shell[1]
        n = self.shell[2]

        L = l + m + n

        # Normalises primitives
        for ia in range(self.num_exps):

            self.norm[ia] = np.sqrt(np.power(2, 2 * (l + m + n) + 1.5) * np.power(self.exps[ia], l + m + n + 1.5) / fact2(2 * l - 1) / fact2(2 * m - 1) / fact2(2 * n - 1) / np.power(pi, 1.5))

        # Normalises contracted functions
        prefactor = np.power(pi, 1.5) * fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1) / np.power(2.0, L)

        N = 0.0

        for ia in range(self.num_exps):
            for ib in range(self.num_exps):

                N += self.norm[ia] * self.norm[ib] * self.coefs[ia] * self.coefs[ib] / np.power(self.exps[ia] + self.exps[ib], L + 1.5)

        N *= prefactor
        N = np.power(N,-0.5)

        for ia in range(self.num_exps): self.coefs[ia] *= N






def fact2(n):

    """
    
    Ensures the correct response of double factorial on a negative number.

    """

    if n == -1: 
    
        return factorial2(n, extend="complex")

    return factorial2(n)









def gaussian_product_center(double a, A, double b, B):

    """
    
    Calculates the centre of a product of Gaussians.
    
    """

    return (a * np.asarray(A) + b * np.asarray(B)) / (a + b)









cdef inline double E(int i, int j, int t, double Qx, double a, double b, int n = 0, double Ax = 0.0, dict cache = None):

    cdef double p = a + b
    cdef double u = a * b / p
    cdef double res = 0.0

    if n == 0:

        if t < 0 or t > (i + j): return 0.0

        elif i == 0 and j == 0 and t == 0: return exp(-u * Qx * Qx)
        elif j == 0: 
            
            res = (1 / (2 * p)) * E(i - 1, j, t - 1, Qx, a, b, 0, 0.0, cache) - (u * Qx / a) * E(i - 1, j, t, Qx, a, b, 0, 0.0, cache) + (t + 1) * E(i - 1, j, t + 1, Qx, a, b, 0, 0.0, cache)
        
        else: 
        
            res = (1 / (2 * p)) * E(i, j - 1, t - 1, Qx, a, b, 0, 0.0, cache) + (u * Qx / b) * E(i, j - 1, t, Qx, a, b, 0, 0.0, cache) + (t + 1) * E(i, j - 1, t + 1, Qx, a, b, 0, 0.0, cache)
    
    else:

        res =  E(i + 1, j, t, Qx, a, b, n - 1, Ax, cache) + Ax * E(i, j, t, Qx, a, b, n - 1, Ax, cache)

    return res












cdef inline double boys(double m, double T, dict cache = None):

    """
    
    Calculates the boys function.
    
    """

    return hyp1f1(m + 0.5, m + 1.5, -T) / (2.0 * m + 1.0) 








cpdef double [:, :, :, :] doERIs(long N,
                                double [:, :, :, :] TwoE,
                                list bfs,
                                bint use_diatomic_parity=True):

    cdef:
        long i, j, k, l, ij, kl
        double val
        Basis bi, bj, bk, bl

    for i in range(N):
        bi = <Basis>bfs[i]
        for j in range(i + 1):
            bj = <Basis>bfs[j]

            ij = (i * (i + 1) // 2 + j)

            for k in range(N):
                bk = <Basis>bfs[k]
                for l in range(k + 1):
                    bl = <Basis>bfs[l]

                    kl = (k * (k + 1) // 2 + l)

                    if ij >= kl:

                        # Exact diatomic (z-axis) parity screening
                        if use_diatomic_parity and (
                            ((bi.shell[0] + bj.shell[0] + bk.shell[0] + bl.shell[0]) & 1) or
                            ((bi.shell[1] + bj.shell[1] + bk.shell[1] + bl.shell[1]) & 1)
                        ):
                            val = 0.0
                        else:
                            val = ERI(bi, bj, bk, bl)

                        TwoE[i, j, k, l] = val
                        TwoE[k, l, i, j] = val
                        TwoE[j, i, l, k] = val
                        TwoE[l, k, j, i] = val
                        TwoE[j, i, k, l] = val
                        TwoE[l, k, i, j] = val
                        TwoE[i, j, l, k] = val
                        TwoE[k, l, j, i] = val

    return TwoE










cpdef double ERI(Basis a, Basis b, Basis c, Basis d):

    """
    
    Calculates the electron repulsion integrals for contracted Gaussians.

    """

    cdef double eri = 0.0
    cdef long ja, jb, jc, jd
    cdef double ca, cb, cc, cd
    cdef double an, bn, cn, dn
    cdef double ac, bc, cc2, dc
    cdef double pref

    for ja in range(a.num_exps):
        for jb in range(b.num_exps):
            for jc in range(c.num_exps):
                for jd in range(d.num_exps):

                    an = a.norm[ja]
                    ac = a.coefs[ja]
                    bn = b.norm[jb]
                    bc = b.coefs[jb]
                    cn = c.norm[jc]
                    cc2 = c.coefs[jc]   # renamed to avoid clash with "cc"
                    dn = d.norm[jd]
                    dc = d.coefs[jd]

                    pref = an * bn * cn * dn * ac * bc * cc2 * dc

                    eri += pref * electron_repulsion(a.exps[ja], a.shell, a.origin, b.exps[jb], b.shell, b.origin, c.exps[jc], c.shell, c.origin, d.exps[jd], d.shell, d.origin)
    
    return eri





cdef inline void fill_boys_table(int M, double T, double* F):
    """
    Fill F[0..M] with Boys values F_m(T).
    Uses ONE hyp1f1 evaluation at m=M, then downward recurrence:
        F_{m-1} = (2*T*F_m + exp(-T)) / (2m - 1)
    """
    cdef int m
    cdef double e, twoT

    if T == 0.0:
        for m in range(M + 1):
            F[m] = 1.0 / (2.0 * m + 1.0)
        return

    F[M] = hyp1f1(M + 0.5, M + 1.5, -T) / (2.0 * M + 1.0)

    e = exp(-T)
    twoT = 2.0 * T
    for m in range(M, 0, -1):
        F[m - 1] = (twoT * F[m] + e) / (2.0 * m - 1.0)


cdef inline void fill_pow_table(int M, double p, double* P):
    """
    P[n] = (-2*p)^n for n=0..M (avoids pow() in the base case).
    """
    cdef int n
    cdef double fac = -2.0 * p
    P[0] = 1.0
    for n in range(1, M + 1):
        P[n] = P[n - 1] * fac


cdef inline double R(int t, int u, int v, int n,
                     double p, double PCx, double PCy, double PCz, double RPC,
                     dict cache = None,
                     double* boys_tab = NULL,
                     double* pow_tab  = NULL):

    cdef double val = 0.0
    cdef long long key
    cdef object tmp

    if cache is not None:
        key = ((<long long>t) << 48) | ((<long long>u) << 32) | ((<long long>v) << 16) | (<long long>n)
        tmp = cache.get(key, None)
        if tmp is not None:
            return <double>tmp

    if t == 0 and u == 0 and v == 0:
        # IMPORTANT: other code (e.g. nuclear_attraction) calls R without tables.
        if boys_tab != NULL and pow_tab != NULL:
            val = pow_tab[n] * boys_tab[n]
        else:
            val = pow(-2.0 * p, n) * boys(n, p * RPC * RPC)

    elif t == 0 and u == 0:
        if v > 1:
            val += (v - 1) * R(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)
        val += PCz * R(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)

    elif t == 0:
        if u > 1:
            val += (u - 1) * R(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)
        val += PCy * R(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)

    else:
        if t > 1:
            val += (t - 1) * R(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)
        val += PCx * R(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC, cache, boys_tab, pow_tab)

    if cache is not None:
        cache[key] = val

    return val


cdef double electron_repulsion(double a, long *lmn1, double *A,
                               double b, long *lmn2, double *B,
                               double c, long *lmn3, double *C,
                               double d, long *lmn4, double *D):

    """
    Recursive calculation of electron repulsion integrals between primitive Gaussians.
    Optimization: precompute Boys table (one hyp1f1 call) and (-2*alpha)^n table once per quartet.
    Also precompute all 1D E-coefficients once per axis, per primitive pair.
    """

    cdef:
        long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
        long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
        long l3 = lmn3[0], m3 = lmn3[1], n3 = lmn3[2]
        long l4 = lmn4[0], m4 = lmn4[1], n4 = lmn4[2]

        double p = a + b
        double q = c + d
        double alpha = p * q / (p + q)

        double Px = (a * A[0] + b * B[0]) / p
        double Py = (a * A[1] + b * B[1]) / p
        double Pz = (a * A[2] + b * B[2]) / p
        double Qx = (c * C[0] + d * D[0]) / q
        double Qy = (c * C[1] + d * D[1]) / q
        double Qz = (c * C[2] + d * D[2]) / q

        double dx = Px - Qx
        double dy = Py - Qy
        double dz = Pz - Qz
        double RPQ = sqrt(dx * dx + dy * dy + dz * dz)

        double ABx = A[0] - B[0]
        double ABy = A[1] - B[1]
        double ABz = A[2] - B[2]
        double CDx = C[0] - D[0]
        double CDy = C[1] - D[1]
        double CDz = C[2] - D[2]

        long t, u, v, tau, nu, phi
        double val = 0.0
        double Eu, sign, pref

        dict R_cache = {}

        int nt   = <int>(l1 + l2 + 1)
        int nuu  = <int>(m1 + m2 + 1)
        int nv   = <int>(n1 + n2 + 1)
        int ntau = <int>(l3 + l4 + 1)
        int nnu  = <int>(m3 + m4 + 1)
        int nphi = <int>(n3 + n4 + 1)

        int Mmax = <int>((l1 + m1 + n1) + (l2 + m2 + n2) + (l3 + m3 + n3) + (l4 + m4 + n4))
        double Tval = alpha * RPQ * RPQ

        double *Etx = NULL
        double *Ety = NULL
        double *Etz = NULL
        double *Eux = NULL
        double *Euy = NULL
        double *Euz = NULL

        double *boys_tab = NULL
        double *pow_tab  = NULL

    if PyErr_CheckSignals() != 0:
        return 0.0

    Etx = <double*>malloc(nt   * sizeof(double))
    Ety = <double*>malloc(nuu  * sizeof(double))
    Etz = <double*>malloc(nv   * sizeof(double))
    Eux = <double*>malloc(ntau * sizeof(double))
    Euy = <double*>malloc(nnu  * sizeof(double))
    Euz = <double*>malloc(nphi * sizeof(double))

    if Etx == NULL or Ety == NULL or Etz == NULL or Eux == NULL or Euy == NULL or Euz == NULL:
        if Etx != NULL: free(Etx)
        if Ety != NULL: free(Ety)
        if Etz != NULL: free(Etz)
        if Eux != NULL: free(Eux)
        if Euy != NULL: free(Euy)
        if Euz != NULL: free(Euz)
        return 0.0

    boys_tab = <double*>malloc((Mmax + 1) * sizeof(double))
    pow_tab  = <double*>malloc((Mmax + 1) * sizeof(double))
    if boys_tab == NULL or pow_tab == NULL:
        if boys_tab != NULL: free(boys_tab)
        if pow_tab  != NULL: free(pow_tab)
        free(Etx); free(Ety); free(Etz)
        free(Eux); free(Euy); free(Euz)
        return 0.0

    fill_boys_table(Mmax, Tval, boys_tab)
    fill_pow_table(Mmax, alpha, pow_tab)

    for t in range(nt):
        Etx[t] = E(l1, l2, t, ABx, a, b)
    for u in range(nuu):
        Ety[u] = E(m1, m2, u, ABy, a, b)
    for v in range(nv):
        Etz[v] = E(n1, n2, v, ABz, a, b)

    for tau in range(ntau):
        Eux[tau] = E(l3, l4, tau, CDx, c, d)
    for nu in range(nnu):
        Euy[nu] = E(m3, m4, nu, CDy, c, d)
    for phi in range(nphi):
        Euz[phi] = E(n3, n4, phi, CDz, c, d)

    for t in range(nt):
        for u in range(nuu):
            for v in range(nv):

                Eu = Etx[t] * Ety[u] * Etz[v]

                for tau in range(ntau):
                    for nu in range(nnu):
                        for phi in range(nphi):

                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0

                            val += Eu * Eux[tau] * Euy[nu] * Euz[phi] * sign * \
                                   R(t + tau, u + nu, v + phi, 0,
                                     alpha, Px - Qx, Py - Qy, Pz - Qz, RPQ,
                                     R_cache, boys_tab, pow_tab)

    pref = 2.0 * pow(pi, 2.5) / (p * q * sqrt(p + q))
    val *= pref

    free(Etx); free(Ety); free(Etz)
    free(Eux); free(Euy); free(Euz)
    free(boys_tab)
    free(pow_tab)

    return val







cpdef double S(object a, object b):
    
    """
    
    Calculates overlap integrals between contracted Gaussians.
    
    """

    cdef double s = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):

            s += a.norm[ia] * b.norm[ib] * ca * cb * overlap(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin)

    return s








cpdef double Mu(object a, object b, C, str direction):

    """
    
    Calculates dipole integrals between contracted Gaussians.
    
    """

    cdef double mu = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):

            mu += a.norm[ia] * b.norm[ib] * ca * cb * dipole(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin, C, direction)

    return mu










cpdef double T(object a, object b):

    """
    
    Calculates kinetic integrals between contracted Gaussians.
    
    """

    cdef double t = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):

            t += a.norm[ia] * b.norm[ib] * ca * cb * kinetic(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin)

    return t








cpdef double V(object a, object b, double [:] C): 

    """
    
    Calculates nuclear attraction integrals between contracted Gaussians.
    
    """

    cdef double v = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):

            v += a.norm[ia] * b.norm[ib] * ca * cb * nuclear_attraction(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin, C)

    return v









def overlap(a,lmn1,A,b,lmn2,B):

    """
    
    Calculates overlap integrals between primitive Gaussians.
    
    """

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    S1 = E(l1, l2, 0, A[0] - B[0], a, b)
    S2 = E(m1, m2, 0, A[1] - B[1], a, b)
    S3 = E(n1, n2, 0, A[2] - B[2], a, b)

    return S1 * S2 * S3 * np.power(pi / (a + b), 1.5)









def dipole(a, lmn1, A, b, lmn2, B, C, direction):

    """
    
    Calculates dipole integrals between primitive Gaussians.
    
    """

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    P = gaussian_product_center(a, A, b, B)

    if direction.lower() == "x":

        XPC = P[0] - C[0]

        D  = E(l1, l2, 1, A[0] - B[0], a, b) + XPC * E(l1, l2, 0, A[0] - B[0], a, b)
        S2 = E(m1, m2, 0, A[1] - B[1], a, b)
        S3 = E(n1, n2, 0, A[2] - B[2], a, b)

        return D * S2 * S3 * np.power(pi / (a + b), 1.5)


    elif direction.lower() == 'y':

        YPC = P[1] - C[1]

        S1 = E(l1, l2, 0, A[0] - B[0], a, b)
        D  = E(m1, m2, 1, A[1] - B[1], a, b) + YPC * E(m1, m2, 0, A[1] - B[1], a, b)
        S3 = E(n1, n2, 0, A[2] - B[2], a, b)

        return S1 * D * S3 * np.power(pi / (a + b), 1.5)


    elif direction.lower() == 'z':

        ZPC = P[2] - C[2]

        S1 = E(l1, l2, 0, A[0] - B[0], a, b)
        S2 = E(m1, m2, 0, A[1] - B[1], a, b)
        D  = E(n1, n2, 1, A[2] - B[2], a, b) + ZPC * E(n1, n2, 0, A[2] - B[2], a, b)

        return S1 * S2 * D * np.power(pi / (a + b), 1.5)







def kinetic(a, lmn1, A, b, lmn2, B):

    """
    
    Calculates kinetic integrals between primitive Gaussians.
    
    """

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    Ax, Ay, Az = (2 * np.asarray(lmn2) + 1) * b
    Bx = By = Bz = -2 * np.power(b, 2)
    Cx, Cy, Cz = -0.5 * np.asarray(lmn2) * (np.asarray(lmn2) - 1) 

    Tx = Ax * E(l1, l2, 0, A[0] - B[0], a, b) + Bx * E(l1, l2 + 2, 0, A[0] - B[0], a, b) + Cx * E(l1, l2 - 2, 0, A[0] - B[0], a, b)
    Tx *= E(m1, m2, 0, A[1] - B[1], a, b)
    Tx *= E(n1, n2, 0, A[2] - B[2], a, b)

    Ty = Ay * E(m1, m2, 0, A[1] - B[1], a, b) + By * E(m1, m2 + 2, 0, A[1] - B[1], a, b) + Cy * E(m1, m2 - 2, 0, A[1] - B[1], a, b)
    Ty *= E(l1, l2, 0, A[0] - B[0], a, b)
    Ty *= E(n1, n2, 0, A[2] - B[2], a, b)

    Tz = Az * E(n1, n2, 0, A[2] - B[2], a, b) + Bz * E(n1, n2 + 2, 0, A[2] - B[2], a, b) + Cz * E(n1, n2 - 2, 0, A[2] - B[2], a, b)
    Tz *= E(l1, l2, 0, A[0] - B[0], a, b)
    Tz *= E(m1, m2, 0, A[1] - B[1], a, b)


    return (Tx + Ty + Tz) * np.power(pi / (a + b), 1.5)
          













def nuclear_attraction(a, lmn1, A, b, lmn2, B, C):

    """
    
    Calculates nuclear attraction integrals between primitive Gaussians.
    
    """

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    p = a + b
    P = np.asarray(gaussian_product_center(a, A, b, B))

    RPC = np.linalg.norm(P - C)

    val = 0.0

    for t in xrange(l1 + l2 + 1):
        for u in xrange(m1 + m2 + 1):
            for v in xrange(n1 + n2 + 1):

                val += E(l1, l2, t, A[0] - B[0], a, b) * E(m1, m2, u, A[1] - B[1], a, b) * E(n1, n2, v, A[2] - B[2], a, b) * R(t, u, v, 0, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], RPC) 

    val *= 2 * pi / p 

    return val 




