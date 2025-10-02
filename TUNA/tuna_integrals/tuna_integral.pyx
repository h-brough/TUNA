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






cdef inline double R(int t, int u, int v, int n, double p, double PCx, double PCy, double PCz, double RPC, dict cache = None):

    cdef double T = p * RPC * RPC
    cdef double val = 0.0

    if t == 0 and u == 0 and v == 0: val += pow(-2 * p, n) * boys(n, T) 

    elif t == 0 and u == 0:

        if v > 1: val += (v - 1) * R(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC, cache)   

        val += PCz * R(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC, cache)

    elif t == 0:

        if u > 1: val += (u - 1) * R(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC, cache)  

        val += PCy * R(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC, cache)

    else:

        if t > 1: val += (t - 1) * R(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC, cache)  
        val += PCx * R(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC, cache)

    return val







cdef inline double boys(double m, double T, dict cache = None):

    """
    
    Calculates the boys function.
    
    """

    return hyp1f1(m + 0.5, m + 1.5, -T) / (2.0 * m + 1.0) 








cpdef double [:, :, :, :] doERIs(long N, double [:, :, :, :] TwoE, list bfs):

    """
    
    Calculates the four-dimensional array of two-electron integrals.
    
    """

    cdef:

        long i, j, k, l, ij, kl

    for i in range(N):
        for j in range(i + 1):

            ij = (i * (i + 1) // 2 + j)

            for k in range(N):
                for l in range(k + 1):

                    kl = (k * (k + 1) // 2 + l)

                    if ij >= kl:

                       val = ERI(bfs[i], bfs[j], bfs[k], bfs[l])

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








cdef double electron_repulsion(double a, long *lmn1, double *A, double b, long *lmn2, double *B, double c, long *lmn3, double *C, double d, long *lmn4, double *D):
    
    """
    
    Recursive calculation of electron repulsion integrals between primitive Gaussians.
    
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

        long t, u, v, tau, nu, phi
        double val = 0.0

    if PyErr_CheckSignals() != 0: return  0

    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                for tau in range(l3 + l4 + 1):
                    for nu in range(m3 + m4 + 1):
                        for phi in range(n3 + n4 + 1):
    
                            val += E(l1, l2, t, A[0] - B[0], a, b) * E(m1, m2, u, A[1] - B[1], a, b) * E(n1, n2, v, A[2] - B[2], a, b) * E(l3, l4, tau, C[0] - D[0], c, d) * E(m3, m4, nu, C[1] - D[1], c, d) * E(n3, n4, phi, C[2] - D[2], c, d) * pow(-1,tau + nu + phi) * R(t + tau, u + nu, v + phi, 0, alpha, Px - Qx, Py - Qy, Pz - Qz, RPQ) 

    val *= 2 * pow(pi, 2.5) / (p * q * sqrt(p + q))

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









def RxDel(a, b, C, direction):

    """
    
    Calculates angular integrals between contracted Gaussians.
    
    """

    l = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):

            l += a.norm[ia] * b.norm[ib] * ca * cb * angular(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin, C, direction)

    return l









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
          










def angular(a, lmn1, A, b, lmn2, B, C, direction):

    """
    
    Calculates angular integrals between primitive Gaussians.
    
    """

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    P = gaussian_product_center(a, A, b, B)

    XPC = P[0] - C[0]
    YPC = P[1] - C[1]
    ZPC = P[2] - C[2]

    S0x = E(l1, l2, 0, A[0] - B[0], a, b) 
    S0y = E(m1, m2, 0, A[1] - B[1], a, b) 
    S0z = E(n1, n2, 0, A[2] - B[2], a, b) 

    S1x = E(l1, l2, 0, A[0] - B[0], a, b, 1, A[0] - C[0])
    S1y = E(m1, m2, 0, A[1] - B[1], a, b, 1, A[1] - C[1])
    S1z = E(n1, n2, 0, A[2] - B[2], a, b, 1, A[2] - C[2])
    
    D1x = l2 * E(l1, l2 - 1, 0, A[0] - B[0], a, b) - 2 * b * E(l1, l2 + 1, 0, A[0] - B[0], a, b)
    D1y = m2 * E(m1, m2 - 1, 0, A[1] - B[1], a, b) - 2 * b * E(m1, m2 + 1, 0, A[1] - B[1], a, b)
    D1z = n2 * E(n1, n2 - 1, 0, A[2] - B[2], a, b) - 2 * b * E(n1, n2 + 1, 0, A[2] - B[2], a, b)

    if direction.lower() == "x": return -S0x * (S1y * D1z - S1z * D1y) * np.power(pi / (a + b), 1.5) 

    elif direction.lower() == "y": return -S0y * (S1z * D1x - S1x * D1z) * np.power(pi / (a + b), 1.5) 

    elif direction.lower() == "z": return -S0z * (S1x * D1y - S1y * D1x) * np.power(pi / (a + b), 1.5) 








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




