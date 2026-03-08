# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from __future__ import division
cimport cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, fabs
from libc.stdlib cimport malloc, free
from scipy.special.cython_special cimport hyp1f1
from scipy.special import factorial2
from cpython.exc cimport PyErr_CheckSignals

cdef double pi = 3.141592653589793238462643383279
cdef double LINEAR_TOL = 1.0e-14


cdef class Basis:

    """
    Defines basis functions from primitive Gaussians.
    """

    cdef:
        double *origin
        long   *shell
        long    num_exps
        double *exps
        double *coefs
        double *norm

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

        for ia in range(self.num_exps):
            self.norm[ia] = np.sqrt(
                np.power(2, 2 * (l + m + n) + 1.5)
                * np.power(self.exps[ia], l + m + n + 1.5)
                / fact2(2 * l - 1)
                / fact2(2 * m - 1)
                / fact2(2 * n - 1)
                / np.power(pi, 1.5)
            )

        prefactor = (
            np.power(pi, 1.5)
            * fact2(2 * l - 1)
            * fact2(2 * m - 1)
            * fact2(2 * n - 1)
            / np.power(2.0, L)
        )

        N = 0.0
        for ia in range(self.num_exps):
            for ib in range(self.num_exps):
                N += (
                    self.norm[ia] * self.norm[ib]
                    * self.coefs[ia] * self.coefs[ib]
                    / np.power(self.exps[ia] + self.exps[ib], L + 1.5)
                )

        N *= prefactor
        N = np.power(N, -0.5)

        for ia in range(self.num_exps):
            self.coefs[ia] *= N


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


cdef inline double E(int i, int j, int t, double Qx, double a, double b,
                     int n=0, double Ax=0.0, dict cache=None):

    cdef double p = a + b
    cdef double u = a * b / p
    cdef double res = 0.0

    if n == 0:

        if t < 0 or t > (i + j):
            return 0.0

        elif i == 0 and j == 0 and t == 0:
            return exp(-u * Qx * Qx)

        elif j == 0:
            res = (
                (1.0 / (2.0 * p)) * E(i - 1, j, t - 1, Qx, a, b, 0, 0.0, cache)
                - (u * Qx / a) * E(i - 1, j, t, Qx, a, b, 0, 0.0, cache)
                + (t + 1) * E(i - 1, j, t + 1, Qx, a, b, 0, 0.0, cache)
            )

        else:
            res = (
                (1.0 / (2.0 * p)) * E(i, j - 1, t - 1, Qx, a, b, 0, 0.0, cache)
                + (u * Qx / b) * E(i, j - 1, t, Qx, a, b, 0, 0.0, cache)
                + (t + 1) * E(i, j - 1, t + 1, Qx, a, b, 0, 0.0, cache)
            )

    else:
        res = E(i + 1, j, t, Qx, a, b, n - 1, Ax, cache) + Ax * E(i, j, t, Qx, a, b, n - 1, Ax, cache)

    return res


cdef inline double boys(double m, double T, dict cache=None):
    """
    Calculates the Boys function.
    """
    return hyp1f1(m + 0.5, m + 1.5, -T) / (2.0 * m + 1.0)


cdef inline double odd_double_factorial_from_even(int n_even):
    """
    Returns (n_even - 1)!! for even n_even >= 0.
    Examples:
        n_even = 0 -> (-1)!! = 1
        n_even = 2 -> 1
        n_even = 4 -> 3
        n_even = 6 -> 15
    """
    cdef int k
    cdef double out = 1.0

    if n_even <= 0:
        return 1.0

    for k in range(n_even - 1, 0, -2):
        out *= k

    return out


cdef inline bint quartet_is_linear_z(double* A, double* B, double* C, double* D):
    """
    Exact linear-z specialization is valid whenever all four centres share
    the same x and the same y coordinate (up to a tiny tolerance).
    """
    return (
        fabs(A[0] - B[0]) < LINEAR_TOL and
        fabs(A[0] - C[0]) < LINEAR_TOL and
        fabs(A[0] - D[0]) < LINEAR_TOL and
        fabs(A[1] - B[1]) < LINEAR_TOL and
        fabs(A[1] - C[1]) < LINEAR_TOL and
        fabs(A[1] - D[1]) < LINEAR_TOL
    )


cpdef double[:, :, :, :] doERIs(long N,
                                double[:, :, :, :] TwoE,
                                list bfs,
                                bint use_diatomic_parity=True):

    cdef:
        long i, j, k, l, l_stop
        double val
        Basis bi, bj, bk, bl

    for i in range(N):
        bi = <Basis>bfs[i]

        for j in range(i + 1):
            bj = <Basis>bfs[j]

            for k in range(i + 1):
                bk = <Basis>bfs[k]

                # Enforces (k,l) <= (i,j) in pair-index ordering without forming ij/kl
                if k == i:
                    l_stop = j + 1
                else:
                    l_stop = k + 1

                for l in range(l_stop):
                    bl = <Basis>bfs[l]

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
    Uses a specialized exact kernel for quartets whose centres lie on the z axis.
    """

    cdef double eri = 0.0
    cdef double primitive = 0.0
    cdef long ja, jb, jc, jd
    cdef double an, bn, cn, dn
    cdef double ac, bc, cc2, dc
    cdef double pref
    cdef bint use_linear_z

    use_linear_z = quartet_is_linear_z(a.origin, b.origin, c.origin, d.origin)

    for ja in range(a.num_exps):
        for jb in range(b.num_exps):
            for jc in range(c.num_exps):
                for jd in range(d.num_exps):

                    an = a.norm[ja]
                    ac = a.coefs[ja]
                    bn = b.norm[jb]
                    bc = b.coefs[jb]
                    cn = c.norm[jc]
                    cc2 = c.coefs[jc]
                    dn = d.norm[jd]
                    dc = d.coefs[jd]

                    pref = an * bn * cn * dn * ac * bc * cc2 * dc

                    if use_linear_z:
                        primitive = electron_repulsion_linear_z(
                            a.exps[ja], a.shell, a.origin,
                            b.exps[jb], b.shell, b.origin,
                            c.exps[jc], c.shell, c.origin,
                            d.exps[jd], d.shell, d.origin
                        )
                    else:
                        primitive = electron_repulsion(
                            a.exps[ja], a.shell, a.origin,
                            b.exps[jb], b.shell, b.origin,
                            c.exps[jc], c.shell, c.origin,
                            d.exps[jd], d.shell, d.origin
                        )

                    eri += pref * primitive

    return eri


cdef inline void fill_boys_table(int M, double T, double* F):
    """
    Fill F[0..M] with Boys values F_m(T).
    Uses one hyp1f1 evaluation at m=M, then downward recurrence:
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
    P[n] = (-2*p)^n for n=0..M.
    """
    cdef int n
    cdef double fac = -2.0 * p

    P[0] = 1.0
    for n in range(1, M + 1):
        P[n] = P[n - 1] * fac


cdef inline void fill_Rz_linear_table(int Vmax, int Nmax, double PCz,
                                      double* boys_tab, double* pow_tab,
                                      double* Rz):
    """
    Fills a row-major table Rz[v, n] = R(0,0,v,n) for the linear-z case.
    stride = Nmax + 1
    """
    cdef int v, n, stride = Nmax + 1

    # Base row: Rz[0, n] = (-2*alpha)^n F_n(T)
    for n in range(Nmax + 1):
        Rz[n] = pow_tab[n] * boys_tab[n]

    # Recurrence:
    # Rz(v,n) = PCz * Rz(v-1,n+1) + (v-1) * Rz(v-2,n+1)
    for v in range(1, Vmax + 1):
        for n in range(Nmax - v, -1, -1):
            Rz[v * stride + n] = PCz * Rz[(v - 1) * stride + (n + 1)]
            if v > 1:
                Rz[v * stride + n] += (v - 1) * Rz[(v - 2) * stride + (n + 1)]


cdef inline double R(int t, int u, int v, int n,
                     double p, double PCx, double PCy, double PCz, double RPC,
                     dict cache=None,
                     double* boys_tab=NULL,
                     double* pow_tab=NULL):

    cdef double val = 0.0
    cdef long long key
    cdef object tmp

    if cache is not None:
        key = ((<long long>t) << 48) | ((<long long>u) << 32) | ((<long long>v) << 16) | (<long long>n)
        tmp = cache.get(key, None)
        if tmp is not None:
            return <double>tmp

    if t == 0 and u == 0 and v == 0:
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


cdef double electron_repulsion_linear_z(double a, long *lmn1, double *A,
                                        double b, long *lmn2, double *B,
                                        double c, long *lmn3, double *C,
                                        double d, long *lmn4, double *D):
    """
    Exact primitive ERI specialization for quartets whose centres lie on the z axis
    (more precisely: all four centres share the same x and y coordinates).

    Uses:
      - x/y parity selection rules
      - x/y collapse of the Hermite Coulomb recurrence
      - a z-only auxiliary table Rz(v,n) = R(0,0,v,n)
    """

    cdef:
        long l1 = lmn1[0]
        long m1 = lmn1[1]
        long n1 = lmn1[2]
        long l2 = lmn2[0]
        long m2 = lmn2[1]
        long n2 = lmn2[2]
        long l3 = lmn3[0]
        long m3 = lmn3[1]
        long n3 = lmn3[2]
        long l4 = lmn4[0]
        long m4 = lmn4[1]
        long n4 = lmn4[2]

        double p = a + b
        double q = c + d
        double alpha = p * q / (p + q)

        double Pz = (a * A[2] + b * B[2]) / p
        double Qz = (c * C[2] + d * D[2]) / q
        double PCz = Pz - Qz
        double RPQ = fabs(PCz)

        double ABz = A[2] - B[2]
        double CDz = C[2] - D[2]

        int ntx = <int>(l1 + l2 + 1)
        int nty = <int>(m1 + m2 + 1)
        int ntz = <int>(n1 + n2 + 1)

        int nkx = <int>(l3 + l4 + 1)
        int nky = <int>(m3 + m4 + 1)
        int nkz = <int>(n3 + n4 + 1)

        int Vmax = <int>(n1 + n2 + n3 + n4)
        int Mmax = <int>((l1 + m1 + n1) + (l2 + m2 + n2) + (l3 + m3 + n3) + (l4 + m4 + n4))
        int stride = Mmax + 1

        int t, u, v, tau, nu, phi
        int Tx, Uy, W, nx, nxy

        double val = 0.0
        double pref
        double ex, ey, ez, exk, eyk, ezk
        double xfac, xyfac
        double sign_xy, sign

        double *Etx = NULL
        double *Ety = NULL
        double *Etz = NULL
        double *Eux = NULL
        double *Euy = NULL
        double *Euz = NULL

        double *boys_tab = NULL
        double *pow_tab  = NULL
        double *Rz = NULL

    if PyErr_CheckSignals() != 0:
        return 0.0

    Etx = <double*>malloc(ntx * sizeof(double))
    Ety = <double*>malloc(nty * sizeof(double))
    Etz = <double*>malloc(ntz * sizeof(double))
    Eux = <double*>malloc(nkx * sizeof(double))
    Euy = <double*>malloc(nky * sizeof(double))
    Euz = <double*>malloc(nkz * sizeof(double))

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
    Rz       = <double*>malloc((Vmax + 1) * (Mmax + 1) * sizeof(double))

    if boys_tab == NULL or pow_tab == NULL or Rz == NULL:
        if boys_tab != NULL: free(boys_tab)
        if pow_tab  != NULL: free(pow_tab)
        if Rz       != NULL: free(Rz)
        free(Etx); free(Ety); free(Etz)
        free(Eux); free(Euy); free(Euz)
        return 0.0

    fill_boys_table(Mmax, alpha * RPQ * RPQ, boys_tab)
    fill_pow_table(Mmax, alpha, pow_tab)
    fill_Rz_linear_table(Vmax, Mmax, PCz, boys_tab, pow_tab, Rz)

    # x and y displacements are exactly zero in the specialized geometry
    for t in range(ntx):
        if ((l1 + l2 + t) & 1):
            Etx[t] = 0.0
        else:
            Etx[t] = E(l1, l2, t, 0.0, a, b)

    for u in range(nty):
        if ((m1 + m2 + u) & 1):
            Ety[u] = 0.0
        else:
            Ety[u] = E(m1, m2, u, 0.0, a, b)

    for v in range(ntz):
        Etz[v] = E(n1, n2, v, ABz, a, b)

    for tau in range(nkx):
        if ((l3 + l4 + tau) & 1):
            Eux[tau] = 0.0
        else:
            Eux[tau] = E(l3, l4, tau, 0.0, c, d)

    for nu in range(nky):
        if ((m3 + m4 + nu) & 1):
            Euy[nu] = 0.0
        else:
            Euy[nu] = E(m3, m4, nu, 0.0, c, d)

    for phi in range(nkz):
        Euz[phi] = E(n3, n4, phi, CDz, c, d)

    for t in range(ntx):
        ex = Etx[t]
        if ex == 0.0:
            continue

        for tau in range(nkx):
            exk = Eux[tau]
            if exk == 0.0:
                continue

            Tx = t + tau

            # Should be even for surviving quartets, but keep the guard explicit.
            if (Tx & 1):
                continue

            nx = Tx >> 1
            xfac = ex * exk * odd_double_factorial_from_even(Tx)

            for u in range(nty):
                ey = Ety[u]
                if ey == 0.0:
                    continue

                for nu in range(nky):
                    eyk = Euy[nu]
                    if eyk == 0.0:
                        continue

                    Uy = u + nu
                    if (Uy & 1):
                        continue

                    nxy = nx + (Uy >> 1)
                    xyfac = xfac * ey * eyk * odd_double_factorial_from_even(Uy)
                    sign_xy = -1.0 if ((tau + nu) & 1) else 1.0

                    for v in range(ntz):
                        ez = Etz[v]
                        if ez == 0.0:
                            continue

                        for phi in range(nkz):
                            ezk = Euz[phi]
                            if ezk == 0.0:
                                continue

                            W = v + phi
                            sign = -sign_xy if (phi & 1) else sign_xy

                            val += xyfac * ez * ezk * sign * Rz[W * stride + nxy]

    pref = 2.0 * pow(pi, 2.5) / (p * q * sqrt(p + q))
    val *= pref

    free(Etx); free(Ety); free(Etz)
    free(Eux); free(Euy); free(Euz)
    free(boys_tab); free(pow_tab); free(Rz)

    return val


cdef double electron_repulsion(double a, long *lmn1, double *A,
                               double b, long *lmn2, double *B,
                               double c, long *lmn3, double *C,
                               double d, long *lmn4, double *D):
    """
    General primitive ERI kernel retained as the fallback for non-linear quartets.
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

    cdef double s = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia] * b.norm[ib] * ca * cb * overlap(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin)

    return s


cpdef double Mu(object a, object b, C, str direction):

    cdef double mu = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            mu += a.norm[ia] * b.norm[ib] * ca * cb * dipole(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin, C, direction)

    return mu


cpdef double T(object a, object b):

    cdef double t = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia] * b.norm[ib] * ca * cb * kinetic(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin)

    return t


cpdef double V(object a, object b, double[:] C):

    cdef double v = 0.0

    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia] * b.norm[ib] * ca * cb * nuclear_attraction(a.exps[ia], a.shell, a.origin, b.exps[ib], b.shell, b.origin, C)

    return v


def overlap(a, lmn1, A, b, lmn2, B):

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    S1 = E(l1, l2, 0, A[0] - B[0], a, b)
    S2 = E(m1, m2, 0, A[1] - B[1], a, b)
    S3 = E(n1, n2, 0, A[2] - B[2], a, b)

    return S1 * S2 * S3 * np.power(pi / (a + b), 1.5)


def dipole(a, lmn1, A, b, lmn2, B, C, direction):

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    P = gaussian_product_center(a, A, b, B)

    if direction.lower() == "x":

        XPC = P[0] - C[0]
        D  = E(l1, l2, 1, A[0] - B[0], a, b) + XPC * E(l1, l2, 0, A[0] - B[0], a, b)
        S2 = E(m1, m2, 0, A[1] - B[1], a, b)
        S3 = E(n1, n2, 0, A[2] - B[2], a, b)

        return D * S2 * S3 * np.power(pi / (a + b), 1.5)

    elif direction.lower() == "y":

        YPC = P[1] - C[1]
        S1 = E(l1, l2, 0, A[0] - B[0], a, b)
        D  = E(m1, m2, 1, A[1] - B[1], a, b) + YPC * E(m1, m2, 0, A[1] - B[1], a, b)
        S3 = E(n1, n2, 0, A[2] - B[2], a, b)

        return S1 * D * S3 * np.power(pi / (a + b), 1.5)

    elif direction.lower() == "z":

        ZPC = P[2] - C[2]
        S1 = E(l1, l2, 0, A[0] - B[0], a, b)
        S2 = E(m1, m2, 0, A[1] - B[1], a, b)
        D  = E(n1, n2, 1, A[2] - B[2], a, b) + ZPC * E(n1, n2, 0, A[2] - B[2], a, b)

        return S1 * S2 * D * np.power(pi / (a + b), 1.5)


def kinetic(a, lmn1, A, b, lmn2, B):

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

    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2

    p = a + b
    P = np.asarray(gaussian_product_center(a, A, b, B))
    RPC = np.linalg.norm(P - C)

    val = 0.0

    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val += (
                    E(l1, l2, t, A[0] - B[0], a, b)
                    * E(m1, m2, u, A[1] - B[1], a, b)
                    * E(n1, n2, v, A[2] - B[2], a, b)
                    * R(t, u, v, 0, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], RPC)
                )

    val *= 2.0 * pi / p
    return val