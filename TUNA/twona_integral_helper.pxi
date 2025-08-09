import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1 
from libc.stdlib cimport malloc, free
from scipy.special import factorial2 as fact2 



cdef double pi = 3.141592653589793238462643383279





cdef class Basis:
    """ Cython extension class to define primitive Gaussian basis functions"""
    cdef:
        double *origin
        long    *shell
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
            return long(view) 

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

        if self.origin == NULL:
            raise MemoryError()
        if self.shell == NULL:
            raise MemoryError()
        if self.exps == NULL:
            raise MemoryError()
        if self.coefs == NULL:
            raise MemoryError()
        if self.norm == NULL:
            raise MemoryError()

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

    def normalize(self):
        """Routine to normalize the BasisFunction objects.
           Returns self.norm, which is a list of doubles that
           normalizes the contracted Gaussian basis functions (CGBFs)

           First normalized the primitives, then takes the results and
           normalizes the contracted functions. Both steps are required,
           though I could make it one step if need be.
        """
        l = self.shell[0]
        m = self.shell[1]
        n = self.shell[2]
        L = l + m + n
        # normalize primitives first (PGBFs)
        for ia in range(self.num_exps):
            self.norm[ia] = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                            np.power(self.exps[ia],l+m+n+1.5)/
                            fact2(2*l-1)/fact2(2*m-1)/
                            fact2(2*n-1)/np.power(np.pi,1.5))

        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = np.power(np.pi,1.5)*\
            fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)

        N = 0.0
        for ia in range(self.num_exps):
            for ib in range(self.num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/np.power(self.exps[ia] + self.exps[ib],L+1.5)

        N *= prefactor
        N = np.power(N,-0.5)
        for ia in range(self.num_exps):
            self.coefs[ia] *= N

       


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double E(int i,int j,int t,double Qx,double a,double b, int n = 0, double Ax = 0.0):
    p = a + b
    u = a*b/p
    if n == 0:
        if (t < 0) or (t > (i + j)):
            return 0.0
        elif i == j == t == 0:
            return exp(-u*Qx*Qx)
        elif j == 0:
            return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - (u*Qx/a)*E(i-1,j,t,Qx,a,b) + \
                   (t+1)*E(i-1,j,t+1,Qx,a,b)
        else:
            return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + (u*Qx/b)*E(i,j-1,t,Qx,a,b) + \
                   (t+1)*E(i,j-1,t+1,Qx,a,b)
    else:
        return E(i+1,j,t,Qx,a,b,n-1,Ax) + Ax*E(i,j,t,Qx,a,b,n-1,Ax)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double R(int t,int u,int v,int n, double p,double PCx, double PCy, double PCz, double RPC):
    cdef double T = p*RPC*RPC
    cdef double val = 0.0
    if t == u == v == 0:
        val += pow(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)  
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val


@cython.cdivision(True)
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0) 

def gaussian_product_center(double a, A, double b, B):
    return (a*np.asarray(A)+b*np.asarray(B))/(a+b)