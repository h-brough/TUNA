from __future__ import division
import cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1 
from scipy.special import factorial2 as fact2
include "twona_integral_helper.pxi"

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double [:,:,:,:] doERIs(long N,double [:,:,:,:] TwoE, list bfs):
    cdef:
        long i,j,k,l,ij,kl
    for i in (range(N)):
        for j in range(i+1):
            ij = (i*(i+1)//2 + j)
            for k in range(N):
                for l in range(k+1):
                    kl = (k*(k+1)//2 + l)
                    if ij >= kl:
                       val = ERI(bfs[i],bfs[j],bfs[k],bfs[l])
                       TwoE[i,j,k,l] = val
                       TwoE[k,l,i,j] = val
                       TwoE[j,i,l,k] = val
                       TwoE[l,k,j,i] = val
                       TwoE[j,i,k,l] = val
                       TwoE[l,k,i,j] = val
                       TwoE[i,j,l,k] = val
                       TwoE[k,l,j,i] = val
    return TwoE


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(Basis a, Basis b, Basis c, Basis d):
    cdef double eri = 0.0
    cdef long ja, jb, jc, jd
    cdef double ca, cb, cc, cd
    for ja in range(a.num_exps):
        for jb in range(b.num_exps):
            for jc in range(c.num_exps):
                for jd in range(d.num_exps):
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             a.coefs[ja]*b.coefs[jb]*c.coefs[jc]*d.coefs[jd]*\
                             electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                                b.exps[jb],b.shell,b.origin,\
                                                c.exps[jc],c.shell,c.origin,\
                                                d.exps[jd],d.shell,d.origin)
    return eri

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double electron_repulsion(double a, long *lmn1, double *A, double b, long *lmn2, double *B,double c, long *lmn3, double *C,double d, long *lmn4, double *D):
    cdef:
        long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
        long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
        long l3 = lmn3[0], m3 = lmn3[1], n3 = lmn3[2]
        long l4 = lmn4[0], m4 = lmn4[1], n4 = lmn4[2]
        double p = a+b
        double q = c+d
        double alpha = p*q/(p+q)
        double Px = (a*A[0] + b*B[0])/p
        double Py = (a*A[1] + b*B[1])/p
        double Pz = (a*A[2] + b*B[2])/p
        double Qx = (c*C[0] + d*D[0])/q
        double Qy = (c*C[1] + d*D[1])/q
        double Qz = (c*C[2] + d*D[2])/q
        double RPQ = sqrt(pow(Px-Qx,2) + \
                           pow(Py-Qy,2) + \
                           pow(Pz-Qz,2)) 

        long t,u,v,tau,nu,phi
        double val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   pow(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,Px-Qx,Py-Qy,Pz-Qz,RPQ) 

    val *= 2*pow(pi,2.5)/(p*q*sqrt(p+q)) 
    return val 




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double S(object a, object b):
    """ Returns overlap """
    cdef double s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                             b.exps[ib],b.shell,b.origin)
    return s

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double Mu(object a, object b, C, str direction):
    cdef double mu = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            mu += a.norm[ia]*b.norm[ib]*ca*cb*\
                     dipole(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,direction)
    return mu

def RxDel(a, b, C, direction):
    l = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            l += a.norm[ia]*b.norm[ib]*ca*cb*\
                     angular(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C,direction)
    return l

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double T(object a, object b):
    """ Kinetic energy integrals """
    cdef double t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double V(object a, object b, double [:] C): 
    """ Nuclear attraction integrals """
    cdef double v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C)
    return v

def overlap(a,lmn1,A,b,lmn2,B):
    """ Returns overlap between two primitive Gaussian basis functions """
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    S1 = E(l1,l2,0,A[0]-B[0],a,b)
    S2 = E(m1,m2,0,A[1]-B[1],a,b)
    S3 = E(n1,n2,0,A[2]-B[2],a,b)
    return S1*S2*S3*np.power(pi/(a+b),1.5)


def dipole(a,lmn1,A,b,lmn2,B,C,direction):
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    P = gaussian_product_center(a,A,b,B)
    if direction.lower() == 'x':
        XPC = P[0] - C[0]
        # Top call for 'D; works for sure, bottom works in terms of properties,
        # but the gauge-origin is different so the AO ints differ.
        D  = E(l1,l2,1,A[0]-B[0],a,b) + XPC*E(l1,l2,0,A[0]-B[0],a,b)
        #D  = E(l1,l2,0,A[0]-B[0],a,b,1)
        S2 = E(m1,m2,0,A[1]-B[1],a,b)
        S3 = E(n1,n2,0,A[2]-B[2],a,b)
        return D*S2*S3*np.power(pi/(a+b),1.5)
    elif direction.lower() == 'y':
        YPC = P[1] - C[1]
        S1 = E(l1,l2,0,A[0]-B[0],a,b)
        D  = E(m1,m2,1,A[1]-B[1],a,b) + YPC*E(m1,m2,0,A[1]-B[1],a,b)
        #D  = E(m1,m2,0,A[1]-B[1],a,b,1)
        S3 = E(n1,n2,0,A[2]-B[2],a,b)
        return S1*D*S3*np.power(pi/(a+b),1.5)
    elif direction.lower() == 'z':
        ZPC = P[2] - C[2]
        S1 = E(l1,l2,0,A[0]-B[0],a,b)
        S2 = E(m1,m2,0,A[1]-B[1],a,b)
        D  = E(n1,n2,1,A[2]-B[2],a,b) + ZPC*E(n1,n2,0,A[2]-B[2],a,b)
        #D  = E(n1,n2,0,A[2]-B[2],a,b,1)
        return S1*S2*D*np.power(pi/(a+b),1.5)

def kinetic(a,lmn1,A,b,lmn2,B):
    # explicit kinetic in terms of "E" operator
    # generalized to include GIAO derivatives
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    Ax,Ay,Az = (2*np.asarray(lmn2) + 1)*b
    Bx = By = Bz = -2*np.power(b,2) # redundant, I know
    Cx,Cy,Cz = -0.5*np.asarray(lmn2)*(np.asarray(lmn2)-1) 

    Tx = Ax*E(l1,l2  ,0,A[0]-B[0],a,b) + \
         Bx*E(l1,l2+2,0,A[0]-B[0],a,b) + \
         Cx*E(l1,l2-2,0,A[0]-B[0],a,b)
    Tx *= E(m1,m2,0,A[1]-B[1],a,b)
    Tx *= E(n1,n2,0,A[2]-B[2],a,b)

    Ty = Ay*E(m1,m2  ,0,A[1]-B[1],a,b) + \
         By*E(m1,m2+2,0,A[1]-B[1],a,b) + \
         Cy*E(m1,m2-2,0,A[1]-B[1],a,b)
    Ty *= E(l1,l2,0,A[0]-B[0],a,b)
    Ty *= E(n1,n2,0,A[2]-B[2],a,b)

    Tz = Az*E(n1,n2  ,0,A[2]-B[2],a,b) + \
         Bz*E(n1,n2+2,0,A[2]-B[2],a,b) + \
         Cz*E(n1,n2-2,0,A[2]-B[2],a,b)
    Tz *= E(l1,l2,0,A[0]-B[0],a,b)
    Tz *= E(m1,m2,0,A[1]-B[1],a,b)

    return (Tx + Ty + Tz)*np.power(pi/(a+b),1.5)
          

def angular(a, lmn1, A, b, lmn2, B, C, direction):
    # a little extra work at the moment, but not all that more expensive
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    P = gaussian_product_center(a,A,b,B)

    XPC = P[0] - C[0]
    YPC = P[1] - C[1]
    ZPC = P[2] - C[2]

    S0x =    E(l1,l2,0,A[0]-B[0],a,b) 
    S0y =    E(m1,m2,0,A[1]-B[1],a,b) 
    S0z =    E(n1,n2,0,A[2]-B[2],a,b) 

    S1x = E(l1,l2,0,A[0]-B[0],a,b,1,A[0]-C[0])
    S1y = E(m1,m2,0,A[1]-B[1],a,b,1,A[1]-C[1])
    S1z = E(n1,n2,0,A[2]-B[2],a,b,1,A[2]-C[2])
    

    D1x = l2*E(l1,l2-1,0,A[0]-B[0],a,b) - 2*b*E(l1,l2+1,0,A[0]-B[0],a,b)
    D1y = m2*E(m1,m2-1,0,A[1]-B[1],a,b) - 2*b*E(m1,m2+1,0,A[1]-B[1],a,b)
    D1z = n2*E(n1,n2-1,0,A[2]-B[2],a,b) - 2*b*E(n1,n2+1,0,A[2]-B[2],a,b)

    if direction.lower() == 'x':
        return -S0x*(S1y*D1z - S1z*D1y)*np.power(pi/(a+b),1.5) 

    elif direction.lower() == 'y':
        return -S0y*(S1z*D1x - S1x*D1z)*np.power(pi/(a+b),1.5) 

    elif direction.lower() == 'z':
        return -S0z*(S1x*D1y - S1y*D1x)*np.power(pi/(a+b),1.5) 




def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    """ Returns nuclear attraction integral between two primitive Gaussians"""
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    p = a + b
    P = np.asarray(gaussian_product_center(a,A,b,B))
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in xrange(l1+l2+1):
        for u in xrange(m1+m2+1):
            for v in xrange(n1+n2+1):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC) 
    val *= 2*pi/p # Pink book, Eq(9.9.40) 
    return val 




# expose boys function for testing purposes only
def _boys(n,T):
    return boys(n,T)