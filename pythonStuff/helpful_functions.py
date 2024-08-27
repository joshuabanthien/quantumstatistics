import numpy as np
import scipy as sci
import math


#defines kronecker delta

def kronecker_delta(m,n):

    if m == n:

        return 1

    else:

        return 0


#defines SHO energy eigenstates for plotting

def SHO_n(n,x):

    return 1/np.sqrt(np.sqrt(np.pi) * 2**n *math.factorial(n)) * np.exp(-0.5*x**2) * sci.special.hermite(n, monic=False)(x)


#computes the matrix rep. of operators acting on an n-fold product space of non-interacting systems

def kronecker_product_with_identity(n_sys,mat):

    val = mat

    for i in range(n_sys-1):

        val = np.kron(val,np.identity(val.shape[0])) + np.kron(np.identity(val.shape[0]),val)

    return val
