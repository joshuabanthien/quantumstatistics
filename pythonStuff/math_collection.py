import numpy as np
import scipy as sci
import math


#defines kronecker delta

def kronecker_delta(m,n):

    if m == n:

        return 1

    else:

        return 0


#computes the matrix rep. of operators acting on an n-fold product space of non-interacting systems

def kronecker_product_with_identity(n_sys,mat):

    val = mat

    for i in range(n_sys-1):

        val = np.kron(val,np.identity(val.shape[0])) + np.kron(np.identity(val.shape[0]),val)

    return val


#general exponential decay curve

def monoExp(t, a, b, c):

    return a * np.exp(-c * t) + b
    

#heavyside step function

def step_function(x):

    if x < 0:

        return 0

    else:

        return 1


#calculates the real part of the eigenvalue with the real part with the smallest magnitude

def smallest_nonzero_eigval(M):

    eigvals = np.abs(sorted(np.real(np.linalg.eigvals(M))))

    smallest_eigval = min(i for i in eigvals if i > 1e-10)

    return smallest_eigval
